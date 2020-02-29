# Named NDArrays

def check_lengths_match(a, b):
    lens_a = sorted(list(zip(a.dimension_names, a.shape)))
    lens_b = sorted(list(zip(b.dimension_names, b.shape)))
    assert lens_a == lens_b, "Dimension error"

import numpy as np

def _broadcast(lhs, rhs):
    """Here we implement numpy-incompatible broadcasting rules

    There rules are designed for arrays with named axes.  As a result,
    they are much conceptually simpler than those of numpy.  It is not
    compatible with operations which require multiple dimensions,
    e.g. broadcasting matrix multiplications.  The rules are:
    
    0. If either element is a scalar, return immediately.
    
    1. Add all dimensions contained in lhs which are not in rhs to the
    end of rhs, in the order in which they appear in lhs.
    
    2. Add all dimensions contained in rhs which are not in lhs to the
    end of rhs in the order in which they appear in rhs.
    
    3. Rearrange the dimensions in rhs to match the order in lhs.
    """
    # 0. Check for scalars
    if np.isscalar(lhs) or np.isscalar(rhs):
        return (lhs, rhs)
    # 1. Add dimensions to rhs
    to_add_rhs = [d for d in lhs.dimension_names if d not in rhs.dimension_names]
    for d in to_add_rhs:
        rhs = rhs._add_axis(d)
    # 2. Add dimensions to rhs
    to_add_lhs = [d for d in rhs.dimension_names if d not in lhs.dimension_names]
    for d in to_add_lhs:
        lhs = lhs._add_axis(d)
    # 3. Rearrange
    rhs = rhs.transpose(lhs.dimension_names)
    return (lhs,rhs)

def _wrap_binop(funcname, rightop=False):
    """Wrapper for binop functions

    Accept two arrays as input, broadcast them to each other, and then
    perform the specified binop.  `funcname` specifies the binop
    method as a string, and `rightop` specifies whether it is the left
    version (False) or the right version (True) of the binop function.
    """
    def newfunc(self, other):
        if rightop:
            lhs, rhs = _broadcast(other, self)
        else:
            lhs, rhs = _broadcast(self, other)
        dimnames = lhs.dimension_names
        out = getattr(np.ndarray, funcname)(lhs, rhs)
        out.dimension_names = dimnames
        return out
    return newfunc

def _wrap_arrayfunc(funcname):
    """Wrapper for array functions functions which return nndarrays

    Accept one nndarray, fix the dimensions, and return a nndarray
    with dimensions collapsed by the specified dimension.  `funcname`
    should be a string with the name of the function.
    """
    def newfunc(self, axis=None, dtype=None, out=None, keepdims=False, ddof=None):
        extraargs = {"out": out}
        if ddof is not None:
            assert funcname in ["std", "var"], "Invalid ddof argument"
            extraargs["ddof"] = ddof
        if dtype is not None:
            assert funcname not in ["min", "max"], "Invalid dtype argument"
            extraargs["dtype"] = dtype
        if keepdims is not False:
            assert funcname not in ["cumsum", "cumprod"], "Invalid dtype argument"
            extraargs["keepdims"] = keepdims
        getattr(np.ndarray, funcname).__doc__
        new_axis = None
        new_dims = self.dimension_names
        if axis is not None:
            if isinstance(axis, tuple):
                new_axis = self._axis_name2num(axis)
                if not keepdims:
                    new_dims = [a for a in self.dimension_names if a not in axis]
            else:
                new_axis = self._axis_name2num(axis)
                if not keepdims:
                    new_dims = [a for a in self.dimension_names if a != axis]
        res = getattr(np.ndarray, funcname)(np.asarray(self), axis=new_axis, **extraargs)
        if isinstance(res, np.ndarray):
            res = self.__class__(res, new_dims)
        return res
    return newfunc

def _wrap_ndarrayfunc(funcname):
    """Wrapper for array functions functions which return ndarrays

    Accept one nndarray, fix the dimensions, and return a standard
    ndarray.  `funcname` should be a string with the name of the
    function.
    """
    def newfunc(self, *args, **kwargs):
        return np.asarray(getattr(np.ndarray, funcname)(self, *args, **kwargs))
    return newfunc

class nndarray(np.ndarray):
    """Named NDArray - NDArrays with axis names

    The constructor takes two arguments. `input_array` takes anything
    that can be cast to an array (e.g. with asarray), and
    `dimension_names` takes a list of names for the dimensions.  The
    number of dimensions of `input_array` must be equal to the length
    of `dimension_names`.
    """
    def __new__(cls, input_array, dimension_names):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        _input = np.asarray(input_array)
        assert _input.ndim == len(dimension_names), "Invalid dimension names"
        obj = _input.view(cls)
        # add the new attribute to the created instance
        obj.dimension_names = dimension_names
        assert len(dimension_names) == len(set(dimension_names)), "Dimension names must not be duplicated"
        # Finally, we must return the newly created object:
        return obj
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        dimnames = getattr(obj, 'dimension_names', None)
        if dimnames is not None and len(dimnames) != len(self.shape):
            dimnames = None
        self.dimension_names = dimnames
    
    sum = _wrap_arrayfunc("sum")
    prod = _wrap_arrayfunc("prod")
    min = _wrap_arrayfunc("min")
    max = _wrap_arrayfunc("max")
    any = _wrap_arrayfunc("any")
    all = _wrap_arrayfunc("all")
    mean = _wrap_arrayfunc("mean")
    std = _wrap_arrayfunc("std")
    var = _wrap_arrayfunc("var")
    
    def argmin(self, axis=None, dtype=None, out=None, keepdims=False):
        if axis is not None:
            axis = self._axis_name2num(axis)
        return np.ndarray.argmin(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    
    def argmax(self, axis=None, dtype=None, out=None, keepdims=False):
        if axis is not None:
            axis = self._axis_name2num(axis)
        return np.ndarray.argmax(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    
    def _axis_name2num(self, axis):
        assert not isinstance(axis, list), "Use a tuple instead of a list"
        if isinstance(axis, tuple):
            assert all([a in self.dimension_names for a in axis]), f"Invalid axis name {axis} in {self.dimension_names})"
            return tuple([self.dimension_names.index(a) for a in axis])
        else:
            assert axis in self.dimension_names, f"Invalid axis name {axis} in {self.dimension_names})"
            return self.dimension_names.index(axis)
    
    def _axis_num2name(self, axis):
        if isinstance(axis, tuple):
            return tuple([self.dimension_names[a] for a in axis])
        else:
            assert axis in self.dimension_names, "Invalid axis name"
            return self.dimension_names[axis]
    
    def _add_axis(self, axis, pos=-1):
        out = self.__class__(np.expand_dims(self, -1), tuple(list(self.dimension_names)+[axis]))
        if pos != -1:
            print(list(self.dimension_names[0:pos]), [axis], list(self.dimension_names[pos:]))
            permute = tuple(list(self.dimension_names[0:pos]) + [axis] + list(self.dimension_names[pos:]))
            print(permute)
            out = out.transpose(permute)
        return out
    
    def __repr__(self):
        array_repr = np.ndarray.__repr__(self)
        return f"nndarray({np.ndarray.__repr__(np.asarray(self))}, {self.dimension_names})"
    
    def __str__(self):
        return f"{np.ndarray.__str__(self)}\nDimensions: {', '.join(self.dimension_names)}"
    
    __add__  = _wrap_binop("__add__")
    __radd__ = _wrap_binop("__radd__", rightop=True)
    __sub__  = _wrap_binop("__sub__")
    __rsub__ = _wrap_binop("__rsub__", rightop=True)
    __mul__  = _wrap_binop("__mul__")
    __rmul__ = _wrap_binop("__rmul__", rightop=True)
    __div__  = _wrap_binop("__div__")
    __rdiv__ = _wrap_binop("__rdiv__", rightop=True)
    __truediv__  = _wrap_binop("__truediv__")
    __rtruediv__ = _wrap_binop("__rtruediv__", rightop=True)
    __mod__  = _wrap_binop("__mod__")
    __rmod__ = _wrap_binop("__rmod__", rightop=True)
    #__divmod__ = _wrap_binop("__divmod__") # TODO
    #__rdivmod__ = _wrap_binop("__rdivmod__", rightop=True) # TODO
    __pow__  = _wrap_binop("__pow__")
    __rpow__ = _wrap_binop("__rpow__", rightop=True)
    __lshift__  = _wrap_binop("__lshift__")
    __rlshift__ = _wrap_binop("__rlshift__", rightop=True)
    __rshift__  = _wrap_binop("__rshift__")
    __rrshift__ = _wrap_binop("__rrshift__", rightop=True)
    __and__  = _wrap_binop("__and__")
    __rand__ = _wrap_binop("__rand__", rightop=True)
    __or__  = _wrap_binop("__or__")
    __ror__ = _wrap_binop("__ror__", rightop=True)
    __xor__  = _wrap_binop("__xor__")
    __rxor__ = _wrap_binop("__rxor__", rightop=True)
    __eq__ = _wrap_binop("__eq__")
    __neq__ = _wrap_binop("__neq__")
    __lt__ = _wrap_binop("__lt__")
    __gt__ = _wrap_binop("__gt__")
    __le__ = _wrap_binop("__le__")
    __ge__ = _wrap_binop("__ge__")
    
    def transpose(self, *axes):
        # Default with no arguments is to reverse the axes
        if axes is None or len(axes) == 0:
            axes = list(reversed(self.dimension_names))
        if len(axes) == 1:
            axes = axes[0]
        
        new = np.ndarray.transpose(self, self._axis_name2num(tuple(axes)))
        new.dimension_names = [self.dimension_names[self._axis_name2num(a)] for a in axes]
        check_lengths_match(self, new)
        return new
    
    @property
    def T(self):
        return self.transpose()
    
    def squeeze(self, axis=None):
        assert not isinstance(axis, list), "Use a tuple instead of a list"
        # If axis isn't specified, choose all of those with dimension 1
        if axis is None:
            axis = tuple(x for x,s in zip(self.dimension_names, self.shape) if s == 1)
        # Get indices and new 
        axis_i = self._axis_name2num(axis)
        if isinstance(axis, tuple):
            new_dimensions = [d for d in self.dimension_names if d not in axis]
        else:
            new_dimensions = [d for d in self.dimension_names if d != axis]
        new = np.ndarray.squeeze(self, axis_i)
        new.dimension_names = new_dimensions
        return new
    
    def swapaxes(self, axis1, axis2):
        axis1_i = self._axis_name2num(axis1)
        axis2_i = self._axis_name2num(axis2)
        new_dimnames = self.dimension_names.copy()
        new_dimnames[axis1_i] = axis2
        new_dimnames[axis2_i] = axis1
        new = np.ndarray.swapaxes(self, axis1_i, axis2_i)
        new.dimension_names = new_dimnames
        return new
    
    def repeat(self, repeats, axis=None):
        if axis is not None:
            axis = self._axis_name2num(axis)
        new = np.ndarray.repeat(self, repeats, axis=axis)
        if axis:
            new.dimension_names = self.dimension_names
            return new
        else:
            return np.asarray(new)
    
    ravel = _wrap_ndarrayfunc("ravel")
    flatten = _wrap_ndarrayfunc("flatten")
    reshape = _wrap_ndarrayfunc("reshape")
    diagonal = _wrap_ndarrayfunc("diagonal")
    argsort = _wrap_ndarrayfunc("argsort")
    
    def sort(self, axis=-1, kind=None, order=None):
        if axis != -1:
            axis = self._axis_name2num(axis)
        return np.ndarray.sort(self, axis=axis, kind=kind, order=order)
    
    def argsort(self, axis=-1, kind=None, order=None):
        if axis != -1:
            axis = self._axis_name2num(axis)
        return np.asarray(np.ndarray.argsort(self, axis=axis, kind=kind, order=order))
    
    def cumsum(self, axis=None, out=None):
        if axis is not None:
            axis = self._axis_name2num(axis)
        return np.asarray(np.ndarray.cumsum(self, axis=axis, out=out))
    
    def cumprod(self, axis=None, out=None):
        if axis is not None:
            axis = self._axis_name2num(axis)
        return np.asarray(np.ndarray.cumprod(self, axis=axis, out=out))
    
    def __matmul__(self, other):
        # There are four cases handled by this, according to the numpy documentation:
        #
        # 1. If both arguments are 2-D they are multiplied like conventional matrices.
        # 2. If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        # 3. If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
        # 4. If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
        #
        # We handle 1, 3, and 4 here.  We leave 2 unimplemented for the moment.  TODO.
        assert self.ndim <= 2 and other.ndim <= 2, "Stack of matrices currently unimplemented."
        assert isinstance(other, self.__class__), "Must be named ndarrays (nndarray)"
        
        # Convert vectors to matrices if necessary
        lhs = self
        rhs = other
        if lhs.ndim == 1:
            lhs = lhs._add_axis("__DUMMY_DIM__")
        if rhs.ndim == 1:
            rhs = rhs._add_axis("__DUMMY_DIM__")
        # If the matrices are already in the correct orientation, multiply
        if lhs.dimension_names[1] == rhs.dimension_names[0]:
            out = np.ndarray.__matmul__(lhs, rhs)
            out.dimension_names = (lhs.dimension_names[0], rhs.dimension_names[1])
        # First try transposing the second matrix
        elif lhs.dimension_names[1] == rhs.dimension_names[1]:
            out = np.ndarray.__matmul__(lhs, rhs.T)
            out.dimension_names = (lhs.dimension_names[0], rhs.dimension_names[0])
        # Now try transposing the first matrix
        elif lhs.dimension_names[0] == rhs.dimension_names[0]:
            out = np.ndarray.__matmul__(lhs.T, rhs)
            out.dimension_names = (lhs.dimension_names[1], rhs.dimension_names[1])
        # Now try both
        elif lhs.dimension_names[0] == rhs.dimension_names[1]:
            out = np.ndarray.__matmul__(lhs.T, rhs.T)
            out.dimension_names = (lhs.dimension_names[1], rhs.dimension_names[0])
        else:
            raise ValueError("Invalid dimensions for matrix multiplication")
        # Make sure the output doesn't have any dummy dimensions
        if "__DUMMY_DIM__" in out.dimension_names:
            out = out.squeeze("__DUMMY_DIM__")
        return out

nnumpy = object()

def _nnumpy_concatenate(A, axis=None):
    if len(A) == 1:
        return A[0]
    for i in range(0, len(A)-1):
        a,b = _broadcast(A[i], A[i+1])
        A[i] = a
        A[i+1] = b
    for i in reversed(range(0, len(A)-1)):
        a,b = _broadcast(A[i], A[i+1])
        A[i] = a
        A[i+1] = b
    if axis is None:
        newaxis = 0
    else:
        newaxis = A[0]._axis_name2num(axis)
    return nndarray(np.concatenate(A, axis=newaxis), A[0].dimension_names)
