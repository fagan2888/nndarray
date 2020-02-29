# Named numpy arrays

The goal with this project was to have named numpy arrays.  Many of
the mistakes which are made in code which requires lots of linear
algebra can be traced to transposed matrices, accidental broadcasting,
applying functions over the wrong axis, and dimensions which don't
line up for, e.g., matrix multiplication.  This attempts to fix this
by providing names for the dimensions.  It throws errors when
dimensions don't line up, and uses smarter broadcasting which respects
the names of the dimensions.  So for example, to find the price to
manufacture each gadget based on :

    from nndarray import nndarray
    price_per_part = nndarray([[5.5, 2.9, 3]], ["price", "part"])
    parts_per_gadget = nndarray([[4, 6, 5], [1, 1, 6], [10, 10, 10]], ["gadget", "part"])
    fixed_costs = nndarray([[3, 4, 1]], ["price", "gadget"])
    print(parts_per_gadget @ price_per_part + fixed_costs)

Notice how the arrays are not transposed correctly, but they can be
properly broadcasted based on their axis labels.

Likewise, suppose we have a matrix "survey_results" representing
responses to a survey which was taken on multiple days, which has axes
"person", "question", and "day".  No matter what transposition we have
of the matrix, we can average across people with

    survey_results.mean(axis="person")

or across days with

    survey_results.mean(axis="person")

## Status of the project

Unfortunately, I have abandoned this project because it turns out that
there are a lot of subtleties in numpy and it would take a very long
time to implement all of them.  But even if I did, there are several
aspects of the numpy API which are incompatible with this concept, and
may cause edge case bugs.

For example, consider the "concatenate" function.  The "axis" keyword
argument defaults to "0", but can also take the value "None".  "None"
means that the arrays should be flattened before being concatenated.
First, there is no reasonable way to flatten arrays while preserving
dimensions.  Second, there is no reasonable default argument which can
be used in this function, since there is no equivalent to "0" when
arrays are named.  Thus, since the existing API was not designed with
these in mind, there would need to be non-trivial changes to the API,
which is beyond the scope of this project.

Another example of why this concept can't work like I want it to is
because it may make sense for axes to have the same label.  For
example, a covariance matrix should have the same label for both
dimensions.  However, using the same label will screw up all of the
broadcasting and matching operations that make this package useful,
and thus, this isn't allowed.  There is no reasonable way around this
problem within the proposed framework.

For reasons such as these, I am no longer developing this project.

## License

This is available under the MIT license.
