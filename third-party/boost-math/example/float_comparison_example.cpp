//!file
//! \brief floating-point comparison from Boost.Test
// Copyright Paul A. Bristow 2015.
// Copyright John Maddock 2015.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/math/special_functions/next.hpp>

#include <iostream>
#include <limits> // for std::numeric_limits<T>::epsilon().

int main()
{
  std::cout << "Compare floats using Boost.Math functions/classes" << std::endl;


//[compare_floats_using
/*`Some using statements will ensure that the functions we need are accessible.
*/

  using namespace boost::math;

//`or

  using boost::math::relative_difference;
  using boost::math::epsilon_difference;
  using boost::math::float_next;
  using boost::math::float_prior;

//] [/compare_floats_using]


//[compare_floats_example_1
/*`The following examples display values with all possibly significant digits.
Newer compilers should provide `std::numeric_limits<FPT>::max_digits10`
for this purpose, and here we use `float` precision where `max_digits10` = 9
to avoid displaying a distracting number of decimal digits.

[note Older compilers can use this formula to calculate `max_digits10`
from `std::numeric_limits<FPT>::digits10`:
__spaces `int max_digits10 = 2 + std::numeric_limits<FPT>::digits10 * 3010/10000;`
] [/note]

One can set the display including all trailing zeros
(helpful for this example to show all potentially significant digits),
and also to display `bool` values as words rather than integers:
*/
  std::cout.precision(std::numeric_limits<float>::max_digits10);
  std::cout << std::boolalpha << std::showpoint << std::endl;

//] [/compare_floats_example_1]

//[compare_floats_example_2]
/*`
When comparing values that are ['quite close] or ['approximately equal],
we could use either `float_distance` or `relative_difference`/`epsilon_difference`, for example
with type `float`, these two values are adjacent to each other:
*/

  float a = 1;
  float b = 1 + std::numeric_limits<float>::epsilon();
  std::cout << "a = " << a << std::endl;
  std::cout << "b = " << b << std::endl;
  std::cout << "float_distance = " << float_distance(a, b) << std::endl;
  std::cout << "relative_difference = " << relative_difference(a, b) << std::endl;
  std::cout << "epsilon_difference = " << epsilon_difference(a, b) << std::endl;

/*`
Which produces the output:

[pre
a = 1.00000000
b = 1.00000012
float_distance = 1.00000000
relative_difference = 1.19209290e-007
epsilon_difference = 1.00000000
]
*/
  //] [/compare_floats_example_2]

//[compare_floats_example_3]
/*`
In the example above, it just so happens that the edit distance as measured by `float_distance`, and the
difference measured in units of epsilon were equal.  However, due to the way floating point
values are represented, that is not always the case:*/

  a = 2.0f / 3.0f;   // 2/3 inexactly represented as a float
  b = float_next(float_next(float_next(a))); // 3 floating point values above a
  std::cout << "a = " << a << std::endl;
  std::cout << "b = " << b << std::endl;
  std::cout << "float_distance = " << float_distance(a, b) << std::endl;
  std::cout << "relative_difference = " << relative_difference(a, b) << std::endl;
  std::cout << "epsilon_difference = " << epsilon_difference(a, b) << std::endl;

/*`
Which produces the output:

[pre
a = 0.666666687
b = 0.666666865
float_distance = 3.00000000
relative_difference = 2.68220901e-007
epsilon_difference = 2.25000000
]

There is another important difference between `float_distance` and the
`relative_difference/epsilon_difference` functions in that `float_distance`
returns a signed result that reflects which argument is larger in magnitude,
where as `relative_difference/epsilon_difference` simply return an unsigned
value that represents how far apart the values are.  For example if we swap
the order of the arguments:
*/

  std::cout << "float_distance = " << float_distance(b, a) << std::endl;
  std::cout << "relative_difference = " << relative_difference(b, a) << std::endl;
  std::cout << "epsilon_difference = " << epsilon_difference(b, a) << std::endl;

  /*`
  The output is now:

  [pre
  float_distance = -3.00000000
  relative_difference = 2.68220901e-007
  epsilon_difference = 2.25000000
  ]
*/
  //] [/compare_floats_example_3]

//[compare_floats_example_4]
/*`
Zeros are always treated as equal, as are infinities as long as they have the same sign:*/

  a = 0;
  b = -0;  // signed zero
  std::cout << "relative_difference = " << relative_difference(a, b) << std::endl;
  a = b = std::numeric_limits<float>::infinity();
  std::cout << "relative_difference = " << relative_difference(a, b) << std::endl;
  std::cout << "relative_difference = " << relative_difference(a, -b) << std::endl;

/*`
Which produces the output:

[pre
relative_difference = 0.000000000
relative_difference = 0.000000000
relative_difference = 3.40282347e+038
]
*/
//] [/compare_floats_example_4]

//[compare_floats_example_5]
/*`
Note that finite values are always infinitely far away from infinities even if those finite values are very large:*/

  a = (std::numeric_limits<float>::max)();
  b = std::numeric_limits<float>::infinity();
  std::cout << "a = " << a << std::endl;
  std::cout << "b = " << b << std::endl;
  std::cout << "relative_difference = " << relative_difference(a, b) << std::endl;
  std::cout << "epsilon_difference = " << epsilon_difference(a, b) << std::endl;

/*`
Which produces the output:

[pre
a = 3.40282347e+038
b = 1.#INF0000
relative_difference = 3.40282347e+038
epsilon_difference = 3.40282347e+038
]
*/
//] [/compare_floats_example_5]

//[compare_floats_example_6]
/*`
Finally, all denormalized values and zeros are treated as being effectively equal:*/

  a = std::numeric_limits<float>::denorm_min();
  b = a * 2;
  std::cout << "a = " << a << std::endl;
  std::cout << "b = " << b << std::endl;
  std::cout << "float_distance = " << float_distance(a, b) << std::endl;
  std::cout << "relative_difference = " << relative_difference(a, b) << std::endl;
  std::cout << "epsilon_difference = " << epsilon_difference(a, b) << std::endl;
  a = 0;
  std::cout << "a = " << a << std::endl;
  std::cout << "b = " << b << std::endl;
  std::cout << "float_distance = " << float_distance(a, b) << std::endl;
  std::cout << "relative_difference = " << relative_difference(a, b) << std::endl;
  std::cout << "epsilon_difference = " << epsilon_difference(a, b) << std::endl;

/*`
Which produces the output:

[pre
a = 1.40129846e-045
b = 2.80259693e-045
float_distance = 1.00000000
relative_difference = 0.000000000
epsilon_difference = 0.000000000
a = 0.000000000
b = 2.80259693e-045
float_distance = 2.00000000
relative_difference = 0.000000000
epsilon_difference = 0.000000000]

Notice how, in the above example, two denormalized values that are a factor of 2 apart are
none the less only one representation apart!

*/
//] [/compare_floats_example_6]


#if 0
//[old_compare_floats_example_3
//`The simplest use is to compare two values with a tolerance thus:

  bool is_close = is_close_to(1.F, 1.F + epsilon, epsilon); // One epsilon apart is close enough.
  std::cout << "is_close_to(1.F, 1.F + epsilon, epsilon); is " << is_close << std::endl; // true

  is_close = is_close_to(1.F, 1.F + 2 * epsilon, epsilon); // Two epsilon apart isn't close enough.
  std::cout << "is_close_to(1.F, 1.F + epsilon, epsilon); is " << is_close << std::endl; // false

/*`
[note The type FPT of the tolerance and the type of the values [*must match].

So `is_close(0.1F, 1., 1.)` will fail to compile because "template parameter 'FPT' is ambiguous".
Always provide the same type, using `static_cast<FPT>` if necessary.]
*/


/*`An instance of class `close_at_tolerance` is more convenient
when multiple tests with the same conditions are planned.
A class that stores a tolerance of three epsilon (and the default ['strong] test) is:
*/

  close_at_tolerance<float> three_rounds(3 * epsilon); // 'strong' by default.

//`and we can confirm these settings:

  std::cout << "fraction_tolerance = "
    << three_rounds.fraction_tolerance()
    << std::endl; // +3.57627869e-007
  std::cout << "strength = "
    << (three_rounds.strength() == FPC_STRONG ? "strong" : "weak")
    << std::endl; // strong

//`To start, let us use two values that are truly equal (having identical bit patterns)

  float a = 1.23456789F;
  float b = 1.23456789F;

//`and make a comparison using our 3*epsilon `three_rounds` functor:

  bool close = three_rounds(a, b);
  std::cout << "three_rounds(a, b) = " << close << std::endl; // true

//`Unsurprisingly, the result is true, and the failed fraction is zero.

  std::cout << "failed_fraction = " << three_rounds.failed_fraction() << std::endl;

/*`To get some nearby values, it is convenient to use the Boost.Math __next_float functions,
for which we need an include

  #include <boost/math/special_functions/next.hpp>

and some using declarations:
*/

  using boost::math::float_next;
  using boost::math::float_prior;
  using boost::math::nextafter;
  using boost::math::float_distance;

//`To add a few __ulp to one value:
  b = float_next(a); // Add just one ULP to a.
  b = float_next(b); // Add another one ULP.
  b = float_next(b); // Add another one ULP.
  // 3 epsilon would pass.
  b = float_next(b); // Add another one ULP.

//`and repeat our comparison:

  close = three_rounds(a, b);
  std::cout << "three_rounds(a, b) = " << close << std::endl; // false
  std::cout << "failed_fraction = " << three_rounds.failed_fraction()
    << std::endl;  // abs(u-v) / abs(v) = 3.86237957e-007

//`We can also 'measure' the number of bits different using the `float_distance` function:

  std::cout << "float_distance = " << float_distance(a, b) << std::endl; // 4

/*`Now consider two values that are much further apart
than one might expect from ['computational noise],
perhaps the result of two measurements of some physical property like length
where an uncertainty of a percent or so might be expected.
*/
  float fp1 = 0.01000F;
  float fp2 = 0.01001F; // Slightly different.

  float tolerance = 0.0001F;

  close_at_tolerance<float> strong(epsilon); // Default is strong.
  bool rs = strong(fp1, fp2);
  std::cout << "strong(fp1, fp2) is " << rs << std::endl;

//`Or we could contrast using the ['weak] criterion:
  close_at_tolerance<float> weak(epsilon, FPC_WEAK); // Explicitly weak.
  bool rw = weak(fp1, fp2); //
  std::cout << "weak(fp1, fp2) is " << rw << std::endl;

//`We can also construct, setting tolerance and strength, and compare in one statement:

  std::cout << a << " #= " << b << " is "
    << close_at_tolerance<float>(epsilon, FPC_STRONG)(a, b) << std::endl;
  std::cout << a << " ~= " << b << " is "
    << close_at_tolerance<float>(epsilon, FPC_WEAK)(a, b) << std::endl;

//`but this has little advantage over using function `is_close_to` directly.

//] [/old_compare_floats_example_3]


/*When the floating-point values become very small and near zero, using
//a relative test becomes unhelpful because one is dividing by zero or tiny,

//Instead, an absolute test is needed, comparing one (or usually both) values with zero,
//using a tolerance.
//This is provided by the `small_with_tolerance` class and `is_small` function.

  namespace boost {
  namespace math {
  namespace fpc {


  template<typename FPT>
  class small_with_tolerance
  {
  public:
  // Public typedefs.
  typedef bool result_type;

  // Constructor.
  explicit small_with_tolerance(FPT tolerance); // tolerance >= 0

  // Functor
  bool operator()(FPT value) const; // return true if <= absolute tolerance (near zero).
  };

  template<typename FPT>
  bool
  is_small(FPT value, FPT tolerance); // return true if value <= absolute tolerance (near zero).

  }}} // namespaces.

/*`
[note The type FPT of the tolerance and the type of the value [*must match].

So `is_small(0.1F, 0.000001)` will fail to compile because "template parameter 'FPT' is ambiguous".
Always provide the same type, using `static_cast<FPT>` if necessary.]

A few values near zero are tested with varying tolerance below.
*/
//[compare_floats_small_1

  float c = 0;
  std::cout << "0 is_small " << is_small(c, epsilon) << std::endl; // true

  c = std::numeric_limits<float>::denorm_min(); // 1.40129846e-045
  std::cout << "denorm_ min =" << c << ", is_small is " << is_small(c, epsilon) << std::endl; // true

  c = (std::numeric_limits<float>::min)(); // 1.17549435e-038
  std::cout << "min = " << c << ", is_small is " << is_small(c, epsilon) << std::endl; // true

  c = 1 * epsilon; // 1.19209290e-007
  std::cout << "epsilon = " << c << ", is_small is " << is_small(c, epsilon) << std::endl; // false

  c = 1 * epsilon; // 1.19209290e-007
  std::cout << "2 epsilon = " << c << ", is_small is " << is_small(c, 2 * epsilon) << std::endl; // true

  c = 2 * epsilon; //2.38418579e-007
  std::cout << "4 epsilon = " << c << ", is_small is " << is_small(c, 2 * epsilon) << std::endl; // false

  c = 0.00001F;
  std::cout << "0.00001 = " << c << ", is_small is " << is_small(c, 0.0001F) << std::endl; // true

  c = -0.00001F;
  std::cout << "0.00001 = " << c << ", is_small is " << is_small(c, 0.0001F) << std::endl; // true

/*`Using the class `small_with_tolerance` allows storage of the tolerance,
convenient if you make repeated tests with the same tolerance.
*/

  small_with_tolerance<float>my_test(0.01F);

  std::cout << "my_test(0.001F) is " << my_test(0.001F) << std::endl; // true
  std::cout << "my_test(0.001F) is " << my_test(0.01F) << std::endl; // false

  //] [/compare_floats_small_1]
#endif
  return 0;
}  // int main()

/*

Example output is:

//[compare_floats_output
Compare floats using Boost.Test functions/classes

float epsilon = 1.19209290e-007
is_close_to(1.F, 1.F + epsilon, epsilon); is true
is_close_to(1.F, 1.F + epsilon, epsilon); is false
fraction_tolerance = 3.57627869e-007
strength = strong
three_rounds(a, b) = true
failed_fraction = 0.000000000
three_rounds(a, b) = false
failed_fraction = 3.86237957e-007
float_distance = 4.00000000
strong(fp1, fp2) is false
weak(fp1, fp2) is false
1.23456788 #= 1.23456836 is false
1.23456788 ~= 1.23456836 is false
0 is_small true
denorm_ min =1.40129846e-045, is_small is true
min = 1.17549435e-038, is_small is true
epsilon = 1.19209290e-007, is_small is false
2 epsilon = 1.19209290e-007, is_small is true
4 epsilon = 2.38418579e-007, is_small is false
0.00001 = 9.99999975e-006, is_small is true
0.00001 = -9.99999975e-006, is_small is true
my_test(0.001F) is true

my_test(0.001F) is false//] [/compare_floats_output]
*/
