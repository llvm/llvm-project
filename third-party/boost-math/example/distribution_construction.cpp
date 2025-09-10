// distribution_construction.cpp

// Copyright Paul A. Bristow 2007, 2010, 2012.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Caution: this file contains Quickbook markup as well as code
// and comments, don't change any of the special comment markups!

#ifdef _MSC_VER
#  pragma warning (disable : 4996) // disable -D_SCL_SECURE_NO_WARNINGS C++ 'Checked Iterators'
#endif

#include <iostream>
#include <exception>

//[distribution_construction_1

/*`
The structure of distributions is rather different from some other statistical libraries,
for example, those written in less object-oriented language like FORTRAN and C that
provide a few arguments to each free function.

Boost.Math library instead provides each distribution as a template C++ class.
A distribution is constructed with a few arguments, and then
member and non-member functions are used to find values of the
distribution, often a function of a random variate.

For this demonstration, first we need some includes to access the
negative binomial distribution (and the binomial, beta and gamma distributions too).

To demonstrate the use with a high precision User-defined floating-point type
`cpp_bin_float`, we also need an include from Boost.Multiprecision.
(We could equally well have used a `cpp_dec_float` multiprecision type).

We choose a typedef `cpp_bin_float_50` to provide a 50 decimal digit type,
but we could equally have chosen at 128-bit type `cpp_bin_float_quad`, 
or on some platforms `__float128`, providing about 35 decimal digits.
*/

#include <boost/math/distributions/negative_binomial.hpp> // for negative_binomial_distribution
  using boost::math::negative_binomial_distribution; // default type is double.
  using boost::math::negative_binomial; // typedef provides default type is double.
#include <boost/math/distributions/binomial.hpp> // for binomial_distribution.
#include <boost/math/distributions/beta.hpp> // for beta_distribution.
#include <boost/math/distributions/gamma.hpp> // for gamma_distribution.
#include <boost/math/distributions/normal.hpp> // for normal_distribution.

#include <boost/multiprecision/cpp_bin_float.hpp> // for cpp_bin_float_50
/*`
Several examples of constructing distributions follow:
*/
//] [/distribution_construction_1 end of Quickbook in C++ markup]

int main()
{
  try
  {
//[distribution_construction_2
/*`
First, a negative binomial distribution with 8 successes
and a success fraction 0.25, 25% or 1 in 4, is constructed like this:
*/
  boost::math::negative_binomial_distribution<double> mydist0(8., 0.25);
  /*`
  But this is inconveniently long, so we might be tempted to write
  */
  using namespace boost::math;
  /*`
  but this might risk ambiguity with names in `std random` so
  [*much] better is explicit `using boost::math::` statements, for example:
  */
  using boost::math::negative_binomial_distribution;
  /*`
  and we can still reduce typing.

  Since the vast majority of applications use will be using `double` precision,
  the template argument to the distribution (`RealType`) defaults
  to type `double`, so we can also write:
  */

  negative_binomial_distribution<> mydist9(8., 0.25); // Uses default `RealType = double`.

  /*`
  But the name `negative_binomial_distribution` is still inconveniently long,
  so, for most distributions, a convenience `typedef` is provided, for example:

     typedef negative_binomial_distribution<double> negative_binomial; // Reserved name of type double.

  [caution
  This convenience typedef is [*not provided] if a clash would occur
  with the name of a function; currently only `beta` and `gamma`
  fall into this category.
  ]

  So, after a using statement,
  */

  using boost::math::negative_binomial;

  /*`
  we have a convenient typedef to `negative_binomial_distribution<double>`:
  */
  negative_binomial mydist(8., 0.25);

  /*`
  Some more examples using the convenience typedef:
  */
  negative_binomial mydist10(5., 0.4); // Both arguments double.
  /*`
  And automatic conversion of arguments takes place, so you can use integers and floats:
  */
  negative_binomial mydist11(5, 0.4); // Using provided typedef of type double, and int and double arguments.
  /*`
  This is probably the most common usage.
  Other combination are possible too:
  */
  negative_binomial mydist12(5., 0.4F); // Double and float arguments.
  negative_binomial mydist13(5, 1); // Both arguments integer.

  /*`
  Similarly for most other distributions like the binomial.
  */
  binomial mybinomial(1, 0.5); // is more concise than
  binomial_distribution<> mybinomd1(1, 0.5);

  /*`
  For cases when the typdef distribution name would clash with a math special function
  (currently only beta and gamma)
  the typedef is deliberately not provided, and the longer version of the name
  must be used, so for example, do not use:

     using boost::math::beta;
     beta mybetad0(1, 0.5); // Error beta is a math FUNCTION!

  Which produces the error messages:

  [pre
  error C2146: syntax error : missing ';' before identifier 'mybetad0'
  warning C4551: function call missing argument list
  error C3861: 'mybetad0': identifier not found
  ]

  Instead you should use:
  */
  using boost::math::beta_distribution;
  beta_distribution<> mybetad1(1, 0.5);
  /*`
  or for the gamma distribution:
  */
  gamma_distribution<> mygammad1(1, 0.5);

  /*`
  We can, of course, still provide the type explicitly thus:
  */

  // Explicit double precision:  both arguments are double:
  negative_binomial_distribution<double>        mydist1(8., 0.25);

  // Explicit float precision, double arguments are truncated to float:
  negative_binomial_distribution<float>         mydist2(8., 0.25);

  // Explicit float precision, integer & double arguments converted to float:
  negative_binomial_distribution<float>         mydist3(8, 0.25);

  // Explicit float precision, float arguments, so no conversion:
  negative_binomial_distribution<float>         mydist4(8.F, 0.25F);

  // Explicit float precision, integer arguments promoted to float.
  negative_binomial_distribution<float>         mydist5(8, 1);

  // Explicit double precision:
  negative_binomial_distribution<double>        mydist6(5., 0.4);

  // Explicit long double precision:
  negative_binomial_distribution<long double>   mydist7(8., 0.25);
     
  /*`
  And you can use your own template RealType,
  for example, `boost::math::cpp_bin_float_50` (an arbitrary 50 decimal digits precision type),
  then we can write:
  */
  using namespace boost::multiprecision;
  negative_binomial_distribution<cpp_bin_float_50>  mydist8(8, 0.25);

  // `integer` arguments are promoted to your RealType exactly, but
  // `double` argument are converted to RealType,
  // most likely losing precision!
  
  // So DON'T be tempted to write the 'obvious':
  negative_binomial_distribution<cpp_bin_float_50>  mydist20(8, 0.23456789012345678901234567890);
 // to avoid truncation of second parameter to `0.2345678901234567` and loss of precision.

 // Instead pass a quoted decimal digit string:
  negative_binomial_distribution<cpp_bin_float_50>  mydist21(8, cpp_bin_float_50("0.23456789012345678901234567890") );

  // Ensure that all potentially significant digits are shown.
  std::cout.precision(std::numeric_limits<cpp_bin_float_50>::digits10);
  // 
  cpp_bin_float_50 x("1.23456789012345678901234567890");
  std::cout << pdf(mydist8, x) << std::endl;
/*` showing  0.00012630010495970320103876754721976419438231705359935
             0.00012630010495970320103876754721976419438231528547467

[warning When using multiprecision, it is all too easy to get accidental truncation!]

For example, if you write
*/
  std::cout << pdf(mydist8, 1.23456789012345678901234567890) << std::endl;
/*`
showing  0.00012630010495970318465064569310967179576805651692929,
which is wrong at about the 17th decimal digit!

This is because the value provided is truncated to a `double`, effectively
  `double x = 1.23456789012345678901234567890;`

Then the now `double x` is passed to function `pdf`,
and this truncated `double` value is finally promoted to `cpp_bin_float_50`.

Another way of quietly getting the wrong answer is to write:
*/
  std::cout << pdf(mydist8, cpp_bin_float_50(1.23456789012345678901234567890)) << std::endl;
/*`
A correct way from a multi-digit string value is
*/
  std::cout << pdf(mydist8, cpp_bin_float_50("1.23456789012345678901234567890")) << std::endl;
/*`

[tip Getting about 17 decimal digits followed by many zeros is often a sign of accidental truncation.]
*/

/*`
[h4 Default arguments to distribution constructors.]

Note that default constructor arguments are only provided for some distributions.
So if you wrongly assume a default argument, you will get an error message, for example:

   negative_binomial_distribution<> mydist8;

[pre error C2512 no appropriate default constructor available.]

No default constructors are provided for the `negative binomial` distribution,
because it is difficult to chose any sensible default values for this distribution.

For other distributions, like the normal distribution,
it is obviously very useful to provide 'standard'
defaults for the mean (zero) and standard deviation (unity) thus:

    normal_distribution(RealType mean = 0, RealType sd = 1);

So in this case we can more tersely write:
*/
  using boost::math::normal;

  normal norm1;       // Standard normal distribution N[0,1].
  normal norm2(2);    // Mean = 2, std deviation = 1.
  normal norm3(2, 3); // Mean = 2, std deviation = 3.

  }
  catch(std::exception &ex)
  {
    std::cout << ex.what() << std::endl;
  }

  return 0;
}  // int main()

/*`There is no useful output from this demonstration program, of course. */

//] [/end of distribution_construction_2]

/*
//[distribution_construction_output

  0.00012630010495970320103876754721976419438231705359935
  0.00012630010495970318465064569310967179576805651692929
  0.00012630010495970318465064569310967179576805651692929
  0.00012630010495970320103876754721976419438231705359935

//] [/distribution_construction_output]


  0.00012630010495970320103876754721976419438231528547467
  0.0001263001049597031846506456931096717957680547488046
  0.0001263001049597031846506456931096717957680547488046
  0.00012630010495970320103876754721976419438231528547467


*/



