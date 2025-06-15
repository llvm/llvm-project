// root_finding_fith.cpp

// Copyright Paul A. Bristow 2014.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of finding fifth root using Newton-Raphson, Halley, Schroder, TOMS748 .

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

// To get (copious!) diagnostic output, add make this define here or elsewhere.
//#define BOOST_MATH_INSTRUMENT


//[root_fifth_headers
/*
This example demonstrates how to use the Boost.Math tools for root finding,
taking the fifth root function (fifth_root) as an example.
It shows how use of derivatives can improve the speed.

First some includes that will be needed.
Using statements are provided to list what functions are being used in this example:
you can of course qualify the names in other ways.
*/

#include <boost/math/tools/roots.hpp>
using boost::math::policies::policy;
using boost::math::tools::newton_raphson_iterate;
using boost::math::tools::halley_iterate;
using boost::math::tools::eps_tolerance; // Binary functor for specified number of bits.
using boost::math::tools::bracket_and_solve_root;
using boost::math::tools::toms748_solve;

#include <boost/math/special_functions/next.hpp>

#include <tuple>
#include <utility> // pair, make_pair

//] [/root_finding_headers]

#include <iostream>
using std::cout; using std::endl;
#include <iomanip>
using std::setw; using std::setprecision;
#include <limits>
using std::numeric_limits;

/*
//[root_finding_fifth_1
Let's suppose we want to find the fifth root of a number.

The equation we want to solve is:

__spaces ['f](x) = x[fifth]

We will first solve this without using any information
about the slope or curvature of the fifth function.

If your differentiation is a little rusty
(or you are faced with an equation whose complexity is daunting,
then you can get help, for example from the invaluable

http://www.wolframalpha.com/ site

entering the command

  differentiate x^5

or the Wolfram Language command

  D[x^5, x]

gives the output

  d/dx(x^5) = 5 x^4

and to get the second differential, enter

  second differentiate x^5

or the Wolfram Language

  D[x^5, {x, 2}]

to get the output

  d^2/dx^2(x^5) = 20 x^3

or

  20 x^3

To get a reference value we can enter

  fifth root 3126

or

  N[3126^(1/5), 50]

to get a result with a precision of  50 decimal digits

  5.0003199590478625588206333405631053401128722314376

(We could also get a reference value using Boost.Multiprecision).

We then show how adding what we can know, for this function, about the slope,
the 1st derivation /f'(x)/, will speed homing in on the solution,
and then finally how adding the curvature /f''(x)/ as well will improve even more.

The 1st and 2nd derivatives of x[fifth] are:

__spaces ['f]\'(x) = 2x[sup2]

__spaces ['f]\'\'(x) = 6x

*/

//] [/root_finding_fifth_1]

//[root_finding_fifth_functor_noderiv

template <class T>
struct fifth_functor_noderiv
{ // fifth root of x using only function - no derivatives.
  fifth_functor_noderiv(T const& to_find_root_of) : value(to_find_root_of)
  { // Constructor stores value to find root of.
    // For example: calling fifth_functor<T>(x) to get fifth root of x.
  }
  T operator()(T const& x)
  { //! \returns f(x) - value.
    T fx = x*x*x*x*x - value; // Difference (estimate x^5 - value).
    return fx;
  }
private:
  T value; // to be 'fifth_rooted'.
};

//] [/root_finding_fifth_functor_noderiv]

//cout  << ", std::numeric_limits<" << typeid(T).name()  << ">::digits = " << digits
//   << ", accuracy " << get_digits << " bits."<< endl;


/*`Implementing the fifth root function itself is fairly trivial now:
the hardest part is finding a good approximation to begin with.
In this case we'll just divide the exponent by five.
(There are better but more complex guess algorithms used in 'real-life'.)

fifth root function is 'Really Well Behaved' in that it is monotonic
and has only one root
(we leave negative values 'as an exercise for the student').
*/

//[root_finding_fifth_noderiv

template <class T>
T fifth_noderiv(T x)
{ //! \returns fifth root of x using bracket_and_solve (no derivatives).
  using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools; // For bracket_and_solve_root.

  int exponent;
  frexp(x, &exponent); // Get exponent of z (ignore mantissa).
  T guess = ldexp(1., exponent / 5); // Rough guess is to divide the exponent by five.
  T factor = 2; // To multiply and divide guess to bracket.
  // digits used to control how accurate to try to make the result.
  // int digits =  3 * std::numeric_limits<T>::digits / 4; // 3/4 maximum possible binary digits accuracy for type T.
  int digits = std::numeric_limits<T>::digits; // Maximum possible binary digits accuracy for type T.

  //std::uintmax_t maxit = (std::numeric_limits<std::uintmax_t>::max)();
  // (std::numeric_limits<std::uintmax_t>::max)() = 18446744073709551615
  // which is more than anyone might wish to wait for!!!
  // so better to choose some reasonable estimate of how many iterations may be needed.

  const std::uintmax_t maxit = 50; // Chosen max iterations,
  // but updated on exit with actual iteration count.

  // We could also have used a maximum iterations provided by any policy:
  // std::uintmax_t max_it = policies::get_max_root_iterations<Policy>();

  std::uintmax_t it = maxit; // Initially our chosen max iterations,

  bool is_rising = true; // So if result if guess^5 is too low, try increasing guess.
  eps_tolerance<double> tol(digits);
  std::pair<T, T> r =
    bracket_and_solve_root(fifth_functor_noderiv<T>(x), guess, factor, is_rising, tol, it);
  // because the iteration count is updating,
  // you can't call with a literal maximum iterations value thus:
  //bracket_and_solve_root(fifth_functor_noderiv<T>(x), guess, factor, is_rising, tol, 20);

  // Can show how many iterations (this information is lost outside fifth_noderiv).
  cout << "Iterations " << it << endl;
  if (it >= maxit)
  { // Failed to converge (or is jumping between bracket values).
    cout << "Unable to locate solution in chosen iterations:"
      " Current best guess is between " << r.first << " and " << r.second << endl;
  }
  T distance = boost::math::float_distance(r.first, r.second);
  if (distance > 0)
  { //
    std::cout << distance << " bits separate the bracketing values." << std::endl;
    for (int i = 0; i < distance; i++)
    { // Show all the values within the bracketing values.
      std::cout << boost::math::float_advance(r.first, i) << std::endl;
    }
  }
  else
  { // distance == 0 and  r.second == r.first
    std::cout << "Converged to a single value " << r.first << std::endl;
  }

  return r.first + (r.second - r.first) / 2;  // return midway between bracketed interval.
} // T fifth_noderiv(T x)

//] [/root_finding_fifth_noderiv]



// maxit = 10
// Unable to locate solution in chosen iterations: Current best guess is between 3.0365889718756613 and 3.0365889718756627


/*`
We now solve the same problem, but using more information about the function,
to show how this can speed up finding the best estimate of the root.

For this function, the 1st differential (the slope of the tangent to a curve at any point) is known.

[@http://en.wikipedia.org/wiki/Derivative#Derivatives_of_elementary_functions Derivatives]
gives some reminders.

Using the rule that the derivative of x^n for positive n (actually all nonzero n) is nx^n-1,
allows use to get the 1st differential as 3x^2.

To see how this extra information is used to find the root, view this demo:
[@http://en.wikipedia.org/wiki/Newton%27s_methodNewton Newton-Raphson iterations].

We need to define a different functor that returns
both the evaluation of the function to solve, along with its first derivative:

To \'return\' two values, we use a pair of floating-point values:
*/

//[root_finding_fifth_functor_1stderiv

template <class T>
struct fifth_functor_1stderiv
{ // Functor returning function and 1st derivative.

  fifth_functor_1stderiv(T const& target) : value(target)
  { // Constructor stores the value to be 'fifth_rooted'.
  }

  std::pair<T, T> operator()(T const& z) // z is best estimate so far.
  { // Return both f(x) and first derivative f'(x).
    T fx = z*z*z*z*z - value; // Difference estimate fx = x^5 - value.
    T d1x = 5 * z*z*z*z; // 1st derivative d1x = 5x^4.
    return std::make_pair(fx, d1x); // 'return' both fx and d1x.
  }
private:
  T value; // to be 'fifth_rooted'.
}; // fifth_functor_1stderiv

//] [/root_finding_fifth_functor_1stderiv]


/*`Our fifth root function using fifth_functor_1stderiv is now:*/

//[root_finding_fifth_1deriv

template <class T>
T fifth_1deriv(T x)
{ //! \return fifth root of x using 1st derivative and Newton_Raphson.
  using namespace std; // For frexp, ldexp, numeric_limits.
  using namespace boost::math::tools; // For newton_raphson_iterate.

  int exponent;
  frexp(x, &exponent); // Get exponent of x (ignore mantissa).
  T guess = ldexp(1., exponent / 5); // Rough guess is to divide the exponent by three.
  // Set an initial bracket interval.
  T min = ldexp(0.5, exponent / 5); // Minimum possible value is half our guess.
  T max = ldexp(2., exponent / 5);// Maximum possible value is twice our guess.

  // digits used to control how accurate to try to make the result.
  int digits = std::numeric_limits<T>::digits; // Maximum possible binary digits accuracy for type T.

  const std::uintmax_t maxit = 20; // Optionally limit the number of iterations.
  std::uintmax_t it = maxit; // limit the number of iterations.
  //cout << "Max Iterations " << maxit << endl; //
  T result = newton_raphson_iterate(fifth_functor_1stderiv<T>(x), guess, min, max, digits, it);
  // Can check and show how many iterations (updated by newton_raphson_iterate).
  cout << it << " iterations (from max of " << maxit << ")" << endl;
  return result;
} // fifth_1deriv

//] [/root_finding_fifth_1deriv]

//  int get_digits = (digits * 2) /3; // Two thirds of maximum possible accuracy.

//std::uintmax_t maxit = (std::numeric_limits<std::uintmax_t>::max)();
// the default (std::numeric_limits<std::uintmax_t>::max)() = 18446744073709551615
// which is more than we might wish to wait for!!!  so we can reduce it

/*`
Finally need to define yet another functor that returns
both the evaluation of the function to solve,
along with its first and second derivatives:

f''(x) = 3 * 3x

To \'return\' three values, we use a tuple of three floating-point values:
*/

//[root_finding_fifth_functor_2deriv

template <class T>
struct fifth_functor_2deriv
{ // Functor returning both 1st and 2nd derivatives.
  fifth_functor_2deriv(T const& to_find_root_of) : value(to_find_root_of)
  { // Constructor stores value to find root of, for example:
  }

  // using boost::math::tuple; // to return three values.
  std::tuple<T, T, T> operator()(T const& x)
  { // Return both f(x) and f'(x) and f''(x).
    T fx = x*x*x*x*x - value; // Difference (estimate x^3 - value).
    T dx = 5 * x*x*x*x; // 1st derivative = 5x^4.
    T d2x = 20 * x*x*x; // 2nd derivative = 20 x^3
    return std::make_tuple(fx, dx, d2x); // 'return' fx, dx and d2x.
  }
private:
  T value; // to be 'fifth_rooted'.
}; // struct fifth_functor_2deriv

//] [/root_finding_fifth_functor_2deriv]


/*`Our fifth function is now:*/

//[root_finding_fifth_2deriv

template <class T>
T fifth_2deriv(T x)
{ // return fifth root of x using 1st and 2nd derivatives and Halley.
  using namespace std;  // Help ADL of std functions.
  using namespace boost::math; // halley_iterate

  int exponent;
  frexp(x, &exponent); // Get exponent of z (ignore mantissa).
  T guess = ldexp(1., exponent / 5); // Rough guess is to divide the exponent by three.
  T min = ldexp(0.5, exponent / 5); // Minimum possible value is half our guess.
  T max = ldexp(2., exponent / 5); // Maximum possible value is twice our guess.

  int digits = std::numeric_limits<T>::digits / 2; // Half maximum possible binary digits accuracy for type T.
  const std::uintmax_t maxit = 50;
  std::uintmax_t it = maxit;
  T result = halley_iterate(fifth_functor_2deriv<T>(x), guess, min, max, digits, it);
  // Can show how many iterations (updated by halley_iterate).
  cout << it << " iterations (from max of " << maxit << ")" << endl;

  return result;
} // fifth_2deriv(x)

//] [/root_finding_fifth_2deriv]

int main()
{

  //[root_finding_example_1
  cout << "fifth Root finding (fifth) Example." << endl;
  // Show all possibly significant decimal digits.
  cout.precision(std::numeric_limits<double>::max_digits10);
  // or use   cout.precision(max_digits10 = 2 + std::numeric_limits<double>::digits * 3010/10000);
  try
  { // Always use try'n'catch blocks with Boost.Math to get any error messages.

    double v27 = 3125; // Example of a value that has an exact integer fifth root.
    // exact value of fifth root is exactly 5.

    std::cout << "Fifth root  of " << v27 << " is " << 5 << std::endl;

    double v28 = v27+1; // Example of a value whose fifth root is *not* exactly representable.
    // Value of fifth root is 5.0003199590478625588206333405631053401128722314376 (50 decimal digits precision)
    // and to std::numeric_limits<double>::max_digits10 double precision (usually 17) is

    double root5v2 = static_cast<double>(5.0003199590478625588206333405631053401128722314376);

    std::cout << "Fifth root  of " << v28 << " is "  << root5v2 << std::endl;

    // Using bracketing:
    double r = fifth_noderiv(v27);
    cout << "fifth_noderiv(" << v27 << ") = " << r << endl;

    r = fifth_noderiv(v28);
    cout << "fifth_noderiv(" << v28 << ") = " << r << endl;

    // Using 1st differential Newton-Raphson:
    r = fifth_1deriv(v27);
    cout << "fifth_1deriv(" << v27 << ") = " << r << endl;
    r = fifth_1deriv(v28);
    cout << "fifth_1deriv(" << v28 << ") = " << r << endl;

    // Using Halley with 1st and 2nd differentials.
    r = fifth_2deriv(v27);
    cout << "fifth_2deriv(" << v27 << ") = " << r << endl;
    r = fifth_2deriv(v28);
    cout << "fifth_2deriv(" << v28 << ") = " << r << endl;
  }
  catch (const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
  //] [/root_finding_example_1
  return 0;
}  // int main()

//[root_finding_example_output
/*`
Normal output is:

[pre
1>  Description: Autorun "J:\Cpp\MathToolkit\test\Math_test\Release\root_finding_fifth.exe"
1>  fifth Root finding (fifth) Example.
1>  Fifth root  of 3125 is 5
1>  Fifth root  of 3126 is 5.0003199590478626
1>  Iterations 10
1>  Converged to a single value 5
1>  fifth_noderiv(3125) = 5
1>  Iterations 11
1>  2 bits separate the bracketing values.
1>  5.0003199590478609
1>  5.0003199590478618
1>  fifth_noderiv(3126) = 5.0003199590478618
1>  6 iterations (from max of 20)
1>  fifth_1deriv(3125) = 5
1>  7 iterations (from max of 20)
1>  fifth_1deriv(3126) = 5.0003199590478626
1>  4 iterations (from max of 50)
1>  fifth_2deriv(3125) = 5
1>  4 iterations (from max of 50)
1>  fifth_2deriv(3126) = 5.0003199590478626
[/pre]

to get some (much!) diagnostic output we can add

#define BOOST_MATH_INSTRUMENT

[pre
1>  fifth Root finding (fifth) Example.
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:537 a = 4 b = 8 fa = -2101 fb = 29643 count = 18
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:340  a = 4.264742943548387 b = 8
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:352  a = 4.264742943548387 b = 5.1409225585147951
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:259  a = 4.264742943548387 b = 5.1409225585147951 d = 8 e = 4 fa = -1714.2037505671719 fb = 465.91652114644285 fd = 29643 fe = -2101
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:267 q11 = -3.735257056451613 q21 = -0.045655399937094755 q31 = 0.68893005658139972 d21 = -2.9047328414222999 d31 = -0.18724955838500826
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:275 q22 = -0.15074699539567221 q32 = 0.007740525571111408 d32 = -0.13385363287680208 q33 = 0.074868009790687237 c = 5.0362815354915851
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:388  a = 4.264742943548387 b = 5.0362815354915851
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:259  a = 4.264742943548387 b = 5.0362815354915851 d = 5.1409225585147951 e = 8 fa = -1714.2037505671719 fb = 115.03721886368339 fd = 465.91652114644285 fe = 29643
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:267 q11 = -0.045655399937094755 q21 = -0.034306988726112195 q31 = 0.7230181097615842 d21 = -0.1389480117493222 d31 = -0.048520482181613811
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:275 q22 = -0.00036345624935100459 q32 = 0.011175908093791367 d32 = -0.0030375853617102483 q33 = 0.00014618657296010219 c = 4.999083147976723
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:408  a = 4.999083147976723 b = 5.0362815354915851
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:433  a = 4.999083147976723 b = 5.0008904277935091
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:434  tol = -0.00036152225583956088
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:259  a = 4.999083147976723 b = 5.0008904277935091 d = 5.0362815354915851 e = 4.264742943548387 fa = -2.8641119933622576 fb = 2.7835781082976609 fd = 115.03721886368339 fe = -1714.2037505671719
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:267 q11 = -0.048520482181613811 q21 = -0.00087760104664616457 q31 = 0.00091652546535745522 d21 = -0.036268708744722128 d31 = -0.00089075435142862297
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:275 q22 = -1.9862562616034592e-005 q32 = 3.1952597740788757e-007 d32 = -1.2833778805050512e-005 q33 = 1.1763429980834706e-008 c = 5.0000000047314881
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:388  a = 4.999083147976723 b = 5.0000000047314881
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:259  a = 4.999083147976723 b = 5.0000000047314881 d = 5.0008904277935091 e = 5.0362815354915851 fa = -2.8641119933622576 fb = 1.4785900475544622e-005 fd = 2.7835781082976609 fe = 115.03721886368339
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:267 q11 = -0.00087760104664616457 q21 = -4.7298032238887272e-009 q31 = 0.00091685202154135855 d21 = -0.00089042779182425238 d31 = -4.7332236912279757e-009
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:275 q22 = -1.6486403607318402e-012 q32 = 1.7346209428817704e-012 d32 = -1.6858463963666777e-012 q33 = 9.0382569995250912e-016 c = 5
1>  I:\modular-boost\boost/math/tools/toms748_solve.hpp:592 max_iter = 10 count = 7
1>  Iterations 20
1>  0 bits separate brackets.
1>  fifth_noderiv(3125) = 5
]
*/
//] [/root_finding_example_output]
