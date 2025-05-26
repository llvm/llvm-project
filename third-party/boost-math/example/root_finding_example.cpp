// root_finding_example.cpp

// Copyright Paul A. Bristow 2010, 2015

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of finding roots using Newton-Raphson, Halley.

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//#define BOOST_MATH_INSTRUMENT

/*
This example demonstrates how to use the various tools for root finding
taking the simple cube root function (`cbrt`) as an example.

It shows how use of derivatives can improve the speed.
(But is only a demonstration and does not try to make the ultimate improvements of 'real-life'
implementation of `boost::math::cbrt`, mainly by using a better computed initial 'guess'
at `<boost/math/special_functions/cbrt.hpp>`).

Then we show how a higher root (fifth) can be computed,
and in `root_finding_n_example.cpp` a generic method
for the ['n]th root that constructs the derivatives at compile-time,

These methods should be applicable to other functions that can be differentiated easily.

First some `#includes` that will be needed.

[tip For clarity, `using` statements are provided to list what functions are being used in this example:
you can of course partly or fully qualify the names in other ways.
(For your application, you may wish to extract some parts into header files,
but you should never use `using` statements globally in header files).]
*/

//[root_finding_include_1

#include <boost/math/tools/roots.hpp>
//using boost::math::policies::policy;
//using boost::math::tools::newton_raphson_iterate;
//using boost::math::tools::halley_iterate; //
//using boost::math::tools::eps_tolerance; // Binary functor for specified number of bits.
//using boost::math::tools::bracket_and_solve_root;
//using boost::math::tools::toms748_solve;

#include <boost/math/special_functions/next.hpp> // For float_distance.
#include <tuple> // for std::tuple and std::make_tuple.
#include <boost/math/special_functions/cbrt.hpp> // For boost::math::cbrt.
#include <boost/math/special_functions/pow.hpp> // boost::math::pow<5,double>

//] [/root_finding_include_1]

// using boost::math::tuple;
// using boost::math::make_tuple;
// using boost::math::tie;
// which provide convenient aliases for various implementations,
// including std::tr1, depending on what is available.

#include <iostream>
//using std::cout; using std::endl;
#include <iomanip>
//using std::setw; using std::setprecision;
#include <limits>
//using std::numeric_limits;

/*

Let's suppose we want to find the root of a number ['a], and to start, compute the cube root.

So the equation we want to solve is:

__spaces ['f](x) = x[cubed] - a

We will first solve this without using any information
about the slope or curvature of the cube root function.

We then show how adding what we can know about this function, first just the slope,
the 1st derivation /f'(x)/, will speed homing in on the solution.

Lastly we show how adding the curvature /f''(x)/ too will speed convergence even more.

*/

//[root_finding_noderiv_1

template <class T>
struct cbrt_functor_noderiv
{ 
  //  cube root of x using only function - no derivatives.
  cbrt_functor_noderiv(T const& to_find_root_of) : a(to_find_root_of)
  { /* Constructor just stores value a to find root of. */ }
  T operator()(T const& x)
  {
    T fx = x*x*x - a; // Difference (estimate x^3 - a).
    return fx;
  }
private:
  T a; // to be 'cube_rooted'.
};
//] [/root_finding_noderiv_1

/*
Implementing the cube root function itself is fairly trivial now:
the hardest part is finding a good approximation to begin with.
In this case we'll just divide the exponent by three.
(There are better but more complex guess algorithms used in 'real-life'.)

Cube root function is 'Really Well Behaved' in that it is monotonic
and has only one root (we leave negative values 'as an exercise for the student').
*/

//[root_finding_noderiv_2

template <class T>
T cbrt_noderiv(T x)
{ 
  // return cube root of x using bracket_and_solve (no derivatives).
  using namespace std;                          // Help ADL of std functions.
  using namespace boost::math::tools;           // For bracket_and_solve_root.

  int exponent;
  frexp(x, &exponent);                          // Get exponent of z (ignore mantissa).
  T guess = ldexp(1., exponent/3);              // Rough guess is to divide the exponent by three.
  T factor = 2;                                 // How big steps to take when searching.

  const std::uintmax_t maxit = 20;            // Limit to maximum iterations.
  std::uintmax_t it = maxit;                  // Initially our chosen max iterations, but updated with actual.
  bool is_rising = true;                        // So if result if guess^3 is too low, then try increasing guess.
  int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  // Some fraction of digits is used to control how accurate to try to make the result.
  int get_digits = digits - 3;                  // We have to have a non-zero interval at each step, so
                                                // maximum accuracy is digits - 1.  But we also have to
                                                // allow for inaccuracy in f(x), otherwise the last few
                                                // iterations just thrash around.
  eps_tolerance<T> tol(get_digits);             // Set the tolerance.
  std::pair<T, T> r = bracket_and_solve_root(cbrt_functor_noderiv<T>(x), guess, factor, is_rising, tol, it);
  return r.first + (r.second - r.first)/2;      // Midway between brackets is our result, if necessary we could
                                                // return the result as an interval here.
}

/*`

[note The final parameter specifying a maximum number of iterations is optional.
However, it defaults to `std::uintmax_t maxit = (std::numeric_limits<std::uintmax_t>::max)();`
which is `18446744073709551615` and is more than anyone would wish to wait for!

So it may be wise to chose some reasonable estimate of how many iterations may be needed, 
In this case the function is so well behaved that we can chose a low value of 20.

Internally when Boost.Math uses these functions, it sets the maximum iterations to
`policies::get_max_root_iterations<Policy>();`.]

Should we have wished we can show how many iterations were used in `bracket_and_solve_root` 
(this information is lost outside `cbrt_noderiv`), for example with:

  if (it >= maxit)
  {
    std::cout << "Unable to locate solution in " << maxit << " iterations:"
      " Current best guess is between " << r.first << " and " << r.second << std::endl;
  }
  else
  {
    std::cout << "Converged after " << it << " (from maximum of " << maxit << " iterations)." << std::endl;
  }

for output like

  Converged after 11 (from maximum of 20 iterations).
*/
//] [/root_finding_noderiv_2]


// Cube root with 1st derivative (slope)

/*
We now solve the same problem, but using more information about the function,
to show how this can speed up finding the best estimate of the root.

For the root function, the 1st differential (the slope of the tangent to a curve at any point) is known.

If you need some reminders then
[@http://en.wikipedia.org/wiki/Derivative#Derivatives_of_elementary_functions Derivatives of elementary functions]
may help.

Using the rule that the derivative of ['x[super n]] for positive n (actually all nonzero n) is ['n x[super n-1]],
allows us to get the 1st differential as ['3x[super 2]].

To see how this extra information is used to find a root, view
[@http://en.wikipedia.org/wiki/Newton%27s_method Newton-Raphson iterations]
and the [@http://en.wikipedia.org/wiki/Newton%27s_method#mediaviewer/File:NewtonIteration_Ani.gif animation].

We need to define a different functor `cbrt_functor_deriv` that returns
both the evaluation of the function to solve, along with its first derivative:

To \'return\' two values, we use a `std::pair` of floating-point values
(though we could equally have used a std::tuple):
*/

//[root_finding_1_deriv_1

template <class T>
struct cbrt_functor_deriv
{ // Functor also returning 1st derivative.
  cbrt_functor_deriv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor stores value a to find root of,
    // for example: calling cbrt_functor_deriv<T>(a) to use to get cube root of a.
  }
  std::pair<T, T> operator()(T const& x)
  { 
    // Return both f(x) and f'(x).
    T fx = x*x*x - a;                // Difference (estimate x^3 - value).
    T dx =  3 * x*x;                 // 1st derivative = 3x^2.
    return std::make_pair(fx, dx);   // 'return' both fx and dx.
  }
private:
  T a;                               // Store value to be 'cube_rooted'.
};

/*`Our cube root function is now:*/

template <class T>
T cbrt_deriv(T x)
{ 
  // return cube root of x using 1st derivative and Newton_Raphson.
  using namespace boost::math::tools;
  int exponent;
  frexp(x, &exponent);                                // Get exponent of z (ignore mantissa).
  T guess = ldexp(1., exponent/3);                    // Rough guess is to divide the exponent by three.
  T min = ldexp(0.5, exponent/3);                     // Minimum possible value is half our guess.
  T max = ldexp(2., exponent/3);                      // Maximum possible value is twice our guess.
  const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  int get_digits = static_cast<int>(digits * 0.6);    // Accuracy doubles with each step, so stop when we have
                                                      // just over half the digits correct.
  const std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = newton_raphson_iterate(cbrt_functor_deriv<T>(x), guess, min, max, get_digits, it);
  return result;
}

//] [/root_finding_1_deriv_1]


/*
[h3:cbrt_2_derivatives Cube root with 1st & 2nd derivative (slope & curvature)]

Finally we define yet another functor `cbrt_functor_2deriv` that returns
both the evaluation of the function to solve,
along with its first *and second* derivatives:

__spaces[''f](x) = 6x

To \'return\' three values, we use a `tuple` of three floating-point values:
*/

//[root_finding_2deriv_1

template <class T>
struct cbrt_functor_2deriv
{ 
  // Functor returning both 1st and 2nd derivatives.
  cbrt_functor_2deriv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor stores value a to find root of, for example:
    // calling cbrt_functor_2deriv<T>(x) to get cube root of x,
  }
  std::tuple<T, T, T> operator()(T const& x)
  { 
    // Return both f(x) and f'(x) and f''(x).
    T fx = x*x*x - a;                     // Difference (estimate x^3 - value).
    T dx = 3 * x*x;                       // 1st derivative = 3x^2.
    T d2x = 6 * x;                        // 2nd derivative = 6x.
    return std::make_tuple(fx, dx, d2x);  // 'return' fx, dx and d2x.
  }
private:
  T a; // to be 'cube_rooted'.
};

/*`Our cube root function is now:*/

template <class T>
T cbrt_2deriv(T x)
{ 
  // return cube root of x using 1st and 2nd derivatives and Halley.
  //using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools;
  int exponent;
  frexp(x, &exponent);                                // Get exponent of z (ignore mantissa).
  T guess = ldexp(1., exponent/3);                    // Rough guess is to divide the exponent by three.
  T min = ldexp(0.5, exponent/3);                     // Minimum possible value is half our guess.
  T max = ldexp(2., exponent/3);                      // Maximum possible value is twice our guess.
  const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  // digits used to control how accurate to try to make the result.
  int get_digits = static_cast<int>(digits * 0.4);    // Accuracy triples with each step, so stop when just
                                                      // over one third of the digits are correct.
  std::uintmax_t maxit = 20;
  T result = halley_iterate(cbrt_functor_2deriv<T>(x), guess, min, max, get_digits, maxit);
  return result;
}

//] [/root_finding_2deriv_1]

//[root_finding_2deriv_lambda

template <class T>
T cbrt_2deriv_lambda(T x)
{
   // return cube root of x using 1st and 2nd derivatives and Halley.
   //using namespace std;  // Help ADL of std functions.
   using namespace boost::math::tools;
   int exponent;
   frexp(x, &exponent);                                // Get exponent of z (ignore mantissa).
   T guess = ldexp(1., exponent / 3);                    // Rough guess is to divide the exponent by three.
   T min = ldexp(0.5, exponent / 3);                     // Minimum possible value is half our guess.
   T max = ldexp(2., exponent / 3);                      // Maximum possible value is twice our guess.
   const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
   // digits used to control how accurate to try to make the result.
   int get_digits = static_cast<int>(digits * 0.4);    // Accuracy triples with each step, so stop when just
   // over one third of the digits are correct.
   std::uintmax_t maxit = 20;
   T result = halley_iterate(
      // lambda function:
      [x](const T& g){ return std::make_tuple(g * g * g - x, 3 * g * g, 6 * g); }, 
      guess, min, max, get_digits, maxit);
   return result;
}

//] [/root_finding_2deriv_lambda]
/*

[h3 Fifth-root function]
Let's now suppose we want to find the [*fifth root] of a number ['a].

The equation we want to solve is :

__spaces['f](x) = x[super 5] - a

If your differentiation is a little rusty
(or you are faced with an equation whose complexity is daunting),
then you can get help, for example from the invaluable
[@http://www.wolframalpha.com/ WolframAlpha site.]

For example, entering the command: `differentiate x ^ 5`

or the Wolfram Language command: ` D[x ^ 5, x]`

gives the output: `d/dx(x ^ 5) = 5 x ^ 4`

and to get the second differential, enter: `second differentiate x ^ 5`

or the Wolfram Language command: `D[x ^ 5, { x, 2 }]`

to get the output: `d ^ 2 / dx ^ 2(x ^ 5) = 20 x ^ 3`

To get a reference value, we can enter: [^fifth root 3126]

or: `N[3126 ^ (1 / 5), 50]`

to get a result with a precision of 50 decimal digits:

5.0003199590478625588206333405631053401128722314376

(We could also get a reference value using Boost.Multiprecision - see below).

The 1st and 2nd derivatives of x[super 5] are:

__spaces['f]\'(x) = 5x[super 4]

__spaces['f]\'\'(x) = 20x[super 3]

*/

//[root_finding_fifth_1
//] [/root_finding_fifth_1]


//[root_finding_fifth_functor_2deriv

/*`Using these expressions for the derivatives, the functor is:
*/

template <class T>
struct fifth_functor_2deriv
{ 
  // Functor returning both 1st and 2nd derivatives.
  fifth_functor_2deriv(T const& to_find_root_of) : a(to_find_root_of)
  { /* Constructor stores value a to find root of, for example: */ }

  std::tuple<T, T, T> operator()(T const& x)
  { 
    // Return both f(x) and f'(x) and f''(x).
    T fx = boost::math::pow<5>(x) - a;    // Difference (estimate x^3 - value).
    T dx = 5 * boost::math::pow<4>(x);    // 1st derivative = 5x^4.
    T d2x = 20 * boost::math::pow<3>(x);  // 2nd derivative = 20 x^3
    return std::make_tuple(fx, dx, d2x);  // 'return' fx, dx and d2x.
  }
private:
  T a;                                    // to be 'fifth_rooted'.
}; // struct fifth_functor_2deriv

//] [/root_finding_fifth_functor_2deriv]

//[root_finding_fifth_2deriv

/*`Our fifth-root function is now:
*/

template <class T>
T fifth_2deriv(T x)
{ 
  // return fifth root of x using 1st and 2nd derivatives and Halley.
  using namespace std;                  // Help ADL of std functions.
  using namespace boost::math::tools;   // for halley_iterate.

  int exponent;
  frexp(x, &exponent);                  // Get exponent of z (ignore mantissa).
  T guess = ldexp(1., exponent / 5);    // Rough guess is to divide the exponent by five.
  T min = ldexp(0.5, exponent / 5);     // Minimum possible value is half our guess.
  T max = ldexp(2., exponent / 5);      // Maximum possible value is twice our guess.
  // Stop when slightly more than one of the digits are correct:
  const int digits = static_cast<int>(std::numeric_limits<T>::digits * 0.4); 
  const std::uintmax_t maxit = 50;
  std::uintmax_t it = maxit;
  T result = halley_iterate(fifth_functor_2deriv<T>(x), guess, min, max, digits, it);
  return result;
}

//] [/root_finding_fifth_2deriv]


int main()
{
  std::cout << "Root finding  Examples." << std::endl;
  std::cout.precision(std::numeric_limits<double>::max_digits10);
  // Show all possibly significant decimal digits for double.
  // std::cout.precision(std::numeric_limits<double>::digits10);
  // Show all guaranteed significant decimal digits for double.


//[root_finding_main_1
  try
  {
    double threecubed = 27.;   // Value that has an *exactly representable* integer cube root.
    double threecubedp1 = 28.; // Value whose cube root is *not* exactly representable.

    std::cout << "cbrt(28) " << boost::math::cbrt(28.) << std::endl; // boost::math:: version of cbrt.
    std::cout << "std::cbrt(28) " << std::cbrt(28.) << std::endl;    // std:: version of cbrt.
    std::cout <<" cast double " << static_cast<double>(3.0365889718756625194208095785056696355814539772481111) << std::endl;

    // Cube root using bracketing:
    double r = cbrt_noderiv(threecubed);
    std::cout << "cbrt_noderiv(" << threecubed << ") = " << r << std::endl;
    r = cbrt_noderiv(threecubedp1);
    std::cout << "cbrt_noderiv(" << threecubedp1 << ") = " << r << std::endl;
//] [/root_finding_main_1]
    //[root_finding_main_2

    // Cube root using 1st differential Newton-Raphson:
    r = cbrt_deriv(threecubed);
    std::cout << "cbrt_deriv(" << threecubed << ") = " << r << std::endl;
    r = cbrt_deriv(threecubedp1);
    std::cout << "cbrt_deriv(" << threecubedp1 << ") = " << r << std::endl;

    // Cube root using Halley with 1st and 2nd differentials.
    r = cbrt_2deriv(threecubed);
    std::cout << "cbrt_2deriv(" << threecubed << ") = " << r << std::endl;
    r = cbrt_2deriv(threecubedp1);
    std::cout << "cbrt_2deriv(" << threecubedp1 << ") = " << r << std::endl;

    // Cube root using lambda's:
    r = cbrt_2deriv_lambda(threecubed);
    std::cout << "cbrt_2deriv(" << threecubed << ") = " << r << std::endl;
    r = cbrt_2deriv_lambda(threecubedp1);
    std::cout << "cbrt_2deriv(" << threecubedp1 << ") = " << r << std::endl;

    // Fifth root.

    double fivepowfive = 3125; // Example of a value that has an exact integer fifth root.
    // Exact value of fifth root is exactly 5.
    std::cout << "Fifth root  of " << fivepowfive << " is " << 5 << std::endl;

    double fivepowfivep1 = fivepowfive + 1; // Example of a value whose fifth root is *not* exactly representable.
    // Value of fifth root is 5.0003199590478625588206333405631053401128722314376 (50 decimal digits precision)
    // and to std::numeric_limits<double>::max_digits10 double precision (usually 17) is

    double root5v2 = static_cast<double>(5.0003199590478625588206333405631053401128722314376);
        std::cout << "Fifth root  of " << fivepowfivep1 << " is " << root5v2 << std::endl;

    // Using Halley with 1st and 2nd differentials.
    r = fifth_2deriv(fivepowfive);
    std::cout << "fifth_2deriv(" << fivepowfive << ") = " << r << std::endl;
    r = fifth_2deriv(fivepowfivep1);
    std::cout << "fifth_2deriv(" << fivepowfivep1 << ") = " << r << std::endl;
//] [/root_finding_main_?]
  }
  catch(const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
  return 0;
}  // int main()

//[root_finding_example_output
/*`
Normal output is:

[pre
  root_finding_example.cpp
  Generating code
  Finished generating code
  root_finding_example.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\root_finding_example.exe
  Cube Root finding (cbrt) Example.
  Iterations 10
  cbrt_1(27) = 3
  Iterations 10
  Unable to locate solution in chosen iterations: Current best guess is between 3.0365889718756613 and 3.0365889718756627
  cbrt_1(28) = 3.0365889718756618
  cbrt_1(27) = 3
  cbrt_2(28) = 3.0365889718756627
  Iterations 4
  cbrt_3(27) = 3
  Iterations 5
  cbrt_3(28) = 3.0365889718756627

] [/pre]

to get some (much!) diagnostic output we can add

#define BOOST_MATH_INSTRUMENT

[pre

]
*/
//] [/root_finding_example_output]

/*

cbrt(28) 3.0365889718756622
std::cbrt(28) 3.0365889718756627

*/
