// Copyright John Maddock 2015

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Comparison of finding roots using TOMS748, Newton-Raphson, Halley & Schroder algorithms.
// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!
// This program also writes files in Quickbook tables mark-up format.

#include <boost/cstdlib.hpp>
#include <boost/config.hpp>
#include <boost/array.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
template <class T>
struct cbrt_functor_noderiv
{
   //  cube root of x using only function - no derivatives.
   cbrt_functor_noderiv(T const& to_find_root_of) : a(to_find_root_of)
   { /* Constructor just stores value a to find root of. */
   }
   T operator()(T const& x)
   {
      T fx = x*x*x - a; // Difference (estimate x^3 - a).
      return fx;
   }
private:
   T a; // to be 'cube_rooted'.
};
//] [/root_finding_noderiv_1

template <class T>
std::uintmax_t cbrt_noderiv(T x, T guess)
{
   // return cube root of x using bracket_and_solve (no derivatives).
   using namespace std;                          // Help ADL of std functions.
   using namespace boost::math::tools;           // For bracket_and_solve_root.

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
   bracket_and_solve_root(cbrt_functor_noderiv<T>(x), guess, factor, is_rising, tol, it);
   return it;
}

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
      T dx = 3 * x*x;                 // 1st derivative = 3x^2.
      return std::make_pair(fx, dx);   // 'return' both fx and dx.
   }
private:
   T a;                               // Store value to be 'cube_rooted'.
};

template <class T>
std::uintmax_t cbrt_deriv(T x, T guess)
{
   // return cube root of x using 1st derivative and Newton_Raphson.
   using namespace boost::math::tools;
   T min = guess / 100;                     // We don't really know what this should be!
   T max = guess * 100;                     // We don't really know what this should be!
   const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
   int get_digits = static_cast<int>(digits * 0.6);    // Accuracy doubles with each step, so stop when we have
   // just over half the digits correct.
   const std::uintmax_t maxit = 20;
   std::uintmax_t it = maxit;
   newton_raphson_iterate(cbrt_functor_deriv<T>(x), guess, min, max, get_digits, it);
   return it;
}

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

template <class T>
std::uintmax_t cbrt_2deriv(T x, T guess)
{ 
   // return cube root of x using 1st and 2nd derivatives and Halley.
   //using namespace std;  // Help ADL of std functions.
   using namespace boost::math::tools;
   T min = guess / 100;                     // We don't really know what this should be!
   T max = guess * 100;                     // We don't really know what this should be!
   const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
   // digits used to control how accurate to try to make the result.
   int get_digits = static_cast<int>(digits * 0.4);    // Accuracy triples with each step, so stop when just
   // over one third of the digits are correct.
   std::uintmax_t maxit = 20;
   halley_iterate(cbrt_functor_2deriv<T>(x), guess, min, max, get_digits, maxit);
   return maxit;
}

template <class T>
std::uintmax_t cbrt_2deriv_s(T x, T guess)
{ 
   // return cube root of x using 1st and 2nd derivatives and Halley.
   //using namespace std;  // Help ADL of std functions.
   using namespace boost::math::tools;
   T min = guess / 100;                     // We don't really know what this should be!
   T max = guess * 100;                     // We don't really know what this should be!
   const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
   // digits used to control how accurate to try to make the result.
   int get_digits = static_cast<int>(digits * 0.4);    // Accuracy triples with each step, so stop when just
   // over one third of the digits are correct.
   std::uintmax_t maxit = 20;
   schroder_iterate(cbrt_functor_2deriv<T>(x), guess, min, max, get_digits, maxit);
   return maxit;
}

template <typename T = double>
struct elliptic_root_functor_noderiv
{ 
   elliptic_root_functor_noderiv(T const& arc, T const& radius) : m_arc(arc), m_radius(radius)
   { // Constructor just stores value a to find root of.
   }
   T operator()(T const& x)
   {
      // return the difference between required arc-length, and the calculated arc-length for an
      // ellipse with radii m_radius and x:
      T a = (std::max)(m_radius, x);
      T b = (std::min)(m_radius, x);
      T k = sqrt(1 - b * b / (a * a));
      return 4 * a * boost::math::ellint_2(k) - m_arc;
   }
private:
   T m_arc;     // length of arc.
   T m_radius;  // one of the two radii of the ellipse
}; // template <class T> struct elliptic_root_functor_noderiv

template <class T = double>
std::uintmax_t elliptic_root_noderiv(T radius, T arc, T guess)
{ // return the other radius of an ellipse, given one radii and the arc-length
   using namespace std;  // Help ADL of std functions.
   using namespace boost::math::tools; // For bracket_and_solve_root.

   T factor = 2;                       // How big steps to take when searching.

   const std::uintmax_t maxit = 50;  // Limit to maximum iterations.
   std::uintmax_t it = maxit;        // Initially our chosen max iterations, but updated with actual.
   bool is_rising = true;              // arc-length increases if one radii increases, so function is rising
   // Define a termination condition, stop when nearly all digits are correct, but allow for
   // the fact that we are returning a range, and must have some inaccuracy in the elliptic integral:
   eps_tolerance<T> tol(std::numeric_limits<T>::digits - 2);
   // Call bracket_and_solve_root to find the solution, note that this is a rising function:
   bracket_and_solve_root(elliptic_root_functor_noderiv<T>(arc, radius), guess, factor, is_rising, tol, it);
   return it;
} 

template <class T = double>
struct elliptic_root_functor_1deriv
{ // Functor also returning 1st derivative.
   static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");

   elliptic_root_functor_1deriv(T const& arc, T const& radius) : m_arc(arc), m_radius(radius)
   { // Constructor just stores value a to find root of.
   }
   std::pair<T, T> operator()(T const& x)
   {
      // Return the difference between required arc-length, and the calculated arc-length for an
      // ellipse with radii m_radius and x, plus it's derivative.
      // See http://www.wolframalpha.com/input/?i=d%2Fda+[4+*+a+*+EllipticE%281+-+b^2%2Fa^2%29]
      // We require two elliptic integral calls, but from these we can calculate both
      // the function and it's derivative:
      T a = (std::max)(m_radius, x);
      T b = (std::min)(m_radius, x);
      T a2 = a * a;
      T b2 = b * b;
      T k = sqrt(1 - b2 / a2);
      T Ek = boost::math::ellint_2(k);
      T Kk = boost::math::ellint_1(k);
      T fx = 4 * a * Ek - m_arc;
      T dfx = 4 * (a2 * Ek - b2 * Kk) / (a2 - b2);
      return std::make_pair(fx, dfx);
   }
private:
   T m_arc;     // length of arc.
   T m_radius;  // one of the two radii of the ellipse
};  // struct elliptic_root__functor_1deriv

template <class T = double>
std::uintmax_t elliptic_root_1deriv(T radius, T arc, T guess)
{
   using namespace std;  // Help ADL of std functions.
   using namespace boost::math::tools; // For newton_raphson_iterate.

   static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");

   T min = 0;   // Minimum possible value is zero.
   T max = arc; // Maximum possible value is the arc length.

   // Accuracy doubles at each step, so stop when just over half of the digits are
   // correct, and rely on that step to polish off the remainder:
   int get_digits = static_cast<int>(std::numeric_limits<T>::digits * 0.6);
   const std::uintmax_t maxit = 20;
   std::uintmax_t it = maxit;
   newton_raphson_iterate(elliptic_root_functor_1deriv<T>(arc, radius), guess, min, max, get_digits, it);
   return it;
}

template <class T = double>
struct elliptic_root_functor_2deriv
{ // Functor returning both 1st and 2nd derivatives.
   static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");

   elliptic_root_functor_2deriv(T const& arc, T const& radius) : m_arc(arc), m_radius(radius) {}
   std::tuple<T, T, T> operator()(T const& x)
   {
      // Return the difference between required arc-length, and the calculated arc-length for an
      // ellipse with radii m_radius and x, plus it's derivative.
      // See http://www.wolframalpha.com/input/?i=d^2%2Fda^2+[4+*+a+*+EllipticE%281+-+b^2%2Fa^2%29]
      // for the second derivative.
      T a = (std::max)(m_radius, x);
      T b = (std::min)(m_radius, x);
      T a2 = a * a;
      T b2 = b * b;
      T k = sqrt(1 - b2 / a2);
      T Ek = boost::math::ellint_2(k);
      T Kk = boost::math::ellint_1(k);
      T fx = 4 * a * Ek - m_arc;
      T dfx = 4 * (a2 * Ek - b2 * Kk) / (a2 - b2);
      T dfx2 = 4 * b2 * ((a2 + b2) * Kk - 2 * a2 * Ek) / (a * (a2 - b2) * (a2 - b2));
      return std::make_tuple(fx, dfx, dfx2);
   }
private:
   T m_arc;     // length of arc.
   T m_radius;  // one of the two radii of the ellipse
};

template <class T = double>
std::uintmax_t elliptic_root_2deriv(T radius, T arc, T guess)
{
   using namespace std;                // Help ADL of std functions.
   using namespace boost::math::tools; // For halley_iterate.

   static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");

   T min = 0;                                   // Minimum possible value is zero.
   T max = arc;                                 // radius can't be larger than the arc length.

   // Accuracy triples at each step, so stop when just over one-third of the digits
   // are correct, and the last iteration will polish off the remaining digits:
   int get_digits = static_cast<int>(std::numeric_limits<T>::digits * 0.4);
   const std::uintmax_t maxit = 20;
   std::uintmax_t it = maxit;
   halley_iterate(elliptic_root_functor_2deriv<T>(arc, radius), guess, min, max, get_digits, it);
   return it;
} // nth_2deriv Halley
//]
// Using 1st and 2nd derivatives using Schroder algorithm.

template <class T = double>
std::uintmax_t elliptic_root_2deriv_s(T radius, T arc, T guess)
{ // return nth root of x using 1st and 2nd derivatives and Schroder.

   using namespace std;  // Help ADL of std functions.
   using namespace boost::math::tools; // For schroder_iterate.

   static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");

   T min = 0; // Minimum possible value is zero.
   T max = arc; // radius can't be larger than the arc length.

   int digits = std::numeric_limits<T>::digits; // Maximum possible binary digits accuracy for type T.
   int get_digits = static_cast<int>(digits * 0.4);
   const std::uintmax_t maxit = 20;
   std::uintmax_t it = maxit;
   schroder_iterate(elliptic_root_functor_2deriv<T>(arc, radius), guess, min, max, get_digits, it);
   return it;
} // T elliptic_root_2deriv_s Schroder


int main()
{
   try
   {
      double to_root = 500;
      double answer = 7.93700525984;

      std::cout << "[table\n"
         << "[[Initial Guess=][-500% ([approx]1.323)][-100% ([approx]3.97)][-50% ([approx]3.96)][-20% ([approx]6.35)][-10% ([approx]7.14)][-5% ([approx]7.54)]"
         "[5% ([approx]8.33)][10% ([approx]8.73)][20% ([approx]9.52)][50% ([approx]11.91)][100% ([approx]15.87)][500 ([approx]47.6)]]\n";
      std::cout << "[[bracket_and_solve_root]["
         << cbrt_noderiv(to_root, answer / 6)
         << "][" << cbrt_noderiv(to_root, answer / 2)
         << "][" << cbrt_noderiv(to_root, answer - answer * 0.5)
         << "][" << cbrt_noderiv(to_root, answer - answer * 0.2)
         << "][" << cbrt_noderiv(to_root, answer - answer * 0.1)
         << "][" << cbrt_noderiv(to_root, answer - answer * 0.05)
         << "][" << cbrt_noderiv(to_root, answer + answer * 0.05)
         << "][" << cbrt_noderiv(to_root, answer + answer * 0.1)
         << "][" << cbrt_noderiv(to_root, answer + answer * 0.2)
         << "][" << cbrt_noderiv(to_root, answer + answer * 0.5)
         << "][" << cbrt_noderiv(to_root, answer + answer)
         << "][" << cbrt_noderiv(to_root, answer + answer * 5) << "]]\n";

      std::cout << "[[newton_iterate]["
         << cbrt_deriv(to_root, answer / 6)
         << "][" << cbrt_deriv(to_root, answer / 2)
         << "][" << cbrt_deriv(to_root, answer - answer * 0.5)
         << "][" << cbrt_deriv(to_root, answer - answer * 0.2)
         << "][" << cbrt_deriv(to_root, answer - answer * 0.1)
         << "][" << cbrt_deriv(to_root, answer - answer * 0.05)
         << "][" << cbrt_deriv(to_root, answer + answer * 0.05)
         << "][" << cbrt_deriv(to_root, answer + answer * 0.1)
         << "][" << cbrt_deriv(to_root, answer + answer * 0.2)
         << "][" << cbrt_deriv(to_root, answer + answer * 0.5)
         << "][" << cbrt_deriv(to_root, answer + answer)
         << "][" << cbrt_deriv(to_root, answer + answer * 5) << "]]\n";

      std::cout << "[[halley_iterate]["
         << cbrt_2deriv(to_root, answer / 6)
         << "][" << cbrt_2deriv(to_root, answer / 2)
         << "][" << cbrt_2deriv(to_root, answer - answer * 0.5)
         << "][" << cbrt_2deriv(to_root, answer - answer * 0.2)
         << "][" << cbrt_2deriv(to_root, answer - answer * 0.1)
         << "][" << cbrt_2deriv(to_root, answer - answer * 0.05)
         << "][" << cbrt_2deriv(to_root, answer + answer * 0.05)
         << "][" << cbrt_2deriv(to_root, answer + answer * 0.1)
         << "][" << cbrt_2deriv(to_root, answer + answer * 0.2)
         << "][" << cbrt_2deriv(to_root, answer + answer * 0.5)
         << "][" << cbrt_2deriv(to_root, answer + answer)
         << "][" << cbrt_2deriv(to_root, answer + answer * 5) << "]]\n";

      std::cout << "[[schr'''&#xf6;'''der_iterate]["
         << cbrt_2deriv_s(to_root, answer / 6)
         << "][" << cbrt_2deriv_s(to_root, answer / 2)
         << "][" << cbrt_2deriv_s(to_root, answer - answer * 0.5)
         << "][" << cbrt_2deriv_s(to_root, answer - answer * 0.2)
         << "][" << cbrt_2deriv_s(to_root, answer - answer * 0.1)
         << "][" << cbrt_2deriv_s(to_root, answer - answer * 0.05)
         << "][" << cbrt_2deriv_s(to_root, answer + answer * 0.05)
         << "][" << cbrt_2deriv_s(to_root, answer + answer * 0.1)
         << "][" << cbrt_2deriv_s(to_root, answer + answer * 0.2)
         << "][" << cbrt_2deriv_s(to_root, answer + answer * 0.5)
         << "][" << cbrt_2deriv_s(to_root, answer + answer)
         << "][" << cbrt_2deriv_s(to_root, answer + answer * 5) << "]]\n]\n\n";


      double radius_a = 10;
      double arc_length = 500;
      double radius_b = 123.6216507967705;

      std::cout << std::setprecision(4) << "[table\n"
         << "[[Initial Guess=][-500% ([approx]" << radius_b / 6 << ")][-100% ([approx]" << radius_b / 2 << ")][-50% ([approx]"
         << radius_b - radius_b * 0.5 << ")][-20% ([approx]" << radius_b - radius_b * 0.2 << ")][-10% ([approx]" << radius_b - radius_b * 0.1 << ")][-5% ([approx]" << radius_b - radius_b * 0.05 << ")]"
         "[5% ([approx]" << radius_b + radius_b * 0.05 << ")][10% ([approx]" << radius_b + radius_b * 0.1 << ")][20% ([approx]" << radius_b + radius_b * 0.2 << ")][50% ([approx]" << radius_b + radius_b * 0.5 
         << ")][100% ([approx]" << radius_b + radius_b << ")][500 ([approx]" << radius_b + radius_b * 5 << ")]]\n";
      std::cout << "[[bracket_and_solve_root]["
         << elliptic_root_noderiv(radius_a, arc_length, radius_b / 6)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b / 2)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b - radius_b * 0.5)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b - radius_b * 0.2)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b - radius_b * 0.1)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b - radius_b * 0.05)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b + radius_b * 0.05)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b + radius_b * 0.1)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b + radius_b * 0.2)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b + radius_b * 0.5)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b + radius_b)
         << "][" << elliptic_root_noderiv(radius_a, arc_length, radius_b + radius_b * 5) << "]]\n";

      std::cout << "[[newton_iterate]["
         << elliptic_root_1deriv(radius_a, arc_length, radius_b / 6)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b / 2)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b - radius_b * 0.5)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b - radius_b * 0.2)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b - radius_b * 0.1)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b - radius_b * 0.05)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b + radius_b * 0.05)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b + radius_b * 0.1)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b + radius_b * 0.2)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b + radius_b * 0.5)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b + radius_b)
         << "][" << elliptic_root_1deriv(radius_a, arc_length, radius_b + radius_b * 5) << "]]\n";

      std::cout << "[[halley_iterate]["
         << elliptic_root_2deriv(radius_a, arc_length, radius_b / 6)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b / 2)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b - radius_b * 0.5)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b - radius_b * 0.2)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b - radius_b * 0.1)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b - radius_b * 0.05)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b + radius_b * 0.05)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b + radius_b * 0.1)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b + radius_b * 0.2)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b + radius_b * 0.5)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b + radius_b)
         << "][" << elliptic_root_2deriv(radius_a, arc_length, radius_b + radius_b * 5) << "]]\n";

      std::cout << "[[schr'''&#xf6;'''der_iterate]["
         << elliptic_root_2deriv_s(radius_a, arc_length, radius_b / 6)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b / 2)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b - radius_b * 0.5)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b - radius_b * 0.2)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b - radius_b * 0.1)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b - radius_b * 0.05)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b + radius_b * 0.05)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b + radius_b * 0.1)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b + radius_b * 0.2)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b + radius_b * 0.5)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b + radius_b)
         << "][" << elliptic_root_2deriv_s(radius_a, arc_length, radius_b + radius_b * 5) << "]]\n]\n\n";

      return boost::exit_success;
   }
   catch(std::exception ex)
   {
      std::cout << "exception thrown: " << ex.what() << std::endl;
      return boost::exit_failure;
   }
} // int main()

