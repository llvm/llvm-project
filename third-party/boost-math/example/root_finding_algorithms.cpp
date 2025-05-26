// Copyright Paul A. Bristow 2015

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Comparison of finding roots using TOMS748, Newton-Raphson, Schroder & Halley algorithms.

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

// root_finding_algorithms.cpp

#include <boost/cstdlib.hpp>
#include <boost/config.hpp>
#include <boost/array.hpp>
#include "table_type.hpp"
// Copy of i:\modular-boost\libs\math\test\table_type.hpp
// #include "handle_test_result.hpp"
// Copy of i:\modular - boost\libs\math\test\handle_test_result.hpp

#include <boost/math/tools/roots.hpp>
//using boost::math::policies::policy;
//using boost::math::tools::newton_raphson_iterate;
//using boost::math::tools::halley_iterate; //
//using boost::math::tools::eps_tolerance; // Binary functor for specified number of bits.
//using boost::math::tools::bracket_and_solve_root;
//using boost::math::tools::toms748_solve;
//using boost::math::tools::schroder_iterate;

#include <boost/math/special_functions/next.hpp> // For float_distance.
#include <tuple> // for tuple and make_tuple.
#include <boost/math/special_functions/cbrt.hpp> // For boost::math::cbrt.

#include <boost/multiprecision/cpp_bin_float.hpp> // is binary.
//#include <boost/multiprecision/cpp_dec_float.hpp> // is decimal.
using boost::multiprecision::cpp_bin_float_100;
using boost::multiprecision::cpp_bin_float_50;

#include <boost/timer/timer.hpp>
#include <boost/system/error_code.hpp>
#include <boost/multiprecision/cpp_bin_float/io.hpp>
#include <boost/preprocessor/stringize.hpp>

// STL
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <limits>
#include <fstream> // std::ofstream
#include <cmath>
#include <typeinfo> // for type name using typid(thingy).name();
#include <type_traits>

#ifndef BOOST_ROOT
# define BOOST_ROOT i:/modular-boost/
#endif
// Need to find this 

#ifdef __FILE__
std::string sourcefilename = __FILE__;
#endif

std::string chop_last(std::string s)
{
   std::string::size_type pos = s.find_last_of("\\/");
   if(pos != std::string::npos)
      s.erase(pos);
   else if(s.empty())
      abort();
   else
      s.erase();
   return s;
}

std::string make_root()
{
   std::string result;
   if(sourcefilename.find_first_of(":") != std::string::npos)
   {
      result = chop_last(sourcefilename); // lose filename part
      result = chop_last(result);   // lose /example/
      result = chop_last(result);   // lose /math/
      result = chop_last(result);   // lose /libs/
   }
   else
   {
      result = chop_last(sourcefilename); // lose filename part
      if(result.empty())
         result = ".";
      result += "/../../..";
   }
   return result;
}

std::string short_file_name(std::string s)
{
   std::string::size_type pos = s.find_last_of("\\/");
   if(pos != std::string::npos)
      s.erase(0, pos + 1);
   return s;
}

std::string boost_root = make_root();

#ifdef _MSC_VER
  std::string filename = boost_root.append("/libs/math/doc/roots/root_comparison_tables_msvc.qbk");
#else // assume GCC
  std::string filename = boost_root.append("/libs/math/doc/roots/root_comparison_tables_gcc.qbk");
#endif

std::ofstream fout (filename.c_str(), std::ios_base::out);

//std::array<std::string, 6> float_type_names =
//{
//  "float", "double", "long double", "cpp_bin_128", "cpp_dec_50", "cpp_dec_100"
//};

std::vector<std::string> algo_names =
{
  "cbrt", "TOMS748", "Newton", "Halley", "Schr'''&#xf6;'''der"
};

std::vector<int> max_digits10s;
std::vector<std::string> typenames; // Full computer generated type name.
std::vector<std::string> names; // short name.

uintmax_t iters; // Global as iterations is not returned by rooting function.

const int count = 1000000; // Number of iterations to average.
 
struct root_info
{ // for a floating-point type, float, double ...
  std::size_t max_digits10; // for type.
  std::string full_typename; // for type from type_id.name().
  std::string short_typename; // for type "float", "double", "cpp_bin_float_50" ....

  std::size_t bin_digits;  // binary in floating-point type numeric_limits<T>::digits;  
  int get_digits; // fraction of maximum possible accuracy required.
  // = digits * digits_accuracy
  // Vector of values for each algorithm, std::cbrt, boost::math::cbrt, TOMS748, Newton, Halley.
  //std::vector< std::int_least64_t> times;  converted to int.
  std::vector<int> times;
  //std::int_least64_t min_time = std::numeric_limits<std::int_least64_t>::max(); // Used to normalize times (as int).
  std::vector<double> normed_times;
  std::int_least64_t min_time = (std::numeric_limits<std::int_least64_t>::max)(); // Used to normalize times.
  std::vector<uintmax_t> iterations;
  std::vector<long int> distances;
  std::vector<cpp_bin_float_100> full_results;
}; // struct root_info

std::vector<root_info> root_infos;  // One element for each type used.

int type_no = -1; // float = 0, double = 1, ... indexing root_infos.

inline std::string build_test_name(const char* type_name, const char* test_name)
{
  std::string result(BOOST_COMPILER);
  result += "|";
  result += BOOST_STDLIB;
  result += "|";
  result += BOOST_PLATFORM;
  result += "|";
  result += type_name;
  result += "|";
  result += test_name;
#if defined(_DEBUG ) || !defined(NDEBUG)
  result += "|";
  result += " debug";
#else
  result += "|";
  result += " release";
#endif
  result += "|";
    return result;
}

// No derivatives - using TOMS748 internally.
template <class T>
struct cbrt_functor_noderiv
{ //  cube root of x using only function - no derivatives.
  cbrt_functor_noderiv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor just stores value a to find root of.
  }
  T operator()(T const& x)
  {
    T fx = x*x*x - a; // Difference (estimate x^3 - a).
    return fx;
  }
private:
  T a; // to be 'cube_rooted'.
}; // template <class T> struct cbrt_functor_noderiv

template <class T>
T cbrt_noderiv(T x)
{ // return cube root of x using bracket_and_solve (using NO derivatives).
  using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools; // For bracket_and_solve_root.

  // Maybe guess should be double, or use enable_if to avoid warning about conversion double to float here?
  T guess;
  if (std::is_fundamental<T>::value)
  { 
    int exponent;
    frexp(x, &exponent); // Get exponent of z (ignore mantissa).
    guess = ldexp((T)1., exponent / 3); // Rough guess is to divide the exponent by three.
  }
  else
  { // (boost::is_class<T>)
    double dx = static_cast<double>(x);
    guess = boost::math::cbrt<T>(dx); // Get guess using double.
  }
  
  T factor = 2; // How big steps to take when searching.

  const std::uintmax_t maxit = 50; // Limit to maximum iterations.
  std::uintmax_t it = maxit; // Initially our chosen max iterations, but updated with actual.
  bool is_rising = true; // So if result if guess^3 is too low, then try increasing guess.
  // Some fraction of digits is used to control how accurate to try to make the result.
  int get_digits = static_cast<int>(std::numeric_limits<T>::digits - 2);

  eps_tolerance<T> tol(get_digits); // Set the tolerance.
  std::pair<T, T> r =
    bracket_and_solve_root(cbrt_functor_noderiv<T>(x), guess, factor, is_rising, tol, it);
  iters = it;
  T result = r.first + (r.second - r.first) / 2;  // Midway between brackets.
  return result;
} // template <class T> T cbrt_noderiv(T x)


// Using 1st derivative only Newton-Raphson

template <class T>
struct cbrt_functor_deriv
{ // Functor also returning 1st derivative.
  cbrt_functor_deriv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor stores value a to find root of,
    // for example: calling cbrt_functor_deriv<T>(x) to use to get cube root of x.
  }
  std::pair<T, T> operator()(T const& x)
  { // Return both f(x) and f'(x).
    T fx = x*x*x - a; // Difference (estimate x^3 - value).
    T dx = 3 * x*x; // 1st derivative = 3x^2.
    return std::make_pair(fx, dx); // 'return' both fx and dx.
  }
private:
  T a; // to be 'cube_rooted'.
};

template <class T>
T cbrt_deriv(T x)
{ // return cube root of x using 1st derivative and Newton_Raphson.
  using namespace boost::math::tools;
  int exponent;
  T guess;
  if(std::is_fundamental<T>::value)
  {
     frexp(x, &exponent); // Get exponent of z (ignore mantissa).
     guess = ldexp(static_cast<T>(1), exponent / 3); // Rough guess is to divide the exponent by three.
  }
  else
     guess = boost::math::cbrt(static_cast<double>(x));
  T min = guess / 2; // Minimum possible value is half our guess.
  T max = 2 * guess; // Maximum possible value is twice our guess.
  int get_digits = static_cast<int>(std::numeric_limits<T>::digits * 0.6);
  const std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = newton_raphson_iterate(cbrt_functor_deriv<T>(x), guess, min, max, get_digits, it);
  iters = it;
  return result;
}

// Using 1st and 2nd derivatives with Halley algorithm.

template <class T>
struct cbrt_functor_2deriv
{ // Functor returning both 1st and 2nd derivatives.
  cbrt_functor_2deriv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor stores value a to find root of, for example:
    // calling cbrt_functor_2deriv<T>(x) to get cube root of x,
  }
  std::tuple<T, T, T> operator()(T const& x)
  { // Return both f(x) and f'(x) and f''(x).
    T fx = x*x*x - a; // Difference (estimate x^3 - value).
    T dx = 3 * x*x; // 1st derivative = 3x^2.
    T d2x = 6 * x; // 2nd derivative = 6x.
    return std::make_tuple(fx, dx, d2x); // 'return' fx, dx and d2x.
  }
private:
  T a; // to be 'cube_rooted'.
};

template <class T>
T cbrt_2deriv(T x)
{ // return cube root of x using 1st and 2nd derivatives and Halley.
  //using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools;
  int exponent;
  T guess;
  if(std::is_fundamental<T>::value)
  {
     frexp(x, &exponent); // Get exponent of z (ignore mantissa).
     guess = ldexp(static_cast<T>(1), exponent / 3); // Rough guess is to divide the exponent by three.
  }
  else
     guess = boost::math::cbrt(static_cast<double>(x));
  T min = guess / 2; // Minimum possible value is half our guess.
  T max = 2 * guess; // Maximum possible value is twice our guess.
  // digits used to control how accurate to try to make the result.
  int get_digits = static_cast<int>(std::numeric_limits<T>::digits * 0.4);
  std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = halley_iterate(cbrt_functor_2deriv<T>(x), guess, min, max, get_digits, it);
  iters = it;
  return result;
}

// Using 1st and 2nd derivatives using Schroder algorithm.

template <class T>
T cbrt_2deriv_s(T x)
{ // return cube root of x using 1st and 2nd derivatives and Schroder algorithm.
  //using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools;
  int exponent;
  T guess;
  if(std::is_fundamental<T>::value)
  {
     frexp(x, &exponent); // Get exponent of z (ignore mantissa).
     guess = ldexp(static_cast<T>(1), exponent / 3); // Rough guess is to divide the exponent by three.
  }
  else
     guess = boost::math::cbrt(static_cast<double>(x));
  T min = guess / 2; // Minimum possible value is half our guess.
  T max = 2 * guess; // Maximum possible value is twice our guess.
  // digits used to control how accurate to try to make the result.
  int get_digits = static_cast<int>(std::numeric_limits<T>::digits * 0.4);
  const std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = schroder_iterate(cbrt_functor_2deriv<T>(x), guess, min, max, get_digits, it);
  iters = it;
  return result;
} // template <class T> T cbrt_2deriv_s(T x)



template <typename T>
int test_root(cpp_bin_float_100 big_value, cpp_bin_float_100 answer, const char* type_name)
{
  //T value = 28.; // integer (exactly representable as floating-point)
  // whose cube root is *not* exactly representable.
  // Wolfram Alpha command N[28 ^ (1 / 3), 100] computes cube root to 100 decimal digits.
  // 3.036588971875662519420809578505669635581453977248111123242141654169177268411884961770250390838097895
  
  std::size_t max_digits = 2 + std::numeric_limits<T>::digits * 3010 / 10000;
  // For new versions use max_digits10
  // std::cout.precision(std::numeric_limits<T>::max_digits10);
  std::cout.precision(max_digits);
  std::cout << std::showpoint << std::endl; // Trailing zeros too.

  root_infos.push_back(root_info());
  type_no++;  // Another type.

  root_infos[type_no].max_digits10 = max_digits;
  root_infos[type_no].full_typename = typeid(T).name(); // Full typename.
  root_infos[type_no].short_typename = type_name; // Short typename.

  root_infos[type_no].bin_digits = std::numeric_limits<T>::digits;

  root_infos[type_no].get_digits = std::numeric_limits<T>::digits;

  T to_root = static_cast<T>(big_value);
  T result; // root
  T ans = static_cast<T>(answer);
  int algo = 0; // Count of algorithms used.
 
  using boost::timer::nanosecond_type;
  using boost::timer::cpu_times;
  using boost::timer::cpu_timer;

  cpu_times now; // Holds wall, user and system times.
  T sum = 0;

  // std::cbrt is much the fastest, but not useful for this comparison because it only handles fundamental types.
  // Using enable_if allows us to avoid a compile fail with multiprecision types, but still distorts the results too much.

  //{
  //  algorithm_names.push_back("std::cbrt"); 
  //  cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
  //  ti.start();
  //  for (long i = 0; i < count; ++i)
  //  {
  //    stdcbrt(big_value);
  //  }
  //  now = ti.elapsed();
  //  int time = static_cast<int>(now.user / count);
  //  root_infos[type_no].times.push_back(time); // CPU time taken per root.
  //  if (time < root_infos[type_no].min_time)
  //  {
  //    root_infos[type_no].min_time = time;
  //  }
  //  ti.stop();
  //  long int distance = static_cast<int>(boost::math::float_distance<T>(result, ans));
  //  root_infos[type_no].distances.push_back(distance);
  //  root_infos[type_no].iterations.push_back(0); // Not known.
  //  root_infos[type_no].full_results.push_back(result);
  //  algo++;
  //}
  //{
  //  //algorithm_names.push_back("boost::math::cbrt"); // .
  //  cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
  //  ti.start();
  //  for (long i = 0; i < count; ++i)
  //  {
  //    result = boost::math::cbrt(to_root); // 
  //  }
  //  now = ti.elapsed();
  //  int time = static_cast<int>(now.user / count);
  //  root_infos[type_no].times.push_back(time); // CPU time taken.
  //  ti.stop();
  //  if (time < root_infos[type_no].min_time)
  //  {
  //    root_infos[type_no].min_time = time;
  //  }
  //  long int distance = static_cast<int>(boost::math::float_distance<T>(result, ans));
  //  root_infos[type_no].distances.push_back(distance);
  //  root_infos[type_no].iterations.push_back(0); // Iterations not knowable.
  //  root_infos[type_no].full_results.push_back(result);
  //}



  {
    //algorithm_names.push_back("boost::math::cbrt"); // .
    result = 0;
    cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
    ti.start();
    for (long i = 0; i < count; ++i)
    {
      result = boost::math::cbrt(to_root); // 
      sum += result;
    }
    now = ti.elapsed();

    long time = static_cast<long>(now.user/1000); // convert nanoseconds to microseconds (assuming this is resolution).
    root_infos[type_no].times.push_back(time); // CPU time taken.
    ti.stop();
    if (time < root_infos[type_no].min_time)
    {
      root_infos[type_no].min_time = time;
    }
    long int distance = static_cast<int>(boost::math::float_distance<T>(result, ans));
    root_infos[type_no].distances.push_back(distance);
    root_infos[type_no].iterations.push_back(0); // Iterations not knowable.
    root_infos[type_no].full_results.push_back(result);
  }
  {
    //algorithm_names.push_back("TOMS748"); // 
    cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
    ti.start();
    for (long i = 0; i < count; ++i)
    {
      result = cbrt_noderiv<T>(to_root); // 
      sum += result;
    }
    now = ti.elapsed();
//    int time = static_cast<int>(now.user / count);
    long time = static_cast<long>(now.user/1000);
    root_infos[type_no].times.push_back(time); // CPU time taken.
    if (time < root_infos[type_no].min_time)
    {
      root_infos[type_no].min_time = time;
    }
    ti.stop();
    long int distance = static_cast<int>(boost::math::float_distance<T>(result, ans));
    root_infos[type_no].distances.push_back(distance);
    root_infos[type_no].iterations.push_back(iters); // 
    root_infos[type_no].full_results.push_back(result);
  }
  {
   // algorithm_names.push_back("Newton"); // algorithm
    cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
    ti.start();
    for (long i = 0; i < count; ++i)
    {
      result = cbrt_deriv(to_root); // 
      sum += result;
    }
    now = ti.elapsed();
//    int time = static_cast<int>(now.user / count);
    long time = static_cast<long>(now.user/1000);
    root_infos[type_no].times.push_back(time); // CPU time taken.
    if (time < root_infos[type_no].min_time)
    {
      root_infos[type_no].min_time = time;
    }

    ti.stop();
    long int distance = static_cast<int>(boost::math::float_distance<T>(result, ans));
    root_infos[type_no].distances.push_back(distance);
    root_infos[type_no].iterations.push_back(iters); //
    root_infos[type_no].full_results.push_back(result);
  }
  {
  //algorithm_names.push_back("Halley"); // algorithm
    cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
    ti.start();
    for (long i = 0; i < count; ++i)
    {
      result = cbrt_2deriv(to_root); // 
      sum += result;
    }
    now = ti.elapsed(); 
//    int time = static_cast<int>(now.user / count);
    long time = static_cast<long>(now.user/1000);
    root_infos[type_no].times.push_back(time); // CPU time taken.
    ti.stop();
    if (time < root_infos[type_no].min_time)
    {
      root_infos[type_no].min_time = time;
    }
    long int distance = static_cast<int>(boost::math::float_distance<T>(result, ans));
    root_infos[type_no].distances.push_back(distance);
    root_infos[type_no].iterations.push_back(iters); // 
    root_infos[type_no].full_results.push_back(result);
  }

  {
   // algorithm_names.push_back("Shroeder"); // algorithm
    cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
    ti.start();
    for (long i = 0; i < count; ++i)
    {
      result = cbrt_2deriv_s(to_root); // 
      sum += result;
    }
    now = ti.elapsed();
//    int time = static_cast<int>(now.user / count);
    long time = static_cast<long>(now.user/1000);
    root_infos[type_no].times.push_back(time); // CPU time taken.
    if (time < root_infos[type_no].min_time)
    {
      root_infos[type_no].min_time = time;
    }
    ti.stop();
    long int distance = static_cast<int>(boost::math::float_distance<T>(result, ans));
    root_infos[type_no].distances.push_back(distance);
    root_infos[type_no].iterations.push_back(iters); // 
    root_infos[type_no].full_results.push_back(result);
  }
  for (size_t i = 0; i != root_infos[type_no].times.size(); i++)
  { // Normalize times.
    double normed_time = static_cast<double>(root_infos[type_no].times[i]);
    normed_time /= root_infos[type_no].min_time;
    root_infos[type_no].normed_times.push_back(normed_time);
  }
  algo++;
  std::cout << "Accumulated sum was " << sum << std::endl;
  return algo;  // Count of how many algorithms used.
} // test_root

void table_root_info(cpp_bin_float_100 full_value, cpp_bin_float_100 full_answer)
{
   // Fill the elements. 
  test_root<float>(full_value, full_answer, "float");
  test_root<double>(full_value, full_answer, "double");
  test_root<long double>(full_value, full_answer, "long double");
  test_root<cpp_bin_float_50>(full_value, full_answer, "cpp_bin_float_50");
  //test_root<cpp_bin_float_100>(full_value, full_answer, "cpp_bin_float_100");

  std::cout << root_infos.size() << " floating-point types tested:" << std::endl;
#ifndef NDEBUG
  std::cout << "Compiled in debug mode." << std::endl;
#else
  std::cout << "Compiled in optimise mode." << std::endl;
#endif


  for (size_t tp = 0; tp != root_infos.size(); tp++)
  { // For all types:

    std::cout << std::endl;

    std::cout << "Floating-point type = " << root_infos[tp].short_typename << std::endl;
    std::cout << "Floating-point type = " << root_infos[tp].full_typename << std::endl;
    std::cout << "Max_digits10 = " << root_infos[tp].max_digits10 << std::endl;
    std::cout << "Binary digits = " << root_infos[tp].bin_digits << std::endl;
    std::cout << "Accuracy digits = " << root_infos[tp].get_digits - 2 << ", " << static_cast<int>(root_infos[tp].get_digits * 0.6) << ", " << static_cast<int>(root_infos[tp].get_digits * 0.4) << std::endl;
    std::cout << "min_time = " << root_infos[tp].min_time << std::endl;

    std::cout << std::setprecision(root_infos[tp].max_digits10 ) << "Roots = ";
    std::copy(root_infos[tp].full_results.begin(), root_infos[tp].full_results.end(), std::ostream_iterator<cpp_bin_float_100>(std::cout, " "));
    std::cout << std::endl;

    // Header row.
    std::cout << "Algorithm         " << "Iterations  " << "Times  " << "Norm_times  " << "Distance" << std::endl;

    // Row for all algorithms.
    for (unsigned algo = 0; algo != algo_names.size(); algo++)
    { 
      std::cout
        << std::left << std::setw(20) << algo_names[algo] << "  "
        << std::setw(8) << std::setprecision(2) << root_infos[tp].iterations[algo] << "  "
        << std::setw(8) << std::setprecision(5) << root_infos[tp].times[algo] << " "
        << std::setw(8) << std::setprecision(3) << root_infos[tp].normed_times[algo] << " "
        << std::setw(8) << std::setprecision(2) << root_infos[tp].distances[algo]
        << std::endl;
    } // for algo
  } // for tp

  // Print info as Quickbook table.
#if 0
  fout << "[table:cbrt_5  Info for float, double, long double and cpp_bin_float_50\n"
    << "[[type name] [max_digits10] [binary digits] [required digits]]\n";// header.

  for (size_t tp = 0; tp != root_infos.size(); tp++)
  { // For all types:
    fout << "["
     <<  "[" << root_infos[tp].short_typename << "]" 
      << "[" << root_infos[tp].max_digits10 << "]"  // max_digits10
      << "["  << root_infos[tp].bin_digits << "]"// < "Binary digits 
      << "["  << root_infos[tp].get_digits << "]]\n"; // Accuracy digits.
  } // tp
  fout << "] [/table cbrt_5] \n" << std::endl;
#endif
  // Prepare Quickbook table of floating-point types.
  fout << "[table:cbrt_4 Cube root(28) for float, double, long double and cpp_bin_float_50\n"
    << "[[][float][][][] [][double][][][] [][long d][][][] [][cpp50][][]]\n"
    << "[[Algorithm]"; 
  for (size_t tp = 0; tp != root_infos.size(); tp++)
  { // For all types:
    fout << "[Its]" << "[Times]" << "[Norm]" << "[Dis]" << "[ ]";
  }
  fout << "]" << std::endl;

  // Row for all algorithms.
  for (size_t algo = 0; algo != algo_names.size(); algo++)
  {
    fout << "[[" << std::left << std::setw(9) << algo_names[algo] << "]";
    for (size_t tp = 0; tp != root_infos.size(); tp++)
    { // For all types:

       fout
          << "[" << std::right << std::showpoint
          << std::setw(3) << std::setprecision(2) << root_infos[tp].iterations[algo] << "]["
          << std::setw(5) << std::setprecision(5) << root_infos[tp].times[algo] << "][";
       if(fabs(root_infos[tp].normed_times[algo]) <= 1.05)
          fout << "[role blue " << std::setw(3) << std::setprecision(2) << root_infos[tp].normed_times[algo] << "]";
       else if(fabs(root_infos[tp].normed_times[algo]) > 4)
          fout << "[role red " << std::setw(3) << std::setprecision(2) << root_infos[tp].normed_times[algo] << "]";
       else
          fout << std::setw(3) << std::setprecision(2) << root_infos[tp].normed_times[algo];
       fout
        << "]["
        << std::setw(3) << std::setprecision(2) << root_infos[tp].distances[algo] << "][ ]";
    } // tp
     fout <<"]" << std::endl;
  } // for algo
  fout << "] [/end of table cbrt_4]\n";
} // void table_root_info

int main()
{
  using namespace boost::multiprecision;
  using namespace boost::math;
 
  try
  {
    std::cout << "Tests run with " << BOOST_COMPILER << ", "
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << ", ";

    if (fout.is_open())
    {
      std::cout << "\nOutput to " << filename << std::endl;
    }
    else
    { // Failed to open.
      std::cout << " Open file " << filename << " for output failed!" << std::endl;
      std::cout << "error" << errno << std::endl;
      return boost::exit_failure;
    }

    fout <<
      "[/""\n"
      "Copyright 2015 Paul A. Bristow.""\n"
      "Copyright 2015 John Maddock.""\n"
      "Distributed under the Boost Software License, Version 1.0.""\n"
      "(See accompanying file LICENSE_1_0.txt or copy at""\n"
      "http://www.boost.org/LICENSE_1_0.txt).""\n"
      "]""\n"
      << std::endl;
    std::string debug_or_optimize;
#ifdef _DEBUG
#if (_DEBUG == 0)
    debug_or_optimize = "Compiled in debug mode.";
#else
    debug_or_optimize = "Compiled in optimise mode.";
#endif
#endif

    // Print out the program/compiler/stdlib/platform names as a Quickbook comment:
    fout << "\n[h5 Program " << short_file_name(sourcefilename) << ", "
      << BOOST_COMPILER << ", "
      << BOOST_STDLIB << ", "
      << BOOST_PLATFORM << (sizeof(void*) == 8 ? ", x64" : ", x86")
      << debug_or_optimize << "[br]"
      << count << " evaluations of each of " << algo_names.size() << " root_finding algorithms."
      << "]"
      << std::endl;
    
    std::cout << count << " evaluations of root_finding." << std::endl;

    BOOST_MATH_CONTROL_FP;

    cpp_bin_float_100 full_value("28");

    cpp_bin_float_100 full_answer ("3.036588971875662519420809578505669635581453977248111123242141654169177268411884961770250390838097895");

    std::copy(max_digits10s.begin(), max_digits10s.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    table_root_info(full_value, full_answer);


    return boost::exit_success;
  }
  catch (std::exception const& ex)
  {
    std::cout << "exception thrown: " << ex.what() << std::endl;
    return boost::exit_failure;
  }
} // int main()

/*
debug

1>  float, maxdigits10 = 9
1>  6 algorithms used.
1>  Digits required = 24.0000000
1>  find root of 28.0000000, expected answer = 3.03658897
1>  Times 156 312 18750 4375 3437 3906
1>  Iterations: 0 0 8 6 4 5
1>  Distance: 0 0 -1 0 0 0
1>  Roots: 3.03658891 3.03658891 3.03658915 3.03658891 3.03658891 3.03658891

release

1>  float, maxdigits10 = 9
1>  6 algorithms used.
1>  Digits required = 24.0000000
1>  find root of 28.0000000, expected answer = 3.03658897
1>  Times 0 312 6875 937 937 937
1>  Iterations: 0 0 8 6 4 5
1>  Distance: 0 0 -1 0 0 0
1>  Roots: 3.03658891 3.03658891 3.03658915 3.03658891 3.03658891 3.03658891


1>
1>  5 algorithms used:
1>  10 algorithms used:
1>  boost::math::cbrt TOMS748 Newton Halley Shroeder boost::math::cbrt TOMS748 Newton Halley Shroeder
1>  2 types compared.
1>  Precision of full type = 102 decimal digits
1>  Find root of 28.000000000000000,
1>  Expected answer = 3.0365889718756625
1>  typeid(T).name()float, maxdigits10 = 9
1>  find root of 28.0000000, expected answer = 3.03658897
1>
1>  Iterations: 0 8 6 4 5
1>  Times 468 8437 4375 3593 4062
1>  Min Time 468
1>  Normalized Times 1.00 18.0 9.35 7.68 8.68
1>  Distance: 0 -1 0 0 0
1>  Roots: 3.03658891 3.03658915 3.03658891 3.03658891 3.03658891
1>  ==================================================================
1>  typeid(T).name()double, maxdigits10 = 17
1>  find root of 28.000000000000000, expected answer = 3.0365889718756625
1>
1>  Iterations: 0 11 7 5 6
1>  Times 312 15000 4531 3906 4375
1>  Min Time 312
1>  Normalized Times 1.00 48.1 14.5 12.5 14.0
1>  Distance: 1 2 0 0 0
1>  Roots: 3.0365889718756622 3.0365889718756618 3.0365889718756627 3.0365889718756627 3.0365889718756627
1>  ==================================================================


Release

1>  5 algorithms used:
1>  10 algorithms used:
1>  boost::math::cbrt TOMS748 Newton Halley Shroeder boost::math::cbrt TOMS748 Newton Halley Shroeder
1>  2 types compared.
1>  Precision of full type = 102 decimal digits
1>  Find root of 28.000000000000000,
1>  Expected answer = 3.0365889718756625
1>  typeid(T).name()float, maxdigits10 = 9
1>  find root of 28.0000000, expected answer = 3.03658897
1>
1>  Iterations: 0 8 6 4 5
1>  Times 312 781 937 937 937
1>  Min Time 312
1>  Normalized Times 1.00 2.50 3.00 3.00 3.00
1>  Distance: 0 -1 0 0 0
1>  Roots: 3.03658891 3.03658915 3.03658891 3.03658891 3.03658891
1>  ==================================================================
1>  typeid(T).name()double, maxdigits10 = 17
1>  find root of 28.000000000000000, expected answer = 3.0365889718756625
1>
1>  Iterations: 0 11 7 5 6
1>  Times 312 1093 937 937 937
1>  Min Time 312
1>  Normalized Times 1.00 3.50 3.00 3.00 3.00
1>  Distance: 1 2 0 0 0
1>  Roots: 3.0365889718756622 3.0365889718756618 3.0365889718756627 3.0365889718756627 3.0365889718756627
1>  ==================================================================



1>  5 algorithms used:
1>  15 algorithms used:
1>  boost::math::cbrt TOMS748 Newton Halley Shroeder boost::math::cbrt TOMS748 Newton Halley Shroeder boost::math::cbrt TOMS748 Newton Halley Shroeder
1>  3 types compared.
1>  Precision of full type = 102 decimal digits
1>  Find root of 28.00000000000000000000000000000000000000000000000000,
1>  Expected answer = 3.036588971875662519420809578505669635581453977248111
1>  typeid(T).name()float, maxdigits10 = 9
1>  find root of 28.0000000, expected answer = 3.03658897
1>
1>  Iterations: 0 8 6 4 5
1>  Times 156 781 937 1093 937
1>  Min Time 156
1>  Normalized Times 1.00 5.01 6.01 7.01 6.01
1>  Distance: 0 -1 0 0 0
1>  Roots: 3.03658891 3.03658915 3.03658891 3.03658891 3.03658891
1>  ==================================================================
1>  typeid(T).name()double, maxdigits10 = 17
1>  find root of 28.000000000000000, expected answer = 3.0365889718756625
1>
1>  Iterations: 0 11 7 5 6
1>  Times 312 1093 937 937 937
1>  Min Time 312
1>  Normalized Times 1.00 3.50 3.00 3.00 3.00
1>  Distance: 1 2 0 0 0
1>  Roots: 3.0365889718756622 3.0365889718756618 3.0365889718756627 3.0365889718756627 3.0365889718756627
1>  ==================================================================
1>  typeid(T).name()class boost::multiprecision::number<class boost::multiprecision::backends::cpp_bin_float<50,10,void,int,0,0>,0>, maxdigits10 = 52
1>  find root of 28.00000000000000000000000000000000000000000000000000, expected answer = 3.036588971875662519420809578505669635581453977248111
1>
1>  Iterations: 0 13 9 6 7
1>  Times 8750 177343 30312 52968 58125
1>  Min Time 8750
1>  Normalized Times 1.00 20.3 3.46 6.05 6.64
1>  Distance: 0 0 -1 0 0
1>  Roots: 3.036588971875662519420809578505669635581453977248106 3.036588971875662519420809578505669635581453977248106 3.036588971875662519420809578505669635581453977248117 3.036588971875662519420809578505669635581453977248106 3.036588971875662519420809578505669635581453977248106
1>  ==================================================================

Reduce accuracy required to 0.5

1>  5 algorithms used:
1>  15 algorithms used:
1>  boost::math::cbrt TOMS748 Newton Halley Shroeder
1>  3 floating_point types compared.
1>  Precision of full type = 102 decimal digits
1>  Find root of 28.00000000000000000000000000000000000000000000000000,
1>  Expected answer = 3.036588971875662519420809578505669635581453977248111
1>  typeid(T).name() = float, maxdigits10 = 9
1>  Digits accuracy fraction required = 0.500000000
1>  find root of 28.0000000, expected answer = 3.03658897
1>
1>  Iterations: 0 8 5 3 4
1>  Times 156 5937 1406 1250 1250
1>  Min Time 156
1>  Normalized Times 1.0 38. 9.0 8.0 8.0
1>  Distance: 0 -1 0 0 0
1>  Roots: 3.03658891 3.03658915 3.03658891 3.03658891 3.03658891
1>  ==================================================================
1>  typeid(T).name() = double, maxdigits10 = 17
1>  Digits accuracy fraction required = 0.50000000000000000
1>  find root of 28.000000000000000, expected answer = 3.0365889718756625
1>
1>  Iterations: 0 8 6 4 5
1>  Times 156 6250 1406 1406 1250
1>  Min Time 156
1>  Normalized Times 1.0 40. 9.0 9.0 8.0
1>  Distance: 1 3695766 0 0 0
1>  Roots: 3.0365889718756622 3.0365889702344129 3.0365889718756627 3.0365889718756627 3.0365889718756627
1>  ==================================================================
1>  typeid(T).name() = class boost::multiprecision::number<class boost::multiprecision::backends::cpp_bin_float<50,10,void,int,0,0>,0>, maxdigits10 = 52
1>  Digits accuracy fraction required = 0.5000000000000000000000000000000000000000000000000000
1>  find root of 28.00000000000000000000000000000000000000000000000000, expected answer = 3.036588971875662519420809578505669635581453977248111
1>
1>  Iterations: 0 11 8 5 6
1>  Times 11562 239843 34843 47500 47812
1>  Min Time 11562
1>  Normalized Times 1.0 21. 3.0 4.1 4.1
1>  Distance: 0 0 -1 0 0
1>  Roots: 3.036588971875662519420809578505669635581453977248106 3.036588971875662519420809578505669635581453977248106 3.036588971875662519420809578505669635581453977248117 3.036588971875662519420809578505669635581453977248106 3.036588971875662519420809578505669635581453977248106
1>  ==================================================================



*/
