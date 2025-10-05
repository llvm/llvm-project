// Copyright Paul A. Bristow 2015

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Comparison of finding roots using TOMS748, Newton-Raphson, Halley & Schroder algorithms.
// root_n_finding_algorithms.cpp  Generalised for nth root version.

// http://en.wikipedia.org/wiki/Cube_root

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!
// This program also writes files in Quickbook tables mark-up format.

#include <boost/cstdlib.hpp>
#include <boost/config.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/tools/roots.hpp>

//using boost::math::policies::policy;
//using boost::math::tools::eps_tolerance; // Binary functor for specified number of bits.
//using boost::math::tools::bracket_and_solve_root;
//using boost::math::tools::toms748_solve;
//using boost::math::tools::halley_iterate; 
//using boost::math::tools::newton_raphson_iterate;
//using boost::math::tools::schroder_iterate;

#include <boost/math/special_functions/next.hpp> // For float_distance.
#include <boost/math/special_functions/pow.hpp> // For pow<N>.
#include <boost/math/tools/tuple.hpp> // for tuple and make_tuple.

#include <boost/multiprecision/cpp_bin_float.hpp> // is binary.
using boost::multiprecision::cpp_bin_float_100;
using boost::multiprecision::cpp_bin_float_50;

#include <boost/timer/timer.hpp>
#include <boost/system/error_code.hpp>
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

#ifdef __FILE__
  std::string sourcefilename = __FILE__;
#else
  std::string sourcefilename("");
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


std::string fp_hardware; // Any hardware features like SEE or AVX

const std::string roots_name = "libs/math/doc/roots/";

const std::string full_roots_name(boost_root + "/libs/math/doc/roots/");

const std::size_t nooftypes = 4;
const std::size_t noofalgos = 4;

double digits_accuracy = 1.0; // 1 == maximum possible accuracy.

std::stringstream ss;

std::ofstream fout;

std::vector<std::string> algo_names =
{
  "TOMS748", "Newton", "Halley", "Schr'''&#xf6;'''der"
};

std::vector<std::string> names =
{
  "float", "double", "long double", "cpp_bin_float50"
};

uintmax_t iters; // Global as value of iterations is not returned.

struct root_info
{ // for a floating-point type, float, double ...
  std::size_t max_digits10; // for type.
  std::string full_typename; // for type from type_id.name().
  std::string short_typename; // for type "float", "double", "cpp_bin_float_50" ....
  std::size_t bin_digits;  // binary in floating-point type numeric_limits<T>::digits;  
  int get_digits; // fraction of maximum possible accuracy required.
  // = digits * digits_accuracy
  // Vector of values (4) for each algorithm, TOMS748, Newton, Halley & Schroder.
  //std::vector< std::int_least64_t> times;  converted to int.
  std::vector<int> times; // arbitrary units (ticks).
  //std::int_least64_t min_time = std::numeric_limits<std::int_least64_t>::max(); // Used to normalize times (as int).
  std::vector<double> normed_times;
  int min_time = (std::numeric_limits<int>::max)(); // Used to normalize times.
  std::vector<uintmax_t> iterations;
  std::vector<long int> distances;
  std::vector<cpp_bin_float_100> full_results;
}; // struct root_info

std::vector<root_info> root_infos;  // One element for each floating-point type used.

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
#if defined(_DEBUG) || !defined(NDEBUG)
  result += "|";
  result += " debug";
#else
  result += "|";
  result += " release";
#endif
  result += "|";
  return result;
} // std::string build_test_name

// Algorithms //////////////////////////////////////////////

// No derivatives - using TOMS748 internally.

template <int N, typename T = double>
struct nth_root_functor_noderiv
{ //  Nth root of x using only function - no derivatives.
  nth_root_functor_noderiv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor just stores value a to find root of.
  }
  T operator()(T const& x)
  {
    using boost::math::pow;
    T fx = pow<N>(x) -a; // Difference (estimate x^n - a).
    return fx;
  }
private:
  T a; // to be 'cube_rooted'.
}; // template <int N, class T> struct nth_root_functor_noderiv

template <int N, class T = double>
T nth_root_noderiv(T x)
{ // return Nth root of x using bracket_and_solve (using NO derivatives).
  using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools; // For bracket_and_solve_root.

  typedef double guess_type;

  int exponent;
  frexp(static_cast<guess_type>(x), &exponent); // Get exponent of z (ignore mantissa).
  T guess = static_cast<T>(ldexp(static_cast<guess_type>(1.), exponent / N)); // Rough guess is to divide the exponent by n.
  //T min = static_cast<T>(ldexp(static_cast<guess_type>(1.) / 2, exponent / N)); // Minimum possible value is half our guess.
  //T max = static_cast<T>(ldexp(static_cast<guess_type>(2.), exponent / N)); // Maximum possible value is twice our guess.

  T factor = 2; // How big steps to take when searching.

  const std::uintmax_t maxit = 50; // Limit to maximum iterations.
  std::uintmax_t it = maxit; // Initially our chosen max iterations, but updated with actual.
  bool is_rising = true; // So if result if guess^3 is too low, then try increasing guess.
  // Some fraction of digits is used to control how accurate to try to make the result.
  int get_digits = std::numeric_limits<T>::digits - 2;
  eps_tolerance<T> tol(get_digits); // Set the tolerance.
  std::pair<T, T> r;
  r =  bracket_and_solve_root(nth_root_functor_noderiv<N, T>(x), guess, factor, is_rising, tol, it);
  iters = it;
  T result = r.first + (r.second - r.first) / 2;  // Midway between brackets.
  return result;
} // template <class T> T nth_root_noderiv(T x)

// Using 1st derivative only Newton-Raphson

template <int N, class T = double>
struct nth_root_functor_1deriv
{ // Functor also returning 1st derivative.
  static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");
  static_assert((N > 0) == true, "root N must be > 0!");

  nth_root_functor_1deriv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor stores value a to find root of, for example:
  }
  std::pair<T, T> operator()(T const& x)
  { // Return both f(x) and f'(x).
    using boost::math::pow; // // Compile-time integral power.
    T p = pow<N - 1>(x);
    return std::make_pair(p * x - a, N * p); // 'return' both fx and dx.
  }

private:
  T a; // to be 'nth_rooted'.
}; // struct nthroot__functor_1deriv

template <int N, class T = double>
T nth_root_1deriv(T x)
{ // return nth root of x using 1st derivative and Newton_Raphson.
  using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools; // For newton_raphson_iterate.

  static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");
  static_assert((N > 0) == true, "root N must be > 0!");
  static_assert((N > 1000) == false, "root N is too big!");

  typedef double guess_type;

  int exponent;
  frexp(static_cast<guess_type>(x), &exponent); // Get exponent of z (ignore mantissa).
  T guess = static_cast<T>(ldexp(static_cast<guess_type>(1.), exponent / N)); // Rough guess is to divide the exponent by n.
  T min = static_cast<T>(ldexp(static_cast<guess_type>(1.) / 2, exponent / N)); // Minimum possible value is half our guess.
  T max = static_cast<T>(ldexp(static_cast<guess_type>(2.), exponent / N)); // Maximum possible value is twice our guess.

  int digits = std::numeric_limits<T>::digits; // Maximum possible binary digits accuracy for type T.
  int get_digits = static_cast<int>(digits * 0.6);
  const std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = newton_raphson_iterate(nth_root_functor_1deriv<N, T>(x), guess, min, max, get_digits, it);
  iters = it;
  return result;
} // T nth_root_1_deriv  Newton-Raphson

// Using 1st and 2nd derivatives with Halley algorithm.

template <int N, class T = double>
struct nth_root_functor_2deriv
{ // Functor returning both 1st and 2nd derivatives.
  static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");
  static_assert((N > 0) == true, "root N must be > 0!");

  nth_root_functor_2deriv(T const& to_find_root_of) : a(to_find_root_of)
  { // Constructor stores value a to find root of, for example:
  }

  // using boost::math::tuple; // to return three values.
  std::tuple<T, T, T> operator()(T const& x)
  { // Return f(x), f'(x) and f''(x).
    using boost::math::pow; // Compile-time integral power.
    T p = pow<N - 2>(x);

    return std::make_tuple(p * x * x - a, p * x * N, p * N * (N - 1)); // 'return' fx, dx and d2x.
  }
private:
  T a; // to be 'nth_rooted'.
};

template <int N, class T = double>
T nth_root_2deriv(T x)
{ // return nth root of x using 1st and 2nd derivatives and Halley.

  using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools; // For halley_iterate.

  static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");
  static_assert((N > 0) == true, "root N must be > 0!");
  static_assert((N > 1000) == false, "root N is too big!");

  typedef double guess_type;

  int exponent;
  frexp(static_cast<guess_type>(x), &exponent); // Get exponent of z (ignore mantissa).
  T guess = static_cast<T>(ldexp(static_cast<guess_type>(1.), exponent / N)); // Rough guess is to divide the exponent by n.
  T min = static_cast<T>(ldexp(static_cast<guess_type>(1.) / 2, exponent / N)); // Minimum possible value is half our guess.
  T max = static_cast<T>(ldexp(static_cast<guess_type>(2.), exponent / N)); // Maximum possible value is twice our guess.

  int digits = std::numeric_limits<T>::digits; // Maximum possible binary digits accuracy for type T.
  int get_digits = static_cast<int>(digits * 0.4);
  const std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = halley_iterate(nth_root_functor_2deriv<N, T>(x), guess, min, max, get_digits, it);
  iters = it;

  return result;
} // nth_2deriv Halley

template <int N, class T = double>
T nth_root_2deriv_s(T x)
{ // return nth root of x using 1st and 2nd derivatives and Schroder.

  using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools; // For schroder_iterate.

  static_assert(boost::is_integral<T>::value == false, "Only floating-point type types can be used!");
  static_assert((N > 0) == true, "root N must be > 0!");
  static_assert((N > 1000) == false, "root N is too big!");

  typedef double guess_type;

  int exponent;
  frexp(static_cast<guess_type>(x), &exponent); // Get exponent of z (ignore mantissa).
  T guess = static_cast<T>(ldexp(static_cast<guess_type>(1.), exponent / N)); // Rough guess is to divide the exponent by n.
  T min = static_cast<T>(ldexp(static_cast<guess_type>(1.) / 2, exponent / N)); // Minimum possible value is half our guess.
  T max = static_cast<T>(ldexp(static_cast<guess_type>(2.), exponent / N)); // Maximum possible value is twice our guess.

  int get_digits = static_cast<int>(std::numeric_limits<T>::digits * 0.4);
  const std::uintmax_t maxit = 20;
  std::uintmax_t it = maxit;
  T result = schroder_iterate(nth_root_functor_2deriv<N, T>(x), guess, min, max, get_digits, it);
  iters = it;

  return result;
} // T nth_root_2deriv_s Schroder

//////////////////////////////////////////////////////// end of algorithms - perhaps in a separate .hpp?

//! Print 4 floating-point types info: max_digits10, digits and required accuracy digits as a Quickbook table.
int table_type_info(double digits_accuracy)
{
  std::string qbk_name = full_roots_name; // Prefix by boost_root file.

  qbk_name += "type_info_table";
  std::stringstream ss;
  ss.precision(3);
  ss << "_" << digits_accuracy * 100;
  qbk_name += ss.str();

#ifdef _MSC_VER
  qbk_name += "_msvc.qbk";
#else // assume GCC
  qbk_name += "_gcc.qbk";
#endif

  // Example: type_info_table_100_msvc.qbk
  fout.open(qbk_name, std::ios_base::out);

  if (fout.is_open())
  {
    std::cout << "Output type table to " << qbk_name << std::endl;
  }
  else
  { // Failed to open.
    std::cout << " Open file " << qbk_name << " for output failed!" << std::endl;
    std::cout << "errno " << errno << std::endl;
    return errno;
  }

  fout <<
    "[/"
    << qbk_name
    << "\n"
    "Copyright 2015 Paul A. Bristow.""\n"
    "Copyright 2015 John Maddock.""\n"
    "Distributed under the Boost Software License, Version 1.0.""\n"
    "(See accompanying file LICENSE_1_0.txt or copy at""\n"
    "http://www.boost.org/LICENSE_1_0.txt).""\n"
    "]""\n"
    << std::endl;

  fout << "[h6 Fraction of maximum possible bits of accuracy required is " << digits_accuracy << ".]\n" << std::endl;

  std::string table_id("type_info");
  table_id += ss.str(); // Fraction digits accuracy.

#ifdef _MSC_VER
  table_id += "_msvc";
#else // assume GCC
  table_id += "_gcc";
#endif

  fout << "[table:" << table_id << " Digits for float, double, long double and cpp_bin_float_50\n"
    << "[[type name] [max_digits10] [binary digits] [required digits]]\n";// header.

  // For all fout types:

  fout  << "[[" << "float" << "]"
    << "[" << std::numeric_limits<float>::max_digits10 << "]"  // max_digits10
    << "[" << std::numeric_limits<float>::digits << "]"// < "Binary digits 
    << "[" << static_cast<int>(std::numeric_limits<float>::digits * digits_accuracy) << "]]\n"; // Accuracy digits.

  fout << "[[" << "float" << "]"
    << "[" << std::numeric_limits<double>::max_digits10 << "]"  // max_digits10
    << "[" << std::numeric_limits<double>::digits << "]"// < "Binary digits 
    << "[" << static_cast<int>(std::numeric_limits<double>::digits * digits_accuracy) << "]]\n"; // Accuracy digits.

  fout << "[[" << "long double" << "]"
    << "[" << std::numeric_limits<long double>::max_digits10 << "]"  // max_digits10
    << "[" << std::numeric_limits<long double>::digits << "]"// < "Binary digits 
    << "[" << static_cast<int>(std::numeric_limits<long double>::digits * digits_accuracy) << "]]\n"; // Accuracy digits.

  fout << "[[" << "cpp_bin_float_50" << "]"
    << "[" << std::numeric_limits<cpp_bin_float_50>::max_digits10 << "]"  // max_digits10
    << "[" << std::numeric_limits<cpp_bin_float_50>::digits << "]"// < "Binary digits 
    << "[" << static_cast<int>(std::numeric_limits<cpp_bin_float_50>::digits * digits_accuracy) << "]]\n"; // Accuracy digits.

  fout << "] [/table table_id_msvc] \n" << std::endl; // End of table.

  fout.close();
  return 0;
} // type_table

//! Evaluate root N timing for each algorithm, and for one floating-point type T. 
template <int N, typename T>
int test_root(cpp_bin_float_100 big_value, cpp_bin_float_100 answer, const char* type_name, std::size_t type_no)
{
  std::size_t max_digits = 2 + std::numeric_limits<T>::digits * 3010 / 10000;
  // For new versions use max_digits10
  // std::cout.precision(std::numeric_limits<T>::max_digits10);
  std::cout.precision(max_digits);
  std::cout << std::showpoint << std::endl; // Show trailing zeros too.

  root_infos.push_back(root_info()); 

  root_infos[type_no].max_digits10 = max_digits;
  root_infos[type_no].full_typename = typeid(T).name(); // Full typename.
  root_infos[type_no].short_typename = type_name; // Short typename.
  root_infos[type_no].bin_digits = std::numeric_limits<T>::digits;
  root_infos[type_no].get_digits = static_cast<int>(std::numeric_limits<T>::digits * digits_accuracy);

  T to_root = static_cast<T>(big_value);

  T result; // root
  T sum = 0;
  T ans = static_cast<T>(answer);

  using boost::timer::nanosecond_type;
  using boost::timer::cpu_times;
  using boost::timer::cpu_timer;

  int eval_count = std::is_floating_point<T>::value ? 10000000 : 100000; // To give a sufficiently stable timing for the fast built-in types,
  //int eval_count = 1000000; // To give a sufficiently stable timing for the fast built-in types,
  // This takes an inconveniently long time for multiprecision cpp_bin_float_50 etc  types.

  cpu_times now; // Holds wall, user and system times.

  { // Evaluate times etc for each algorithm.
    //algorithm_names.push_back("TOMS748"); // 
    cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
    ti.start();
    for (long i = 0; i < eval_count; ++i)
    {
      result = nth_root_noderiv<N, T>(to_root); // 
      sum += result;
    }
    now = ti.elapsed();
    int time = static_cast<int>(now.user / eval_count);
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
    for (long i = 0; i < eval_count; ++i)
    {
      result = nth_root_1deriv<N, T>(to_root); // 
      sum += result;
    }
    now = ti.elapsed();
    int time = static_cast<int>(now.user / eval_count);
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
    for (long i = 0; i < eval_count; ++i)
    {
      result = nth_root_2deriv<N>(to_root); // 
      sum += result;
    }
    now = ti.elapsed();
    int time = static_cast<int>(now.user / eval_count);
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
    // algorithm_names.push_back("Schroder"); // algorithm
    cpu_timer ti; // Can start, pause, resume and stop, and read elapsed.
    ti.start();
    for (long i = 0; i < eval_count; ++i)
    {
      result = nth_root_2deriv_s<N>(to_root); // 
      sum += result;
    }
    now = ti.elapsed();
    int time = static_cast<int>(now.user / eval_count);
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
  for (size_t i = 0; i != root_infos[type_no].times.size(); i++) // For each time.
  { // Normalize times.
    root_infos[type_no].normed_times.push_back(static_cast<double>(root_infos[type_no].times[i]) / root_infos[type_no].min_time);
  }

  std::cout << "Accumulated result was: " << sum << std::endl;

  return 4;  // eval_count of how many algorithms used.
} // test_root

/*! Fill array of times, iterations, etc for Nth root for all 4 types,
 and write a table of results in Quickbook format.
 */
template <int N>
void table_root_info(cpp_bin_float_100 full_value)
{
   using std::abs;
  std::cout << nooftypes << " floating-point types tested:" << std::endl;
#if defined(_DEBUG) || !defined(NDEBUG)
  std::cout << "Compiled in debug mode." << std::endl;
#else
  std::cout << "Compiled in optimise mode." << std::endl;
#endif
  std::cout << "FP hardware " << fp_hardware << std::endl;
  // Compute the 'right' answer for root N at 100 decimal digits.
  cpp_bin_float_100 full_answer = nth_root_noderiv<N, cpp_bin_float_100>(full_value);

  root_infos.clear(); // Erase any previous data.
  // Fill the elements of the array for each floating-point type.

  test_root<N, float>(full_value, full_answer, "float", 0);
  test_root<N, double>(full_value, full_answer, "double", 1);
  test_root<N, long double>(full_value, full_answer, "long double", 2);
  test_root<N, cpp_bin_float_50>(full_value, full_answer, "cpp_bin_float_50", 3);

  // Use info from 4 floating point types to

  // Prepare Quickbook table for a single root
  // with columns of times, iterations, distances repeated for various floating-point types,
  // and 4 rows for each algorithm.

  std::stringstream table_info;
  table_info.precision(3);
  table_info << "[table:root_" << N << " " << N << "th root(" << static_cast<float>(full_value) << ") for float, double, long double and cpp_bin_float_50 types";
  if (fp_hardware != "")
  {
    table_info << ", using " << fp_hardware;
  }
  table_info << std::endl;

  fout << table_info.str()
    << "[[][float][][][] [][double][][][] [][long d][][][] [][cpp50][][]]\n"
    << "[[Algo     ]";
  for (size_t tp = 0; tp != nooftypes; tp++)
  { // For all types:
    fout << "[Its]" << "[Times]" << "[Norm]" << "[Dis]" << "[ ]";
  }
  fout << "]" << std::endl;

  // Row for all algorithms.
  for (std::size_t algo = 0; algo != noofalgos; algo++)
  {
    fout << "[[" << std::left << std::setw(9) << algo_names[algo] << "]";
    for (size_t tp = 0; tp != nooftypes; tp++)
    { // For all types:
      fout
        << "[" << std::right << std::showpoint
        << std::setw(3) << std::setprecision(2) << root_infos[tp].iterations[algo] << "]["
        << std::setw(5) << std::setprecision(5) << root_infos[tp].times[algo] << "][";
      fout << std::setw(3) << std::setprecision(3);
        double normed_time = root_infos[tp].normed_times[algo];
        if (abs(normed_time - 1.00) <= 0.05)
        { // At or near the best time, so show as blue.
          fout << "[role blue " << normed_time << "]";
        }
        else if (abs(normed_time) > 4.)
        { // markedly poor so show as red.
          fout << "[role red " << normed_time << "]";
        }
        else
        { // Not the best, so normal black.
          fout << normed_time;
        }
        fout << "]["
        << std::setw(3) << std::setprecision(2) << root_infos[tp].distances[algo] << "][ ]";
    } // tp
    fout << "]" << std::endl;
  } // for algo
  fout << "] [/end of table root]\n";
} // void table_root_info

/*! Output program header, table of type info, and tables for 4 algorithms and 4 floating-point types,
 for Nth root required digits_accuracy.
 */

int roots_tables(cpp_bin_float_100 full_value, double digits_accuracy)
{
  ::digits_accuracy = digits_accuracy;
  // Save globally so that it is available to root-finding algorithms. Ugly :-(

#if defined(_DEBUG) || !defined(NDEBUG)
  std::string debug_or_optimize("Compiled in debug mode.");
#else
     std::string debug_or_optimize("Compiled in optimise mode.");
#endif

  // Create filename for roots_table
  std::string qbk_name = full_roots_name;
  qbk_name += "roots_table";

  std::stringstream ss;
  ss.precision(3);
  // ss << "_" << N // now put all the tables in one .qbk file?
    ss << "_" << digits_accuracy * 100
    << std::flush;
  // Assume only save optimize mode runs, so don't add any  _DEBUG info.
  qbk_name += ss.str();

#ifdef _MSC_VER
  qbk_name += "_msvc";
#else // assume GCC
  qbk_name += "_gcc";
#endif 
  if (fp_hardware != "")
  {
    qbk_name += fp_hardware;
  }
  qbk_name += ".qbk";

  fout.open(qbk_name, std::ios_base::out);

  if (fout.is_open())
  {
    std::cout << "Output root table to " << qbk_name << std::endl;
  }
  else
  { // Failed to open.
    std::cout << " Open file " << qbk_name << " for output failed!" << std::endl;
    std::cout << "errno " << errno << std::endl;
    return errno;
  }

  fout <<
    "[/"
    << qbk_name
    << "\n"
    "Copyright 2015 Paul A. Bristow.""\n"
    "Copyright 2015 John Maddock.""\n"
    "Distributed under the Boost Software License, Version 1.0.""\n"
    "(See accompanying file LICENSE_1_0.txt or copy at""\n"
    "http://www.boost.org/LICENSE_1_0.txt).""\n"
    "]""\n"
    << std::endl;

  // Print out the program/compiler/stdlib/platform names as a Quickbook comment:
  fout << "\n[h6 Program " << sourcefilename << ",\n "
    << BOOST_COMPILER << ", "
    << BOOST_STDLIB << ", "
    << BOOST_PLATFORM << "\n"
    << debug_or_optimize 
    << ((fp_hardware != "") ? ", " + fp_hardware : "")
    << "]" // [h6 close].
    << std::endl;

  fout << "Fraction of full accuracy " << digits_accuracy << std::endl;

  table_root_info<5>(full_value);
  table_root_info<7>(full_value);
  table_root_info<11>(full_value);

  fout.close();

  //   table_type_info(digits_accuracy);

  return 0;
} // roots_tables


int main()
{
  using namespace boost::multiprecision;
  using namespace boost::math;


  try
  {
    std::cout << "Tests run with " << BOOST_COMPILER << ", "
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << ", ";

// How to: Configure Visual C++ Projects to Target 64-Bit Platforms
// https://msdn.microsoft.com/en-us/library/9yb4317s.aspx

#ifdef _M_X64 // Defined for compilations that target x64 processors.
    std::cout << "X64 " << std::endl;
    fp_hardware += "_X64";
#else
#  ifdef _M_IX86
     std::cout << "X32 " << std::endl;
     fp_hardware += "_X86";
#  endif
#endif

#ifdef _M_AMD64
    std::cout << "AMD64 " << std::endl;
 //   fp_hardware += "_AMD64";
#endif

// https://msdn.microsoft.com/en-us/library/7t5yh4fd.aspx  
// /arch (x86) options /arch:[IA32|SSE|SSE2|AVX|AVX2]
// default is to use SSE and SSE2 instructions by default.
// https://msdn.microsoft.com/en-us/library/jj620901.aspx
// /arch (x64) options /arch:AVX and /arch:AVX2

// MSVC doesn't bother to set these SSE macros!
// http://stackoverflow.com/questions/18563978/sse-sse2-is-enabled-control-in-visual-studio
// https://msdn.microsoft.com/en-us/library/b0084kay.aspx  predefined macros.

// But some of these macros are *not* defined by MSVC, 
// unlike AVX (but *are* defined by GCC and Clang). 
// So the macro code above does define them.
#if (defined(_M_AMD64) || defined (_M_X64))
#ifndef _M_X64
#  define _M_X64
#endif
#ifndef __SSE2__
#  define __SSE2__
#endif
#else
#  ifdef _M_IX86_FP // Expands to an integer literal value indicating which /arch compiler option was used:
    std::cout << "Floating-point _M_IX86_FP = " << _M_IX86_FP << std::endl;
#  if (_M_IX86_FP == 2) // 2 if /arch:SSE2, /arch:AVX or /arch:AVX2 
#    define __SSE2__ // x32
#  elif (_M_IX86_FP == 1) // 1 if /arch:SSE was used.
#    define __SSE__ // x32
#  elif (_M_IX86_FP == 0) // 0 if /arch:IA32 was used.
#    define _X32 // No special FP instructions.
#  endif
# endif
#endif
// Set the fp_hardware that is used in the .qbk filename.
#ifdef __AVX2__
    std::cout << "Floating-point AVX2 " << std::endl;
    fp_hardware += "_AVX2";
#  else 
#    ifdef __AVX__
    std::cout << "Floating-point AVX " << std::endl;
    fp_hardware += "_AVX";
#    else
#      ifdef __SSE2__
    std::cout << "Floating-point SSE2 " << std::endl;
    fp_hardware += "_SSE2";
#      else
#        ifdef __SSE__
    std::cout << "Floating-point SSE " << std::endl;
    fp_hardware += "_SSE";
#        endif
#      endif
#   endif
# endif

#ifdef _M_IX86
    std::cout << "Floating-point X86 _M_IX86 = " << _M_IX86 << std::endl;
    // https://msdn.microsoft.com/en-us/library/aa273918%28v=vs.60%29.aspx#_predir_table_1..3
    // 600 = Pentium Pro
#endif

#ifdef _MSC_FULL_VER
    std::cout << "Floating-point _MSC_FULL_VER " << _MSC_FULL_VER << std::endl;
#endif

#ifdef __MSVC_RUNTIME_CHECKS
    std::cout << "Runtime __MSVC_RUNTIME_CHECKS " << std::endl;
#endif

    BOOST_MATH_CONTROL_FP;

    cpp_bin_float_100 full_value("28.");
    // Compute full answer to more than precision of tests.
    //T value = 28.; // integer (exactly representable as floating-point)
    // whose cube root is *not* exactly representable.
    // Wolfram Alpha command N[28 ^ (1 / 3), 100] computes cube root to 100 decimal digits.
    // 3.036588971875662519420809578505669635581453977248111123242141654169177268411884961770250390838097895

    std::cout.precision(100);
    std::cout << "value " << full_value << std::endl;
   // std::cout << ",\n""answer = " << full_answer << std::endl;
    std::cout.precision(6);
   // cbrt cpp_bin_float_100 full_answer("3.036588971875662519420809578505669635581453977248111123242141654169177268411884961770250390838097895");

    // Output the table of types, maxdigits10 and digits and required digits for some accuracies.

    // Output tables for some roots at full accuracy.
    roots_tables(full_value, 1.);

    // Output tables for some roots at less accuracy.
    //roots_tables(full_value, 0.75);

    return boost::exit_success;
  }
  catch (std::exception const& ex)
  {
    std::cout << "exception thrown: " << ex.what() << std::endl;
    return boost::exit_failure;
  }
} // int main()

/*

*/
