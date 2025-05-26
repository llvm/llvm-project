//  Copyright (c) 2006-7 John Maddock
//  Copyright (c) 2021 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_CONFIG_HPP
#define BOOST_MATH_TOOLS_CONFIG_HPP

#ifdef _MSC_VER
#pragma once
#endif

#ifndef __CUDACC_RTC__

#include <boost/math/tools/is_standalone.hpp>

// Minimum language standard transition
#ifdef _MSVC_LANG
#  if _MSVC_LANG < 201402L
#    pragma message("Boost.Math requires C++14");
#  endif
#  if _MSC_VER == 1900
#    pragma message("MSVC 14.0 has broken C++14 constexpr support. Support for this compiler will be removed in Boost 1.86")
#  endif
#else
#  if __cplusplus < 201402L
#    warning "Boost.Math requires C++14"
#  endif
#endif

#ifndef BOOST_MATH_STANDALONE
#include <boost/config.hpp>


// The following are all defined as standalone macros as well
// If Boost.Config is available just use those definitions because they are more fine-grained

// Could be defined in TR1
#ifndef BOOST_MATH_PREVENT_MACRO_SUBSTITUTION
#  define BOOST_MATH_PREVENT_MACRO_SUBSTITUTION BOOST_PREVENT_MACRO_SUBSTITUTION
#endif

#define BOOST_MATH_CXX14_CONSTEXPR BOOST_CXX14_CONSTEXPR
#ifdef BOOST_NO_CXX14_CONSTEXPR
#  define BOOST_MATH_NO_CXX14_CONSTEXPR
#endif

#define BOOST_MATH_IF_CONSTEXPR BOOST_IF_CONSTEXPR
#ifdef BOOST_NO_CXX17_IF_CONSTEXPR
#  define BOOST_MATH_NO_CXX17_IF_CONSTEXPR
#endif

#ifdef BOOST_NO_CXX17_HDR_EXECUTION
#  define BOOST_MATH_NO_CXX17_HDR_EXECUTION
#endif

#ifdef BOOST_HAS_THREADS
#  define BOOST_MATH_HAS_THREADS
#endif
#ifdef BOOST_DISABLE_THREADS
#  define BOOST_MATH_DISABLE_THREADS
#endif
#ifdef BOOST_NO_CXX11_THREAD_LOCAL
#  define BOOST_MATH_NO_CXX11_THREAD_LOCAL
#endif

#ifdef BOOST_NO_EXCEPTIONS
#  define BOOST_MATH_NO_EXCEPTIONS
#endif

#ifdef BOOST_NO_TYPEID
#  define BOOST_MATH_NO_TYPEID
#endif
#ifdef BOOST_NO_RTTI
#  define BOOST_MATH_NO_RTTI
#endif

#define BOOST_MATH_NOINLINE BOOST_NOINLINE
#define BOOST_MATH_FORCEINLINE BOOST_FORCEINLINE

#define BOOST_MATH_JOIN(X, Y) BOOST_JOIN(X, Y)
#define BOOST_MATH_STRINGIZE(X) BOOST_STRINGIZE(X)

#else // Things from boost/config that are required, and easy to replicate

#define BOOST_MATH_PREVENT_MACRO_SUBSTITUTION
#define BOOST_MATH_NO_REAL_CONCEPT_TESTS
#define BOOST_MATH_NO_DISTRIBUTION_CONCEPT_TESTS
#define BOOST_MATH_NO_LEXICAL_CAST

// Since Boost.Multiprecision is in active development some tests do not fully cooperate yet.
#define BOOST_MATH_NO_MP_TESTS

#if (__cplusplus > 201400L || _MSVC_LANG > 201400L)
#define BOOST_MATH_CXX14_CONSTEXPR constexpr
#else
#define BOOST_MATH_CXX14_CONSTEXPR
#define BOOST_MATH_NO_CXX14_CONSTEXPR
#endif // BOOST_MATH_CXX14_CONSTEXPR

#if (__cplusplus > 201700L || _MSVC_LANG > 201700L)
#define BOOST_MATH_IF_CONSTEXPR if constexpr

// Clang on mac provides the execution header with none of the functionality. TODO: Check back on this
// https://en.cppreference.com/w/cpp/compiler_support "Standardization of Parallelism TS"
#  if !__has_include(<execution>) || (defined(__APPLE__) && defined(__clang__))
#  define BOOST_MATH_NO_CXX17_HDR_EXECUTION
#  endif
#else
#  define BOOST_MATH_IF_CONSTEXPR if
#  define BOOST_MATH_NO_CXX17_IF_CONSTEXPR
#  define BOOST_MATH_NO_CXX17_HDR_EXECUTION
#endif

#if __cpp_lib_gcd_lcm >= 201606L
#define BOOST_MATH_HAS_CXX17_NUMERIC
#endif

#define BOOST_MATH_JOIN(X, Y) BOOST_MATH_DO_JOIN(X, Y)
#define BOOST_MATH_DO_JOIN(X, Y) BOOST_MATH_DO_JOIN2(X,Y)
#define BOOST_MATH_DO_JOIN2(X, Y) X##Y

#define BOOST_MATH_STRINGIZE(X) BOOST_MATH_DO_STRINGIZE(X)
#define BOOST_MATH_DO_STRINGIZE(X) #X

#ifdef BOOST_MATH_DISABLE_THREADS // No threads, do nothing
// Detect thread support via STL implementation
#elif defined(__has_include)
#  if !__has_include(<thread>) || !__has_include(<mutex>) || !__has_include(<future>) || !__has_include(<atomic>)
#     define BOOST_MATH_DISABLE_THREADS
#  else
#     define BOOST_MATH_HAS_THREADS
#  endif 
#else
#  define BOOST_MATH_HAS_THREADS // The default assumption is that the machine has threads
#endif // Thread Support

#ifdef BOOST_MATH_DISABLE_THREADS
#  define BOOST_MATH_NO_CXX11_THREAD_LOCAL
#endif // BOOST_MATH_DISABLE_THREADS

#ifdef __GNUC__
#  if !defined(__EXCEPTIONS) && !defined(BOOST_MATH_NO_EXCEPTIONS)
#     define BOOST_MATH_NO_EXCEPTIONS
#  endif
   //
   // Make sure we have some std lib headers included so we can detect __GXX_RTTI:
   //
#  include <algorithm>  // for min and max
#  include <limits>
#  ifndef __GXX_RTTI
#     ifndef BOOST_MATH_NO_TYPEID
#        define BOOST_MATH_NO_TYPEID
#     endif
#     ifndef BOOST_MATH_NO_RTTI
#        define BOOST_MATH_NO_RTTI
#     endif
#  endif
#endif

#if !defined(BOOST_MATH_NOINLINE)
#  if defined(_MSC_VER)
#    define BOOST_MATH_NOINLINE __declspec(noinline)
#  elif defined(__GNUC__) && __GNUC__ > 3
     // Clang also defines __GNUC__ (as 4)
#    if defined(__CUDACC__)
       // nvcc doesn't always parse __noinline__,
       // see: https://svn.boost.org/trac/boost/ticket/9392
#      define BOOST_MATH_NOINLINE __attribute__ ((noinline))
#    elif defined(__HIP__)
       // See https://github.com/boostorg/config/issues/392
#      define BOOST_MATH_NOINLINE __attribute__ ((noinline))
#    else
#      define BOOST_MATH_NOINLINE __attribute__ ((__noinline__))
#    endif
#  else
#    define BOOST_MATH_NOINLINE
#  endif
#endif

#if !defined(BOOST_MATH_FORCEINLINE)
#  if defined(_MSC_VER)
#    define BOOST_MATH_FORCEINLINE __forceinline
#  elif defined(__GNUC__) && __GNUC__ > 3
     // Clang also defines __GNUC__ (as 4)
#    define BOOST_MATH_FORCEINLINE inline __attribute__ ((__always_inline__))
#  else
#    define BOOST_MATH_FORCEINLINE inline
#  endif
#endif

#endif // BOOST_MATH_STANDALONE

// Support compilers with P0024R2 implemented without linking TBB
// https://en.cppreference.com/w/cpp/compiler_support
#if !defined(BOOST_MATH_NO_CXX17_HDR_EXECUTION) && defined(BOOST_MATH_HAS_THREADS)
#  define BOOST_MATH_EXEC_COMPATIBLE
#endif

// C++23
#if __cplusplus > 202002L || (defined(_MSVC_LANG) &&_MSVC_LANG > 202002L)
#  if __GNUC__ >= 13
     // libstdc++3 only defines to/from_chars for std::float128_t when one of these defines are set
     // otherwise we're right out of luck...
#    if defined(_GLIBCXX_LDOUBLE_IS_IEEE_BINARY128) || defined(_GLIBCXX_HAVE_FLOAT128_MATH)
#      include <cstring> // std::strlen is used with from_chars
#      include <charconv>
#      include <stdfloat>
#      define BOOST_MATH_USE_CHARCONV_FOR_CONVERSION
#    endif
#  endif
#endif

#include <algorithm>  // for min and max
#include <limits>
#include <cmath>
#include <climits>
#include <cfloat>

#include <boost/math/tools/user.hpp>

#if (defined(__NetBSD__)\
   || (defined(__hppa) && !defined(__OpenBSD__)) || (defined(__NO_LONG_DOUBLE_MATH) && (DBL_MANT_DIG != LDBL_MANT_DIG))) \
   && !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
//#  define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif

#if defined(__EMSCRIPTEN__) && !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
#  define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif

#ifdef __IBMCPP__
//
// For reasons I don't understand, the tests with IMB's compiler all
// pass at long double precision, but fail with real_concept, those tests
// are disabled for now.  (JM 2012).
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#  define BOOST_MATH_NO_REAL_CONCEPT_TESTS
#endif // BOOST_MATH_NO_REAL_CONCEPT_TESTS
#endif
#ifdef sun
// Any use of __float128 in program startup code causes a segfault  (tested JM 2015, Solaris 11).
#  define BOOST_MATH_DISABLE_FLOAT128
#endif
#ifdef __HAIKU__
//
// Not sure what's up with the math detection on Haiku, but linking fails with
// float128 code enabled, and we don't have an implementation of __expl, so
// disabling long double functions for now as well.
#  define BOOST_MATH_DISABLE_FLOAT128
#  define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif
#if (defined(macintosh) || defined(__APPLE__) || defined(__APPLE_CC__)) && ((LDBL_MANT_DIG == 106) || (__LDBL_MANT_DIG__ == 106)) && !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
//
// Darwin's rather strange "double double" is rather hard to
// support, it should be possible given enough effort though...
//
#  define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif
#if !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && (LDBL_MANT_DIG == 106) && (LDBL_MIN_EXP > DBL_MIN_EXP)
//
// Generic catch all case for gcc's "double-double" long double type.
// We do not support this as it's not even remotely IEEE conforming:
//
#  define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif
#if defined(unix) && defined(__INTEL_COMPILER) && (__INTEL_COMPILER <= 1000) && !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
//
// Intel compiler prior to version 10 has sporadic problems
// calling the long double overloads of the std lib math functions:
// calling ::powl is OK, but std::pow(long double, long double) 
// may segfault depending upon the value of the arguments passed 
// and the specific Linux distribution.
//
// We'll be conservative and disable long double support for this compiler.
//
// Comment out this #define and try building the tests to determine whether
// your Intel compiler version has this issue or not.
//
#  define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#endif
#if defined(unix) && defined(__INTEL_COMPILER)
//
// Intel compiler has sporadic issues compiling std::fpclassify depending on
// the exact OS version used.  Use our own code for this as we know it works
// well on Intel processors:
//
#define BOOST_MATH_DISABLE_STD_FPCLASSIFY
#endif

#if defined(_MSC_VER) && !defined(_WIN32_WCE)
   // Better safe than sorry, our tests don't support hardware exceptions:
#  define BOOST_MATH_CONTROL_FP _control87(MCW_EM,MCW_EM)
#endif

#ifdef __IBMCPP__
#  define BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS
#endif

#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901))
#  define BOOST_MATH_USE_C99
#endif

#if (defined(__hpux) && !defined(__hppa))
#  define BOOST_MATH_USE_C99
#endif

#if defined(__GNUC__) && defined(_GLIBCXX_USE_C99)
#  define BOOST_MATH_USE_C99
#endif

#if defined(_LIBCPP_VERSION) && !defined(_MSC_VER)
#  define BOOST_MATH_USE_C99
#endif

#if defined(__CYGWIN__) || defined(__HP_aCC) || defined(__INTEL_COMPILER) \
  || defined(BOOST_NO_NATIVE_LONG_DOUBLE_FP_CLASSIFY) \
  || (defined(__GNUC__) && !defined(BOOST_MATH_USE_C99))\
  || defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
#  define BOOST_MATH_NO_NATIVE_LONG_DOUBLE_FP_CLASSIFY
#endif

#if defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x590)

namespace boost { namespace math { namespace tools { namespace detail {
template <typename T>
struct type {};

template <typename T, T n>
struct non_type {};
}}}} // Namespace boost, math tools, detail

#  define BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(t)              boost::math::tools::detail::type<t>* = 0
#  define BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(t)         boost::math::tools::detail::type<t>*
#  define BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE(t, v)       boost::math::tools::detail::non_type<t, v>* = 0
#  define BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)  boost::math::tools::detail::non_type<t, v>*

#  define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(t)         \
             , BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(t)
#  define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(t)    \
             , BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(t)
#  define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_NON_TYPE(t, v)  \
             , BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE(t, v)
#  define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)  \
             , BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)

#else

// no workaround needed: expand to nothing

#  define BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(t)
#  define BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(t)
#  define BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE(t, v)
#  define BOOST_MATH_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)

#  define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE(t)
#  define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(t)
#  define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_NON_TYPE(t, v)
#  define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_NON_TYPE_SPEC(t, v)


#endif // __SUNPRO_CC

#if (defined(__SUNPRO_CC) || defined(__hppa) || defined(__GNUC__)) && !defined(BOOST_MATH_SMALL_CONSTANT)
// Sun's compiler emits a hard error if a constant underflows,
// as does aCC on PA-RISC, while gcc issues a large number of warnings:
#  define BOOST_MATH_SMALL_CONSTANT(x) 0.0
#else
#  define BOOST_MATH_SMALL_CONSTANT(x) x
#endif

//
// Tune performance options for specific compilers,
// but check at each step that nothing has been previously defined by the user first
//
#ifdef _MSC_VER
#  ifndef BOOST_MATH_POLY_METHOD
#    define BOOST_MATH_POLY_METHOD 2
#  endif
#if _MSC_VER <= 1900
#  ifndef BOOST_MATH_POLY_METHOD
#    define BOOST_MATH_RATIONAL_METHOD 1
#  endif
#else
#  ifndef BOOST_MATH_RATIONAL_METHOD
#    define BOOST_MATH_RATIONAL_METHOD 2
#  endif
#endif
#if _MSC_VER > 1900
#  ifndef BOOST_MATH_INT_TABLE_TYPE
#    define BOOST_MATH_INT_TABLE_TYPE(RT, IT) RT
#  endif
#  ifndef BOOST_MATH_INT_VALUE_SUFFIX
#    define BOOST_MATH_INT_VALUE_SUFFIX(RV, SUF) RV##.0L
#  endif
#endif

#elif defined(__INTEL_COMPILER)
#  ifndef BOOST_MATH_POLY_METHOD
#    define BOOST_MATH_POLY_METHOD 2
#  endif
#  ifndef BOOST_MATH_RATIONAL_METHOD
#    define BOOST_MATH_RATIONAL_METHOD 1
#  endif

#elif defined(__GNUC__)
#  ifndef BOOST_MATH_POLY_METHOD
#    define BOOST_MATH_POLY_METHOD 3
#  endif
#  ifndef BOOST_MATH_RATIONAL_METHOD
#    define BOOST_MATH_RATIONAL_METHOD 3
#  endif

#elif defined(__clang__)

#if __clang__ > 6
#  ifndef BOOST_MATH_POLY_METHOD
#    define BOOST_MATH_POLY_METHOD 3
#  endif
#  ifndef BOOST_MATH_RATIONAL_METHOD
#    define BOOST_MATH_RATIONAL_METHOD 3
#  endif
#  ifndef BOOST_MATH_INT_TABLE_TYPE
#    define BOOST_MATH_INT_TABLE_TYPE(RT, IT) RT
#  endif
#  ifndef BOOST_MATH_INT_VALUE_SUFFIX
#    define BOOST_MATH_INT_VALUE_SUFFIX(RV, SUF) RV##.0L
#  endif
#endif

#endif

//
// noexcept support:
//
#include <type_traits>
#define BOOST_MATH_NOEXCEPT(T) noexcept(std::is_floating_point<T>::value)
#define BOOST_MATH_IS_FLOAT(T) (std::is_floating_point<T>::value)

//
// The maximum order of polynomial that will be evaluated 
// via an unrolled specialisation:
//
#ifndef BOOST_MATH_MAX_POLY_ORDER
#  define BOOST_MATH_MAX_POLY_ORDER 20
#endif 
//
// Set the method used to evaluate polynomials and rationals:
//
#ifndef BOOST_MATH_POLY_METHOD
#  define BOOST_MATH_POLY_METHOD 2
#endif 
#ifndef BOOST_MATH_RATIONAL_METHOD
#  define BOOST_MATH_RATIONAL_METHOD 1
#endif 
//
// decide whether to store constants as integers or reals:
//
#ifndef BOOST_MATH_INT_TABLE_TYPE
#  define BOOST_MATH_INT_TABLE_TYPE(RT, IT) IT
#endif
#ifndef BOOST_MATH_INT_VALUE_SUFFIX
#  define BOOST_MATH_INT_VALUE_SUFFIX(RV, SUF) RV##SUF
#endif
//
// And then the actual configuration:
//
#if defined(BOOST_MATH_STANDALONE) && defined(_GLIBCXX_USE_FLOAT128) && defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__) && !defined(__STRICT_ANSI__) \
   && !defined(BOOST_MATH_DISABLE_FLOAT128) && !defined(BOOST_MATH_USE_FLOAT128)
#  define BOOST_MATH_USE_FLOAT128
#elif defined(BOOST_HAS_FLOAT128) && !defined(BOOST_MATH_USE_FLOAT128) && !defined(BOOST_MATH_DISABLE_FLOAT128)
#  define BOOST_MATH_USE_FLOAT128
#endif
#ifdef BOOST_MATH_USE_FLOAT128
//
// Only enable this when the compiler really is GCC as clang and probably 
// intel too don't support __float128 yet :-(
//
#  if defined(__INTEL_COMPILER) && defined(__GNUC__)
#    if (__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 6))
#      define BOOST_MATH_FLOAT128_TYPE __float128
#    endif
#  elif defined(__GNUC__)
#      define BOOST_MATH_FLOAT128_TYPE __float128
#  endif

#  ifndef BOOST_MATH_FLOAT128_TYPE
#      define BOOST_MATH_FLOAT128_TYPE _Quad
#  endif
#endif
//
// Check for WinCE with no iostream support:
//
#if defined(_WIN32_WCE) && !defined(__SGI_STL_PORT)
#  define BOOST_MATH_NO_LEXICAL_CAST
#endif

//
// Helper macro for controlling the FP behaviour:
//
#ifndef BOOST_MATH_CONTROL_FP
#  define BOOST_MATH_CONTROL_FP
#endif
//
// Helper macro for using statements:
//
#define BOOST_MATH_STD_USING_CORE \
   using std::abs;\
   using std::acos;\
   using std::cos;\
   using std::fmod;\
   using std::modf;\
   using std::tan;\
   using std::asin;\
   using std::cosh;\
   using std::frexp;\
   using std::pow;\
   using std::tanh;\
   using std::atan;\
   using std::exp;\
   using std::ldexp;\
   using std::sin;\
   using std::atan2;\
   using std::fabs;\
   using std::log;\
   using std::sinh;\
   using std::ceil;\
   using std::floor;\
   using std::log10;\
   using std::sqrt;\
   using std::log2;\
   using std::ilogb;

#define BOOST_MATH_STD_USING BOOST_MATH_STD_USING_CORE

namespace boost{ namespace math{
namespace tools
{

template <class T>
inline T max BOOST_MATH_PREVENT_MACRO_SUBSTITUTION(T a, T b, T c) BOOST_MATH_NOEXCEPT(T)
{
   return (std::max)((std::max)(a, b), c);
}

template <class T>
inline T max BOOST_MATH_PREVENT_MACRO_SUBSTITUTION(T a, T b, T c, T d) BOOST_MATH_NOEXCEPT(T)
{
   return (std::max)((std::max)(a, b), (std::max)(c, d));
}

} // namespace tools

template <class T>
void suppress_unused_variable_warning(const T&) BOOST_MATH_NOEXCEPT(T)
{
}

namespace detail{

template <class T>
struct is_integer_for_rounding
{
   static constexpr bool value = std::is_integral<T>::value || (std::numeric_limits<T>::is_specialized && std::numeric_limits<T>::is_integer);
};

}

}} // namespace boost namespace math

#ifdef __GLIBC_PREREQ
#  if __GLIBC_PREREQ(2,14)
#     define BOOST_MATH_HAVE_FIXED_GLIBC
#  endif
#endif

#if ((defined(__linux__) && !defined(__UCLIBC__) && !defined(BOOST_MATH_HAVE_FIXED_GLIBC)) || defined(__QNX__) || defined(__IBMCPP__))
//
// This code was introduced in response to this glibc bug: http://sourceware.org/bugzilla/show_bug.cgi?id=2445
// Basically powl and expl can return garbage when the result is small and certain exception flags are set
// on entrance to these functions.  This appears to have been fixed in Glibc 2.14 (May 2011).
// Much more information in this message thread: https://groups.google.com/forum/#!topic/boost-list/ZT99wtIFlb4
//

#include <cfenv>

#  ifdef FE_ALL_EXCEPT

namespace boost{ namespace math{
   namespace detail
   {
   struct fpu_guard
   {
      fpu_guard()
      {
         fegetexceptflag(&m_flags, FE_ALL_EXCEPT);
         feclearexcept(FE_ALL_EXCEPT);
      }
      ~fpu_guard()
      {
         fesetexceptflag(&m_flags, FE_ALL_EXCEPT);
      }
   private:
      fexcept_t m_flags;
   };

   } // namespace detail
   }} // namespaces

#    define BOOST_FPU_EXCEPTION_GUARD boost::math::detail::fpu_guard local_guard_object;
#    define BOOST_MATH_INSTRUMENT_FPU do{ fexcept_t cpu_flags; fegetexceptflag(&cpu_flags, FE_ALL_EXCEPT); BOOST_MATH_INSTRUMENT_VARIABLE(cpu_flags); } while(0); 

#  else

#    define BOOST_FPU_EXCEPTION_GUARD
#    define BOOST_MATH_INSTRUMENT_FPU

#  endif

#else // All other platforms.
#  define BOOST_FPU_EXCEPTION_GUARD
#  define BOOST_MATH_INSTRUMENT_FPU
#endif

#ifdef BOOST_MATH_INSTRUMENT

#  include <iostream>
#  include <iomanip>
#  include <typeinfo>

#  define BOOST_MATH_INSTRUMENT_CODE(x) \
      std::cout << std::setprecision(35) << __FILE__ << ":" << __LINE__ << " " << x << std::endl;
#  define BOOST_MATH_INSTRUMENT_VARIABLE(name) BOOST_MATH_INSTRUMENT_CODE(#name << " = " << name)

#else

#  define BOOST_MATH_INSTRUMENT_CODE(x)
#  define BOOST_MATH_INSTRUMENT_VARIABLE(name)

#endif

//
// Thread local storage:
//
#ifndef BOOST_MATH_DISABLE_THREADS
#  define BOOST_MATH_THREAD_LOCAL thread_local
#else
#  define BOOST_MATH_THREAD_LOCAL 
#endif

//
// Some mingw flavours have issues with thread_local and types with non-trivial destructors
// See https://sourceforge.net/p/mingw-w64/bugs/527/
//
#if (defined(__MINGW32__) && (__GNUC__ < 9) && !defined(__clang__))
#  define BOOST_MATH_NO_THREAD_LOCAL_WITH_NON_TRIVIAL_TYPES
#endif


//
// Can we have constexpr tables?
//
#if (!defined(BOOST_MATH_NO_CXX14_CONSTEXPR)) || (defined(_MSC_VER) && _MSC_VER >= 1910)
#define BOOST_MATH_HAVE_CONSTEXPR_TABLES
#define BOOST_MATH_CONSTEXPR_TABLE_FUNCTION constexpr
#else
#define BOOST_MATH_CONSTEXPR_TABLE_FUNCTION
#endif

//
// CUDA support:
//

#ifdef __CUDACC__

// We have to get our include order correct otherwise you get compilation failures
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/cstdint>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/complex>

#  define BOOST_MATH_CUDA_ENABLED __host__ __device__
#  define BOOST_MATH_HAS_GPU_SUPPORT

#  ifndef BOOST_MATH_ENABLE_CUDA
#    define BOOST_MATH_ENABLE_CUDA
#  endif

// Device code can not handle exceptions
#  ifndef BOOST_MATH_NO_EXCEPTIONS
#    define BOOST_MATH_NO_EXCEPTIONS
#  endif

// We want to use force inline from CUDA instead of the host compiler
#  undef BOOST_MATH_FORCEINLINE
#  define BOOST_MATH_FORCEINLINE __forceinline__

#elif defined(SYCL_LANGUAGE_VERSION)

#  define BOOST_MATH_SYCL_ENABLED SYCL_EXTERNAL
#  define BOOST_MATH_HAS_GPU_SUPPORT

#  ifndef BOOST_MATH_ENABLE_SYCL
#    define BOOST_MATH_ENABLE_SYCL
#  endif

#  ifndef BOOST_MATH_NO_EXCEPTIONS
#    define BOOST_MATH_NO_EXCEPTIONS
#  endif

// spir64 does not support long double
#  define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#  define BOOST_MATH_NO_REAL_CONCEPT_TESTS

#  undef BOOST_MATH_FORCEINLINE
#  define BOOST_MATH_FORCEINLINE inline

#endif

#ifndef BOOST_MATH_CUDA_ENABLED
#  define BOOST_MATH_CUDA_ENABLED
#endif

#ifndef BOOST_MATH_SYCL_ENABLED
#  define BOOST_MATH_SYCL_ENABLED
#endif

// Not all functions that allow CUDA allow SYCL (e.g. Recursion is disallowed by SYCL)
#  define BOOST_MATH_GPU_ENABLED BOOST_MATH_CUDA_ENABLED BOOST_MATH_SYCL_ENABLED

// Additional functions that need replaced/marked up
#ifdef BOOST_MATH_HAS_GPU_SUPPORT
template <class T>
BOOST_MATH_GPU_ENABLED constexpr void gpu_safe_swap(T& a, T& b) { T t(a); a = b; b = t; }
template <class T>
BOOST_MATH_GPU_ENABLED constexpr T gpu_safe_min(const T& a, const T& b) { return a < b ? a : b; }
template <class T>
BOOST_MATH_GPU_ENABLED constexpr T gpu_safe_max(const T& a, const T& b) { return a > b ? a : b; }

#define BOOST_MATH_GPU_SAFE_SWAP(a, b) gpu_safe_swap(a, b)
#define BOOST_MATH_GPU_SAFE_MIN(a, b) gpu_safe_min(a, b)
#define BOOST_MATH_GPU_SAFE_MAX(a, b) gpu_safe_max(a, b)

#else

#define BOOST_MATH_GPU_SAFE_SWAP(a, b) std::swap(a, b)
#define BOOST_MATH_GPU_SAFE_MIN(a, b) (std::min)(a, b)
#define BOOST_MATH_GPU_SAFE_MAX(a, b) (std::max)(a, b)

#endif

// Static variables are not allowed with CUDA or C++20 modules
// See if we can inline them instead

#if defined(__cpp_inline_variables) && __cpp_inline_variables >= 201606L
#  define BOOST_MATH_INLINE_CONSTEXPR inline constexpr
#  define BOOST_MATH_STATIC static
#  ifndef BOOST_MATH_HAS_GPU_SUPPORT
#    define BOOST_MATH_STATIC_LOCAL_VARIABLE static
#  else
#    define BOOST_MATH_STATIC_LOCAL_VARIABLE
#  endif
#else
#  ifndef BOOST_MATH_HAS_GPU_SUPPORT
#    define BOOST_MATH_INLINE_CONSTEXPR static constexpr
#    define BOOST_MATH_STATIC static
#    define BOOST_MATH_STATIC_LOCAL_VARIABLE
#  else
#    define BOOST_MATH_INLINE_CONSTEXPR constexpr
#    define BOOST_MATH_STATIC constexpr
#    define BOOST_MATH_STATIC_LOCAL_VARIABLE static
#  endif
#endif

#define BOOST_MATH_FP_NAN FP_NAN
#define BOOST_MATH_FP_INFINITE FP_INFINITE
#define BOOST_MATH_FP_ZERO FP_ZERO
#define BOOST_MATH_FP_SUBNORMAL FP_SUBNORMAL
#define BOOST_MATH_FP_NORMAL FP_NORMAL

#else // Special section for CUDA NVRTC to ensure we consume no STL headers

#ifndef BOOST_MATH_STANDALONE
#  define BOOST_MATH_STANDALONE
#endif

#define BOOST_MATH_HAS_NVRTC
#define BOOST_MATH_ENABLE_CUDA
#define BOOST_MATH_HAS_GPU_SUPPORT

#define BOOST_MATH_GPU_ENABLED __host__ __device__
#define BOOST_MATH_CUDA_ENABLED __host__ __device__

#define BOOST_MATH_STATIC static
#define BOOST_MATH_STATIC_LOCAL_VARIABLE

#define BOOST_MATH_NOEXCEPT(T) noexcept(boost::math::is_floating_point_v<T>)
#define BOOST_MATH_EXPLICIT_TEMPLATE_TYPE(T) 
#define BOOST_MATH_EXPLICIT_TEMPLATE_TYPE_SPEC(T) 
#define BOOST_MATH_APPEND_EXPLICIT_TEMPLATE_TYPE_SPEC(T) 
#define BOOST_MATH_BIG_CONSTANT(T, N, V) static_cast<T>(V)
#define BOOST_MATH_FORCEINLINE __forceinline__
#define BOOST_MATH_STD_USING  
#define BOOST_MATH_IF_CONSTEXPR if
#define BOOST_MATH_IS_FLOAT(T) (boost::math::is_floating_point<T>::value)
#define BOOST_MATH_CONSTEXPR_TABLE_FUNCTION constexpr
#define BOOST_MATH_NO_EXCEPTIONS
#define BOOST_MATH_PREVENT_MACRO_SUBSTITUTION 

// This should be defined to nothing but since it is not specifically a math macro
// we need to undef before proceeding
#ifdef BOOST_FPU_EXCEPTION_GUARD
#  undef BOOST_FPU_EXCEPTION_GUARD
#endif

#define BOOST_FPU_EXCEPTION_GUARD

template <class T>
BOOST_MATH_GPU_ENABLED constexpr void gpu_safe_swap(T& a, T& b) { T t(a); a = b; b = t; }

#define BOOST_MATH_GPU_SAFE_SWAP(a, b) gpu_safe_swap(a, b)
#define BOOST_MATH_GPU_SAFE_MIN(a, b) (::min)(a, b)
#define BOOST_MATH_GPU_SAFE_MAX(a, b) (::max)(a, b)

#define BOOST_MATH_FP_NAN 0
#define BOOST_MATH_FP_INFINITE 1
#define BOOST_MATH_FP_ZERO 2
#define BOOST_MATH_FP_SUBNORMAL 3
#define BOOST_MATH_FP_NORMAL 4

#define BOOST_MATH_INT_VALUE_SUFFIX(RV, SUF) RV##SUF
#define BOOST_MATH_INT_TABLE_TYPE(RT, IT) IT

#if defined(__cpp_inline_variables) && __cpp_inline_variables >= 201606L
#  define BOOST_MATH_INLINE_CONSTEXPR inline constexpr
#else
#  define BOOST_MATH_INLINE_CONSTEXPR constexpr
#endif

#define BOOST_MATH_INSTRUMENT_VARIABLE(x)
#define BOOST_MATH_INSTRUMENT_CODE(x) 

#endif // NVRTC

#endif // BOOST_MATH_TOOLS_CONFIG_HPP




