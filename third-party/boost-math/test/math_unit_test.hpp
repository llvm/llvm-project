// Copyright Nick Thompson, 2019
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TEST_TEST_HPP
#define BOOST_MATH_TEST_TEST_HPP
#include <atomic>
#include <iostream>
#include <iomanip>
#include <cmath> // for std::isnan
#include <string>
#include <type_traits>
#include <boost/math/tools/assert.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/special_functions/trunc.hpp>
#if defined __has_include
#  if __has_include(<cxxabi.h>)
#define BOOST_MATH_HAS_CXX_ABI 1
#    include <cxxabi.h>
#  endif
#endif
namespace boost { namespace math { namespace  test {

namespace detail {
    static std::atomic<int64_t> global_error_count{0};
    static std::atomic<int64_t> total_ulp_distance{0};

    inline std::string demangle(char const * name)
    {
        int status = 0;
        std::size_t size = 0;
#if BOOST_MATH_HAS_CXX_ABI
        std::string s {abi::__cxa_demangle( name, NULL, &size, &status )};
#else
        std::string s {name};
#endif
        return s;
    }
}

template<class Real>
bool check_mollified_close(Real expected, Real computed, Real tol, std::string const & filename, std::string const & function, int line)
{
    using std::isnan;
    BOOST_MATH_ASSERT_MSG(!isnan(tol), "Tolerance cannot be a nan.");
    BOOST_MATH_ASSERT_MSG(!isnan(expected), "Expected value cannot be a nan.");
    BOOST_MATH_ASSERT_MSG(tol >= 0, "Tolerance must be non-negative.");
    if (isnan(computed)) {
        std::ios_base::fmtflags f( std::cerr.flags() );
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                  << " \033[0m Computed value is a nan\n";
        std::cerr.flags(f);
        ++detail::global_error_count;
        return false;
    }
    using std::max;
    using std::abs;
    Real denom = (max)(abs(expected), Real(1));
    Real mollified_relative_error = abs(expected - computed)/denom;
    if (mollified_relative_error > tol)
    {
        Real dist = abs(boost::math::float_distance(expected, computed));
        detail::total_ulp_distance += static_cast<int64_t>(dist);
        std::ios_base::fmtflags f( std::cerr.flags() );
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                  << " \033[0m Mollified relative error in " << detail::demangle(typeid(Real).name())<< " precision is " << mollified_relative_error
                  << ", which exceeds " << tol << ", error/tol  = " << mollified_relative_error/tol << ".\n"
                  << std::setprecision(std::numeric_limits<Real>::max_digits10) << std::showpos
                  << "  Expected: " << std::defaultfloat << std::fixed << expected << std::hexfloat << " = " << expected << "\n"
                  << "  Computed: " << std::defaultfloat << std::fixed << computed << std::hexfloat << " = " << computed << "\n"
                  << std::defaultfloat
                  << "  ULP distance: " << dist << "\n";
        std::cerr.flags(f);
        ++detail::global_error_count;

        return false;
    }
    return true;
}

template<class PreciseReal, class Real>
bool check_ulp_close(PreciseReal expected1, Real computed, size_t ulps, std::string const & filename, std::string const & function, int line)
{
    using std::max;
    using std::abs;
    using std::isnan;
    using boost::math::lltrunc;
    // Of course integers can be expected values, and they are exact:
    if (!std::is_integral<PreciseReal>::value) {
    if (boost::math::isnan(expected1)) {
        std::ostringstream oss;
        oss << "Error in CHECK_ULP_CLOSE: Expected value cannot be a nan. Callsite: " << filename << ":" << function << ":" << line << "."; 
        throw std::domain_error(oss.str());
    }
        if (sizeof(PreciseReal) < sizeof(Real)) {
            std::ostringstream err;
            err << "\n\tThe expected number must be computed in higher (or equal) precision than the number being tested.\n";
            err << "\tType of expected is " << detail::demangle(typeid(PreciseReal).name()) << ", which occupies " << sizeof(PreciseReal) << " bytes.\n";
            err << "\tType of computed is " << detail::demangle(typeid(Real).name()) << ", which occupies " << sizeof(Real) << " bytes.\n";
            throw std::logic_error(err.str());
        }
    }

    if (boost::math::isnan(computed))
    {
        std::ios_base::fmtflags f( std::cerr.flags() );
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                  << " \033[0m Computed value is a nan\n";
        std::cerr.flags(f);
        ++detail::global_error_count;
        return false;
    }

    Real expected = Real(expected1);
    Real dist = abs(boost::math::float_distance(expected, computed));
    if (dist > ulps)
    {
        detail::total_ulp_distance += static_cast<int64_t>(lltrunc(dist));
        Real abs_expected = abs(expected);
        Real denom = (max)(abs_expected, Real(1));
        Real mollified_relative_error = abs(expected - computed)/denom;
        std::ios_base::fmtflags f( std::cerr.flags() );
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                  << " \033[0m ULP distance in " << detail::demangle(typeid(Real).name())<< " precision is " << dist
                  << ", which exceeds " << ulps;
                  if (ulps > 0)
                  {
                      std::cerr << ", error/ulps  = " << dist/static_cast<Real>(ulps) << ".\n";
                  }
                  else
                  {
                      std::cerr << ".\n";
                  }
        std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10) << std::showpos
                  << "  Expected: " << std::defaultfloat << std::fixed << expected << std::hexfloat << " = " << expected << "\n"
                  << "  Computed: " << std::defaultfloat << std::fixed << computed << std::hexfloat << " = " << computed << "\n"
                  << std::defaultfloat
                  << "  Mollified relative error: " << mollified_relative_error << "\n";
        std::cerr.flags(f);
        ++detail::global_error_count;
        return false;
    }
    return true;
}

template<typename Real>
bool check_le(Real lesser, Real greater, std::string const & filename, std::string const & function, int line)
{
    using std::max;
    using std::abs;
    using std::isnan;

    if (std::is_floating_point<Real>::value) {
        if (boost::math::isnan(lesser))
        {
            std::ios_base::fmtflags f( std::cerr.flags() );
            std::cerr << std::setprecision(3);
            std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                    << " \033[0m Lesser value is a nan\n";
            std::cerr.flags(f);
            ++detail::global_error_count;
            return false;
        }

        if (boost::math::isnan(greater))
        {
            std::ios_base::fmtflags f( std::cerr.flags() );
            std::cerr << std::setprecision(3);
            std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                    << " \033[0m Greater value is a nan\n";
            std::cerr.flags(f);
            ++detail::global_error_count;
            return false;
        }
    }

    if (lesser > greater)
    {
        std::ios_base::fmtflags f( std::cerr.flags() );
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                  << " \033[0m Condition " << lesser << " \u2264 " << greater << " is violated in " << detail::demangle(typeid(Real).name()) << " precision.\n";
        std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10) << std::showpos
                  << "  \"Lesser\" : " << std::defaultfloat << std::fixed << lesser  << " = " << std::scientific << lesser  << std::hexfloat << " = " << lesser << "\n"
                  << "  \"Greater\": " << std::defaultfloat << std::fixed << greater << " = " << std::scientific << greater << std::hexfloat << " = " << greater << "\n"
                  << std::defaultfloat;
        std::cerr.flags(f);
        ++detail::global_error_count;
        return false;
    }
    return true;
}


template<class PreciseReal, class Real>
bool check_conditioned_error(Real abscissa, PreciseReal expected1, PreciseReal expected_derivative, Real computed, Real acceptable_badness, std::string const & filename, std::string const & function, int line)
{
    using std::max;
    using std::abs;
    using std::isnan;
    // Of course integers can be expected values, and they are exact:
    if (!std::is_integral<PreciseReal>::value) {
        BOOST_MATH_ASSERT_MSG(sizeof(PreciseReal) >= sizeof(Real),
                         "The expected number must be computed in higher (or equal) precision than the number being tested.");
        BOOST_MATH_ASSERT_MSG(!isnan(abscissa), "Expected abscissa cannot be a nan.");
        BOOST_MATH_ASSERT_MSG(!isnan(expected1), "Expected value cannot be a nan.");
        BOOST_MATH_ASSERT_MSG(!isnan(expected_derivative), "Expected derivative cannot be a nan.");
    }
    BOOST_MATH_ASSERT_MSG(acceptable_badness >= 1, "Acceptable badness scale must be >= 1, and in general should = 1 exactly.");

    if (isnan(computed))
    {
        std::ios_base::fmtflags f(std::cerr.flags());
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                  << " \033[0m Computed value is a nan\n";
        std::cerr.flags(f);
        ++detail::global_error_count;
        return false;
    }

    Real mu = std::numeric_limits<Real>::epsilon()/2;
    Real expected = Real(expected1);
    // Relative error is undefined. Therefore we must use |f(x(1+eps))| le mu|xf'(x)|.
    if (expected == 0)
    {
        Real tol = acceptable_badness*mu*abs(abscissa*expected_derivative);
        if (abs(computed) > tol)
        {
            std::ios_base::fmtflags f( std::cerr.flags() );
            std::cerr << std::setprecision(3);
            std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n";
            std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10) << std::showpos;
            std::cerr << "\033[0m  Error at abscissa " << std::defaultfloat << std::fixed << abscissa << " = " << std::hexfloat << abscissa << "\n";
            std::cerr << "  Given that the expected value is zero, the computed value in " << detail::demangle(typeid(Real).name()) << " precision  must satisfy |f(x)| <= " << tol << ".\n";
            std::cerr << "  But the computed value is " << std::defaultfloat << std::fixed << computed << std::hexfloat << " = " << computed << "\n";
            std::cerr.flags(f);
            ++detail::global_error_count;
            return false;            
        }
    }
    // 1 ULP accuracy * acceptable_badness is always acceptable, independent of condition number:
    if (abs(boost::math::float_distance(Real(expected), computed)) <= acceptable_badness)
    {
        return true;
    }
    Real expected_prime = Real(expected_derivative);
    PreciseReal precise_abscissa = abscissa;
    PreciseReal cond = abs(precise_abscissa*expected_prime/expected);
    PreciseReal relative_error = abs((expected - PreciseReal(computed))/expected);
    // If the condition number is small, then we revert to allowing 1ULP accuracy, i.e., one incorrect bit.
    Real tol = cond*mu;
    tol *= acceptable_badness;
    if (relative_error > tol)
    {
        std::ios_base::fmtflags f( std::cerr.flags() );
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << "\n";
        std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10);
        std::cerr << "\033[0m  The relative error at abscissa x = " << std::defaultfloat << std::fixed << abscissa << " = " << std::hexfloat << abscissa
                  << " in " << detail::demangle(typeid(Real).name()) << " precision is " << std::scientific << relative_error << "\n"
                  << "  This exceeds the tolerance " << tol << "\n"
                  << std::showpos
                  << "  Expected: " << std::defaultfloat << std::fixed << expected << " = " << std::scientific << expected << std::hexfloat << " = " << expected << "\n"
                  << "  Computed: " << std::defaultfloat << std::fixed << computed << " = " << std::scientific << computed << std::hexfloat << " = " << computed << "\n"
                  << "  Condition number of function evaluation: " << std::noshowpos << std::defaultfloat << std::scientific << cond  << " = " << std::fixed << cond << "\n"
                  << "  Badness scale required to make this message go away: " << std::defaultfloat << relative_error/(cond*mu) << "\n";
        std::cerr.flags(f);
        ++detail::global_error_count;
        return false;
    }
    return true;
}


template<class PreciseReal, class Real>
bool check_absolute_error(PreciseReal expected1, Real computed, Real acceptable_error, std::string const & filename, std::string const & function, int line)
{
    using std::max;
    using std::abs;
    using std::isnan;
    // Of course integers can be expected values, and they are exact:
    if (!std::is_integral<PreciseReal>::value) {
        BOOST_MATH_ASSERT_MSG(sizeof(PreciseReal) >= sizeof(Real),
                         "The expected number must be computed in higher (or equal) precision than the number being tested.");
        BOOST_MATH_ASSERT_MSG(!isnan(expected1), "Expected value cannot be a nan (use CHECK_NAN if this is your intention).");
    }
    BOOST_MATH_ASSERT_MSG(acceptable_error > 0, "Error must be > 0.");

    if (isnan(computed))
    {
        std::ios_base::fmtflags f(std::cerr.flags());
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << ":\n"
                  << " \033[0m Computed value is a nan\n";
        std::cerr.flags(f);
        ++detail::global_error_count;
        return false;
    }

    Real expected = Real(expected1);
    Real error = abs(expected - computed);
    if (error > acceptable_error)
    {
        std::ios_base::fmtflags f( std::cerr.flags() );
        std::cerr << std::setprecision(3);
        std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << "\n";
        std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10);
        std::cerr << "\033[0m  The absolute error in " << detail::demangle(typeid(Real).name()) << " precision is " << std::scientific << error << "\n"
                  << "  This exceeds the acceptable error " << acceptable_error << "\n"
                  << std::showpos
                  << "  Expected: " << std::defaultfloat << std::fixed << expected << " = " << std::scientific << expected << std::hexfloat << " = " << expected << "\n"
                  << "  Computed: " << std::defaultfloat << std::fixed << computed << " = " << std::scientific << computed<< std::hexfloat << " = " << computed << "\n"
                  << "  Error/Acceptable error: " << std::defaultfloat << error/acceptable_error << "\n";
        std::cerr.flags(f);
        ++detail::global_error_count;
        return false;
    }
    return true;
}

template<class Real>
bool check_nan(Real x, std::string const & filename, std::string const & function, int line)
{
    using std::isnan;
    if (!isnan(x)) {
      std::ios_base::fmtflags f( std::cerr.flags() );
      std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << "\n";
      std::cerr << "\033[0m  The computed value should be a nan, but is instead " << std::defaultfloat << std::fixed << x << " = " << std::scientific << x << std::hexfloat << " = " << x << "\n";
      std::cerr.flags(f);
      ++detail::global_error_count;
      return false;
    }
    return true;
}

template<class Real>
bool check_equal(Real x, Real y, std::string const & filename, std::string const & function, int line)
{
  using std::isnan;
  if (x != y) {
    std::ios_base::fmtflags f( std::cerr.flags() );
    std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << "\n";
    std::cerr << "\033[0m  Condition '" << x << " == " << y << "' is not satisfied:\n";
    if (std::is_floating_point<Real>::value) {
      std::cerr << "  Expected =  " << std::defaultfloat << std::fixed << x << " = " << std::scientific << x << std::hexfloat << " = " << x << "\n";
      std::cerr << "  Computed =  " << std::defaultfloat << std::fixed << y << " = " << std::scientific << y << std::hexfloat << " = " << y << "\n";
    } else {
      std::cerr << "  Expected: " << x << " = " << "0x" << std::hex << x << "\n";
      std::cerr << std::dec;
      std::cerr << "  Computed: " << y << " = " << "0x" << std::hex << y << "\n";
    }
    std::cerr.flags(f);
    ++detail::global_error_count;
    return false;
  }
  return true;
}


bool check_true(bool condition, std::string const & filename, std::string const & function, int line)
{
  if (!condition) {
    std::ios_base::fmtflags f( std::cerr.flags() );
    std::cerr << "\033[0;31mError at " << filename << ":" << function << ":" << line << "\n";
    std::cerr << "\033[0m  Boolean condition is not satisfied:\n";
    std::cerr.flags(f);
    ++detail::global_error_count;
    return false;
  }
  return true;
}

void report_non_throw(const std::string& file, int line)
{
   std::cerr << "Expected exception not thrown in test at: " << file << ":" << line << std::endl;
   ++detail::global_error_count;
}

void report_incorrect_throw(const std::string& file, int line)
{
   std::cerr << "Exception of the wrong type thrown in test at: " << file << ":" << line << std::endl;
   ++detail::global_error_count;
}

int report_errors()
{
    if (detail::global_error_count > 0)
    {
        std::cerr << "\033[0;31mError count: " << detail::global_error_count;
        if (detail::total_ulp_distance > 0) {
            std::cerr << ", total ulp distance = " << detail::total_ulp_distance << "\n\033[0m";
        }
        else {
            // else we overflowed the ULPs counter and all we could print is a bizarre negative number.
            std::cerr << "\n\033[0m";
        }

        detail::global_error_count = 0;
        detail::total_ulp_distance = 0;
        return 1;
    }
    std::cout << "\x1B[32mNo errors detected.\n\033[0m";
    return 0;
}

}}}

#define CHECK_MOLLIFIED_CLOSE(X, Y, Z) boost::math::test::check_mollified_close< typename std::remove_reference<decltype((Y))>::type>((X), (Y), (Z), __FILE__, __func__, __LINE__)

#define CHECK_ULP_CLOSE(X, Y, Z) boost::math::test::check_ulp_close((X), (Y), (Z), __FILE__, __func__, __LINE__)

#define CHECK_GE(X, Y) boost::math::test::check_le((Y), (X), __FILE__, __func__, __LINE__)

#define CHECK_LE(X, Y) boost::math::test::check_le((X), (Y), __FILE__, __func__, __LINE__)

#define CHECK_NAN(X) boost::math::test::check_nan((X), __FILE__, __func__, __LINE__)

#define CHECK_EQUAL(X, Y) boost::math::test::check_equal((X), (Y), __FILE__, __func__, __LINE__)

#define CHECK_CONDITIONED_ERROR(V, W, X, Y, Z) boost::math::test::check_conditioned_error((V), (W), (X), (Y), (Z), __FILE__, __func__, __LINE__)

#define CHECK_ABSOLUTE_ERROR(X, Y, Z) boost::math::test::check_absolute_error((X), (Y), (Z), __FILE__, __func__, __LINE__)

#define CHECK_TRUE(X) boost::math::test::check_true((X), __FILE__, __func__, __LINE__)

#define CHECK_THROW(x, what) try{ x; boost::math::test::report_non_throw(__FILE__, __LINE__); }catch(const what&){} catch(...){ boost::math::test::report_incorrect_throw(__FILE__, __LINE__); }

#endif
