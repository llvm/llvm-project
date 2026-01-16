// Copyright Nick Thompson, 2017
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

/*
 * This class performs exp-sinh quadrature on half infinite intervals.
 *
 * References:
 *
 * 1) Tanaka, Ken'ichiro, et al. "Function classes for double exponential integration formulas." Numerische Mathematik 111.4 (2009): 631-655.
 */

#ifndef BOOST_MATH_QUADRATURE_EXP_SINH_HPP
#define BOOST_MATH_QUADRATURE_EXP_SINH_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/quadrature/detail/exp_sinh_detail.hpp>

#ifndef BOOST_MATH_HAS_NVRTC

#include <cmath>
#include <limits>
#include <memory>
#include <string>

namespace boost{ namespace math{ namespace quadrature {

template<class Real, class Policy = policies::policy<> >
class exp_sinh
{
public:
   exp_sinh(size_t max_refinements = 9)
      : m_imp(std::make_shared<detail::exp_sinh_detail<Real, Policy>>(max_refinements)) {}

    template<class F>
    auto integrate(const F& f, Real a, Real b, Real tol = boost::math::tools::root_epsilon<Real>(), Real* error = nullptr, Real* L1 = nullptr, std::size_t* levels = nullptr) const ->decltype(std::declval<F>()(std::declval<Real>()));
    template<class F>
    auto integrate(const F& f, Real tol = boost::math::tools::root_epsilon<Real>(), Real* error = nullptr, Real* L1 = nullptr, std::size_t* levels = nullptr) const ->decltype(std::declval<F>()(std::declval<Real>()));

private:
    std::shared_ptr<detail::exp_sinh_detail<Real, Policy>> m_imp;
};

template<class Real, class Policy>
template<class F>
auto exp_sinh<Real, Policy>::integrate(const F& f, Real a, Real b, Real tolerance, Real* error, Real* L1, std::size_t* levels) const ->decltype(std::declval<F>()(std::declval<Real>()))
{
    typedef decltype(f(a)) K;
    static_assert(!std::is_integral<K>::value,
                  "The return type cannot be integral, it must be either a real or complex floating point type.");
    using std::abs;
    using boost::math::constants::half;
    using boost::math::quadrature::detail::exp_sinh_detail;

    static const char* function = "boost::math::quadrature::exp_sinh<%1%>::integrate";

    // Neither limit may be a NaN:
    if((boost::math::isnan)(a) || (boost::math::isnan)(b))
    {
       return static_cast<K>(policies::raise_domain_error(function, "NaN supplied as one limit of integration - sorry I don't know what to do", a, Policy()));
     }
    // Right limit is infinite:
    if ((boost::math::isfinite)(a) && (b >= boost::math::tools::max_value<Real>()))
    {
        // If a = 0, don't use an additional level of indirection:
        if (a == static_cast<Real>(0))
        {
            return m_imp->integrate(f, error, L1, function, tolerance, levels);
        }
        const auto u = [&](Real t)->K { return f(t + a); };
        return m_imp->integrate(u, error, L1, function, tolerance, levels);
    }

    if ((boost::math::isfinite)(b) && a <= -boost::math::tools::max_value<Real>())
    {
        const auto u = [&](Real t)->K { return f(b-t);};
        return m_imp->integrate(u, error, L1, function, tolerance, levels);
    }

    // Infinite limits:
    if ((a <= -boost::math::tools::max_value<Real>()) && (b >= boost::math::tools::max_value<Real>()))
    {
        return static_cast<K>(policies::raise_domain_error(function, "Use sinh_sinh quadrature for integration over the whole real line; exp_sinh is for half infinite integrals.", a, Policy()));
    }
    // If we get to here then both ends must necessarily be finite:
    return static_cast<K>(policies::raise_domain_error(function, "Use tanh_sinh quadrature for integration over finite domains; exp_sinh is for half infinite integrals.", a, Policy()));
}

template<class Real, class Policy>
template<class F>
auto exp_sinh<Real, Policy>::integrate(const F& f, Real tolerance, Real* error, Real* L1, std::size_t* levels) const ->decltype(std::declval<F>()(std::declval<Real>()))
{
    static const char* function = "boost::math::quadrature::exp_sinh<%1%>::integrate";
    using std::abs;
    if (abs(tolerance) > 1) {
        return policies::raise_domain_error(function, "The tolerance provided (%1%) is unusually large; did you confuse it with a domain bound?", tolerance, Policy());
    }
    return m_imp->integrate(f, error, L1, function, tolerance, levels);
}


}}}

#endif // BOOST_MATH_HAS_NVRTC

#ifdef BOOST_MATH_ENABLE_CUDA

#include <boost/math/tools/type_traits.hpp>
#include <boost/math/tools/cstdint.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/constants/constants.hpp>

namespace boost { 
namespace math { 
namespace quadrature {

template <class F, class Real, class Policy = policies::policy<> >
__device__ auto exp_sinh_integrate(const F& f, Real a, Real b, Real tolerance, Real* error, Real* L1, boost::math::size_t* levels)
{
    BOOST_MATH_STD_USING

    using K = decltype(f(a));
    static_assert(!boost::math::is_integral<K>::value,
                  "The return type cannot be integral, it must be either a real or complex floating point type.");

    constexpr auto function = "boost::math::quadrature::exp_sinh<%1%>::integrate";

    // Neither limit may be a NaN:
    if((boost::math::isnan)(a) || (boost::math::isnan)(b))
    {
       return static_cast<K>(policies::raise_domain_error(function, "NaN supplied as one limit of integration - sorry I don't know what to do", a, Policy()));
    }
    // Right limit is infinite:
    if ((boost::math::isfinite)(a) && (b >= boost::math::tools::max_value<Real>()))
    {
        // If a = 0, don't use an additional level of indirection:
        if (a == static_cast<Real>(0))
        {
            return detail::exp_sinh_integrate_impl(f, tolerance, error, L1, levels);
        }
        const auto u = [&](Real t)->K { return f(t + a); };
        return detail::exp_sinh_integrate_impl(u, tolerance, error, L1, levels);
    }

    if ((boost::math::isfinite)(b) && a <= -boost::math::tools::max_value<Real>())
    {
        const auto u = [&](Real t)->K { return f(b-t);};
        return detail::exp_sinh_integrate_impl(u, tolerance, error, L1, levels);
    }

    // Infinite limits:
    if ((a <= -boost::math::tools::max_value<Real>()) && (b >= boost::math::tools::max_value<Real>()))
    {
        return static_cast<K>(policies::raise_domain_error(function, "Use sinh_sinh quadrature for integration over the whole real line; exp_sinh is for half infinite integrals.", a, Policy()));
    }
    // If we get to here then both ends must necessarily be finite:
    return static_cast<K>(policies::raise_domain_error(function, "Use tanh_sinh quadrature for integration over finite domains; exp_sinh is for half infinite integrals.", a, Policy()));
}

template <class F, class Real, class Policy = policies::policy<> >
__device__ auto exp_sinh_integrate(const F& f, Real tolerance, Real* error, Real* L1, boost::math::size_t* levels)
{
    BOOST_MATH_STD_USING
    constexpr auto function = "boost::math::quadrature::exp_sinh<%1%>::integrate";
    if (abs(tolerance) > 1) {
        return policies::raise_domain_error(function, "The tolerance provided (%1%) is unusually large; did you confuse it with a domain bound?", tolerance, Policy());
    }
    return detail::exp_sinh_integrate_impl(f, tolerance, error, L1, levels);
}

} // namespace quadrature
} // namespace math
} // namespace boost

#endif // BOOST_MATH_ENABLE_CUDA

#endif // BOOST_MATH_QUADRATURE_EXP_SINH_HPP
