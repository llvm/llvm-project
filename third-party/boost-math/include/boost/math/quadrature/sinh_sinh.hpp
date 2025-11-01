// Copyright Nick Thompson, 2017
// Copyright Matt Borland, 2024
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

/*
 * This class performs sinh-sinh quadrature over the entire real line.
 *
 * References:
 *
 * 1) Tanaka, Ken'ichiro, et al. "Function classes for double exponential integration formulas." Numerische Mathematik 111.4 (2009): 631-655.
 */

#ifndef BOOST_MATH_QUADRATURE_SINH_SINH_HPP
#define BOOST_MATH_QUADRATURE_SINH_SINH_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/tools/cstdint.hpp>
#include <boost/math/quadrature/detail/sinh_sinh_detail.hpp>
#include <boost/math/policies/error_handling.hpp>

#ifndef BOOST_MATH_HAS_NVRTC

#include <cmath>
#include <limits>
#include <memory>

namespace boost{ namespace math{ namespace quadrature {

template<class Real, class Policy = boost::math::policies::policy<> >
class sinh_sinh
{
public:
    sinh_sinh(size_t max_refinements = 9)
        : m_imp(std::make_shared<detail::sinh_sinh_detail<Real, Policy> >(max_refinements)) {}

    template<class F>
    auto integrate(const F f, Real tol = boost::math::tools::root_epsilon<Real>(), Real* error = nullptr, Real* L1 = nullptr, std::size_t* levels = nullptr) const ->decltype(std::declval<F>()(std::declval<Real>()))
    {
        return m_imp->integrate(f, tol, error, L1, levels);
    }

private:
    std::shared_ptr<detail::sinh_sinh_detail<Real, Policy>> m_imp;
};

}}}

#endif // BOOST_MATH_HAS_NVRTC

#ifdef BOOST_MATH_ENABLE_CUDA

namespace boost {
namespace math {
namespace quadrature {

template <class F, class Real, class Policy = boost::math::policies::policy<> >
__device__ auto sinh_sinh_integrate(const F& f, Real tol = boost::math::tools::root_epsilon<Real>(), Real* error = nullptr, Real* L1 = nullptr, boost::math::size_t* levels = nullptr)
{
    return detail::sinh_sinh_integrate_impl(f, tol, error, L1, levels);
}

} // namespace quadrature
} // namespace math
} // namespace boost

#endif // BOOST_MATH_ENABLE_CUDA

#endif // BOOST_MATH_QUADRATURE_SINH_SINH_HPP
