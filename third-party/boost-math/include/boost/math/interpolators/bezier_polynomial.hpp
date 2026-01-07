// Copyright Nick Thompson, 2021
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_INTERPOLATORS_BEZIER_POLYNOMIAL_HPP
#define BOOST_MATH_INTERPOLATORS_BEZIER_POLYNOMIAL_HPP
#include <memory>
#include <boost/math/interpolators/detail/bezier_polynomial_detail.hpp>

#ifdef BOOST_MATH_NO_THREAD_LOCAL_WITH_NON_TRIVIAL_TYPES
#warning "Thread local storage support is necessary for the Bezier polynomial class to work."
#endif

namespace boost::math::interpolators {

template <class RandomAccessContainer>
class bezier_polynomial
{
public:
    using Point = typename RandomAccessContainer::value_type;
    using Real = typename Point::value_type;
    using Z = typename RandomAccessContainer::size_type;

    bezier_polynomial(RandomAccessContainer && control_points)
    : m_imp(std::make_shared<detail::bezier_polynomial_imp<RandomAccessContainer>>(std::move(control_points)))
    {
    }

    inline Point operator()(Real t) const
    {
        return (*m_imp)(t);
    }

    inline Point prime(Real t) const
    {
        return m_imp->prime(t);
    }

    void edit_control_point(Point const & p, Z index)
    {
        m_imp->edit_control_point(p, index);
    }

    RandomAccessContainer const & control_points() const
    {
        return m_imp->control_points();
    }

    friend std::ostream& operator<<(std::ostream& out, bezier_polynomial<RandomAccessContainer> const & bp) {
        out << *bp.m_imp;
        return out;
    }

private:
    std::shared_ptr<detail::bezier_polynomial_imp<RandomAccessContainer>> m_imp;
};

}
#endif
