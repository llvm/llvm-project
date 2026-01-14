// Copyright Nick Thompson, 2021
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_INTERPOLATORS_BEZIER_POLYNOMIAL_DETAIL_HPP
#define BOOST_MATH_INTERPOLATORS_BEZIER_POLYNOMIAL_DETAIL_HPP

#include <stdexcept>
#include <iostream>
#include <string>
#include <limits>

namespace boost::math::interpolators::detail {


template <class RandomAccessContainer>
static inline RandomAccessContainer& get_bezier_storage()
{
    static thread_local RandomAccessContainer the_storage;
    return the_storage;
}


template <class RandomAccessContainer>
class bezier_polynomial_imp
{
public:
    using Point = typename RandomAccessContainer::value_type;
    using Real = typename Point::value_type;
    using Z = typename RandomAccessContainer::size_type;

    bezier_polynomial_imp(RandomAccessContainer && control_points)
    {
        using std::to_string;
        if (control_points.size() < 2) {
            std::string err = std::string(__FILE__) + ":" + to_string(__LINE__)
               + " At least two points are required to form a Bezier curve. Only " + to_string(control_points.size())  + " points have been provided.";
            throw std::logic_error(err);
        }
        Z dimension = control_points[0].size();
        for (Z i = 0; i < control_points.size(); ++i) {
            if (control_points[i].size() != dimension) {
                std::string err = std::string(__FILE__) + ":" + to_string(__LINE__)
                + " All points passed to the Bezier polynomial must have the same dimension.";
                throw std::logic_error(err);
            }
        }
        control_points_ = std::move(control_points);
        auto & storage = get_bezier_storage<RandomAccessContainer>();
        if (storage.size() < control_points_.size() -1) {
            storage.resize(control_points_.size() -1);
        }
    }

    inline Point operator()(Real t) const
    {
        if (t < 0 || t > 1) {
            std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << "\n";
            std::cerr << "Querying the Bezier curve interpolator at t = " << t << " is not allowed; t in [0,1] is required.\n";
            Point p;
            for (Z i = 0; i < p.size(); ++i) {
                p[i] = std::numeric_limits<Real>::quiet_NaN();
            }
            return p;
        }

        auto & scratch_space = get_bezier_storage<RandomAccessContainer>();
        for (Z i = 0; i < control_points_.size() - 1; ++i) {
            for (Z j = 0; j < control_points_[0].size(); ++j) {
                scratch_space[i][j] = (1-t)*control_points_[i][j] + t*control_points_[i+1][j];
            }
        }

        decasteljau_recursion(scratch_space, control_points_.size() - 1, t);
        return scratch_space[0];
    }

    Point prime(Real t) {
        auto & scratch_space = get_bezier_storage<RandomAccessContainer>();
        for (Z i = 0; i < control_points_.size() - 1; ++i) {
            for (Z j = 0; j < control_points_[0].size(); ++j) {
                scratch_space[i][j] = control_points_[i+1][j] - control_points_[i][j];
            }
        }
        decasteljau_recursion(scratch_space, control_points_.size() - 1, t);
        for (Z j = 0; j < control_points_[0].size(); ++j) {
            scratch_space[0][j] *= (control_points_.size()-1);
        }
        return scratch_space[0];
    }


    void edit_control_point(Point const & p, Z index)
    {
        if (index >= control_points_.size()) {
            std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << "\n";
            std::cerr << "Attempting to edit a control point outside the bounds of the container; requested edit of index " << index << ", but there are only " << control_points_.size() << " control points.\n";
            return;
        }
        control_points_[index] = p;
    }

    RandomAccessContainer const & control_points() const {
        return control_points_;
    }

    // See "Bezier and B-spline techniques", section 2.7:
    // I cannot figure out why this doesn't work.
    /*RandomAccessContainer indefinite_integral() const {
        using std::fma;
        // control_points_.size() == n + 1
        RandomAccessContainer c(control_points_.size() + 1);
        // This is the constant of integration, chosen arbitrarily to be zero:
        for (Z j = 0; j < control_points_[0].size(); ++j) {
            c[0][j] = Real(0);
        }

        // Make the reciprocal approximation to unroll the iteration into a pile of fma's:
        Real rnp1 = Real(1)/control_points_.size();
        for (Z i = 1; i < c.size(); ++i) {
            for (Z j = 0; j < control_points_[0].size(); ++j) {
                //c[i][j] = c[i-1][j] + control_points_[i-1][j]*rnp1;
                c[i][j] = fma(rnp1, control_points_[i-1][j], c[i-1][j]);
            }
        }
        return c;
    }*/

    friend std::ostream& operator<<(std::ostream& out, bezier_polynomial_imp<RandomAccessContainer> const & bp) {
        out << "{";
        for (Z i = 0; i < bp.control_points_.size() - 1; ++i) {
            out << "(";
            for (Z j = 0; j < bp.control_points_[0].size() - 1; ++j) {
                out << bp.control_points_[i][j] << ", ";
            }
            out << bp.control_points_[i][bp.control_points_[0].size() - 1] << "), ";
        }
        out << "(";
        for (Z j = 0; j < bp.control_points_[0].size() - 1; ++j) {
            out << bp.control_points_.back()[j] << ", ";
        }
        out << bp.control_points_.back()[bp.control_points_[0].size() - 1] << ")}";
        return out;
    }

private:

    void decasteljau_recursion(RandomAccessContainer & points, Z n, Real t) const {
        if (n <= 1) {
            return;
        }
        for (Z i = 0; i < n - 1; ++i) {
            for (Z j = 0; j < points[0].size(); ++j) {
                points[i][j] = (1-t)*points[i][j] + t*points[i+1][j];
            }
        }
        decasteljau_recursion(points, n - 1, t);
    }

    RandomAccessContainer control_points_;
};


}
#endif
