//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef REVERSE_MODE_AUTODIFF_STL_NO_ET_HPP
#define REVERSE_MODE_AUTODIFF_STL_NO_ET_HPP

#include <boost/math/differentiation/detail/reverse_mode_autodiff_stl_expressions.hpp>
namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> fabs(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return fabs_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}
template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> abs(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return fabs(arg);
}
template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> ceil(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return ceil_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}
template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> floor(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return floor_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> exp(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return exp_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}

template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
rvar<RealType, DerivativeOrder> pow(const expression<RealType, DerivativeOrder, LHS> &lhs,
                                    const expression<RealType, DerivativeOrder, RHS> &rhs)
{
    return pow_expr<RealType, DerivativeOrder, LHS, RHS>(lhs, rhs);
}

template<typename RealType2,
         typename RealType1,
         size_t DerivativeOrder,
         typename ARG,
         typename = typename std::enable_if<!detail::is_expression<RealType2>::value>::type>
rvar<RealType1, DerivativeOrder> pow(const expression<RealType1, DerivativeOrder, ARG> &arg,
                                     const RealType2                                   &v)
{
    return expr_pow_float_expr<RealType1, DerivativeOrder, ARG>(arg, static_cast<RealType1>(v));
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> pow(const RealType                                   &v,
                                    const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return float_pow_expr_expr<RealType, DerivativeOrder, ARG>(arg, v);
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> log(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return log_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> sqrt(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return sqrt_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> frexp(const expression<RealType, DerivativeOrder, ARG> &arg, int *i)
{
    BOOST_MATH_STD_USING
    frexp(arg.evaluate(), i);
    return arg / pow(static_cast<RealType>(2.0), *i);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
auto ldexp(const expression<RealType, DerivativeOrder, ARG> &arg, const int &i)
{
    BOOST_MATH_STD_USING
    return arg * pow(static_cast<RealType>(2.0), i);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> cos(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return cos_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> sin(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return sin_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> tan(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return tan_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> acos(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return acos_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> asin(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return asin_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> atan(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return atan_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
};

template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
rvar<RealType, DerivativeOrder> atan2(const expression<RealType, DerivativeOrder, LHS> &lhs,
                                      const expression<RealType, DerivativeOrder, RHS> &rhs)
{
    return atan2_expr<RealType, DerivativeOrder, LHS, RHS>(lhs, rhs);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> atan2(const expression<RealType, DerivativeOrder, ARG> &arg,
                                      const RealType                                   &v)
{
    return atan2_right_float_expr<RealType, DerivativeOrder, ARG>(arg, v);
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> atan2(const RealType                                   &v,
                                      const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return atan2_left_float_expr<RealType, DerivativeOrder, ARG>(arg, v);
};

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> trunc(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return trunc_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}

template<typename RealType, size_t DerivativeOrder, typename LHS, typename RHS>
auto fmod(const expression<RealType, DerivativeOrder, LHS> &lhs,
          const expression<RealType, DerivativeOrder, RHS> &rhs)
{
    return fmod_expr<RealType, DerivativeOrder, LHS, RHS>(lhs, rhs);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
auto fmod(const expression<RealType, DerivativeOrder, ARG> &lhs, const RealType rhs)
{
    return fmod_right_float_expr<RealType, DerivativeOrder, ARG>(lhs, rhs);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
auto fmod(const RealType lhs, const expression<RealType, DerivativeOrder, ARG> &rhs)
{
    return fmod_left_float_expr<RealType, DerivativeOrder, ARG>(rhs, lhs);
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> round(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return round_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
int iround(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    rvar<RealType, DerivativeOrder> tmp = arg.evaluate();
    return iround(tmp.item());
}
template<typename RealType, size_t DerivativeOrder, typename ARG>
long lround(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    BOOST_MATH_STD_USING
    rvar<RealType, DerivativeOrder> tmp = arg.evaluate();
    return lround(tmp.item());
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
long long llround(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    rvar<RealType, DerivativeOrder> tmp = arg.evaluate();
    return llround(tmp.item());
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
int itrunc(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    rvar<RealType, DerivativeOrder> tmp = arg.evaluate();
    return itrunc(tmp.item());
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
long ltrunc(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    rvar<RealType, DerivativeOrder> tmp = arg.evaluate();
    return ltrunc(tmp.item());
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
long long lltrunc(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    rvar<RealType, DerivativeOrder> tmp = arg.evaluate();
    return lltrunc(tmp.item());
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> sinh(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return sinh_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> cosh(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return cosh_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> tanh(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return tanh_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}

template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> log10(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return log10_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}
template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> asinh(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return asinh_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}
template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> acosh(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return acosh_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}
template<typename RealType, size_t DerivativeOrder, typename ARG>
rvar<RealType, DerivativeOrder> atanh(const expression<RealType, DerivativeOrder, ARG> &arg)
{
    return atanh_expr<RealType, DerivativeOrder, ARG>(arg, static_cast<RealType>(0.0));
}
} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost

#endif
