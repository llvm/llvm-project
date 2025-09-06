/*
 * Copyright Nick Thompson, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifdef _MSC_VER
#pragma warning(disable:4459)
#endif

#if !defined(_MSC_VER) || (_MSC_VER >= 1900)

#include <boost/math/quadrature/naive_monte_carlo.hpp>
#include "test_compile_result.hpp"

using boost::math::quadrature::naive_monte_carlo;
void compile_and_link_test()
{
    auto g = [&](std::vector<double> const &)
    {
        return 1.873;
    };
    std::vector<std::pair<double, double>> bounds{{0, 1}, {0, 1}, {0, 1}};
    naive_monte_carlo<double, decltype(g)> mc(g, bounds, 1.0);

    auto task = mc.integrate();
    check_result<double>(task.get());
}

#else
void compile_and_link_test()
{
}
#endif
