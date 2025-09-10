//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstdint>
#include <vector>
#include <boost/math/statistics/detail/rank.hpp>
#include <boost/math/tools/config.hpp>
#include "math_unit_test.hpp"

template <typename T>
void test()
{
    std::vector<T> test_vals {T(1.0), T(3.2), T(2.4), T(5.6), T(4.1)};
    auto rank_vector = boost::math::statistics::detail::rank(test_vals.begin(), test_vals.end());

    CHECK_EQUAL(static_cast<std::size_t>(0), rank_vector[0]);
    CHECK_EQUAL(static_cast<std::size_t>(2), rank_vector[1]);
    CHECK_EQUAL(static_cast<std::size_t>(1), rank_vector[2]);
    CHECK_EQUAL(static_cast<std::size_t>(4), rank_vector[3]);
    CHECK_EQUAL(static_cast<std::size_t>(3), rank_vector[4]);

    // Remove duplicates
    test_vals.push_back(T(4.1));
    test_vals.push_back(T(2.4));
    rank_vector = boost::math::statistics::detail::rank(test_vals.begin(), test_vals.end());

    // Check the size is correct and the ordering is not disrupted
    CHECK_EQUAL(static_cast<std::size_t>(5), rank_vector.size());
    CHECK_EQUAL(static_cast<std::size_t>(0), rank_vector[0]);
    CHECK_EQUAL(static_cast<std::size_t>(2), rank_vector[1]);
    CHECK_EQUAL(static_cast<std::size_t>(1), rank_vector[2]);
    CHECK_EQUAL(static_cast<std::size_t>(4), rank_vector[3]);
    CHECK_EQUAL(static_cast<std::size_t>(3), rank_vector[4]);
}

template <typename T>
void container_test()
{
    std::vector<T> test_vals {T(1.0), T(3.2), T(2.4), T(5.6), T(4.1)};
    auto rank_vector = boost::math::statistics::detail::rank(test_vals);

    CHECK_EQUAL(static_cast<std::size_t>(0), rank_vector[0]);
    CHECK_EQUAL(static_cast<std::size_t>(2), rank_vector[1]);
    CHECK_EQUAL(static_cast<std::size_t>(1), rank_vector[2]);
    CHECK_EQUAL(static_cast<std::size_t>(4), rank_vector[3]);
    CHECK_EQUAL(static_cast<std::size_t>(3), rank_vector[4]);
}

#ifdef BOOST_MATH_EXEC_COMPATIBLE

#include <execution>

template<typename T, typename ExecutionPolicy>
void execution_test(ExecutionPolicy&& exec)
{
    std::vector<T> test_vals {T(1.0), T(3.2), T(2.4), T(5.6), T(4.1)};
    auto rank_vector = boost::math::statistics::detail::rank(exec, test_vals.begin(), test_vals.end());

    CHECK_EQUAL(static_cast<std::size_t>(0), rank_vector[0]);
    CHECK_EQUAL(static_cast<std::size_t>(2), rank_vector[1]);
    CHECK_EQUAL(static_cast<std::size_t>(1), rank_vector[2]);
    CHECK_EQUAL(static_cast<std::size_t>(4), rank_vector[3]);
    CHECK_EQUAL(static_cast<std::size_t>(3), rank_vector[4]);

    // Remove duplicates
    test_vals.push_back(T(4.1));
    test_vals.push_back(T(2.4));
    rank_vector = boost::math::statistics::detail::rank(exec, test_vals.begin(), test_vals.end());

    // Check the size is correct and the ordering is not disrupted
    CHECK_EQUAL(static_cast<std::size_t>(5), rank_vector.size());
    CHECK_EQUAL(static_cast<std::size_t>(0), rank_vector[0]);
    CHECK_EQUAL(static_cast<std::size_t>(2), rank_vector[1]);
    CHECK_EQUAL(static_cast<std::size_t>(1), rank_vector[2]);
    CHECK_EQUAL(static_cast<std::size_t>(4), rank_vector[3]);
    CHECK_EQUAL(static_cast<std::size_t>(3), rank_vector[4]);
}

#endif // BOOST_MATH_EXEC_COMPATIBLE

int main(void)
{
    test<float>();
    test<double>();
    test<long double>();

    container_test<float>();
    container_test<double>();
    container_test<long double>();

    #ifdef BOOST_MATH_EXEC_COMPATIBLE

    execution_test<float>(std::execution::par);
    execution_test<double>(std::execution::par);
    execution_test<long double>(std::execution::par);

    #endif // BOOST_MATH_EXEC_COMPATIBLE

    return boost::math::test::report_errors();
}
