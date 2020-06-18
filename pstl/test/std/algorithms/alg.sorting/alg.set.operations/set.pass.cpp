// -*- C++ -*-
//===-- set.pass.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

#include "support/pstl_test_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <functional>

#include "support/utils.h"

using namespace TestUtils;

template <typename T>
struct Num
{
    T val;

    Num() : val{} {}
    Num(const T& v) : val(v) {}

    //for "includes" checks
    template <typename T1>
    bool
    operator<(const Num<T1>& v1) const
    {
        return val < v1.val;
    }

    //The types Type1 and Type2 must be such that an object of type InputIt can be dereferenced and then implicitly converted to both of them
    template <typename T1>
    operator Num<T1>() const
    {
        return Num<T1>((T1)val);
    }

    friend bool
    operator==(const Num& v1, const Num& v2)
    {
        return v1.val == v2.val;
    }
};

template <typename Type>
struct test_set_union
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename std::enable_if<!TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        using T1 = typename std::iterator_traits<InputIterator1>::value_type;

        auto n1 = std::distance(first1, last1);
        auto n2 = std::distance(first2, last2);
        auto n = n1 + n2;
        Sequence<T1> expect(n);
        Sequence<T1> out(n);

        auto expect_res = std::set_union(first1, last1, first2, last2, expect.begin(), comp);
        auto res = std::set_union(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_union");
        EXPECT_EQ_N(expect.begin(), out.begin(), std::distance(out.begin(), res), "wrong set_union effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename std::enable_if<TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
    }
};

template <typename Type>
struct test_set_intersection
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename std::enable_if<!TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        using T1 = typename std::iterator_traits<InputIterator1>::value_type;

        auto n1 = std::distance(first1, last1);
        auto n2 = std::distance(first2, last2);
        auto n = n1 + n2;
        Sequence<T1> expect(n);
        Sequence<T1> out(n);

        auto expect_res = std::set_intersection(first1, last1, first2, last2, expect.begin(), comp);
        auto res = std::set_intersection(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_intersection");
        EXPECT_EQ_N(expect.begin(), out.begin(), std::distance(out.begin(), res), "wrong set_intersection effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename std::enable_if<TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
    }
};

template <typename Type>
struct test_set_difference
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename std::enable_if<!TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        using T1 = typename std::iterator_traits<InputIterator1>::value_type;

        auto n1 = std::distance(first1, last1);
        auto n2 = std::distance(first2, last2);
        auto n = n1 + n2;
        Sequence<T1> expect(n);
        Sequence<T1> out(n);

        auto expect_res = std::set_difference(first1, last1, first2, last2, expect.begin(), comp);
        auto res = std::set_difference(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_difference");
        EXPECT_EQ_N(expect.begin(), out.begin(), std::distance(out.begin(), res), "wrong set_difference effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename std::enable_if<TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
    }
};

template <typename Type>
struct test_set_symmetric_difference
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename std::enable_if<!TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        using T1 = typename std::iterator_traits<InputIterator1>::value_type;

        auto n1 = std::distance(first1, last1);
        auto n2 = std::distance(first2, last2);
        auto n = n1 + n2;
        Sequence<T1> expect(n);
        Sequence<T1> out(n);

        auto expect_res = std::set_symmetric_difference(first1, last1, first2, last2, expect.begin(), comp);
        auto res = std::set_symmetric_difference(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_symmetric_difference");
        EXPECT_EQ_N(expect.begin(), out.begin(), std::distance(out.begin(), res),
                    "wrong set_symmetric_difference effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename std::enable_if<TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2, Compare)
    {
    }
};

template <typename T1, typename T2, typename Compare>
void
test_set(Compare compare)
{

    const std::size_t n_max = 100000;

    // The rand()%(2*n+1) encourages generation of some duplicates.
    std::srand(4200);

    for (std::size_t n = 0; n < n_max; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        for (std::size_t m = 0; m < n_max; m = m <= 16 ? m + 1 : size_t(2.71828 * m))
        {
            //prepare the input ranges
            Sequence<T1> in1(n, [n](std::size_t k) { return rand() % (2 * k + 1); });
            Sequence<T2> in2(m, [m](std::size_t k) { return (m % 2) * rand() + rand() % (k + 1); });

            std::sort(in1.begin(), in1.end(), compare);
            std::sort(in2.begin(), in2.end(), compare);

            invoke_on_all_policies(test_set_union<T1>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(),
                                        compare);

            invoke_on_all_policies(test_set_intersection<T1>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(),
                                        compare);

            invoke_on_all_policies(test_set_difference<T1>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(),
                                        compare);

            invoke_on_all_policies(test_set_symmetric_difference<T1>(), in1.begin(), in1.end(), in2.cbegin(),
                                        in2.cend(), compare);
        }
    }
}

template <typename T>
struct test_non_const_set_difference
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_difference(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(std::less<T>()));
    }
};

template <typename T>
struct test_non_const_set_intersection
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_intersection(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(std::less<T>()));
    }
};

template <typename T>
struct test_non_const_set_symmetric_difference
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_symmetric_difference(exec, input_iter, input_iter, input_iter, input_iter, out_iter,
                                 non_const(std::less<T>()));
    }
};

template <typename T>
struct test_non_const_set_union
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_union(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(std::less<T>()));
    }
};

int
main()
{

    test_set<float64_t, float64_t>(std::less<>());
    test_set<Num<int64_t>, Num<int32_t>>([](const Num<int64_t>& x, const Num<int32_t>& y) { return x < y; });

    test_set<MemoryChecker, MemoryChecker>([](const MemoryChecker& val1, const MemoryChecker& val2) -> bool {
        return val1.value() < val2.value();
    });
    EXPECT_FALSE(MemoryChecker::alive_objects() < 0, "wrong effect from set algorithms: number of ctors calls < num of dtors calls");
    EXPECT_FALSE(MemoryChecker::alive_objects() > 0, "wrong effect from set algorithms: number of ctors calls > num of dtors calls");

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_set_difference<int32_t>>());

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_set_intersection<int32_t>>());

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_set_symmetric_difference<int32_t>>());

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_set_union<int32_t>>());

    std::cout << done() << std::endl;

    return 0;
}
