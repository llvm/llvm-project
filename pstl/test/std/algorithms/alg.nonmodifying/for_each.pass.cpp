// -*- C++ -*-
//===-- for_each.pass.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include "support/pstl_test_config.h"

#include <execution>
#include <algorithm>

#include "support/utils.h"

using namespace TestUtils;

template <typename Type>
struct Gen
{
    Type
    operator()(std::size_t k)
    {
        return Type(k % 5 != 1 ? 3 * k - 7 : 0);
    };
};

template <typename T>
struct Flip
{
    int32_t val;
    Flip(int32_t y) : val(y) {}
    T
    operator()(T& x) const
    {
        return x = val - x;
    }
};

struct test_one_policy
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator expected_first, Iterator expected_last, Size n)
    {
        typedef typename std::iterator_traits<Iterator>::value_type T;

        // Try for_each
        std::for_each(expected_first, expected_last, Flip<T>(1));
        for_each(exec, first, last, Flip<T>(1));
        EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_each");

        // Try for_each_n
        std::for_each_n(std::execution::seq, expected_first, n, Flip<T>(1));
        for_each_n(exec, first, n, Flip<T>(1));
        EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_each_n");
    }
};

template <typename T>
void
test()
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> inout(n, Gen<T>());
        Sequence<T> expected(n, Gen<T>());
        invoke_on_all_policies(test_one_policy(), inout.begin(), inout.end(), expected.begin(), expected.end(),
                               inout.size());
    }
}

struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        invoke_if(exec, [&]() {
            auto f = [](typename std::iterator_traits<Iterator>::reference x) { x = x + 1; };

            for_each(exec, iter, iter, non_const(f));
            for_each_n(exec, iter, 0, non_const(f));
        });
    }
};

int
main()
{
    test<int32_t>();
    test<uint16_t>();
    test<float64_t>();

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const>());

    std::cout << done() << std::endl;
    return 0;
}
