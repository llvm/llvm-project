// -*- C++ -*-
//===-- any_of.pass.cpp ---------------------------------------------------===//
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

/*
  TODO: consider implementing the following tests for a better code coverage
  - correctness
  - bad input argument (if applicable)
  - data corruption around/of input and output
  - correctly work with nested parallelism
  - check that algorithm does not require anything more than is described in its requirements section
*/

using namespace TestUtils;

struct test_any_of
{
    template <typename ExecutionPolicy, typename Iterator, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator begin, Iterator end, Predicate pred, bool expected)
    {

        auto actualr = std::any_of(exec, begin, end, pred);
        EXPECT_EQ(expected, actualr, "result for any_of");
    }
};

template <typename T>
void
test(size_t bits)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {

        // Sequence of odd values
        Sequence<T> in(n, [n, bits](size_t) { return T(2 * HashBits(n, bits - 1) ^ 1); });

        // Even value, or false when T is bool.
        T spike(2 * HashBits(n, bits - 1));
        Sequence<T> inCopy(in);

        invoke_on_all_policies(test_any_of(), in.begin(), in.end(), is_equal_to<T>(spike), false);
        invoke_on_all_policies(test_any_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), false);
        EXPECT_EQ(in, inCopy, "any_of modified input sequence");
        if (n > 0)
        {
            // Sprinkle in a hit
            in[2 * n / 3] = spike;
            invoke_on_all_policies(test_any_of(), in.begin(), in.end(), is_equal_to<T>(spike), true);
            invoke_on_all_policies(test_any_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), true);

            // Sprinkle in a few more hits
            in[n / 2] = spike;
            in[n / 3] = spike;
            invoke_on_all_policies(test_any_of(), in.begin(), in.end(), is_equal_to<T>(spike), true);
            invoke_on_all_policies(test_any_of(), in.cbegin(), in.cend(), is_equal_to<T>(spike), true);
        }
    }
}

struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto is_even = [&](float64_t v) {
            uint32_t i = (uint32_t)v;
            return i % 2 == 0;
        };
        any_of(exec, iter, iter, non_const(is_even));
    }
};

int
main()
{
    test<int32_t>(8 * sizeof(int32_t));
    test<uint16_t>(8 * sizeof(uint16_t));
    test<float64_t>(53);
#if !_PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN
    test<bool>(1);
#endif

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const>());

    std::cout << done() << std::endl;
    return 0;
}
