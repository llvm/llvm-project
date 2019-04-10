// -*- C++ -*-
//===-- copy_if.pass.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for copy_if and remove_copy_if
#include "support/pstl_test_config.h"

#ifdef PSTL_STANDALONE_TESTS
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/utils.h"

using namespace TestUtils;

struct run_copy_if
{
#if _PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN // dummy specializations to skip testing in case of broken configuration
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(pstl::execution::parallel_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size n,
               Predicate pred, T trash)
    {
    }
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size n, Predicate pred, T trash)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size n,
               Predicate pred, T trash)
    {
        // Cleaning
        std::fill_n(expected_first, n, trash);
        std::fill_n(out_first, n, trash);

        // Run copy_if
        auto i = copy_if(first, last, expected_first, pred);
        auto k = copy_if(exec, first, last, out_first, pred);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong copy_if effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from copy_if");

        // Cleaning
        std::fill_n(expected_first, n, trash);
        std::fill_n(out_first, n, trash);
        // Run remove_copy_if
        i = remove_copy_if(first, last, expected_first, [=](const T& x) { return !pred(x); });
        k = remove_copy_if(exec, first, last, out_first, [=](const T& x) { return !pred(x); });
        EXPECT_EQ_N(expected_first, out_first, n, "wrong remove_copy_if effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from remove_copy_if");
    }
};

template <typename T, typename Predicate, typename Convert>
void
test(T trash, Predicate pred, Convert convert, bool check_weakness = true)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        size_t count = GuardSize;
        Sequence<T> in(n, [&](size_t k) -> T {
            T val = convert(n ^ k);
            count += pred(val) ? 1 : 0;
            return val;
        });

        Sequence<T> out(count, [=](size_t) { return trash; });
        Sequence<T> expected(count, [=](size_t) { return trash; });
        if (check_weakness)
        {
            auto expected_result = copy_if(in.cfbegin(), in.cfend(), expected.begin(), pred);
            size_t m = expected_result - expected.begin();
            EXPECT_TRUE(n / 4 <= m && m <= 3 * (n + 1) / 4, "weak test for copy_if");
        }
        invoke_on_all_policies(run_copy_if(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                               expected.end(), count, pred, trash);
        invoke_on_all_policies(run_copy_if(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(),
                               expected.end(), count, pred, trash);
    }
}

struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        auto is_even = [&](float64_t v) {
            uint32_t i = (uint32_t)v;
            return i % 2 == 0;
        };
        copy_if(exec, input_iter, input_iter, out_iter, non_const(is_even));

        invoke_if(exec, [&]() { remove_copy_if(exec, input_iter, input_iter, out_iter, non_const(is_even)); });
    }
};

int32_t
main()
{
    test<float64_t>(-666.0, [](const float64_t& x) { return x * x <= 1024; },
                    [](size_t j) { return ((j + 1) % 7 & 2) != 0 ? float64_t(j % 32) : float64_t(j % 33 + 34); });

    test<int32_t>(-666, [](const int32_t& x) { return x != 42; },
                  [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? int32_t(j + 1) : 42; });

#if !_PSTL_ICC_17_TEST_MAC_RELEASE_32_BROKEN
    test<Number>(Number(42, OddTag()), IsMultiple(3, OddTag()), [](int32_t j) { return Number(j, OddTag()); });
#endif

#if !_PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN
    test<int32_t>(-666, [](const int32_t& x) { return true; }, [](size_t j) { return j; }, false);
#endif

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const>());

    std::cout << done() << std::endl;
    return 0;
}
