// -*- C++ -*-
//===-- unique_copy_equal.pass.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Tests for unique_copy
#include "support/pstl_test_config.h"

#include <execution>
#include <algorithm>

#include "support/utils.h"

using namespace TestUtils;

struct run_unique_copy
{
#if defined(_PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN) // dummy specializations to skip testing in case of broken configuration
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
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2, Size n, Predicate pred,
               T trash)
    {
        // Cleaning
        std::fill_n(expected_first, n, trash);
        std::fill_n(out_first, n, trash);

        // Run unique_copy
        auto i = unique_copy(first, last, expected_first);
        auto k = unique_copy(exec, first, last, out_first);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong unique_copy effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from unique_copy");

        // Cleaning
        std::fill_n(expected_first, n, trash);
        std::fill_n(out_first, n, trash);
        // Run unique_copy with predicate
        i = unique_copy(first, last, expected_first, pred);
        k = unique_copy(exec, first, last, out_first, pred);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong unique_copy with predicate effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from unique_copy with predicate");
    }
};

template <typename T, typename BinaryPredicate, typename Convert>
void
test(T trash, BinaryPredicate pred, Convert convert, bool check_weakness = true)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        Sequence<T> in(n, [&](size_t k) -> T { return convert(k ^ n); });
        using namespace std;
        size_t count = GuardSize;
        for (size_t k = 0; k < in.size(); ++k)
            count += k == 0 || !pred(in[k], in[k - 1]) ? 1 : 0;
        Sequence<T> out(count, [=](size_t) { return trash; });
        Sequence<T> expected(count, [=](size_t) { return trash; });
        if (check_weakness)
        {
            auto expected_result = unique_copy(in.begin(), in.end(), expected.begin(), pred);
            size_t m = expected_result - expected.begin();
            EXPECT_TRUE(n / (n < 10000 ? 4 : 6) <= m && m <= (3 * n + 1) / 4, "weak test for unique_copy");
        }
        invoke_on_all_policies(run_unique_copy(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                               expected.end(), count, pred, trash);
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        unique_copy(exec, input_iter, input_iter, out_iter, non_const(std::equal_to<T>()));
    }
};

int
main()
{
    test<Number>(Number(42, OddTag()), std::equal_to<Number>(),
                 [](int32_t j) { return Number(3 * j / 13 ^ (j & 8), OddTag()); });

    test<float32_t>(float32_t(42), std::equal_to<float32_t>(),
                    [](int32_t j) { return float32_t(5 * j / 23 ^ (j / 7)); });
#if !defined(_PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN)
    test<float32_t>(float32_t(42), [](float32_t, float32_t) { return false; }, [](int32_t j) { return float32_t(j); },
                    false);
#endif

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
