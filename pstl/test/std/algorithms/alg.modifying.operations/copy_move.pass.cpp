// -*- C++ -*-
//===-- copy_move.pass.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for copy, move and copy_n

#include "support/pstl_test_config.h"

#include <execution>
#include <algorithm>

#include "support/utils.h"

using namespace TestUtils;

struct run_copy
{

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size, typename T>
    void
    operator()(pstl::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size size,
               Size n, T trash)
    {
    }

    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size, typename T>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size size, Size n, T trash)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size size,
               Size n, T trash)
    {
        // Cleaning
        std::fill_n(expected_first, size, trash);
        std::fill_n(out_first, size, trash);

        // Run copy
        copy(first, last, expected_first);
        auto k = copy(exec, first, last, out_first);
        for (size_t j = 0; j < GuardSize; ++j)
            ++k;
        EXPECT_EQ_N(expected_first, out_first, size, "wrong effect from copy");
        EXPECT_TRUE(out_last == k, "wrong return value from copy");

        // Cleaning
        std::fill_n(out_first, size, trash);
        // Run copy_n
        k = copy_n(exec, first, n, out_first);
        for (size_t j = 0; j < GuardSize; ++j)
            ++k;
        EXPECT_EQ_N(expected_first, out_first, size, "wrong effect from copy_n");
        EXPECT_TRUE(out_last == k, "wrong return value from copy_n");
    }
};

template <typename T>
struct run_move
{

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(pstl::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size size,
               Size n, T trash)
    {
    }

    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size size, Size n, T trash)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size size,
               Size n, T trash)
    {
        // Cleaning
        std::fill_n(expected_first, size, trash);
        std::fill_n(out_first, size, trash);

        // Run move
        move(first, last, expected_first);
        auto k = move(exec, first, last, out_first);
        for (size_t j = 0; j < GuardSize; ++j)
            ++k;
        EXPECT_EQ_N(expected_first, out_first, size, "wrong effect from move");
        EXPECT_TRUE(out_last == k, "wrong return value from move");
    }
};

template <typename T>
struct run_move<Wrapper<T>>
{

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(pstl::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size size,
               Size n, Wrapper<T> trash)
    {
    }

    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size size, Size n, Wrapper<T> trash)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size size,
               Size n, Wrapper<T> trash)
    {
        // Cleaning
        std::fill_n(out_first, size, trash);
        Wrapper<T>::SetMoveCount(0);

        // Run move
        auto k = move(exec, first, last, out_first);
        for (size_t j = 0; j < GuardSize; ++j)
            ++k;
        EXPECT_TRUE(Wrapper<T>::MoveCount() == size, "wrong effect from move");
        EXPECT_TRUE(out_last == k, "wrong return value from move");
    }
};

template <typename T, typename Convert>
void
test(T trash, Convert convert)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        Sequence<T> in(n, [&](size_t k) -> T {
            T val = convert(n ^ k);
            return val;
        });

        const size_t outN = n + GuardSize;
        Sequence<T> out(outN, [=](size_t) { return trash; });
        Sequence<T> expected(outN, [=](size_t) { return trash; });
        invoke_on_all_policies(run_copy(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                               expected.end(), outN, n, trash);
        invoke_on_all_policies(run_copy(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(),
                               expected.end(), outN, n, trash);
        invoke_on_all_policies(run_move<T>(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                               expected.end(), n, n, trash);

        // For this test const iterator isn't suitable
        // because const rvalue-reference call copy assignment operator
    }
}

int32_t
main()
{
    test<int32_t>(-666, [](size_t j) { return int32_t(j); });
    test<Wrapper<float64_t>>(Wrapper<float64_t>(-666.0), [](int32_t j) { return Wrapper<float64_t>(j); });

#if !_PSTL_ICC_16_17_TEST_64_TIMEOUT
    test<float64_t>(-666.0, [](size_t j) { return float64_t(j); });
    test<Number>(Number(42, OddTag()), [](int32_t j) { return Number(j, OddTag()); });
#endif
    std::cout << done() << std::endl;
    return 0;
}
