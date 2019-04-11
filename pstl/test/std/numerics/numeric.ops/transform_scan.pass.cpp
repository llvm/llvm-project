// -*- C++ -*-
//===-- transform_scan.pass.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/pstl_test_config.h"

#ifdef PSTL_STANDALONE_TESTS
#include "pstl/execution"
#include "pstl/numeric"
#else
#include <execution>
#include <numeric>
#endif // PSTL_STANDALONE_TESTS

#include "support/utils.h"

using namespace TestUtils;

// Most of the framework required for testing inclusive and exclusive transform-scans is identical,
// so the tests for both are in this file.  Which is being tested is controlled by the global
// flag inclusive, which is set to each alternative by main().
static bool inclusive;

template <typename Iterator, typename Size, typename T>
void
check_and_reset(Iterator expected_first, Iterator out_first, Size n, T trash)
{
    EXPECT_EQ_N(expected_first, out_first, n,
                inclusive ? "wrong result from transform_inclusive_scan"
                          : "wrong result from transform_exclusive_scan");
    std::fill_n(out_first, n, trash);
}

struct test_transform_scan
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp,
              typename T, typename BinaryOp>
    typename std::enable_if<!TestUtils::isReverse<InputIterator>::value, void>::type
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator expected_first, OutputIterator expected_last, Size n,
               UnaryOp unary_op, T init, BinaryOp binary_op, T trash)
    {
        using namespace std;

        auto orr1 = inclusive ? transform_inclusive_scan(pstl::execution::seq, first, last, expected_first, binary_op,
                                                         unary_op, init)
                              : transform_exclusive_scan(pstl::execution::seq, first, last, expected_first, init,
                                                         binary_op, unary_op);
        auto orr2 = inclusive ? transform_inclusive_scan(exec, first, last, out_first, binary_op, unary_op, init)
                              : transform_exclusive_scan(exec, first, last, out_first, init, binary_op, unary_op);
        EXPECT_TRUE(out_last == orr2, "transform...scan returned wrong iterator");
        check_and_reset(expected_first, out_first, n, trash);

        // Checks inclusive scan if init is not provided
        if (inclusive && n > 0)
        {
            orr1 = transform_inclusive_scan(pstl::execution::seq, first, last, expected_first, binary_op, unary_op);
            orr2 = transform_inclusive_scan(exec, first, last, out_first, binary_op, unary_op);
            EXPECT_TRUE(out_last == orr2, "transform...scan returned wrong iterator");
            check_and_reset(expected_first, out_first, n, trash);
        }
    }

    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp,
              typename T, typename BinaryOp>
    typename std::enable_if<TestUtils::isReverse<InputIterator>::value, void>::type
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator expected_first, OutputIterator expected_last, Size n,
               UnaryOp unary_op, T init, BinaryOp binary_op, T trash)
    {
    }
};

const uint32_t encryption_mask = 0x314;

template <typename InputIterator, typename OutputIterator, typename UnaryOperation, typename T,
          typename BinaryOperation>
std::pair<OutputIterator, T>
transform_inclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op,
                                T init, BinaryOperation binary_op) noexcept
{
    for (; first != last; ++first, ++result)
    {
        init = binary_op(init, unary_op(*first));
        *result = init;
    }
    return std::make_pair(result, init);
}

template <typename InputIterator, typename OutputIterator, typename UnaryOperation, typename T,
          typename BinaryOperation>
std::pair<OutputIterator, T>
transform_exclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op,
                                T init, BinaryOperation binary_op) noexcept
{
    for (; first != last; ++first, ++result)
    {
        *result = init;
        init = binary_op(init, unary_op(*first));
    }
    return std::make_pair(result, init);
}

template <typename In, typename Out, typename UnaryOp, typename BinaryOp>
void
test(UnaryOp unary_op, Out init, BinaryOp binary_op, Out trash)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [](size_t k) { return In(k ^ encryption_mask); });

        Out tmp = init;
        Sequence<Out> expected(n, [&](size_t k) -> Out {
            if (inclusive)
            {
                tmp = binary_op(tmp, unary_op(in[k]));
                return tmp;
            }
            else
            {
                Out val = tmp;
                tmp = binary_op(tmp, unary_op(in[k]));
                return val;
            }
        });

        Sequence<Out> out(n, [&](size_t) { return trash; });

        auto result =
            inclusive
                ? transform_inclusive_scan_serial(in.cbegin(), in.cend(), out.fbegin(), unary_op, init, binary_op)
                : transform_exclusive_scan_serial(in.cbegin(), in.cend(), out.fbegin(), unary_op, init, binary_op);
        check_and_reset(expected.begin(), out.begin(), out.size(), trash);

        invoke_on_all_policies(test_transform_scan(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                               expected.end(), in.size(), unary_op, init, binary_op, trash);
        invoke_on_all_policies(test_transform_scan(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(),
                               expected.end(), in.size(), unary_op, init, binary_op, trash);
    }
}

template <typename In, typename Out, typename UnaryOp, typename BinaryOp>
void
test_matrix(UnaryOp unary_op, Out init, BinaryOp binary_op, Out trash)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [](size_t k) { return In(k, k + 1); });

        Sequence<Out> out(n, [&](size_t) { return trash; });
        Sequence<Out> expected(n, [&](size_t) { return trash; });

        invoke_on_all_policies(test_transform_scan(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                               expected.end(), in.size(), unary_op, init, binary_op, trash);
        invoke_on_all_policies(test_transform_scan(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(),
                               expected.end(), in.size(), unary_op, init, binary_op, trash);
    }
}

int32_t
main()
{
    for (int32_t mode = 0; mode < 2; ++mode)
    {
        inclusive = mode != 0;
#if !_PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN
        test_matrix<Matrix2x2<int32_t>, Matrix2x2<int32_t>>([](const Matrix2x2<int32_t> x) { return x; },
                                                            Matrix2x2<int32_t>(), multiply_matrix<int32_t>,
                                                            Matrix2x2<int32_t>(-666, 666));
#endif
        test<int32_t, uint32_t>([](int32_t x) { return x++; }, -123, [](int32_t x, int32_t y) { return x + y; }, 666);
    }
    std::cout << done() << std::endl;
    return 0;
}
