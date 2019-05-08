// -*- C++ -*-
//===-- reduce.pass.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/pstl_test_config.h"

#include <execution>
#include <numeric>

#include "support/utils.h"

using namespace TestUtils;

struct test_long_forms_for_one_policy
{
    template <typename Policy, typename Iterator, typename T, typename BinaryOp>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, T init, BinaryOp binary, T expected)
    {
        T result_r = std::reduce(exec, first, last, init, binary);
        EXPECT_EQ(expected, result_r, "bad result from reduce(exec, first, last, init, binary_op)");
    }
};

template <typename T, typename BinaryOp, typename F>
void
test_long_form(T init, BinaryOp binary_op, F f)
{
    // Try sequences of various lengths
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        T expected(init);
        Sequence<T> in(n, [n, f](size_t k) { return f((int32_t(k ^ n) % 1000 - 500)); });
        for (size_t k = 0; k < n; ++k)
            expected = binary_op(expected, in[k]);

        using namespace std;

        T result = transform_reduce_serial(in.cfbegin(), in.cfend(), init, binary_op, [](const T& t) { return t; });
        EXPECT_EQ(expected, result, "bad result from reduce(first, last, init, binary_op_op)");

        invoke_on_all_policies(test_long_forms_for_one_policy(), in.begin(), in.end(), init, binary_op, expected);
        invoke_on_all_policies(test_long_forms_for_one_policy(), in.cbegin(), in.cend(), init, binary_op, expected);
    }
}

struct test_two_short_forms
{

#if _PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename Iterator>
    void
    operator()(pstl::execution::parallel_policy, Iterator first, Iterator last, Sum init, Sum expected)
    {
    }
    template <typename Iterator>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, Iterator first, Iterator last, Sum init, Sum expected)
    {
    }
#endif

    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Sum init, Sum expected)
    {
        using namespace std;

        Sum r0 = init + reduce(exec, first, last);
        EXPECT_EQ(expected, r0, "bad result from reduce(exec, first, last)");

        Sum r1 = reduce(exec, first, last, init);
        EXPECT_EQ(expected, r1, "bad result from reduce(exec, first, last, init)");
    }
};

// Test forms of reduce(...) that omit the binary_op or init operands.
void
test_short_forms()
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sum init(42, OddTag());
        Sum expected(init);
        Sequence<Sum> in(n, [n](size_t k) { return Sum((int32_t(k ^ n) % 1000 - 500), OddTag()); });
        for (size_t k = 0; k < n; ++k)
            expected = expected + in[k];
        invoke_on_all_policies(test_two_short_forms(), in.begin(), in.end(), init, expected);
        invoke_on_all_policies(test_two_short_forms(), in.cbegin(), in.cend(), init, expected);
    }
}

int32_t
main()
{
    // Test for popular types
    test_long_form(42, std::plus<int32_t>(), [](int32_t x) { return x; });
    test_long_form(42.0, std::plus<float64_t>(), [](float64_t x) { return x; });

    // Test for strict types
    test_long_form<Number>(Number(42, OddTag()), Add(OddTag()), [](int32_t x) { return Number(x, OddTag()); });

    // Short forms are just facade for long forms, so just test with a single type.
    test_short_forms();
    std::cout << done() << std::endl;
    return 0;
}
