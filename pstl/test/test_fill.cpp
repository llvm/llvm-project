// -*- C++ -*-
//===-- test_fill.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for fill/fill_n

#include "pstl/execution"
#include "pstl/algorithm"
#include "utils.h"

using namespace TestUtils;

struct test_fill
{
    template <typename It, typename T>
    bool
    check(It first, It last, const T& value)
    {
        for (; first != last; ++first)
            if (*first != value)
                return false;
        return true;
    }

    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& value)
    {
        fill(first, last, T(value + 1)); // initialize memory with different value

        fill(exec, first, last, value);
        EXPECT_TRUE(check(first, last, value), "fill wrong result");
    }
};

struct test_fill_n
{
    template <typename It, typename Size, typename T>
    bool
    check(It first, Size n, const T& value)
    {
        for (Size i = 0; i < n; ++i, ++first)
            if (*first != value)
                return false;
        return true;
    }

    template <typename Policy, typename Iterator, typename Size, typename T>
    void
    operator()(Policy&& exec, Iterator first, Size n, const T& value)
    {
        fill_n(first, n, T(value + 1)); // initialize memory with different value

        const Iterator one_past_last = fill_n(exec, first, n, value);
        const Iterator expected_return = std::next(first, n);

        EXPECT_TRUE(expected_return == one_past_last, "fill_n should return Iterator to one past the element assigned");
        EXPECT_TRUE(check(first, n, value), "fill_n wrong result");

        //n == -1
        const Iterator res = fill_n(exec, first, -1, value);
        EXPECT_TRUE(res == first, "fill_n wrong result for n == -1");
    }
};

template <typename T>
void
test_fill_by_type(std::size_t n)
{
    Sequence<T> in(n, [](std::size_t v) -> T { return T(0); }); //fill with zeros
    T value = -1;

    invoke_on_all_policies(test_fill(), in.begin(), in.end(), value);
    invoke_on_all_policies(test_fill_n(), in.begin(), n, value);
}

int32_t
main()
{

    const std::size_t N = 100000;

    for (std::size_t n = 0; n < N; n = n < 16 ? n + 1 : size_t(3.1415 * n))
    {
        test_fill_by_type<int32_t>(n);
        test_fill_by_type<float64_t>(n);
    }

    std::cout << done() << std::endl;

    return 0;
}
