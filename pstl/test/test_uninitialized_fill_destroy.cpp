// -*- C++ -*-
//===-- test_uninitialized_fill_destroy.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for the destroy, destroy_n, uninitialized_fill, uninitialized_fill_n algorithms

#include "pstl/execution"
#include "pstl/memory"
#include "pstl/algorithm"
#include "utils.h"

using namespace TestUtils;

struct test_uninitialized_fill_destroy
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, std::size_t n, std::false_type)
    {
        using namespace std;
        {
            T::SetCount(0);
            uninitialized_fill(exec, first, last, in);
            size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
            EXPECT_TRUE(n == count, "wrong work of uninitialized_fill");
            destroy(exec, first, last);
            EXPECT_TRUE(T::Count() == 0, "wrong work of destroy");
        }

        {
            auto res = uninitialized_fill_n(exec, first, n, in);
            EXPECT_TRUE(res == last, "wrong result of uninitialized_fill_n");
            size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
            EXPECT_TRUE(n == count, "wrong work of uninitialized_fill_n");
            destroy_n(exec, first, n);
            EXPECT_TRUE(T::Count() == 0, "wrong work of destroy_n");
        }
    }
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, std::size_t n, std::true_type)
    {
        using namespace std;
        {
            destroy(exec, first, last);
            uninitialized_fill(exec, first, last, in);
            size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
            EXPECT_EQ(n, count, "wrong work of uninitialized:_fill");
        }
        {
            destroy_n(exec, first, n);
            auto res = uninitialized_fill_n(exec, first, n, in);
            size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
            EXPECT_EQ(n, count, "wrong work of uninitialized_fill_n");
            EXPECT_TRUE(res == last, "wrong result of uninitialized_fill_n");
        }
    }
};

template <typename T>
void
test_uninitialized_fill_destroy_by_type()
{
    std::size_t N = 100000;
    for (size_t n = 0; n <= N; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        std::unique_ptr<T[]> p(new T[n]);
        invoke_on_all_policies(test_uninitialized_fill_destroy(), p.get(), std::next(p.get(), n), T(), n,
                               std::is_trivial<T>());
    }
}

int32_t
main()
{
    // for trivial types
    test_uninitialized_fill_destroy_by_type<int32_t>();
    test_uninitialized_fill_destroy_by_type<float64_t>();

    // for user-defined types
    test_uninitialized_fill_destroy_by_type<Wrapper<std::string>>();
    test_uninitialized_fill_destroy_by_type<Wrapper<int8_t*>>();
    std::cout << done() << std::endl;

    return 0;
}
