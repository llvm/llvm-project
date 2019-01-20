// -*- C++ -*-
//===-- test_generate.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for generate
#include <atomic>

#include "pstl/execution"
#include "pstl/algorithm"
#include "utils.h"

using namespace TestUtils;

template <typename T>
struct Generator_count
{
    const T def_val = T(-1);
    T
    operator()()
    {
        return def_val;
    }
    T
    default_value() const
    {
        return def_val;
    }
};

struct test_generate
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using namespace std;
        typedef typename std::iterator_traits<Iterator>::value_type T;

        // Try random-access iterator
        {
            Generator_count<T> g;
            generate(exec, first, last, g);
            EXPECT_TRUE(std::count(first, last, g.default_value()) == n, "generate wrong result for generate");
            std::fill(first, last, T(0));
        }

        {
            Generator_count<T> g;
            const auto m = n / 2;
            auto last = generate_n(exec, first, m, g);
            EXPECT_TRUE(std::count(first, last, g.default_value()) == m && last == std::next(first, m),
                        "generate_n wrong result for generate_n");
            std::fill(first, last, T(0));
        }
    }
};

template <typename T>
void
test_generate_by_type()
{
    for (size_t n = 0; n <= 100000; n = n < 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> in(n, [](size_t v) -> T { return T(0); }); //fill by zero

        invoke_on_all_policies(test_generate(), in.begin(), in.end(), in.size());
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto gen = []() { return T(0); };

        generate(exec, iter, iter, non_const(gen));
        generate_n(exec, iter, 0, non_const(gen));
    }
};

int32_t
main()
{

    test_generate_by_type<int32_t>();
    test_generate_by_type<float64_t>();

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
