// -*- C++ -*-
//===-- count.pass.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Tests for count and count_if
#include "support/pstl_test_config.h"

#include <execution>
#include <algorithm>

#include "support/utils.h"

using namespace TestUtils;

struct test_count
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, T needle)
    {
        auto expected = std::count(first, last, needle);
        auto result = std::count(exec, first, last, needle);
        EXPECT_EQ(expected, result, "wrong count result");
    }
};

struct test_count_if
{
    template <typename Policy, typename Iterator, typename Predicate>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred)
    {
        auto expected = std::count_if(first, last, pred);
        auto result = std::count_if(exec, first, last, pred);
        EXPECT_EQ(expected, result, "wrong count_if result");
    }
};

template <typename T>
class IsEqual
{
    T value;

  public:
    IsEqual(T value_, OddTag) : value(value_) {}
    bool
    operator()(const T& x) const
    {
        return x == value;
    }
};

template <typename In, typename T, typename Predicate, typename Convert>
void
test(T needle, Predicate pred, Convert convert)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [=](size_t k) -> In {
            // Sprinkle "42" and "50" early, so that short sequences have non-zero count.
            return convert((n - k - 1) % 3 == 0 ? 42 : (n - k - 2) % 5 == 0 ? 50 : 3 * (int(k) % 1000 - 500));
        });
        invoke_on_all_policies(test_count(), in.begin(), in.end(), needle);
        invoke_on_all_policies(test_count_if(), in.begin(), in.end(), pred);

        invoke_on_all_policies(test_count(), in.cbegin(), in.cend(), needle);
        invoke_on_all_policies(test_count_if(), in.cbegin(), in.cend(), pred);
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
        count_if(exec, iter, iter, non_const(is_even));
    }
};

int
main()
{
    test<int32_t>(42, IsEqual<int32_t>(50, OddTag()), [](int32_t j) { return j; });
#if !_PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN
    test<int32_t>(42, [](const int32_t&) { return true; }, [](int32_t j) { return j; });
#endif
    test<float64_t>(42, IsEqual<float64_t>(50, OddTag()), [](int32_t j) { return float64_t(j); });
    test<Number>(Number(42, OddTag()), IsEqual<Number>(Number(50, OddTag()), OddTag()),
                 [](int32_t j) { return Number(j, OddTag()); });

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const>());

    std::cout << done() << std::endl;
    return 0;
}
