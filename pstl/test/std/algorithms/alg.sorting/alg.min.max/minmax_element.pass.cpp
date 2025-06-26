// -*- C++ -*-
//===-- minmax_element.pass.cpp -------------------------------------------===//
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
#include <set>
#include <cmath>

#include "support/utils.h"

using namespace TestUtils;

struct check_minelement
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        typedef typename std::iterator_traits<Iterator>::value_type T;
        const Iterator expect = std::min_element(begin, end);
        const Iterator result = std::min_element(exec, begin, end);
        const Iterator result_pred = std::min_element(exec, begin, end, std::less<T>());
        EXPECT_TRUE(expect == result, "wrong return result from min_element");
        EXPECT_TRUE(expect == result_pred, "wrong return result from min_element");
    }
};

struct check_maxelement
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        typedef typename std::iterator_traits<Iterator>::value_type T;
        const Iterator expect = std::max_element(begin, end);
        const Iterator result = std::max_element(exec, begin, end);
        const Iterator result_pred = std::max_element(exec, begin, end, std::less<T>());
        EXPECT_TRUE(expect == result, "wrong return result from max_element");
        EXPECT_TRUE(expect == result_pred, "wrong return result from max_element");
    }
};

struct check_minmaxelement
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        typedef typename std::iterator_traits<Iterator>::value_type T;
        const std::pair<Iterator, Iterator> expect = std::minmax_element(begin, end);
        const std::pair<Iterator, Iterator> got = std::minmax_element(exec, begin, end);
        const std::pair<Iterator, Iterator> got_pred = std::minmax_element(exec, begin, end, std::less<T>());
        EXPECT_TRUE(expect.first == got.first, "wrong return result from minmax_element (min part)");
        EXPECT_TRUE(expect.second == got.second, "wrong return result from minmax_element (max part)");
        EXPECT_TRUE(expect == got_pred, "wrong return result from minmax_element");
    }
};

template <typename T>
struct sequence_wrapper
{
    TestUtils::Sequence<T> seq;
    const T min_value;
    const T max_value;
    static const std::size_t bits = 30; // We assume that T can handle signed 2^bits+1 value

    // TestUtils::HashBits returns value between 0 and (1<<bits)-1,
    // therefore we could threat 1<<bits as maximum and -(1<<bits) as a minimum
    sequence_wrapper(std::size_t n) : seq(n), min_value(-(1 << bits)), max_value(1 << bits) {}

    void
    pattern_fill()
    {
        seq.fill([](std::size_t i) -> T { return T(TestUtils::HashBits(i, bits)); });
    }

    // sets first one at position `at` and bunch of them farther
    void
    set_desired_value(std::size_t at, T value)
    {
        if (seq.size() == 0)
            return;
        seq[at] = value;

        //Producing serveral red herrings
        for (std::size_t i = at + 1; i < seq.size(); i += 1 + TestUtils::HashBits(i, 5))
            seq[i] = value;
    }
};

template <typename T>
void
test_by_type(std::size_t n)
{
    sequence_wrapper<T> wseq(n);

    // to avoid overtesing we use std::set to leave only unique indexes
    std::set<std::size_t> targets{0};
    if (n > 1)
    {
        targets.insert(1);
        targets.insert(2.718282 * n / 3);
        targets.insert(n / 2);
        targets.insert(n / 7.389056);
        targets.insert(n - 1); // last
    }

    for (std::set<std::size_t>::iterator it = targets.begin(); it != targets.end(); ++it)
    {
        wseq.pattern_fill();
        wseq.set_desired_value(*it, wseq.min_value);
        TestUtils::invoke_on_all_policies(check_minelement(), wseq.seq.cbegin(), wseq.seq.cend());
        TestUtils::invoke_on_all_policies(check_minelement(), wseq.seq.begin(), wseq.seq.end());

        wseq.set_desired_value(*it, wseq.max_value);
        TestUtils::invoke_on_all_policies(check_maxelement(), wseq.seq.cbegin(), wseq.seq.cend());
        TestUtils::invoke_on_all_policies(check_maxelement(), wseq.seq.begin(), wseq.seq.end());

        if (targets.size() > 1)
        {
            for (std::set<std::size_t>::reverse_iterator rit = targets.rbegin(); rit != targets.rend(); ++rit)
            {
                if (*rit == *it) // we requires at least 2 unique indexes in targets
                    break;
                wseq.pattern_fill();
                wseq.set_desired_value(*it, wseq.min_value);  // setting minimum element
                wseq.set_desired_value(*rit, wseq.max_value); // setting maximum element
                TestUtils::invoke_on_all_policies(check_minmaxelement(), wseq.seq.cbegin(), wseq.seq.cend());
                TestUtils::invoke_on_all_policies(check_minmaxelement(), wseq.seq.begin(), wseq.seq.end());
            }
        }
        else
        { // we must check this corner case; it can not be tested in loop above
            TestUtils::invoke_on_all_policies(check_minmaxelement(), wseq.seq.cbegin(), wseq.seq.cend());
            TestUtils::invoke_on_all_policies(check_minmaxelement(), wseq.seq.begin(), wseq.seq.end());
        }
    }
}

// should provide minimal requirements only
struct OnlyLessCompare
{
    int32_t val;
    OnlyLessCompare() : val(0) {}
    OnlyLessCompare(int32_t val_) : val(val_) {}
    bool
    operator<(const OnlyLessCompare& other) const
    {
        return val < other.val;
    }
};

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        max_element(exec, iter, iter, non_const(std::less<T>()));
        min_element(exec, iter, iter, non_const(std::less<T>()));
        minmax_element(exec, iter, iter, non_const(std::less<T>()));
    }
};

int
main()
{
    using TestUtils::float64_t;
    const std::size_t N = 100000;

    for (std::size_t n = 0; n < N; n = n < 16 ? n + 1 : size_t(3.14159 * n))
    {
        test_by_type<float64_t>(n);
        test_by_type<OnlyLessCompare>(n);
    }

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << TestUtils::done() << std::endl;
    return 0;
}
