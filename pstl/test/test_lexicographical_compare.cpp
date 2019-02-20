// -*- C++ -*-
//===-- test_lexicographical_compare.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pstl_test_config.h"
#include <string>
#include <iostream>

#include "pstl/execution"
#include "pstl/algorithm"
#include "utils.h"

using namespace TestUtils;

struct test_one_policy
{

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2,
               Predicate pred)
    {
        const bool expected = std::lexicographical_compare(begin1, end1, begin2, end2, pred);
        const bool actual = std::lexicographical_compare(exec, begin1, end1, begin2, end2, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from lexicographical compare with predicate");
    }

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2)
    {
        const bool expected = std::lexicographical_compare(begin1, end1, begin2, end2);
        const bool actual = std::lexicographical_compare(exec, begin1, end1, begin2, end2);
        EXPECT_TRUE(actual == expected, "wrong return result from lexicographical compare without predicate");
    }
};

template <typename T1, typename T2, typename Predicate>
void
test(Predicate pred)
{

    const std::size_t max_n = 1000000;
    Sequence<T1> in1(max_n, [](std::size_t k) { return T1(k); });
    Sequence<T2> in2(2 * max_n, [](std::size_t k) { return T2(k); });

    std::size_t n2;

    // Test case: Call algorithm's version without predicate.
    invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cbegin() + max_n, in2.cbegin() + 3 * max_n / 10,
                           in2.cbegin() + 5 * max_n / 10);

    // Test case: If one range is a prefix of another, the shorter range is lexicographically less than the other.
    std::size_t max_n2 = max_n / 10;
    invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + max_n, in2.cbegin(), in2.cbegin() + max_n2,
                           pred);
    invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + max_n, in2.begin() + max_n2,
                           in2.begin() + 3 * max_n2, pred);

    // Test case: If one range is a prefix of another, the shorter range is lexicographically less than the other.
    max_n2 = 2 * max_n;
    invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cbegin() + max_n, in2.begin(), in2.begin() + max_n2,
                           pred);

    for (std::size_t n1 = 0; n1 <= max_n; n1 = n1 <= 16 ? n1 + 1 : std::size_t(3.1415 * n1))
    {
        // Test case: If two ranges have equivalent elements and are of the same length, then the ranges are lexicographically equal.
        n2 = n1;
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n1, in2.begin(), in2.begin() + n2, pred);

        n2 = n1;
        // Test case: two ranges have different elements and are of the same length (second sequence less than first)
        std::size_t ind = n1 / 2;
        in2[ind] = T2(-1);
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n1, in2.begin(), in2.begin() + n2, pred);
        in2[ind] = T2(ind);

        // Test case: two ranges have different elements and are of the same length (first sequence less than second)
        ind = n1 / 5;
        in1[ind] = T1(-1);
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n1, in2.cbegin(), in2.cbegin() + n2, pred);
        in1[ind] = T1(ind);
    }
}

template <typename Predicate>
void
test_string(Predicate pred)
{

    const std::size_t max_n = 1000000;
    std::string in1 = "";
    std::string in2 = "";
    for (std::size_t n1 = 0; n1 <= max_n; ++n1)
    {
        in1 += n1;
    }

    for (std::size_t n1 = 0; n1 <= 2 * max_n; ++n1)
    {
        in2 += n1;
    }

    std::size_t n2;

    for (std::size_t n1 = 0; n1 < in1.size(); n1 = n1 <= 16 ? n1 + 1 : std::size_t(3.1415 * n1))
    {
        // Test case: If two ranges have equivalent elements and are of the same length, then the ranges are lexicographically equal.
        n2 = n1;
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n1, in2.begin(), in2.begin() + n2, pred);

        n2 = n1;
        // Test case: two ranges have different elements and are of the same length (second sequence less than first)
        in2[n1 / 2] = 'a';
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n1, in2.begin(), in2.begin() + n2, pred);

        // Test case: two ranges have different elements and are of the same length (first sequence less than second)
        in1[n1 / 5] = 'a';
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n1, in2.cbegin(), in2.cbegin() + n2, pred);
    }
    invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cbegin() + max_n, in2.cbegin() + 3 * max_n / 10,
                           in2.cbegin() + 5 * max_n / 10);
}

template <typename T>
struct LocalWrapper
{
    explicit LocalWrapper(std::size_t k) : my_val(k) {}
    bool
    operator<(const LocalWrapper<T>& w) const
    {
        return my_val < w.my_val;
    }

  private:
    T my_val;
};

template <typename T>
struct test_non_const
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        invoke_if(exec, [&]() {
            lexicographical_compare(exec, first_iter, first_iter, second_iter, second_iter, non_const(std::less<T>()));
        });
    }
};

int32_t
main()
{
    test<uint16_t, float64_t>(std::less<float64_t>());
    test<float32_t, int32_t>(std::greater<float32_t>());
#if !__PSTL_ICC_18_TEST_EARLY_EXIT_AVX_RELEASE_BROKEN
    test<float64_t, int32_t>([](const float64_t x, const int32_t y) { return x * x < y * y; });
#endif
    test<LocalWrapper<int32_t>, LocalWrapper<int32_t>>(
        [](const LocalWrapper<int32_t>& x, const LocalWrapper<int32_t>& y) { return x < y; });
    test_string([](const char x, const char y) { return x < y; });

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
