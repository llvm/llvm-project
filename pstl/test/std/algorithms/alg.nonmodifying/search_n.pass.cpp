// -*- C++ -*-
//===-- search_n.pass.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/pstl_test_config.h"

#include <execution>
#include <algorithm>

#include "support/utils.h"

using namespace TestUtils;

struct test_one_policy
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename Iterator, typename Size, typename T, typename Predicate>
    void
    operator()(pstl::execution::unsequenced_policy, Iterator b, Iterator e, Size count, const T& value, Predicate pred)
    {
    }
    template <typename Iterator, typename Size, typename T, typename Predicate>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, Iterator b, Iterator e, Size count, const T& value,
               Predicate pred)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator, typename Size, typename T, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator b, Iterator e, Size count, const T& value, Predicate pred)
    {
        using namespace std;
        auto expected = search_n(b, e, count, value, pred);
        auto actual = search_n(exec, b, e, count, value);
        EXPECT_TRUE(actual == expected, "wrong return result from search_n");

        actual = search_n(exec, b, e, count, value, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from search_n with a predicate");
    }
};

template <typename T>
void
test()
{

    const std::size_t max_n1 = 100000;
    const T value = T(1);
    for (std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1))
    {
        std::size_t sub_n[] = {0, 1, 3, n1, (n1 * 10) / 8};
        std::size_t res[] = {0, 1, n1 / 2, n1};
        for (auto n2 : sub_n)
        {
            // Some of standard libraries return "first" in this case. We return "last" according to the standard
            if (n2 == 0)
            {
                continue;
            }
            for (auto r : res)
            {
                Sequence<T> in(n1, [](std::size_t k) { return T(0); });
                std::size_t i = r, isub = 0;
                for (; i < n1 & isub < n2; ++i, ++isub)
                    in[i] = value;

                invoke_on_all_policies(test_one_policy(), in.begin(), in.begin() + n1, n2, value, std::equal_to<T>());
                invoke_on_all_policies(test_one_policy(), in.cbegin(), in.cbegin() + n1, n2, value, std::equal_to<T>());
            }
        }
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        invoke_if(exec, [&]() { search_n(exec, iter, iter, 0, T(0), non_const(std::equal_to<T>())); });
    }
};

int32_t
main()
{
    test<int32_t>();
    test<uint16_t>();
    test<float64_t>();
#if !_PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN
    test<bool>();
#endif

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
