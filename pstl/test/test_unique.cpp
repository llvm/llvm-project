// -*- C++ -*-
//===-- test_unique.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test for unique
#include "pstl_test_config.h"

#include "pstl/execution"
#include "pstl/algorithm"
#include "utils.h"

using namespace TestUtils;

struct run_unique
{
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename ForwardIt, typename Generator>
    void
    operator()(pstl::execution::unsequenced_policy, ForwardIt first1, ForwardIt last1, ForwardIt first2,
               ForwardIt last2, Generator generator)
    {
    }

    template <typename ForwardIt, typename Generator>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, ForwardIt first1, ForwardIt last1, ForwardIt first2,
               ForwardIt last2, Generator generator)
    {
    }

    template <typename ForwardIt, typename BinaryPred, typename Generator>
    void
    operator()(pstl::execution::unsequenced_policy, ForwardIt first1, ForwardIt last1, ForwardIt first2,
               ForwardIt last2, BinaryPred pred, Generator generator)
    {
    }

    template <typename ForwardIt, typename BinaryPred, typename Generator>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, ForwardIt first1, ForwardIt last1, ForwardIt first2,
               ForwardIt last2, BinaryPred pred, Generator generator)
    {
    }
#endif

    template <typename Policy, typename ForwardIt, typename Generator>
    void
    operator()(Policy&& exec, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, Generator generator)
    {
        using namespace std;

        // Preparation
        fill_data(first1, last1, generator);
        fill_data(first2, last2, generator);

        ForwardIt i = unique(first1, last1);
        ForwardIt k = unique(exec, first2, last2);

        auto n = std::distance(first1, i);
        EXPECT_TRUE(std::distance(first2, k) == n, "wrong return value from unique without predicate");
        EXPECT_EQ_N(first1, first2, n, "wrong effect from unique without predicate");
    }

    template <typename Policy, typename ForwardIt, typename BinaryPred, typename Generator>
    void
    operator()(Policy&& exec, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, BinaryPred pred,
               Generator generator)
    {
        using namespace std;

        // Preparation
        fill_data(first1, last1, generator);
        fill_data(first2, last2, generator);

        ForwardIt i = unique(first1, last1, pred);
        ForwardIt k = unique(exec, first2, last2, pred);

        auto n = std::distance(first1, i);
        EXPECT_TRUE(std::distance(first2, k) == n, "wrong return value from unique with predicate");
        EXPECT_EQ_N(first1, first2, n, "wrong effect from unique with predicate");
    }
};

template <typename T, typename Generator, typename Predicate>
void
test(Generator generator, Predicate pred)
{
    const std::size_t max_size = 1000000;
    Sequence<T> in(max_size, [](size_t v) { return T(v); });
    Sequence<T> exp(max_size, [](size_t v) { return T(v); });

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_policies(run_unique(), exp.begin(), exp.begin() + n, in.begin(), in.begin() + n, generator);
        invoke_on_all_policies(run_unique(), exp.begin(), exp.begin() + n, in.begin(), in.begin() + n, pred, generator);
    }
}

template <typename T>
struct LocalWrapper
{
    T my_val;

    explicit LocalWrapper(T k) : my_val(k) {}
    LocalWrapper(LocalWrapper&& input) : my_val(std::move(input.my_val)) {}
    LocalWrapper&
    operator=(LocalWrapper&& input)
    {
        my_val = std::move(input.my_val);
        return *this;
    }
    friend bool
    operator==(const LocalWrapper<T>& x, const LocalWrapper<T>& y)
    {
        return x.my_val == y.my_val;
    }
};

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        invoke_if(exec, [&]() { unique(exec, iter, iter, non_const(std::equal_to<T>())); });
    }
};

int32_t
main()
{
#if !__PSTL_ICC_16_17_18_TEST_UNIQUE_MASK_RELEASE_BROKEN
    test<int32_t>([](size_t j) { return j / 3; },
                  [](const int32_t& val1, const int32_t& val2) { return val1 * val1 == val2 * val2; });
    test<float64_t>([](size_t) { return float64_t(1); },
                    [](const float64_t& val1, const float64_t& val2) { return val1 != val2; });
#endif
    test<LocalWrapper<uint32_t>>([](size_t j) { return LocalWrapper<uint32_t>(j); },
                                 [](const LocalWrapper<uint32_t>& val1, const LocalWrapper<uint32_t>& val2) {
                                     return val1.my_val != val2.my_val;
                                 });

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
