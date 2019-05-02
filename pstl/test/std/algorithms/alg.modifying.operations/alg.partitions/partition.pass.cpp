// -*- C++ -*-
//===-- partition.pass.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for stable_partition and partition
#include "support/pstl_test_config.h"

#include <execution>
#include <algorithm>
#include <iterator>
#include <type_traits>

#include "support/utils.h"

using namespace TestUtils;

template <typename T>
struct DataType
{
    explicit DataType(int32_t k) : my_val(k) {}
    DataType(DataType&& input) { my_val = std::move(input.my_val); }
    DataType&
    operator=(DataType&& input)
    {
        my_val = std::move(input.my_val);
        return *this;
    }
    T
    get_val() const
    {
        return my_val;
    }

    friend std::ostream&
    operator<<(std::ostream& stream, const DataType<T>& input)
    {
        return stream << input.my_val;
    }

  private:
    T my_val;
};

template <typename Iterator>
typename std::enable_if<std::is_trivial<typename std::iterator_traits<Iterator>::value_type>::value, bool>::type
is_equal(Iterator first, Iterator last, Iterator d_first)
{
    return std::equal(first, last, d_first);
}

template <typename Iterator>
typename std::enable_if<!std::is_trivial<typename std::iterator_traits<Iterator>::value_type>::value, bool>::type
is_equal(Iterator first, Iterator last, Iterator d_first)
{
    return true;
}

struct test_one_policy
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specializations to skip testing in case of broken configuration
    template <typename BiDirIt, typename Size, typename UnaryOp, typename Generator>
    void
    operator()(pstl::execution::unsequenced_policy, BiDirIt first, BiDirIt last, BiDirIt exp_first, BiDirIt exp_last,
               Size n, UnaryOp unary_op, Generator generator)
    {
    }

    template <typename BiDirIt, typename Size, typename UnaryOp, typename Generator>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, BiDirIt first, BiDirIt last, BiDirIt exp_first,
               BiDirIt exp_last, Size n, UnaryOp unary_op, Generator generator)
    {
    }
#elif _PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN //dummy specializations to skip testing in case of broken configuration
    template <typename BiDirIt, typename Size, typename UnaryOp, typename Generator>
    void
    operator()(pstl::execution::parallel_policy, BiDirIt first, BiDirIt last, BiDirIt exp_first, BiDirIt exp_last,
               Size n, UnaryOp unary_op, Generator generator)
    {
    }

    template <typename BiDirIt, typename Size, typename UnaryOp, typename Generator>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, BiDirIt first, BiDirIt last, BiDirIt exp_first,
               BiDirIt exp_last, Size n, UnaryOp unary_op, Generator generator)
    {
    }
#endif

    template <typename Policy, typename BiDirIt, typename Size, typename UnaryOp, typename Generator>
    typename std::enable_if<!is_same_iterator_category<BiDirIt, std::forward_iterator_tag>::value, void>::type
    operator()(Policy&& exec, BiDirIt first, BiDirIt last, BiDirIt exp_first, BiDirIt exp_last, Size n,
               UnaryOp unary_op, Generator generator)
    {
        // partition
        {
            fill_data(first, last, generator);
            BiDirIt actual_ret = std::partition(exec, first, last, unary_op);
            EXPECT_TRUE(std::all_of(first, actual_ret, unary_op) && !std::any_of(actual_ret, last, unary_op),
                        "wrong effect from partition");
        }
        // stable_partition
        {
            fill_data(exp_first, exp_last, generator);
            BiDirIt exp_ret = std::stable_partition(exp_first, exp_last, unary_op);
            fill_data(first, last, generator);
            BiDirIt actual_ret = std::stable_partition(exec, first, last, unary_op);

            EXPECT_TRUE(std::distance(first, actual_ret) == std::distance(exp_first, exp_ret),
                        "wrong result from stable_partition");
            EXPECT_TRUE((is_equal<BiDirIt>(exp_first, exp_last, first)), "wrong effect from stable_partition");
        }
    }
    template <typename Policy, typename BiDirIt, typename Size, typename UnaryOp, typename Generator>
    typename std::enable_if<is_same_iterator_category<BiDirIt, std::forward_iterator_tag>::value, void>::type
    operator()(Policy&& exec, BiDirIt first, BiDirIt last, BiDirIt exp_first, BiDirIt exp_last, Size n,
               UnaryOp unary_op, Generator generator)
    {
    }
};

template <typename T, typename Generator, typename UnaryPred>
void
test_by_type(Generator generator, UnaryPred pred)
{

    using namespace std;
    size_t max_size = 100000;
    Sequence<T> in(max_size, [](size_t v) { return T(v); });
    Sequence<T> exp(max_size, [](size_t v) { return T(v); });

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_policies(test_one_policy(), in.begin(), in.begin() + n, exp.begin(), exp.begin() + n, n, pred,
                               generator);
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
        invoke_if(exec, [&]() {
            partition(exec, iter, iter, non_const(is_even));
            stable_partition(exec, iter, iter, non_const(is_even));
        });
    }
};

int32_t
main()
{
#if !_PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN
    test_by_type<int32_t>([](int32_t i) { return i; }, [](int32_t) { return true; });
#endif
    test_by_type<float64_t>([](int32_t i) { return -i; }, [](const float64_t x) { return x < 0; });
    test_by_type<int64_t>([](int32_t i) { return i + 1; }, [](int64_t x) { return x % 3 == 0; });
    test_by_type<DataType<float32_t>>([](int32_t i) { return DataType<float32_t>(2 * i + 1); },
                                      [](const DataType<float32_t>& x) { return x.get_val() < 0; });

    test_algo_basic_single<int32_t>(run_for_rnd_bi<test_non_const>());

    std::cout << done() << std::endl;
    return 0;
}
