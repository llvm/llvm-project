// -*- C++ -*-
//===-- uninitialized_copy_move.pass.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for uninitialized_copy, uninitialized_copy_n, uninitialized_move, uninitialized_move_n

#include "support/pstl_test_config.h"

#include <execution>
#include <memory>

#include "support/utils.h"

using namespace TestUtils;

// function of checking correctness for uninitialized.construct.value
template <typename InputIterator, typename OutputIterator, typename Size>
bool
IsCheckValueCorrectness(InputIterator first1, OutputIterator first2, Size n)
{
    for (Size i = 0; i < n; ++i, ++first1, ++first2)
    {
        if (*first1 != *first2)
        {
            return false;
        }
    }
    return true;
}

struct test_uninitialized_copy_move
{
    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::false_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        std::destroy_n(exec, out_first, n);

        // reset counter of constructors
        T::SetCount(0);
        // run algorithm
        std::uninitialized_copy(exec, first, last, out_first);
        // compare counter of constructors to length of container
        EXPECT_TRUE(T::Count() == n, "wrong uninitialized_copy");
        // destroy objects for testing new algorithms on same memory
        std::destroy_n(exec, out_first, n);

        std::uninitialized_copy_n(exec, first, n, out_first);
        EXPECT_TRUE(T::Count() == n, "wrong uninitialized_copy_n");
        std::destroy_n(exec, out_first, n);

        // For move
        std::uninitialized_move(exec, first, last, out_first);
        // compare counter of constructors to length of container
        EXPECT_TRUE(T::MoveCount() == n, "wrong uninitialized_move");
        // destroy objects for testing new algorithms on same memory
        std::destroy_n(exec, out_first, n);

        std::uninitialized_move_n(exec, first, n, out_first);
        EXPECT_TRUE(T::MoveCount() == n, "wrong uninitialized_move_n");
        std::destroy_n(exec, out_first, n);
    }

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN
    template <typename InputIterator, typename OutputIterator>
    void
    operator()(pstl::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }
    template <typename InputIterator, typename OutputIterator>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::true_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        std::uninitialized_copy(exec, first, last, out_first);
        EXPECT_TRUE(IsCheckValueCorrectness(first, out_first, n), "wrong uninitialized_copy");
        std::destroy_n(exec, out_first, n);

        std::uninitialized_copy_n(exec, first, n, out_first);
        EXPECT_TRUE(IsCheckValueCorrectness(first, out_first, n), "wrong uninitialized_copy_n");
        std::destroy_n(exec, out_first, n);

        std::uninitialized_move(exec, first, last, out_first);
        EXPECT_TRUE(IsCheckValueCorrectness(first, out_first, n), "wrong uninitialized_move");
        std::destroy_n(exec, out_first, n);

        std::uninitialized_move_n(exec, first, n, out_first);
        EXPECT_TRUE(IsCheckValueCorrectness(first, out_first, n), "wrong uninitialized_move_n");
        std::destroy_n(exec, out_first, n);
    }
};

template <typename T>
void
test_uninitialized_copy_move_by_type()
{
    std::size_t N = 100000;
    for (size_t n = 0; n <= N; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> in(n, [=](size_t k) -> T { return T(k); });
        std::unique_ptr<T[]> p(new T[n]);
        invoke_on_all_policies(test_uninitialized_copy_move(), in.begin(), in.end(), p.get(), n, std::is_trivial<T>());
    }
}

int32_t
main()
{

    // for trivial types
    test_uninitialized_copy_move_by_type<int16_t>();
    test_uninitialized_copy_move_by_type<float64_t>();

    // for user-defined types
#if !_PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN && !_PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN &&   \
    !_PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN
    test_uninitialized_copy_move_by_type<Wrapper<int8_t>>();
#endif

    std::cout << done() << std::endl;
    return 0;
}
