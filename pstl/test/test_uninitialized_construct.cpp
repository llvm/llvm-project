// -*- C++ -*-
//===-- test_uninitialized_construct.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for uninitialized_default_construct, uninitialized_default_construct_n,
//           uninitialized_value_construct,   uninitialized_value_construct_n

#include "pstl_test_config.h"

#include "pstl/execution"
#include "pstl/memory"
#include "utils.h"

using namespace TestUtils;

// function of checking correctness for uninitialized.construct.value
template <typename T, typename Iterator>
bool
IsCheckValueCorrectness(Iterator begin, Iterator end)
{
    for (; begin != end; ++begin)
    {
        if (*begin != T())
        {
            return false;
        }
    }
    return true;
}

struct test_uninit_construct
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end, size_t n, /*is_trivial<T>=*/std::false_type)
    {
        typedef typename std::iterator_traits<Iterator>::value_type T;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        std::destroy_n(exec, begin, n);

        // reset counter of constructors
        T::SetCount(0);
        // run algorithm
        std::uninitialized_default_construct(exec, begin, end);
        // compare counter of constructors to length of container
        EXPECT_TRUE(T::Count() == n, "wrong uninitialized_default_construct");
        // destroy objects for testing new algorithms on same memory
        std::destroy(exec, begin, end);

        std::uninitialized_default_construct_n(exec, begin, n);
        EXPECT_TRUE(T::Count() == n, "wrong uninitialized_default_construct_n");
        std::destroy_n(exec, begin, n);

        std::uninitialized_value_construct(exec, begin, end);
        EXPECT_TRUE(T::Count() == n, "wrong uninitialized_value_construct");
        std::destroy(exec, begin, end);

        std::uninitialized_value_construct_n(exec, begin, n);
        EXPECT_TRUE(T::Count() == n, "wrong uninitialized_value_construct_n");
    }

    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end, size_t n, /*is_trivial<T>=*/std::true_type)
    {
        typedef typename std::iterator_traits<Iterator>::value_type T;

        std::uninitialized_default_construct(exec, begin, end);
        std::destroy(exec, begin, end);

        std::uninitialized_default_construct_n(exec, begin, n);
        std::destroy_n(exec, begin, n);

        std::uninitialized_value_construct(exec, begin, end);
        // check correctness for uninitialized.construct.value
        EXPECT_TRUE(IsCheckValueCorrectness<T>(begin, end), "wrong uninitialized_value_construct");
        std::destroy(exec, begin, end);

        std::uninitialized_value_construct_n(exec, begin, n);
        EXPECT_TRUE(IsCheckValueCorrectness<T>(begin, end), "wrong uninitialized_value_construct_n");
        std::destroy_n(exec, begin, n);
    }
};

template <typename T>
void
test_uninit_construct_by_type()
{
    std::size_t N = 100000;
    for (size_t n = 0; n <= N; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        std::unique_ptr<T[]> p(new T[n]);
        invoke_on_all_policies(test_uninit_construct(), p.get(), std::next(p.get(), n), n, std::is_trivial<T>());
    }
}

int32_t
main()
{

    // for user-defined types
#if !__PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN
    test_uninit_construct_by_type<Wrapper<int32_t>>();
    test_uninit_construct_by_type<Wrapper<std::vector<std::string>>>();
#endif

    // for trivial types
    test_uninit_construct_by_type<int8_t>();
    test_uninit_construct_by_type<float64_t>();

    std::cout << done() << std::endl;
    return 0;
}
