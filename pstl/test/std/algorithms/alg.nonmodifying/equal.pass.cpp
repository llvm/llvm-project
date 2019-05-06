// -*- C++ -*-
//===-- equal.pass.cpp ----------------------------------------------------===//
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

#define CPP14_ENABLED 0

struct UserType
{
    float32_t f;
    float64_t d;
    int32_t i;
    size_t key;

    bool
    operator()(UserType a, UserType b)
    {
        return a.key < b.key;
    }
    bool
    operator<(UserType a)
    {
        return a.key < key;
    }
    bool
    operator>=(UserType a)
    {
        return a.key <= key;
    }
    bool
    operator<=(UserType a)
    {
        return a.key >= key;
    }
    bool
    operator==(UserType a)
    {
        return a.key == key;
    }
    bool
    operator==(UserType a) const
    {
        return a.key == key;
    }
    bool
    operator!=(UserType a)
    {
        return a.key != key;
    }
    UserType operator!()
    {
        UserType tmp;
        tmp.key = !key;
        return tmp;
    }
    friend std::ostream&
    operator<<(std::ostream& stream, const UserType a)
    {
        stream << a.key;
        return stream;
    }

    UserType() : key(-1), f(0.0f), d(0.0), i(0) {}
    UserType(size_t Number) : key(Number), f(0.0f), d(0.0), i(0) {}
    UserType&
    operator=(const UserType& other)
    {
        key = other.key;
        return *this;
    }
    UserType(const UserType& other) : key(other.key), f(other.f), d(other.d), i(other.i) {}
    UserType(UserType&& other) : key(other.key), f(other.f), d(other.d), i(other.i)
    {
        other.key = -1;
        other.f = 0.0f;
        other.d = 0.0;
        other.i = 0;
    }
};

struct test_one_policy
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, bool is_true_equal)
    {
        using namespace std;

        auto expected = equal(first1, last1, first2);
        auto actual = equal(exec, first1, last1, first2);
        EXPECT_EQ(expected, actual, "result for equal for random-access iterator, checking against std::equal()");

        // testing bool
        EXPECT_TRUE(is_true_equal == actual, "result for equal for random-access iterator, bool");

//add C++14 equal symantics tests
//add more cases for inCopy size less than in
#if CPP14_ENABLED
        auto actualr14 = std::equal(in.cbegin(), in.cend(), inCopy.cbegin(), inCopy.cend());
        EXPECT_EQ(expected, actualr14, "result for equal for random-access iterator");
#endif
    }
};

template <typename T>
void
test(size_t bits)
{
    for (size_t n = 1; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {

        // Sequence of odd values
        Sequence<T> in(n, [bits](size_t k) { return T(2 * HashBits(k, bits - 1) ^ 1); });
        Sequence<T> inCopy(in);

        invoke_on_all_policies(test_one_policy(), in.begin(), in.end(), inCopy.begin(), true);
        invoke_on_all_policies(test_one_policy(), in.cbegin(), in.cend(), inCopy.cbegin(), true);

        // testing bool !equal()
        inCopy[0] = !inCopy[0];
        invoke_on_all_policies(test_one_policy(), in.begin(), in.end(), inCopy.begin(), false);
        invoke_on_all_policies(test_one_policy(), in.cbegin(), in.cend(), inCopy.cbegin(), false);
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        equal(exec, first_iter, first_iter, second_iter, second_iter, non_const(std::equal_to<T>()));
    }
};

int32_t
main()
{

    test<int32_t>(8 * sizeof(int32_t));
    test<uint16_t>(8 * sizeof(uint16_t));
    test<float64_t>(53);
#if !_PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN
    test<bool>(1);
#endif
    test<UserType>(256);

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
