// -*- C++ -*-
//===-- rotate.pass.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include "support/pstl_test_config.h"

#include <iterator>
#include <execution>
#include <algorithm>

#include "support/utils.h"

using namespace TestUtils;

template <typename T>
struct wrapper
{
    T t;
    int move_count;
    explicit wrapper(T t_) : t(t_), move_count(0) {}
    wrapper&
    operator=(const T& t_)
    {
        t = t_;
        return *this;
    }

    wrapper(const wrapper<T>& a) : move_count(0) { t = a.t; }

    wrapper<T>&
    operator=(wrapper<T>& a)
    {
        t = a.t;
        return *this;
    }

    wrapper<T>&
    operator=(wrapper<T>&& a)
    {
        t = a.t;
        move_count += 1;
        return *this;
    }
};

template <typename T>
struct compare
{
    bool
    operator()(const T& a, const T& b)
    {
        return a == b;
    }
};

template <typename T>
struct compare<wrapper<T>>
{
    bool
    operator()(const wrapper<T>& a, const wrapper<T>& b)
    {
        return a.t == b.t;
    }
};
#include <typeinfo>

struct test_one_policy
{

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                             \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN // dummy specializations to skip testing in case of broken configuration
    template <typename Iterator, typename Size>
    void
    operator()(pstl::execution::unsequenced_policy, Iterator data_b, Iterator data_e, Iterator actual_b,
               Iterator actual_e, Size shift)
    {
    }
    template <typename Iterator, typename Size>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, Iterator data_b, Iterator data_e, Iterator actual_b,
               Iterator actual_e, Size shift)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator, typename Size>
    void
    operator()(ExecutionPolicy&& exec, Iterator data_b, Iterator data_e, Iterator actual_b, Iterator actual_e,
               Size shift)
    {
        using namespace std;
        using T = typename iterator_traits<Iterator>::value_type;
        Iterator actual_m = std::next(actual_b, shift);

        copy(data_b, data_e, actual_b);
        Iterator actual_return = rotate(exec, actual_b, actual_m, actual_e);

        EXPECT_TRUE(actual_return == std::next(actual_b, std::distance(actual_m, actual_e)), "wrong result of rotate");
        auto comparator = compare<T>();
        bool check = std::equal(actual_return, actual_e, data_b, comparator);
        check = check && std::equal(actual_b, actual_return, std::next(data_b, shift), comparator);

        EXPECT_TRUE(check, "wrong effect of rotate");
        EXPECT_TRUE(check_move(exec, actual_b, actual_e, shift), "wrong move test of rotate");
    }

    template <typename ExecutionPolicy, typename Iterator, typename Size>
    typename std::enable_if<
        is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value &&
            !std::is_same<ExecutionPolicy, std::execution::sequenced_policy>::value &&
            std::is_same<typename std::iterator_traits<Iterator>::value_type, wrapper<float32_t>>::value,
        bool>::type
    check_move(ExecutionPolicy&&, Iterator b, Iterator e, Size shift)
    {
        bool result = all_of(b, e, [](wrapper<float32_t>& a) {
            bool temp = a.move_count > 0;
            a.move_count = 0;
            return temp;
        });
        return shift == 0 || result;
    }

    template <typename ExecutionPolicy, typename Iterator, typename Size>
    typename std::enable_if<
        !(is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value &&
          !std::is_same<ExecutionPolicy, std::execution::sequenced_policy>::value &&
          std::is_same<typename std::iterator_traits<Iterator>::value_type, wrapper<float32_t>>::value),
        bool>::type
    check_move(ExecutionPolicy&&, Iterator, Iterator, Size)
    {
        return true;
    }
};

template <typename T>
void
test()
{
    const int32_t max_len = 100000;

    Sequence<T> actual(max_len, [](std::size_t i) { return T(i); });
    Sequence<T> data(max_len, [](std::size_t i) { return T(i); });

    for (int32_t len = 0; len < max_len; len = len <= 16 ? len + 1 : int32_t(3.1415 * len))
    {
        int32_t shifts[] = {0, 1, 2, len / 3, (2 * len) / 3, len - 1};
        for (auto shift : shifts)
        {
            if (shift >= 0 && shift < len)
            {
                invoke_on_all_policies(test_one_policy(), data.begin(), data.begin() + len, actual.begin(),
                                       actual.begin() + len, shift);
            }
        }
    }
}

int
main()
{
    test<int32_t>();
    test<wrapper<float64_t>>();
    test<MemoryChecker>();
    EXPECT_FALSE(MemoryChecker::alive_objects() < 0, "wrong effect from rotate: number of ctors calls < num of dtors calls");
    EXPECT_FALSE(MemoryChecker::alive_objects() > 0, "wrong effect from rotate: number of ctors calls > num of dtors calls");

    std::cout << done() << std::endl;
    return 0;
}
