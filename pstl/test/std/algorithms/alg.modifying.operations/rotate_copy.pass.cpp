// -*- C++ -*-
//===-- rotate_copy.pass.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/pstl_test_config.h"

#ifdef PSTL_STANDALONE_TESTS
#include <iterator>

#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/utils.h"

using namespace TestUtils;

template <typename T>
struct wrapper;

template <typename T>
bool
compare(const wrapper<T>& a, const wrapper<T>& b)
{
    return a.t == b.t;
}

template <typename T>
bool
compare(const T& a, const T& b)
{
    return a == b;
}

template <typename T>
struct wrapper
{
    explicit wrapper(T t_) : t(t_) {}
    wrapper&
    operator=(const T& t_)
    {
        t = t_;
        return *this;
    }
    friend bool
    compare<T>(const wrapper<T>& a, const wrapper<T>& b);

  private:
    T t;
};

template <typename T, typename It1, typename It2>
struct comparator
{
    using T1 = typename std::iterator_traits<It1>::value_type;
    using T2 = typename std::iterator_traits<It2>::value_type;
    bool
    operator()(T1 a, T2 b)
    {
        T temp = a;
        return compare(temp, b);
    }
};

struct test_one_policy
{

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN // dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
    operator()(pstl::execution::unsequenced_policy, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b,
               Iterator2 actual_e, std::size_t shift)
    {
    }
    template <typename Iterator1, typename Iterator2>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
    operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b,
               Iterator2 actual_e, std::size_t shift)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e,
               std::size_t shift)
    {
        using namespace std;
        using T = typename iterator_traits<Iterator2>::value_type;
        Iterator1 data_m = std::next(data_b, shift);

        fill(actual_b, actual_e, T(-123));
        Iterator2 actual_return = rotate_copy(exec, data_b, data_m, data_e, actual_b);

        EXPECT_TRUE(actual_return == actual_e, "wrong result of rotate_copy");
        auto comparer = comparator<T, Iterator1, Iterator2>();
        bool check = std::equal(data_m, data_e, actual_b, comparer);
        check = check && std::equal(data_b, data_m, std::next(actual_b, std::distance(data_m, data_e)), comparer);

        EXPECT_TRUE(check, "wrong effect of rotate_copy");
    }
};

template <typename T1, typename T2>
void
test()
{

    const std::size_t max_len = 100000;

    Sequence<T2> actual(max_len, [](std::size_t i) { return T1(i); });

    Sequence<T1> data(max_len, [](std::size_t i) { return T1(i); });

    for (std::size_t len = 0; len < max_len; len = len <= 16 ? len + 1 : std::size_t(3.1415 * len))
    {
        std::size_t shifts[] = {0, 1, 2, len / 3, (2 * len) / 3, len - 1};
        for (std::size_t shift : shifts)
        {
            if (shift > 0 && shift < len)
            {
                invoke_on_all_policies(test_one_policy(), data.begin(), data.begin() + len, actual.begin(),
                                       actual.begin() + len, shift);
                invoke_on_all_policies(test_one_policy(), data.cbegin(), data.cbegin() + len, actual.begin(),
                                       actual.begin() + len, shift);
            }
        }
    }
}

int32_t
main()
{
    test<int32_t, int8_t>();
    test<uint16_t, float32_t>();
    test<float64_t, int64_t>();
    test<wrapper<float64_t>, wrapper<float64_t>>();

    std::cout << done() << std::endl;
    return 0;
}
