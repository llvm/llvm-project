// -*- C++ -*-
//===-- test_adjacent_difference.cpp --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pstl_test_config.h"

#include <iterator>

#include "pstl/execution"
#include "pstl/algorithm"
#include "pstl/numeric"
#include "utils.h"

using namespace TestUtils;

template <typename T>
struct wrapper
{
    T t;
    explicit wrapper(T t_) : t(t_) {}
    template <typename T2>
    wrapper(const wrapper<T2>& a)
    {
        t = a.t;
    }
    template <typename T2>
    void
    operator=(const wrapper<T2>& a)
    {
        t = a.t;
    }
    wrapper<T>
    operator-(const wrapper<T>& a) const
    {
        return wrapper<T>(t - a.t);
    }
};

template <typename T>
bool
compare(const T& a, const T& b)
{
    return a == b;
}

template <typename T>
bool
compare(const wrapper<T>& a, const wrapper<T>& b)
{
    return a.t == b.t;
}

template <typename Iterator1, typename Iterator2, typename T, typename Function>
typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
compute_and_check(Iterator1 first, Iterator1 last, Iterator2 d_first, T, Function f)
{
    using T2 = typename std::iterator_traits<Iterator2>::value_type;

    if (first == last)
        return true;

    T2 temp(*first);
    if (!compare(temp, *d_first))
        return false;
    Iterator1 second = std::next(first);

    ++d_first;
    for (; second != last; ++first, ++second, ++d_first)
    {
        T2 temp(f(*second, *first));
        if (!compare(temp, *d_first))
            return false;
    }

    return true;
}

// we don't want to check equality here
// because we can't be sure it will be strictly equal for floating point types
template <typename Iterator1, typename Iterator2, typename T, typename Function>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
compute_and_check(Iterator1 first, Iterator1 last, Iterator2 d_first, T, Function)
{
    return true;
}

struct test_one_policy
{
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN // dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2, typename T, typename Function>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
    operator()(pstl::execution::unsequenced_policy, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b,
               Iterator2 actual_e, T trash, Function f)
    {
    }
    template <typename Iterator1, typename Iterator2, typename T, typename Function>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
    operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b,
               Iterator2 actual_e, T trash, Function f)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T, typename Function>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e,
               T trash, Function f)
    {
        using namespace std;
        using T2 = typename std::iterator_traits<Iterator1>::value_type;

        fill(actual_b, actual_e, trash);

        Iterator2 actual_return = adjacent_difference(exec, data_b, data_e, actual_b);
        EXPECT_TRUE(compute_and_check(data_b, data_e, actual_b, T2(0), std::minus<T2>()),
                    "wrong effect of adjacent_difference");
        EXPECT_TRUE(actual_return == actual_e, "wrong result of adjacent_difference");

        fill(actual_b, actual_e, trash);

        actual_return = adjacent_difference(exec, data_b, data_e, actual_b, f);
        EXPECT_TRUE(compute_and_check(data_b, data_e, actual_b, T2(0), f),
                    "wrong effect of adjacent_difference with functor");
        EXPECT_TRUE(actual_return == actual_e, "wrong result of adjacent_difference with functor");
    }
};

template <typename T1, typename T2, typename Pred>
void
test(Pred pred)
{
    typedef typename Sequence<T2>::iterator iterator_type;

    const std::size_t max_len = 100000;

    const T2 value = T2(77);
    const T1 trash = T1(31);

    Sequence<T1> actual(max_len, [](std::size_t i) { return T1(i); });

    Sequence<T2> data(max_len, [&value](std::size_t i) { return i % 3 == 2 ? T2(i * i) : value; });

    for (std::size_t len = 0; len < max_len; len = len <= 16 ? len + 1 : std::size_t(3.1415 * len))
    {
        invoke_on_all_policies(test_one_policy(), data.begin(), data.begin() + len, actual.begin(),
                               actual.begin() + len, trash, pred);
        invoke_on_all_policies(test_one_policy(), data.cbegin(), data.cbegin() + len, actual.begin(),
                               actual.begin() + len, trash, pred);
    }
}

int32_t
main()
{
    test<uint8_t, uint32_t>([](uint32_t a, uint32_t b) { return a - b; });
    test<int32_t, int64_t>([](int64_t a, int64_t b) { return a / (b + 1); });
    test<int64_t, float32_t>([](float32_t a, float32_t b) { return (a + b) / 2; });
    test<wrapper<int32_t>, wrapper<int64_t>>(
        [](const wrapper<int64_t>& a, const wrapper<int64_t>& b) { return a - b; });

    std::cout << done() << std::endl;
    return 0;
}
