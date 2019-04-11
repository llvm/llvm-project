// -*- C++ -*-
//===-- nth_element.pass.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/pstl_test_config.h"

#ifdef PSTL_STANDALONE_TESTS
#include <algorithm>
#include <iostream>
#include "pstl/execution"
#include "pstl/algorithm"

#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/utils.h"

using namespace TestUtils;

// User defined type with minimal requirements
template <typename T>
struct DataType
{
    explicit DataType(int32_t k) : my_val(k) {}
    DataType(DataType&& input)
    {
        my_val = std::move(input.my_val);
        input.my_val = T(0);
    }
    DataType&
    operator=(DataType&& input)
    {
        my_val = std::move(input.my_val);
        input.my_val = T(0);
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

template <typename T>
bool
is_equal(const DataType<T>& x, const DataType<T>& y)
{
    return x.get_val() == y.get_val();
}

template <typename T>
bool
is_equal(const T& x, const T& y)
{
    return x == y;
}

struct test_one_policy
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN // dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Size, typename Generator1, typename Generator2, typename Compare>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
    operator()(pstl::execution::unsequenced_policy, Iterator1 first1, Iterator1 last1, Iterator1 first2,
               Iterator1 last2, Size n, Size m, Generator1 generator1, Generator2 generator2, Compare comp)
    {
    }
    template <typename Iterator1, typename Size, typename Generator1, typename Generator2, typename Compare>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
    operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 first1, Iterator1 last1, Iterator1 first2,
               Iterator1 last2, Size n, Size m, Generator1 generator1, Generator2 generator2, Compare comp)
    {
    }
#endif

    // nth_element works only with random access iterators
    template <typename Policy, typename Iterator1, typename Size, typename Generator1, typename Generator2,
              typename Compare>
    typename std::enable_if<is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator1 first2, Iterator1 last2, Size n, Size m,
               Generator1 generator1, Generator2 generator2, Compare comp)
    {

        using T = typename std::iterator_traits<Iterator1>::value_type;
        const Iterator1 mid1 = std::next(first1, m);
        const Iterator1 mid2 = std::next(first2, m);

        fill_data(first1, mid1, generator1);
        fill_data(mid1, last1, generator2);
        fill_data(first2, mid2, generator1);
        fill_data(mid2, last2, generator2);
        std::nth_element(first1, mid1, last1, comp);
        std::nth_element(exec, first2, mid2, last2, comp);
        if (m > 0 && m < n)
        {
            EXPECT_TRUE(is_equal(*mid1, *mid2), "wrong result from nth_element with predicate");
        }
        EXPECT_TRUE(std::find_first_of(first2, mid2, mid2, last2, [comp](T& x, T& y) { return comp(y, x); }) == mid2,
                    "wrong effect from nth_element with predicate");
    }

    template <typename Policy, typename Iterator1, typename Size, typename Generator1, typename Generator2,
              typename Compare>
    typename std::enable_if<!is_same_iterator_category<Iterator1, std::random_access_iterator_tag>::value, void>::type
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator1 first2, Iterator1 last2, Size n, Size m,
               Generator1 generator1, Generator2 generator2, Compare comp)
    {
    }
};

template <typename T, typename Generator1, typename Generator2, typename Compare>
void
test_by_type(Generator1 generator1, Generator2 generator2, Compare comp)
{
    using namespace std;
    size_t max_size = 10000;
    Sequence<T> in1(max_size, [](size_t v) { return T(v); });
    Sequence<T> exp(max_size, [](size_t v) { return T(v); });
    size_t m;

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        m = 0;
        invoke_on_all_policies(test_one_policy(), exp.begin(), exp.begin() + n, in1.begin(), in1.begin() + n, n, m,
                               generator1, generator2, comp);
        m = n / 7;
        invoke_on_all_policies(test_one_policy(), exp.begin(), exp.begin() + n, in1.begin(), in1.begin() + n, n, m,
                               generator1, generator2, comp);
        m = 3 * n / 5;
        invoke_on_all_policies(test_one_policy(), exp.begin(), exp.begin() + n, in1.begin(), in1.begin() + n, n, m,
                               generator1, generator2, comp);
    }
    invoke_on_all_policies(test_one_policy(), exp.begin(), exp.begin() + max_size, in1.begin(), in1.begin() + max_size,
                           max_size, max_size, generator1, generator2, comp);
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        invoke_if(exec, [&]() { nth_element(exec, iter, iter, iter, non_const(std::less<T>())); });
    }
};

int32_t
main()
{
    test_by_type<int32_t>([](int32_t i) { return 10 * i; }, [](int32_t i) { return i + 1; }, std::less<int32_t>());
    test_by_type<int32_t>([](int32_t) { return 0; }, [](int32_t) { return 0; }, std::less<int32_t>());

    test_by_type<float64_t>([](int32_t i) { return -2 * i; }, [](int32_t i) { return -(2 * i + 1); },
                            [](const float64_t x, const float64_t y) { return x > y; });

    test_by_type<DataType<float32_t>>(
        [](int32_t i) { return DataType<float32_t>(2 * i + 1); }, [](int32_t i) { return DataType<float32_t>(2 * i); },
        [](const DataType<float32_t>& x, const DataType<float32_t>& y) { return x.get_val() < y.get_val(); });

    test_algo_basic_single<int32_t>(run_for_rnd<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
