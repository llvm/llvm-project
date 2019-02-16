// -*- C++ -*-
//===-- test_inplace_merge.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pstl_test_config.h"

#include <algorithm>
#include "pstl/execution"
#include "pstl/algorithm"

#include "utils.h"

using namespace TestUtils;

struct test_one_policy
{
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                            \
    __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN // dummy specialization by policy type, in case of broken configuration
    template <typename BiDirIt1, typename Size, typename Generator1, typename Generator2, typename Compare>
    void
    operator()(pstl::execution::unsequenced_policy, BiDirIt1 first1, BiDirIt1 last1, BiDirIt1 first2, BiDirIt1 last2,
               Size n, Size m, Generator1 generator1, Generator2 generator2, Compare comp)
    {
    }

    template <typename BiDirIt1, typename Size, typename Generator1, typename Generator2, typename Compare>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, BiDirIt1 first1, BiDirIt1 last1, BiDirIt1 first2,
               BiDirIt1 last2, Size n, Size m, Generator1 generator1, Generator2 generator2, Compare comp)
    {
    }
#endif

    // inplace_merge works with bidirectional iterators at least
    template <typename Policy, typename BiDirIt1, typename Size, typename Generator1, typename Generator2,
              typename Compare>
    typename std::enable_if<!is_same_iterator_category<BiDirIt1, std::forward_iterator_tag>::value, void>::type
    operator()(Policy&& exec, BiDirIt1 first1, BiDirIt1 last1, BiDirIt1 first2, BiDirIt1 last2, Size n, Size m,
               Generator1 generator1, Generator2 generator2, Compare comp)
    {

        using T = typename std::iterator_traits<BiDirIt1>::value_type;
        const BiDirIt1 mid1 = std::next(first1, m);
        fill_data(first1, mid1, generator1);
        fill_data(mid1, last1, generator2);

        const BiDirIt1 mid2 = std::next(first2, m);
        fill_data(first2, mid2, generator1);
        fill_data(mid2, last2, generator2);

        std::inplace_merge(first1, mid1, last1, comp);
        std::inplace_merge(exec, first2, mid2, last2, comp);
        EXPECT_EQ_N(first1, first2, n, "wrong effect from inplace_merge with predicate");
    }

    template <typename Policy, typename BiDirIt1, typename Size, typename Generator1, typename Generator2,
              typename Compare>
    typename std::enable_if<is_same_iterator_category<BiDirIt1, std::forward_iterator_tag>::value, void>::type
    operator()(Policy&& exec, BiDirIt1 first1, BiDirIt1 last1, BiDirIt1 first2, BiDirIt1 last2, Size n, Size m,
               Generator1 generator1, Generator2 generator2, Compare comp)
    {
    }
};

template <typename T, typename Generator1, typename Generator2, typename Compare>
void
test_by_type(Generator1 generator1, Generator2 generator2, Compare comp)
{
    using namespace std;
    size_t max_size = 100000;
    Sequence<T> in1(max_size, [](size_t v) { return T(v); });
    Sequence<T> exp(max_size, [](size_t v) { return T(v); });
    size_t m;

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        m = 0;
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                               generator1, generator2, comp);

        m = n / 3;
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                               generator1, generator2, comp);

        m = 2 * n / 3;
        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                               generator1, generator2, comp);
    }
}

template <typename T>
struct LocalWrapper
{
    explicit LocalWrapper(int32_t k) : my_val(k) {}
    LocalWrapper(LocalWrapper&& input) { my_val = std::move(input.my_val); }
    LocalWrapper&
    operator=(LocalWrapper&& input)
    {
        my_val = std::move(input.my_val);
        return *this;
    }
    bool
    operator<(const LocalWrapper<T>& w) const
    {
        return my_val < w.my_val;
    }
    friend bool
    operator==(const LocalWrapper<T>& x, const LocalWrapper<T>& y)
    {
        return x.my_val == y.my_val;
    }
    friend std::ostream&
    operator<<(std::ostream& stream, const LocalWrapper<T>& input)
    {
        return stream << input.my_val;
    }

  private:
    T my_val;
};

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        invoke_if(exec, [&]() { inplace_merge(exec, iter, iter, iter, non_const(std::less<T>())); });
    }
};

int32_t
main()
{
    test_by_type<float64_t>([](int32_t i) { return -2 * i; }, [](int32_t i) { return -(2 * i + 1); },
                            [](const float64_t x, const float64_t y) { return x > y; });

    test_by_type<int32_t>([](int32_t i) { return 10 * i; }, [](int32_t i) { return i + 1; }, std::less<int32_t>());

    test_by_type<LocalWrapper<float32_t>>([](int32_t i) { return LocalWrapper<float32_t>(2 * i + 1); },
                                          [](int32_t i) { return LocalWrapper<float32_t>(2 * i); },
                                          std::less<LocalWrapper<float32_t>>());

    test_algo_basic_single<int32_t>(run_for_rnd_bi<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
