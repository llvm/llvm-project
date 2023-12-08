// -*- C++ -*-
//===-- transform_binary.pass.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include "support/pstl_test_config.h"

#include <execution>
#include <algorithm>

#include "support/utils.h"

using namespace TestUtils;

template <typename In1, typename In2, typename Out>
class TheOperation
{
    Out val;

  public:
    TheOperation(Out v) : val(v) {}
    Out
    operator()(const In1& x, const In2& y) const
    {
        return Out(val + x - y);
    }
};

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
void
check_and_reset(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator out_first)
{
    typedef typename std::iterator_traits<OutputIterator>::value_type Out;
    typename std::iterator_traits<OutputIterator>::difference_type k = 0;
    for (; first1 != last1; ++first1, ++first2, ++out_first, ++k)
    {
        // check
        Out expected = Out(1.5) + *first1 - *first2;
        Out actual = *out_first;
        if (std::is_floating_point<Out>::value)
        {
            EXPECT_TRUE((expected > actual ? expected - actual : actual - expected) < 1e7,
                        "wrong value in output sequence");
        }
        else
        {
            EXPECT_EQ(expected, actual, "wrong value in output sequence");
        }
        // reset
        *out_first = k % 7 != 4 ? 7 * k - 5 : 0;
    }
}

struct test_one_policy
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
              typename BinaryOp>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2,
               OutputIterator out_first, OutputIterator, BinaryOp op)
    {
        auto result = std::transform(exec, first1, last1, first2, out_first, op);
        (void)result;
        check_and_reset(first1, last1, first2, out_first);
    }
};

template <typename In1, typename In2, typename Out, typename Predicate>
void
test(Predicate pred)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In1> in1(n, [](size_t k) { return k % 5 != 1 ? 3 * k - 7 : 0; });
        Sequence<In2> in2(n, [](size_t k) { return k % 7 != 2 ? 5 * k - 5 : 0; });

        Sequence<Out> out(n, [](size_t) { return -1; });

        invoke_on_all_policies(test_one_policy(), in1.begin(), in1.end(), in2.begin(), in2.end(), out.begin(),
                               out.end(), pred);
        invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cend(), in2.cbegin(), in2.cend(), out.begin(),
                               out.end(), pred);
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        invoke_if(exec, [&]() {
            InputIterator input_iter2 = input_iter;
            transform(exec, input_iter, input_iter, input_iter2, out_iter, non_const(std::plus<T>()));
        });
    }
};

int
main()
{
    //const operator()
    test<int32_t, int32_t, int32_t>(TheOperation<int32_t, int32_t, int32_t>(1));
    test<float32_t, float32_t, float32_t>(TheOperation<float32_t, float32_t, float32_t>(1.5));
    //non-const operator()
    test<int32_t, float32_t, float32_t>(non_const(TheOperation<int32_t, float32_t, float32_t>(1.5)));
    test<int64_t, float64_t, float32_t>(non_const(TheOperation<int64_t, float64_t, float32_t>(1.5)));
    //lambda
    test<int8_t, float64_t, int8_t>([](const int8_t& x, const float64_t& y) { return int8_t(int8_t(1.5) + x - y); });

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
