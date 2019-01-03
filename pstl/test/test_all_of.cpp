// -*- C++ -*-
//===-- test_all_of.cpp ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pstl_test_config.h"

#include "pstl/execution"
#include "pstl/algorithm"
#include "utils.h"

/*
  TODO: consider implementing the following tests for a better code coverage
  - correctness
  - bad input argument (if applicable)
  - data corruption around/of input and output
  - correctly work with nested parallelism
  - check that algorithm does not require anything more than is described in its requirements section
*/

using namespace TestUtils;

struct test_all_of
{
    template <typename ExecutionPolicy, typename Iterator, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator begin, Iterator end, Predicate pred, bool expected)
    {

        auto actualr = std::all_of(exec, begin, end, pred);
        EXPECT_EQ(expected, actualr, "result for all_of");
    }
};

template <typename T>
struct Parity
{
    bool parity;

  public:
    Parity(bool parity_) : parity(parity_) {}
    bool
    operator()(T value) const
    {
        return (size_t(value) ^ parity) % 2 == 0;
    }
};

template <typename T>
void
test(size_t bits)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {

        // Sequence of odd values
        Sequence<T> in(n, [n, bits](size_t k) { return T(2 * HashBits(n, bits - 1) ^ 1); });

        // Even value, or false when T is bool.
        T spike(2 * HashBits(n, bits - 1));
        Sequence<T> inCopy(in);

        invoke_on_all_policies(test_all_of(), in.begin(), in.end(), Parity<T>(1), true);
        invoke_on_all_policies(test_all_of(), in.cbegin(), in.cend(), Parity<T>(1), true);
        EXPECT_EQ(in, inCopy, "all_of modified input sequence");
        if (n > 0)
        {
            // Sprinkle in a miss
            in[2 * n / 3] = spike;
            invoke_on_all_policies(test_all_of(), in.begin(), in.end(), Parity<T>(1), false);
            invoke_on_all_policies(test_all_of(), in.cbegin(), in.cend(), Parity<T>(1), false);

            // Sprinkle in a few more misses
            in[n / 2] = spike;
            in[n / 3] = spike;
            invoke_on_all_policies(test_all_of(), in.begin(), in.end(), Parity<T>(1), false);
            invoke_on_all_policies(test_all_of(), in.cbegin(), in.cend(), Parity<T>(1), false);
        }
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
        all_of(exec, iter, iter, non_const(is_even));
    }
};

int32_t
main()
{
    test<int32_t>(8 * sizeof(int32_t));
    test<uint16_t>(8 * sizeof(uint16_t));
    test<float64_t>(53);
#if !__PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN
    test<bool>(1);
#endif

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const>());

    std::cout << done() << std::endl;
    return 0;
}
