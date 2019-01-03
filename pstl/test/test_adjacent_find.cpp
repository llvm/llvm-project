// -*- C++ -*-
//===-- test_adjacent_find.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Tests for adjacent_find

#include "pstl/execution"
#include "pstl/algorithm"
#include "utils.h"

using namespace TestUtils;

struct test_adjacent_find
{
    template <typename Policy, typename Iterator, typename Pred>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Pred pred)
    {
        using namespace std;

        auto k = std::adjacent_find(first, last, pred);
        auto i = adjacent_find(exec, first, last, pred);
        EXPECT_TRUE(i == k, "wrong return value from adjacent_find with predicate");

        i = adjacent_find(exec, first, last);
        EXPECT_TRUE(i == k, "wrong return value from adjacent_find without predicate");
    }
};

template <typename T>
void
test_adjacent_find_by_type()
{

    size_t counts[] = {2, 3, 500};
    for (int32_t c = 0; c < const_size(counts); ++c)
    {

        for (int32_t e = 0; e < (counts[c] >= 64 ? 64 : (counts[c] == 2 ? 1 : 2)); ++e)
        {
            Sequence<T> in(counts[c], [](int32_t v) -> T { return T(v); }); //fill 0...n
            in[e] = in[e + 1] = -1;                                         //make an adjacent pair

            auto i = std::adjacent_find(in.cbegin(), in.cend(), std::equal_to<T>());
            EXPECT_TRUE(i == in.cbegin() + e, "std::adjacent_find returned wrong result");

            invoke_on_all_policies(test_adjacent_find(), in.begin(), in.end(), std::equal_to<T>());
            invoke_on_all_policies(test_adjacent_find(), in.cbegin(), in.cend(), std::equal_to<T>());
        }
    }

    //special cases: size=0, size=1;
    for (int32_t expect = 0; expect < 1; ++expect)
    {
        Sequence<T> in(expect, [](int32_t v) -> T { return T(v); }); //fill 0...n
        auto i = std::adjacent_find(in.cbegin(), in.cend(), std::equal_to<T>());
        EXPECT_TRUE(i == in.cbegin() + expect, "std::adjacent_find returned wrong result");

        invoke_on_all_policies(test_adjacent_find(), in.begin(), in.end(), std::equal_to<T>());
        invoke_on_all_policies(test_adjacent_find(), in.cbegin(), in.cend(), std::equal_to<T>());
    }

    //special cases:
    Sequence<T> a1 = {5, 5, 5, 6, 7, 8, 9};
    invoke_on_all_policies(test_adjacent_find(), a1.begin(), a1.end(), std::equal_to<T>());
    invoke_on_all_policies(test_adjacent_find(), a1.begin() + 1, a1.end(), std::equal_to<T>());

    invoke_on_all_policies(test_adjacent_find(), a1.cbegin(), a1.cend(), std::equal_to<T>());
    invoke_on_all_policies(test_adjacent_find(), a1.cbegin() + 1, a1.cend(), std::equal_to<T>());

    Sequence<T> a2 = {5, 6, 7, 8, 9, 9};
    invoke_on_all_policies(test_adjacent_find(), a2.begin(), a2.end(), std::equal_to<T>());
    invoke_on_all_policies(test_adjacent_find(), a2.begin(), a2.end() - 1, std::equal_to<T>());

    invoke_on_all_policies(test_adjacent_find(), a2.cbegin(), a2.cend(), std::equal_to<T>());
    invoke_on_all_policies(test_adjacent_find(), a2.cbegin(), a2.cend() - 1, std::equal_to<T>());

    Sequence<T> a3 = {5, 6, 6, 6, 7, 9, 9, 9, 9};
    invoke_on_all_policies(test_adjacent_find(), a3.begin(), a3.end(), std::equal_to<T>());

    invoke_on_all_policies(test_adjacent_find(), a3.cbegin(), a3.cend(), std::equal_to<T>());
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        adjacent_find(exec, iter, iter, non_const(std::equal_to<T>()));
    }
};

int32_t
main()
{

    test_adjacent_find_by_type<int32_t>();
    test_adjacent_find_by_type<float64_t>();

    test_algo_basic_single<int32_t>(run_for_rnd_bi<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
