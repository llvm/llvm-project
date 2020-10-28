// -*- C++ -*-
//===-- scan.fail.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include <execution>
#include <numeric>

struct CustomPolicy
{
    constexpr std::false_type
    __allow_vector()
    {
        return std::false_type{};
    }
    constexpr std::false_type
    __allow_parallel()
    {
        return std::false_type{};
    }
} policy;

int32_t
main()
{
    int *first = nullptr, *last = nullptr, *result = nullptr;

    std::exclusive_scan(policy, first, last, result, 0); // expected-error {{no matching function for call to 'exclusive_scan'}}
    std::exclusive_scan(policy, first, last, result, 0, std::plus<int>()); // expected-error {{no matching function for call to 'exclusive_scan'}}

    return 0;
}
