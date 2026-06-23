//===- MacroUtilsTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's MacroUtils.h APIs.
//
// These are compile-time invariants — the static_asserts pin the contract
// of ORC_RT_DEPAREN. The trivial TEST exists so the file produces a gtest
// case to run.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/MacroUtils.h"

#include "gtest/gtest.h"

#include <tuple>
#include <type_traits>

// Multi-element list.
static_assert(
    std::is_same_v<std::tuple<ORC_RT_DEPAREN((int, double, char))>,
                   std::tuple<int, double, char>>,
    "ORC_RT_DEPAREN should strip outer parens from a multi-element list");

// Single-element list.
static_assert(
    std::is_same_v<std::tuple<ORC_RT_DEPAREN((int))>, std::tuple<int>>,
    "ORC_RT_DEPAREN should strip outer parens from a single-element list");

// Empty list.
static_assert(std::is_same_v<std::tuple<ORC_RT_DEPAREN(())>, std::tuple<>>,
              "ORC_RT_DEPAREN should produce nothing from an empty list");

TEST(MacroUtilsTest, DeParenCompiles) {
  // The real coverage is in the static_asserts above; this case exists so
  // the file contributes a runnable gtest entry.
}
