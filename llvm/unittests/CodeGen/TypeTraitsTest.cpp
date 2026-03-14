//===- llvm/unittest/CodeGen/TypeTraitsTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RDFRegisters.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "gtest/gtest.h"
#include <functional>
#include <type_traits>
#include <utility>

using namespace llvm;

static_assert(std::is_trivially_copyable_v<PressureChange>,
              "trivially copyable");
static_assert(std::is_trivially_copyable_v<SDep>, "trivially copyable");
static_assert(std::is_trivially_copyable_v<SDValue>, "trivially copyable");
static_assert(std::is_trivially_copyable_v<SlotIndex>, "trivially copyable");
static_assert(std::is_trivially_copyable_v<IdentifyingPassPtr>,
              "trivially copyable");

// https://llvm.org/PR105169
// Verify that we won't accidently specialize std::less and std::equal_to in a
// wrong way.
// C++17 [namespace.std]/2, C++20/23 [namespace.std]/5:
//   A program may explicitly instantiate a template defined in the standard
//   library only if the declaration
//   - depends on the name of a user-defined type and
//   - the instantiation meets the standard library requirements for the
//   original template.
template <class Fn> constexpr bool CheckStdCmpRequirements() {
  // std::less and std::equal_to are literal, default constructible, and
  // copyable classes.
  Fn f1{};
  auto f2 = f1;
  auto f3 = std::move(f2);
  f2 = f3;
  f2 = std::move(f3);

  // Properties held on all known implementations, although not guaranteed by
  // the standard.
  static_assert(std::is_empty_v<Fn>);
  static_assert(std::is_trivially_default_constructible_v<Fn>);
  static_assert(std::is_trivially_copyable_v<Fn>);

  return true;
}

static_assert(CheckStdCmpRequirements<std::less<rdf::RegisterRef>>(),
              "same as the original template");
static_assert(CheckStdCmpRequirements<std::equal_to<rdf::RegisterRef>>(),
              "same as the original template");
