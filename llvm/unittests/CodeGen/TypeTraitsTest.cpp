//===- llvm/unittest/CodeGen/TypeTraitsTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace llvm;

#if __has_feature(is_trivially_copyable) || (defined(__GNUC__) && __GNUC__ >= 5)
static_assert(std::is_trivially_copyable_v<PressureChange>,
              "trivially copyable");
static_assert(std::is_trivially_copyable_v<SDep>, "trivially copyable");
static_assert(std::is_trivially_copyable_v<SDValue>, "trivially copyable");
static_assert(std::is_trivially_copyable_v<SlotIndex>, "trivially copyable");
static_assert(std::is_trivially_copyable_v<IdentifyingPassPtr>,
              "trivially copyable");
#endif

