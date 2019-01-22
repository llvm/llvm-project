//===- llvm/unittest/CodeGen/TypeTraitsTest.cpp --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "gtest/gtest.h"

using namespace llvm;

#if __has_feature(is_trivially_copyable) || (defined(__GNUC__) && __GNUC__ >= 5)
static_assert(is_trivially_copyable<PressureChange>::value, "trivially copyable");
static_assert(is_trivially_copyable<SDep>::value, "trivially copyable");
static_assert(is_trivially_copyable<SDValue>::value, "trivially copyable");
static_assert(is_trivially_copyable<SlotIndex>::value, "trivially copyable");
static_assert(is_trivially_copyable<IdentifyingPassPtr>::value, "trivially copyable");
#endif

