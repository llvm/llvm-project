//===- CCStateTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

#include "MFCommon.inc"

TEST(CCStateTest, NegativeOffsets) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);

  SmallVector<CCValAssign, 8> Locs;
  CCState Info(CallingConv::C, /*IsVarArg=*/false, *MF, Locs, Ctx,
               /*NegativeOffsets=*/true);

  ASSERT_EQ(Info.AllocateStack(1, Align(1)), -1);
  ASSERT_EQ(Info.AllocateStack(1, Align(2)), -2);
  ASSERT_EQ(Info.AllocateStack(1, Align(2)), -4);
  ASSERT_EQ(Info.AllocateStack(1, Align(1)), -5);
  ASSERT_EQ(Info.AllocateStack(2, Align(2)), -8);
  ASSERT_EQ(Info.AllocateStack(2, Align(2)), -10);
  ASSERT_EQ(Info.AllocateStack(2, Align(1)), -12);
  ASSERT_EQ(Info.AllocateStack(1, Align(1)), -13);
  ASSERT_EQ(Info.getStackSize(), 13u);
  ASSERT_EQ(Info.getAlignedCallFrameSize(), 14u);
}

} // namespace
