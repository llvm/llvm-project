//===- RegisterPressureTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterPressure.h"
#include "CodeGenTestBase.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/Config/Targets.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

class RegisterPressureTest : public CodeGenTestBase {
public:
  static void SetUpTestCase() {
#if LLVM_HAS_AMDGPU_TARGET
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
#else
    GTEST_SKIP();
#endif
  }

  void SetUp() override { setUpImpl("amdgpu9.50--", "", ""); }
};

/// Replicates the scheduler's effect on \p LIS on an intra-block move of \p
/// MI right before \p MoveBefore, which must be in the same block as \p MI.
static void moveMIAndAdjustLiveness(MachineBasicBlock::iterator MoveBefore,
                                    MachineInstr &MI, LiveIntervals &LIS) {
  MachineBasicBlock &MBB = *MI.getParent();
  const MachineFunction &MF = *MBB.getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  MBB.splice(MoveBefore, &MBB, MI.getIterator());
  LIS.handleMove(MI);

  RegisterOperands RegOpers;
  RegOpers.collect(MI, TRI, MRI, true, /*IgnoreDead=*/false);
  RegOpers.adjustLaneLiveness(LIS, MRI, MI);
}

/// When a scheduler move turns a live subregister def into the dead, last def
/// of its virtual register, `adjustLaneLiveness` must add a dead flag on the
/// def operand. Here %0.sub1 defines a lane that is never used: before the move
/// it is a partially-dead def sitting in the middle of %0's merged live
/// interval (legal without a dead flag), but sinking it below the use of
/// %0.sub0 makes it the interval's terminal segment, which ends on a dead slot.
TEST_F(RegisterPressureTest, MarkDeadDefAfterMoveBeyondLastUse) {
  StringRef MIRString = R"(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    undef %0.sub0:vreg_64 = IMPLICIT_DEF
    %0.sub1:vreg_64 = IMPLICIT_DEF
    %1:vgpr_32 = V_ADD_U32_e32 %0.sub0, %0.sub0, implicit $exec
    
  bb.1:
    S_NOP 0, implicit %1
    S_ENDPGM 0
...
  )";
  ASSERT_TRUE(parseMIR(MIRString));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  MachineInstr &Sub1Def = *std::next(MBB0.begin());

  // Sink %0.sub1's (dead) def to the end of bb.0, past the use of %0.sub0. This
  // replicates a scheduler move: the def becomes the last, dead def of %0.
  moveMIAndAdjustLiveness(MBB0.end(), Sub1Def, LIS);

  EXPECT_TRUE(MF.verify(&LIS, /*Indexes=*/nullptr, /*Banner=*/nullptr,
                        /*OS=*/&errs(), /*AbortOnError=*/false));
}