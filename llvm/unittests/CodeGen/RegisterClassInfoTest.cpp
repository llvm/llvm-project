//===- RegisterClassInfoTest.cpp - RegisterClassInfo tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterClassInfo.h"
#include "CodeGenTestBase.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Config/Targets.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class RegisterClassInfoTest : public CodeGenTestBase {
public:
  /// Register the AMDGPU target components needed by this test suite.
  static void SetUpTestCase() {
#if LLVM_HAS_AMDGPU_TARGET
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
#else
    GTEST_SKIP();
#endif
  }

  /// Use a GCN triple so VGPR registers are available to the test.
  void SetUp() override { setUpImpl("amdgpu9.00-amd-amdhsa", "", /*FS=*/""); }
};

/// Force every RegisterClassInfo cache entry to be populated so
/// updateReservedRegs exercises incremental compaction instead of lazy
/// recomputation on the next query.
static void materializeAll(RegisterClassInfo &RCI,
                           const TargetRegisterInfo &TRI) {
  for (const TargetRegisterClass &RC : TRI.regclasses()) {
    (void)RCI.getOrder(&RC);
    (void)RCI.getNumAllocatableRegs(&RC);
    (void)RCI.isProperSubClass(&RC);
    (void)RCI.getMinCost(&RC);
    (void)RCI.getLastCostChange(&RC);
  }

  for (unsigned I = 0; I != TRI.getNumRegPressureSets(); ++I)
    (void)RCI.getRegPressureSetLimit(I);
}

/// Compare every cached RegisterClassInfo field against a freshly built object.
static void expectEqual(RegisterClassInfo &Incremental,
                        RegisterClassInfo &Recomputed,
                        const TargetRegisterInfo &TRI) {
  for (const TargetRegisterClass &RC : TRI.regclasses()) {
    SCOPED_TRACE(TRI.getRegClassName(&RC));
    EXPECT_EQ(Incremental.getOrder(&RC), Recomputed.getOrder(&RC));
    EXPECT_EQ(Incremental.getNumAllocatableRegs(&RC),
              Recomputed.getNumAllocatableRegs(&RC));
    EXPECT_EQ(Incremental.isProperSubClass(&RC),
              Recomputed.isProperSubClass(&RC));
    EXPECT_EQ(Incremental.getMinCost(&RC), Recomputed.getMinCost(&RC));
    EXPECT_EQ(Incremental.getLastCostChange(&RC),
              Recomputed.getLastCostChange(&RC));
  }

  for (unsigned I = 0; I != TRI.getNumRegPressureSets(); ++I) {
    SCOPED_TRACE(I);
    EXPECT_EQ(Incremental.getRegPressureSetLimit(I),
              Recomputed.getRegPressureSetLimit(I));
  }
}

/// Verify that updateReservedRegs matches rebuilding RegisterClassInfo from
/// scratch.
TEST_F(RegisterClassInfoTest, IncrementalUpdateMatchesRecompute) {
  ASSERT_TRUE(parseMIR(R"MIR(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    S_ENDPGM 0
...
)MIR"));

  MachineFunction &MF = getMF("func");
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MRI.freezeReservedRegs();

  RegisterClassInfo Incremental;
  Incremental.runOnMachineFunction(MF);
  materializeAll(Incremental, TRI);

  MCRegister VGPR0 = AMDGPU::VGPR0;
  ASSERT_FALSE(MRI.isReserved(VGPR0));
  ASSERT_TRUE(
      llvm::any_of(TRI.regclasses(), [&](const TargetRegisterClass &RC) {
        return llvm::is_contained(Incremental.getOrder(&RC),
                                  static_cast<MCPhysReg>(VGPR0.id()));
      }));

  MRI.reserveReg(VGPR0, &TRI);
  for (MCRegAliasIterator Alias(VGPR0, &TRI, /*IncludeSubRegs=*/true);
       Alias.isValid(); ++Alias)
    EXPECT_TRUE(MRI.isReserved(*Alias)) << TRI.getName(*Alias);

  Incremental.updateReservedRegs(MRI.getReservedRegs());

  // Construct an independent baseline from the updated reserved-register set.
  RegisterClassInfo Recomputed;
  Recomputed.runOnMachineFunction(MF);

  expectEqual(Incremental, Recomputed, TRI);
}

} // namespace
