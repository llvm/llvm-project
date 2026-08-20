//===- llvm/unittest/CodeGen/AMDGPUMetadataTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Test that MFMA co-exec rules are defined for all gfx950 MFMA instructions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "gtest/gtest.h"

#include "AMDGPUCoExecInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/TargetParser/SubtargetFeature.h"

#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunction.h"

#define GET_AVAILABLE_OPCODE_CHECKER
#include "AMDGPUGenInstrInfo.inc"

using namespace llvm;
using namespace llvm::AMDGPU;

class AMDGPUMFMACoExecRules : public AMDGPUTestBase {};

TEST_F(AMDGPUMFMACoExecRules, Basic) {
  std::unique_ptr<GCNTargetMachine> TM =
      createAMDGPUTargetMachine(Triple("amdgpu-amd-amdhsa"), "gfx950", "");
  ASSERT_NE(TM, nullptr);
  auto ST =
      std::make_unique<GCNSubtarget>(TM->getTargetTriple(), TM->getTargetCPU(),
                                     TM->getTargetFeatureString(), *TM);
  ASSERT_NE(ST, nullptr);

  const MCInstrInfo *MCII = TM->getMCInstrInfo();

  // Get feature bits for gfx950
  const FeatureBitset &Features950 = ST->getFeatureBits();
  FeatureBitset Available950 = AMDGPU_MC::computeAvailableFeatures(Features950);

  for (unsigned Op = 0; Op < MCII->getNumOpcodes(); ++Op) {
    ASSERT_NE(ST->getInstrInfo(), nullptr);
    if (!ST->getInstrInfo()->isMFMAorWMMA(Op))
      continue;
    if (MCII->get(Op).isPseudo())
      continue;

    FeatureBitset Required = AMDGPU_MC::computeRequiredFeatures(Op);
    if (!Required.test(AMDGPU_MC::Feature_HasGFX950InstsBit))
      continue;

    FeatureBitset Missing = (Available950 & Required) ^ Required;
    bool AvailableOnGfx950 = Missing.none();

    CoExecInfo Info = getMFMACoExecInfo(Op);
    if (AvailableOnGfx950) {
      EXPECT_EQ(Info.Slots[0].Mask, CoExecMask::None) << MCII->getName(Op);
    } else {
      // Ignore for now/not part of this test
    }
  }
}
