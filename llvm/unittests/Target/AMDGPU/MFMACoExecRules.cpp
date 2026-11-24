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

#include "GCNHazardRecognizer.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

#define GET_AVAILABLE_OPCODE_CHECKER
#include "AMDGPUGenInstrInfo.inc"

using namespace llvm;
using namespace llvm::AMDGPU;

class AMDGPUMFMACoExecRules : public AMDGPUTestBase {};

TEST_F(AMDGPUMFMACoExecRules, GFX950) {
  std::unique_ptr<GCNTargetMachine> TM =
      createAMDGPUTargetMachine(Triple("amdgpu-amd-amdhsa"), "gfx950", "");
  auto ST =
      std::make_unique<GCNSubtarget>(TM->getTargetTriple(), TM->getTargetCPU(),
                                     TM->getTargetFeatureString(), *TM);

  const MCInstrInfo *MCII = TM->getMCInstrInfo();

  for (unsigned Op = 0; Op < MCII->getNumOpcodes(); ++Op) {
    if (!ST->getInstrInfo()->isMFMAorWMMA(Op))
      continue;

    // Filter out pre-gfx950 MFMA instructions
    FeatureBitset Required = AMDGPU_MC::computeRequiredFeatures(Op);
    if (!Required.test(AMDGPU_MC::Feature_HasGFX950InstsBit))
      continue;

    CoExecInfo Info = getMFMACoExecInfo(Op, *ST);
    // None of gfx950 MFMAs can co-execute anything in their issue slot and our
    // fallback for unrecognized MFMA instructions is to allow everything in
    // every slot, so here we check that we did not encounter the fallback
    // definition of co-exec rules.
    EXPECT_EQ(Info.Slots[0].Mask, CoExecMask::None) << MCII->getName(Op);
  }
}

TEST_F(AMDGPUMFMACoExecRules, AllMFMAsHaveVariant) {
  for (StringRef TT : {"amdgpu9.42-amd-amdhsa", "amdgpu9.50-amd-amdhsa"}) {
    std::unique_ptr<GCNTargetMachine> TM =
        createAMDGPUTargetMachine(Triple(TT), "", "");
    std::unique_ptr<GCNSubtarget> ST = std::make_unique<GCNSubtarget>(
        TM->getTargetTriple(), TM->getTargetCPU(), TM->getTargetFeatureString(),
        *TM);
    const MCInstrInfo *MCII = TM->getMCInstrInfo();

    for (unsigned Op = 0; Op < MCII->getNumOpcodes(); ++Op) {
      if (!ST->getInstrInfo()->isMFMA(Op) ||
          !AMDGPU_MC::isOpcodeAvailable(Op, ST->getFeatureBits()))
        continue;
      const MAIInstInfo *Info = getMAIInstInfoHelper(Op);
      EXPECT_TRUE(Info && Info->Variant != MFMAVariant::Unknown)
          << TT.str() << ": " << MCII->getName(Op).str();
    }
  }
}

static unsigned getFirstStage(const CoExecInfo &Info, CoExecMaskT Mask) {
  for (unsigned I = 0; I < Info.TotalWindow; ++I) {
    if ((Info.Slots[I].Mask & Mask) == Mask)
      return I;
  }
  return Info.TotalWindow;
}

TEST_F(AMDGPUMFMACoExecRules, StartStages) {
  using Stages = std::array<unsigned, 6>;
  struct Case {
    const char *CPU;
    unsigned Opcode;
    Stages Expected;
  };
  const Case Cases[] = {
      {"gfx950", V_MFMA_F32_16X16X32_BF16_e64, {1, 2, 2, 4, 7, 8}},
      {"gfx950", V_MFMA_F32_32X32X16_BF16_e64, {1, 2, 2, 8, 11, 12}},
      {"gfx950", V_MFMA_F32_16X16X128_F8F6F4_f8_f8_e64, {1, 2, 3, 8, 11, 12}},
      {"gfx950", V_MFMA_F32_32X32X64_F8F6F4_f8_f8_e64, {1, 2, 3, 16, 19, 20}},
      {"gfx950", V_SMFMAC_F32_16X16X64_F16_e64, {1, 2, 3, 4, 7, 8}},
      {"gfx950", V_SMFMAC_F32_32X32X32_F16_e64, {1, 3, 3, 8, 11, 12}},
      {"gfx950", V_MFMA_F64_16X16X4F64_e64, {1, 1, 18, 16, 18, 19}},
      {"gfx942", V_MFMA_F64_16X16X4F64_e64, {1, 1, 10, 8, 10, 11}},
      {"gfx950", V_MFMA_F32_4X4X4F16_e64, {1, 1, 1, 2, 4, 5}},
      {"gfx950", V_MFMA_F32_16X16X16F16_e64, {1, 1, 1, 4, 6, 7}},
      {"gfx950", V_MFMA_F32_32X32X8F16_e64, {1, 1, 1, 8, 10, 11}},
      {"gfx950", V_MFMA_F32_32X32X4F16_e64, {1, 1, 1, 16, 18, 19}},
      {"gfx950", V_SMFMAC_F32_16X16X32_F16_e64, {1, 2, 2, 4, 6, 7}},
      {"gfx950", V_SMFMAC_F32_32X32X16_F16_e64, {1, 2, 2, 8, 10, 11}},
      {"gfx950", V_MFMA_F32_4X4X1F32_e64, {1, 1, 4, 2, 4, 5}},
      {"gfx950", V_MFMA_F32_16X16X4F32_e64, {1, 1, 10, 8, 10, 11}},
      {"gfx950", V_MFMA_F32_32X32X2F32_e64, {1, 1, 18, 16, 18, 19}},
      {"gfx950", V_MFMA_F64_4X4X4F64_e64, {1, 1, 6, 4, 6, 7}},
  };
  for (const Case &C : Cases) {
    std::unique_ptr<GCNTargetMachine> TM =
        createAMDGPUTargetMachine(Triple("amdgpu-amd-amdhsa"), C.CPU, "");
    GCNSubtarget ST(TM->getTargetTriple(), TM->getTargetCPU(),
                    TM->getTargetFeatureString(), *TM);
    CoExecInfo Info = getMFMACoExecInfo(C.Opcode, ST);
    Stages Actual = {getFirstStage(Info, CoExecMask::SALU),
                     getFirstStage(Info, CoExecMask::DS | CoExecMask::VMEM),
                     getFirstStage(Info, CoExecMask::VALU),
                     getFirstStage(Info, CoExecMask::WMMA),
                     getFirstStage(Info, CoExecMask::PK_F32),
                     Info.TotalWindow};
    EXPECT_EQ(Actual, C.Expected)
        << C.CPU << ": " << TM->getMCInstrInfo()->getName(C.Opcode).str();
  }
}

TEST_F(AMDGPUMFMACoExecRules, CoExecMaskClasses) {
  std::unique_ptr<GCNTargetMachine> TM =
      createAMDGPUTargetMachine(Triple("amdgpu-amd-amdhsa"), "gfx950", "");
  GCNSubtarget ST(TM->getTargetTriple(), TM->getTargetCPU(),
                  TM->getTargetFeatureString(), *TM);
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  Mod.setDataLayout(TM->getTargetTriple().computeDataLayout());
  Function *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                                 GlobalValue::ExternalLinkage, "Test", &Mod);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  const SIInstrInfo &TII = *ST.getInstrInfo();

  auto Check = [&](ArrayRef<unsigned> Opcs, CoExecMaskT Expected) {
    for (unsigned Opc : Opcs) {
      const MCInstrDesc &Desc = TII.get(Opc);
      MachineInstr *MI = MF.CreateMachineInstr(Desc, DebugLoc());
      for (auto [Idx, Info] : enumerate(Desc.operands())) {
        if (Info.RegClass == -1)
          MI->addOperand(MF, MachineOperand::CreateImm(0));
        else
          MI->addOperand(MF, MachineOperand::CreateReg(Register(), Idx < 1));
      }
      EXPECT_EQ(GCNHazardRecognizer::getCoExecMaskForMI(*MI, TII), Expected)
          << TII.getName(Opc);
    }
  };
  Check({V_PK_ADD_F32, V_PK_FMA_F32, V_PK_MOV_B32, V_PK_MUL_F32},
        CoExecMask::PK_F32);
  Check({V_PK_ADD_F16}, CoExecMask::VALU);
}
