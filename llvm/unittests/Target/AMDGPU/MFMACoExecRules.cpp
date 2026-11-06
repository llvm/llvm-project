//===- llvm/unittests/Target/AMDGPU/MFMACoExecRules.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Test MFMA co-exec rules.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "gtest/gtest.h"

#include "AMDGPUCoExecInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/TargetParser/SubtargetFeature.h"

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

    CoExecInfo Info = getMFMACoExecInfo(Op);
    // None of gfx950 MFMAs can co-execute anything in their issue slot and our
    // fallback for unrecognized MFMA instructions is to allow everything in
    // every slot, so here we check that we did not encounter the fallback
    // definition of co-exec rules.
    EXPECT_EQ(Info.Slots[0].Mask, CoExecMask::None) << MCII->getName(Op);
  }
}

namespace {
class MFMACoExecTest {
public:
  explicit MFMACoExecTest(std::unique_ptr<GCNTargetMachine> TheTM)
      : TM(std::move(TheTM)),
        ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
           std::string(TM->getTargetFeatureString()), *TM),
        Mod("Module", Ctx), MMI(TM.get()) {
    Mod.setDataLayout(TM->getTargetTriple().computeDataLayout());
    auto *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                               GlobalValue::ExternalLinkage, "Test", &Mod);
    MF = std::make_unique<MachineFunction>(*F, *TM, ST, MMI.getContext(), 0);
  }

  const SIInstrInfo &getInstrInfo() const { return *ST.getInstrInfo(); }

  // Builds an instruction with placeholder operands.
  MachineInstr &buildMI(unsigned Opc) {
    const MCInstrDesc &Desc = getInstrInfo().get(Opc);
    MachineInstr *MI = MF->CreateMachineInstr(Desc, DebugLoc());
    for (auto [Idx, Info] : enumerate(Desc.operands())) {
      if (Info.RegClass == -1)
        MI->addOperand(*MF, MachineOperand::CreateImm(0));
      else
        MI->addOperand(*MF, MachineOperand::CreateReg(Register(), Idx < 1));
    }
    return *MI;
  }

  // First stage at which an instruction with opcode Opc can issue after an
  // MFMA with opcode MFMAOpc.
  unsigned getIssueStage(unsigned MFMAOpc, unsigned Opc) {
    return getMFMACoExecInfo(MFMAOpc).getFirstAllowedStage(
        getCoExecMaskForMI(buildMI(Opc), getInstrInfo()));
  }

private:
  std::unique_ptr<GCNTargetMachine> TM;
  GCNSubtarget ST;
  LLVMContext Ctx;
  Module Mod;
  MachineModuleInfo MMI;
  std::unique_ptr<MachineFunction> MF;
};
} // namespace

TEST_F(AMDGPUMFMACoExecRules, GFX950IssueStages) {
  auto TM =
      createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx950", "");
  if (!TM)
    GTEST_SKIP();
  MFMACoExecTest Test(std::move(TM));
  const SIInstrInfo &TII = Test.getInstrInfo();
  StringMap<unsigned> Opcodes;
  for (unsigned Op = 0; Op < TII.getNumOpcodes(); ++Op)
    if (TII.isMFMA(Op))
      Opcodes[TII.getName(Op)] = Op;

  struct {
    // Opcode names without the _e64, _vgprcd_e64 or _mac*_e64 suffix.
    std::vector<StringRef> MFMAs;
    struct {
      unsigned VALU, PackedF32, DS, VMEM, SALU;
    } Expected;
  } Groups[] = {
      {{"V_MFMA_F32_16X16X32_BF16", "V_MFMA_F32_16X16X32_F16",
        "V_MFMA_I32_16X16X64_I8", "V_MFMA_F32_16X16X128_F8F6F4_f4_f4",
        "V_MFMA_F32_16X16X128_F8F6F4_f4_f6",
        "V_MFMA_F32_16X16X128_F8F6F4_f6_f4",
        "V_MFMA_F32_16X16X128_F8F6F4_f6_f6",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6"},
       {2, 7, 2, 2, 1}},
      {{"V_MFMA_F32_16X16X128_F8F6F4_f4_f8",
        "V_MFMA_F32_16X16X128_F8F6F4_f6_f8",
        "V_MFMA_F32_16X16X128_F8F6F4_f8_f4",
        "V_MFMA_F32_16X16X128_F8F6F4_f8_f6",
        "V_MFMA_F32_16X16X128_F8F6F4_f8_f8",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6",
        "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8"},
       {3, 11, 2, 2, 1}},
      {{"V_MFMA_F32_32X32X16_BF16", "V_MFMA_F32_32X32X16_F16",
        "V_MFMA_I32_32X32X32_I8", "V_MFMA_F32_32X32X64_F8F6F4_f4_f4",
        "V_MFMA_F32_32X32X64_F8F6F4_f4_f6", "V_MFMA_F32_32X32X64_F8F6F4_f6_f4",
        "V_MFMA_F32_32X32X64_F8F6F4_f6_f6",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6"},
       {2, 11, 2, 2, 1}},
      {{"V_MFMA_F32_32X32X64_F8F6F4_f4_f8", "V_MFMA_F32_32X32X64_F8F6F4_f6_f8",
        "V_MFMA_F32_32X32X64_F8F6F4_f8_f4", "V_MFMA_F32_32X32X64_F8F6F4_f8_f6",
        "V_MFMA_F32_32X32X64_F8F6F4_f8_f8",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6",
        "V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8"},
       {3, 19, 2, 2, 1}},
      {{"V_SMFMAC_F32_16X16X64_BF16", "V_SMFMAC_F32_16X16X64_F16",
        "V_SMFMAC_I32_16X16X128_I8", "V_SMFMAC_F32_16X16X128_BF8_BF8",
        "V_SMFMAC_F32_16X16X128_BF8_FP8", "V_SMFMAC_F32_16X16X128_FP8_BF8",
        "V_SMFMAC_F32_16X16X128_FP8_FP8"},
       {3, 7, 2, 2, 1}},
      {{"V_SMFMAC_F32_32X32X32_BF16", "V_SMFMAC_F32_32X32X32_F16",
        "V_SMFMAC_I32_32X32X64_I8", "V_SMFMAC_F32_32X32X64_BF8_BF8",
        "V_SMFMAC_F32_32X32X64_BF8_FP8", "V_SMFMAC_F32_32X32X64_FP8_BF8",
        "V_SMFMAC_F32_32X32X64_FP8_FP8"},
       {3, 11, 3, 3, 1}},
      {{"V_MFMA_F64_16X16X4F64"}, {18, 18, 1, 1, 1}},
  };
  for (const auto &G : Groups) {
    for (StringRef Base : G.MFMAs) {
      unsigned NumForms = 0;
      for (StringRef Suffix :
           {"_e64", "_vgprcd_e64", "_mac_e64", "_mac_vgprcd_e64"}) {
        auto It = Opcodes.find((Base + Suffix).str());
        if (It == Opcodes.end())
          continue;
        ++NumForms;
        unsigned MFMAOpc = It->second;
        StringRef Name = It->getKey();
        EXPECT_EQ(Test.getIssueStage(MFMAOpc, V_FMA_F32_e64), G.Expected.VALU)
            << Name;
        EXPECT_EQ(Test.getIssueStage(MFMAOpc, V_PK_ADD_F16), G.Expected.VALU)
            << Name;
        EXPECT_EQ(Test.getIssueStage(MFMAOpc, V_PK_FMA_F32),
                  G.Expected.PackedF32)
            << Name;
        CoExecInfo Info = getMFMACoExecInfo(MFMAOpc);
        EXPECT_EQ(Info.getFirstAllowedStage(CoExecMask::DS), G.Expected.DS)
            << Name;
        EXPECT_EQ(Info.getFirstAllowedStage(CoExecMask::VMEM), G.Expected.VMEM)
            << Name;
        EXPECT_EQ(Info.getFirstAllowedStage(CoExecMask::SALU), G.Expected.SALU)
            << Name;
      }
      EXPECT_GT(NumForms, 0u) << Base;
    }
  }
}

TEST_F(AMDGPUMFMACoExecRules, CoExecMaskClasses) {
  auto TM =
      createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx950", "");
  if (!TM)
    GTEST_SKIP();
  MFMACoExecTest Test(std::move(TM));
  const SIInstrInfo &TII = Test.getInstrInfo();
  auto Check = [&](ArrayRef<unsigned> Opcs, CoExecMaskT Expected) {
    for (unsigned Opc : Opcs)
      EXPECT_EQ(getCoExecMaskForMI(Test.buildMI(Opc), TII), Expected)
          << TII.getName(Opc);
  };
  Check({V_PK_ADD_F32, V_PK_FMA_F32, V_PK_MOV_B32, V_PK_MUL_F32},
        CoExecMask::PK_F32);
  Check({V_FMA_F32_e64, V_PK_ADD_F16}, CoExecMask::VALU);
}
