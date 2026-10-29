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
struct IssueCycles {
  unsigned MFMAOpc;
  int64_t CBSZ, BLGP;
  unsigned VALU, PackedF32;
};

class MFMAIssueCycleTest {
public:
  explicit MFMAIssueCycleTest(std::unique_ptr<GCNTargetMachine> TheTM)
      : TM(std::move(TheTM)),
        ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
           std::string(TM->getTargetFeatureString()), *TM),
        Mod("Module", Ctx), MMI(TM.get()) {
    Mod.setDataLayout(TM->getTargetTriple().computeDataLayout());
    auto *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                               GlobalValue::ExternalLinkage, "Test", &Mod);
    MF = std::make_unique<MachineFunction>(*F, *TM, ST, MMI.getContext(), 0);
  }

  // Builds an MFMA with placeholder operands; CBSZ/BLGP select F8F6F4 formats.
  const MachineInstr &buildMFMA(unsigned Opc, int64_t CBSZ, int64_t BLGP) {
    const SIInstrInfo &TII = *ST.getInstrInfo();
    const MCInstrDesc &Desc = TII.get(Opc);
    MachineInstr *MI = MF->CreateMachineInstr(Desc, DebugLoc());
    for (auto [Idx, Info] : enumerate(Desc.operands())) {
      if (Info.RegClass == -1)
        MI->addOperand(*MF, MachineOperand::CreateImm(0));
      else
        MI->addOperand(*MF, MachineOperand::CreateReg(Register(), Idx < 1));
    }
    if (MachineOperand *Op = TII.getNamedOperand(*MI, AMDGPU::OpName::cbsz))
      Op->setImm(CBSZ);
    if (MachineOperand *Op = TII.getNamedOperand(*MI, AMDGPU::OpName::blgp))
      Op->setImm(BLGP);
    return *MI;
  }

  void check(ArrayRef<IssueCycles> Cases) {
    const SIInstrInfo &TII = *ST.getInstrInfo();
    for (const IssueCycles &C : Cases) {
      const MachineInstr &MFMA = buildMFMA(C.MFMAOpc, C.CBSZ, C.BLGP);
      StringRef Name = TII.getName(C.MFMAOpc);
      EXPECT_EQ(getVALUIssueCycle(MFMA, TII), C.VALU) << Name;
      EXPECT_EQ(getPackedVALUIssueCycle(MFMA, V_PK_ADD_F16, TII), C.VALU)
          << Name;
      EXPECT_EQ(getPackedVALUIssueCycle(MFMA, V_PK_FMA_F32, TII), C.PackedF32)
          << Name;
      EXPECT_EQ(getPackedVALUIssueCycle(MFMA, V_DOT2_F32_F16, TII), C.PackedF32)
          << Name;
      EXPECT_EQ(getPackedVALUIssueCycle(MFMA, /*IsFMALike=*/true,
                                        /*IsDOTLike=*/false, TII),
                C.PackedF32)
          << Name;
    }
  }

  void checkPacked(unsigned MFMAOpc, ArrayRef<unsigned> PackedOpcs,
                   unsigned Expected) {
    const SIInstrInfo &TII = *ST.getInstrInfo();
    const MachineInstr &MFMA = buildMFMA(MFMAOpc, 0, 0);
    for (unsigned PK : PackedOpcs)
      EXPECT_EQ(getPackedVALUIssueCycle(MFMA, PK, TII), Expected)
          << TII.getName(PK);
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

// Expected cycles were measured with ATT on MI308X with the peak TOPS limiter
// enabled and on MI350X; one cycle is 4 clocks. Scaled MFMAs were measured
// issuing right behind another MFMA.
TEST_F(AMDGPUMFMACoExecRules, GFX942IssueCycles) {
  auto TM =
      createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx942", "");
  if (!TM)
    GTEST_SKIP();
  MFMAIssueCycleTest Test(std::move(TM));
  Test.checkPacked(V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64,
                   {V_PK_MUL_F32, V_PK_ADD_F32, V_PK_MOV_B32, V_DOT2_I32_I16},
                   6);
  Test.checkPacked(V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64,
                   {V_PK_FMA_F16, V_PK_MUL_F16, V_PK_ADD_U16, V_PK_MUL_LO_U16},
                   1);
  Test.checkPacked(V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64,
                   {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32}, 6);
  Test.checkPacked(V_MFMA_F32_32X32X8BF16_1K_vgprcd_e64,
                   {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32}, 10);
  Test.checkPacked(V_MFMA_F32_16X16X4F32_vgprcd_e64,
                   {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32}, 10);
  Test.check({
      {V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64, 0, 0, 1, 6},
      {V_MFMA_F32_16X16X16BF16_1K_e64, 0, 0, 1, 6},
      {V_MFMA_F32_4X4X4BF16_1K_vgprcd_e64, 0, 0, 1, 4},
      {V_MFMA_F32_32X32X8BF16_1K_vgprcd_e64, 0, 0, 1, 10},
      {V_MFMA_F32_16X16X16F16_vgprcd_e64, 0, 0, 1, 6},
      {V_MFMA_I32_16X16X32I8_vgprcd_e64, 0, 0, 1, 6},
      {V_MFMA_F32_16X16X32_FP8_FP8_vgprcd_e64, 0, 0, 1, 6},
      {V_MFMA_F32_16X16X8XF32_vgprcd_e64, 0, 0, 1, 6},
      {V_SMFMAC_F32_16X16X32_F16_e64, 0, 0, 2, 6},
      {V_SMFMAC_F32_32X32X16_F16_e64, 0, 0, 2, 10},
      {V_MFMA_F32_16X16X4F32_vgprcd_e64, 0, 0, 10, 10},
      {V_MFMA_F64_16X16X4F64_vgprcd_e64, 0, 0, 10, 10},
      {V_MFMA_F64_4X4X4F64_vgprcd_e64, 0, 0, 6, 6},
  });
}

TEST_F(AMDGPUMFMACoExecRules, GFX950IssueCycles) {
  auto TM =
      createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx950", "");
  if (!TM)
    GTEST_SKIP();
  MFMAIssueCycleTest Test(std::move(TM));
  Test.checkPacked(
      V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64,
      {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32, V_DOT2C_F32_BF16_e32}, 6);
  Test.checkPacked(V_MFMA_F32_32X32X8BF16_1K_vgprcd_e64,
                   {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32}, 10);
  Test.checkPacked(V_MFMA_F32_16X16X4F32_vgprcd_e64,
                   {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32}, 10);
  Test.checkPacked(
      V_MFMA_F32_16X16X32_BF16_vgprcd_e64,
      {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32, V_DOT2C_F32_BF16_e32}, 7);
  Test.checkPacked(V_MFMA_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64,
                   {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32}, 11);
  Test.checkPacked(V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64,
                   {V_DOT2C_F32_F16_e32, V_DOT4C_I32_I8_e32}, 11);
  Test.check({
      {V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64, 0, 0, 1, 6},
      {V_MFMA_F32_4X4X4BF16_1K_vgprcd_e64, 0, 0, 1, 4},
      {V_MFMA_F32_32X32X8BF16_1K_vgprcd_e64, 0, 0, 1, 10},
      {V_SMFMAC_F32_16X16X32_F16_e64, 0, 0, 2, 6},
      {V_SMFMAC_F32_32X32X16_F16_e64, 0, 0, 2, 10},
      {V_MFMA_F32_16X16X32_BF16_vgprcd_e64, 0, 0, 2, 7},
      {V_MFMA_F32_16X16X32_BF16_e64, 0, 0, 2, 7},
      {V_MFMA_F32_16X16X32_F16_vgprcd_e64, 0, 0, 2, 7},
      {V_MFMA_I32_16X16X64_I8_vgprcd_e64, 0, 0, 2, 7},
      {V_MFMA_F32_32X32X16_BF16_vgprcd_e64, 0, 0, 2, 11},
      {V_MFMA_F32_32X32X16_F16_vgprcd_e64, 0, 0, 2, 11},
      {V_MFMA_I32_32X32X32_I8_vgprcd_e64, 0, 0, 2, 11},
      {V_MFMA_F32_16X16X128_F8F6F4_f4_f4_vgprcd_e64, 4, 4, 2, 7},
      {V_MFMA_F32_16X16X128_F8F6F4_f6_f6_vgprcd_e64, 2, 2, 2, 7},
      {V_MFMA_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64, 0, 0, 3, 11},
      {V_MFMA_F32_16X16X128_F8F6F4_f8_f4_vgprcd_e64, 0, 4, 3, 11},
      {V_MFMA_F32_32X32X64_F8F6F4_f4_f4_vgprcd_e64, 4, 4, 2, 11},
      {V_MFMA_F32_32X32X64_F8F6F4_f8_f8_vgprcd_e64, 0, 0, 3, 19},
      {V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64, 0, 0, 3, 11},
      {V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_vgprcd_e64, 4, 4, 2, 7},
      {V_SMFMAC_F32_16X16X64_BF16_e64, 0, 0, 3, 7},
      {V_SMFMAC_F32_32X32X32_BF16_e64, 0, 0, 3, 11},
      {V_SMFMAC_F32_16X16X128_FP8_FP8_e64, 0, 0, 3, 7},
      {V_SMFMAC_F32_32X32X64_FP8_FP8_e64, 0, 0, 3, 11},
      {V_MFMA_F32_16X16X4F32_vgprcd_e64, 0, 0, 10, 10},
      {V_MFMA_F64_16X16X4F64_vgprcd_e64, 0, 0, 18, 18},
      {V_MFMA_F64_4X4X4F64_vgprcd_e64, 0, 0, 6, 6},
  });
}
