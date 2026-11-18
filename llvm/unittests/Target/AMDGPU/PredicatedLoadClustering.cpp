//===- PredicatedLoadClustering.cpp
//----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class PredicatedLoadClusteringTest : public AMDGPUTestBase {
protected:
  LLVMContext Ctx;
  std::unique_ptr<const GCNTargetMachine> TM;
  std::unique_ptr<GCNSubtarget> ST;
  std::unique_ptr<Module> Mod;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MachineFunction> MF;
  MachineBasicBlock *BB;
  const SIInstrInfo *TII;
  Register Addr, SAddr, VOffset, Pred;

  void SetUp() override {
    TM = createAMDGPUTargetMachine(Triple("amdgcn-amd-"), "gfx1100",
                                   "+wavefrontsize32");
    if (!TM)
      GTEST_SKIP();
    ST = std::make_unique<GCNSubtarget>(
        TM->getTargetTriple(), std::string(TM->getTargetCPU()),
        std::string(TM->getTargetFeatureString()), *TM);
    Mod = std::make_unique<Module>("test", Ctx);
    Mod->setDataLayout(TM->createDataLayout());
    auto *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                               GlobalValue::ExternalLinkage, "test", *Mod);
    F->addFnAttr("amdgpu-max-memory-cluster-dwords", "8");
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    MF = std::make_unique<MachineFunction>(*F, *TM, *ST, MMI->getContext(), 0);
    MF->initTargetMachineFunctionInfo(*ST);
    BB = MF->CreateMachineBasicBlock();
    MF->push_back(BB);
    TII = ST->getInstrInfo();
    Addr = makeReg(AMDGPU::VReg_64RegClass);
    SAddr = makeReg(AMDGPU::SReg_64_XEXEC_XNULLRegClass);
    VOffset = makeReg(AMDGPU::VGPR_32RegClass);
    Pred = makeReg(AMDGPU::SReg_32RegClass);
  }

  Register makeReg(const TargetRegisterClass &RC) {
    Register R = MF->getRegInfo().createVirtualRegister(&RC);
    BuildMI(*BB, BB->end(), DebugLoc(), TII->get(TargetOpcode::IMPLICIT_DEF),
            R);
    return R;
  }

  MachineInstr *load(unsigned Opc = AMDGPU::PRED_GLOBAL_LOAD_DWORD_RA,
                     int64_t Offset = 0) {
    unsigned DstIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
    const auto *RC = ST->getRegisterInfo()->getAllocatableClass(
        TII->getRegClass(TII->get(Opc), DstIdx));
    Register Dst = makeReg(*RC);
    auto MIB = BuildMI(*BB, BB->end(), DebugLoc(), TII->get(Opc), Dst);
    if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::pred) >= 0)
      MIB.addDef(
          MF->getRegInfo().createVirtualRegister(&AMDGPU::SReg_32RegClass),
          RegState::EarlyClobber);
    if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::voffset) >= 0)
      MIB.addReg(SAddr).addReg(VOffset).addImm(Offset).addImm(0);
    else {
      MIB.addReg(Addr).addImm(Offset);
      if (Opc == AMDGPU::GLOBAL_LOAD_DWORD)
        MIB.addImm(0);
    }
    if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::pred) >= 0)
      MIB.addReg(Pred).addReg(Dst);
    return MIB.getInstr();
  }

  bool canCluster(const MachineInstr &A, const MachineInstr &B,
                  unsigned Count = 2, unsigned Bytes = 8) {
    SmallVector<const MachineOperand *, 4> BasesA, BasesB;
    int64_t OffsetA = 0, OffsetB = 0;
    bool ScalableA = true, ScalableB = true;
    LocationSize WidthA = LocationSize::precise(0);
    LocationSize WidthB = LocationSize::precise(0);
    if (!TII->getMemOperandsWithOffsetWidth(A, BasesA, OffsetA, ScalableA,
                                            WidthA) ||
        !TII->getMemOperandsWithOffsetWidth(B, BasesB, OffsetB, ScalableB,
                                            WidthB))
      return false;
    return TII->shouldClusterMemOps(BasesA, OffsetA, ScalableA, BasesB, OffsetB,
                                    ScalableB, Count, Bytes);
  }
};

// These are target-hook tests, independent of scheduler reachability and SCC
// dependencies. Success here does not assert that a DAG cluster will be formed.
TEST_F(PredicatedLoadClusteringTest, AddressesOffsetsAndWidths) {
  const unsigned Opcodes[] = {AMDGPU::PRED_GLOBAL_LOAD_DWORD_RA,
                              AMDGPU::PRED_GLOBAL_LOAD_DWORDX2_RA,
                              AMDGPU::PRED_GLOBAL_LOAD_DWORDX4_RA,
                              AMDGPU::PRED_GLOBAL_LOAD_DWORD_RA_SADDR,
                              AMDGPU::PRED_GLOBAL_LOAD_DWORDX2_RA_SADDR,
                              AMDGPU::PRED_GLOBAL_LOAD_DWORDX4_RA_SADDR};
  for (unsigned I = 0; I != 6; ++I) {
    SCOPED_TRACE(TII->getName(Opcodes[I]).str());
    auto *A = load(Opcodes[I], -16);
    SmallVector<const MachineOperand *, 4> Bases;
    int64_t Offset = 0;
    bool Scalable = true;
    LocationSize Width = LocationSize::precise(0);
    ASSERT_TRUE(
        TII->getMemOperandsWithOffsetWidth(*A, Bases, Offset, Scalable, Width));
    ASSERT_EQ(Bases.size(), I < 3 ? 1u : 2u);
    EXPECT_EQ(Bases[0]->getReg(), I < 3 ? Addr : SAddr);
    if (I >= 3)
      EXPECT_EQ(Bases[1]->getReg(), VOffset);
    EXPECT_EQ(Offset, -16);
    EXPECT_FALSE(Scalable);
    EXPECT_EQ(AMDGPU::supportsScaleOffset(*TII, Opcodes[I]), I >= 3);
    EXPECT_EQ(Width, LocationSize::precise(4u << (I % 3)));
    auto *B = load(Opcodes[I], 16);
    EXPECT_TRUE(canCluster(*A, *B, 2, 2 * (4u << (I % 3))));
  }
}

TEST_F(PredicatedLoadClusteringTest, PredicateIdentity) {
  auto *A = load();
  auto *B = load();
  EXPECT_TRUE(canCluster(*A, *B));
  auto *P = TII->getNamedOperand(*B, AMDGPU::OpName::pred);
  P->setIsKill(true);
  EXPECT_TRUE(canCluster(*A, *B));
  P->setIsKill(false);
  P->setIsUndef(true);
  EXPECT_FALSE(canCluster(*A, *B));
  EXPECT_FALSE(canCluster(*B, *A));
  P->setIsUndef(false);
  P->setReg(makeReg(AMDGPU::SReg_32RegClass));
  EXPECT_FALSE(canCluster(*A, *B));
  EXPECT_FALSE(canCluster(*B, *A));
}

TEST_F(PredicatedLoadClusteringTest, PredicateSubregisters) {
  Pred = makeReg(AMDGPU::SReg_64RegClass);
  auto *A = load();
  auto *B = load();
  TII->getNamedOperand(*A, AMDGPU::OpName::pred)->setSubReg(AMDGPU::sub0);
  TII->getNamedOperand(*B, AMDGPU::OpName::pred)->setSubReg(AMDGPU::sub0);
  EXPECT_TRUE(canCluster(*A, *B));
  TII->getNamedOperand(*B, AMDGPU::OpName::pred)->setSubReg(AMDGPU::sub1);
  EXPECT_FALSE(canCluster(*A, *B));
}

TEST_F(PredicatedLoadClusteringTest, PhysicalOrMultiplyDefinedPredicate) {
  auto *A = load();
  auto *B = load();
  BuildMI(*BB, BB->end(), DebugLoc(), TII->get(AMDGPU::S_MOV_B32), Pred)
      .addImm(0);
  EXPECT_FALSE(canCluster(*A, *B));
  TII->getNamedOperand(*A, AMDGPU::OpName::pred)->setReg(AMDGPU::SGPR0);
  TII->getNamedOperand(*B, AMDGPU::OpName::pred)->setReg(AMDGPU::SGPR0);
  EXPECT_FALSE(canCluster(*A, *B));
}

TEST_F(PredicatedLoadClusteringTest, OrdinaryLoadsRemainSeparate) {
  auto *P = load();
  auto *A = load(AMDGPU::GLOBAL_LOAD_DWORD);
  auto *B = load(AMDGPU::GLOBAL_LOAD_DWORD, 4);
  EXPECT_TRUE(canCluster(*A, *B));
  EXPECT_FALSE(canCluster(*P, *A));
  EXPECT_FALSE(canCluster(*A, *P));
}

TEST_F(PredicatedLoadClusteringTest, ExistingBaseAndSizePolicy) {
  auto *A = load();
  auto *B = load();
  EXPECT_TRUE(canCluster(*A, *B, 8, 32));
  EXPECT_FALSE(canCluster(*A, *B, 9, 36));
  TII->getNamedOperand(*B, AMDGPU::OpName::vaddr)
      ->setReg(makeReg(AMDGPU::VReg_64RegClass));
  EXPECT_FALSE(canCluster(*A, *B));
}
} // namespace
