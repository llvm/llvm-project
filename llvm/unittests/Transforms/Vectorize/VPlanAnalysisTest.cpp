//===- llvm/unittests/Transforms/Vectorize/VPlanAnalysisTest.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Transforms/Vectorize/VPlanAnalysis.h"
#include "VPlanTestBase.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/IR/Instruction.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MockRegPressureTTI
    : public TargetTransformInfoImplCRTPBase<MockRegPressureTTI> {
  bool ModelResultPressure = false;
  bool CheckLiveInCastOperand = false;
  mutable bool SawLiveInCastOperand = false;
  SmallBitVector ReusableOperandsMask;

public:
  explicit MockRegPressureTTI(const DataLayout &DL)
      : TargetTransformInfoImplCRTPBase<MockRegPressureTTI>(DL) {}

  void setModelResultPressure(bool B) { ModelResultPressure = B; }
  void checkLiveInCastOperand() { CheckLiveInCastOperand = true; }
  bool sawLiveInCastOperand() const { return SawLiveInCastOperand; }
  void setReusableOperandsMask(SmallBitVector Mask) {
    ReusableOperandsMask = std::move(Mask);
  }

  unsigned getNumberOfRegisters(unsigned ClassID) const {
    (void)ClassID;
    return 32;
  }

  unsigned getRegisterClassForType(bool Vector, Type *Ty) const {
    (void)Vector;
    (void)Ty;
    return 1;
  }

  unsigned getRegUsageForType(Type *Ty) const {
    (void)Ty;
    return 1;
  }

  std::optional<SmallBitVector> getResultRegisterReuseMask(
      unsigned Opcode, Type *ResultType,
      ArrayRef<TTI::RegisterUsageOperandInfo> Operands) const {
    if (!ModelResultPressure)
      return std::nullopt;
    if (Opcode != Instruction::Add)
      return std::nullopt;

    EXPECT_TRUE(ResultType->isVectorTy());
    EXPECT_EQ(Operands.size(), 2u);
    if (CheckLiveInCastOperand && Operands.size() == 2) {
      LLVMContext &Ctx = ResultType->getContext();
      EXPECT_EQ(Operands[1].ValueType,
                FixedVectorType::get(Type::getInt32Ty(Ctx), 4));
      EXPECT_EQ(Operands[1].SourceType,
                FixedVectorType::get(Type::getInt16Ty(Ctx), 4));
      EXPECT_EQ(Operands[1].DefOpcode, Instruction::SExt);
      EXPECT_TRUE(Operands[1].IsUniform);
      SawLiveInCastOperand = true;
    }

    SmallBitVector ReusableOperands = ReusableOperandsMask;
    ReusableOperands.resize(Operands.size());
    return ReusableOperands;
  }
};

class VPRegisterUsageAnalysisTest : public VPlanTestBase {
protected:
  VPlan &buildSingleAddPlan(bool KeepFirstOperandLive = false,
                            bool RepeatFirstOperand = false,
                            bool UseLiveInCast = false) {
    VPlan &Plan = getPlan();
    auto *I32Ty = Type::getInt32Ty(C);

    VPBasicBlock *Preheader = Plan.getEntry();
    VPBasicBlock *Header = Plan.createVPBasicBlock("header");
    VPBasicBlock *Latch = Plan.createVPBasicBlock("latch");
    auto *LoopRegion = Plan.createLoopRegion(I32Ty, DebugLoc::getUnknown(),
                                             "vector.loop", Header, Latch);

    auto *LiveInA = Plan.getConstantInt(I32Ty, 1);
    auto *LiveInB = Plan.getConstantInt(I32Ty, 2);
    auto *Seed0 =
        new VPWidenRecipe(Instruction::Mul, {LiveInA, LiveInB},
                          VPIRFlags::getDefaultFlags(Instruction::Mul));
    auto *Seed1 =
        new VPWidenRecipe(Instruction::Sub, {LiveInA, LiveInB},
                          VPIRFlags::getDefaultFlags(Instruction::Sub));
    Header->appendRecipe(Seed0);
    Header->appendRecipe(Seed1);

    VPValue *SecondAddOperand = Seed1;
    if (RepeatFirstOperand)
      SecondAddOperand = Seed0;
    if (UseLiveInCast) {
      auto *Ext =
          new SExtInst(ConstantInt::get(Type::getInt16Ty(C), 1), I32Ty, "ext",
                       ScalarHeader->getTerminator()->getIterator());
      SecondAddOperand = Plan.getOrAddLiveIn(Ext);
    }
    auto *Add = new VPInstruction(Instruction::Add, {Seed0, SecondAddOperand},
                                  VPIRFlags::getDefaultFlags(Instruction::Add));
    Header->appendRecipe(Add);
    VPValue *SecondConsumeOperand =
        KeepFirstOperandLive ? static_cast<VPValue *>(Seed0)
        : RepeatFirstOperand ? static_cast<VPValue *>(Seed1)
                             : static_cast<VPValue *>(LiveInA);
    auto *Consume =
        new VPWidenRecipe(Instruction::Sub, {Add, SecondConsumeOperand},
                          VPIRFlags::getDefaultFlags(Instruction::Sub));
    Latch->appendRecipe(Consume);
    Latch->appendRecipe(
        new VPInstruction(VPInstruction::BranchOnCond, {Plan.getTrue()}));

    VPBlockUtils::connectBlocks(Header, Latch);
    VPBlockUtils::connectBlocks(Preheader, LoopRegion);
    VPBlockUtils::connectBlocks(LoopRegion, Plan.getScalarHeader());
    return Plan;
  }

  unsigned getMaxLocalUsers(VPlan &Plan, TargetTransformInfo &TTI,
                            VPRegisterUsageMode Mode) {
    SmallVector<VPRegisterUsage, 1> Usage = calculateRegisterUsageForPlan(
        Plan, {ElementCount::getFixed(4)}, TTI, Mode);
    EXPECT_EQ(Usage.size(), 1u);
    return Usage[0].MaxLocalUsers.lookup(1);
  }
};

TEST_F(VPRegisterUsageAnalysisTest, DefaultPressureDoesNotCountResultAtDef) {
  VPlan &Plan = buildSingleAddPlan();
  auto Impl = std::make_unique<MockRegPressureTTI>(Plan.getDataLayout());
  Impl->setModelResultPressure(true);
  auto TTI = TargetTransformInfo(std::move(Impl));

  auto Usage =
      calculateRegisterUsageForPlan(Plan, {ElementCount::getFixed(4)}, TTI);
  ASSERT_EQ(Usage.size(), 1u);
  EXPECT_EQ(Usage[0].MaxLocalUsers.lookup(1), 2u);
  EXPECT_EQ(getMaxLocalUsers(Plan, TTI, VPRegisterUsageMode::LiveIntervals),
            2u);
  EXPECT_EQ(getMaxLocalUsers(Plan, TTI, VPRegisterUsageMode::ConservativePeak),
            3u);
}

TEST_F(VPRegisterUsageAnalysisTest,
       ConservativePressureCountsResultWithoutReusableDyingOperand) {
  VPlan &Plan = buildSingleAddPlan();
  auto Impl = std::make_unique<MockRegPressureTTI>(Plan.getDataLayout());
  Impl->setModelResultPressure(true);
  auto TTI = TargetTransformInfo(std::move(Impl));

  EXPECT_EQ(getMaxLocalUsers(Plan, TTI, VPRegisterUsageMode::ConservativePeak),
            3u);
}

TEST_F(VPRegisterUsageAnalysisTest,
       ConservativePressureKeepsPeakAtTwoWithReusableDyingOperand) {
  VPlan &Plan = buildSingleAddPlan();

  auto Impl = std::make_unique<MockRegPressureTTI>(Plan.getDataLayout());
  Impl->setModelResultPressure(true);
  SmallBitVector ReusableOperands(2, false);
  ReusableOperands.set(0);
  Impl->setReusableOperandsMask(std::move(ReusableOperands));
  auto TTI = TargetTransformInfo(std::move(Impl));

  EXPECT_EQ(getMaxLocalUsers(Plan, TTI, VPRegisterUsageMode::ConservativePeak),
            2u);
}

TEST_F(VPRegisterUsageAnalysisTest,
       ConservativePressureDoesNotReuseLiveOperand) {
  VPlan &Plan = buildSingleAddPlan(/*KeepFirstOperandLive=*/true);

  auto Impl = std::make_unique<MockRegPressureTTI>(Plan.getDataLayout());
  Impl->setModelResultPressure(true);
  SmallBitVector ReusableOperands(2, false);
  ReusableOperands.set(0);
  Impl->setReusableOperandsMask(std::move(ReusableOperands));
  auto TTI = TargetTransformInfo(std::move(Impl));

  EXPECT_EQ(getMaxLocalUsers(Plan, TTI, VPRegisterUsageMode::ConservativePeak),
            3u);
}

TEST_F(VPRegisterUsageAnalysisTest,
       ConservativePressureRequiresAllDuplicateUsesToBeReusable) {
  VPlan &Plan = buildSingleAddPlan(/*KeepFirstOperandLive=*/false,
                                   /*RepeatFirstOperand=*/true);

  auto Impl = std::make_unique<MockRegPressureTTI>(Plan.getDataLayout());
  Impl->setModelResultPressure(true);
  SmallBitVector ReusableOperands(2, false);
  ReusableOperands.set(0);
  Impl->setReusableOperandsMask(std::move(ReusableOperands));
  auto TTI = TargetTransformInfo(std::move(Impl));

  EXPECT_EQ(getMaxLocalUsers(Plan, TTI, VPRegisterUsageMode::ConservativePeak),
            3u);
}

TEST_F(VPRegisterUsageAnalysisTest,
       ConservativePressureDescribesLiveInCastOperand) {
  VPlan &Plan = buildSingleAddPlan(/*KeepFirstOperandLive=*/false,
                                   /*RepeatFirstOperand=*/false,
                                   /*UseLiveInCast=*/true);

  auto Impl = std::make_unique<MockRegPressureTTI>(Plan.getDataLayout());
  Impl->setModelResultPressure(true);
  Impl->checkLiveInCastOperand();
  MockRegPressureTTI *ImplPtr = Impl.get();
  auto TTI = TargetTransformInfo(std::move(Impl));

  (void)getMaxLocalUsers(Plan, TTI, VPRegisterUsageMode::ConservativePeak);
  EXPECT_TRUE(ImplPtr->sawLiveInCastOperand());
}

} // namespace
