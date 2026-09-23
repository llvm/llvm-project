//===- ProfDataUtilsTest.cpp - Profiling metadata tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ProfDataUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "gtest/gtest.h"
#include <initializer_list>

using namespace llvm;

namespace {
static BitVector makeBitVector(unsigned Size,
                               std::initializer_list<unsigned> SetBits) {
  BitVector Result(Size);
  for (unsigned Index : SetBits)
    Result.set(Index);
  return Result;
}

class WaveProfileTest : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;

  void expectUnmeasured(Function &F, unsigned Index) {
    SmallVector<uint64_t> Counts;
    BitVector Valid;
    uint64_t Entry;
    if (extractMappedBlockWaveCounts(F, Counts, Valid, Entry))
      EXPECT_FALSE(Valid[Index]);
    if (extractBlockWaveCounts(F, Counts, &Valid))
      EXPECT_FALSE(Valid[Index]);
  }

  void SetUp() override {
    SMDiagnostic Error;
    M = parseAssemblyString(R"(
      define void @diamond(i1 %condition) {
      entry:
        br i1 %condition, label %left, label %right
      left:
        br label %exit
      right:
        br label %exit
      exit:
        ret void
      })",
                            Error, Context);
    ASSERT_TRUE(M);
  }
};

TEST_F(WaveProfileTest, RoundTripAndReplacement) {
  Function &F = *M->getFunction("diamond");
  F.setEntryCount(6400);
  SmallVector<uint64_t> Counts{99};
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());
  setBlockWaveCounts(F, {100, 100, 100, 100});
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 100, 100, 100}));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  std::string Text;
  raw_string_ostream OS(Text);
  M->print(OS, nullptr);
  SMDiagnostic Error;
  std::unique_ptr<Module> Reloaded = parseAssemblyString(Text, Error, Context);
  ASSERT_TRUE(Reloaded);
  EXPECT_TRUE(
      extractBlockWaveCounts(*Reloaded->getFunction("diamond"), Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 100, 100, 100}));

  setBlockWaveCounts(F, {200, 0, 200, 200});
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{200, 0, 200, 200}));
}

TEST_F(WaveProfileTest, PreserveAcrossInstructionChanges) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  cast<CondBrInst>(F.getEntryBlock().getTerminator())
      ->setCondition(ConstantInt::getTrue(Context));
  SmallVector<uint64_t> Counts{99};
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 100, 100, 100}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, PreserveAcrossConditionalSuccessorSwap) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  cast<CondBrInst>(F.getEntryBlock().getTerminator())->swapSuccessors();
  SmallVector<uint64_t> Counts;
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 10, 90, 100}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, PreserveIdentityAcrossBlockReordering) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  Left->moveAfter(Left->getNextNode());
  SmallVector<uint64_t> Counts;
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 90, 10, 100}));
}

TEST_F(WaveProfileTest, PreserveAcrossEntryCountChangeButRejectRename) {
  Function &F = *M->getFunction("diamond");
  F.setEntryCount(6400);
  setBlockWaveCounts(F, {100, 100, 100, 100});
  F.setEntryCount(3200);
  SmallVector<uint64_t> Counts;
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts));
  F.setName("specialized");
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
}

TEST_F(WaveProfileTest, RejectRedirectedEdge) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Right = Left->getNextNode();
  cast<UncondBrInst>(Left->getTerminator())->setSuccessor(Right);
  SmallVector<uint64_t> Counts;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
}

TEST_F(WaveProfileTest, RejectDuplicateBlockIdentity) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Right = Left->getNextNode();
  Right->getTerminator()->setMetadata(
      LLVMContext::MD_wave_profile_block,
      Left->getTerminator()->getMetadata(LLVMContext::MD_wave_profile_block));
  SmallVector<uint64_t> Counts;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
}

TEST_F(WaveProfileTest, RejectSplitBlock) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  BasicBlock &Entry = F.getEntryBlock();
  Entry.splitBasicBlock(Entry.begin(), "split");
  SmallVector<uint64_t> Counts;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, MapCountsAfterRemovingOriginalBlock) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Exit = &F.back();
  Left->replaceAllUsesWith(Exit);
  Left->eraseFromParent();

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 90, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(3, {1}));
}

TEST_F(WaveProfileTest, MapCountsAfterRemovingOriginalEntry) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  F.getEntryBlock().eraseFromParent();

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{10, 90, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(3, {0, 1, 2}));
}

TEST_F(WaveProfileTest, MapCountsAroundNewUnmeasuredBlock) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Exit = &F.back();
  BasicBlock *Inserted = BasicBlock::Create(Context, "inserted", &F, Exit);
  UncondBrInst::Create(Exit, Inserted);
  cast<UncondBrInst>(Left->getTerminator())->setSuccessor(Inserted);

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 10, 90, 0, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(5, {0, 2}));
}

TEST_F(WaveProfileTest, MapCountsAroundDuplicateBlockIdentity) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Right = Left->getNextNode();
  Right->getTerminator()->setMetadata(
      LLVMContext::MD_wave_profile_block,
      Left->getTerminator()->getMetadata(LLVMContext::MD_wave_profile_block));

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 0, 0, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(4, {}));
}

TEST_F(WaveProfileTest, MapCountsAroundRedirectedEdge) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Right = Left->getNextNode();
  cast<UncondBrInst>(Left->getTerminator())->setSuccessor(Right);

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_TRUE(extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 10, 90, 100}));
  EXPECT_EQ(HasCounts, makeBitVector(4, {0}));
}

TEST_F(WaveProfileTest, RepresentExplicitlyUnmeasuredBlocks) {
  Function &F = *M->getFunction("diamond");
  BasicBlock &Entry = F.getEntryBlock();
  Entry.splitBasicBlock(Entry.begin(), "synthetic");

  SmallVector<uint64_t> ExpectedCounts{100, 0, 100, 100, 100};
  BitVector ExpectedHasCounts(ExpectedCounts.size(), true);
  ExpectedHasCounts.reset(1);
  setBlockWaveCounts(F, ExpectedCounts, ExpectedHasCounts);

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  EXPECT_TRUE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_EQ(Counts, ExpectedCounts);
  EXPECT_EQ(HasCounts, ExpectedHasCounts);
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());
}

TEST_F(WaveProfileTest, RejectUnmeasuredOriginalEntry) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  Instruction *EntryTerminator = F.getEntryBlock().getTerminator();
  MDNode *EntryMD =
      EntryTerminator->getMetadata(LLVMContext::MD_wave_profile_block);
  SmallVector<Metadata *> Ops(EntryMD->op_begin(), EntryMD->op_end());
  Ops[3] =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Context), 0));
  EntryTerminator->setMetadata(LLVMContext::MD_wave_profile_block,
                               MDNode::get(Context, Ops));

  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount = 0;
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_FALSE(extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 0u);
  EXPECT_TRUE(Counts.empty());
  EXPECT_TRUE(HasCounts.empty());
}

TEST_F(WaveProfileTest, RejectUnsupportedOrMalformedMetadata) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 100, 100, 100});
  MDNode *Valid = F.getMetadata(LLVMContext::MD_wave_profile);
  SmallVector<Metadata *> Ops(Valid->op_begin(), Valid->op_end());
  SmallVector<uint64_t> Counts{99};

  Ops[0] =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Context), 3));
  F.setMetadata(LLVMContext::MD_wave_profile, MDNode::get(Context, Ops));
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());
  EXPECT_FALSE(verifyModule(*M, &errs()));

  Ops[0] = Valid->getOperand(0);
  Ops[2] =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), 100));
  F.setMetadata(LLVMContext::MD_wave_profile, MDNode::get(Context, Ops));
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());

  Ops[2] = nullptr;
  F.setMetadata(LLVMContext::MD_wave_profile, MDNode::get(Context, Ops));
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());

  Ops[2] = Valid->getOperand(2);
  Ops.pop_back();
  F.setMetadata(LLVMContext::MD_wave_profile, MDNode::get(Context, Ops));
  EXPECT_FALSE(extractBlockWaveCounts(F, Counts));
  EXPECT_TRUE(Counts.empty());
  EXPECT_FALSE(verifyModule(*M, &errs()));
}
TEST_F(WaveProfileTest, TransferAcrossEdgeSplit) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 80, 60, 100});
  BlockWaveCountPreserver Profile(F);
  auto *Br = cast<CondBrInst>(F.getEntryBlock().getTerminator());
  BasicBlock *Left = Br->getSuccessor(0);
  BasicBlock *Edge = BasicBlock::Create(Context, "edge", &F);
  UncondBrInst::Create(Left, Edge);
  Br->setSuccessor(0, Edge);
  Profile.restore();

  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Entry, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 80, 60, 100, 0}));
  EXPECT_EQ(Valid, makeBitVector(5, {0, 1, 2, 3}));
  EXPECT_FALSE(verifyModule(*M, &errs()));

  BlockWaveCountPreserver Again(F);
  Again.restore();
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Entry, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 80, 60, 100, 0}));
  EXPECT_EQ(Valid, makeBitVector(5, {0, 1, 2, 3}));
}

TEST_F(WaveProfileTest, TransferDoesNotResurrectInvalidCounts) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 80, 60, 100});
  auto *Br = cast<CondBrInst>(F.getEntryBlock().getTerminator());
  BasicBlock *Left = Br->getSuccessor(0);
  BasicBlock *Edge = BasicBlock::Create(Context, "edge", &F);
  UncondBrInst::Create(Left, Edge);
  Br->setSuccessor(0, Edge);

  // Capture after an unsupported rewrite has invalidated the entry and left.
  BlockWaveCountPreserver Profile(F);
  Profile.restore();
  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Entry, 100u);
  EXPECT_EQ(Valid, makeBitVector(5, {2, 3}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferInvalidatesChangedExecutionEvent) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 80, 60, 100});
  BlockWaveCountPreserver Profile(F);
  Profile.invalidate(F.getEntryBlock());
  Profile.restore();
  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Entry, 100u);
  EXPECT_EQ(Valid, makeBitVector(4, {1, 2, 3}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferDoesNotResurrectDuplicatedCounts) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 80, 60, 100});
  auto *Br = cast<CondBrInst>(F.getEntryBlock().getTerminator());
  BasicBlock *Left = Br->getSuccessor(0);
  BasicBlock *Copy = BasicBlock::Create(Context, "copy", &F);
  UncondBrInst *CopyBr = UncondBrInst::Create(Left->getSingleSuccessor(), Copy);
  CopyBr->setMetadata(
      LLVMContext::MD_wave_profile_block,
      Left->getTerminator()->getMetadata(LLVMContext::MD_wave_profile_block));
  BlockWaveCountPreserver Profile(F);
  Profile.restore();
  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Entry, 100u);
  EXPECT_EQ(Valid, makeBitVector(5, {2}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferAfterRemovingOriginalEntry) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 80, 60, 100});
  BlockWaveCountPreserver Profile(F);
  F.getEntryBlock().eraseFromParent();
  Profile.restore();
  BlockWaveCountPreserver Again(F);
  Again.restore();

  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Entry, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{80, 60, 100}));
  EXPECT_EQ(Valid, makeBitVector(3, {0, 1, 2}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferDoesNotFollowReplacedBlock) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 80, 60, 100});
  BlockWaveCountPreserver Profile(F);
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BasicBlock *Replacement = BasicBlock::Create(Context, "replacement", &F);
  UncondBrInst::Create(Left->getSingleSuccessor(), Replacement);
  Left->replaceAllUsesWith(Replacement);
  Left->eraseFromParent();
  Profile.restore();

  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Entry, 100u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 60, 100, 0}));
  EXPECT_EQ(Valid, makeBitVector(4, {0, 1, 2}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, SparseMeasuredZeroAndReplacement) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {32, 0, 32, 0}, makeBitVector(4, {0, 1, 2}));
  SmallVector<uint64_t> Counts;
  BitVector HasCounts;
  uint64_t EntryCount;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_EQ(EntryCount, 32u);
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{32, 0, 32, 0}));
  EXPECT_EQ(HasCounts, makeBitVector(4, {0, 1, 2}));

  // Replacing a measured block with an unmeasured one clears its old state.
  setBlockWaveCounts(F, {32, 0, 32, 0}, makeBitVector(4, {0, 2}));
  ASSERT_TRUE(extractBlockWaveCounts(F, Counts, &HasCounts));
  EXPECT_EQ(HasCounts, makeBitVector(4, {0, 2}));
  clearBlockWaveCounts(F);
  EXPECT_FALSE(F.hasMetadata(LLVMContext::MD_wave_profile));
  for (BasicBlock &BB : F)
    EXPECT_FALSE(
        BB.getTerminator()->hasMetadata(LLVMContext::MD_wave_profile_block));
  EXPECT_FALSE(extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
  EXPECT_TRUE(Counts.empty());
  EXPECT_TRUE(HasCounts.empty());
  EXPECT_EQ(EntryCount, 0u);
}

TEST_F(WaveProfileTest, RejectConflictingFunctionRecords) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  MDNode *First = F.getMetadata(LLVMContext::MD_wave_profile);
  setBlockWaveCounts(F, {200, 20, 180, 200});
  MDNode *Second = F.getMetadata(LLVMContext::MD_wave_profile);
  for (bool Reverse : {false, true}) {
    F.setMetadata(LLVMContext::MD_wave_profile, Reverse ? Second : First);
    F.addMetadata(LLVMContext::MD_wave_profile, Reverse ? *First : *Second);
    SmallVector<uint64_t> Counts{99};
    BitVector HasCounts;
    uint64_t EntryCount = 99;
    EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &HasCounts));
    EXPECT_TRUE(Counts.empty());
    EXPECT_FALSE(
        extractMappedBlockWaveCounts(F, Counts, HasCounts, EntryCount));
    EXPECT_TRUE(Counts.empty());
    EXPECT_TRUE(HasCounts.empty());
    EXPECT_EQ(EntryCount, 0u);
    EXPECT_FALSE(verifyModule(*M, &errs()));
  }
}

TEST_F(WaveProfileTest, RejectSpecializedFunctionNames) {
  Function &F = *M->getFunction("diamond");
  for (StringRef Name : {"compute", "compute.llvm.123", "compute.__uniq.123",
                         "compute.content.123"}) {
    SCOPED_TRACE(Name.str());
    F.setName(Name);
    setBlockWaveCounts(F, {100, 10, 90, 100});
    SmallVector<uint64_t> Counts;
    BitVector Valid;
    uint64_t Entry;
    ASSERT_TRUE(extractBlockWaveCounts(F, Counts));
    ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
    // Function specialization appends this suffix after any promotion suffix.
    ValueToValueMapTy VMap;
    Function *Clone = CloneFunction(&F, VMap);
    Clone->setName(Name + ".specialized.1");
    EXPECT_FALSE(extractBlockWaveCounts(*Clone, Counts));
    EXPECT_TRUE(Counts.empty());
    EXPECT_FALSE(extractMappedBlockWaveCounts(*Clone, Counts, Valid, Entry));
    EXPECT_TRUE(Counts.empty());
    EXPECT_TRUE(Valid.empty());
    EXPECT_EQ(Entry, 0u);
    EXPECT_FALSE(verifyFunction(*Clone, &errs()));
    Clone->eraseFromParent();
  }
}

TEST_F(WaveProfileTest, TransferRespectsNestedInvalidation) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Outer(F);
  BlockWaveCountPreserver Inner(F);
  Inner.invalidate(*F.getEntryBlock().getNextNode());
  Inner.restore();
  Outer.restore();

  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 10, 90, 100}));
  EXPECT_EQ(Valid, makeBitVector(4, {0, 2, 3}));
}

TEST_F(WaveProfileTest, TransferRespectsProfileRemoval) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  clearBlockWaveCounts(F);
  Profile.restore();
  EXPECT_FALSE(F.hasMetadata(LLVMContext::MD_wave_profile));
  for (BasicBlock &BB : F)
    EXPECT_FALSE(
        BB.getTerminator()->hasMetadata(LLVMContext::MD_wave_profile_block));
}

TEST_F(WaveProfileTest, TransferRespectsProfileReplacement) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  setBlockWaveCounts(F, {200, 20, 180, 200});
  Profile.restore();
  SmallVector<uint64_t> Counts;
  ASSERT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{200, 20, 180, 200}));
}

TEST_F(WaveProfileTest, TransferRespectsBlockMetadataRemoval) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  Instruction *LeftTerm = F.getEntryBlock().getNextNode()->getTerminator();
  LeftTerm->setMetadata(LLVMContext::MD_wave_profile_block, nullptr);
  Profile.restore();
  EXPECT_FALSE(LeftTerm->hasMetadata(LLVMContext::MD_wave_profile_block));
  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_FALSE(Valid[1]);
}

TEST_F(WaveProfileTest, TransferAcrossTerminatorReplacement) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  Instruction *OldTerm = Left->getTerminator();
  UncondBrInst::Create(&F.back(), OldTerm->getIterator());
  OldTerm->eraseFromParent();
  Profile.restore();
  SmallVector<uint64_t> Counts;
  ASSERT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{100, 10, 90, 100}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, RejectMalformedBlockRecordsAndIgnoreTheirSwap) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  auto *Br = cast<CondBrInst>(F.getEntryBlock().getTerminator());
  MDNode *Original = Br->getMetadata(LLVMContext::MD_wave_profile_block);
  // All record fields are i64, including IDs and the measured flag.
  for (unsigned I = 0; I != Original->getNumOperands(); ++I) {
    for (Metadata *Bad :
         {static_cast<Metadata *>(nullptr),
          static_cast<Metadata *>(ConstantAsMetadata::get(
              ConstantInt::get(Type::getInt32Ty(Context), 0)))}) {
      SCOPED_TRACE(I);
      SmallVector<Metadata *> Ops(Original->op_begin(), Original->op_end());
      Ops[I] = Bad;
      MDNode *Malformed = MDNode::get(Context, Ops);
      Br->setMetadata(LLVMContext::MD_wave_profile_block, Malformed);
      SmallVector<uint64_t> Counts{99};
      BitVector Valid;
      uint64_t Entry = 99;
      EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &Valid));
      EXPECT_TRUE(Counts.empty());
      EXPECT_FALSE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
      EXPECT_TRUE(Counts.empty());
      EXPECT_TRUE(Valid.empty());
      EXPECT_EQ(Entry, 0u);
      Br->swapSuccessors();
      EXPECT_EQ(Br->getMetadata(LLVMContext::MD_wave_profile_block), Malformed);
      Br->swapSuccessors();
    }
  }
}

TEST_F(WaveProfileTest, RejectUnsupportedBlockVersionAndInvalidFlag) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  auto *Br = cast<CondBrInst>(F.getEntryBlock().getTerminator());
  MDNode *Original = Br->getMetadata(LLVMContext::MD_wave_profile_block);
  for (unsigned I : {0u, 3u}) {
    SmallVector<Metadata *> Ops(Original->op_begin(), Original->op_end());
    Ops[I] =
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Context), 3));
    MDNode *Unsupported = MDNode::get(Context, Ops);
    Br->setMetadata(LLVMContext::MD_wave_profile_block, Unsupported);
    SmallVector<uint64_t> Counts;
    BitVector Valid;
    uint64_t Entry;
    EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &Valid));
    EXPECT_FALSE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
    Br->swapSuccessors();
    EXPECT_EQ(Br->getMetadata(LLVMContext::MD_wave_profile_block), Unsupported);
    Br->swapSuccessors();
    if (I == 0)
      EXPECT_FALSE(verifyModule(*M, &errs()));
  }
}

TEST_F(WaveProfileTest, TransferRespectsMetadataOnReplacementTerminator) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  Instruction *OldTerm = Left->getTerminator();
  MDNode *MD = OldTerm->getMetadata(LLVMContext::MD_wave_profile_block);
  SmallVector<Metadata *> Ops(MD->op_begin(), MD->op_end());
  Ops[3] =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(Context), 0));
  auto *NewTerm = UncondBrInst::Create(&F.back(), OldTerm->getIterator());
  NewTerm->setMetadata(LLVMContext::MD_wave_profile_block,
                       MDNode::get(Context, Ops));
  OldTerm->eraseFromParent();
  Profile.restore();
  SmallVector<uint64_t> Counts;
  BitVector Valid;
  ASSERT_TRUE(extractBlockWaveCounts(F, Counts, &Valid));
  EXPECT_EQ(Valid, makeBitVector(4, {0, 2, 3}));
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferKeepsInvalidationsAfterSuccessorSwap) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  Profile.invalidate(*F.getEntryBlock().getNextNode());
  auto *Br = cast<CondBrInst>(F.getEntryBlock().getTerminator());
  IRBuilder<> Builder(Br);
  Br->setCondition(Builder.CreateNot(Br->getCondition()));
  Br->swapSuccessors();
  Profile.restore();

  expectUnmeasured(F, 1);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferKeepsInvalidationsAfterNestedRestore) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BlockWaveCountPreserver Outer(F);
  Outer.invalidate(*Left->getNextNode());
  BlockWaveCountPreserver Inner(F);
  Inner.invalidate(*Left);
  Inner.restore();
  Outer.restore();

  expectUnmeasured(F, 1);
  expectUnmeasured(F, 2);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest,
       TransferRespectsInvalidationBeforeTerminatorReplacement) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  BlockWaveCountPreserver Outer(F);
  BlockWaveCountPreserver Inner(F);
  Inner.invalidate(*Left);
  Inner.restore();
  Instruction *OldTerm = Left->getTerminator();
  UncondBrInst::Create(&F.back(), OldTerm->getIterator());
  OldTerm->eraseFromParent();
  Outer.restore();

  expectUnmeasured(F, 1);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferRespectsInPlaceMetadataChanges) {
  Function &F = *M->getFunction("diamond");
  for (bool ReplaceTerminator : {false, true}) {
    SCOPED_TRACE(ReplaceTerminator);
    setBlockWaveCounts(F, {100, 10, 90, 100});
    BlockWaveCountPreserver Profile(F);
    Instruction *OldTerm = F.getEntryBlock().getNextNode()->getTerminator();
    MDNode *MD = OldTerm->getMetadata(LLVMContext::MD_wave_profile_block);
    MD->replaceOperandWith(3, ConstantAsMetadata::get(ConstantInt::get(
                                  Type::getInt64Ty(Context), 0)));
    if (ReplaceTerminator) {
      UncondBrInst::Create(&F.back(), OldTerm->getIterator());
      OldTerm->eraseFromParent();
    }
    Profile.restore();
    expectUnmeasured(F, 1);
    EXPECT_FALSE(verifyModule(*M, &errs()));
  }
}

TEST_F(WaveProfileTest, TransferKeepsInvalidationsAfterTableChange) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  Profile.invalidate(*F.getEntryBlock().getNextNode());
  F.getMetadata(LLVMContext::MD_wave_profile)
      ->replaceOperandWith(2, ConstantAsMetadata::get(ConstantInt::get(
                                  Type::getInt64Ty(Context), 200)));
  Profile.restore();
  expectUnmeasured(F, 1);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferKeepsInvalidationsAfterProfileReplacement) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  Profile.invalidate(*F.getEntryBlock().getNextNode());
  setBlockWaveCounts(F, {200, 20, 180, 200});
  Profile.restore();
  expectUnmeasured(F, 1);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferConsumesSnapshot) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  Profile.invalidate(*F.getEntryBlock().getNextNode());
  Profile.restore();
  Profile.restore();
  SmallVector<uint64_t> Counts;
  BitVector Valid;
  uint64_t Entry;
  ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, Entry));
  EXPECT_EQ(Valid, makeBitVector(4, {0, 2, 3}));
  setBlockWaveCounts(F, {200, 20, 180, 200});
  Profile.restore();
  ASSERT_TRUE(extractBlockWaveCounts(F, Counts));
  EXPECT_EQ(Counts, (SmallVector<uint64_t>{200, 20, 180, 200}));
}

TEST_F(WaveProfileTest,
       TransferRespectsReattachedProfileAfterTerminatorReplacement) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 0, 90, 100});
  BlockWaveCountPreserver Profile(F);
  setBlockWaveCounts(F, {100, 0, 90, 100}, makeBitVector(4, {0, 2, 3}));
  Instruction *OldTerm = F.getEntryBlock().getNextNode()->getTerminator();
  UncondBrInst::Create(&F.back(), OldTerm->getIterator());
  OldTerm->eraseFromParent();
  Profile.restore();
  expectUnmeasured(F, 1);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST_F(WaveProfileTest, TransferInvalidatesDiscardedMetadataEdit) {
  Function &F = *M->getFunction("diamond");
  setBlockWaveCounts(F, {100, 10, 90, 100});
  BlockWaveCountPreserver Profile(F);
  BasicBlock *Left = F.getEntryBlock().getNextNode();
  Instruction *OldTerm = Left->getTerminator();
  OldTerm->setMetadata(LLVMContext::MD_wave_profile_block, nullptr);
  Profile.invalidate(*Left);
  UncondBrInst::Create(&F.back(), OldTerm->getIterator());
  OldTerm->eraseFromParent();
  Profile.restore();
  expectUnmeasured(F, 1);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

} // namespace
