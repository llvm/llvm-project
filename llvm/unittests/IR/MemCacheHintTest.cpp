//===- MemCacheHintTest.cpp - Cache hint metadata tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/MemCacheHint.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MemCacheHintTest : public testing::Test {
protected:
  LLVMContext Context;
  Module M{"test", Context};
  IRBuilder<> Builder{Context};
  Value *Dest;
  Value *Source;

  MemCacheHintTest() {
    Type *PtrTy = Builder.getPtrTy();
    auto *F = Function::Create(
        FunctionType::get(Builder.getVoidTy(), {PtrTy, PtrTy}, false),
        Function::ExternalLinkage, "test", M);
    auto Arg = F->arg_begin();
    Dest = &*Arg++;
    Source = &*Arg;
    Builder.SetInsertPoint(BasicBlock::Create(Context, "entry", F));
    auto *Ret = Builder.CreateRetVoid();
    Builder.SetInsertPoint(Ret);
  }

  void TearDown() override { EXPECT_FALSE(verifyModule(M, &errs())); }

  MDNode *hint(StringRef Value) {
    return MDNode::get(Context, {MDString::get(Context, "test.cache"),
                                 MDString::get(Context, Value)});
  }

  Metadata *index(unsigned I) {
    return ConstantAsMetadata::get(Builder.getInt32(I));
  }

  MDNode *attachment(unsigned I, MDNode *Hint) {
    return MDNode::get(Context, {index(I), Hint});
  }

  MDNode *twoHints(MDNode *DestHint, MDNode *SourceHint) {
    return MDNode::get(Context, {index(0), DestHint, index(1), SourceHint});
  }

  CallInst *createMemCpy() {
    return Builder.CreateMemCpy(Dest, Align(4), Source, Align(4),
                                Builder.getInt64(4));
  }
};

TEST_F(MemCacheHintTest, GetLoadHint) {
  auto *Load = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  EXPECT_EQ(getMemCacheHint(*Load), nullptr);
  MDNode *Hint = hint("load");
  Load->setMetadata(LLVMContext::MD_mem_cache_hint, attachment(0, Hint));
  EXPECT_EQ(getMemCacheHint(*Load), Hint);
  EXPECT_EQ(getMemCacheHint(*Load, 1), nullptr);
}

TEST_F(MemCacheHintTest, GetStoreHint) {
  auto *Store = Builder.CreateStore(Builder.getInt32(1), Dest);
  MDNode *Hint = hint("store");
  Store->setMetadata(LLVMContext::MD_mem_cache_hint, attachment(1, Hint));
  EXPECT_EQ(getMemCacheHint(*Store, 1), Hint);
  EXPECT_EQ(getMemCacheHint(*Store), nullptr);
}

TEST_F(MemCacheHintTest, GetIntrinsicHints) {
  auto *Copy = createMemCpy();
  MDNode *A = hint("destination"), *B = hint("source");
  Copy->setMetadata(LLVMContext::MD_mem_cache_hint, twoHints(A, B));
  EXPECT_EQ(getMemCacheHint(*Copy, 0), A);
  EXPECT_EQ(getMemCacheHint(*Copy, 1), B);
  EXPECT_EQ(getMemCacheHint(*Copy, 2), nullptr);
}

TEST_F(MemCacheHintTest, SetInOperandOrder) {
  auto *Copy = createMemCpy();
  MDNode *A = hint("destination"), *B = hint("source");
  setMemCacheHint(*Copy, 1, B);
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint),
            attachment(1, B));
  setMemCacheHint(*Copy, 0, A);
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint), twoHints(A, B));
}

TEST_F(MemCacheHintTest, SetAfterExistingOperand) {
  auto *Copy = createMemCpy();
  MDNode *A = hint("destination"), *B = hint("source");
  setMemCacheHint(*Copy, 0, A);
  setMemCacheHint(*Copy, 1, B);
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint), twoHints(A, B));
}

TEST_F(MemCacheHintTest, ReplacePreservesOtherHintsAndMetadata) {
  auto *Copy = createMemCpy();
  MDNode *A = hint("destination"), *B = hint("source"),
         *C = hint("replacement");
  Copy->setMetadata(LLVMContext::MD_mem_cache_hint, twoHints(A, B));
  MDNode *Other = MDNode::get(Context, {});
  Copy->setMetadata("test.other", Other);
  setMemCacheHint(*Copy, 0, C);
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint), twoHints(C, B));
  EXPECT_EQ(Copy->getMetadata("test.other"), Other);
}

TEST_F(MemCacheHintTest, RemovePreservesOtherOperand) {
  auto *Copy = createMemCpy();
  MDNode *A = hint("destination"), *B = hint("source");
  Copy->setMetadata(LLVMContext::MD_mem_cache_hint, twoHints(A, B));
  setMemCacheHint(*Copy, 0, nullptr);
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint),
            attachment(1, B));
  setMemCacheHint(*Copy, 1, nullptr);
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
}

TEST_F(MemCacheHintTest, RemoveAbsentHint) {
  auto *Copy = createMemCpy();
  setMemCacheHint(*Copy, 0, nullptr);
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
  MDNode *MD = attachment(1, hint("source"));
  Copy->setMetadata(LLVMContext::MD_mem_cache_hint, MD);
  setMemCacheHint(*Copy, 0, nullptr);
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint), MD);
}

TEST_F(MemCacheHintTest, CommonReadHints) {
  auto *Ty = FixedVectorType::get(Builder.getInt32Ty(), 4);
  Value *Mask = Builder.CreateVectorSplat(4, Builder.getTrue());
  Value *Pointers = Builder.CreateVectorSplat(4, Source);
  auto *Load = Builder.CreateLoad(Ty, Source);
  auto *MaskedLoad = Builder.CreateMaskedLoad(Ty, Source, Align(4), Mask);
  auto *Gather = Builder.CreateMaskedGather(Ty, Pointers, Align(4), Mask);
  Value *Sources[] = {Load, MaskedLoad, Gather};
  MDNode *Hint = hint("read");
  for (Value *V : Sources)
    setMemCacheHint(*cast<Instruction>(V), 0, Hint);
  auto *Target = Builder.CreateLoad(Ty, Dest);
  propagateMemCacheHint(*Target, Sources);
  EXPECT_EQ(Target->getMetadata(LLVMContext::MD_mem_cache_hint),
            attachment(0, Hint));
}

TEST_F(MemCacheHintTest, CommonWriteHints) {
  Value *Data = Builder.CreateVectorSplat(4, Builder.getInt32(1));
  Value *Mask = Builder.CreateVectorSplat(4, Builder.getTrue());
  Value *Pointers = Builder.CreateVectorSplat(4, Dest);
  auto *Store = Builder.CreateStore(Data, Dest);
  auto *MaskedStore = Builder.CreateMaskedStore(Data, Dest, Align(4), Mask);
  auto *Scatter = Builder.CreateMaskedScatter(Data, Pointers, Align(4), Mask);
  Value *Sources[] = {Store, MaskedStore, Scatter};
  MDNode *Hint = hint("write");
  for (Value *V : Sources)
    setMemCacheHint(*cast<Instruction>(V), 1, Hint);
  auto *Target = Builder.CreateStore(Data, Dest);
  setMemCacheHint(*Target, 1, hint("old"));
  propagateMemCacheHint(*Target, Sources);
  EXPECT_EQ(Target->getMetadata(LLVMContext::MD_mem_cache_hint),
            attachment(1, Hint));
}

TEST_F(MemCacheHintTest, MissingOrConflictingHints) {
  auto *A = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  auto *B = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  auto *Target = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  Value *Sources[] = {A, B};
  MDNode *Hint = hint("read");
  setMemCacheHint(*A, 0, Hint);
  setMemCacheHint(*Target, 0, hint("old"));
  propagateMemCacheHint(*Target, Sources);
  EXPECT_EQ(Target->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
  setMemCacheHint(*B, 0, hint("different"));
  setMemCacheHint(*Target, 0, hint("old"));
  propagateMemCacheHint(*Target, Sources);
  EXPECT_EQ(Target->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
  setMemCacheHint(*B, 0, Hint);
  propagateMemCacheHint(*Target, Sources);
  EXPECT_EQ(Target->getMetadata(LLVMContext::MD_mem_cache_hint),
            attachment(0, Hint));
  setMemCacheHint(*A, 0, nullptr);
  propagateMemCacheHint(*Target, Sources);
  EXPECT_EQ(Target->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
}

TEST_F(MemCacheHintTest, ReadWriteMismatch) {
  auto *Load = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  auto *Store = Builder.CreateStore(Builder.getInt32(1), Dest);
  auto *TargetLoad = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  auto *TargetStore = Builder.CreateStore(Builder.getInt32(1), Dest);
  MDNode *Read = hint("read"), *Write = hint("write");
  setMemCacheHint(*Load, 0, Read);
  setMemCacheHint(*Store, 1, Write);
  setMemCacheHint(*TargetLoad, 0, Read);
  setMemCacheHint(*TargetStore, 1, Write);
  propagateMemCacheHint(*TargetLoad, {Store});
  propagateMemCacheHint(*TargetStore, {Load});
  EXPECT_EQ(TargetLoad->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
  EXPECT_EQ(TargetStore->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
}

TEST_F(MemCacheHintTest, UnsupportedAccess) {
  auto *Load = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  auto *Copy = createMemCpy();
  auto *Target = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  MDNode *Hint = hint("read");
  setMemCacheHint(*Load, 0, Hint);
  setMemCacheHint(*Copy, 0, Hint);
  propagateMemCacheHint(*Copy, {Load});
  EXPECT_EQ(Copy->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
  setMemCacheHint(*Copy, 0, Hint);
  setMemCacheHint(*Target, 0, Hint);
  propagateMemCacheHint(*Target, {Load, Copy});
  EXPECT_EQ(Target->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
}

TEST_F(MemCacheHintTest, EmptyOrNonInstructionSources) {
  auto *Load = Builder.CreateLoad(Builder.getInt32Ty(), Source);
  MDNode *Hint = hint("read");
  setMemCacheHint(*Load, 0, Hint);
  propagateMemCacheHint(*Load, {});
  EXPECT_EQ(Load->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
  setMemCacheHint(*Load, 0, Hint);
  propagateMemCacheHint(*Load, {Builder.getInt32(0)});
  EXPECT_EQ(Load->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
  setMemCacheHint(*Load, 0, Hint);
  propagateMemCacheHint(*Load, {Load, Builder.getInt32(0)});
  EXPECT_EQ(Load->getMetadata(LLVMContext::MD_mem_cache_hint), nullptr);
}

} // namespace
