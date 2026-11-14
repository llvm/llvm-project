//===- MetadataUtilsTest.cpp - Metadata tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MetadataUtils.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MemCacheHint.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class MetadataUtilsTest : public testing::Test {
protected:
  LLVMContext C;
  Module M{"test", C};
  IRBuilder<> B{C};
  Value *P;
  Value *Q;

  MetadataUtilsTest() {
    M.setDataLayout("e-p:64:64");
    auto *F = Function::Create(
        FunctionType::get(B.getVoidTy(), {B.getPtrTy(), B.getPtrTy()}, false),
        Function::ExternalLinkage, "test", M);
    P = F->getArg(0);
    Q = F->getArg(1);
    B.SetInsertPoint(BasicBlock::Create(C, "entry", F));
    B.SetInsertPoint(B.CreateRetVoid());
  }

  void TearDown() override { EXPECT_FALSE(verifyModule(M, &errs())); }

  MDNode *empty() { return MDNode::get(C, {}); }
  Metadata *number(uint64_t N) {
    return ConstantAsMetadata::get(B.getInt64(N));
  }
  MDNode *hint() {
    return MDNode::get(
        C, {MDString::get(C, "nvvm.l1_eviction"), MDString::get(C, "last")});
  }
  MDNode *range() {
    return MDNode::get(C, {ConstantAsMetadata::get(B.getInt32(0)),
                           ConstantAsMetadata::get(B.getInt32(100))});
  }
  MDNode *scope() {
    MDBuilder MDB(C);
    return MDNode::get(C, {MDB.createAnonymousAliasScope(
                              MDB.createAnonymousAliasScopeDomain())});
  }
};

TEST_F(MetadataUtilsTest, NarrowedLoadPreservesElementFacts) {
  auto *Old = B.CreateLoad(FixedVectorType::get(B.getInt32Ty(), 4), P);
  auto *Ptr = B.CreateInBoundsGEP(B.getInt32Ty(), P, B.getInt64(2));
  auto *New = B.CreateLoad(B.getInt32Ty(), Ptr);
  auto *Range = range();
  auto *Hint = hint();
  auto *Scope = scope();
  Old->setMetadata(LLVMContext::MD_range, Range);
  Old->setMetadata(LLVMContext::MD_noundef, empty());
  Old->setMetadata(LLVMContext::MD_invariant_load, empty());
  Old->setMetadata(LLVMContext::MD_invariant_group, empty());
  Old->setMetadata(LLVMContext::MD_alias_scope, Scope);
  setMemCacheHint(*Old, 0, Hint);
  copyMetadataForMemoryAccess(*New, *Old, 8);
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_range), Range);
  EXPECT_TRUE(New->hasMetadata(LLVMContext::MD_noundef));
  EXPECT_TRUE(New->hasMetadata(LLVMContext::MD_invariant_load));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_invariant_group));
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_alias_scope), Scope);
  EXPECT_EQ(getMemCacheHint(*New), Hint);
}

TEST_F(MetadataUtilsTest, InvariantGroupRequiresPointerIdentity) {
  auto *Old = B.CreateLoad(FixedVectorType::get(B.getInt32Ty(), 4), P);
  auto *Same = B.CreateLoad(B.getInt32Ty(), P);
  auto *Different = B.CreateLoad(B.getInt32Ty(), Q);
  Old->setMetadata(LLVMContext::MD_invariant_group, empty());
  copyMetadataForMemoryAccess(*Same, *Old);
  copyMetadataForMemoryAccess(*Different, *Old);
  EXPECT_TRUE(Same->hasMetadata(LLVMContext::MD_invariant_group));
  EXPECT_FALSE(Different->hasMetadata(LLVMContext::MD_invariant_group));
}

TEST_F(MetadataUtilsTest, IntegerFragmentPreservesDefinednessButNotRange) {
  auto *Old = B.CreateLoad(B.getInt64Ty(), P);
  MDBuilder MDB(C);
  Old->setMetadata(LLVMContext::MD_noundef, empty());
  Old->setMetadata(LLVMContext::MD_range,
                   MDB.createRange(B.getInt64(0), B.getInt64(100)));
  for (int64_t Offset : {0, 4}) {
    auto *Ptr = B.CreateInBoundsGEP(B.getInt8Ty(), P, B.getInt64(Offset));
    auto *New = B.CreateLoad(B.getInt32Ty(), Ptr);
    copyMetadataForMemoryAccess(*New, *Old, Offset);
    EXPECT_TRUE(New->hasMetadata(LLVMContext::MD_noundef));
    EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_range));
  }
}

TEST_F(MetadataUtilsTest, WideningDropsFactsAboutOriginalBytes) {
  auto *Old = B.CreateLoad(B.getInt32Ty(), P);
  auto *New = B.CreateLoad(FixedVectorType::get(B.getInt32Ty(), 4), P);
  auto *Hint = hint();
  auto *Annotation = MDNode::get(C, {MDString::get(C, "original")});
  Old->setMetadata(LLVMContext::MD_range, range());
  Old->setMetadata(LLVMContext::MD_noundef, empty());
  Old->setMetadata(LLVMContext::MD_invariant_load, empty());
  Old->setMetadata(LLVMContext::MD_invariant_group, empty());
  Old->setMetadata(LLVMContext::MD_alias_scope, scope());
  Old->setMetadata(LLVMContext::MD_annotation, Annotation);
  Old->setMetadata("test.unknown", empty());
  setMemCacheHint(*Old, 0, Hint);
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_range));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_noundef));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_invariant_load));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_invariant_group));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_alias_scope));
  EXPECT_FALSE(New->getMetadata("test.unknown"));
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_annotation), Annotation);
  EXPECT_EQ(getMemCacheHint(*New), Hint);
}

TEST_F(MetadataUtilsTest, EarlierAddressIsNotASubset) {
  auto *Ptr = B.CreateInBoundsGEP(B.getInt32Ty(), P, B.getInt64(2));
  auto *Old = B.CreateLoad(FixedVectorType::get(B.getInt32Ty(), 4), Ptr);
  auto *New = B.CreateLoad(B.getInt32Ty(), P);
  Old->setMetadata(LLVMContext::MD_range, range());
  copyMetadataForMemoryAccess(*New, *Old, -8);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_range));
}

TEST_F(MetadataUtilsTest, BitcastDropsDifferentIntegerElementRange) {
  auto *Old = B.CreateLoad(FixedVectorType::get(B.getInt32Ty(), 2), P);
  auto *New = B.CreateLoad(B.getInt64Ty(), P);
  Old->setMetadata(LLVMContext::MD_range, range());
  Old->setMetadata(LLVMContext::MD_noundef, empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_range));
  EXPECT_TRUE(New->hasMetadata(LLVMContext::MD_noundef));
}

TEST_F(MetadataUtilsTest, BitcastDropsFloatingPointValueFacts) {
  auto *Old = B.CreateLoad(FixedVectorType::get(B.getFloatTy(), 2), P);
  auto *New = B.CreateLoad(B.getInt64Ty(), P);
  Old->setMetadata(LLVMContext::MD_nofpclass,
                   MDNode::get(C, {ConstantAsMetadata::get(B.getInt32(3))}));
  Old->setMetadata(LLVMContext::MD_fpmath,
                   MDNode::get(C, {ConstantAsMetadata::get(
                                      ConstantFP::get(B.getFloatTy(), 2.5))}));
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_nofpclass));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_fpmath));
}

TEST_F(MetadataUtilsTest, PointerNonnullBecomesIntegerRange) {
  auto *Old = B.CreateLoad(B.getPtrTy(), P);
  auto *New = B.CreateLoad(B.getInt64Ty(), P);
  Old->setMetadata(LLVMContext::MD_nonnull, empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_nonnull));
  ASSERT_TRUE(New->hasMetadata(LLVMContext::MD_range));
  auto *Range = New->getMetadata(LLVMContext::MD_range);
  EXPECT_EQ(mdconst::extract<ConstantInt>(Range->getOperand(0))->getZExtValue(),
            1u);
  EXPECT_EQ(mdconst::extract<ConstantInt>(Range->getOperand(1))->getZExtValue(),
            0u);
}

TEST_F(MetadataUtilsTest, UnknownOffsetKeepsAccessTagsButDropsFieldLayout) {
  auto *Old = B.CreateLoad(FixedVectorType::get(B.getInt32Ty(), 4), P);
  auto *Index = B.CreateAnd(B.CreateLoad(B.getInt64Ty(), Q), B.getInt64(3));
  auto *Ptr = B.CreateInBoundsGEP(B.getInt32Ty(), P, Index);
  auto *New = B.CreateLoad(B.getInt32Ty(), Ptr);
  MDBuilder MDB(C);
  auto *Char = MDB.createTBAAScalarTypeNode("char", MDB.createTBAARoot("root"));
  auto *Tag = MDB.createTBAAStructTagNode(Char, Char, 0);
  auto *Scope = scope();
  Old->setMetadata(LLVMContext::MD_tbaa, Tag);
  Old->setMetadata(LLVMContext::MD_tbaa_struct,
                   MDNode::get(C, {number(0), number(4), Tag}));
  Old->setMetadata(LLVMContext::MD_alias_scope, Scope);
  copyMetadataForMemoryAccess(*New, *Old, std::nullopt);
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_tbaa), Tag);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_tbaa_struct));
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_alias_scope), Scope);
}

TEST_F(MetadataUtilsTest, ConstantOffsetSelectsFieldTag) {
  auto *Old = B.CreateLoad(FixedVectorType::get(B.getInt32Ty(), 2), P);
  auto *New = B.CreateLoad(B.getInt32Ty(), Q);
  MDBuilder MDB(C);
  auto *Root = MDB.createTBAARoot("root");
  auto *Char = MDB.createTBAAScalarTypeNode("char", Root);
  auto *Int = MDB.createTBAAScalarTypeNode("int", Char);
  auto *CharTag = MDB.createTBAAStructTagNode(Char, Char, 0);
  auto *IntTag = MDB.createTBAAStructTagNode(Int, Int, 0);
  Old->setMetadata(LLVMContext::MD_tbaa_struct,
                   MDNode::get(C, {number(0), number(4), CharTag, number(4),
                                   number(4), IntTag}));
  copyMetadataForMemoryAccess(*New, *Old, 4);
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_tbaa), IntTag);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_tbaa_struct));
}

TEST_F(MetadataUtilsTest, WholeAccessSelectsSingleFieldTag) {
  auto *Old = B.CreateLoad(B.getInt64Ty(), P);
  auto *New = B.CreateLoad(B.getDoubleTy(), P);
  MDBuilder MDB(C);
  auto *Char = MDB.createTBAAScalarTypeNode("char", MDB.createTBAARoot("root"));
  auto *Tag = MDB.createTBAAStructTagNode(Char, Char, 0);
  Old->setMetadata(LLVMContext::MD_tbaa_struct,
                   MDNode::get(C, {number(0), number(8), Tag}));
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_tbaa), Tag);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_tbaa_struct));
}

TEST_F(MetadataUtilsTest, WholeAccessRetainsMultipleFieldTags) {
  auto *Ty = FixedVectorType::get(B.getInt32Ty(), 2);
  auto *Old = B.CreateLoad(Ty, P);
  auto *New = B.CreateLoad(Ty, P);
  MDBuilder MDB(C);
  auto *Char = MDB.createTBAAScalarTypeNode("char", MDB.createTBAARoot("root"));
  auto *Int = MDB.createTBAAScalarTypeNode("int", Char);
  auto *CharTag = MDB.createTBAAStructTagNode(Char, Char, 0);
  auto *IntTag = MDB.createTBAAStructTagNode(Int, Int, 0);
  auto *Fields = MDNode::get(
      C, {number(0), number(4), CharTag, number(4), number(4), IntTag});
  Old->setMetadata(LLVMContext::MD_tbaa_struct, Fields);
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_tbaa));
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_tbaa_struct), Fields);
}

TEST_F(MetadataUtilsTest, MemoryTransferRemapsBothOperands) {
  auto *Copy = B.CreateMemCpy(P, Align(4), Q, Align(4), B.getInt64(4));
  auto *Read = B.CreateLoad(B.getInt32Ty(), Q);
  auto *Write = B.CreateStore(B.getInt32(1), P);
  auto *ReadHint = hint();
  auto *WriteHint = MDNode::get(
      C, {MDString::get(C, "nvvm.l1_eviction"), MDString::get(C, "first")});
  setMemCacheHint(*Copy, 0, WriteHint);
  setMemCacheHint(*Copy, 1, ReadHint);
  copyMetadataForMemoryAccess(*Read, *Copy);
  copyMetadataForMemoryAccess(*Write, *Copy);
  EXPECT_EQ(getMemCacheHint(*Read, 0), ReadHint);
  EXPECT_EQ(getMemCacheHint(*Write, 1), WriteHint);
  EXPECT_FALSE(getMemCacheHint(*Read, 1));
  EXPECT_FALSE(getMemCacheHint(*Write, 0));
}

TEST_F(MetadataUtilsTest, RuntimeMemoryTransferLength) {
  auto *Length = B.CreateLoad(B.getInt64Ty(), Q);
  B.CreateAssumption(B.CreateICmpUGE(Length, B.getInt64(8)));
  auto *Copy = B.CreateMemCpy(P, Align(4), Q, Align(4), Length);
  auto *Ptr = B.CreateInBoundsGEP(B.getInt8Ty(), Q, B.getInt64(4));
  auto *Read = B.CreateLoad(B.getInt32Ty(), Ptr);
  auto *Hint = hint();
  setMemCacheHint(*Copy, 1, Hint);
  copyMetadataForMemoryAccess(*Read, *Copy, 4);
  EXPECT_EQ(getMemCacheHint(*Read), Hint);
}

TEST_F(MetadataUtilsTest, StoreTypeChangePreservesHints) {
  auto *Old = B.CreateStore(B.getInt32(0), P);
  auto *New = B.CreateStore(ConstantFP::get(B.getFloatTy(), 0), P);
  auto *Hint = hint();
  setMemCacheHint(*Old, 1, Hint);
  Old->setMetadata(LLVMContext::MD_invariant_group, empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_EQ(getMemCacheHint(*New, 1), Hint);
  EXPECT_TRUE(New->hasMetadata(LLVMContext::MD_invariant_group));
}

TEST_F(MetadataUtilsTest, NonMemoryScalarizationPreservesFPMath) {
  auto *Ty = FixedVectorType::get(B.getFloatTy(), 2);
  auto *Old =
      cast<Instruction>(B.CreateFAdd(B.CreateLoad(Ty, P), B.CreateLoad(Ty, Q)));
  auto *New = cast<Instruction>(B.CreateFAdd(B.CreateLoad(B.getFloatTy(), P),
                                             B.CreateLoad(B.getFloatTy(), Q)));
  auto *Tag = MDNode::get(
      C, {ConstantAsMetadata::get(ConstantFP::get(B.getFloatTy(), 2.5))});
  Old->setMetadata(LLVMContext::MD_fpmath, Tag);
  copyMetadataForScalarizedInstruction(*New, *Old);
  EXPECT_EQ(New->getMetadata(LLVMContext::MD_fpmath), Tag);
}

TEST_F(MetadataUtilsTest, CSEIntersectsCacheHints) {
  auto *A = B.CreateLoad(B.getInt32Ty(), P);
  auto *BLoad = B.CreateLoad(B.getInt32Ty(), P);
  auto *Hint = hint();
  setMemCacheHint(*A, 0, Hint);
  setMemCacheHint(*BLoad, 0, Hint);
  combineMetadataForCSE(A, BLoad, false);
  EXPECT_EQ(getMemCacheHint(*A), Hint);
  BLoad->setMetadata(LLVMContext::MD_mem_cache_hint, nullptr);
  combineMetadataForCSE(A, BLoad, false);
  EXPECT_FALSE(getMemCacheHint(*A));
}

TEST_F(MetadataUtilsTest, AddressSpaceCastChangesInvariantGroupIdentity) {
  auto *Old = B.CreateLoad(B.getInt32Ty(), P);
  auto *NewPtr = B.CreateAddrSpaceCast(P, B.getPtrTy(1));
  auto *New = B.CreateLoad(B.getInt32Ty(), NewPtr);
  Old->setMetadata(LLVMContext::MD_invariant_group, empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_invariant_group));
}

TEST_F(MetadataUtilsTest, DefinedAggregateDoesNotDefinePadding) {
  auto *Ty = StructType::get(C, {B.getInt8Ty(), B.getInt32Ty()});
  ASSERT_EQ(M.getDataLayout().getTypeStoreSize(Ty).getFixedValue(), 8u);
  auto *Old = B.CreateLoad(Ty, P);
  auto *New = B.CreateLoad(B.getInt64Ty(), P);
  Old->setMetadata(LLVMContext::MD_noundef, empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_noundef));
}

TEST_F(MetadataUtilsTest, DefinedIntegerDoesNotDefineSourcePadding) {
  auto *Old = B.CreateLoad(B.getInt1Ty(), P);
  auto *New = B.CreateLoad(B.getInt8Ty(), P);
  Old->setMetadata(LLVMContext::MD_noundef, empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_noundef));
}

TEST_F(MetadataUtilsTest, DestinationPaddingDoesNotInvalidateDefinedness) {
  auto *Old = B.CreateLoad(B.getInt8Ty(), P);
  auto *New = B.CreateLoad(B.getInt1Ty(), P);
  Old->setMetadata(LLVMContext::MD_noundef, empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_TRUE(New->hasMetadata(LLVMContext::MD_noundef));
}

TEST_F(MetadataUtilsTest, NonnullDoesNotConstrainTruncatedPointerBits) {
  auto *Old = B.CreateLoad(B.getPtrTy(), P);
  auto *New = B.CreateLoad(B.getIntNTy(63), P);
  Old->setMetadata(LLVMContext::MD_nonnull, empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_range));
}

TEST_F(MetadataUtilsTest, IntegerRangeBecomesPointerNonnull) {
  auto *Old = B.CreateLoad(B.getInt64Ty(), P);
  auto *New = B.CreateLoad(B.getPtrTy(), P);
  MDBuilder MDB(C);
  Old->setMetadata(LLVMContext::MD_range,
                   MDB.createRange(B.getInt64(1), B.getInt64(100)));
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_TRUE(New->hasMetadata(LLVMContext::MD_nonnull));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_range));
}

TEST_F(MetadataUtilsTest, PointerFactsRequireSamePointerType) {
  auto *Old = B.CreateLoad(B.getPtrTy(), P);
  auto *New = B.CreateLoad(B.getPtrTy(1), P);
  Old->setMetadata(LLVMContext::MD_nonnull, empty());
  Old->setMetadata(LLVMContext::MD_align, MDNode::get(C, {number(16)}));
  Old->setMetadata(LLVMContext::MD_dereferenceable,
                   MDNode::get(C, {number(32)}));
  Old->setMetadata(LLVMContext::MD_dereferenceable_or_null,
                   MDNode::get(C, {number(32)}));
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_nonnull));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_align));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_dereferenceable));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_dereferenceable_or_null));
}

TEST_F(MetadataUtilsTest, PointerFactsRequirePointerResults) {
  auto *Old = B.CreateLoad(B.getPtrTy(), P);
  auto *Pointer = B.CreateLoad(B.getPtrTy(), P);
  auto *Integer = B.CreateLoad(B.getInt64Ty(), P);
  auto *Align = MDNode::get(C, {number(16)});
  auto *Deref = MDNode::get(C, {number(32)});
  Old->setMetadata(LLVMContext::MD_nonnull, empty());
  Old->setMetadata(LLVMContext::MD_align, Align);
  Old->setMetadata(LLVMContext::MD_dereferenceable, Deref);
  Old->setMetadata(LLVMContext::MD_dereferenceable_or_null, Deref);
  copyMetadataForMemoryAccess(*Pointer, *Old);
  copyMetadataForMemoryAccess(*Integer, *Old);
  EXPECT_TRUE(Pointer->hasMetadata(LLVMContext::MD_nonnull));
  EXPECT_EQ(Pointer->getMetadata(LLVMContext::MD_align), Align);
  EXPECT_EQ(Pointer->getMetadata(LLVMContext::MD_dereferenceable), Deref);
  EXPECT_EQ(Pointer->getMetadata(LLVMContext::MD_dereferenceable_or_null),
            Deref);
  EXPECT_FALSE(Integer->hasMetadata(LLVMContext::MD_nonnull));
  EXPECT_FALSE(Integer->hasMetadata(LLVMContext::MD_align));
  EXPECT_FALSE(Integer->hasMetadata(LLVMContext::MD_dereferenceable));
  EXPECT_FALSE(Integer->hasMetadata(LLVMContext::MD_dereferenceable_or_null));
  EXPECT_TRUE(Integer->hasMetadata(LLVMContext::MD_range));
}

TEST_F(MetadataUtilsTest, StoreFactsRequireWholeAccessAndPointerValue) {
  auto *Old = B.CreateStore(Q, P);
  auto *Pointer = B.CreateStore(Q, P);
  auto *Integer = B.CreateStore(B.CreatePtrToInt(Q, B.getInt64Ty()), P);
  auto *Part = B.CreateStore(B.getInt32(0), P);
  auto *AssignID = DIAssignID::getDistinct(C);
  auto *Captures = MDNode::get(C, {MDString::get(C, "address")});
  Old->setMetadata(LLVMContext::MD_DIAssignID, AssignID);
  Old->setMetadata(LLVMContext::MD_captures, Captures);
  copyMetadataForMemoryAccess(*Pointer, *Old);
  copyMetadataForMemoryAccess(*Integer, *Old);
  copyMetadataForMemoryAccess(*Part, *Old);
  EXPECT_EQ(Pointer->getMetadata(LLVMContext::MD_DIAssignID), AssignID);
  EXPECT_EQ(Pointer->getMetadata(LLVMContext::MD_captures), Captures);
  EXPECT_EQ(Integer->getMetadata(LLVMContext::MD_DIAssignID), AssignID);
  EXPECT_FALSE(Integer->hasMetadata(LLVMContext::MD_captures));
  EXPECT_FALSE(Part->hasMetadata(LLVMContext::MD_DIAssignID));
  EXPECT_FALSE(Part->hasMetadata(LLVMContext::MD_captures));
}

TEST_F(MetadataUtilsTest, IntrinsicProfilesAndUnknownMetadataAreNotCopied) {
  auto *Ty = FixedVectorType::get(B.getInt32Ty(), 2);
  auto *Mask = B.CreateLoad(FixedVectorType::get(B.getInt1Ty(), 2), Q);
  auto *Old = B.CreateMaskedLoad(Ty, P, Align(4), Mask);
  auto *New = B.CreateLoad(Ty, P);
  auto *Profile = MDNode::get(C, {MDString::get(C, "branch_weights"),
                                  ConstantAsMetadata::get(B.getInt32(10))});
  Old->setMetadata(LLVMContext::MD_prof, Profile);
  Old->setMetadata("test.unknown", empty());
  copyMetadataForMemoryAccess(*New, *Old);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_prof));
  EXPECT_FALSE(New->getMetadata("test.unknown"));
}

TEST_F(MetadataUtilsTest, SmallerAccessOutsideSourceRangeIsNotASubset) {
  auto *Old = B.CreateLoad(FixedVectorType::get(B.getInt32Ty(), 4), P);
  auto *Ptr = B.CreateInBoundsGEP(B.getInt32Ty(), P, B.getInt64(4));
  auto *New = B.CreateLoad(B.getInt32Ty(), Ptr);
  Old->setMetadata(LLVMContext::MD_range, range());
  Old->setMetadata(LLVMContext::MD_noundef, empty());
  Old->setMetadata(LLVMContext::MD_invariant_load, empty());
  Old->setMetadata(LLVMContext::MD_alias_scope, scope());
  Old->setMetadata(LLVMContext::MD_noalias, scope());
  copyMetadataForMemoryAccess(*New, *Old, 16);
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_range));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_noundef));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_invariant_load));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_alias_scope));
  EXPECT_FALSE(New->hasMetadata(LLVMContext::MD_noalias));
}

} // namespace
