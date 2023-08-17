//===- ValueMapper.cpp - Unit tests for ValueMapper -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ValueMapperTest, mapMDNode) {
  LLVMContext Context;
  auto *U = MDTuple::get(Context, std::nullopt);

  // The node should be unchanged.
  ValueToValueMapTy VM;
  EXPECT_EQ(U, ValueMapper(VM).mapMDNode(*U));
}

TEST(ValueMapperTest, mapMDNodeCycle) {
  LLVMContext Context;
  MDNode *U0;
  MDNode *U1;
  {
    Metadata *Ops[] = {nullptr};
    auto T = MDTuple::getTemporary(Context, Ops);
    Ops[0] = T.get();
    U0 = MDTuple::get(Context, Ops);
    T->replaceOperandWith(0, U0);
    U1 = MDNode::replaceWithUniqued(std::move(T));
    U0->resolveCycles();
  }

  EXPECT_TRUE(U0->isResolved());
  EXPECT_TRUE(U0->isUniqued());
  EXPECT_TRUE(U1->isResolved());
  EXPECT_TRUE(U1->isUniqued());
  EXPECT_EQ(U1, U0->getOperand(0));
  EXPECT_EQ(U0, U1->getOperand(0));

  // Cycles shouldn't be duplicated.
  {
    ValueToValueMapTy VM;
    EXPECT_EQ(U0, ValueMapper(VM).mapMDNode(*U0));
    EXPECT_EQ(U1, ValueMapper(VM).mapMDNode(*U1));
  }

  // Check the other order.
  {
    ValueToValueMapTy VM;
    EXPECT_EQ(U1, ValueMapper(VM).mapMDNode(*U1));
    EXPECT_EQ(U0, ValueMapper(VM).mapMDNode(*U0));
  }
}

TEST(ValueMapperTest, mapMDNodeDuplicatedCycle) {
  LLVMContext Context;
  auto *PtrTy = PointerType::get(Context, 0);
  std::unique_ptr<GlobalVariable> G0 = std::make_unique<GlobalVariable>(
      PtrTy, false, GlobalValue::ExternalLinkage, nullptr, "G0");
  std::unique_ptr<GlobalVariable> G1 = std::make_unique<GlobalVariable>(
      PtrTy, false, GlobalValue::ExternalLinkage, nullptr, "G1");

  // Create a cycle that references G0.
  MDNode *N0; // !0 = !{!1}
  MDNode *N1; // !1 = !{!0, i8* @G0}
  {
    auto T0 = MDTuple::getTemporary(Context, nullptr);
    Metadata *Ops1[] = {T0.get(), ConstantAsMetadata::get(G0.get())};
    N1 = MDTuple::get(Context, Ops1);
    T0->replaceOperandWith(0, N1);
    N0 = MDNode::replaceWithUniqued(std::move(T0));
  }

  // Resolve N0 and N1.
  ASSERT_FALSE(N0->isResolved());
  ASSERT_FALSE(N1->isResolved());
  N0->resolveCycles();
  ASSERT_TRUE(N0->isResolved());
  ASSERT_TRUE(N1->isResolved());

  // Seed the value map to map G0 to G1 and map the nodes.  The output should
  // have new nodes that reference G1 (instead of G0).
  ValueToValueMapTy VM;
  VM[G0.get()] = G1.get();
  MDNode *MappedN0 = ValueMapper(VM).mapMDNode(*N0);
  MDNode *MappedN1 = ValueMapper(VM).mapMDNode(*N1);
  EXPECT_NE(N0, MappedN0);
  EXPECT_NE(N1, MappedN1);
  EXPECT_EQ(ConstantAsMetadata::get(G1.get()), MappedN1->getOperand(1));

  // Check that the output nodes are resolved.
  EXPECT_TRUE(MappedN0->isResolved());
  EXPECT_TRUE(MappedN1->isResolved());
}

TEST(ValueMapperTest, mapMDNodeUnresolved) {
  LLVMContext Context;
  TempMDTuple T = MDTuple::getTemporary(Context, std::nullopt);

  ValueToValueMapTy VM;
  EXPECT_EQ(T.get(), ValueMapper(VM, RF_NoModuleLevelChanges).mapMDNode(*T));
}

TEST(ValueMapperTest, mapMDNodeDistinct) {
  LLVMContext Context;
  auto *D = MDTuple::getDistinct(Context, std::nullopt);

  {
    // The node should be cloned.
    ValueToValueMapTy VM;
    EXPECT_NE(D, ValueMapper(VM).mapMDNode(*D));
  }
  {
    // The node should be moved.
    ValueToValueMapTy VM;
    EXPECT_EQ(D, ValueMapper(VM, RF_ReuseAndMutateDistinctMDs).mapMDNode(*D));
  }
}

TEST(ValueMapperTest, mapMDNodeDistinctOperands) {
  LLVMContext Context;
  Metadata *Old = MDTuple::getDistinct(Context, std::nullopt);
  auto *D = MDTuple::getDistinct(Context, Old);
  ASSERT_EQ(Old, D->getOperand(0));

  Metadata *New = MDTuple::getDistinct(Context, std::nullopt);
  ValueToValueMapTy VM;
  VM.MD()[Old].reset(New);

  // Make sure operands are updated.
  EXPECT_EQ(D, ValueMapper(VM, RF_ReuseAndMutateDistinctMDs).mapMDNode(*D));
  EXPECT_EQ(New, D->getOperand(0));
}

TEST(ValueMapperTest, mapMDNodeSeeded) {
  LLVMContext Context;
  auto *D = MDTuple::getDistinct(Context, std::nullopt);

  // The node should be moved.
  ValueToValueMapTy VM;
  EXPECT_EQ(std::nullopt, VM.getMappedMD(D));

  VM.MD().insert(std::make_pair(D, TrackingMDRef(D)));
  EXPECT_EQ(D, *VM.getMappedMD(D));
  EXPECT_EQ(D, ValueMapper(VM).mapMDNode(*D));
}

TEST(ValueMapperTest, mapMDNodeSeededWithNull) {
  LLVMContext Context;
  auto *D = MDTuple::getDistinct(Context, std::nullopt);

  // The node should be moved.
  ValueToValueMapTy VM;
  EXPECT_EQ(std::nullopt, VM.getMappedMD(D));

  VM.MD().insert(std::make_pair(D, TrackingMDRef()));
  EXPECT_EQ(nullptr, *VM.getMappedMD(D));
  EXPECT_EQ(nullptr, ValueMapper(VM).mapMDNode(*D));
}

TEST(ValueMapperTest, mapMetadataNullMapGlobalWithIgnoreMissingLocals) {
  LLVMContext C;
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(C), Type::getInt8Ty(C), false);
  std::unique_ptr<Function> F(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F"));

  ValueToValueMapTy VM;
  RemapFlags Flags = RF_IgnoreMissingLocals | RF_NullMapMissingGlobalValues;
  EXPECT_EQ(nullptr, ValueMapper(VM, Flags).mapValue(*F));
}

TEST(ValueMapperTest, mapMetadataMDString) {
  LLVMContext C;
  auto *S1 = MDString::get(C, "S1");
  ValueToValueMapTy VM;

  // Make sure S1 maps to itself, but isn't memoized.
  EXPECT_EQ(S1, ValueMapper(VM).mapMetadata(*S1));
  EXPECT_EQ(std::nullopt, VM.getMappedMD(S1));

  // We still expect VM.MD() to be respected.
  auto *S2 = MDString::get(C, "S2");
  VM.MD()[S1].reset(S2);
  EXPECT_EQ(S2, ValueMapper(VM).mapMetadata(*S1));
}

TEST(ValueMapperTest, mapMetadataGetMappedMD) {
  LLVMContext C;
  auto *N0 = MDTuple::get(C, std::nullopt);
  auto *N1 = MDTuple::get(C, N0);

  // Make sure hasMD and getMappedMD work correctly.
  ValueToValueMapTy VM;
  EXPECT_FALSE(VM.hasMD());
  EXPECT_EQ(N0, ValueMapper(VM).mapMetadata(*N0));
  EXPECT_EQ(N1, ValueMapper(VM).mapMetadata(*N1));
  EXPECT_TRUE(VM.hasMD());
  ASSERT_NE(std::nullopt, VM.getMappedMD(N0));
  ASSERT_NE(std::nullopt, VM.getMappedMD(N1));
  EXPECT_EQ(N0, *VM.getMappedMD(N0));
  EXPECT_EQ(N1, *VM.getMappedMD(N1));
}

TEST(ValueMapperTest, mapMetadataNoModuleLevelChanges) {
  LLVMContext C;
  auto *N0 = MDTuple::get(C, std::nullopt);
  auto *N1 = MDTuple::get(C, N0);

  // Nothing should be memoized when RF_NoModuleLevelChanges.
  ValueToValueMapTy VM;
  EXPECT_FALSE(VM.hasMD());
  EXPECT_EQ(N0, ValueMapper(VM, RF_NoModuleLevelChanges).mapMetadata(*N0));
  EXPECT_EQ(N1, ValueMapper(VM, RF_NoModuleLevelChanges).mapMetadata(*N1));
  EXPECT_FALSE(VM.hasMD());
  EXPECT_EQ(std::nullopt, VM.getMappedMD(N0));
  EXPECT_EQ(std::nullopt, VM.getMappedMD(N1));
}

TEST(ValueMapperTest, mapMetadataConstantAsMetadata) {
  LLVMContext C;
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(C), Type::getInt8Ty(C), false);
  std::unique_ptr<Function> F(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F"));

  auto *CAM = ConstantAsMetadata::get(F.get());
  {
    // ConstantAsMetadata shouldn't be memoized.
    ValueToValueMapTy VM;
    EXPECT_EQ(CAM, ValueMapper(VM).mapMetadata(*CAM));
    EXPECT_FALSE(VM.MD().count(CAM));
    EXPECT_EQ(CAM, ValueMapper(VM, RF_IgnoreMissingLocals).mapMetadata(*CAM));
    EXPECT_FALSE(VM.MD().count(CAM));

    // But it should respect a mapping that gets seeded.
    auto *N = MDTuple::get(C, std::nullopt);
    VM.MD()[CAM].reset(N);
    EXPECT_EQ(N, ValueMapper(VM).mapMetadata(*CAM));
    EXPECT_EQ(N, ValueMapper(VM, RF_IgnoreMissingLocals).mapMetadata(*CAM));
  }

  std::unique_ptr<Function> F2(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F2"));
  ValueToValueMapTy VM;
  VM[F.get()] = F2.get();
  auto *F2MD = ValueMapper(VM).mapMetadata(*CAM);
  EXPECT_FALSE(VM.MD().count(CAM));
  EXPECT_TRUE(F2MD);
  EXPECT_EQ(F2.get(), cast<ConstantAsMetadata>(F2MD)->getValue());
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(ValueMapperTest, mapMetadataLocalAsMetadata) {
  LLVMContext C;
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(C), Type::getInt8Ty(C), false);
  std::unique_ptr<Function> F(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F"));
  Argument &A = *F->arg_begin();

  // mapMetadata doesn't support LocalAsMetadata.  The only valid container for
  // LocalAsMetadata is a MetadataAsValue instance, so use it directly.
  auto *LAM = LocalAsMetadata::get(&A);
  ValueToValueMapTy VM;
  EXPECT_DEATH(ValueMapper(VM).mapMetadata(*LAM), "Unexpected local metadata");
  EXPECT_DEATH(ValueMapper(VM, RF_IgnoreMissingLocals).mapMetadata(*LAM),
               "Unexpected local metadata");
}
#endif
#endif

TEST(ValueMapperTest, mapValueLocalAsMetadata) {
  LLVMContext C;
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(C), Type::getInt8Ty(C), false);
  std::unique_ptr<Function> F(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F"));
  Argument &A = *F->arg_begin();

  auto *LAM = LocalAsMetadata::get(&A);
  auto *MAV = MetadataAsValue::get(C, LAM);

  // The principled answer to a LocalAsMetadata of an unmapped SSA value would
  // be to return nullptr (regardless of RF_IgnoreMissingLocals).
  //
  // However, algorithms that use RemapInstruction assume that each instruction
  // only references SSA values from previous instructions.  Arguments of
  // such as "metadata i32 %x" don't currently successfully maintain that
  // property.  To keep RemapInstruction from crashing we need a non-null
  // return here, but we also shouldn't reference the unmapped local.  Use
  // "metadata !{}".
  auto *N0 = MDTuple::get(C, std::nullopt);
  auto *N0AV = MetadataAsValue::get(C, N0);
  ValueToValueMapTy VM;
  EXPECT_EQ(N0AV, ValueMapper(VM).mapValue(*MAV));
  EXPECT_EQ(nullptr, ValueMapper(VM, RF_IgnoreMissingLocals).mapValue(*MAV));
  EXPECT_FALSE(VM.count(MAV));
  EXPECT_FALSE(VM.count(&A));
  EXPECT_EQ(std::nullopt, VM.getMappedMD(LAM));

  VM[MAV] = MAV;
  EXPECT_EQ(MAV, ValueMapper(VM).mapValue(*MAV));
  EXPECT_EQ(MAV, ValueMapper(VM, RF_IgnoreMissingLocals).mapValue(*MAV));
  EXPECT_TRUE(VM.count(MAV));
  EXPECT_FALSE(VM.count(&A));

  VM[MAV] = &A;
  EXPECT_EQ(&A, ValueMapper(VM).mapValue(*MAV));
  EXPECT_EQ(&A, ValueMapper(VM, RF_IgnoreMissingLocals).mapValue(*MAV));
  EXPECT_TRUE(VM.count(MAV));
  EXPECT_FALSE(VM.count(&A));
}

TEST(ValueMapperTest, mapValueLocalInArgList) {
  LLVMContext C;
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(C), Type::getInt8Ty(C), false);
  std::unique_ptr<Function> F(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F"));
  Argument &A = *F->arg_begin();

  auto *LAM = LocalAsMetadata::get(&A);
  std::vector<ValueAsMetadata*> Elts;
  Elts.push_back(LAM);
  auto *ArgList = DIArgList::get(C, Elts);
  auto *MAV = MetadataAsValue::get(C, ArgList);

  // The principled answer to a LocalAsMetadata of an unmapped SSA value would
  // be to return nullptr (regardless of RF_IgnoreMissingLocals).
  //
  // However, algorithms that use RemapInstruction assume that each instruction
  // only references SSA values from previous instructions.  Arguments of
  // such as "metadata i32 %x" don't currently successfully maintain that
  // property.  To keep RemapInstruction from crashing we need a non-null
  // return here, but we also shouldn't reference the unmapped local.  Use
  // undef for uses in a DIArgList.
  auto *N0 = UndefValue::get(Type::getInt8Ty(C));
  auto *N0AM = ValueAsMetadata::get(N0);
  std::vector<ValueAsMetadata*> N0Elts;
  N0Elts.push_back(N0AM);
  auto *N0ArgList = DIArgList::get(C, N0Elts);
  auto *N0AV = MetadataAsValue::get(C, N0ArgList);
  ValueToValueMapTy VM;
  EXPECT_EQ(N0AV, ValueMapper(VM).mapValue(*MAV));
  EXPECT_EQ(MAV, ValueMapper(VM, RF_IgnoreMissingLocals).mapValue(*MAV));
  EXPECT_FALSE(VM.count(MAV));
  EXPECT_FALSE(VM.count(&A));
  EXPECT_EQ(std::nullopt, VM.getMappedMD(LAM));
  EXPECT_EQ(std::nullopt, VM.getMappedMD(ArgList));

  VM[MAV] = MAV;
  EXPECT_EQ(MAV, ValueMapper(VM).mapValue(*MAV));
  EXPECT_EQ(MAV, ValueMapper(VM, RF_IgnoreMissingLocals).mapValue(*MAV));
  EXPECT_TRUE(VM.count(MAV));
  EXPECT_FALSE(VM.count(&A));

  VM[MAV] = &A;
  EXPECT_EQ(&A, ValueMapper(VM).mapValue(*MAV));
  EXPECT_EQ(&A, ValueMapper(VM, RF_IgnoreMissingLocals).mapValue(*MAV));
  EXPECT_TRUE(VM.count(MAV));
  EXPECT_FALSE(VM.count(&A));
}

TEST(ValueMapperTest, mapValueLocalAsMetadataToConstant) {
  LLVMContext Context;
  auto *Int8 = Type::getInt8Ty(Context);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context), Int8, false);
  std::unique_ptr<Function> F(
      Function::Create(FTy, GlobalValue::ExternalLinkage, "F"));

  // Map a local value to a constant.
  Argument &A = *F->arg_begin();
  Constant &C = *ConstantInt::get(Int8, 42);
  ValueToValueMapTy VM;
  VM[&A] = &C;

  // Look up the metadata-as-value wrapper.  Don't crash.
  auto *MDA = MetadataAsValue::get(Context, ValueAsMetadata::get(&A));
  auto *MDC = MetadataAsValue::get(Context, ValueAsMetadata::get(&C));
  EXPECT_TRUE(isa<LocalAsMetadata>(MDA->getMetadata()));
  EXPECT_TRUE(isa<ConstantAsMetadata>(MDC->getMetadata()));
  EXPECT_EQ(&C, ValueMapper(VM).mapValue(A));
  EXPECT_EQ(MDC, ValueMapper(VM).mapValue(*MDA));
}

// Type remapper which remaps all types to same destination.
class TestTypeRemapper : public ValueMapTypeRemapper {
public:
  TestTypeRemapper(Type *Ty) : DstTy(Ty) { }
  Type *remapType(Type *srcTy) { return DstTy; }
private:
  Type *DstTy;
};

TEST(ValueMapperTest, mapValuePoisonWithTypeRemap) {
  LLVMContext C;
  Type *OldTy = Type::getInt8Ty(C);
  Type *NewTy = Type::getInt32Ty(C);

  TestTypeRemapper TM(NewTy);
  ValueToValueMapTy VM;
  ValueMapper Mapper(VM, RF_None, &TM);

  // Check that poison is still poison and has not been converted to undef.
  auto *OldPoison = PoisonValue::get(OldTy);
  auto *NewPoison = PoisonValue::get(NewTy);
  EXPECT_EQ(NewPoison, Mapper.mapValue(*OldPoison));
}

TEST(ValueMapperTest, mapValueConstantTargetNoneToLayoutTypeNullValue) {
  LLVMContext C;
  auto *OldTy = TargetExtType::get(C, "spirv.Image");
  Type *NewTy = OldTy->getLayoutType();

  TestTypeRemapper TM(NewTy);
  ValueToValueMapTy VM;
  ValueMapper Mapper(VM, RF_None, &TM);

  // Check that ConstantTargetNone is mapped to '0' constant of its layout type.
  auto *OldConstant = ConstantTargetNone::get(OldTy);
  auto *NewConstant = Constant::getNullValue(NewTy);
  EXPECT_EQ(NewConstant, Mapper.mapValue(*OldConstant));
}

} // end namespace
