//===-- SPIRVLegalizePointerCast.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The LLVM IR has multiple legal patterns we cannot lower to Logical SPIR-V.
// This pass modifies such loads to have an IR we can directly lower to valid
// logical SPIR-V.
// OpenCL can avoid this because they rely on ptrcast, which is not supported
// by logical SPIR-V.
//
// This pass relies on the assign_ptr_type intrinsic to deduce the type of the
// pointed values, must replace all occurences of `ptrcast`. This is why
// unhandled cases are reported as unreachable: we MUST cover all cases.
//
// 1. Loading the first element of an array
//
//        %array = [10 x i32]
//        %value = load i32, ptr %array
//
//    LLVM can skip the GEP instruction, and only request loading the first 4
//    bytes. In logical SPIR-V, we need an OpAccessChain to access the first
//    element. This pass will add a getelementptr instruction before the load.
//
//
// 2. Implicit downcast from load
//
//        %1 = getelementptr <4 x i32>, ptr %vec4, i64 0
//        %2 = load <3 x i32>, ptr %1
//
//    The pointer in the GEP instruction is only used for offset computations,
//    but it doesn't NEED to match the pointed type. OpAccessChain however
//    requires this. Also, LLVM loads define the bitwidth of the load, not the
//    pointer. In this example, we can guess %vec4 is a vec4 thanks to the GEP
//    instruction basetype, but we only want to load the first 3 elements, hence
//    do a partial load. In logical SPIR-V, this is not legal. What we must do
//    is load the full vector (basetype), extract 3 elements, and recombine them
//    to form a 3-element vector.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;

namespace {
class SPIRVLegalizePointerCast : public FunctionPass {

  // Builds the `spv_assign_type` assigning |Ty| to |Value| at the current
  // builder position.
  void buildAssignType(IRBuilder<> &B, Type *Ty, Value *Arg) {
    Value *OfType = PoisonValue::get(Ty);
    CallInst *AssignCI = buildIntrWithMD(Intrinsic::spv_assign_type,
                                         {Arg->getType()}, OfType, Arg, {}, B);
    GR->addAssignPtrTypeInstr(Arg, AssignCI);
  }

  // Loads parts of the vector of type |SourceType| from the pointer |Source|
  // and create a new vector of type |TargetType|. |TargetType| must be a vector
  // type, and element types of |TargetType| and |SourceType| must match.
  // Returns the loaded value.
  Value *loadVectorFromVector(IRBuilder<> &B, FixedVectorType *SourceType,
                              FixedVectorType *TargetType, Value *Source) {
    assert(TargetType->getNumElements() <= SourceType->getNumElements());
    LoadInst *NewLoad = B.CreateLoad(SourceType, Source);
    buildAssignType(B, SourceType, NewLoad);
    Value *AssignValue = NewLoad;
    if (TargetType->getElementType() != SourceType->getElementType()) {
      AssignValue = B.CreateIntrinsic(Intrinsic::spv_bitcast,
                                      {TargetType, SourceType}, {NewLoad});
      buildAssignType(B, TargetType, AssignValue);
    }

    SmallVector<int> Mask(/* Size= */ TargetType->getNumElements());
    for (unsigned I = 0; I < TargetType->getNumElements(); ++I)
      Mask[I] = I;
    Value *Output = B.CreateShuffleVector(AssignValue, AssignValue, Mask);
    buildAssignType(B, TargetType, Output);
    return Output;
  }

  // Loads the first value in an aggregate pointed by |Source| of containing
  // elements of type |ElementType|. Load flags will be copied from |BadLoad|,
  // which should be the load being legalized. Returns the loaded value.
  Value *loadFirstValueFromAggregate(IRBuilder<> &B, Type *ElementType,
                                     Value *Source, LoadInst *BadLoad) {
    SmallVector<Type *, 2> Types = {BadLoad->getPointerOperandType(),
                                    BadLoad->getPointerOperandType()};
    SmallVector<Value *, 3> Args{/* isInBounds= */ B.getInt1(false), Source,
                                 B.getInt32(0), B.getInt32(0)};
    auto *GEP = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
    GR->buildAssignPtr(B, ElementType, GEP);

    LoadInst *LI = B.CreateLoad(ElementType, GEP);
    LI->setAlignment(BadLoad->getAlign());
    buildAssignType(B, ElementType, LI);
    return LI;
  }

  // Replaces the load instruction to get rid of the ptrcast used as source
  // operand.
  void transformLoad(IRBuilder<> &B, LoadInst *LI, Value *CastedOperand,
                     Value *OriginalOperand) {
    Type *FromTy = GR->findDeducedElementType(OriginalOperand);
    Type *ToTy = GR->findDeducedElementType(CastedOperand);
    Value *Output = nullptr;

    auto *SAT = dyn_cast<ArrayType>(FromTy);
    auto *SVT = dyn_cast<FixedVectorType>(FromTy);
    auto *SST = dyn_cast<StructType>(FromTy);
    auto *DVT = dyn_cast<FixedVectorType>(ToTy);

    B.SetInsertPoint(LI);

    // Destination is the element type of Source, and source is an array ->
    // Loading 1st element.
    // - float a = array[0];
    if (SAT && SAT->getElementType() == ToTy)
      Output = loadFirstValueFromAggregate(B, SAT->getElementType(),
                                           OriginalOperand, LI);
    // Destination is the element type of Source, and source is a vector ->
    // Vector to scalar.
    // - float a = vector.x;
    else if (!DVT && SVT && SVT->getElementType() == ToTy) {
      Output = loadFirstValueFromAggregate(B, SVT->getElementType(),
                                           OriginalOperand, LI);
    }
    // Destination is a smaller vector than source or different vector type.
    // - float3 v3 = vector4;
    // - float4 v2 = int4;
    else if (SVT && DVT)
      Output = loadVectorFromVector(B, SVT, DVT, OriginalOperand);
    // Destination is the scalar type stored at the start of an aggregate.
    // - struct S { float m };
    // - float v = s.m;
    else if (SST && SST->getTypeAtIndex(0u) == ToTy)
      Output = loadFirstValueFromAggregate(B, ToTy, OriginalOperand, LI);
    else
      llvm_unreachable("Unimplemented implicit down-cast from load.");

    GR->replaceAllUsesWith(LI, Output, /* DeleteOld= */ true);
    DeadInstructions.push_back(LI);
  }

  // Creates an spv_insertelt instruction (equivalent to llvm's insertelement).
  Value *makeInsertElement(IRBuilder<> &B, Value *Vector, Value *Element,
                           unsigned Index) {
    Type *Int32Ty = Type::getInt32Ty(B.getContext());
    SmallVector<Type *, 4> Types = {Vector->getType(), Vector->getType(),
                                    Element->getType(), Int32Ty};
    SmallVector<Value *> Args = {Vector, Element, B.getInt32(Index)};
    Instruction *NewI =
        B.CreateIntrinsic(Intrinsic::spv_insertelt, {Types}, {Args});
    buildAssignType(B, Vector->getType(), NewI);
    return NewI;
  }

  // Creates an spv_extractelt instruction (equivalent to llvm's
  // extractelement).
  Value *makeExtractElement(IRBuilder<> &B, Type *ElementType, Value *Vector,
                            unsigned Index) {
    Type *Int32Ty = Type::getInt32Ty(B.getContext());
    SmallVector<Type *, 3> Types = {ElementType, Vector->getType(), Int32Ty};
    SmallVector<Value *> Args = {Vector, B.getInt32(Index)};
    Instruction *NewI =
        B.CreateIntrinsic(Intrinsic::spv_extractelt, {Types}, {Args});
    buildAssignType(B, ElementType, NewI);
    return NewI;
  }

  // Stores the given Src vector operand into the Dst vector, adjusting the size
  // if required.
  Value *storeVectorFromVector(IRBuilder<> &B, Value *Src, Value *Dst,
                               Align Alignment) {
    FixedVectorType *SrcType = cast<FixedVectorType>(Src->getType());
    FixedVectorType *DstType =
        cast<FixedVectorType>(GR->findDeducedElementType(Dst));
    auto dstNumElements = DstType->getNumElements();
    auto srcNumElements = SrcType->getNumElements();

    // if the element type differs, it is a bitcast.
    if (DstType->getElementType() != SrcType->getElementType()) {
      // Support bitcast between vectors of different sizes only if
      // the total bitwidth is the same.
      [[maybe_unused]] auto dstBitWidth =
          DstType->getElementType()->getScalarSizeInBits() * dstNumElements;
      [[maybe_unused]] auto srcBitWidth =
          SrcType->getElementType()->getScalarSizeInBits() * srcNumElements;
      assert(dstBitWidth == srcBitWidth &&
             "Unsupported bitcast between vectors of different sizes.");

      Src =
          B.CreateIntrinsic(Intrinsic::spv_bitcast, {DstType, SrcType}, {Src});
      buildAssignType(B, DstType, Src);
      SrcType = DstType;

      StoreInst *SI = B.CreateStore(Src, Dst);
      SI->setAlignment(Alignment);
      return SI;
    }

    assert(DstType->getNumElements() >= SrcType->getNumElements());
    LoadInst *LI = B.CreateLoad(DstType, Dst);
    LI->setAlignment(Alignment);
    Value *OldValues = LI;
    buildAssignType(B, OldValues->getType(), OldValues);
    Value *NewValues = Src;

    for (unsigned I = 0; I < SrcType->getNumElements(); ++I) {
      Value *Element =
          makeExtractElement(B, SrcType->getElementType(), NewValues, I);
      OldValues = makeInsertElement(B, OldValues, Element, I);
    }

    StoreInst *SI = B.CreateStore(OldValues, Dst);
    SI->setAlignment(Alignment);
    return SI;
  }

  void buildGEPIndexChain(IRBuilder<> &B, Type *Search, Type *Aggregate,
                          SmallVectorImpl<Value *> &Indices) {
    Indices.push_back(B.getInt32(0));

    if (Search == Aggregate)
      return;

    if (auto *ST = dyn_cast<StructType>(Aggregate))
      buildGEPIndexChain(B, Search, ST->getTypeAtIndex(0u), Indices);
    else if (auto *AT = dyn_cast<ArrayType>(Aggregate))
      buildGEPIndexChain(B, Search, AT->getElementType(), Indices);
    else if (auto *VT = dyn_cast<FixedVectorType>(Aggregate))
      buildGEPIndexChain(B, Search, VT->getElementType(), Indices);
    else
      llvm_unreachable("Bad access chain?");
  }

  // Stores the given Src value into the first entry of the Dst aggregate.
  Value *storeToFirstValueAggregate(IRBuilder<> &B, Value *Src, Value *Dst,
                                    Type *DstPointeeType, Align Alignment) {
    SmallVector<Type *, 2> Types = {Dst->getType(), Dst->getType()};
    SmallVector<Value *, 3> Args{/* isInBounds= */ B.getInt1(true), Dst};
    buildGEPIndexChain(B, Src->getType(), DstPointeeType, Args);
    auto *GEP = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
    GR->buildAssignPtr(B, Src->getType(), GEP);
    StoreInst *SI = B.CreateStore(Src, GEP);
    SI->setAlignment(Alignment);
    return SI;
  }

  bool isTypeFirstElementAggregate(Type *Search, Type *Aggregate) {
    if (Search == Aggregate)
      return true;
    if (auto *ST = dyn_cast<StructType>(Aggregate))
      return isTypeFirstElementAggregate(Search, ST->getTypeAtIndex(0u));
    if (auto *VT = dyn_cast<FixedVectorType>(Aggregate))
      return isTypeFirstElementAggregate(Search, VT->getElementType());
    if (auto *AT = dyn_cast<ArrayType>(Aggregate))
      return isTypeFirstElementAggregate(Search, AT->getElementType());
    return false;
  }

  // Transforms a store instruction (or SPV intrinsic) using a ptrcast as
  // operand into a valid logical SPIR-V store with no ptrcast.
  void transformStore(IRBuilder<> &B, Instruction *BadStore, Value *Src,
                      Value *Dst, Align Alignment) {
    Type *ToTy = GR->findDeducedElementType(Dst);
    Type *FromTy = Src->getType();

    auto *S_VT = dyn_cast<FixedVectorType>(FromTy);
    auto *D_ST = dyn_cast<StructType>(ToTy);
    auto *D_VT = dyn_cast<FixedVectorType>(ToTy);

    B.SetInsertPoint(BadStore);
    if (D_ST && isTypeFirstElementAggregate(FromTy, D_ST))
      storeToFirstValueAggregate(B, Src, Dst, D_ST, Alignment);
    else if (D_VT && S_VT)
      storeVectorFromVector(B, Src, Dst, Alignment);
    else if (D_VT && !S_VT && FromTy == D_VT->getElementType())
      storeToFirstValueAggregate(B, Src, Dst, D_VT, Alignment);
    else
      llvm_unreachable("Unsupported ptrcast use in store. Please fix.");

    DeadInstructions.push_back(BadStore);
  }

  void legalizePointerCast(IntrinsicInst *II) {
    Value *CastedOperand = II;
    Value *OriginalOperand = II->getOperand(0);

    IRBuilder<> B(II->getContext());
    std::vector<Value *> Users;
    for (Use &U : II->uses())
      Users.push_back(U.getUser());

    for (Value *User : Users) {
      if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
        transformLoad(B, LI, CastedOperand, OriginalOperand);
        continue;
      }

      if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
        transformStore(B, SI, SI->getValueOperand(), OriginalOperand,
                       SI->getAlign());
        continue;
      }

      if (IntrinsicInst *Intrin = dyn_cast<IntrinsicInst>(User)) {
        if (Intrin->getIntrinsicID() == Intrinsic::spv_assign_ptr_type) {
          DeadInstructions.push_back(Intrin);
          continue;
        }

        if (Intrin->getIntrinsicID() == Intrinsic::spv_gep) {
          GR->replaceAllUsesWith(CastedOperand, OriginalOperand,
                                 /* DeleteOld= */ false);
          continue;
        }

        if (Intrin->getIntrinsicID() == Intrinsic::spv_store) {
          Align Alignment;
          if (ConstantInt *C = dyn_cast<ConstantInt>(Intrin->getOperand(3)))
            Alignment = Align(C->getZExtValue());
          transformStore(B, Intrin, Intrin->getArgOperand(0), OriginalOperand,
                         Alignment);
          continue;
        }
      }

      llvm_unreachable("Unsupported ptrcast user. Please fix.");
    }

    DeadInstructions.push_back(II);
  }

public:
  SPIRVLegalizePointerCast(SPIRVTargetMachine *TM) : FunctionPass(ID), TM(TM) {}

  virtual bool runOnFunction(Function &F) override {
    const SPIRVSubtarget &ST = TM->getSubtarget<SPIRVSubtarget>(F);
    GR = ST.getSPIRVGlobalRegistry();
    DeadInstructions.clear();

    std::vector<IntrinsicInst *> WorkList;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *II = dyn_cast<IntrinsicInst>(&I);
        if (II && II->getIntrinsicID() == Intrinsic::spv_ptrcast)
          WorkList.push_back(II);
      }
    }

    for (IntrinsicInst *II : WorkList)
      legalizePointerCast(II);

    for (Instruction *I : DeadInstructions)
      I->eraseFromParent();

    return DeadInstructions.size() != 0;
  }

private:
  SPIRVTargetMachine *TM = nullptr;
  SPIRVGlobalRegistry *GR = nullptr;
  std::vector<Instruction *> DeadInstructions;

public:
  static char ID;
};
} // namespace

char SPIRVLegalizePointerCast::ID = 0;
INITIALIZE_PASS(SPIRVLegalizePointerCast, "spirv-legalize-bitcast",
                "SPIRV legalize bitcast pass", false, false)

FunctionPass *llvm::createSPIRVLegalizePointerCastPass(SPIRVTargetMachine *TM) {
  return new SPIRVLegalizePointerCast(TM);
}
