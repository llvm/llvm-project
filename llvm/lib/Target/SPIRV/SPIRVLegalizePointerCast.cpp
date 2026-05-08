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

#include "SPIRVLegalizePointerCast.h"
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
class SPIRVLegalizePointerCastImpl {

  // Builds the `spv_assign_type` assigning |Ty| to |Value| at the current
  // builder position.
  void buildAssignType(IRBuilder<> &B, Type *Ty, Value *Arg) {
    Value *OfType = PoisonValue::get(Ty);
    CallInst *AssignCI = buildIntrWithMD(Intrinsic::spv_assign_type,
                                         {Arg->getType()}, OfType, Arg, {}, B);
    GR->addAssignPtrTypeInstr(Arg, AssignCI);
  }

  static FixedVectorType *makeVectorFromTotalBits(Type *ElemTy,
                                                  TypeSize TotalBits) {
    unsigned ElemBits = ElemTy->getScalarSizeInBits();
    assert(ElemBits && TotalBits % ElemBits == 0 &&
           "TotalBits must be divisible by element bit size");
    return FixedVectorType::get(ElemTy, TotalBits / ElemBits);
  }

  Value *resizeVectorBitsWithShuffle(IRBuilder<> &B, Value *V,
                                     FixedVectorType *DstTy) {
    auto *SrcTy = cast<FixedVectorType>(V->getType());
    assert(SrcTy->getElementType() == DstTy->getElementType() &&
           "shuffle resize expects identical element types");

    const unsigned NumNeeded = DstTy->getNumElements();
    const unsigned NumSource = SrcTy->getNumElements();

    SmallVector<int> Mask(NumNeeded);
    for (unsigned I = 0; I < NumNeeded; ++I)
      Mask[I] = (I < NumSource) ? static_cast<int>(I) : -1;

    Value *Resized = B.CreateShuffleVector(V, V, Mask);
    buildAssignType(B, DstTy, Resized);
    return Resized;
  }

  // Loads parts of the vector of type |SourceType| from the pointer |Source|
  // and create a new vector of type |TargetType|. |TargetType| must be a vector
  // type.
  // Returns the loaded value.
  Value *loadVectorFromVector(IRBuilder<> &B, FixedVectorType *SourceType,
                              FixedVectorType *TargetType, Value *Source) {
    LoadInst *NewLoad = B.CreateLoad(SourceType, Source);
    buildAssignType(B, SourceType, NewLoad);
    Value *AssignValue = NewLoad;
    if (TargetType->getElementType() != SourceType->getElementType()) {
      const DataLayout &DL = B.GetInsertBlock()->getModule()->getDataLayout();
      TypeSize TargetTypeSize = DL.getTypeSizeInBits(TargetType);
      TypeSize SourceTypeSize = DL.getTypeSizeInBits(SourceType);

      Value *BitcastSrcVal = NewLoad;
      FixedVectorType *BitcastSrcTy =
          cast<FixedVectorType>(BitcastSrcVal->getType());
      FixedVectorType *BitcastDstTy = TargetType;

      if (TargetTypeSize != SourceTypeSize) {
        unsigned TargetElemBits =
            TargetType->getElementType()->getScalarSizeInBits();
        if (SourceTypeSize % TargetElemBits == 0) {
          // No Resize needed. Same total bits as source, but use target element
          // type.
          BitcastDstTy = makeVectorFromTotalBits(TargetType->getElementType(),
                                                 SourceTypeSize);
        } else {
          // Resize source to target total bitwidth using source element type.
          BitcastSrcTy = makeVectorFromTotalBits(SourceType->getElementType(),
                                                 TargetTypeSize);
          BitcastSrcVal = resizeVectorBitsWithShuffle(B, NewLoad, BitcastSrcTy);
        }
      }
      AssignValue =
          B.CreateIntrinsic(Intrinsic::spv_bitcast,
                            {BitcastDstTy, BitcastSrcTy}, {BitcastSrcVal});
      buildAssignType(B, BitcastDstTy, AssignValue);
      if (BitcastDstTy == TargetType)
        return AssignValue;
    }

    assert(TargetType->getNumElements() < SourceType->getNumElements());
    SmallVector<int> Mask(/* Size= */ TargetType->getNumElements());
    for (unsigned I = 0; I < TargetType->getNumElements(); ++I)
      Mask[I] = I;
    Value *Output = B.CreateShuffleVector(AssignValue, AssignValue, Mask);
    buildAssignType(B, TargetType, Output);
    return Output;
  }

  // Returns true if |FromTy| has a memory layout compatible with loading or
  // storing |ToTy|.
  bool isCompatibleMemoryLayout(Type *ToTy, Type *FromTy) {
    if (ToTy == FromTy)
      return true;
    auto *SVT = dyn_cast<FixedVectorType>(FromTy);
    auto *DVT = dyn_cast<FixedVectorType>(ToTy);
    if (SVT && DVT)
      return true;
    auto *SAT = dyn_cast<ArrayType>(FromTy);
    if (SAT && DVT) {
      if (SAT->getElementType() == DVT->getElementType())
        return true;
      if (auto *MAT = dyn_cast<FixedVectorType>(SAT->getElementType()))
        if (MAT->getElementType() == DVT->getElementType())
          return true;
    }
    return false;
  }

  // Traverses the aggregate type to find the first sub-type that matches
  // the TargetElemType's memory layout, optionally emitting a GEP intrinsic.
  std::optional<std::pair<Value *, Type *>>
  getPointerToFirstCompatibleType(IRBuilder<> &B, Value *BasePtr,
                                  Type *PointerType, Type *TargetElemType,
                                  bool IsInBounds) {
    Type *CurrentTy = GR->findDeducedElementType(BasePtr);
    assert(CurrentTy && "Could not deduce aggregate type");
    SmallVector<Value *, 8> Args{/* isInBounds= */ B.getInt1(IsInBounds),
                                 BasePtr};
    Args.push_back(B.getInt32(0)); // Pointer offset

    while (!isCompatibleMemoryLayout(TargetElemType, CurrentTy)) {
      if (auto *ST = dyn_cast<StructType>(CurrentTy)) {
        if (ST->getNumElements() == 0)
          return std::nullopt;
        CurrentTy = ST->getTypeAtIndex(0u);
      } else if (auto *AT = dyn_cast<ArrayType>(CurrentTy)) {
        CurrentTy = AT->getElementType();
      } else if (auto *VT = dyn_cast<FixedVectorType>(CurrentTy)) {
        CurrentTy = VT->getElementType();
      } else {
        return std::nullopt;
      }
      Args.push_back(B.getInt32(0));
    }

    Value *GEP = BasePtr;
    if (Args.size() > 3) {
      std::array<Type *, 2> Types = {PointerType, BasePtr->getType()};
      GEP = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, CurrentTy, GEP);
    }

    return std::make_pair(GEP, CurrentTy);
  }

  // Builds a legalized load from a pointer, drilling down through
  // memory layouts to find a compatible type. Load flags will be
  // copied from |BadLoad|, which should be the load being legalized.
  Value *buildLegalizedLoad(IRBuilder<> &B, Type *ElementType, Value *Source,
                            LoadInst *BadLoad) {
    auto ResultOpt = getPointerToFirstCompatibleType(
        B, Source, BadLoad->getPointerOperandType(), ElementType, false);
    assert(ResultOpt && "Failed to load from aggregate: "
                        "Could not find compatible memory layout.");
    auto [GEP, CurrentTy] = *ResultOpt;

    auto *SAT = dyn_cast<ArrayType>(CurrentTy);
    auto *SVT = dyn_cast<FixedVectorType>(CurrentTy);
    auto *DVT = dyn_cast<FixedVectorType>(ElementType);
    auto *MAT =
        SAT ? dyn_cast<FixedVectorType>(SAT->getElementType()) : nullptr;

    if (ElementType == CurrentTy) {
      LoadInst *LI = B.CreateLoad(ElementType, GEP);
      LI->setAlignment(BadLoad->getAlign());
      buildAssignType(B, ElementType, LI);
      return LI;
    }
    if (SVT && DVT)
      return loadVectorFromVector(B, SVT, DVT, GEP);
    if (SAT && DVT && SAT->getElementType() == DVT->getElementType())
      return loadVectorFromArray(B, DVT, GEP);
    if (MAT && DVT && MAT->getElementType() == DVT->getElementType())
      return loadVectorFromMatrixArray(B, DVT, GEP, MAT);

    llvm_unreachable("Failed to load from aggregate.");
  }
  Value *
  buildVectorFromLoadedElements(IRBuilder<> &B, FixedVectorType *TargetType,
                                SmallVector<Value *, 4> &LoadedElements) {
    // Build the vector from the loaded elements.
    Value *NewVector = PoisonValue::get(TargetType);
    buildAssignType(B, TargetType, NewVector);

    for (unsigned I = 0, E = TargetType->getNumElements(); I < E; ++I) {
      Value *Index = B.getInt32(I);
      SmallVector<Type *, 4> Types = {TargetType, TargetType,
                                      TargetType->getElementType(),
                                      Index->getType()};
      SmallVector<Value *> Args = {NewVector, LoadedElements[I], Index};
      NewVector = B.CreateIntrinsic(Intrinsic::spv_insertelt, {Types}, {Args});
      buildAssignType(B, TargetType, NewVector);
    }
    return NewVector;
  }

  // Loads elements from a matrix with an array of vector memory layout and
  // constructs a vector.
  Value *loadVectorFromMatrixArray(IRBuilder<> &B, FixedVectorType *TargetType,
                                   Value *Source,
                                   FixedVectorType *ArrElemVecTy) {
    Type *TargetElemTy = TargetType->getElementType();
    unsigned ScalarsPerArrayElement = ArrElemVecTy->getNumElements();
    // Load each element of the array.
    SmallVector<Value *, 4> LoadedElements;
    std::array<Type *, 2> Types = {Source->getType(), Source->getType()};
    for (unsigned I = 0, E = TargetType->getNumElements(); I < E; ++I) {
      unsigned ArrayIndex = I / ScalarsPerArrayElement;
      unsigned ElementIndexInArrayElem = I % ScalarsPerArrayElement;
      // Create a GEP to access the i-th element of the array.
      std::array<Value *, 4> Args = {
          B.getInt1(/*Inbounds=*/false), Source, B.getInt32(0),
          ConstantInt::get(B.getInt32Ty(), ArrayIndex)};
      auto *ElementPtr = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, ArrElemVecTy, ElementPtr);
      Value *LoadVec = B.CreateLoad(ArrElemVecTy, ElementPtr);
      buildAssignType(B, ArrElemVecTy, LoadVec);
      LoadedElements.push_back(makeExtractElement(B, TargetElemTy, LoadVec,
                                                  ElementIndexInArrayElem));
    }
    return buildVectorFromLoadedElements(B, TargetType, LoadedElements);
  }
  // Loads elements from an array and constructs a vector.
  Value *loadVectorFromArray(IRBuilder<> &B, FixedVectorType *TargetType,
                             Value *Source) {
    // Load each element of the array.
    SmallVector<Value *, 4> LoadedElements;
    std::array<Type *, 2> Types = {Source->getType(), Source->getType()};
    for (unsigned I = 0, E = TargetType->getNumElements(); I < E; ++I) {
      // Create a GEP to access the i-th element of the array.
      std::array<Value *, 4> Args = {B.getInt1(/*Inbounds=*/false), Source,
                                     B.getInt32(0),
                                     ConstantInt::get(B.getInt32Ty(), I)};
      auto *ElementPtr = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, TargetType->getElementType(), ElementPtr);

      // Load the value from the element pointer.
      Value *Load = B.CreateLoad(TargetType->getElementType(), ElementPtr);
      buildAssignType(B, TargetType->getElementType(), Load);
      LoadedElements.push_back(Load);
    }
    return buildVectorFromLoadedElements(B, TargetType, LoadedElements);
  }

  // Stores elements from a vector into a matrix (an array of vectors).
  void storeMatrixArrayFromVector(IRBuilder<> &B, Value *SrcVector,
                                  Value *DstArrayPtr, ArrayType *ArrTy,
                                  Align Alignment) {
    auto *SrcVecTy = cast<FixedVectorType>(SrcVector->getType());
    auto *ArrElemVecTy = cast<FixedVectorType>(ArrTy->getElementType());
    Type *ElemTy = ArrElemVecTy->getElementType();
    unsigned ScalarsPerArrayElement = ArrElemVecTy->getNumElements();
    unsigned SrcNumElements = SrcVecTy->getNumElements();
    assert(
        SrcNumElements % ScalarsPerArrayElement == 0 &&
        "Source vector size must be a multiple of array element vector size");

    std::array<Type *, 2> Types = {DstArrayPtr->getType(),
                                   DstArrayPtr->getType()};

    for (unsigned I = 0; I < SrcNumElements; I += ScalarsPerArrayElement) {
      unsigned ArrayIndex = I / ScalarsPerArrayElement;
      // Create a GEP to access the array element.
      std::array<Value *, 4> Args = {
          B.getInt1(/*Inbounds=*/false), DstArrayPtr, B.getInt32(0),
          ConstantInt::get(B.getInt32Ty(), ArrayIndex)};
      auto *ElementPtr = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, ArrElemVecTy, ElementPtr);

      // Extract scalar elements from the source vector for this array slot.
      SmallVector<Value *, 4> Elements;
      for (unsigned J = 0; J < ScalarsPerArrayElement; ++J)
        Elements.push_back(makeExtractElement(B, ElemTy, SrcVector, I + J));

      // Build a vector from the extracted elements and store it.
      Value *Vec = buildVectorFromLoadedElements(B, ArrElemVecTy, Elements);
      StoreInst *SI = B.CreateStore(Vec, ElementPtr);
      SI->setAlignment(Alignment);
    }
  }

  // Stores elements from a vector into an array.
  void storeArrayFromVector(IRBuilder<> &B, Value *SrcVector,
                            Value *DstArrayPtr, ArrayType *ArrTy,
                            Align Alignment) {
    auto *VecTy = cast<FixedVectorType>(SrcVector->getType());
    Type *ElemTy = ArrTy->getElementType();

    // Ensure the element types of the array and vector are the same.
    assert(VecTy->getElementType() == ElemTy &&
           "Element types of array and vector must be the same.");
    std::array<Type *, 2> Types = {DstArrayPtr->getType(),
                                   DstArrayPtr->getType()};

    for (unsigned I = 0, E = VecTy->getNumElements(); I < E; ++I) {
      // Create a GEP to access the i-th element of the array.
      std::array<Value *, 4> Args = {B.getInt1(/*Inbounds=*/false), DstArrayPtr,
                                     B.getInt32(0),
                                     ConstantInt::get(B.getInt32Ty(), I)};
      auto *ElementPtr = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
      GR->buildAssignPtr(B, ElemTy, ElementPtr);

      // Extract the element from the vector and store it.
      Value *Element = makeExtractElement(B, ElemTy, SrcVector, I);
      StoreInst *SI = B.CreateStore(Element, ElementPtr);
      SI->setAlignment(Alignment);
    }
  }

  // Replaces the load instruction to get rid of the ptrcast used as source
  // operand.
  void transformLoad(IRBuilder<> &B, LoadInst *LI, Value *CastedOperand,
                     Value *OriginalOperand) {
    Type *ToTy = GR->findDeducedElementType(CastedOperand);
    B.SetInsertPoint(LI);

    Value *Output = buildLegalizedLoad(B, ToTy, OriginalOperand, LI);

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

  // Builds a legalized store to a pointer, drilling down through
  // memory layouts to find a compatible type.
  void buildLegalizedStore(IRBuilder<> &B, Value *Src, Value *Dst,
                           Align Alignment) {
    auto ResultOpt = getPointerToFirstCompatibleType(B, Dst, Dst->getType(),
                                                     Src->getType(), true);
    assert(ResultOpt && "Failed to store to aggregate: "
                        "Could not find compatible memory layout.");
    auto [GEP, CurrentTy] = *ResultOpt;

    auto *DAT = dyn_cast<ArrayType>(CurrentTy);
    auto *DVT = dyn_cast<FixedVectorType>(CurrentTy);
    auto *SVT = dyn_cast<FixedVectorType>(Src->getType());
    auto *DMAT =
        DAT ? dyn_cast<FixedVectorType>(DAT->getElementType()) : nullptr;

    if (Src->getType() == CurrentTy) {
      StoreInst *SI = B.CreateStore(Src, GEP);
      SI->setAlignment(Alignment);
      return;
    }
    if (DVT && SVT) {
      storeVectorFromVector(B, Src, GEP, Alignment);
      return;
    }
    if (DAT && SVT && SVT->getElementType() == DAT->getElementType()) {
      storeArrayFromVector(B, Src, GEP, DAT, Alignment);
      return;
    }
    if (DMAT && SVT && DMAT->getElementType() == SVT->getElementType()) {
      storeMatrixArrayFromVector(B, Src, GEP, DAT, Alignment);
      return;
    }

    llvm_unreachable("Failed to store to aggregate.");
  }

  // Transforms a store instruction (or SPV intrinsic) using a ptrcast as
  // operand into a valid logical SPIR-V store with no ptrcast.
  void transformStore(IRBuilder<> &B, Instruction *BadStore, Value *Src,
                      Value *Dst, Align Alignment) {
    B.SetInsertPoint(BadStore);
    buildLegalizedStore(B, Src, Dst, Alignment);
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
  SPIRVLegalizePointerCastImpl(const SPIRVTargetMachine &TM) : TM(TM) {}

  bool run(Function &F) {
    const SPIRVSubtarget &ST = TM.getSubtarget<SPIRVSubtarget>(F);
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
  const SPIRVTargetMachine &TM;
  SPIRVGlobalRegistry *GR = nullptr;
  std::vector<Instruction *> DeadInstructions;
};

class SPIRVLegalizePointerCastLegacy : public FunctionPass {
public:
  static char ID;
  SPIRVLegalizePointerCastLegacy(const SPIRVTargetMachine &TM)
      : FunctionPass(ID), TM(TM) {}

  bool runOnFunction(Function &F) override {
    return SPIRVLegalizePointerCastImpl(TM).run(F);
  }

private:
  const SPIRVTargetMachine &TM;
};
} // namespace

PreservedAnalyses SPIRVLegalizePointerCast::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  return SPIRVLegalizePointerCastImpl(TM).run(F) ? PreservedAnalyses::none()
                                                 : PreservedAnalyses::all();
}

char SPIRVLegalizePointerCastLegacy::ID = 0;
INITIALIZE_PASS(SPIRVLegalizePointerCastLegacy, "spirv-legalize-pointer-cast",
                "SPIRV legalize pointer cast pass", false, false)

FunctionPass *llvm::createSPIRVLegalizePointerCastPass(SPIRVTargetMachine *TM) {
  return new SPIRVLegalizePointerCastLegacy(*TM);
}
