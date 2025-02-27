//===-- SPIRVLegalizePointerLoad.cpp ----------------------*- C++ -*-===//
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
// pointee. All occurrences of `ptrcast` must be replaced because the lead to
// invalid SPIR-V. Unhandled cases result in an error.
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
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;

namespace llvm {
void initializeSPIRVLegalizePointerLoadPass(PassRegistry &);
}

class SPIRVLegalizePointerLoad : public FunctionPass {

  // Replace all uses of a |Old| with |New| updates the global registry type
  // mappings.
  void replaceAllUsesWith(Value *Old, Value *New) {
    Old->replaceAllUsesWith(New);
    GR->updateIfExistDeducedElementType(Old, New, /* deleteOld = */ true);
    GR->updateIfExistAssignPtrTypeInstr(Old, New, /* deleteOld = */ true);
  }

  // Builds the `spv_assign_type` assigning |Ty| to |Value| at the current
  // builder position.
  void buildAssignType(IRBuilder<> &B, Type *Ty, Value *Arg) {
    Value *OfType = PoisonValue::get(Ty);
    CallInst *AssignCI = buildIntrWithMD(Intrinsic::spv_assign_type,
                                         {Arg->getType()}, OfType, Arg, {}, B);
    GR->addAssignPtrTypeInstr(Arg, AssignCI);
  }

  // Loads a single scalar of type |To| from the vector pointed by |Source| of
  // the type |From|. Returns the loaded value.
  Value *loadScalarFromVector(IRBuilder<> &B, Value *Source,
                              FixedVectorType *From) {

    LoadInst *NewLoad = B.CreateLoad(From, Source);
    buildAssignType(B, From, NewLoad);

    SmallVector<Value *, 2> Args = {NewLoad, B.getInt64(0)};
    SmallVector<Type *, 3> Types = {From->getElementType(), Args[0]->getType(),
                                    Args[1]->getType()};
    Value *Extracted =
        B.CreateIntrinsic(Intrinsic::spv_extractelt, {Types}, {Args});
    buildAssignType(B, Extracted->getType(), Extracted);
    return Extracted;
  }

  // Loads parts of the vector of type |From| from the pointer |Source| and
  // create a new vector of type |To|. |To| must be a vector type, and element
  // types of |To| and |From| must match. Returns the loaded value.
  Value *loadVectorFromVector(IRBuilder<> &B, FixedVectorType *From,
                              FixedVectorType *To, Value *Source) {
    // We expect the codegen to avoid doing implicit bitcast from a load.
    assert(To->getElementType() == From->getElementType());
    assert(To->getNumElements() < From->getNumElements());

    LoadInst *NewLoad = B.CreateLoad(From, Source);
    buildAssignType(B, From, NewLoad);

    auto ConstInt = ConstantInt::get(IntegerType::get(B.getContext(), 32), 0);
    ElementCount VecElemCount = ElementCount::getFixed(To->getNumElements());
    Value *Output = ConstantVector::getSplat(VecElemCount, ConstInt);
    for (unsigned I = 0; I < To->getNumElements(); ++I) {
      Value *Extracted = nullptr;
      {
        SmallVector<Value *, 2> Args = {NewLoad, B.getInt64(I)};
        SmallVector<Type *, 3> Types = {To->getElementType(),
                                        Args[0]->getType(), Args[1]->getType()};
        Extracted =
            B.CreateIntrinsic(Intrinsic::spv_extractelt, {Types}, {Args});
        buildAssignType(B, Extracted->getType(), Extracted);
      }
      assert(Extracted != nullptr);

      {
        SmallVector<Value *, 3> Args = {Output, Extracted, B.getInt64(I)};
        SmallVector<Type *, 4> Types = {Args[0]->getType(), Args[0]->getType(),
                                        Args[1]->getType(), Args[2]->getType()};
        Output = B.CreateIntrinsic(Intrinsic::spv_insertelt, {Types}, {Args});
        buildAssignType(B, Output->getType(), Output);
      }
    }
    return Output;
  }

  // Loads the first value in an array pointed by |Source| of type |From|. Load
  // flags will be copied from |BadLoad|, which should be the illegal load being
  // legalized. Returns the loaded value.
  Value *loadFirstValueFromArray(IRBuilder<> &B, ArrayType *From, Value *Source,
                                 LoadInst *BadLoad) {
    SmallVector<Type *, 2> Types = {BadLoad->getPointerOperandType(),
                                    BadLoad->getPointerOperandType()};
    SmallVector<Value *, 3> Args{/* isInBounds= */ B.getInt1(true), Source,
                                 B.getInt64(0), B.getInt64(0)};
    auto *GEP = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
    GR->buildAssignPtr(B, From->getElementType(), GEP);

    const auto *TLI = TM->getSubtargetImpl()->getTargetLowering();
    MachineMemOperand::Flags Flags = TLI->getLoadMemOperandFlags(
        *BadLoad, BadLoad->getFunction()->getDataLayout());
    Instruction *LI = B.CreateIntrinsic(
        Intrinsic::spv_load, {BadLoad->getOperand(0)->getType()},
        {GEP, B.getInt16(Flags), B.getInt8(BadLoad->getAlign().value())});
    buildAssignType(B, From->getElementType(), LI);
    return LI;
  }

  // Transforms an illegal partial load into a sequence we can lower to logical
  // SPIR-V.
  void transformLoad(IRBuilder<> &B, LoadInst *LI, Value *CastedOperand,
                     Value *OriginalOperand) {
    Type *FromTy = GR->findDeducedElementType(OriginalOperand);
    Type *ToTy /*-fruity*/ = GR->findDeducedElementType(CastedOperand);
    Value *Output = nullptr;

    auto *SAT = dyn_cast<ArrayType>(FromTy);
    auto *SVT = dyn_cast<FixedVectorType>(FromTy);
    auto *DVT = dyn_cast<FixedVectorType>(ToTy);

    B.SetInsertPoint(LI);

    // Destination is the element type of Source, and source is an array ->
    // Loading 1st element.
    // - float a = array[0];
    if (SAT && SAT->getElementType() == ToTy)
      Output = loadFirstValueFromArray(B, SAT, OriginalOperand, LI);
    // Destination is the element type of Source, and source is a vector ->
    // Vector to scalar.
    // - float a = vector.x;
    else if (!DVT && SVT && SVT->getElementType() == ToTy) {
      Output = loadScalarFromVector(B, OriginalOperand, SVT);
    }
    // Destination is a smaller vector than source.
    // - float3 v3 = vector4;
    else if (SVT && DVT)
      Output = loadVectorFromVector(B, SVT, DVT, OriginalOperand);
    else
      llvm_unreachable("Unimplemented implicit down-cast from load.");

    replaceAllUsesWith(LI, Output);
    DeadInstructions.push_back(LI);
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

      IntrinsicInst *Intrin = dyn_cast<IntrinsicInst>(User);
      if (Intrin->getIntrinsicID() == Intrinsic::spv_assign_ptr_type) {
        DeadInstructions.push_back(Intrin);
        continue;
      }

      llvm_unreachable("Unsupported ptrcast user. Please fix.");
    }

    DeadInstructions.push_back(II);
  }

public:
  SPIRVLegalizePointerLoad(SPIRVTargetMachine *TM) : FunctionPass(ID), TM(TM) {
    initializeSPIRVLegalizePointerLoadPass(*PassRegistry::getPassRegistry());
  };

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

char SPIRVLegalizePointerLoad::ID = 0;
INITIALIZE_PASS(SPIRVLegalizePointerLoad, "spirv-legalize-bitcast",
                "SPIRV legalize bitcast pass", false, false)

FunctionPass *llvm::createSPIRVLegalizePointerLoadPass(SPIRVTargetMachine *TM) {
  return new SPIRVLegalizePointerLoad(TM);
}
