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
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;

namespace llvm {
void initializeSPIRVLegalizePointerCastPass(PassRegistry &);
}

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
    // We expect the codegen to avoid doing implicit bitcast from a load.
    assert(TargetType->getElementType() == SourceType->getElementType());
    assert(TargetType->getNumElements() < SourceType->getNumElements());

    LoadInst *NewLoad = B.CreateLoad(SourceType, Source);
    buildAssignType(B, SourceType, NewLoad);

    SmallVector<int> Mask(/* Size= */ TargetType->getNumElements(),
                          /* Value= */ 0);
    Value *Output = B.CreateShuffleVector(NewLoad, NewLoad, Mask);
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

    const auto *TLI = TM->getSubtargetImpl()->getTargetLowering();
    MachineMemOperand::Flags Flags = TLI->getLoadMemOperandFlags(
        *BadLoad, BadLoad->getFunction()->getDataLayout());
    Instruction *LI = B.CreateIntrinsic(
        Intrinsic::spv_load, {BadLoad->getOperand(0)->getType()},
        {GEP, B.getInt16(Flags), B.getInt8(BadLoad->getAlign().value())});
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
    // Destination is a smaller vector than source.
    // - float3 v3 = vector4;
    else if (SVT && DVT)
      Output = loadVectorFromVector(B, SVT, DVT, OriginalOperand);
    else
      llvm_unreachable("Unimplemented implicit down-cast from load.");

    GR->replaceAllUsesWith(LI, Output, /* DeleteOld= */ true);
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
  SPIRVLegalizePointerCast(SPIRVTargetMachine *TM) : FunctionPass(ID), TM(TM) {
    initializeSPIRVLegalizePointerCastPass(*PassRegistry::getPassRegistry());
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

char SPIRVLegalizePointerCast::ID = 0;
INITIALIZE_PASS(SPIRVLegalizePointerCast, "spirv-legalize-bitcast",
                "SPIRV legalize bitcast pass", false, false)

FunctionPass *llvm::createSPIRVLegalizePointerCastPass(SPIRVTargetMachine *TM) {
  return new SPIRVLegalizePointerCast(TM);
}
