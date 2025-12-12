//===- SPIRVLegalizeZeroSizeArrays.cpp - Legalize zero-size arrays -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPIR-V does not support zero-size arrays unless it is within a shader. This
// pass legalizes zero-size arrays ([0 x T]) in unsupported cases.
//
//===----------------------------------------------------------------------===//

#include "SPIRVLegalizeZeroSizeArrays.h"
#include "SPIRV.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "spirv-legalize-zero-size-arrays"

using namespace llvm;

namespace {

bool hasZeroSizeArray(const Type *Ty) {
  if (const ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
    if (ArrTy->getNumElements() == 0)
      return true;
    return hasZeroSizeArray(ArrTy->getElementType());
  }

  if (const StructType *StructTy = dyn_cast<StructType>(Ty)) {
    for (Type *ElemTy : StructTy->elements()) {
      if (hasZeroSizeArray(ElemTy))
        return true;
    }
  }

  return false;
}

class SPIRVLegalizeZeroSizeArraysImpl
    : public InstVisitor<SPIRVLegalizeZeroSizeArraysImpl> {
  friend class InstVisitor<SPIRVLegalizeZeroSizeArraysImpl>;

public:
  bool runOnModule(Module &M);

  // TODO: Handle GEP, PHI
  void visitAllocaInst(AllocaInst &AI);
  void visitLoadInst(LoadInst &LI);
  void visitStoreInst(StoreInst &SI);
  void visitSelectInst(SelectInst &Sel);
  void visitExtractValueInst(ExtractValueInst &EVI);
  void visitInsertValueInst(InsertValueInst &IVI);

private:
  Type *legalizeType(Type *Ty);
  Constant *legalizeConstant(Constant *C);

  DenseMap<Type *, Type *> TypeMap;
  DenseMap<GlobalVariable *, GlobalVariable *> GlobalMap;
  SmallVector<Instruction *, 16> ToErase;
  bool Modified = false;
};

class SPIRVLegalizeZeroSizeArraysLegacy : public ModulePass {
public:
  static char ID;
  SPIRVLegalizeZeroSizeArraysLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "SPIRV Legalize Zero-Size Arrays";
  }
  bool runOnModule(Module &M) override {
    SPIRVLegalizeZeroSizeArraysImpl Impl;
    return Impl.runOnModule(M);
  }
};

Type *SPIRVLegalizeZeroSizeArraysImpl::legalizeType(Type *Ty) {
  auto It = TypeMap.find(Ty);
  if (It != TypeMap.end())
    return It->second;

  Type *LegalizedTy = Ty;

  if (ArrayType *ArrTy = dyn_cast<ArrayType>(Ty)) {
    if (ArrTy->getNumElements() == 0) {
      LegalizedTy = PointerType::getUnqual(Ty->getContext());
    } else if (Type *ElemTy = legalizeType(ArrTy->getElementType());
               ElemTy != ArrTy->getElementType()) {
      LegalizedTy = ArrayType::get(ElemTy, ArrTy->getNumElements());
    }
  } else if (StructType *StructTy = dyn_cast<StructType>(Ty)) {
    SmallVector<Type *, 8> ElemTypes;
    bool Changed = false;
    for (Type *ElemTy : StructTy->elements()) {
      Type *LegalizedElemTy = legalizeType(ElemTy);
      ElemTypes.push_back(LegalizedElemTy);
      Changed |= LegalizedElemTy != ElemTy;
    }
    if (Changed) {
      LegalizedTy =
          StructTy->hasName()
              ? StructType::create(StructTy->getContext(), ElemTypes,
                                   (StructTy->getName() + ".legalized").str(),
                                   StructTy->isPacked())
              : StructType::get(StructTy->getContext(), ElemTypes,
                                StructTy->isPacked());
    }
  }

  TypeMap[Ty] = LegalizedTy;
  return LegalizedTy;
}

Constant *SPIRVLegalizeZeroSizeArraysImpl::legalizeConstant(Constant *C) {
  if (!C || !hasZeroSizeArray(C->getType()))
    return C;

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
    return GlobalMap.lookup(GV) ? GlobalMap[GV] : C;

  Type *NewTy = legalizeType(C->getType());
  if (isa<UndefValue>(C) || isa<PoisonValue>(C))
    return PoisonValue::get(NewTy);
  if (isa<ConstantAggregateZero>(C))
    return Constant::getNullValue(NewTy);

  if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
    SmallVector<Constant *, 8> Elems;
    for (Use &U : CA->operands())
      Elems.push_back(legalizeConstant(cast<Constant>(U)));
    return ConstantArray::get(cast<ArrayType>(NewTy), Elems);
  }

  if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
    SmallVector<Constant *, 8> Fields;
    for (Use &U : CS->operands())
      Fields.push_back(legalizeConstant(cast<Constant>(U)));
    return ConstantStruct::get(cast<StructType>(NewTy), Fields);
  }

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    // Don't legalize GEP constant expressions, the backend deals with them
    // fine.
    if (CE->getOpcode() == Instruction::GetElementPtr)
      return CE;
    SmallVector<Constant *, 4> Ops;
    bool Changed = false;
    for (Use &U : CE->operands()) {
      Constant *LegalizedOp = legalizeConstant(cast<Constant>(U));
      Ops.push_back(LegalizedOp);
      Changed |= LegalizedOp != cast<Constant>(U.get());
    }
    if (Changed)
      return CE->getWithOperands(Ops);
  }

  return C;
}

void SPIRVLegalizeZeroSizeArraysImpl::visitAllocaInst(AllocaInst &AI) {
  if (!hasZeroSizeArray(AI.getAllocatedType()))
    return;

  // TODO: Handle nested arrays and structs containing zero-size arrays
  ArrayType *ArrTy = dyn_cast<ArrayType>(AI.getAllocatedType());
  if (ArrTy && ArrTy->getNumElements() == 0) {
    IRBuilder<> Builder(&AI);
    AllocaInst *NewAI = Builder.CreateAlloca(ArrTy->getElementType(),
                                             AI.getArraySize(), AI.getName());
    NewAI->setAlignment(AI.getAlign());
    NewAI->setDebugLoc(AI.getDebugLoc());
    AI.replaceAllUsesWith(NewAI);
    ToErase.push_back(&AI);
    Modified = true;
  }
}

void SPIRVLegalizeZeroSizeArraysImpl::visitLoadInst(LoadInst &LI) {
  if (!hasZeroSizeArray(LI.getType()))
    return;

  // TODO: Handle nested arrays and structs containing zero-size arrays
  ArrayType *ArrTy = dyn_cast<ArrayType>(LI.getType());
  if (ArrTy && ArrTy->getNumElements() == 0) {
    LI.replaceAllUsesWith(PoisonValue::get(LI.getType()));
    ToErase.push_back(&LI);
    Modified = true;
  }
}

void SPIRVLegalizeZeroSizeArraysImpl::visitStoreInst(StoreInst &SI) {
  Type *StoreTy = SI.getValueOperand()->getType();

  // TODO: Handle nested arrays and structs containing zero-size arrays
  ArrayType *ArrTy = dyn_cast<ArrayType>(StoreTy);
  if (ArrTy && ArrTy->getNumElements() == 0) {
    ToErase.push_back(&SI);
    Modified = true;
  }
}

void SPIRVLegalizeZeroSizeArraysImpl::visitSelectInst(SelectInst &Sel) {
  if (!hasZeroSizeArray(Sel.getType()))
    return;

  // TODO: Handle nested arrays and structs containing zero-size arrays
  ArrayType *ArrTy = dyn_cast<ArrayType>(Sel.getType());
  if (ArrTy && ArrTy->getNumElements() == 0) {
    Sel.replaceAllUsesWith(PoisonValue::get(Sel.getType()));
    ToErase.push_back(&Sel);
    Modified = true;
  }
}

void SPIRVLegalizeZeroSizeArraysImpl::visitExtractValueInst(
    ExtractValueInst &EVI) {
  if (!hasZeroSizeArray(EVI.getAggregateOperand()->getType()))
    return;

  // TODO: Handle nested arrays and structs containing zero-size arrays
  ArrayType *ArrTy = dyn_cast<ArrayType>(EVI.getType());
  if (ArrTy && ArrTy->getNumElements() == 0) {
    EVI.replaceAllUsesWith(PoisonValue::get(EVI.getType()));
    ToErase.push_back(&EVI);
    Modified = true;
  }
}

void SPIRVLegalizeZeroSizeArraysImpl::visitInsertValueInst(
    InsertValueInst &IVI) {
  if (!hasZeroSizeArray(IVI.getAggregateOperand()->getType()))
    return;

  // TODO: Handle nested arrays and structs containing zero-size arrays
  ArrayType *ArrTy =
      dyn_cast<ArrayType>(IVI.getInsertedValueOperand()->getType());
  if (ArrTy && ArrTy->getNumElements() == 0) {
    IVI.replaceAllUsesWith(IVI.getAggregateOperand());
    ToErase.push_back(&IVI);
    Modified = true;
  }
}

bool SPIRVLegalizeZeroSizeArraysImpl::runOnModule(Module &M) {
  TypeMap.clear();
  GlobalMap.clear();
  ToErase.clear();
  Modified = false;

  // Runtime arrays are allowed for shaders, so we don't need to do anything.
  Triple Triple(M.getTargetTriple());
  if (Triple.getOS() == Triple::Vulkan)
    return false;

  // First pass: create new globals and track mapping (don't erase old ones
  // yet).
  SmallVector<GlobalVariable *, 8> OldGlobals;
  for (GlobalVariable &GV : M.globals()) {
    if (!hasZeroSizeArray(GV.getValueType()))
      continue;

    // llvm.embedded.module is handled by SPIRVPrepareGlobals
    if (GV.getName() == "llvm.embedded.module")
      continue;

    Type *NewTy = legalizeType(GV.getValueType());
    // Use an empty name and initializer for now, we will update them in the
    // following steps.
    GlobalVariable *NewGV = new GlobalVariable(
        M, NewTy, GV.isConstant(), GV.getLinkage(), /*Initializer=*/nullptr,
        /*Name=*/"", &GV, GV.getThreadLocalMode(), GV.getAddressSpace(),
        GV.isExternallyInitialized());
    NewGV->copyAttributesFrom(&GV);
    NewGV->copyMetadata(&GV, 0);
    NewGV->setComdat(GV.getComdat());
    NewGV->setAlignment(GV.getAlign());
    GlobalMap[&GV] = NewGV;
    OldGlobals.push_back(&GV);
    Modified = true;
  }

  // Second pass: set initializers now that all globals are mapped.
  for (GlobalVariable *GV : OldGlobals) {
    GlobalVariable *NewGV = cast<GlobalVariable>(GlobalMap[GV]);
    if (GV->hasInitializer())
      NewGV->setInitializer(legalizeConstant(GV->getInitializer()));
  }

  // Third pass: replace uses, transfer names, and erase old globals.
  for (GlobalVariable *GV : OldGlobals) {
    GlobalVariable *NewGV = GlobalMap[GV];
    GV->replaceAllUsesWith(ConstantExpr::getBitCast(NewGV, GV->getType()));
    NewGV->takeName(GV);
    GV->eraseFromParent();
  }

  for (Function &F : M)
    for (Instruction &I : instructions(F))
      visit(I);

  for (Instruction *I : ToErase)
    I->eraseFromParent();

  return Modified;
}

} // namespace

PreservedAnalyses SPIRVLegalizeZeroSizeArrays::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  SPIRVLegalizeZeroSizeArraysImpl Impl;
  if (Impl.runOnModule(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

char SPIRVLegalizeZeroSizeArraysLegacy::ID = 0;

INITIALIZE_PASS(SPIRVLegalizeZeroSizeArraysLegacy,
                "spirv-legalize-zero-size-arrays",
                "Legalize SPIR-V zero-size arrays", false, false)

ModulePass *llvm::createSPIRVLegalizeZeroSizeArraysPass() {
  return new SPIRVLegalizeZeroSizeArraysLegacy();
}
