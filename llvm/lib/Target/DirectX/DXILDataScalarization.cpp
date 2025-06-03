//===- DXILDataScalarization.cpp - Perform DXIL Data Legalization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "DXILDataScalarization.h"
#include "DirectX.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "dxil-data-scalarization"
static const int MaxVecSize = 4;

using namespace llvm;

class DXILDataScalarizationLegacy : public ModulePass {

public:
  bool runOnModule(Module &M) override;
  DXILDataScalarizationLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
};

static bool findAndReplaceVectors(Module &M);

class DataScalarizerVisitor : public InstVisitor<DataScalarizerVisitor, bool> {
public:
  DataScalarizerVisitor() : GlobalMap() {}
  bool visit(Function &F);
  // InstVisitor methods.  They return true if the instruction was scalarized,
  // false if nothing changed.
  bool visitAllocaInst(AllocaInst &AI);
  bool visitInstruction(Instruction &I) { return false; }
  bool visitSelectInst(SelectInst &SI) { return false; }
  bool visitICmpInst(ICmpInst &ICI) { return false; }
  bool visitFCmpInst(FCmpInst &FCI) { return false; }
  bool visitUnaryOperator(UnaryOperator &UO) { return false; }
  bool visitBinaryOperator(BinaryOperator &BO) { return false; }
  bool visitGetElementPtrInst(GetElementPtrInst &GEPI);
  bool visitCastInst(CastInst &CI) { return false; }
  bool visitBitCastInst(BitCastInst &BCI) { return false; }
  bool visitInsertElementInst(InsertElementInst &IEI) { return false; }
  bool visitExtractElementInst(ExtractElementInst &EEI) { return false; }
  bool visitShuffleVectorInst(ShuffleVectorInst &SVI) { return false; }
  bool visitPHINode(PHINode &PHI) { return false; }
  bool visitLoadInst(LoadInst &LI);
  bool visitStoreInst(StoreInst &SI);
  bool visitCallInst(CallInst &ICI) { return false; }
  bool visitFreezeInst(FreezeInst &FI) { return false; }
  friend bool findAndReplaceVectors(llvm::Module &M);

private:
  GlobalVariable *lookupReplacementGlobal(Value *CurrOperand);
  DenseMap<GlobalVariable *, GlobalVariable *> GlobalMap;
};

bool DataScalarizerVisitor::visit(Function &F) {
  bool MadeChange = false;
  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (BasicBlock *BB : make_early_inc_range(RPOT)) {
    for (Instruction &I : make_early_inc_range(*BB))
      MadeChange |= InstVisitor::visit(I);
  }
  return MadeChange;
}

GlobalVariable *
DataScalarizerVisitor::lookupReplacementGlobal(Value *CurrOperand) {
  if (GlobalVariable *OldGlobal = dyn_cast<GlobalVariable>(CurrOperand)) {
    auto It = GlobalMap.find(OldGlobal);
    if (It != GlobalMap.end()) {
      return It->second; // Found, return the new global
    }
  }
  return nullptr; // Not found
}

// Recursively creates an array version of the given vector type.
static Type *replaceVectorWithArray(Type *T, LLVMContext &Ctx) {
  if (auto *VecTy = dyn_cast<VectorType>(T))
    return ArrayType::get(VecTy->getElementType(),
                          dyn_cast<FixedVectorType>(VecTy)->getNumElements());
  if (auto *ArrayTy = dyn_cast<ArrayType>(T)) {
    Type *NewElementType =
        replaceVectorWithArray(ArrayTy->getElementType(), Ctx);
    return ArrayType::get(NewElementType, ArrayTy->getNumElements());
  }
  // If it's not a vector or array, return the original type.
  return T;
}

static bool isArrayOfVectors(Type *T) {
  if (ArrayType *ArrType = dyn_cast<ArrayType>(T))
    return isa<VectorType>(ArrType->getElementType());
  return false;
}

bool DataScalarizerVisitor::visitAllocaInst(AllocaInst &AI) {
  if (!isArrayOfVectors(AI.getAllocatedType()))
    return false;

  ArrayType *ArrType = cast<ArrayType>(AI.getAllocatedType());
  IRBuilder<> Builder(&AI);
  LLVMContext &Ctx = AI.getContext();
  Type *NewType = replaceVectorWithArray(ArrType, Ctx);
  AllocaInst *ArrAlloca =
      Builder.CreateAlloca(NewType, nullptr, AI.getName() + ".scalarize");
  ArrAlloca->setAlignment(AI.getAlign());
  AI.replaceAllUsesWith(ArrAlloca);
  AI.eraseFromParent();
  return true;
}

bool DataScalarizerVisitor::visitLoadInst(LoadInst &LI) {
  unsigned NumOperands = LI.getNumOperands();
  for (unsigned I = 0; I < NumOperands; ++I) {
    Value *CurrOpperand = LI.getOperand(I);
    ConstantExpr *CE = dyn_cast<ConstantExpr>(CurrOpperand);
    if (CE && CE->getOpcode() == Instruction::GetElementPtr) {
      GetElementPtrInst *OldGEP =
          cast<GetElementPtrInst>(CE->getAsInstruction());
      OldGEP->insertBefore(LI.getIterator());
      IRBuilder<> Builder(&LI);
      LoadInst *NewLoad =
          Builder.CreateLoad(LI.getType(), OldGEP, LI.getName());
      NewLoad->setAlignment(LI.getAlign());
      LI.replaceAllUsesWith(NewLoad);
      LI.eraseFromParent();
      visitGetElementPtrInst(*OldGEP);
      return true;
    }
    if (GlobalVariable *NewGlobal = lookupReplacementGlobal(CurrOpperand))
      LI.setOperand(I, NewGlobal);
  }
  return false;
}

bool DataScalarizerVisitor::visitStoreInst(StoreInst &SI) {
  unsigned NumOperands = SI.getNumOperands();
  for (unsigned I = 0; I < NumOperands; ++I) {
    Value *CurrOpperand = SI.getOperand(I);
    ConstantExpr *CE = dyn_cast<ConstantExpr>(CurrOpperand);
    if (CE && CE->getOpcode() == Instruction::GetElementPtr) {
      GetElementPtrInst *OldGEP =
          cast<GetElementPtrInst>(CE->getAsInstruction());
      OldGEP->insertBefore(SI.getIterator());
      IRBuilder<> Builder(&SI);
      StoreInst *NewStore = Builder.CreateStore(SI.getValueOperand(), OldGEP);
      NewStore->setAlignment(SI.getAlign());
      SI.replaceAllUsesWith(NewStore);
      SI.eraseFromParent();
      visitGetElementPtrInst(*OldGEP);
      return true;
    }
    if (GlobalVariable *NewGlobal = lookupReplacementGlobal(CurrOpperand))
      SI.setOperand(I, NewGlobal);
  }
  return false;
}

bool DataScalarizerVisitor::visitGetElementPtrInst(GetElementPtrInst &GEPI) {

  unsigned NumOperands = GEPI.getNumOperands();
  GlobalVariable *NewGlobal = nullptr;
  for (unsigned I = 0; I < NumOperands; ++I) {
    Value *CurrOpperand = GEPI.getOperand(I);
    NewGlobal = lookupReplacementGlobal(CurrOpperand);
    if (NewGlobal)
      break;
  }
  if (!NewGlobal)
    return false;

  IRBuilder<> Builder(&GEPI);
  SmallVector<Value *, MaxVecSize> Indices(GEPI.indices());

  Value *NewGEP =
      Builder.CreateGEP(NewGlobal->getValueType(), NewGlobal, Indices,
                        GEPI.getName(), GEPI.getNoWrapFlags());
  GEPI.replaceAllUsesWith(NewGEP);
  GEPI.eraseFromParent();
  return true;
}

Constant *transformInitializer(Constant *Init, Type *OrigType, Type *NewType,
                               LLVMContext &Ctx) {
  // Handle ConstantAggregateZero (zero-initialized constants)
  if (isa<ConstantAggregateZero>(Init)) {
    return ConstantAggregateZero::get(NewType);
  }

  // Handle UndefValue (undefined constants)
  if (isa<UndefValue>(Init)) {
    return UndefValue::get(NewType);
  }

  // Handle vector to array transformation
  if (isa<VectorType>(OrigType) && isa<ArrayType>(NewType)) {
    // Convert vector initializer to array initializer
    SmallVector<Constant *, MaxVecSize> ArrayElements;
    if (ConstantVector *ConstVecInit = dyn_cast<ConstantVector>(Init)) {
      for (unsigned I = 0; I < ConstVecInit->getNumOperands(); ++I)
        ArrayElements.push_back(ConstVecInit->getOperand(I));
    } else if (ConstantDataVector *ConstDataVecInit =
                   llvm::dyn_cast<llvm::ConstantDataVector>(Init)) {
      for (unsigned I = 0; I < ConstDataVecInit->getNumElements(); ++I)
        ArrayElements.push_back(ConstDataVecInit->getElementAsConstant(I));
    } else {
      assert(false && "Expected a ConstantVector or ConstantDataVector for "
                      "vector initializer!");
    }

    return ConstantArray::get(cast<ArrayType>(NewType), ArrayElements);
  }

  // Handle array of vectors transformation
  if (auto *ArrayTy = dyn_cast<ArrayType>(OrigType)) {
    auto *ArrayInit = dyn_cast<ConstantArray>(Init);
    assert(ArrayInit && "Expected a ConstantArray for array initializer!");

    SmallVector<Constant *, MaxVecSize> NewArrayElements;
    for (unsigned I = 0; I < ArrayTy->getNumElements(); ++I) {
      // Recursively transform array elements
      Constant *NewElemInit = transformInitializer(
          ArrayInit->getOperand(I), ArrayTy->getElementType(),
          cast<ArrayType>(NewType)->getElementType(), Ctx);
      NewArrayElements.push_back(NewElemInit);
    }

    return ConstantArray::get(cast<ArrayType>(NewType), NewArrayElements);
  }

  // If not a vector or array, return the original initializer
  return Init;
}

static bool findAndReplaceVectors(Module &M) {
  bool MadeChange = false;
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> Builder(Ctx);
  DataScalarizerVisitor Impl;
  for (GlobalVariable &G : M.globals()) {
    Type *OrigType = G.getValueType();

    Type *NewType = replaceVectorWithArray(OrigType, Ctx);
    if (OrigType != NewType) {
      // Create a new global variable with the updated type
      // Note: Initializer is set via transformInitializer
      GlobalVariable *NewGlobal = new GlobalVariable(
          M, NewType, G.isConstant(), G.getLinkage(),
          /*Initializer=*/nullptr, G.getName() + ".scalarized", &G,
          G.getThreadLocalMode(), G.getAddressSpace(),
          G.isExternallyInitialized());

      // Copy relevant attributes
      NewGlobal->setUnnamedAddr(G.getUnnamedAddr());
      if (G.getAlignment() > 0) {
        NewGlobal->setAlignment(G.getAlign());
      }

      if (G.hasInitializer()) {
        Constant *Init = G.getInitializer();
        Constant *NewInit = transformInitializer(Init, OrigType, NewType, Ctx);
        NewGlobal->setInitializer(NewInit);
      }

      // Note: we want to do G.replaceAllUsesWith(NewGlobal);, but it assumes
      // type equality. Instead we will use the visitor pattern.
      Impl.GlobalMap[&G] = NewGlobal;
    }
  }

  for (auto &F : make_early_inc_range(M.functions())) {
    if (F.isDeclaration())
      continue;
    MadeChange |= Impl.visit(F);
  }

  // Remove the old globals after the iteration
  for (auto &[Old, New] : Impl.GlobalMap) {
    Old->eraseFromParent();
    MadeChange = true;
  }
  return MadeChange;
}

PreservedAnalyses DXILDataScalarization::run(Module &M,
                                             ModuleAnalysisManager &) {
  bool MadeChanges = findAndReplaceVectors(M);
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  return PA;
}

bool DXILDataScalarizationLegacy::runOnModule(Module &M) {
  return findAndReplaceVectors(M);
}

char DXILDataScalarizationLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILDataScalarizationLegacy, DEBUG_TYPE,
                      "DXIL Data Scalarization", false, false)
INITIALIZE_PASS_END(DXILDataScalarizationLegacy, DEBUG_TYPE,
                    "DXIL Data Scalarization", false, false)

ModulePass *llvm::createDXILDataScalarizationLegacyPass() {
  return new DXILDataScalarizationLegacy();
}
