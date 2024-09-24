//===- DXILDataScalarization.cpp - Prepare LLVM Module for DXIL Data
// Legalization----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------------------===//

#include "DXILDataScalarization.h"
#include "DirectX.h"
#include "llvm/ADT/PostOrderIterator.h"
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
#include <utility>

#define DEBUG_TYPE "dxil-data-scalarization"
#define Max_VEC_SIZE 4

using namespace llvm;

static void findAndReplaceVectors(Module &M);

class DataScalarizerVisitor : public InstVisitor<DataScalarizerVisitor, bool> {
public:
  DataScalarizerVisitor() : GlobalMap() {}
  bool visit(Function &F);
  // InstVisitor methods.  They return true if the instruction was scalarized,
  // false if nothing changed.
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
  friend void findAndReplaceVectors(llvm::Module &M);

private:
  GlobalVariable *getNewGlobalIfExists(Value *CurrOperand);
  DenseMap<GlobalVariable *, GlobalVariable *> GlobalMap;
  SmallVector<WeakTrackingVH, 32> PotentiallyDeadInstrs;
  bool finish();
};

bool DataScalarizerVisitor::visit(Function &F) {
  assert(!GlobalMap.empty());
  ReversePostOrderTraversal<BasicBlock *> RPOT(&F.getEntryBlock());
  for (BasicBlock *BB : RPOT) {
    for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE;) {
      Instruction *I = &*II;
      bool Done = InstVisitor::visit(I);
      ++II;
      if (Done && I->getType()->isVoidTy())
        I->eraseFromParent();
    }
  }
  return finish();
}

bool DataScalarizerVisitor::finish() {
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(PotentiallyDeadInstrs);
  return true;
}

GlobalVariable *
DataScalarizerVisitor::getNewGlobalIfExists(Value *CurrOperand) {
  if (GlobalVariable *OldGlobal = dyn_cast<GlobalVariable>(CurrOperand)) {
    auto It = GlobalMap.find(OldGlobal);
    if (It != GlobalMap.end()) {
      return It->second; // Found, return the new global
    }
  }
  return nullptr; // Not found
}

bool DataScalarizerVisitor::visitLoadInst(LoadInst &LI) {
  for (unsigned I = 0; I < LI.getNumOperands(); ++I) {
    Value *CurrOpperand = LI.getOperand(I);
    GlobalVariable *NewGlobal = getNewGlobalIfExists(CurrOpperand);
    if (NewGlobal)
      LI.setOperand(I, NewGlobal);
  }
  return false;
}

bool DataScalarizerVisitor::visitStoreInst(StoreInst &SI) {
  for (unsigned I = 0; I < SI.getNumOperands(); ++I) {
    Value *CurrOpperand = SI.getOperand(I);
    GlobalVariable *NewGlobal = getNewGlobalIfExists(CurrOpperand);
    if (NewGlobal) {
      SI.setOperand(I, NewGlobal);
    }
  }
  return false;
}

bool DataScalarizerVisitor::visitGetElementPtrInst(GetElementPtrInst &GEPI) {
  for (unsigned I = 0; I < GEPI.getNumOperands(); ++I) {
    Value *CurrOpperand = GEPI.getOperand(I);
    GlobalVariable *NewGlobal = getNewGlobalIfExists(CurrOpperand);
    if (NewGlobal) {
      IRBuilder<> Builder(&GEPI);

      SmallVector<Value *, Max_VEC_SIZE> Indices;
      for (auto &Index : GEPI.indices())
        Indices.push_back(Index);

      Value *NewGEP =
          Builder.CreateGEP(NewGlobal->getValueType(), NewGlobal, Indices);

      GEPI.replaceAllUsesWith(NewGEP);
      PotentiallyDeadInstrs.emplace_back(&GEPI);
    }
  }
  return true;
}

// Recursively Creates and Array like version of the given vector like type.
static Type *replaceVectorWithArray(Type *T, LLVMContext &Ctx) {
  if (auto *VecTy = dyn_cast<VectorType>(T))
    return ArrayType::get(VecTy->getElementType(),
                          cast<FixedVectorType>(VecTy)->getNumElements());
  if (auto *ArrayTy = dyn_cast<ArrayType>(T)) {
    Type *NewElementType =
        replaceVectorWithArray(ArrayTy->getElementType(), Ctx);
    return ArrayType::get(NewElementType, ArrayTy->getNumElements());
  }
  // If it's not a vector or array, return the original type.
  return T;
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
    auto *VecInit = dyn_cast<ConstantVector>(Init);
    if (!VecInit) {
      llvm_unreachable("Expected a ConstantVector for vector initializer!");
    }

    SmallVector<Constant *, Max_VEC_SIZE> ArrayElements;
    for (unsigned I = 0; I < VecInit->getNumOperands(); ++I) {
      ArrayElements.push_back(VecInit->getOperand(I));
    }

    return ConstantArray::get(cast<ArrayType>(NewType), ArrayElements);
  }

  // Handle array of vectors transformation
  if (auto *ArrayTy = dyn_cast<ArrayType>(OrigType)) {

    auto *ArrayInit = dyn_cast<ConstantArray>(Init);
    if (!ArrayInit) {
      llvm_unreachable("Expected a ConstantArray for array initializer!");
    }

    SmallVector<Constant *, Max_VEC_SIZE> NewArrayElements;
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

static void findAndReplaceVectors(Module &M) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> Builder(Ctx);
  DataScalarizerVisitor Impl;
  for (GlobalVariable &G : M.globals()) {
    Type *OrigType = G.getValueType();

    Type *NewType = replaceVectorWithArray(OrigType, Ctx);
    if (OrigType != NewType) {
      // Create a new global variable with the updated type
      GlobalVariable *NewGlobal = new GlobalVariable(
          M, NewType, G.isConstant(), G.getLinkage(),
          // This is set via: transformInitializer
          nullptr, G.getName() + ".scalarized", &G, G.getThreadLocalMode(),
          G.getAddressSpace(), G.isExternallyInitialized());

      // Copy relevant attributes
      NewGlobal->setUnnamedAddr(G.getUnnamedAddr());
      if (G.getAlignment() > 0) {
        NewGlobal->setAlignment(Align(G.getAlignment()));
      }

      if (G.hasInitializer()) {
        Constant *Init = G.getInitializer();
        Constant *NewInit = transformInitializer(Init, OrigType, NewType, Ctx);
        NewGlobal->setInitializer(NewInit);
      }

      // Note: we want to do G.replaceAllUsesWith(NewGlobal);, but it assumes
      // type equality
      //  So instead we will use the visitor pattern
      Impl.GlobalMap[&G] = NewGlobal;
      for (User *U : G.users()) {
        // Note: The GEPS are stored as constExprs
        // This step flattens them out to instructions
        if (isa<ConstantExpr>(U) && isa<Operator>(U)) {
          ConstantExpr *CE = cast<ConstantExpr>(U);
          convertUsersOfConstantsToInstructions(CE,
                                                /*RestrictToFunc=*/nullptr,
                                                /*RemoveDeadConstants=*/false,
                                                /*IncludeSelf=*/true);
        }
      }
      // Uses should have grown
      std::vector<User *> UsersToProcess;
      // Collect all users first
      // work around so I can delete
      // in a loop body
      for (User *U : G.users()) {
        UsersToProcess.push_back(U);
      }

      // Now process each user
      for (User *U : UsersToProcess) {
        if (isa<Instruction>(U)) {
          Instruction *Inst = cast<Instruction>(U);
          Function *F = Inst->getFunction();
          if (F)
            Impl.visit(*F);
        }
      }
    }
  }

  // Remove the old globals after the iteration
  for (auto Pair : Impl.GlobalMap) {
    GlobalVariable *OldG = Pair.getFirst();
    OldG->eraseFromParent();
  }
}

PreservedAnalyses DXILDataScalarization::run(Module &M,
                                             ModuleAnalysisManager &) {
  findAndReplaceVectors(M);
  return PreservedAnalyses::none();
}

bool DXILDataScalarizationLegacy::runOnModule(Module &M) {
  findAndReplaceVectors(M);
  return true;
}

void DXILDataScalarizationLegacy::getAnalysisUsage(AnalysisUsage &AU) const {}

char DXILDataScalarizationLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILDataScalarizationLegacy, DEBUG_TYPE,
                      "DXIL Data Scalarization", false, false)
INITIALIZE_PASS_END(DXILDataScalarizationLegacy, DEBUG_TYPE,
                    "DXIL Data Scalarization", false, false)

ModulePass *llvm::createDXILDataScalarizationLegacyPass() {
  return new DXILDataScalarizationLegacy();
}