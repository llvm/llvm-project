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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "dxil-data-scalarization"
static const int MaxVecSize = 4;

using namespace llvm;

// Recursively creates an array-like version of a given vector type.
static Type *equivalentArrayTypeFromVector(Type *T) {
  if (auto *VecTy = dyn_cast<VectorType>(T))
    return ArrayType::get(VecTy->getElementType(),
                          dyn_cast<FixedVectorType>(VecTy)->getNumElements());
  if (auto *ArrayTy = dyn_cast<ArrayType>(T)) {
    Type *NewElementType =
        equivalentArrayTypeFromVector(ArrayTy->getElementType());
    return ArrayType::get(NewElementType, ArrayTy->getNumElements());
  }
  // If it's not a vector or array, return the original type.
  return T;
}

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
  bool visitInsertElementInst(InsertElementInst &IEI);
  bool visitExtractElementInst(ExtractElementInst &EEI);
  bool visitShuffleVectorInst(ShuffleVectorInst &SVI) { return false; }
  bool visitPHINode(PHINode &PHI) { return false; }
  bool visitLoadInst(LoadInst &LI);
  bool visitStoreInst(StoreInst &SI);
  bool visitCallInst(CallInst &ICI) { return false; }
  bool visitFreezeInst(FreezeInst &FI) { return false; }
  friend bool findAndReplaceVectors(llvm::Module &M);

private:
  typedef std::pair<AllocaInst *, SmallVector<Value *, 4>> AllocaAndGEPs;
  typedef SmallDenseMap<Value *, AllocaAndGEPs>
      VectorToArrayMap; // A map from a vector-typed Value to its corresponding
                        // AllocaInst and GEPs to each element of an array
  VectorToArrayMap VectorAllocaMap;
  AllocaAndGEPs createArrayFromVector(IRBuilder<> &Builder, Value *Vec,
                                      const Twine &Name);
  bool replaceDynamicInsertElementInst(InsertElementInst &IEI);
  bool replaceDynamicExtractElementInst(ExtractElementInst &EEI);

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
  VectorAllocaMap.clear();
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

// Helper function to check if a type is a vector or an array of vectors
static bool isVectorOrArrayOfVectors(Type *T) {
  if (isa<VectorType>(T))
    return true;
  if (ArrayType *ArrType = dyn_cast<ArrayType>(T))
    return isa<VectorType>(ArrType->getElementType()) ||
           isVectorOrArrayOfVectors(ArrType->getElementType());
  return false;
}

bool DataScalarizerVisitor::visitAllocaInst(AllocaInst &AI) {
  Type *AllocatedType = AI.getAllocatedType();
  if (!isVectorOrArrayOfVectors(AllocatedType))
    return false;

  IRBuilder<> Builder(&AI);
  Type *NewType = equivalentArrayTypeFromVector(AllocatedType);
  AllocaInst *ArrAlloca =
      Builder.CreateAlloca(NewType, nullptr, AI.getName() + ".scalarize");
  ArrAlloca->setAlignment(AI.getAlign());
  AI.replaceAllUsesWith(ArrAlloca);
  AI.eraseFromParent();
  return true;
}

bool DataScalarizerVisitor::visitLoadInst(LoadInst &LI) {
  Value *PtrOperand = LI.getPointerOperand();
  ConstantExpr *CE = dyn_cast<ConstantExpr>(PtrOperand);
  if (CE && CE->getOpcode() == Instruction::GetElementPtr) {
    GetElementPtrInst *OldGEP = cast<GetElementPtrInst>(CE->getAsInstruction());
    OldGEP->insertBefore(LI.getIterator());
    IRBuilder<> Builder(&LI);
    LoadInst *NewLoad = Builder.CreateLoad(LI.getType(), OldGEP, LI.getName());
    NewLoad->setAlignment(LI.getAlign());
    LI.replaceAllUsesWith(NewLoad);
    LI.eraseFromParent();
    visitGetElementPtrInst(*OldGEP);
    return true;
  }
  if (GlobalVariable *NewGlobal = lookupReplacementGlobal(PtrOperand))
    LI.setOperand(LI.getPointerOperandIndex(), NewGlobal);
  return false;
}

bool DataScalarizerVisitor::visitStoreInst(StoreInst &SI) {

  Value *PtrOperand = SI.getPointerOperand();
  ConstantExpr *CE = dyn_cast<ConstantExpr>(PtrOperand);
  if (CE && CE->getOpcode() == Instruction::GetElementPtr) {
    GetElementPtrInst *OldGEP = cast<GetElementPtrInst>(CE->getAsInstruction());
    OldGEP->insertBefore(SI.getIterator());
    IRBuilder<> Builder(&SI);
    StoreInst *NewStore = Builder.CreateStore(SI.getValueOperand(), OldGEP);
    NewStore->setAlignment(SI.getAlign());
    SI.replaceAllUsesWith(NewStore);
    SI.eraseFromParent();
    visitGetElementPtrInst(*OldGEP);
    return true;
  }
  if (GlobalVariable *NewGlobal = lookupReplacementGlobal(PtrOperand))
    SI.setOperand(SI.getPointerOperandIndex(), NewGlobal);

  return false;
}

DataScalarizerVisitor::AllocaAndGEPs
DataScalarizerVisitor::createArrayFromVector(IRBuilder<> &Builder, Value *Vec,
                                             const Twine &Name = "") {
  // If there is already an alloca for this vector, return it
  if (VectorAllocaMap.contains(Vec))
    return VectorAllocaMap[Vec];

  auto InsertPoint = Builder.GetInsertPoint();

  // Allocate the array to hold the vector elements
  Builder.SetInsertPointPastAllocas(Builder.GetInsertBlock()->getParent());
  Type *ArrTy = equivalentArrayTypeFromVector(Vec->getType());
  AllocaInst *ArrAlloca =
      Builder.CreateAlloca(ArrTy, nullptr, Name + ".alloca");
  const uint64_t ArrNumElems = ArrTy->getArrayNumElements();

  // Create loads and stores to populate the array immediately after the
  // original vector's defining instruction if available, else immediately after
  // the alloca
  if (auto *Instr = dyn_cast<Instruction>(Vec))
    Builder.SetInsertPoint(Instr->getNextNode());
  SmallVector<Value *, 4> GEPs(ArrNumElems);
  for (unsigned I = 0; I < ArrNumElems; ++I) {
    Value *EE = Builder.CreateExtractElement(Vec, I, Name + ".extract");
    GEPs[I] = Builder.CreateInBoundsGEP(
        ArrTy, ArrAlloca, {Builder.getInt32(0), Builder.getInt32(I)},
        Name + ".index");
    Builder.CreateStore(EE, GEPs[I]);
  }

  VectorAllocaMap.insert({Vec, {ArrAlloca, GEPs}});
  Builder.SetInsertPoint(InsertPoint);
  return {ArrAlloca, GEPs};
}

/// Returns a pair of Value* with the first being a GEP into ArrAlloca using
/// indices {0, Index}, and the second Value* being a Load of the GEP
static std::pair<Value *, Value *>
dynamicallyLoadArray(IRBuilder<> &Builder, AllocaInst *ArrAlloca, Value *Index,
                     const Twine &Name = "") {
  Type *ArrTy = ArrAlloca->getAllocatedType();
  Value *GEP = Builder.CreateInBoundsGEP(
      ArrTy, ArrAlloca, {Builder.getInt32(0), Index}, Name + ".index");
  Value *Load =
      Builder.CreateLoad(ArrTy->getArrayElementType(), GEP, Name + ".load");
  return std::make_pair(GEP, Load);
}

bool DataScalarizerVisitor::replaceDynamicInsertElementInst(
    InsertElementInst &IEI) {
  IRBuilder<> Builder(&IEI);

  Value *Vec = IEI.getOperand(0);
  Value *Val = IEI.getOperand(1);
  Value *Index = IEI.getOperand(2);

  AllocaAndGEPs ArrAllocaAndGEPs =
      createArrayFromVector(Builder, Vec, IEI.getName());
  AllocaInst *ArrAlloca = ArrAllocaAndGEPs.first;
  Type *ArrTy = ArrAlloca->getAllocatedType();
  SmallVector<Value *, 4> &ArrGEPs = ArrAllocaAndGEPs.second;

  auto GEPAndLoad =
      dynamicallyLoadArray(Builder, ArrAlloca, Index, IEI.getName());
  Value *GEP = GEPAndLoad.first;
  Value *Load = GEPAndLoad.second;

  Builder.CreateStore(Val, GEP);
  Value *NewIEI = PoisonValue::get(Vec->getType());
  for (unsigned I = 0; I < ArrTy->getArrayNumElements(); ++I) {
    Value *Load = Builder.CreateLoad(ArrTy->getArrayElementType(), ArrGEPs[I],
                                     IEI.getName() + ".load");
    NewIEI = Builder.CreateInsertElement(NewIEI, Load, Builder.getInt32(I),
                                         IEI.getName() + ".insert");
  }

  // Store back the original value so the Alloca can be reused for subsequent
  // insertelement instructions on the same vector
  Builder.CreateStore(Load, GEP);

  IEI.replaceAllUsesWith(NewIEI);
  IEI.eraseFromParent();
  return true;
}

bool DataScalarizerVisitor::visitInsertElementInst(InsertElementInst &IEI) {
  // If the index is a constant then we don't need to scalarize it
  Value *Index = IEI.getOperand(2);
  if (isa<ConstantInt>(Index))
    return false;
  return replaceDynamicInsertElementInst(IEI);
}

bool DataScalarizerVisitor::replaceDynamicExtractElementInst(
    ExtractElementInst &EEI) {
  IRBuilder<> Builder(&EEI);

  AllocaAndGEPs ArrAllocaAndGEPs =
      createArrayFromVector(Builder, EEI.getVectorOperand(), EEI.getName());
  AllocaInst *ArrAlloca = ArrAllocaAndGEPs.first;

  auto GEPAndLoad = dynamicallyLoadArray(Builder, ArrAlloca,
                                         EEI.getIndexOperand(), EEI.getName());
  Value *Load = GEPAndLoad.second;

  EEI.replaceAllUsesWith(Load);
  EEI.eraseFromParent();
  return true;
}

bool DataScalarizerVisitor::visitExtractElementInst(ExtractElementInst &EEI) {
  // If the index is a constant then we don't need to scalarize it
  Value *Index = EEI.getIndexOperand();
  if (isa<ConstantInt>(Index))
    return false;
  return replaceDynamicExtractElementInst(EEI);
}

bool DataScalarizerVisitor::visitGetElementPtrInst(GetElementPtrInst &GEPI) {
  GEPOperator *GOp = cast<GEPOperator>(&GEPI);
  Value *PtrOperand = GOp->getPointerOperand();
  Type *NewGEPType = GOp->getSourceElementType();
  bool NeedsTransform = false;

  // Unwrap GEP ConstantExprs to find the base operand and element type
  while (auto *CE = dyn_cast<ConstantExpr>(PtrOperand)) {
    if (auto *GEPCE = dyn_cast<GEPOperator>(CE)) {
      GOp = GEPCE;
      PtrOperand = GEPCE->getPointerOperand();
      NewGEPType = GEPCE->getSourceElementType();
    } else
      break;
  }

  if (GlobalVariable *NewGlobal = lookupReplacementGlobal(PtrOperand)) {
    NewGEPType = NewGlobal->getValueType();
    PtrOperand = NewGlobal;
    NeedsTransform = true;
  } else if (AllocaInst *Alloca = dyn_cast<AllocaInst>(PtrOperand)) {
    Type *AllocatedType = Alloca->getAllocatedType();
    if (isa<ArrayType>(AllocatedType) &&
        AllocatedType != GOp->getResultElementType()) {
      NewGEPType = AllocatedType;
      NeedsTransform = true;
    }
  }

  if (!NeedsTransform)
    return false;

  // Keep scalar GEPs scalar; dxil-flatten-arrays will do flattening later
  if (!isa<ArrayType>(GOp->getSourceElementType()))
    NewGEPType = GOp->getSourceElementType();

  IRBuilder<> Builder(&GEPI);
  SmallVector<Value *, MaxVecSize> Indices(GOp->indices());
  Value *NewGEP = Builder.CreateGEP(NewGEPType, PtrOperand, Indices,
                                    GOp->getName(), GOp->getNoWrapFlags());

  GOp->replaceAllUsesWith(NewGEP);

  if (auto *OldGEPI = dyn_cast<GetElementPtrInst>(GOp))
    OldGEPI->eraseFromParent();

  return true;
}

static Constant *transformInitializer(Constant *Init, Type *OrigType,
                                      Type *NewType, LLVMContext &Ctx) {
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

    Type *NewType = equivalentArrayTypeFromVector(OrigType);
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
