//===- DXILFlattenArrays.cpp - Flattens DXIL Arrays-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
///
/// \file This file contains a pass to flatten arrays for the DirectX Backend.
///
//===----------------------------------------------------------------------===//

#include "DXILFlattenArrays.h"
#include "DirectX.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#define DEBUG_TYPE "dxil-flatten-arrays"

using namespace llvm;
namespace {

class DXILFlattenArraysLegacy : public ModulePass {

public:
  bool runOnModule(Module &M) override;
  DXILFlattenArraysLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
};

struct GEPData {
  ArrayType *ParentArrayType;
  Value *ParendOperand;
  SmallVector<Value *> Indices;
  SmallVector<uint64_t> Dims;
  bool AllIndicesAreConstInt;
};

class DXILFlattenArraysVisitor
    : public InstVisitor<DXILFlattenArraysVisitor, bool> {
public:
  DXILFlattenArraysVisitor() {}
  bool visit(Function &F);
  // InstVisitor methods.  They return true if the instruction was scalarized,
  // false if nothing changed.
  bool visitGetElementPtrInst(GetElementPtrInst &GEPI);
  bool visitAllocaInst(AllocaInst &AI);
  bool visitInstruction(Instruction &I) { return false; }
  bool visitSelectInst(SelectInst &SI) { return false; }
  bool visitICmpInst(ICmpInst &ICI) { return false; }
  bool visitFCmpInst(FCmpInst &FCI) { return false; }
  bool visitUnaryOperator(UnaryOperator &UO) { return false; }
  bool visitBinaryOperator(BinaryOperator &BO) { return false; }
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
  static bool isMultiDimensionalArray(Type *T);
  static std::pair<unsigned, Type *> getElementCountAndType(Type *ArrayTy);

private:
  SmallVector<WeakTrackingVH> PotentiallyDeadInstrs;
  DenseMap<GetElementPtrInst *, GEPData> GEPChainMap;
  bool finish();
  ConstantInt *genConstFlattenIndices(ArrayRef<Value *> Indices,
                                      ArrayRef<uint64_t> Dims,
                                      IRBuilder<> &Builder);
  Value *genInstructionFlattenIndices(ArrayRef<Value *> Indices,
                                      ArrayRef<uint64_t> Dims,
                                      IRBuilder<> &Builder);
  void
  recursivelyCollectGEPs(GetElementPtrInst &CurrGEP,
                         ArrayType *FlattenedArrayType, Value *PtrOperand,
                         unsigned &GEPChainUseCount,
                         SmallVector<Value *> Indices = SmallVector<Value *>(),
                         SmallVector<uint64_t> Dims = SmallVector<uint64_t>(),
                         bool AllIndicesAreConstInt = true);
  bool visitGetElementPtrInstInGEPChain(GetElementPtrInst &GEP);
  bool visitGetElementPtrInstInGEPChainBase(GEPData &GEPInfo,
                                            GetElementPtrInst &GEP);
};
} // namespace

bool DXILFlattenArraysVisitor::finish() {
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(PotentiallyDeadInstrs);
  return true;
}

bool DXILFlattenArraysVisitor::isMultiDimensionalArray(Type *T) {
  if (ArrayType *ArrType = dyn_cast<ArrayType>(T))
    return isa<ArrayType>(ArrType->getElementType());
  return false;
}

std::pair<unsigned, Type *>
DXILFlattenArraysVisitor::getElementCountAndType(Type *ArrayTy) {
  unsigned TotalElements = 1;
  Type *CurrArrayTy = ArrayTy;
  while (auto *InnerArrayTy = dyn_cast<ArrayType>(CurrArrayTy)) {
    TotalElements *= InnerArrayTy->getNumElements();
    CurrArrayTy = InnerArrayTy->getElementType();
  }
  return std::make_pair(TotalElements, CurrArrayTy);
}

ConstantInt *DXILFlattenArraysVisitor::genConstFlattenIndices(
    ArrayRef<Value *> Indices, ArrayRef<uint64_t> Dims, IRBuilder<> &Builder) {
  assert(Indices.size() == Dims.size() &&
         "Indicies and dimmensions should be the same");
  unsigned FlatIndex = 0;
  unsigned Multiplier = 1;

  for (int I = Indices.size() - 1; I >= 0; --I) {
    unsigned DimSize = Dims[I];
    ConstantInt *CIndex = dyn_cast<ConstantInt>(Indices[I]);
    assert(CIndex && "This function expects all indicies to be ConstantInt");
    FlatIndex += CIndex->getZExtValue() * Multiplier;
    Multiplier *= DimSize;
  }
  return Builder.getInt32(FlatIndex);
}

Value *DXILFlattenArraysVisitor::genInstructionFlattenIndices(
    ArrayRef<Value *> Indices, ArrayRef<uint64_t> Dims, IRBuilder<> &Builder) {
  if (Indices.size() == 1)
    return Indices[0];

  Value *FlatIndex = Builder.getInt32(0);
  unsigned Multiplier = 1;

  for (int I = Indices.size() - 1; I >= 0; --I) {
    unsigned DimSize = Dims[I];
    Value *VMultiplier = Builder.getInt32(Multiplier);
    Value *ScaledIndex = Builder.CreateMul(Indices[I], VMultiplier);
    FlatIndex = Builder.CreateAdd(FlatIndex, ScaledIndex);
    Multiplier *= DimSize;
  }
  return FlatIndex;
}

bool DXILFlattenArraysVisitor::visitLoadInst(LoadInst &LI) {
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
  }
  return false;
}

bool DXILFlattenArraysVisitor::visitStoreInst(StoreInst &SI) {
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
  }
  return false;
}

bool DXILFlattenArraysVisitor::visitAllocaInst(AllocaInst &AI) {
  if (!isMultiDimensionalArray(AI.getAllocatedType()))
    return false;

  ArrayType *ArrType = cast<ArrayType>(AI.getAllocatedType());
  IRBuilder<> Builder(&AI);
  auto [TotalElements, BaseType] = getElementCountAndType(ArrType);

  ArrayType *FattenedArrayType = ArrayType::get(BaseType, TotalElements);
  AllocaInst *FlatAlloca =
      Builder.CreateAlloca(FattenedArrayType, nullptr, AI.getName() + ".flat");
  FlatAlloca->setAlignment(AI.getAlign());
  AI.replaceAllUsesWith(FlatAlloca);
  AI.eraseFromParent();
  return true;
}

void DXILFlattenArraysVisitor::recursivelyCollectGEPs(
    GetElementPtrInst &CurrGEP, ArrayType *FlattenedArrayType,
    Value *PtrOperand, unsigned &GEPChainUseCount, SmallVector<Value *> Indices,
    SmallVector<uint64_t> Dims, bool AllIndicesAreConstInt) {
  Value *LastIndex = CurrGEP.getOperand(CurrGEP.getNumOperands() - 1);
  AllIndicesAreConstInt &= isa<ConstantInt>(LastIndex);
  Indices.push_back(LastIndex);
  assert(isa<ArrayType>(CurrGEP.getSourceElementType()));
  Dims.push_back(
      cast<ArrayType>(CurrGEP.getSourceElementType())->getNumElements());
  bool IsMultiDimArr = isMultiDimensionalArray(CurrGEP.getSourceElementType());
  if (!IsMultiDimArr) {
    assert(GEPChainUseCount < FlattenedArrayType->getNumElements());
    GEPChainMap.insert(
        {&CurrGEP,
         {std::move(FlattenedArrayType), PtrOperand, std::move(Indices),
          std::move(Dims), AllIndicesAreConstInt}});
    return;
  }
  bool GepUses = false;
  for (auto *User : CurrGEP.users()) {
    if (GetElementPtrInst *NestedGEP = dyn_cast<GetElementPtrInst>(User)) {
      recursivelyCollectGEPs(*NestedGEP, FlattenedArrayType, PtrOperand,
                             ++GEPChainUseCount, Indices, Dims,
                             AllIndicesAreConstInt);
      GepUses = true;
    }
  }
  // This case is just incase the gep chain doesn't end with a 1d array.
  if (IsMultiDimArr && GEPChainUseCount > 0 && !GepUses) {
    GEPChainMap.insert(
        {&CurrGEP,
         {std::move(FlattenedArrayType), PtrOperand, std::move(Indices),
          std::move(Dims), AllIndicesAreConstInt}});
  }
}

bool DXILFlattenArraysVisitor::visitGetElementPtrInstInGEPChain(
    GetElementPtrInst &GEP) {
  GEPData GEPInfo = GEPChainMap.at(&GEP);
  return visitGetElementPtrInstInGEPChainBase(GEPInfo, GEP);
}
bool DXILFlattenArraysVisitor::visitGetElementPtrInstInGEPChainBase(
    GEPData &GEPInfo, GetElementPtrInst &GEP) {
  IRBuilder<> Builder(&GEP);
  Value *FlatIndex;
  if (GEPInfo.AllIndicesAreConstInt)
    FlatIndex = genConstFlattenIndices(GEPInfo.Indices, GEPInfo.Dims, Builder);
  else
    FlatIndex =
        genInstructionFlattenIndices(GEPInfo.Indices, GEPInfo.Dims, Builder);

  ArrayType *FlattenedArrayType = GEPInfo.ParentArrayType;
  Value *FlatGEP =
      Builder.CreateGEP(FlattenedArrayType, GEPInfo.ParendOperand, FlatIndex,
                        GEP.getName() + ".flat", GEP.isInBounds());

  GEP.replaceAllUsesWith(FlatGEP);
  GEP.eraseFromParent();
  return true;
}

bool DXILFlattenArraysVisitor::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  auto It = GEPChainMap.find(&GEP);
  if (It != GEPChainMap.end())
    return visitGetElementPtrInstInGEPChain(GEP);
  if (!isMultiDimensionalArray(GEP.getSourceElementType()))
    return false;

  ArrayType *ArrType = cast<ArrayType>(GEP.getSourceElementType());
  IRBuilder<> Builder(&GEP);
  auto [TotalElements, BaseType] = getElementCountAndType(ArrType);
  ArrayType *FlattenedArrayType = ArrayType::get(BaseType, TotalElements);

  Value *PtrOperand = GEP.getPointerOperand();

  unsigned GEPChainUseCount = 0;
  recursivelyCollectGEPs(GEP, FlattenedArrayType, PtrOperand, GEPChainUseCount);

  // NOTE: hasNUses(0) is not the same as GEPChainUseCount == 0.
  // Here recursion is used to get the length of the GEP chain.
  // Handle zero uses here because there won't be an update via
  // a child in the chain later.
  if (GEPChainUseCount == 0) {
    SmallVector<Value *> Indices({GEP.getOperand(GEP.getNumOperands() - 1)});
    SmallVector<uint64_t> Dims({ArrType->getNumElements()});
    bool AllIndicesAreConstInt = isa<ConstantInt>(Indices[0]);
    GEPData GEPInfo{std::move(FlattenedArrayType), PtrOperand,
                    std::move(Indices), std::move(Dims), AllIndicesAreConstInt};
    return visitGetElementPtrInstInGEPChainBase(GEPInfo, GEP);
  }

  PotentiallyDeadInstrs.emplace_back(&GEP);
  return false;
}

bool DXILFlattenArraysVisitor::visit(Function &F) {
  bool MadeChange = false;
  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (BasicBlock *BB : make_early_inc_range(RPOT)) {
    for (Instruction &I : make_early_inc_range(*BB))
      MadeChange |= InstVisitor::visit(I);
  }
  finish();
  return MadeChange;
}

static void collectElements(Constant *Init,
                            SmallVectorImpl<Constant *> &Elements) {
  // Base case: If Init is not an array, add it directly to the vector.
  auto *ArrayTy = dyn_cast<ArrayType>(Init->getType());
  if (!ArrayTy) {
    Elements.push_back(Init);
    return;
  }
  unsigned ArrSize = ArrayTy->getNumElements();
  if (isa<ConstantAggregateZero>(Init)) {
    for (unsigned I = 0; I < ArrSize; ++I)
      Elements.push_back(Constant::getNullValue(ArrayTy->getElementType()));
    return;
  }

  // Recursive case: Process each element in the array.
  if (auto *ArrayConstant = dyn_cast<ConstantArray>(Init)) {
    for (unsigned I = 0; I < ArrayConstant->getNumOperands(); ++I) {
      collectElements(ArrayConstant->getOperand(I), Elements);
    }
  } else if (auto *DataArrayConstant = dyn_cast<ConstantDataArray>(Init)) {
    for (unsigned I = 0; I < DataArrayConstant->getNumElements(); ++I) {
      collectElements(DataArrayConstant->getElementAsConstant(I), Elements);
    }
  } else {
    llvm_unreachable(
        "Expected a ConstantArray or ConstantDataArray for array initializer!");
  }
}

static Constant *transformInitializer(Constant *Init, Type *OrigType,
                                      ArrayType *FlattenedType,
                                      LLVMContext &Ctx) {
  // Handle ConstantAggregateZero (zero-initialized constants)
  if (isa<ConstantAggregateZero>(Init))
    return ConstantAggregateZero::get(FlattenedType);

  // Handle UndefValue (undefined constants)
  if (isa<UndefValue>(Init))
    return UndefValue::get(FlattenedType);

  if (!isa<ArrayType>(OrigType))
    return Init;

  SmallVector<Constant *> FlattenedElements;
  collectElements(Init, FlattenedElements);
  assert(FlattenedType->getNumElements() == FlattenedElements.size() &&
         "The number of collected elements should match the FlattenedType");
  return ConstantArray::get(FlattenedType, FlattenedElements);
}

static void
flattenGlobalArrays(Module &M,
                    DenseMap<GlobalVariable *, GlobalVariable *> &GlobalMap) {
  LLVMContext &Ctx = M.getContext();
  for (GlobalVariable &G : M.globals()) {
    Type *OrigType = G.getValueType();
    if (!DXILFlattenArraysVisitor::isMultiDimensionalArray(OrigType))
      continue;

    ArrayType *ArrType = cast<ArrayType>(OrigType);
    auto [TotalElements, BaseType] =
        DXILFlattenArraysVisitor::getElementCountAndType(ArrType);
    ArrayType *FattenedArrayType = ArrayType::get(BaseType, TotalElements);

    // Create a new global variable with the updated type
    // Note: Initializer is set via transformInitializer
    GlobalVariable *NewGlobal =
        new GlobalVariable(M, FattenedArrayType, G.isConstant(), G.getLinkage(),
                           /*Initializer=*/nullptr, G.getName() + ".1dim", &G,
                           G.getThreadLocalMode(), G.getAddressSpace(),
                           G.isExternallyInitialized());

    // Copy relevant attributes
    NewGlobal->setUnnamedAddr(G.getUnnamedAddr());
    if (G.getAlignment() > 0) {
      NewGlobal->setAlignment(G.getAlign());
    }

    if (G.hasInitializer()) {
      Constant *Init = G.getInitializer();
      Constant *NewInit =
          transformInitializer(Init, OrigType, FattenedArrayType, Ctx);
      NewGlobal->setInitializer(NewInit);
    }
    GlobalMap[&G] = NewGlobal;
  }
}

static bool flattenArrays(Module &M) {
  bool MadeChange = false;
  DXILFlattenArraysVisitor Impl;
  DenseMap<GlobalVariable *, GlobalVariable *> GlobalMap;
  flattenGlobalArrays(M, GlobalMap);
  for (auto &F : make_early_inc_range(M.functions())) {
    if (F.isDeclaration())
      continue;
    MadeChange |= Impl.visit(F);
  }
  for (auto &[Old, New] : GlobalMap) {
    Old->replaceAllUsesWith(New);
    Old->eraseFromParent();
    MadeChange = true;
  }
  return MadeChange;
}

PreservedAnalyses DXILFlattenArrays::run(Module &M, ModuleAnalysisManager &) {
  bool MadeChanges = flattenArrays(M);
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  return PA;
}

bool DXILFlattenArraysLegacy::runOnModule(Module &M) {
  return flattenArrays(M);
}

char DXILFlattenArraysLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILFlattenArraysLegacy, DEBUG_TYPE,
                      "DXIL Array Flattener", false, false)
INITIALIZE_PASS_END(DXILFlattenArraysLegacy, DEBUG_TYPE, "DXIL Array Flattener",
                    false, false)

ModulePass *llvm::createDXILFlattenArraysLegacyPass() {
  return new DXILFlattenArraysLegacy();
}
