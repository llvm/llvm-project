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
#include "llvm/Support/MathExtras.h"
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

struct GEPInfo {
  ArrayType *RootFlattenedArrayType;
  Value *RootPointerOperand;
  SmallMapVector<Value *, APInt, 4> VariableOffsets;
  APInt ConstantOffset;
};

class DXILFlattenArraysVisitor
    : public InstVisitor<DXILFlattenArraysVisitor, bool> {
public:
  DXILFlattenArraysVisitor(
      SmallDenseMap<GlobalVariable *, GlobalVariable *> &GlobalMap)
      : GlobalMap(GlobalMap) {}
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
  SmallDenseMap<GEPOperator *, GEPInfo> GEPChainInfoMap;
  SmallDenseMap<GlobalVariable *, GlobalVariable *> &GlobalMap;
  bool finish();
  ConstantInt *genConstFlattenIndices(ArrayRef<Value *> Indices,
                                      ArrayRef<uint64_t> Dims,
                                      IRBuilder<> &Builder);
  Value *genInstructionFlattenIndices(ArrayRef<Value *> Indices,
                                      ArrayRef<uint64_t> Dims,
                                      IRBuilder<> &Builder);
};
} // namespace

bool DXILFlattenArraysVisitor::finish() {
  GEPChainInfoMap.clear();
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
      Builder.CreateAlloca(FattenedArrayType, nullptr, AI.getName() + ".1dim");
  FlatAlloca->setAlignment(AI.getAlign());
  AI.replaceAllUsesWith(FlatAlloca);
  AI.eraseFromParent();
  return true;
}

bool DXILFlattenArraysVisitor::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  // Do not visit GEPs more than once
  if (GEPChainInfoMap.contains(cast<GEPOperator>(&GEP)))
    return false;

  Value *PtrOperand = GEP.getPointerOperand();
  // It shouldn't(?) be possible for the pointer operand of a GEP to be a PHI
  // node unless HLSL has pointers. If this assumption is incorrect or HLSL gets
  // pointer types, then the handling of this case can be implemented later.
  assert(!isa<PHINode>(PtrOperand) &&
         "Pointer operand of GEP should not be a PHI Node");

  // Replace a GEP ConstantExpr pointer operand with a GEP instruction so that
  // it can be visited
  if (auto *PtrOpGEPCE = dyn_cast<ConstantExpr>(PtrOperand);
      PtrOpGEPCE && PtrOpGEPCE->getOpcode() == Instruction::GetElementPtr) {
    GetElementPtrInst *OldGEPI =
        cast<GetElementPtrInst>(PtrOpGEPCE->getAsInstruction());
    OldGEPI->insertBefore(GEP.getIterator());

    IRBuilder<> Builder(&GEP);
    SmallVector<Value *> Indices(GEP.indices());
    Value *NewGEP =
        Builder.CreateGEP(GEP.getSourceElementType(), OldGEPI, Indices,
                          GEP.getName(), GEP.getNoWrapFlags());
    assert(isa<GetElementPtrInst>(NewGEP) &&
           "Expected newly-created GEP to be an instruction");
    GetElementPtrInst *NewGEPI = cast<GetElementPtrInst>(NewGEP);

    GEP.replaceAllUsesWith(NewGEPI);
    GEP.eraseFromParent();
    visitGetElementPtrInst(*OldGEPI);
    visitGetElementPtrInst(*NewGEPI);
    return true;
  }

  // Construct GEPInfo for this GEP
  GEPInfo Info;

  // Obtain the variable and constant byte offsets computed by this GEP
  const DataLayout &DL = GEP.getDataLayout();
  unsigned BitWidth = DL.getIndexTypeSizeInBits(GEP.getType());
  Info.ConstantOffset = {BitWidth, 0};
  [[maybe_unused]] bool Success = GEP.collectOffset(
      DL, BitWidth, Info.VariableOffsets, Info.ConstantOffset);
  assert(Success && "Failed to collect offsets for GEP");

  // If there is a parent GEP, inherit the root array type and pointer, and
  // merge the byte offsets. Otherwise, this GEP is itself the root of a GEP
  // chain and we need to deterine the root array type
  if (auto *PtrOpGEP = dyn_cast<GEPOperator>(PtrOperand)) {

    // If the parent GEP was not processed, then we do not want to process its
    // descendants. This can happen if the GEP chain is for an unsupported type
    // such as a struct -- we do not flatten structs nor GEP chains for structs
    if (!GEPChainInfoMap.contains(PtrOpGEP))
      return false;

    GEPInfo &PGEPInfo = GEPChainInfoMap[PtrOpGEP];
    Info.RootFlattenedArrayType = PGEPInfo.RootFlattenedArrayType;
    Info.RootPointerOperand = PGEPInfo.RootPointerOperand;
    for (auto &VariableOffset : PGEPInfo.VariableOffsets)
      Info.VariableOffsets.insert(VariableOffset);
    Info.ConstantOffset += PGEPInfo.ConstantOffset;
  } else {
    Info.RootPointerOperand = PtrOperand;

    // We should try to determine the type of the root from the pointer rather
    // than the GEP's source element type because this could be a scalar GEP
    // into an array-typed pointer from an Alloca or Global Variable.
    Type *RootTy = GEP.getSourceElementType();
    if (auto *GlobalVar = dyn_cast<GlobalVariable>(PtrOperand)) {
      if (GlobalMap.contains(GlobalVar))
        GlobalVar = GlobalMap[GlobalVar];
      Info.RootPointerOperand = GlobalVar;
      RootTy = GlobalVar->getValueType();
    } else if (auto *Alloca = dyn_cast<AllocaInst>(PtrOperand))
      RootTy = Alloca->getAllocatedType();
    assert(!isMultiDimensionalArray(RootTy) &&
           "Expected root array type to be flattened");

    // If the root type is not an array, we don't need to do any flattening
    if (!isa<ArrayType>(RootTy))
      return false;

    Info.RootFlattenedArrayType = cast<ArrayType>(RootTy);
  }

  // GEPs without users or GEPs with non-GEP users should be replaced such that
  // the chain of GEPs they are a part of are collapsed to a single GEP into a
  // flattened array.
  bool ReplaceThisGEP = GEP.users().empty();
  for (Value *User : GEP.users())
    if (!isa<GetElementPtrInst>(User))
      ReplaceThisGEP = true;

  if (ReplaceThisGEP) {
    unsigned BytesPerElem =
        DL.getTypeAllocSize(Info.RootFlattenedArrayType->getArrayElementType());
    assert(isPowerOf2_32(BytesPerElem) &&
           "Bytes per element should be a power of 2");

    // Compute the 32-bit index for this flattened GEP from the constant and
    // variable byte offsets in the GEPInfo
    IRBuilder<> Builder(&GEP);
    Value *ZeroIndex = Builder.getInt32(0);
    uint64_t ConstantOffset =
        Info.ConstantOffset.udiv(BytesPerElem).getZExtValue();
    assert(ConstantOffset < UINT32_MAX &&
           "Constant byte offset for flat GEP index must fit within 32 bits");
    Value *FlattenedIndex = Builder.getInt32(ConstantOffset);
    for (auto [VarIndex, Multiplier] : Info.VariableOffsets) {
      assert(Multiplier.getActiveBits() <= 32 &&
             "The multiplier for a flat GEP index must fit within 32 bits");
      assert(VarIndex->getType()->isIntegerTy(32) &&
             "Expected i32-typed GEP indices");
      Value *VI;
      if (Multiplier.getZExtValue() % BytesPerElem != 0) {
        // This can happen, e.g., with i8 GEPs. To handle this we just divide
        // by BytesPerElem using an instruction after multiplying VarIndex by
        // Multiplier.
        VI = Builder.CreateMul(VarIndex,
                               Builder.getInt32(Multiplier.getZExtValue()));
        VI = Builder.CreateLShr(VI, Builder.getInt32(Log2_32(BytesPerElem)));
      } else
        VI = Builder.CreateMul(
            VarIndex,
            Builder.getInt32(Multiplier.getZExtValue() / BytesPerElem));
      FlattenedIndex = Builder.CreateAdd(FlattenedIndex, VI);
    }

    // Construct a new GEP for the flattened array to replace the current GEP
    Value *NewGEP = Builder.CreateGEP(
        Info.RootFlattenedArrayType, Info.RootPointerOperand,
        {ZeroIndex, FlattenedIndex}, GEP.getName(), GEP.getNoWrapFlags());

    // If the pointer operand is a global variable and all indices are 0,
    // IRBuilder::CreateGEP will return the global variable instead of creating
    // a GEP instruction or GEP ConstantExpr. In this case we have to create and
    // insert our own GEP instruction.
    if (!isa<GEPOperator>(NewGEP))
      NewGEP = GetElementPtrInst::Create(
          Info.RootFlattenedArrayType, Info.RootPointerOperand,
          {ZeroIndex, FlattenedIndex}, GEP.getNoWrapFlags(), GEP.getName(),
          Builder.GetInsertPoint());

    // Replace the current GEP with the new GEP. Store GEPInfo into the map
    // for later use in case this GEP was not the end of the chain
    GEPChainInfoMap.insert({cast<GEPOperator>(NewGEP), std::move(Info)});
    GEP.replaceAllUsesWith(NewGEP);
    GEP.eraseFromParent();
    return true;
  }

  // This GEP is potentially dead at the end of the pass since it may not have
  // any users anymore after GEP chains have been collapsed. We retain store
  // GEPInfo for GEPs down the chain to use to compute their indices.
  GEPChainInfoMap.insert({cast<GEPOperator>(&GEP), std::move(Info)});
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

static void flattenGlobalArrays(
    Module &M, SmallDenseMap<GlobalVariable *, GlobalVariable *> &GlobalMap) {
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
  SmallDenseMap<GlobalVariable *, GlobalVariable *> GlobalMap;
  flattenGlobalArrays(M, GlobalMap);
  DXILFlattenArraysVisitor Impl(GlobalMap);
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
