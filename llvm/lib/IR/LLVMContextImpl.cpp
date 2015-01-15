//===-- LLVMContextImpl.cpp - Implement LLVMContextImpl -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the opaque LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Module.h"
#include <algorithm>
using namespace llvm;

LLVMContextImpl::LLVMContextImpl(LLVMContext &C)
  : TheTrueVal(nullptr), TheFalseVal(nullptr),
    VoidTy(C, Type::VoidTyID),
    LabelTy(C, Type::LabelTyID),
    HalfTy(C, Type::HalfTyID),
    FloatTy(C, Type::FloatTyID),
    DoubleTy(C, Type::DoubleTyID),
    MetadataTy(C, Type::MetadataTyID),
    X86_FP80Ty(C, Type::X86_FP80TyID),
    FP128Ty(C, Type::FP128TyID),
    PPC_FP128Ty(C, Type::PPC_FP128TyID),
    X86_MMXTy(C, Type::X86_MMXTyID),
    Int1Ty(C, 1),
    Int8Ty(C, 8),
    Int16Ty(C, 16),
    Int32Ty(C, 32),
    Int64Ty(C, 64) {
  InlineAsmDiagHandler = nullptr;
  InlineAsmDiagContext = nullptr;
  DiagnosticHandler = nullptr;
  DiagnosticContext = nullptr;
  RespectDiagnosticFilters = false;
  YieldCallback = nullptr;
  YieldOpaqueHandle = nullptr;
  NamedStructTypesUniqueID = 0;
}

namespace {
struct DropReferences {
  // Takes the value_type of a ConstantUniqueMap's internal map, whose 'second'
  // is a Constant*.
  template <typename PairT> void operator()(const PairT &P) {
    P.second->dropAllReferences();
  }
};

// Temporary - drops pair.first instead of second.
struct DropFirst {
  // Takes the value_type of a ConstantUniqueMap's internal map, whose 'second'
  // is a Constant*.
  template<typename PairT>
  void operator()(const PairT &P) {
    P.first->dropAllReferences();
  }
};
}

LLVMContextImpl::~LLVMContextImpl() {
  // NOTE: We need to delete the contents of OwnedModules, but Module's dtor
  // will call LLVMContextImpl::removeModule, thus invalidating iterators into
  // the container. Avoid iterators during this operation:
  while (!OwnedModules.empty())
    delete *OwnedModules.begin();

  // Drop references for MDNodes.  Do this before Values get deleted to avoid
  // unnecessary RAUW when nodes are still unresolved.
  for (auto *I : DistinctMDNodes)
    I->dropAllReferences();
  for (auto *I : MDTuples)
    I->dropAllReferences();
  for (auto *I : MDLocations)
    I->dropAllReferences();

  // Also drop references that come from the Value bridges.
  for (auto &Pair : ValuesAsMetadata)
    Pair.second->dropUsers();
  for (auto &Pair : MetadataAsValues)
    Pair.second->dropUse();

  // Destroy MDNodes.
  for (UniquableMDNode *I : DistinctMDNodes)
    I->deleteAsSubclass();
  for (MDTuple *I : MDTuples)
    delete I;
  for (MDLocation *I : MDLocations)
    delete I;

  // Free the constants.  This is important to do here to ensure that they are
  // freed before the LeakDetector is torn down.
  std::for_each(ExprConstants.map_begin(), ExprConstants.map_end(),
                DropFirst());
  std::for_each(ArrayConstants.map_begin(), ArrayConstants.map_end(),
                DropFirst());
  std::for_each(StructConstants.map_begin(), StructConstants.map_end(),
                DropFirst());
  std::for_each(VectorConstants.map_begin(), VectorConstants.map_end(),
                DropFirst());
  ExprConstants.freeConstants();
  ArrayConstants.freeConstants();
  StructConstants.freeConstants();
  VectorConstants.freeConstants();
  DeleteContainerSeconds(CAZConstants);
  DeleteContainerSeconds(CPNConstants);
  DeleteContainerSeconds(UVConstants);
  InlineAsms.freeConstants();
  DeleteContainerSeconds(IntConstants);
  DeleteContainerSeconds(FPConstants);
  
  for (StringMap<ConstantDataSequential*>::iterator I = CDSConstants.begin(),
       E = CDSConstants.end(); I != E; ++I)
    delete I->second;
  CDSConstants.clear();

  // Destroy attributes.
  for (FoldingSetIterator<AttributeImpl> I = AttrsSet.begin(),
         E = AttrsSet.end(); I != E; ) {
    FoldingSetIterator<AttributeImpl> Elem = I++;
    delete &*Elem;
  }

  // Destroy attribute lists.
  for (FoldingSetIterator<AttributeSetImpl> I = AttrsLists.begin(),
         E = AttrsLists.end(); I != E; ) {
    FoldingSetIterator<AttributeSetImpl> Elem = I++;
    delete &*Elem;
  }

  // Destroy attribute node lists.
  for (FoldingSetIterator<AttributeSetNode> I = AttrsSetNodes.begin(),
         E = AttrsSetNodes.end(); I != E; ) {
    FoldingSetIterator<AttributeSetNode> Elem = I++;
    delete &*Elem;
  }

  // Destroy MetadataAsValues.
  {
    SmallVector<MetadataAsValue *, 8> MDVs;
    MDVs.reserve(MetadataAsValues.size());
    for (auto &Pair : MetadataAsValues)
      MDVs.push_back(Pair.second);
    MetadataAsValues.clear();
    for (auto *V : MDVs)
      delete V;
  }

  // Destroy ValuesAsMetadata.
  for (auto &Pair : ValuesAsMetadata)
    delete Pair.second;

  // Destroy MDStrings.
  MDStringCache.clear();
}

// ConstantsContext anchors
void UnaryConstantExpr::anchor() { }

void BinaryConstantExpr::anchor() { }

void SelectConstantExpr::anchor() { }

void ExtractElementConstantExpr::anchor() { }

void InsertElementConstantExpr::anchor() { }

void ShuffleVectorConstantExpr::anchor() { }

void ExtractValueConstantExpr::anchor() { }

void InsertValueConstantExpr::anchor() { }

void GetElementPtrConstantExpr::anchor() { }

void CompareConstantExpr::anchor() { }
