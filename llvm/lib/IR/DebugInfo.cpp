//===- DebugInfo.cpp - Debug Information Helper Classes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the helper classes used to build and interpret debug
// information in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/DebugInfo.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <optional>
#include <utility>

using namespace llvm;
using namespace llvm::at;
using namespace llvm::dwarf;

TinyPtrVector<DbgDeclareInst *> llvm::FindDbgDeclareUses(Value *V) {
  // This function is hot. Check whether the value has any metadata to avoid a
  // DenseMap lookup.
  if (!V->isUsedByMetadata())
    return {};
  auto *L = LocalAsMetadata::getIfExists(V);
  if (!L)
    return {};
  auto *MDV = MetadataAsValue::getIfExists(V->getContext(), L);
  if (!MDV)
    return {};

  TinyPtrVector<DbgDeclareInst *> Declares;
  for (User *U : MDV->users()) {
    if (auto *DDI = dyn_cast<DbgDeclareInst>(U))
      Declares.push_back(DDI);
  }

  return Declares;
}

void llvm::findDbgValues(SmallVectorImpl<DbgValueInst *> &DbgValues, Value *V) {
  // This function is hot. Check whether the value has any metadata to avoid a
  // DenseMap lookup.
  if (!V->isUsedByMetadata())
    return;
  // TODO: If this value appears multiple times in a DIArgList, we should still
  // only add the owning DbgValueInst once; use this set to track ArgListUsers.
  // This behaviour can be removed when we can automatically remove duplicates.
  SmallPtrSet<DbgValueInst *, 4> EncounteredDbgValues;
  if (auto *L = LocalAsMetadata::getIfExists(V)) {
    if (auto *MDV = MetadataAsValue::getIfExists(V->getContext(), L)) {
      for (User *U : MDV->users())
        if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(U))
          DbgValues.push_back(DVI);
    }
    for (Metadata *AL : L->getAllArgListUsers()) {
      if (auto *MDV = MetadataAsValue::getIfExists(V->getContext(), AL)) {
        for (User *U : MDV->users())
          if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(U))
            if (EncounteredDbgValues.insert(DVI).second)
              DbgValues.push_back(DVI);
      }
    }
  }
}

void llvm::findDbgUsers(SmallVectorImpl<DbgVariableIntrinsic *> &DbgUsers,
                        Value *V) {
  // This function is hot. Check whether the value has any metadata to avoid a
  // DenseMap lookup.
  if (!V->isUsedByMetadata())
    return;
  // TODO: If this value appears multiple times in a DIArgList, we should still
  // only add the owning DbgValueInst once; use this set to track ArgListUsers.
  // This behaviour can be removed when we can automatically remove duplicates.
  SmallPtrSet<DbgVariableIntrinsic *, 4> EncounteredDbgValues;
  if (auto *L = LocalAsMetadata::getIfExists(V)) {
    if (auto *MDV = MetadataAsValue::getIfExists(V->getContext(), L)) {
      for (User *U : MDV->users())
        if (DbgVariableIntrinsic *DII = dyn_cast<DbgVariableIntrinsic>(U))
          DbgUsers.push_back(DII);
    }
    for (Metadata *AL : L->getAllArgListUsers()) {
      if (auto *MDV = MetadataAsValue::getIfExists(V->getContext(), AL)) {
        for (User *U : MDV->users())
          if (DbgVariableIntrinsic *DII = dyn_cast<DbgVariableIntrinsic>(U))
            if (EncounteredDbgValues.insert(DII).second)
              DbgUsers.push_back(DII);
      }
    }
  }
}

DISubprogram *llvm::getDISubprogram(const MDNode *Scope) {
  if (auto *LocalScope = dyn_cast_or_null<DILocalScope>(Scope))
    return LocalScope->getSubprogram();
  return nullptr;
}

DebugLoc llvm::getDebugValueLoc(DbgVariableIntrinsic *DII) {
  // Original dbg.declare must have a location.
  const DebugLoc &DeclareLoc = DII->getDebugLoc();
  MDNode *Scope = DeclareLoc.getScope();
  DILocation *InlinedAt = DeclareLoc.getInlinedAt();
  // Because no machine insts can come from debug intrinsics, only the scope
  // and inlinedAt is significant. Zero line numbers are used in case this
  // DebugLoc leaks into any adjacent instructions. Produce an unknown location
  // with the correct scope / inlinedAt fields.
  return DILocation::get(DII->getContext(), 0, 0, Scope, InlinedAt);
}

//===----------------------------------------------------------------------===//
// DebugInfoFinder implementations.
//===----------------------------------------------------------------------===//

void DebugInfoFinder::reset() {
  CUs.clear();
  SPs.clear();
  GVs.clear();
  TYs.clear();
  Scopes.clear();
  NodesSeen.clear();
}

void DebugInfoFinder::processModule(const Module &M) {
  for (auto *CU : M.debug_compile_units())
    processCompileUnit(CU);
  for (auto &F : M.functions()) {
    if (auto *SP = cast_or_null<DISubprogram>(F.getSubprogram()))
      processSubprogram(SP);
    // There could be subprograms from inlined functions referenced from
    // instructions only. Walk the function to find them.
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB)
        processInstruction(M, I);
  }
}

void DebugInfoFinder::processCompileUnit(DICompileUnit *CU) {
  if (!addCompileUnit(CU))
    return;
  for (auto *DIG : CU->getGlobalVariables()) {
    if (!addGlobalVariable(DIG))
      continue;
    auto *GV = DIG->getVariable();
    processScope(GV->getScope());
    processType(GV->getType());
  }
  for (auto *ET : CU->getEnumTypes())
    processType(ET);
  for (auto *RT : CU->getRetainedTypes())
    if (auto *T = dyn_cast<DIType>(RT))
      processType(T);
    else
      processSubprogram(cast<DISubprogram>(RT));
  for (auto *Import : CU->getImportedEntities()) {
    auto *Entity = Import->getEntity();
    if (auto *T = dyn_cast<DIType>(Entity))
      processType(T);
    else if (auto *SP = dyn_cast<DISubprogram>(Entity))
      processSubprogram(SP);
    else if (auto *NS = dyn_cast<DINamespace>(Entity))
      processScope(NS->getScope());
    else if (auto *M = dyn_cast<DIModule>(Entity))
      processScope(M->getScope());
  }
}

void DebugInfoFinder::processInstruction(const Module &M,
                                         const Instruction &I) {
  if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(&I))
    processVariable(M, *DVI);

  if (auto DbgLoc = I.getDebugLoc())
    processLocation(M, DbgLoc.get());
}

void DebugInfoFinder::processLocation(const Module &M, const DILocation *Loc) {
  if (!Loc)
    return;
  processScope(Loc->getScope());
  processLocation(M, Loc->getInlinedAt());
}

void DebugInfoFinder::processType(DIType *DT) {
  if (!addType(DT))
    return;
  processScope(DT->getScope());
  if (auto *ST = dyn_cast<DISubroutineType>(DT)) {
    for (DIType *Ref : ST->getTypeArray())
      processType(Ref);
    return;
  }
  if (auto *DCT = dyn_cast<DICompositeType>(DT)) {
    processType(DCT->getBaseType());
    for (Metadata *D : DCT->getElements()) {
      if (auto *T = dyn_cast<DIType>(D))
        processType(T);
      else if (auto *SP = dyn_cast<DISubprogram>(D))
        processSubprogram(SP);
    }
    return;
  }
  if (auto *DDT = dyn_cast<DIDerivedType>(DT)) {
    processType(DDT->getBaseType());
  }
}

void DebugInfoFinder::processScope(DIScope *Scope) {
  if (!Scope)
    return;
  if (auto *Ty = dyn_cast<DIType>(Scope)) {
    processType(Ty);
    return;
  }
  if (auto *CU = dyn_cast<DICompileUnit>(Scope)) {
    addCompileUnit(CU);
    return;
  }
  if (auto *SP = dyn_cast<DISubprogram>(Scope)) {
    processSubprogram(SP);
    return;
  }
  if (!addScope(Scope))
    return;
  if (auto *LB = dyn_cast<DILexicalBlockBase>(Scope)) {
    processScope(LB->getScope());
  } else if (auto *NS = dyn_cast<DINamespace>(Scope)) {
    processScope(NS->getScope());
  } else if (auto *M = dyn_cast<DIModule>(Scope)) {
    processScope(M->getScope());
  }
}

void DebugInfoFinder::processSubprogram(DISubprogram *SP) {
  if (!addSubprogram(SP))
    return;
  processScope(SP->getScope());
  // Some of the users, e.g. CloneFunctionInto / CloneModule, need to set up a
  // ValueMap containing identity mappings for all of the DICompileUnit's, not
  // just DISubprogram's, referenced from anywhere within the Function being
  // cloned prior to calling MapMetadata / RemapInstruction to avoid their
  // duplication later as DICompileUnit's are also directly referenced by
  // llvm.dbg.cu list. Thefore we need to collect DICompileUnit's here as well.
  // Also, DICompileUnit's may reference DISubprogram's too and therefore need
  // to be at least looked through.
  processCompileUnit(SP->getUnit());
  processType(SP->getType());
  for (auto *Element : SP->getTemplateParams()) {
    if (auto *TType = dyn_cast<DITemplateTypeParameter>(Element)) {
      processType(TType->getType());
    } else if (auto *TVal = dyn_cast<DITemplateValueParameter>(Element)) {
      processType(TVal->getType());
    }
  }
}

void DebugInfoFinder::processVariable(const Module &M,
                                      const DbgVariableIntrinsic &DVI) {
  auto *N = dyn_cast<MDNode>(DVI.getVariable());
  if (!N)
    return;

  auto *DV = dyn_cast<DILocalVariable>(N);
  if (!DV)
    return;

  if (!NodesSeen.insert(DV).second)
    return;
  processScope(DV->getScope());
  processType(DV->getType());
}

bool DebugInfoFinder::addType(DIType *DT) {
  if (!DT)
    return false;

  if (!NodesSeen.insert(DT).second)
    return false;

  TYs.push_back(const_cast<DIType *>(DT));
  return true;
}

bool DebugInfoFinder::addCompileUnit(DICompileUnit *CU) {
  if (!CU)
    return false;
  if (!NodesSeen.insert(CU).second)
    return false;

  CUs.push_back(CU);
  return true;
}

bool DebugInfoFinder::addGlobalVariable(DIGlobalVariableExpression *DIG) {
  if (!NodesSeen.insert(DIG).second)
    return false;

  GVs.push_back(DIG);
  return true;
}

bool DebugInfoFinder::addSubprogram(DISubprogram *SP) {
  if (!SP)
    return false;

  if (!NodesSeen.insert(SP).second)
    return false;

  SPs.push_back(SP);
  return true;
}

bool DebugInfoFinder::addScope(DIScope *Scope) {
  if (!Scope)
    return false;
  // FIXME: Ocaml binding generates a scope with no content, we treat it
  // as null for now.
  if (Scope->getNumOperands() == 0)
    return false;
  if (!NodesSeen.insert(Scope).second)
    return false;
  Scopes.push_back(Scope);
  return true;
}

static MDNode *updateLoopMetadataDebugLocationsImpl(
    MDNode *OrigLoopID, function_ref<Metadata *(Metadata *)> Updater) {
  assert(OrigLoopID && OrigLoopID->getNumOperands() > 0 &&
         "Loop ID needs at least one operand");
  assert(OrigLoopID && OrigLoopID->getOperand(0).get() == OrigLoopID &&
         "Loop ID should refer to itself");

  // Save space for the self-referential LoopID.
  SmallVector<Metadata *, 4> MDs = {nullptr};

  for (unsigned i = 1; i < OrigLoopID->getNumOperands(); ++i) {
    Metadata *MD = OrigLoopID->getOperand(i);
    if (!MD)
      MDs.push_back(nullptr);
    else if (Metadata *NewMD = Updater(MD))
      MDs.push_back(NewMD);
  }

  MDNode *NewLoopID = MDNode::getDistinct(OrigLoopID->getContext(), MDs);
  // Insert the self-referential LoopID.
  NewLoopID->replaceOperandWith(0, NewLoopID);
  return NewLoopID;
}

void llvm::updateLoopMetadataDebugLocations(
    Instruction &I, function_ref<Metadata *(Metadata *)> Updater) {
  MDNode *OrigLoopID = I.getMetadata(LLVMContext::MD_loop);
  if (!OrigLoopID)
    return;
  MDNode *NewLoopID = updateLoopMetadataDebugLocationsImpl(OrigLoopID, Updater);
  I.setMetadata(LLVMContext::MD_loop, NewLoopID);
}

/// Return true if a node is a DILocation or if a DILocation is
/// indirectly referenced by one of the node's children.
static bool isDILocationReachable(SmallPtrSetImpl<Metadata *> &Visited,
                                  SmallPtrSetImpl<Metadata *> &Reachable,
                                  Metadata *MD) {
  MDNode *N = dyn_cast_or_null<MDNode>(MD);
  if (!N)
    return false;
  if (isa<DILocation>(N) || Reachable.count(N))
    return true;
  if (!Visited.insert(N).second)
    return false;
  for (auto &OpIt : N->operands()) {
    Metadata *Op = OpIt.get();
    if (isDILocationReachable(Visited, Reachable, Op)) {
      // Don't return just yet as we want to visit all MD's children to
      // initialize DILocationReachable in stripDebugLocFromLoopID
      Reachable.insert(N);
    }
  }
  return Reachable.count(N);
}

static bool isAllDILocation(SmallPtrSetImpl<Metadata *> &Visited,
                            SmallPtrSetImpl<Metadata *> &AllDILocation,
                            const SmallPtrSetImpl<Metadata *> &DIReachable,
                            Metadata *MD) {
  MDNode *N = dyn_cast_or_null<MDNode>(MD);
  if (!N)
    return false;
  if (isa<DILocation>(N) || AllDILocation.count(N))
    return true;
  if (!DIReachable.count(N))
    return false;
  if (!Visited.insert(N).second)
    return false;
  for (auto &OpIt : N->operands()) {
    Metadata *Op = OpIt.get();
    if (Op == MD)
      continue;
    if (!isAllDILocation(Visited, AllDILocation, DIReachable, Op)) {
      return false;
    }
  }
  AllDILocation.insert(N);
  return true;
}

static Metadata *
stripLoopMDLoc(const SmallPtrSetImpl<Metadata *> &AllDILocation,
               const SmallPtrSetImpl<Metadata *> &DIReachable, Metadata *MD) {
  if (isa<DILocation>(MD) || AllDILocation.count(MD))
    return nullptr;

  if (!DIReachable.count(MD))
    return MD;

  MDNode *N = dyn_cast_or_null<MDNode>(MD);
  if (!N)
    return MD;

  SmallVector<Metadata *, 4> Args;
  bool HasSelfRef = false;
  for (unsigned i = 0; i < N->getNumOperands(); ++i) {
    Metadata *A = N->getOperand(i);
    if (!A) {
      Args.push_back(nullptr);
    } else if (A == MD) {
      assert(i == 0 && "expected i==0 for self-reference");
      HasSelfRef = true;
      Args.push_back(nullptr);
    } else if (Metadata *NewArg =
                   stripLoopMDLoc(AllDILocation, DIReachable, A)) {
      Args.push_back(NewArg);
    }
  }
  if (Args.empty() || (HasSelfRef && Args.size() == 1))
    return nullptr;

  MDNode *NewMD = N->isDistinct() ? MDNode::getDistinct(N->getContext(), Args)
                                  : MDNode::get(N->getContext(), Args);
  if (HasSelfRef)
    NewMD->replaceOperandWith(0, NewMD);
  return NewMD;
}

static MDNode *stripDebugLocFromLoopID(MDNode *N) {
  assert(!N->operands().empty() && "Missing self reference?");
  SmallPtrSet<Metadata *, 8> Visited, DILocationReachable, AllDILocation;
  // If we already visited N, there is nothing to do.
  if (!Visited.insert(N).second)
    return N;

  // If there is no debug location, we do not have to rewrite this
  // MDNode. This loop also initializes DILocationReachable, later
  // needed by updateLoopMetadataDebugLocationsImpl; the use of
  // count_if avoids an early exit.
  if (!llvm::count_if(llvm::drop_begin(N->operands()),
                     [&Visited, &DILocationReachable](const MDOperand &Op) {
                       return isDILocationReachable(
                                  Visited, DILocationReachable, Op.get());
                     }))
    return N;

  Visited.clear();
  // If there is only the debug location without any actual loop metadata, we
  // can remove the metadata.
  if (llvm::all_of(llvm::drop_begin(N->operands()),
                   [&Visited, &AllDILocation,
                    &DILocationReachable](const MDOperand &Op) {
                     return isAllDILocation(Visited, AllDILocation,
                                            DILocationReachable, Op.get());
                   }))
    return nullptr;

  return updateLoopMetadataDebugLocationsImpl(
      N, [&AllDILocation, &DILocationReachable](Metadata *MD) -> Metadata * {
        return stripLoopMDLoc(AllDILocation, DILocationReachable, MD);
      });
}

bool llvm::stripDebugInfo(Function &F) {
  bool Changed = false;
  if (F.hasMetadata(LLVMContext::MD_dbg)) {
    Changed = true;
    F.setSubprogram(nullptr);
  }

  DenseMap<MDNode *, MDNode *> LoopIDsMap;
  for (BasicBlock &BB : F) {
    for (Instruction &I : llvm::make_early_inc_range(BB)) {
      if (isa<DbgInfoIntrinsic>(&I)) {
        I.eraseFromParent();
        Changed = true;
        continue;
      }
      if (I.getDebugLoc()) {
        Changed = true;
        I.setDebugLoc(DebugLoc());
      }
      if (auto *LoopID = I.getMetadata(LLVMContext::MD_loop)) {
        auto *NewLoopID = LoopIDsMap.lookup(LoopID);
        if (!NewLoopID)
          NewLoopID = LoopIDsMap[LoopID] = stripDebugLocFromLoopID(LoopID);
        if (NewLoopID != LoopID)
          I.setMetadata(LLVMContext::MD_loop, NewLoopID);
      }
      // Strip other attachments that are or use debug info.
      if (I.hasMetadataOtherThanDebugLoc()) {
        // Heapallocsites point into the DIType system.
        I.setMetadata("heapallocsite", nullptr);
        // DIAssignID are debug info metadata primitives.
        I.setMetadata(LLVMContext::MD_DIAssignID, nullptr);
      }
    }
  }
  return Changed;
}

bool llvm::StripDebugInfo(Module &M) {
  bool Changed = false;

  for (NamedMDNode &NMD : llvm::make_early_inc_range(M.named_metadata())) {
    // We're stripping debug info, and without them, coverage information
    // doesn't quite make sense.
    if (NMD.getName().startswith("llvm.dbg.") ||
        NMD.getName() == "llvm.gcov") {
      NMD.eraseFromParent();
      Changed = true;
    }
  }

  for (Function &F : M)
    Changed |= stripDebugInfo(F);

  for (auto &GV : M.globals()) {
    Changed |= GV.eraseMetadata(LLVMContext::MD_dbg);
  }

  if (GVMaterializer *Materializer = M.getMaterializer())
    Materializer->setStripDebugInfo();

  return Changed;
}

namespace {

/// Helper class to downgrade -g metadata to -gline-tables-only metadata.
class DebugTypeInfoRemoval {
  DenseMap<Metadata *, Metadata *> Replacements;

public:
  /// The (void)() type.
  MDNode *EmptySubroutineType;

private:
  /// Remember what linkage name we originally had before stripping. If we end
  /// up making two subprograms identical who originally had different linkage
  /// names, then we need to make one of them distinct, to avoid them getting
  /// uniqued. Maps the new node to the old linkage name.
  DenseMap<DISubprogram *, StringRef> NewToLinkageName;

  // TODO: Remember the distinct subprogram we created for a given linkage name,
  // so that we can continue to unique whenever possible. Map <newly created
  // node, old linkage name> to the first (possibly distinct) mdsubprogram
  // created for that combination. This is not strictly needed for correctness,
  // but can cut down on the number of MDNodes and let us diff cleanly with the
  // output of -gline-tables-only.

public:
  DebugTypeInfoRemoval(LLVMContext &C)
      : EmptySubroutineType(DISubroutineType::get(C, DINode::FlagZero, 0,
                                                  MDNode::get(C, {}))) {}

  Metadata *map(Metadata *M) {
    if (!M)
      return nullptr;
    auto Replacement = Replacements.find(M);
    if (Replacement != Replacements.end())
      return Replacement->second;

    return M;
  }
  MDNode *mapNode(Metadata *N) { return dyn_cast_or_null<MDNode>(map(N)); }

  /// Recursively remap N and all its referenced children. Does a DF post-order
  /// traversal, so as to remap bottoms up.
  void traverseAndRemap(MDNode *N) { traverse(N); }

private:
  // Create a new DISubprogram, to replace the one given.
  DISubprogram *getReplacementSubprogram(DISubprogram *MDS) {
    auto *FileAndScope = cast_or_null<DIFile>(map(MDS->getFile()));
    StringRef LinkageName = MDS->getName().empty() ? MDS->getLinkageName() : "";
    DISubprogram *Declaration = nullptr;
    auto *Type = cast_or_null<DISubroutineType>(map(MDS->getType()));
    DIType *ContainingType =
        cast_or_null<DIType>(map(MDS->getContainingType()));
    auto *Unit = cast_or_null<DICompileUnit>(map(MDS->getUnit()));
    auto Variables = nullptr;
    auto TemplateParams = nullptr;

    // Make a distinct DISubprogram, for situations that warrent it.
    auto distinctMDSubprogram = [&]() {
      return DISubprogram::getDistinct(
          MDS->getContext(), FileAndScope, MDS->getName(), LinkageName,
          FileAndScope, MDS->getLine(), Type, MDS->getScopeLine(),
          ContainingType, MDS->getVirtualIndex(), MDS->getThisAdjustment(),
          MDS->getFlags(), MDS->getSPFlags(), Unit, TemplateParams, Declaration,
          Variables);
    };

    if (MDS->isDistinct())
      return distinctMDSubprogram();

    auto *NewMDS = DISubprogram::get(
        MDS->getContext(), FileAndScope, MDS->getName(), LinkageName,
        FileAndScope, MDS->getLine(), Type, MDS->getScopeLine(), ContainingType,
        MDS->getVirtualIndex(), MDS->getThisAdjustment(), MDS->getFlags(),
        MDS->getSPFlags(), Unit, TemplateParams, Declaration, Variables);

    StringRef OldLinkageName = MDS->getLinkageName();

    // See if we need to make a distinct one.
    auto OrigLinkage = NewToLinkageName.find(NewMDS);
    if (OrigLinkage != NewToLinkageName.end()) {
      if (OrigLinkage->second == OldLinkageName)
        // We're good.
        return NewMDS;

      // Otherwise, need to make a distinct one.
      // TODO: Query the map to see if we already have one.
      return distinctMDSubprogram();
    }

    NewToLinkageName.insert({NewMDS, MDS->getLinkageName()});
    return NewMDS;
  }

  /// Create a new compile unit, to replace the one given
  DICompileUnit *getReplacementCU(DICompileUnit *CU) {
    // Drop skeleton CUs.
    if (CU->getDWOId())
      return nullptr;

    auto *File = cast_or_null<DIFile>(map(CU->getFile()));
    MDTuple *EnumTypes = nullptr;
    MDTuple *RetainedTypes = nullptr;
    MDTuple *GlobalVariables = nullptr;
    MDTuple *ImportedEntities = nullptr;
    return DICompileUnit::getDistinct(
        CU->getContext(), CU->getSourceLanguage(), File, CU->getProducer(),
        CU->isOptimized(), CU->getFlags(), CU->getRuntimeVersion(),
        CU->getSplitDebugFilename(), DICompileUnit::LineTablesOnly, EnumTypes,
        RetainedTypes, GlobalVariables, ImportedEntities, CU->getMacros(),
        CU->getDWOId(), CU->getSplitDebugInlining(),
        CU->getDebugInfoForProfiling(), CU->getNameTableKind(),
        CU->getRangesBaseAddress(), CU->getSysRoot(), CU->getSDK());
  }

  DILocation *getReplacementMDLocation(DILocation *MLD) {
    auto *Scope = map(MLD->getScope());
    auto *InlinedAt = map(MLD->getInlinedAt());
    if (MLD->isDistinct())
      return DILocation::getDistinct(MLD->getContext(), MLD->getLine(),
                                     MLD->getColumn(), Scope, InlinedAt);
    return DILocation::get(MLD->getContext(), MLD->getLine(), MLD->getColumn(),
                           Scope, InlinedAt);
  }

  /// Create a new generic MDNode, to replace the one given
  MDNode *getReplacementMDNode(MDNode *N) {
    SmallVector<Metadata *, 8> Ops;
    Ops.reserve(N->getNumOperands());
    for (auto &I : N->operands())
      if (I)
        Ops.push_back(map(I));
    auto *Ret = MDNode::get(N->getContext(), Ops);
    return Ret;
  }

  /// Attempt to re-map N to a newly created node.
  void remap(MDNode *N) {
    if (Replacements.count(N))
      return;

    auto doRemap = [&](MDNode *N) -> MDNode * {
      if (!N)
        return nullptr;
      if (auto *MDSub = dyn_cast<DISubprogram>(N)) {
        remap(MDSub->getUnit());
        return getReplacementSubprogram(MDSub);
      }
      if (isa<DISubroutineType>(N))
        return EmptySubroutineType;
      if (auto *CU = dyn_cast<DICompileUnit>(N))
        return getReplacementCU(CU);
      if (isa<DIFile>(N))
        return N;
      if (auto *MDLB = dyn_cast<DILexicalBlockBase>(N))
        // Remap to our referenced scope (recursively).
        return mapNode(MDLB->getScope());
      if (auto *MLD = dyn_cast<DILocation>(N))
        return getReplacementMDLocation(MLD);

      // Otherwise, if we see these, just drop them now. Not strictly necessary,
      // but this speeds things up a little.
      if (isa<DINode>(N))
        return nullptr;

      return getReplacementMDNode(N);
    };
    Replacements[N] = doRemap(N);
  }

  /// Do the remapping traversal.
  void traverse(MDNode *);
};

} // end anonymous namespace

void DebugTypeInfoRemoval::traverse(MDNode *N) {
  if (!N || Replacements.count(N))
    return;

  // To avoid cycles, as well as for efficiency sake, we will sometimes prune
  // parts of the graph.
  auto prune = [](MDNode *Parent, MDNode *Child) {
    if (auto *MDS = dyn_cast<DISubprogram>(Parent))
      return Child == MDS->getRetainedNodes().get();
    return false;
  };

  SmallVector<MDNode *, 16> ToVisit;
  DenseSet<MDNode *> Opened;

  // Visit each node starting at N in post order, and map them.
  ToVisit.push_back(N);
  while (!ToVisit.empty()) {
    auto *N = ToVisit.back();
    if (!Opened.insert(N).second) {
      // Close it.
      remap(N);
      ToVisit.pop_back();
      continue;
    }
    for (auto &I : N->operands())
      if (auto *MDN = dyn_cast_or_null<MDNode>(I))
        if (!Opened.count(MDN) && !Replacements.count(MDN) && !prune(N, MDN) &&
            !isa<DICompileUnit>(MDN))
          ToVisit.push_back(MDN);
  }
}

bool llvm::stripNonLineTableDebugInfo(Module &M) {
  bool Changed = false;

  // First off, delete the debug intrinsics.
  auto RemoveUses = [&](StringRef Name) {
    if (auto *DbgVal = M.getFunction(Name)) {
      while (!DbgVal->use_empty())
        cast<Instruction>(DbgVal->user_back())->eraseFromParent();
      DbgVal->eraseFromParent();
      Changed = true;
    }
  };
  RemoveUses("llvm.dbg.declare");
  RemoveUses("llvm.dbg.label");
  RemoveUses("llvm.dbg.value");

  // Delete non-CU debug info named metadata nodes.
  for (auto NMI = M.named_metadata_begin(), NME = M.named_metadata_end();
       NMI != NME;) {
    NamedMDNode *NMD = &*NMI;
    ++NMI;
    // Specifically keep dbg.cu around.
    if (NMD->getName() == "llvm.dbg.cu")
      continue;
  }

  // Drop all dbg attachments from global variables.
  for (auto &GV : M.globals())
    GV.eraseMetadata(LLVMContext::MD_dbg);

  DebugTypeInfoRemoval Mapper(M.getContext());
  auto remap = [&](MDNode *Node) -> MDNode * {
    if (!Node)
      return nullptr;
    Mapper.traverseAndRemap(Node);
    auto *NewNode = Mapper.mapNode(Node);
    Changed |= Node != NewNode;
    Node = NewNode;
    return NewNode;
  };

  // Rewrite the DebugLocs to be equivalent to what
  // -gline-tables-only would have created.
  for (auto &F : M) {
    if (auto *SP = F.getSubprogram()) {
      Mapper.traverseAndRemap(SP);
      auto *NewSP = cast<DISubprogram>(Mapper.mapNode(SP));
      Changed |= SP != NewSP;
      F.setSubprogram(NewSP);
    }
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto remapDebugLoc = [&](const DebugLoc &DL) -> DebugLoc {
          auto *Scope = DL.getScope();
          MDNode *InlinedAt = DL.getInlinedAt();
          Scope = remap(Scope);
          InlinedAt = remap(InlinedAt);
          return DILocation::get(M.getContext(), DL.getLine(), DL.getCol(),
                                 Scope, InlinedAt);
        };

        if (I.getDebugLoc() != DebugLoc())
          I.setDebugLoc(remapDebugLoc(I.getDebugLoc()));

        // Remap DILocations in llvm.loop attachments.
        updateLoopMetadataDebugLocations(I, [&](Metadata *MD) -> Metadata * {
          if (auto *Loc = dyn_cast_or_null<DILocation>(MD))
            return remapDebugLoc(Loc).get();
          return MD;
        });

        // Strip heapallocsite attachments, they point into the DIType system.
        if (I.hasMetadataOtherThanDebugLoc())
          I.setMetadata("heapallocsite", nullptr);
      }
    }
  }

  // Create a new llvm.dbg.cu, which is equivalent to the one
  // -gline-tables-only would have created.
  for (auto &NMD : M.named_metadata()) {
    SmallVector<MDNode *, 8> Ops;
    for (MDNode *Op : NMD.operands())
      Ops.push_back(remap(Op));

    if (!Changed)
      continue;

    NMD.clearOperands();
    for (auto *Op : Ops)
      if (Op)
        NMD.addOperand(Op);
  }
  return Changed;
}

unsigned llvm::getDebugMetadataVersionFromModule(const Module &M) {
  if (auto *Val = mdconst::dyn_extract_or_null<ConstantInt>(
          M.getModuleFlag("Debug Info Version")))
    return Val->getZExtValue();
  return 0;
}

void Instruction::applyMergedLocation(const DILocation *LocA,
                                      const DILocation *LocB) {
  setDebugLoc(DILocation::getMergedLocation(LocA, LocB));
}

void Instruction::mergeDIAssignID(
    ArrayRef<const Instruction *> SourceInstructions) {
  // Replace all uses (and attachments) of all the DIAssignIDs
  // on SourceInstructions with a single merged value.
  assert(getFunction() && "Uninserted instruction merged");
  // Collect up the DIAssignID tags.
  SmallVector<DIAssignID *, 4> IDs;
  for (const Instruction *I : SourceInstructions) {
    if (auto *MD = I->getMetadata(LLVMContext::MD_DIAssignID))
      IDs.push_back(cast<DIAssignID>(MD));
    assert(getFunction() == I->getFunction() &&
           "Merging with instruction from another function not allowed");
  }

  // Add this instruction's DIAssignID too, if it has one.
  if (auto *MD = getMetadata(LLVMContext::MD_DIAssignID))
    IDs.push_back(cast<DIAssignID>(MD));

  if (IDs.empty())
    return; // No DIAssignID tags to process.

  DIAssignID *MergeID = IDs[0];
  for (auto It = std::next(IDs.begin()), End = IDs.end(); It != End; ++It) {
    if (*It != MergeID)
      at::RAUW(*It, MergeID);
  }
  setMetadata(LLVMContext::MD_DIAssignID, MergeID);
}

void Instruction::updateLocationAfterHoist() { dropLocation(); }

void Instruction::dropLocation() {
  const DebugLoc &DL = getDebugLoc();
  if (!DL)
    return;

  // If this isn't a call, drop the location to allow a location from a
  // preceding instruction to propagate.
  bool MayLowerToCall = false;
  if (isa<CallBase>(this)) {
    auto *II = dyn_cast<IntrinsicInst>(this);
    MayLowerToCall =
        !II || IntrinsicInst::mayLowerToFunctionCall(II->getIntrinsicID());
  }

  if (!MayLowerToCall) {
    setDebugLoc(DebugLoc());
    return;
  }

  // Set a line 0 location for calls to preserve scope information in case
  // inlining occurs.
  DISubprogram *SP = getFunction()->getSubprogram();
  if (SP)
    // If a function scope is available, set it on the line 0 location. When
    // hoisting a call to a predecessor block, using the function scope avoids
    // making it look like the callee was reached earlier than it should be.
    setDebugLoc(DILocation::get(getContext(), 0, 0, SP));
  else
    // The parent function has no scope. Go ahead and drop the location. If
    // the parent function is inlined, and the callee has a subprogram, the
    // inliner will attach a location to the call.
    //
    // One alternative is to set a line 0 location with the existing scope and
    // inlinedAt info. The location might be sensitive to when inlining occurs.
    setDebugLoc(DebugLoc());
}

//===----------------------------------------------------------------------===//
// LLVM C API implementations.
//===----------------------------------------------------------------------===//

static unsigned map_from_llvmDWARFsourcelanguage(LLVMDWARFSourceLanguage lang) {
  switch (lang) {
#define HANDLE_DW_LANG(ID, NAME, LOWER_BOUND, VERSION, VENDOR)                 \
  case LLVMDWARFSourceLanguage##NAME:                                          \
    return ID;
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DW_LANG
  }
  llvm_unreachable("Unhandled Tag");
}

template <typename DIT> DIT *unwrapDI(LLVMMetadataRef Ref) {
  return (DIT *)(Ref ? unwrap<MDNode>(Ref) : nullptr);
}

static DINode::DIFlags map_from_llvmDIFlags(LLVMDIFlags Flags) {
  return static_cast<DINode::DIFlags>(Flags);
}

static LLVMDIFlags map_to_llvmDIFlags(DINode::DIFlags Flags) {
  return static_cast<LLVMDIFlags>(Flags);
}

static DISubprogram::DISPFlags
pack_into_DISPFlags(bool IsLocalToUnit, bool IsDefinition, bool IsOptimized) {
  return DISubprogram::toSPFlags(IsLocalToUnit, IsDefinition, IsOptimized);
}

unsigned LLVMDebugMetadataVersion() {
  return DEBUG_METADATA_VERSION;
}

LLVMDIBuilderRef LLVMCreateDIBuilderDisallowUnresolved(LLVMModuleRef M) {
  return wrap(new DIBuilder(*unwrap(M), false));
}

LLVMDIBuilderRef LLVMCreateDIBuilder(LLVMModuleRef M) {
  return wrap(new DIBuilder(*unwrap(M)));
}

unsigned LLVMGetModuleDebugMetadataVersion(LLVMModuleRef M) {
  return getDebugMetadataVersionFromModule(*unwrap(M));
}

LLVMBool LLVMStripModuleDebugInfo(LLVMModuleRef M) {
  return StripDebugInfo(*unwrap(M));
}

void LLVMDisposeDIBuilder(LLVMDIBuilderRef Builder) {
  delete unwrap(Builder);
}

void LLVMDIBuilderFinalize(LLVMDIBuilderRef Builder) {
  unwrap(Builder)->finalize();
}

void LLVMDIBuilderFinalizeSubprogram(LLVMDIBuilderRef Builder,
                                     LLVMMetadataRef subprogram) {
  unwrap(Builder)->finalizeSubprogram(unwrapDI<DISubprogram>(subprogram));
}

LLVMMetadataRef LLVMDIBuilderCreateCompileUnit(
    LLVMDIBuilderRef Builder, LLVMDWARFSourceLanguage Lang,
    LLVMMetadataRef FileRef, const char *Producer, size_t ProducerLen,
    LLVMBool isOptimized, const char *Flags, size_t FlagsLen,
    unsigned RuntimeVer, const char *SplitName, size_t SplitNameLen,
    LLVMDWARFEmissionKind Kind, unsigned DWOId, LLVMBool SplitDebugInlining,
    LLVMBool DebugInfoForProfiling, const char *SysRoot, size_t SysRootLen,
    const char *SDK, size_t SDKLen) {
  auto File = unwrapDI<DIFile>(FileRef);

  return wrap(unwrap(Builder)->createCompileUnit(
      map_from_llvmDWARFsourcelanguage(Lang), File,
      StringRef(Producer, ProducerLen), isOptimized, StringRef(Flags, FlagsLen),
      RuntimeVer, StringRef(SplitName, SplitNameLen),
      static_cast<DICompileUnit::DebugEmissionKind>(Kind), DWOId,
      SplitDebugInlining, DebugInfoForProfiling,
      DICompileUnit::DebugNameTableKind::Default, false,
      StringRef(SysRoot, SysRootLen), StringRef(SDK, SDKLen)));
}

LLVMMetadataRef
LLVMDIBuilderCreateFile(LLVMDIBuilderRef Builder, const char *Filename,
                        size_t FilenameLen, const char *Directory,
                        size_t DirectoryLen) {
  return wrap(unwrap(Builder)->createFile(StringRef(Filename, FilenameLen),
                                          StringRef(Directory, DirectoryLen)));
}

LLVMMetadataRef
LLVMDIBuilderCreateModule(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentScope,
                          const char *Name, size_t NameLen,
                          const char *ConfigMacros, size_t ConfigMacrosLen,
                          const char *IncludePath, size_t IncludePathLen,
                          const char *APINotesFile, size_t APINotesFileLen) {
  return wrap(unwrap(Builder)->createModule(
      unwrapDI<DIScope>(ParentScope), StringRef(Name, NameLen),
      StringRef(ConfigMacros, ConfigMacrosLen),
      StringRef(IncludePath, IncludePathLen),
      StringRef(APINotesFile, APINotesFileLen)));
}

LLVMMetadataRef LLVMDIBuilderCreateNameSpace(LLVMDIBuilderRef Builder,
                                             LLVMMetadataRef ParentScope,
                                             const char *Name, size_t NameLen,
                                             LLVMBool ExportSymbols) {
  return wrap(unwrap(Builder)->createNameSpace(
      unwrapDI<DIScope>(ParentScope), StringRef(Name, NameLen), ExportSymbols));
}

LLVMMetadataRef LLVMDIBuilderCreateFunction(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, const char *LinkageName, size_t LinkageNameLen,
    LLVMMetadataRef File, unsigned LineNo, LLVMMetadataRef Ty,
    LLVMBool IsLocalToUnit, LLVMBool IsDefinition,
    unsigned ScopeLine, LLVMDIFlags Flags, LLVMBool IsOptimized) {
  return wrap(unwrap(Builder)->createFunction(
      unwrapDI<DIScope>(Scope), {Name, NameLen}, {LinkageName, LinkageNameLen},
      unwrapDI<DIFile>(File), LineNo, unwrapDI<DISubroutineType>(Ty), ScopeLine,
      map_from_llvmDIFlags(Flags),
      pack_into_DISPFlags(IsLocalToUnit, IsDefinition, IsOptimized), nullptr,
      nullptr, nullptr));
}


LLVMMetadataRef LLVMDIBuilderCreateLexicalBlock(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope,
    LLVMMetadataRef File, unsigned Line, unsigned Col) {
  return wrap(unwrap(Builder)->createLexicalBlock(unwrapDI<DIScope>(Scope),
                                                  unwrapDI<DIFile>(File),
                                                  Line, Col));
}

LLVMMetadataRef
LLVMDIBuilderCreateLexicalBlockFile(LLVMDIBuilderRef Builder,
                                    LLVMMetadataRef Scope,
                                    LLVMMetadataRef File,
                                    unsigned Discriminator) {
  return wrap(unwrap(Builder)->createLexicalBlockFile(unwrapDI<DIScope>(Scope),
                                                      unwrapDI<DIFile>(File),
                                                      Discriminator));
}

LLVMMetadataRef
LLVMDIBuilderCreateImportedModuleFromNamespace(LLVMDIBuilderRef Builder,
                                               LLVMMetadataRef Scope,
                                               LLVMMetadataRef NS,
                                               LLVMMetadataRef File,
                                               unsigned Line) {
  return wrap(unwrap(Builder)->createImportedModule(unwrapDI<DIScope>(Scope),
                                                    unwrapDI<DINamespace>(NS),
                                                    unwrapDI<DIFile>(File),
                                                    Line));
}

LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromAlias(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope,
    LLVMMetadataRef ImportedEntity, LLVMMetadataRef File, unsigned Line,
    LLVMMetadataRef *Elements, unsigned NumElements) {
  auto Elts =
      (NumElements > 0)
          ? unwrap(Builder)->getOrCreateArray({unwrap(Elements), NumElements})
          : nullptr;
  return wrap(unwrap(Builder)->createImportedModule(
      unwrapDI<DIScope>(Scope), unwrapDI<DIImportedEntity>(ImportedEntity),
      unwrapDI<DIFile>(File), Line, Elts));
}

LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromModule(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef M,
    LLVMMetadataRef File, unsigned Line, LLVMMetadataRef *Elements,
    unsigned NumElements) {
  auto Elts =
      (NumElements > 0)
          ? unwrap(Builder)->getOrCreateArray({unwrap(Elements), NumElements})
          : nullptr;
  return wrap(unwrap(Builder)->createImportedModule(
      unwrapDI<DIScope>(Scope), unwrapDI<DIModule>(M), unwrapDI<DIFile>(File),
      Line, Elts));
}

LLVMMetadataRef LLVMDIBuilderCreateImportedDeclaration(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef Decl,
    LLVMMetadataRef File, unsigned Line, const char *Name, size_t NameLen,
    LLVMMetadataRef *Elements, unsigned NumElements) {
  auto Elts =
      (NumElements > 0)
          ? unwrap(Builder)->getOrCreateArray({unwrap(Elements), NumElements})
          : nullptr;
  return wrap(unwrap(Builder)->createImportedDeclaration(
      unwrapDI<DIScope>(Scope), unwrapDI<DINode>(Decl), unwrapDI<DIFile>(File),
      Line, {Name, NameLen}, Elts));
}

LLVMMetadataRef
LLVMDIBuilderCreateDebugLocation(LLVMContextRef Ctx, unsigned Line,
                                 unsigned Column, LLVMMetadataRef Scope,
                                 LLVMMetadataRef InlinedAt) {
  return wrap(DILocation::get(*unwrap(Ctx), Line, Column, unwrap(Scope),
                              unwrap(InlinedAt)));
}

unsigned LLVMDILocationGetLine(LLVMMetadataRef Location) {
  return unwrapDI<DILocation>(Location)->getLine();
}

unsigned LLVMDILocationGetColumn(LLVMMetadataRef Location) {
  return unwrapDI<DILocation>(Location)->getColumn();
}

LLVMMetadataRef LLVMDILocationGetScope(LLVMMetadataRef Location) {
  return wrap(unwrapDI<DILocation>(Location)->getScope());
}

LLVMMetadataRef LLVMDILocationGetInlinedAt(LLVMMetadataRef Location) {
  return wrap(unwrapDI<DILocation>(Location)->getInlinedAt());
}

LLVMMetadataRef LLVMDIScopeGetFile(LLVMMetadataRef Scope) {
  return wrap(unwrapDI<DIScope>(Scope)->getFile());
}

const char *LLVMDIFileGetDirectory(LLVMMetadataRef File, unsigned *Len) {
  auto Dir = unwrapDI<DIFile>(File)->getDirectory();
  *Len = Dir.size();
  return Dir.data();
}

const char *LLVMDIFileGetFilename(LLVMMetadataRef File, unsigned *Len) {
  auto Name = unwrapDI<DIFile>(File)->getFilename();
  *Len = Name.size();
  return Name.data();
}

const char *LLVMDIFileGetSource(LLVMMetadataRef File, unsigned *Len) {
  if (auto Src = unwrapDI<DIFile>(File)->getSource()) {
    *Len = Src->size();
    return Src->data();
  }
  *Len = 0;
  return "";
}

LLVMMetadataRef LLVMDIBuilderCreateMacro(LLVMDIBuilderRef Builder,
                                         LLVMMetadataRef ParentMacroFile,
                                         unsigned Line,
                                         LLVMDWARFMacinfoRecordType RecordType,
                                         const char *Name, size_t NameLen,
                                         const char *Value, size_t ValueLen) {
  return wrap(
      unwrap(Builder)->createMacro(unwrapDI<DIMacroFile>(ParentMacroFile), Line,
                                   static_cast<MacinfoRecordType>(RecordType),
                                   {Name, NameLen}, {Value, ValueLen}));
}

LLVMMetadataRef
LLVMDIBuilderCreateTempMacroFile(LLVMDIBuilderRef Builder,
                                 LLVMMetadataRef ParentMacroFile, unsigned Line,
                                 LLVMMetadataRef File) {
  return wrap(unwrap(Builder)->createTempMacroFile(
      unwrapDI<DIMacroFile>(ParentMacroFile), Line, unwrapDI<DIFile>(File)));
}

LLVMMetadataRef LLVMDIBuilderCreateEnumerator(LLVMDIBuilderRef Builder,
                                              const char *Name, size_t NameLen,
                                              int64_t Value,
                                              LLVMBool IsUnsigned) {
  return wrap(unwrap(Builder)->createEnumerator({Name, NameLen}, Value,
                                                IsUnsigned != 0));
}

LLVMMetadataRef LLVMDIBuilderCreateEnumerationType(
  LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
  size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
  uint64_t SizeInBits, uint32_t AlignInBits, LLVMMetadataRef *Elements,
  unsigned NumElements, LLVMMetadataRef ClassTy) {
auto Elts = unwrap(Builder)->getOrCreateArray({unwrap(Elements),
                                               NumElements});
return wrap(unwrap(Builder)->createEnumerationType(
    unwrapDI<DIScope>(Scope), {Name, NameLen}, unwrapDI<DIFile>(File),
    LineNumber, SizeInBits, AlignInBits, Elts, unwrapDI<DIType>(ClassTy)));
}

LLVMMetadataRef LLVMDIBuilderCreateUnionType(
  LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
  size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
  uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags,
  LLVMMetadataRef *Elements, unsigned NumElements, unsigned RunTimeLang,
  const char *UniqueId, size_t UniqueIdLen) {
  auto Elts = unwrap(Builder)->getOrCreateArray({unwrap(Elements),
                                                 NumElements});
  return wrap(unwrap(Builder)->createUnionType(
     unwrapDI<DIScope>(Scope), {Name, NameLen}, unwrapDI<DIFile>(File),
     LineNumber, SizeInBits, AlignInBits, map_from_llvmDIFlags(Flags),
     Elts, RunTimeLang, {UniqueId, UniqueIdLen}));
}


LLVMMetadataRef
LLVMDIBuilderCreateArrayType(LLVMDIBuilderRef Builder, uint64_t Size,
                             uint32_t AlignInBits, LLVMMetadataRef Ty,
                             LLVMMetadataRef *Subscripts,
                             unsigned NumSubscripts) {
  auto Subs = unwrap(Builder)->getOrCreateArray({unwrap(Subscripts),
                                                 NumSubscripts});
  return wrap(unwrap(Builder)->createArrayType(Size, AlignInBits,
                                               unwrapDI<DIType>(Ty), Subs));
}

LLVMMetadataRef
LLVMDIBuilderCreateVectorType(LLVMDIBuilderRef Builder, uint64_t Size,
                              uint32_t AlignInBits, LLVMMetadataRef Ty,
                              LLVMMetadataRef *Subscripts,
                              unsigned NumSubscripts) {
  auto Subs = unwrap(Builder)->getOrCreateArray({unwrap(Subscripts),
                                                 NumSubscripts});
  return wrap(unwrap(Builder)->createVectorType(Size, AlignInBits,
                                                unwrapDI<DIType>(Ty), Subs));
}

LLVMMetadataRef
LLVMDIBuilderCreateBasicType(LLVMDIBuilderRef Builder, const char *Name,
                             size_t NameLen, uint64_t SizeInBits,
                             LLVMDWARFTypeEncoding Encoding,
                             LLVMDIFlags Flags) {
  return wrap(unwrap(Builder)->createBasicType({Name, NameLen},
                                               SizeInBits, Encoding,
                                               map_from_llvmDIFlags(Flags)));
}

LLVMMetadataRef LLVMDIBuilderCreatePointerType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef PointeeTy,
    uint64_t SizeInBits, uint32_t AlignInBits, unsigned AddressSpace,
    const char *Name, size_t NameLen) {
  return wrap(unwrap(Builder)->createPointerType(
      unwrapDI<DIType>(PointeeTy), SizeInBits, AlignInBits, AddressSpace,
      {Name, NameLen}));
}

LLVMMetadataRef LLVMDIBuilderCreateStructType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags,
    LLVMMetadataRef DerivedFrom, LLVMMetadataRef *Elements,
    unsigned NumElements, unsigned RunTimeLang, LLVMMetadataRef VTableHolder,
    const char *UniqueId, size_t UniqueIdLen) {
  auto Elts = unwrap(Builder)->getOrCreateArray({unwrap(Elements),
                                                 NumElements});
  return wrap(unwrap(Builder)->createStructType(
      unwrapDI<DIScope>(Scope), {Name, NameLen}, unwrapDI<DIFile>(File),
      LineNumber, SizeInBits, AlignInBits, map_from_llvmDIFlags(Flags),
      unwrapDI<DIType>(DerivedFrom), Elts, RunTimeLang,
      unwrapDI<DIType>(VTableHolder), {UniqueId, UniqueIdLen}));
}

LLVMMetadataRef LLVMDIBuilderCreateMemberType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNo, uint64_t SizeInBits,
    uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags,
    LLVMMetadataRef Ty) {
  return wrap(unwrap(Builder)->createMemberType(unwrapDI<DIScope>(Scope),
      {Name, NameLen}, unwrapDI<DIFile>(File), LineNo, SizeInBits, AlignInBits,
      OffsetInBits, map_from_llvmDIFlags(Flags), unwrapDI<DIType>(Ty)));
}

LLVMMetadataRef
LLVMDIBuilderCreateUnspecifiedType(LLVMDIBuilderRef Builder, const char *Name,
                                   size_t NameLen) {
  return wrap(unwrap(Builder)->createUnspecifiedType({Name, NameLen}));
}

LLVMMetadataRef
LLVMDIBuilderCreateStaticMemberType(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNumber,
    LLVMMetadataRef Type, LLVMDIFlags Flags, LLVMValueRef ConstantVal,
    uint32_t AlignInBits) {
  return wrap(unwrap(Builder)->createStaticMemberType(
                  unwrapDI<DIScope>(Scope), {Name, NameLen},
                  unwrapDI<DIFile>(File), LineNumber, unwrapDI<DIType>(Type),
                  map_from_llvmDIFlags(Flags), unwrap<Constant>(ConstantVal),
                  AlignInBits));
}

LLVMMetadataRef
LLVMDIBuilderCreateObjCIVar(LLVMDIBuilderRef Builder,
                            const char *Name, size_t NameLen,
                            LLVMMetadataRef File, unsigned LineNo,
                            uint64_t SizeInBits, uint32_t AlignInBits,
                            uint64_t OffsetInBits, LLVMDIFlags Flags,
                            LLVMMetadataRef Ty, LLVMMetadataRef PropertyNode) {
  return wrap(unwrap(Builder)->createObjCIVar(
                  {Name, NameLen}, unwrapDI<DIFile>(File), LineNo,
                  SizeInBits, AlignInBits, OffsetInBits,
                  map_from_llvmDIFlags(Flags), unwrapDI<DIType>(Ty),
                  unwrapDI<MDNode>(PropertyNode)));
}

LLVMMetadataRef
LLVMDIBuilderCreateObjCProperty(LLVMDIBuilderRef Builder,
                                const char *Name, size_t NameLen,
                                LLVMMetadataRef File, unsigned LineNo,
                                const char *GetterName, size_t GetterNameLen,
                                const char *SetterName, size_t SetterNameLen,
                                unsigned PropertyAttributes,
                                LLVMMetadataRef Ty) {
  return wrap(unwrap(Builder)->createObjCProperty(
                  {Name, NameLen}, unwrapDI<DIFile>(File), LineNo,
                  {GetterName, GetterNameLen}, {SetterName, SetterNameLen},
                  PropertyAttributes, unwrapDI<DIType>(Ty)));
}

LLVMMetadataRef
LLVMDIBuilderCreateObjectPointerType(LLVMDIBuilderRef Builder,
                                     LLVMMetadataRef Type) {
  return wrap(unwrap(Builder)->createObjectPointerType(unwrapDI<DIType>(Type)));
}

LLVMMetadataRef
LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef Builder, LLVMMetadataRef Type,
                           const char *Name, size_t NameLen,
                           LLVMMetadataRef File, unsigned LineNo,
                           LLVMMetadataRef Scope, uint32_t AlignInBits) {
  return wrap(unwrap(Builder)->createTypedef(
      unwrapDI<DIType>(Type), {Name, NameLen}, unwrapDI<DIFile>(File), LineNo,
      unwrapDI<DIScope>(Scope), AlignInBits));
}

LLVMMetadataRef
LLVMDIBuilderCreateInheritance(LLVMDIBuilderRef Builder,
                               LLVMMetadataRef Ty, LLVMMetadataRef BaseTy,
                               uint64_t BaseOffset, uint32_t VBPtrOffset,
                               LLVMDIFlags Flags) {
  return wrap(unwrap(Builder)->createInheritance(
                  unwrapDI<DIType>(Ty), unwrapDI<DIType>(BaseTy),
                  BaseOffset, VBPtrOffset, map_from_llvmDIFlags(Flags)));
}

LLVMMetadataRef
LLVMDIBuilderCreateForwardDecl(
    LLVMDIBuilderRef Builder, unsigned Tag, const char *Name,
    size_t NameLen, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned Line,
    unsigned RuntimeLang, uint64_t SizeInBits, uint32_t AlignInBits,
    const char *UniqueIdentifier, size_t UniqueIdentifierLen) {
  return wrap(unwrap(Builder)->createForwardDecl(
                  Tag, {Name, NameLen}, unwrapDI<DIScope>(Scope),
                  unwrapDI<DIFile>(File), Line, RuntimeLang, SizeInBits,
                  AlignInBits, {UniqueIdentifier, UniqueIdentifierLen}));
}

LLVMMetadataRef
LLVMDIBuilderCreateReplaceableCompositeType(
    LLVMDIBuilderRef Builder, unsigned Tag, const char *Name,
    size_t NameLen, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned Line,
    unsigned RuntimeLang, uint64_t SizeInBits, uint32_t AlignInBits,
    LLVMDIFlags Flags, const char *UniqueIdentifier,
    size_t UniqueIdentifierLen) {
  return wrap(unwrap(Builder)->createReplaceableCompositeType(
                  Tag, {Name, NameLen}, unwrapDI<DIScope>(Scope),
                  unwrapDI<DIFile>(File), Line, RuntimeLang, SizeInBits,
                  AlignInBits, map_from_llvmDIFlags(Flags),
                  {UniqueIdentifier, UniqueIdentifierLen}));
}

LLVMMetadataRef
LLVMDIBuilderCreateQualifiedType(LLVMDIBuilderRef Builder, unsigned Tag,
                                 LLVMMetadataRef Type) {
  return wrap(unwrap(Builder)->createQualifiedType(Tag,
                                                   unwrapDI<DIType>(Type)));
}

LLVMMetadataRef
LLVMDIBuilderCreateReferenceType(LLVMDIBuilderRef Builder, unsigned Tag,
                                 LLVMMetadataRef Type) {
  return wrap(unwrap(Builder)->createReferenceType(Tag,
                                                   unwrapDI<DIType>(Type)));
}

LLVMMetadataRef
LLVMDIBuilderCreateNullPtrType(LLVMDIBuilderRef Builder) {
  return wrap(unwrap(Builder)->createNullPtrType());
}

LLVMMetadataRef
LLVMDIBuilderCreateMemberPointerType(LLVMDIBuilderRef Builder,
                                     LLVMMetadataRef PointeeType,
                                     LLVMMetadataRef ClassType,
                                     uint64_t SizeInBits,
                                     uint32_t AlignInBits,
                                     LLVMDIFlags Flags) {
  return wrap(unwrap(Builder)->createMemberPointerType(
                  unwrapDI<DIType>(PointeeType),
                  unwrapDI<DIType>(ClassType), AlignInBits, SizeInBits,
                  map_from_llvmDIFlags(Flags)));
}

LLVMMetadataRef
LLVMDIBuilderCreateBitFieldMemberType(LLVMDIBuilderRef Builder,
                                      LLVMMetadataRef Scope,
                                      const char *Name, size_t NameLen,
                                      LLVMMetadataRef File, unsigned LineNumber,
                                      uint64_t SizeInBits,
                                      uint64_t OffsetInBits,
                                      uint64_t StorageOffsetInBits,
                                      LLVMDIFlags Flags, LLVMMetadataRef Type) {
  return wrap(unwrap(Builder)->createBitFieldMemberType(
                  unwrapDI<DIScope>(Scope), {Name, NameLen},
                  unwrapDI<DIFile>(File), LineNumber,
                  SizeInBits, OffsetInBits, StorageOffsetInBits,
                  map_from_llvmDIFlags(Flags), unwrapDI<DIType>(Type)));
}

LLVMMetadataRef LLVMDIBuilderCreateClassType(LLVMDIBuilderRef Builder,
    LLVMMetadataRef Scope, const char *Name, size_t NameLen,
    LLVMMetadataRef File, unsigned LineNumber, uint64_t SizeInBits,
    uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags,
    LLVMMetadataRef DerivedFrom,
    LLVMMetadataRef *Elements, unsigned NumElements,
    LLVMMetadataRef VTableHolder, LLVMMetadataRef TemplateParamsNode,
    const char *UniqueIdentifier, size_t UniqueIdentifierLen) {
  auto Elts = unwrap(Builder)->getOrCreateArray({unwrap(Elements),
                                                 NumElements});
  return wrap(unwrap(Builder)->createClassType(
                  unwrapDI<DIScope>(Scope), {Name, NameLen},
                  unwrapDI<DIFile>(File), LineNumber,
                  SizeInBits, AlignInBits, OffsetInBits,
                  map_from_llvmDIFlags(Flags), unwrapDI<DIType>(DerivedFrom),
                  Elts, unwrapDI<DIType>(VTableHolder),
                  unwrapDI<MDNode>(TemplateParamsNode),
                  {UniqueIdentifier, UniqueIdentifierLen}));
}

LLVMMetadataRef
LLVMDIBuilderCreateArtificialType(LLVMDIBuilderRef Builder,
                                  LLVMMetadataRef Type) {
  return wrap(unwrap(Builder)->createArtificialType(unwrapDI<DIType>(Type)));
}

uint16_t LLVMGetDINodeTag(LLVMMetadataRef MD) {
  return unwrapDI<DINode>(MD)->getTag();
}

const char *LLVMDITypeGetName(LLVMMetadataRef DType, size_t *Length) {
  StringRef Str = unwrap<DIType>(DType)->getName();
  *Length = Str.size();
  return Str.data();
}

uint64_t LLVMDITypeGetSizeInBits(LLVMMetadataRef DType) {
  return unwrapDI<DIType>(DType)->getSizeInBits();
}

uint64_t LLVMDITypeGetOffsetInBits(LLVMMetadataRef DType) {
  return unwrapDI<DIType>(DType)->getOffsetInBits();
}

uint32_t LLVMDITypeGetAlignInBits(LLVMMetadataRef DType) {
  return unwrapDI<DIType>(DType)->getAlignInBits();
}

unsigned LLVMDITypeGetLine(LLVMMetadataRef DType) {
  return unwrapDI<DIType>(DType)->getLine();
}

LLVMDIFlags LLVMDITypeGetFlags(LLVMMetadataRef DType) {
  return map_to_llvmDIFlags(unwrapDI<DIType>(DType)->getFlags());
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateTypeArray(LLVMDIBuilderRef Builder,
                                                  LLVMMetadataRef *Types,
                                                  size_t Length) {
  return wrap(
      unwrap(Builder)->getOrCreateTypeArray({unwrap(Types), Length}).get());
}

LLVMMetadataRef
LLVMDIBuilderCreateSubroutineType(LLVMDIBuilderRef Builder,
                                  LLVMMetadataRef File,
                                  LLVMMetadataRef *ParameterTypes,
                                  unsigned NumParameterTypes,
                                  LLVMDIFlags Flags) {
  auto Elts = unwrap(Builder)->getOrCreateTypeArray({unwrap(ParameterTypes),
                                                     NumParameterTypes});
  return wrap(unwrap(Builder)->createSubroutineType(
    Elts, map_from_llvmDIFlags(Flags)));
}

LLVMMetadataRef LLVMDIBuilderCreateExpression(LLVMDIBuilderRef Builder,
                                              uint64_t *Addr, size_t Length) {
  return wrap(
      unwrap(Builder)->createExpression(ArrayRef<uint64_t>(Addr, Length)));
}

LLVMMetadataRef
LLVMDIBuilderCreateConstantValueExpression(LLVMDIBuilderRef Builder,
                                           uint64_t Value) {
  return wrap(unwrap(Builder)->createConstantValueExpression(Value));
}

LLVMMetadataRef LLVMDIBuilderCreateGlobalVariableExpression(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, const char *Linkage, size_t LinkLen, LLVMMetadataRef File,
    unsigned LineNo, LLVMMetadataRef Ty, LLVMBool LocalToUnit,
    LLVMMetadataRef Expr, LLVMMetadataRef Decl, uint32_t AlignInBits) {
  return wrap(unwrap(Builder)->createGlobalVariableExpression(
      unwrapDI<DIScope>(Scope), {Name, NameLen}, {Linkage, LinkLen},
      unwrapDI<DIFile>(File), LineNo, unwrapDI<DIType>(Ty), LocalToUnit,
      true, unwrap<DIExpression>(Expr), unwrapDI<MDNode>(Decl),
      nullptr, AlignInBits));
}

LLVMMetadataRef LLVMDIGlobalVariableExpressionGetVariable(LLVMMetadataRef GVE) {
  return wrap(unwrapDI<DIGlobalVariableExpression>(GVE)->getVariable());
}

LLVMMetadataRef LLVMDIGlobalVariableExpressionGetExpression(
    LLVMMetadataRef GVE) {
  return wrap(unwrapDI<DIGlobalVariableExpression>(GVE)->getExpression());
}

LLVMMetadataRef LLVMDIVariableGetFile(LLVMMetadataRef Var) {
  return wrap(unwrapDI<DIVariable>(Var)->getFile());
}

LLVMMetadataRef LLVMDIVariableGetScope(LLVMMetadataRef Var) {
  return wrap(unwrapDI<DIVariable>(Var)->getScope());
}

unsigned LLVMDIVariableGetLine(LLVMMetadataRef Var) {
  return unwrapDI<DIVariable>(Var)->getLine();
}

LLVMMetadataRef LLVMTemporaryMDNode(LLVMContextRef Ctx, LLVMMetadataRef *Data,
                                    size_t Count) {
  return wrap(
      MDTuple::getTemporary(*unwrap(Ctx), {unwrap(Data), Count}).release());
}

void LLVMDisposeTemporaryMDNode(LLVMMetadataRef TempNode) {
  MDNode::deleteTemporary(unwrapDI<MDNode>(TempNode));
}

void LLVMMetadataReplaceAllUsesWith(LLVMMetadataRef TargetMetadata,
                                    LLVMMetadataRef Replacement) {
  auto *Node = unwrapDI<MDNode>(TargetMetadata);
  Node->replaceAllUsesWith(unwrap(Replacement));
  MDNode::deleteTemporary(Node);
}

LLVMMetadataRef LLVMDIBuilderCreateTempGlobalVariableFwdDecl(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, const char *Linkage, size_t LnkLen, LLVMMetadataRef File,
    unsigned LineNo, LLVMMetadataRef Ty, LLVMBool LocalToUnit,
    LLVMMetadataRef Decl, uint32_t AlignInBits) {
  return wrap(unwrap(Builder)->createTempGlobalVariableFwdDecl(
      unwrapDI<DIScope>(Scope), {Name, NameLen}, {Linkage, LnkLen},
      unwrapDI<DIFile>(File), LineNo, unwrapDI<DIType>(Ty), LocalToUnit,
      unwrapDI<MDNode>(Decl), nullptr, AlignInBits));
}

LLVMValueRef
LLVMDIBuilderInsertDeclareBefore(LLVMDIBuilderRef Builder, LLVMValueRef Storage,
                                 LLVMMetadataRef VarInfo, LLVMMetadataRef Expr,
                                 LLVMMetadataRef DL, LLVMValueRef Instr) {
  return wrap(unwrap(Builder)->insertDeclare(
                  unwrap(Storage), unwrap<DILocalVariable>(VarInfo),
                  unwrap<DIExpression>(Expr), unwrap<DILocation>(DL),
                  unwrap<Instruction>(Instr)));
}

LLVMValueRef LLVMDIBuilderInsertDeclareAtEnd(
    LLVMDIBuilderRef Builder, LLVMValueRef Storage, LLVMMetadataRef VarInfo,
    LLVMMetadataRef Expr, LLVMMetadataRef DL, LLVMBasicBlockRef Block) {
  return wrap(unwrap(Builder)->insertDeclare(
                  unwrap(Storage), unwrap<DILocalVariable>(VarInfo),
                  unwrap<DIExpression>(Expr), unwrap<DILocation>(DL),
                  unwrap(Block)));
}

LLVMValueRef LLVMDIBuilderInsertDbgValueBefore(LLVMDIBuilderRef Builder,
                                               LLVMValueRef Val,
                                               LLVMMetadataRef VarInfo,
                                               LLVMMetadataRef Expr,
                                               LLVMMetadataRef DebugLoc,
                                               LLVMValueRef Instr) {
  return wrap(unwrap(Builder)->insertDbgValueIntrinsic(
                  unwrap(Val), unwrap<DILocalVariable>(VarInfo),
                  unwrap<DIExpression>(Expr), unwrap<DILocation>(DebugLoc),
                  unwrap<Instruction>(Instr)));
}

LLVMValueRef LLVMDIBuilderInsertDbgValueAtEnd(LLVMDIBuilderRef Builder,
                                              LLVMValueRef Val,
                                              LLVMMetadataRef VarInfo,
                                              LLVMMetadataRef Expr,
                                              LLVMMetadataRef DebugLoc,
                                              LLVMBasicBlockRef Block) {
  return wrap(unwrap(Builder)->insertDbgValueIntrinsic(
                  unwrap(Val), unwrap<DILocalVariable>(VarInfo),
                  unwrap<DIExpression>(Expr), unwrap<DILocation>(DebugLoc),
                  unwrap(Block)));
}

LLVMMetadataRef LLVMDIBuilderCreateAutoVariable(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, LLVMMetadataRef File, unsigned LineNo, LLVMMetadataRef Ty,
    LLVMBool AlwaysPreserve, LLVMDIFlags Flags, uint32_t AlignInBits) {
  return wrap(unwrap(Builder)->createAutoVariable(
                  unwrap<DIScope>(Scope), {Name, NameLen}, unwrap<DIFile>(File),
                  LineNo, unwrap<DIType>(Ty), AlwaysPreserve,
                  map_from_llvmDIFlags(Flags), AlignInBits));
}

LLVMMetadataRef LLVMDIBuilderCreateParameterVariable(
    LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name,
    size_t NameLen, unsigned ArgNo, LLVMMetadataRef File, unsigned LineNo,
    LLVMMetadataRef Ty, LLVMBool AlwaysPreserve, LLVMDIFlags Flags) {
  return wrap(unwrap(Builder)->createParameterVariable(
                  unwrap<DIScope>(Scope), {Name, NameLen}, ArgNo, unwrap<DIFile>(File),
                  LineNo, unwrap<DIType>(Ty), AlwaysPreserve,
                  map_from_llvmDIFlags(Flags)));
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(LLVMDIBuilderRef Builder,
                                                 int64_t Lo, int64_t Count) {
  return wrap(unwrap(Builder)->getOrCreateSubrange(Lo, Count));
}

LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(LLVMDIBuilderRef Builder,
                                              LLVMMetadataRef *Data,
                                              size_t Length) {
  Metadata **DataValue = unwrap(Data);
  return wrap(unwrap(Builder)->getOrCreateArray({DataValue, Length}).get());
}

LLVMMetadataRef LLVMGetSubprogram(LLVMValueRef Func) {
  return wrap(unwrap<Function>(Func)->getSubprogram());
}

void LLVMSetSubprogram(LLVMValueRef Func, LLVMMetadataRef SP) {
  unwrap<Function>(Func)->setSubprogram(unwrap<DISubprogram>(SP));
}

unsigned LLVMDISubprogramGetLine(LLVMMetadataRef Subprogram) {
  return unwrapDI<DISubprogram>(Subprogram)->getLine();
}

LLVMMetadataRef LLVMInstructionGetDebugLoc(LLVMValueRef Inst) {
  return wrap(unwrap<Instruction>(Inst)->getDebugLoc().getAsMDNode());
}

void LLVMInstructionSetDebugLoc(LLVMValueRef Inst, LLVMMetadataRef Loc) {
  if (Loc)
    unwrap<Instruction>(Inst)->setDebugLoc(DebugLoc(unwrap<MDNode>(Loc)));
  else
    unwrap<Instruction>(Inst)->setDebugLoc(DebugLoc());
}

LLVMMetadataKind LLVMGetMetadataKind(LLVMMetadataRef Metadata) {
  switch(unwrap(Metadata)->getMetadataID()) {
#define HANDLE_METADATA_LEAF(CLASS) \
  case Metadata::CLASS##Kind: \
    return (LLVMMetadataKind)LLVM##CLASS##MetadataKind;
#include "llvm/IR/Metadata.def"
  default:
    return (LLVMMetadataKind)LLVMGenericDINodeMetadataKind;
  }
}

AssignmentInstRange at::getAssignmentInsts(DIAssignID *ID) {
  assert(ID && "Expected non-null ID");
  LLVMContext &Ctx = ID->getContext();
  auto &Map = Ctx.pImpl->AssignmentIDToInstrs;

  auto MapIt = Map.find(ID);
  if (MapIt == Map.end())
    return make_range(nullptr, nullptr);

  return make_range(MapIt->second.begin(), MapIt->second.end());
}

AssignmentMarkerRange at::getAssignmentMarkers(DIAssignID *ID) {
  assert(ID && "Expected non-null ID");
  LLVMContext &Ctx = ID->getContext();

  auto *IDAsValue = MetadataAsValue::getIfExists(Ctx, ID);

  // The ID is only used wrapped in MetadataAsValue(ID), so lets check that
  // one of those already exists first.
  if (!IDAsValue)
    return make_range(Value::user_iterator(), Value::user_iterator());

  return make_range(IDAsValue->user_begin(), IDAsValue->user_end());
}

void at::deleteAssignmentMarkers(const Instruction *Inst) {
  auto Range = getAssignmentMarkers(Inst);
  if (Range.empty())
    return;
  SmallVector<DbgAssignIntrinsic *> ToDelete(Range.begin(), Range.end());
  for (auto *DAI : ToDelete)
    DAI->eraseFromParent();
}

void at::RAUW(DIAssignID *Old, DIAssignID *New) {
  // Replace MetadataAsValue uses.
  if (auto *OldIDAsValue =
          MetadataAsValue::getIfExists(Old->getContext(), Old)) {
    auto *NewIDAsValue = MetadataAsValue::get(Old->getContext(), New);
    OldIDAsValue->replaceAllUsesWith(NewIDAsValue);
  }

  // Replace attachments.
  AssignmentInstRange InstRange = getAssignmentInsts(Old);
  // Use intermediate storage for the instruction ptrs because the
  // getAssignmentInsts range iterators will be invalidated by adding and
  // removing DIAssignID attachments.
  SmallVector<Instruction *> InstVec(InstRange.begin(), InstRange.end());
  for (auto *I : InstVec)
    I->setMetadata(LLVMContext::MD_DIAssignID, New);
}

void at::deleteAll(Function *F) {
  SmallVector<DbgAssignIntrinsic *, 12> ToDelete;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto *DAI = dyn_cast<DbgAssignIntrinsic>(&I))
        ToDelete.push_back(DAI);
      else
        I.setMetadata(LLVMContext::MD_DIAssignID, nullptr);
    }
  }
  for (auto *DAI : ToDelete)
    DAI->eraseFromParent();
}

bool at::calculateFragmentIntersect(
    const DataLayout &DL, const Value *Dest, uint64_t SliceOffsetInBits,
    uint64_t SliceSizeInBits, const DbgAssignIntrinsic *DAI,
    std::optional<DIExpression::FragmentInfo> &Result) {
  // There are multiple offsets at play in this function, so let's break it
  // down. Starting with how variables may be stored in allocas:
  //
  //   1 Simplest case: variable is alloca sized and starts at offset 0.
  //   2 Variable is larger than the alloca: the alloca holds just a part of it.
  //   3 Variable is smaller than the alloca: the alloca may hold multiple
  //   variables.
  //
  // Imagine we have a store to the entire alloca. In case (3) the store
  // affects bits outside of the bounds of each variable. In case (2), where
  // the alloca holds the Xth bit to the Yth bit of a variable, the
  // zero-offset store doesn't represent an assignment at offset zero to the
  // variable. It is an assignment to offset X.
  //
  // # Example 1
  // Obviously, not all stores are alloca-sized and have zero offset. Imagine
  // the lower 32 bits of this store are dead and are going to be DSEd:
  //
  //    store i64 %v, ptr %dest, !DIAssignID !1
  //    dbg.assign(..., !DIExpression(fragment, 128, 32), !1, %dest,
  //               !DIExpression(DW_OP_plus_uconst, 4))
  //
  // Goal: Given our dead bits at offset:0 size:32 for the store, determine the
  // part of the variable, which fits in the fragment expressed by the
  // dbg.assign, that has been killed, if any.
  //
  //     calculateFragmentIntersect(..., SliceOffsetInBits=0,
  //                 SliceSizeInBits=32, Dest=%dest, DAI=dbg.assign)
  //
  // Drawing the store (s) in memory followed by the shortened version ($),
  // then the dbg.assign (d), with the fragment information on a seperate scale
  // underneath:
  //
  // Memory
  // offset
  //   from
  //   dest 0      63
  //        |      |
  //       s[######] - Original stores 64 bits to Dest.
  //       $----[##] - DSE says the lower 32 bits are dead, to be removed.
  //       d    [##] - DAI's address-modifying expression adds 4 bytes to dest.
  // Variable   |  |
  // Fragment   128|
  //  Offsets      159
  //
  // The answer is achieved in a few steps:
  // 1. Add the fragment offset to the store offset:
  //      SliceOffsetInBits:0 + VarFrag.OffsetInBits:128 = 128
  //
  // 2. Subtract the address-modifying expression offset plus difference
  //    between d.address and dest:
  //      128 - (expression_offset:32 + (d.address - dest):0) = 96
  //
  // 3. That offset along with the store size (32) represents the bits of the
  //    variable that'd be affected by the store. Call it SliceOfVariable.
  //    Intersect that with DAI's fragment info:
  //      SliceOfVariable ∩ DAI_fragment = none
  //
  // In this case: none of the dead bits of the store affect DAI.
  //
  // # Example 2
  // Similar example with the same goal. This time the upper 16 bits
  // of the store are going to be DSE'd.
  //
  //    store i64 %v, ptr %dest, !DIAssignID !1
  //    dbg.assign(..., !DIExpression(fragment, 128, 32), !1, %dest,
  //               !DIExpression(DW_OP_plus_uconst, 4))
  //
  //     calculateFragmentIntersect(..., SliceOffsetInBits=48,
  //                 SliceSizeInBits=16, Dest=%dest, DAI=dbg.assign)
  //
  // Memory
  // offset
  //   from
  //   dest 0      63
  //        |      |
  //       s[######] - Original stores 64 bits to Dest.
  //       $[####]-- - DSE says the upper 16 bits are dead, to be removed.
  //       d    [##] - DAI's address-modifying expression adds 4 bytes to dest.
  // Variable   |  |
  // Fragment   128|
  //  Offsets      159
  //
  // Using the same steps in the first example:
  // 1. SliceOffsetInBits:48 + VarFrag.OffsetInBits:128 = 176
  // 2. 176 - (expression_offset:32 + (d.address - dest):0) = 144
  // 3. SliceOfVariable offset = 144, size = 16:
  //      SliceOfVariable ∩ DAI_fragment = (offset: 144, size: 16)
  // SliceOfVariable tells us the bits of the variable described by DAI that are
  // affected by the DSE.
  if (DAI->isKillAddress())
    return false;

  DIExpression::FragmentInfo VarFrag = DAI->getFragmentOrEntireVariable();
  if (VarFrag.SizeInBits == 0)
    return false; // Variable size is unknown.

  // Calculate the difference between Dest and the dbg.assign address +
  // address-modifying expression.
  int64_t PointerOffsetInBits;
  {
    auto DestOffsetInBytes = llvm::isPointerOffset(Dest, DAI->getAddress(), DL);
    if (!DestOffsetInBytes)
      return false; // Can't calculate difference in addresses.

    int64_t ExprOffsetInBytes;
    if (!DAI->getAddressExpression()->extractIfOffset(ExprOffsetInBytes))
      return false;

    int64_t PointerOffsetInBytes = *DestOffsetInBytes + ExprOffsetInBytes;
    PointerOffsetInBits = PointerOffsetInBytes * 8;
  }

  // Adjust the slice offset so that we go from describing the a slice
  // of memory to a slice of the variable.
  int64_t NewOffsetInBits =
      SliceOffsetInBits + VarFrag.OffsetInBits - PointerOffsetInBits;
  if (NewOffsetInBits < 0)
    return false; // Fragment offsets can only be positive.
  DIExpression::FragmentInfo SliceOfVariable(SliceSizeInBits, NewOffsetInBits);
  // Intersect the variable slice with DAI's fragment to trim it down to size.
  DIExpression::FragmentInfo TrimmedSliceOfVariable =
      DIExpression::FragmentInfo::intersect(SliceOfVariable, VarFrag);
  if (TrimmedSliceOfVariable == VarFrag)
    Result = std::nullopt;
  else
    Result = TrimmedSliceOfVariable;
  return true;
}

/// Collect constant properies (base, size, offset) of \p StoreDest.
/// Return std::nullopt if any properties are not constants.
static std::optional<AssignmentInfo>
getAssignmentInfoImpl(const DataLayout &DL, const Value *StoreDest,
                      uint64_t SizeInBits) {
  APInt GEPOffset(DL.getIndexTypeSizeInBits(StoreDest->getType()), 0);
  const Value *Base = StoreDest->stripAndAccumulateConstantOffsets(
      DL, GEPOffset, /*AllowNonInbounds*/ true);
  uint64_t OffsetInBytes = GEPOffset.getLimitedValue();
  // Check for overflow.
  if (OffsetInBytes == UINT64_MAX)
    return std::nullopt;
  if (const auto *Alloca = dyn_cast<AllocaInst>(Base))
    return AssignmentInfo(DL, Alloca, OffsetInBytes * 8, SizeInBits);
  return std::nullopt;
}

std::optional<AssignmentInfo> at::getAssignmentInfo(const DataLayout &DL,
                                                    const MemIntrinsic *I) {
  const Value *StoreDest = I->getRawDest();
  // Assume 8 bit bytes.
  auto *ConstLengthInBytes = dyn_cast<ConstantInt>(I->getLength());
  if (!ConstLengthInBytes)
    // We can't use a non-const size, bail.
    return std::nullopt;
  uint64_t SizeInBits = 8 * ConstLengthInBytes->getZExtValue();
  return getAssignmentInfoImpl(DL, StoreDest, SizeInBits);
}

std::optional<AssignmentInfo> at::getAssignmentInfo(const DataLayout &DL,
                                                    const StoreInst *SI) {
  const Value *StoreDest = SI->getPointerOperand();
  uint64_t SizeInBits = DL.getTypeSizeInBits(SI->getValueOperand()->getType());
  return getAssignmentInfoImpl(DL, StoreDest, SizeInBits);
}

std::optional<AssignmentInfo> at::getAssignmentInfo(const DataLayout &DL,
                                                    const AllocaInst *AI) {
  uint64_t SizeInBits = DL.getTypeSizeInBits(AI->getAllocatedType());
  return getAssignmentInfoImpl(DL, AI, SizeInBits);
}

/// Returns nullptr if the assignment shouldn't be attributed to this variable.
static CallInst *emitDbgAssign(AssignmentInfo Info, Value *Val, Value *Dest,
                               Instruction &StoreLikeInst,
                               const VarRecord &VarRec, DIBuilder &DIB) {
  auto *ID = StoreLikeInst.getMetadata(LLVMContext::MD_DIAssignID);
  assert(ID && "Store instruction must have DIAssignID metadata");
  (void)ID;

  const uint64_t StoreStartBit = Info.OffsetInBits;
  const uint64_t StoreEndBit = Info.OffsetInBits + Info.SizeInBits;

  uint64_t FragStartBit = StoreStartBit;
  uint64_t FragEndBit = StoreEndBit;

  bool StoreToWholeVariable = Info.StoreToWholeAlloca;
  if (auto Size = VarRec.Var->getSizeInBits()) {
    // NOTE: trackAssignments doesn't understand base expressions yet, so all
    // variables that reach here are guaranteed to start at offset 0 in the
    // alloca.
    const uint64_t VarStartBit = 0;
    const uint64_t VarEndBit = *Size;

    // FIXME: trim FragStartBit when nonzero VarStartBit is supported.
    FragEndBit = std::min(FragEndBit, VarEndBit);

    // Discard stores to bits outside this variable.
    if (FragStartBit >= FragEndBit)
      return nullptr;

    StoreToWholeVariable = FragStartBit <= VarStartBit && FragEndBit >= *Size;
  }

  DIExpression *Expr =
      DIExpression::get(StoreLikeInst.getContext(), std::nullopt);
  if (!StoreToWholeVariable) {
    auto R = DIExpression::createFragmentExpression(Expr, FragStartBit,
                                                    FragEndBit - FragStartBit);
    assert(R.has_value() && "failed to create fragment expression");
    Expr = *R;
  }
  DIExpression *AddrExpr =
      DIExpression::get(StoreLikeInst.getContext(), std::nullopt);
  return DIB.insertDbgAssign(&StoreLikeInst, Val, VarRec.Var, Expr, Dest,
                             AddrExpr, VarRec.DL);
}

#undef DEBUG_TYPE // Silence redefinition warning (from ConstantsContext.h).
#define DEBUG_TYPE "assignment-tracking"

void at::trackAssignments(Function::iterator Start, Function::iterator End,
                          const StorageToVarsMap &Vars, const DataLayout &DL,
                          bool DebugPrints) {
  // Early-exit if there are no interesting variables.
  if (Vars.empty())
    return;

  auto &Ctx = Start->getContext();
  auto &Module = *Start->getModule();

  // Undef type doesn't matter, so long as it isn't void. Let's just use i1.
  auto *Undef = UndefValue::get(Type::getInt1Ty(Ctx));
  DIBuilder DIB(Module, /*AllowUnresolved*/ false);

  // Scan the instructions looking for stores to local variables' storage.
  LLVM_DEBUG(errs() << "# Scanning instructions\n");
  for (auto BBI = Start; BBI != End; ++BBI) {
    for (Instruction &I : *BBI) {

      std::optional<AssignmentInfo> Info;
      Value *ValueComponent = nullptr;
      Value *DestComponent = nullptr;
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        // We want to track the variable's stack home from its alloca's
        // position onwards so we treat it as an assignment (where the stored
        // value is Undef).
        Info = getAssignmentInfo(DL, AI);
        ValueComponent = Undef;
        DestComponent = AI;
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Info = getAssignmentInfo(DL, SI);
        ValueComponent = SI->getValueOperand();
        DestComponent = SI->getPointerOperand();
      } else if (auto *MI = dyn_cast<MemTransferInst>(&I)) {
        Info = getAssignmentInfo(DL, MI);
        // May not be able to represent this value easily.
        ValueComponent = Undef;
        DestComponent = MI->getOperand(0);
      } else if (auto *MI = dyn_cast<MemSetInst>(&I)) {
        Info = getAssignmentInfo(DL, MI);
        // If we're zero-initing we can state the assigned value is zero,
        // otherwise use undef.
        auto *ConstValue = dyn_cast<ConstantInt>(MI->getOperand(1));
        if (ConstValue && ConstValue->isZero())
          ValueComponent = ConstValue;
        else
          ValueComponent = Undef;
        DestComponent = MI->getOperand(0);
      } else {
        // Not a store-like instruction.
        continue;
      }

      assert(ValueComponent && DestComponent);
      LLVM_DEBUG(errs() << "SCAN: Found store-like: " << I << "\n");

      // Check if getAssignmentInfo failed to understand this store.
      if (!Info.has_value()) {
        LLVM_DEBUG(
            errs()
            << " | SKIP: Untrackable store (e.g. through non-const gep)\n");
        continue;
      }
      LLVM_DEBUG(errs() << " | BASE: " << *Info->Base << "\n");

      //  Check if the store destination is a local variable with debug info.
      auto LocalIt = Vars.find(Info->Base);
      if (LocalIt == Vars.end()) {
        LLVM_DEBUG(
            errs()
            << " | SKIP: Base address not associated with local variable\n");
        continue;
      }

      DIAssignID *ID =
          cast_or_null<DIAssignID>(I.getMetadata(LLVMContext::MD_DIAssignID));
      if (!ID) {
        ID = DIAssignID::getDistinct(Ctx);
        I.setMetadata(LLVMContext::MD_DIAssignID, ID);
      }

      for (const VarRecord &R : LocalIt->second) {
        auto *Assign =
            emitDbgAssign(*Info, ValueComponent, DestComponent, I, R, DIB);
        (void)Assign;
        LLVM_DEBUG(if (Assign) errs() << " > INSERT: " << *Assign << "\n");
      }
    }
  }
}

bool AssignmentTrackingPass::runOnFunction(Function &F) {
  // No value in assignment tracking without optimisations.
  if (F.hasFnAttribute(Attribute::OptimizeNone))
    return /*Changed*/ false;

  bool Changed = false;
  // Collect a map of {backing storage : dbg.declares} (currently "backing
  // storage" is limited to Allocas). We'll use this to find dbg.declares to
  // delete after running `trackAssignments`.
  DenseMap<const AllocaInst *, SmallPtrSet<DbgDeclareInst *, 2>> DbgDeclares;
  // Create another similar map of {storage : variables} that we'll pass to
  // trackAssignments.
  StorageToVarsMap Vars;
  for (auto &BB : F) {
    for (auto &I : BB) {
      DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(&I);
      if (!DDI)
        continue;
      // FIXME: trackAssignments doesn't let you specify any modifiers to the
      // variable (e.g. fragment) or location (e.g. offset), so we have to
      // leave dbg.declares with non-empty expressions in place.
      if (DDI->getExpression()->getNumElements() != 0)
        continue;
      if (!DDI->getAddress())
        continue;
      if (AllocaInst *Alloca =
              dyn_cast<AllocaInst>(DDI->getAddress()->stripPointerCasts())) {
        // FIXME: Skip VLAs for now (let these variables use dbg.declares).
        if (!Alloca->isStaticAlloca())
          continue;
        DbgDeclares[Alloca].insert(DDI);
        Vars[Alloca].insert(VarRecord(DDI));
      }
    }
  }

  auto DL = std::make_unique<DataLayout>(F.getParent());
  // FIXME: Locals can be backed by caller allocas (sret, byval).
  // Note: trackAssignments doesn't respect dbg.declare's IR positions (as it
  // doesn't "understand" dbg.declares). However, this doesn't appear to break
  // any rules given this description of dbg.declare from
  // llvm/docs/SourceLevelDebugging.rst:
  //
  //   It is not control-dependent, meaning that if a call to llvm.dbg.declare
  //   exists and has a valid location argument, that address is considered to
  //   be the true home of the variable across its entire lifetime.
  trackAssignments(F.begin(), F.end(), Vars, *DL);

  // Delete dbg.declares for variables now tracked with assignment tracking.
  for (auto &P : DbgDeclares) {
    const AllocaInst *Alloca = P.first;
    auto Markers = at::getAssignmentMarkers(Alloca);
    (void)Markers;
    for (DbgDeclareInst *DDI : P.second) {
      // Assert that the alloca that DDI uses is now linked to a dbg.assign
      // describing the same variable (i.e. check that this dbg.declare has
      // been replaced by a dbg.assign). Use DebugVariableAggregate to Discard
      // the fragment part because trackAssignments may alter the
      // fragment. e.g. if the alloca is smaller than the variable, then
      // trackAssignments will create an alloca-sized fragment for the
      // dbg.assign.
      assert(llvm::any_of(Markers, [DDI](DbgAssignIntrinsic *DAI) {
        return DebugVariableAggregate(DAI) == DebugVariableAggregate(DDI);
      }));
      // Delete DDI because the variable location is now tracked using
      // assignment tracking.
      DDI->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

static const char *AssignmentTrackingModuleFlag =
    "debug-info-assignment-tracking";

static void setAssignmentTrackingModuleFlag(Module &M) {
  M.setModuleFlag(Module::ModFlagBehavior::Max, AssignmentTrackingModuleFlag,
                  ConstantAsMetadata::get(
                      ConstantInt::get(Type::getInt1Ty(M.getContext()), 1)));
}

static bool getAssignmentTrackingModuleFlag(const Module &M) {
  Metadata *Value = M.getModuleFlag(AssignmentTrackingModuleFlag);
  return Value && !cast<ConstantAsMetadata>(Value)->getValue()->isZeroValue();
}

bool llvm::isAssignmentTrackingEnabled(const Module &M) {
  return getAssignmentTrackingModuleFlag(M);
}

PreservedAnalyses AssignmentTrackingPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  if (!runOnFunction(F))
    return PreservedAnalyses::all();

  // Record that this module uses assignment tracking. It doesn't matter that
  // some functons in the module may not use it - the debug info in those
  // functions will still be handled properly.
  setAssignmentTrackingModuleFlag(*F.getParent());

  // Q: Can we return a less conservative set than just CFGAnalyses? Can we
  // return PreservedAnalyses::all()?
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

PreservedAnalyses AssignmentTrackingPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  bool Changed = false;
  for (auto &F : M)
    Changed |= runOnFunction(F);

  if (!Changed)
    return PreservedAnalyses::all();

  // Record that this module uses assignment tracking.
  setAssignmentTrackingModuleFlag(M);

  // Q: Can we return a less conservative set than just CFGAnalyses? Can we
  // return PreservedAnalyses::all()?
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

#undef DEBUG_TYPE
