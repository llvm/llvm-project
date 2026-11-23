//===-- DebugLocConversion.cpp - Convert DILocation<->FLDebugLoc ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugLocConversion.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/FunctionLocalMetadata.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace llvm;

FLIndex<CompleteDebugLocation>
DebugLocationStorage::intern(Function *Fn, DILocation *DL,
                             DISubprogram *InlineeSP) {
  if (auto Result = DILMap.find({Fn, DL, InlineeSP}); Result != DILMap.end())
    return Result->second;
  FLIndex<CompleteDebugLocation> InlinedAt;
  if (DL->getInlinedAt()) {
    if (!isa<DILocation>(DL->getInlinedAt())) {
      errs() << "Invalid InlinedAt for DILocation\n";
      return {};
    }
    InlinedAt = intern(Fn, DL->getInlinedAt(), DL->getScope()->getSubprogram());
    if (!InlinedAt)
      return {};
  } else if (!Fn->getSubprogram()) {
    // If this is an end-of-inline-chain location in a nodebug function, we
    // create a CompleteDebugLocation representing that root.
    InlinedAt = DebugLocations.size();
    if (!isa<DILocalScope>(DL->getScope()))
      return {};
    FLIndex<CompleteDebugLocation> NewIdx = DebugLocations.size();
    DebugLocations.push_back(
        CompleteDebugLocation{Fn,
                              0,
                              0,
                              nullptr,
                              {},
                              false,
                              0,
                              0,
                              nullptr,
                              false,
                              DL->getScope()->getSubprogram()});
    DILocations.push_back(nullptr);
    DILMap.insert({{Fn, nullptr, DL->getScope()->getSubprogram()}, NewIdx});
  }
  CompleteDebugLocation CDL = {Fn,
                               DL->getLine(),
                               (uint16_t)DL->getColumn(),
                               DL->getScope(),
                               InlinedAt,
                               DL->isImplicitCode(),
                               DL->getAtomGroup(),
                               DL->getAtomRank(),
                               DL->getIRLayers(),
                               DL->isDistinct(),
                               InlineeSP};
  FLIndex<CompleteDebugLocation> NewIdx = DebugLocations.size();
  DebugLocations.push_back(CDL);
  DILocations.push_back(DL);
  DILMap.insert({{Fn, DL, InlineeSP}, NewIdx});
  return NewIdx;
}

FLIndex<CompleteDebugLocation> DebugLocationStorage::intern(Function *Fn,
                                                            FLDebugLoc DL) {
  if (auto Result = FLDLMap.find({Fn, DL}); Result != FLDLMap.end())
    return Result->second;
  // If we already have a stored context, use that; otherwise fetch it from the
  // Function.
  MDFunctionLocalContext *FLCtx = getFnContext(Fn);
  if (!FLCtx)
    FLCtx = Fn->getFLContext();
  if (!FLCtx || !DL.isValidInContext(FLCtx) || !DL) {
    if (!FLCtx)
      errs() << "dl-conversion error: missing fl context for function "
                "containing locs\n";
    else if (!DL.isValidInContext(FLCtx))
      errs() << "dl-conversion error: invalid debugloc\n";
    else
      errs() << "dl-conversion error: tried to intern empty debugloc\n";
    Fn->printAsOperand(errs());
    errs() << " ";
    DL.printAsOperand(errs());
    errs() << "\n";
    return {};
  }
  if (auto *FnSP = Fn->getSubprogram())
    SPContexts.insert({FnSP, FLCtx});
  else
    NodebugFunctionContexts.insert({Fn, FLCtx});
  FLIndex<CompleteDebugLocation> InlinedAt;
  if (DL.getInlinedAt(FLCtx)) {
    InlinedAt = intern(Fn, DL.getInlinedAt(FLCtx));
    if (!InlinedAt)
      return {};
  }
  uint32_t Line = 0;
  uint16_t Column = 0;
  DILocalScope *Scope = nullptr;
  if (!DL.isNoDebugInlinedCall(FLCtx)) {
    FLSrcLoc SrcLoc = DL.getSrcLoc(FLCtx);
    Line = SrcLoc.Line;
    Column = SrcLoc.Column;
    Scope = DL.getScope(FLCtx);
  }
  bool IsDistinct = DL.isInlinedCall() && !DL.asInlinedCall(FLCtx).Uniquable;
  DISubprogram *InlineeSP = nullptr;
  if (DL.isInlinedCall()) {
    auto *InlineeCtx = DL.asInlinedCall(FLCtx).InlineeContext;
    InlineeSP = cast<DISubprogram>(InlineeCtx->getScope(0).get());
    SPContexts.insert({InlineeSP, InlineeCtx});
  }
  CompleteDebugLocation CDL = {
      Fn,        Line,       Column,       Scope,
      InlinedAt, false,      DL.AtomGroup, (uint8_t)DL.AtomRank,
      nullptr,   IsDistinct, InlineeSP};
  FLIndex<CompleteDebugLocation> NewIdx = DebugLocations.size();
  DebugLocations.push_back(CDL);
  FLDebugLocs.push_back(DL);
  FLDLMap.insert({{Fn, DL}, NewIdx});
  return NewIdx;
}

/// Must be called in-order for entries in CompleteDebugLocation, i.e. must
/// only be called for DebugLocations[i] after calling for all DebugLocations
/// 0..i-1.
void DebugLocationStorage::convertImpl(FLIndex<CompleteDebugLocation> CDLIdx,
                                       FLDebugLoc &New) {
  if (CDLIdx.get() < FLDebugLocs.size()) {
    New = FLDebugLocs[CDLIdx.get()];
    return;
  }
  assert(CDLIdx.get() == FLDebugLocs.size() &&
         "FLDebugLocs being created out-of-order?");
  CompleteDebugLocation CDL = DebugLocations[CDLIdx.get()];
  assert((!CDL.InlinedAt || CDL.InlinedAt.get() < CDLIdx.get()) &&
         "Invalid circular inline chain?");

  // First: ensure a context for the concrete function containing this CDL.
  MDFunctionLocalContext *FnContext;
  DISubprogram *FnSP = CDL.Fn->getSubprogram();
  if (FnSP) {
    auto SPCtxIt = SPContexts.try_emplace(FnSP, nullptr).first;
    if (!SPCtxIt->second) {
      auto *NewCtx = MDFunctionLocalContext::getDistinct(CDL.Fn->getContext());
      SPCtxIt->second = NewCtx;
    }
    FnContext = SPCtxIt->second;
  } else {
    auto FnCtxIt = NodebugFunctionContexts.try_emplace(CDL.Fn, nullptr).first;
    if (!FnCtxIt->second)
      FnCtxIt->second =
          MDFunctionLocalContext::getDistinct(CDL.Fn->getContext(), true);
    FnContext = FnCtxIt->second;
  }
  assert(FnContext && "Failed to find/create a context?");

  // Then, ensure an InlineeContext if necessary.
  if (CDL.InlineeSP) {
    auto SPCtxIt = SPContexts.try_emplace(CDL.InlineeSP, nullptr).first;
    if (!SPCtxIt->second) {
      MDFunctionLocalContext *NewCtx =
          MDFunctionLocalContext::getDistinct(CDL.Fn->getContext());
      SPCtxIt->second = NewCtx;
    }
  }

  // Create the SrcLoc associated with this DebugLocation, which should exist
  // in the SPContext for its non-inlined subprogram scope (unless it has no
  // scope, in which case it is a nodebug inlined call).
  // Now we can perform the actual creation itself.
  FLIndex<FLSrcLoc> SrcLocIdx;
  if (CDL.Scope) {
    auto *SrcCtx = SPContexts.find(CDL.Scope->getSubprogram())->second;
    auto BuilderIt = Builders.try_emplace(SrcCtx, nullptr).first;
    if (!BuilderIt->second)
      BuilderIt->second.reset(
          new FLMDBuilder(SrcCtx, CDL.Scope->getSubprogram()));
    FLMDBuilder *SrcBuilder = BuilderIt->second.get();
    SrcLocIdx = SrcBuilder->getSrcLocWithScope(CDL.Line, CDL.Column, CDL.Scope);
  } else {
    assert(CDL.InlineeSP && "nodebug inlined call without InlineeSP?");
  }

  // If this is an inlined call, then we actually want to create an
  // FLInlinedCall in the function context, and return an InlinedCall loc
  // pointing at that; otherwise, return the FLDebugLoc as constructed.
  FLIndex<FLInlinedCall> InlinedAtIdx =
      CDL.InlinedAt ? FLDebugLocs[CDL.InlinedAt.get()].InlinedAtIdx
                    : FLIndex<FLInlinedCall>();
  if (CDL.InlineeSP) {
    MDFunctionLocalContext *InlineeCtx = SPContexts.find(CDL.InlineeSP)->second;
    FLInlinedCall IC = {SrcLocIdx, InlinedAtIdx, InlineeCtx, !CDL.IsDistinct};
    FLIndex<FLInlinedCall> ICIdx = FnContext->insertInlinedCall(IC);
    New = FLDebugLoc(ICIdx);
  } else {
    // We ignore wrapping atom groups, since at worst we end up affecting
    // is_stmt places a small amount.
    FnContext->updateAtomGroupWaterline(InlinedAtIdx, (uint16_t)CDL.AtomGroup);
    New = FLDebugLoc(SrcLocIdx, InlinedAtIdx, (uint16_t)CDL.AtomGroup,
                     CDL.AtomRank);
  }
  assert(New && "Failed to create valid FLDebugLoc?");
}

void DebugLocationStorage::convertImpl(FLIndex<CompleteDebugLocation> CDLIdx,
                                       DILocation *&New) {
  if (CDLIdx.get() < DILocations.size()) {
    New = DILocations[CDLIdx.get()];
    return;
  }
  assert(CDLIdx.get() == DILocations.size() &&
         "DILocations being created out-of-order?");
  CompleteDebugLocation CDL = DebugLocations[CDLIdx.get()];
  assert((!CDL.InlinedAt || CDL.InlinedAt.get() < CDLIdx.get()) &&
         "Invalid circular inline chain?");

  if (!CDL.Scope) {
    // The outermost location in an inline chain into a nodebug function
    // is not represented as a DILocation; representing these as nullptr
    // gives correct behaviour.
    New = nullptr;
    return;
  }
  DILocation *InlinedAt =
      CDL.InlinedAt ? DILocations[CDL.InlinedAt.get()] : nullptr;
  if (CDL.IsDistinct) {
    New = DILocation::getDistinct(CDL.Scope->getContext(), CDL.Line, CDL.Column,
                                  CDL.Scope, InlinedAt, CDL.IsImplicitCode,
                                  CDL.AtomGroup, CDL.AtomRank, CDL.IRLayers);
  } else {
    New = DILocation::get(CDL.Scope->getContext(), CDL.Line, CDL.Column,
                          CDL.Scope, InlinedAt, CDL.IsImplicitCode,
                          CDL.AtomGroup, CDL.AtomRank, CDL.IRLayers);
  }
}

// Call after conversions, before any changes might be made to the IR.
void DebugLocationStorage::flush() { Builders.clear(); }

/// Returns the FLContext that, according to this storage, should be attached
/// to Fn.
/// This function will not create any new contexts even if they should exist,
/// so this should be used only after performing conversions.
MDFunctionLocalContext *DebugLocationStorage::getFnContext(Function *Fn) const {
  if (auto *FnSP = Fn->getSubprogram())
    return SPContexts.lookup(FnSP);
  return NodebugFunctionContexts.lookup(Fn);
}

bool DebugLocationStorage::convert(Function *Fn, DILocation *DIL,
                                   FLDebugLoc &FLDL) {
  auto Idx = intern(Fn, DIL);
  if (!Idx)
    return true;
  // If Idx has an existing conversion then reuse it.
  if (Idx.get() < FLDebugLocs.size()) {
    FLDL = FLDebugLocs[Idx.get()];
  } else {
    // Otherwise, convert all interned locations, which may be more than just
    // DIL itself, culminating in the newest location, which should be at the
    // back of DebugLocations.
    assert(Idx.get() == DebugLocations.size() - 1 &&
           "New location is not the newest location?");
    for (size_t NewIdx = FLDebugLocs.size(); NewIdx < DebugLocations.size();
         ++NewIdx) {
      convertImpl(NewIdx, FLDL);
      FLDebugLocs.push_back(FLDL);
      FLDLMap.insert({{Fn, FLDL}, NewIdx});
    }
  }
  assert(DebugLocations.size() == DILocations.size() &&
         DebugLocations.size() == FLDebugLocs.size() &&
         "Storage has become imbalanced?");
  return false;
}
bool DebugLocationStorage::convert(Function *Fn, FLDebugLoc FLDL,
                                   DILocation *&DIL) {
  assert(FLDL.isInstrLoc() &&
         "Only expected InstrLocs to be converted directly");
  auto Idx = intern(Fn, FLDL);
  if (!Idx)
    return true;
  // If Idx has an existing conversion then reuse it.
  if (Idx.get() < DILocations.size()) {
    DIL = DILocations[Idx.get()];
  } else {
    // Otherwise, convert all interned locations, which may be more than just
    // FLDL itself, culminating in the newest location, which should be at the
    // back of DebugLocations.
    assert(Idx.get() == DebugLocations.size() - 1 &&
           "New location is not the newest location?");
    for (size_t NewIdx = DILocations.size(); NewIdx < DebugLocations.size();
         ++NewIdx) {
      convertImpl(NewIdx, DIL);
      DILocations.push_back(DIL);
      DILMap.insert({{Fn, DIL, DebugLocations[NewIdx].InlineeSP}, NewIdx});
    }
  }
  assert(DebugLocations.size() == DILocations.size() &&
         DebugLocations.size() == FLDebugLocs.size() &&
         "Storage has become imbalanced?");
  return false;
}

MDNode *DebugLocConversionInfo::getNodeFromLoc(Function *Fn, FLDebugLoc Loc) {
  return DIDbgLoc::get(Fn->getContext(), Fn->getFLContext(), Loc);
}
MDNode *DebugLocConversionInfo::getNodeFromLoc(Function *Fn, DILocation *Loc) {
  return Loc;
}

template <> FLDebugLoc DebugLocConversionInfo::getLocFromNode(const MDNode *N) {
  if (const DIDbgLoc *DL = dyn_cast<DIDbgLoc>(N))
    return DL->getFLDebugLoc();
  return {};
}
template <>
DILocation *DebugLocConversionInfo::getLocFromNode(const MDNode *N) {
  if (const DILocation *DL = dyn_cast<DILocation>(N))
    return const_cast<DILocation *>(DL);
  return {};
}

/// We have to define the constructor/destructor for the conversion class here,
/// instead of in the header, since the unique_ptr for the storage class
/// requires the class definition for its ctor/dtor.
template <typename From, typename To>
DebugLocConversionBase<From, To>::DebugLocConversionBase(Module *M)
    : M(M), Storage(M->getContext().getModuleDLConversionStorage(M)) {}
template <typename From, typename To>
DebugLocConversionBase<From, To>::~DebugLocConversionBase() {
  Storage->flush();
}

template <typename From, typename To>
void DebugLocConversionBase<From, To>::flush() {
  Attachments.clear();
  AttachmentMap.clear();
  NodeMap.clear();
  LoopInstrs.clear();
  Storage->flush();
}
template <typename From, typename To>
bool DebugLocConversionBase<From, To>::convertRecordedLocs() {
  for (auto [Fn, AttachTo, LocSetter, Loc] : Attachments) {
    To NewLoc;
    if (Storage->convert(Fn, Loc, NewLoc))
      return true;
    AttachmentMap.insert({AttachTo, NewLoc});
  }
  for (auto [I, LoopMD] : LoopInstrs) {
    Function *Fn = I->getFunction();
    SmallPtrSet<MDNode *, 4> VisitedNodes;
    SmallVector<MDNode *> ToVisit;
    ToVisit.push_back(LoopMD);
    while (!ToVisit.empty()) {
      MDNode *Next = ToVisit.pop_back_val();
      if (!VisitedNodes.insert(Next).second)
        continue;
      if (From LocOpt = DLInfo::getLocFromNode<From>(Next)) {
        To NewLoc;
        if (Storage->convert(Fn, LocOpt, NewLoc))
          return true;
        NodeMap.insert({{Fn, Next}, NewLoc});
        continue;
      }
      for (auto &Op : Next->operands())
        if (auto *N = dyn_cast_if_present<MDNode>(Op);
            N && !VisitedNodes.contains(N))
          ToVisit.push_back(N);
    }
  }
  return false;
}
template <typename From, typename To>
void DebugLocConversionBase<From, To>::replaceLoopNodes() {
  assert(Storage && "Trying to replace loop nodes pre-conversion.");
  for (auto [I, LoopMD] : LoopInstrs) {
    Function *Fn = I->getFunction();
    auto ExistingIt = RemappedLoopNodes.try_emplace({Fn, LoopMD}, nullptr).first;
    if (ExistingIt->second) {
      I->setMetadata(LLVMContext::MD_loop, ExistingIt->second);
      continue;
    }
    updateLoopMetadataDebugLocations(*I, [&](Metadata *M) -> Metadata * {
      if (auto *N = dyn_cast_if_present<MDNode>(M)) {
        if (auto ReplaceIt = NodeMap.find({Fn, N});
            ReplaceIt != NodeMap.end()) {
          return DLInfo::getNodeFromLoc(Fn, ReplaceIt->second);
        }
      }
      return M;
    });
    ExistingIt->second = I->getMetadata(LLVMContext::MD_loop);
  }
}
template <typename From, typename To>
To DebugLocConversionBase<From, To>::getDebugLoc(const void *AttachedTo) const {
  auto Result = AttachmentMap.find(AttachedTo);
  if (Result == AttachmentMap.end())
    return {};
  return Result->second;
}

namespace llvm {
template class DebugLocConversionBase<DebugLocConversionInfo::NativeLocTy,
                                      DebugLocConversionInfo::ForeignLocTy>;
template class DebugLocConversionBase<DebugLocConversionInfo::ForeignLocTy,
                                      DebugLocConversionInfo::NativeLocTy>;
} // namespace llvm

// Helper to conditionally filter FunctionContext based on the direction we're
// converting in.
template <typename To> MDFunctionLocalContext *filter(MDFunctionLocalContext *);
template <>
MDFunctionLocalContext *filter<DILocation *>(MDFunctionLocalContext *) {
  return nullptr;
}
template <>
MDFunctionLocalContext *filter<FLDebugLoc>(MDFunctionLocalContext *Ctx) {
  return Ctx;
}

DebugLocWriteConversionMap::DebugLocWriteConversionMap(Module *M)
    : DebugLocConversionBase(M) {
  for (auto &F : *M) {
    OldFnFLMDAttachments.emplace_back(&F, F.getMetadata(LLVMContext::MD_flmd));
    for (auto &BB : F) {
      for (auto &I : BB) {
        for (auto &DR : I.getDbgRecordRange())
          addAttachment(&F, &DR, DR.getDebugLoc());
        if (DLInfo::NativeLocTy Loc = I.getDebugLoc())
          addAttachment(&F, &I, Loc);
        if (MDNode *LoopMD = I.getMetadata(LLVMContext::MD_loop))
          addLoopInstr(&I, LoopMD);
      }
    }
  }
  // Convert to new debug locations and remove/replace existing
  // attachments/nodes.
  bool Failed = convertRecordedLocs();
  assert(!Failed && "Failed to convert debug locations for write");
  (void)Failed;
  for (auto &F : *M)
    if (auto *NewCtx = Storage->getFnContext(&F))
      F.setMetadata(LLVMContext::MD_flmd,
                    filter<DebugLocConversionInfo::ForeignLocTy>(NewCtx));
  for (auto [Fn, AttachedTo, Attach, Loc] : Attachments)
    Attach(AttachedTo, {});
  replaceLoopNodes();
  M->renumberMetadataForAssembly();
}
DebugLocWriteConversionMap::~DebugLocWriteConversionMap() {
  // Restore the original module.
  for (auto [I, LoopMD] : LoopInstrs)
    I->setMetadata(LLVMContext::MD_loop, LoopMD);
  for (auto [Fn, AttachedTo, Attach, Loc] : Attachments)
    Attach(AttachedTo, Loc);
  for (auto [F, MD] : OldFnFLMDAttachments)
    F->setMetadata(LLVMContext::MD_flmd, MD);
}

DebugLocConversionInfo::ForeignLocTy
DebugLocWriteConversionMap::getExtraDebugLoc(Function *Fn, void *AttachedTo,
                                             DLInfo::NativeLocTy Loc) {
  auto ExistingResult = AttachmentMap.try_emplace(AttachedTo);
  if (!ExistingResult.second)
    return ExistingResult.first->second;
  DLInfo::ForeignLocTy ConvLoc;
  bool Failed = Storage->convert(Fn, Loc, ConvLoc);
  assert(!Failed && "Failed to convert debug locations for write");
  (void)Failed;
  ExistingResult.first->second = ConvLoc;
  return ConvLoc;
}

DebugLocReadConversionMap::DebugLocReadConversionMap(Module *M)
    : DebugLocConversionBase(M) {}

bool DebugLocReadConversionMap::resolveReferences() {
  if (convertRecordedLocs()) {
    flush();
    return true;
  }
  SmallVector<Function *> FnsToConvert;
  for (auto &F : *M)
    F.setMetadata(
        LLVMContext::MD_flmd,
        filter<DebugLocConversionInfo::NativeLocTy>(Storage->getFnContext(&F)));
  for (auto [Fn, AttachedTo, Attach, Loc] : Attachments)
    Attach(AttachedTo, AttachmentMap.find(AttachedTo)->second);
  replaceLoopNodes();
  flush();
  return false;
}

MDFunctionLocalContext *
DebugLocReadConversionMap::getFnContext(Function *Fn) const {
  return Storage->getFnContext(Fn);
}
