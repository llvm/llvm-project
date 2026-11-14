//===- MetadataUtils.cpp - Metadata utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MetadataUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MemCacheHint.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ProfDataUtils.h"

using namespace llvm;

namespace {
struct MemoryAccessInfo {
  Type *Ty;
  unsigned PointerOperand;
  bool IsRead;
};
} // namespace

/// Get information about a memory instruction required to propagate metadata.
/// Returns std::nullopt if the instruction is not supported and we will not
/// propagate any metadata.
///
/// If IsRead is set and this is a instruction that reads and writes, mark it
/// as reading or writing. This is used to make the source instruction in
/// metadata propagation correspond with the dest.
static std::optional<MemoryAccessInfo>
getMemoryAccessInfo(const Instruction &I,
                    std::optional<bool> IsRead = std::nullopt) {
  if (const auto *LI = dyn_cast<LoadInst>(&I))
    return MemoryAccessInfo{LI->getType(), 0, true};
  if (const auto *SI = dyn_cast<StoreInst>(&I))
    return MemoryAccessInfo{SI->getValueOperand()->getType(), 1, false};
  if (const auto *MI = dyn_cast<AnyMemIntrinsic>(&I)) {
    if (!IsRead || (*IsRead && !isa<AnyMemTransferInst>(MI)))
      return std::nullopt;
    return MemoryAccessInfo{nullptr, *IsRead ? 1u : 0u, *IsRead};
  }
  const auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return std::nullopt;
  if (const auto *VP = dyn_cast<VPIntrinsic>(II)) {
    auto Ptr = VPIntrinsic::getMemoryPointerParamPos(VP->getIntrinsicID());
    if (!Ptr)
      return std::nullopt;
    if (auto Data = VPIntrinsic::getMemoryDataParamPos(VP->getIntrinsicID()))
      return MemoryAccessInfo{VP->getArgOperand(*Data)->getType(), *Ptr, false};
    Type *Ty = VP->getType();
    if (VP->getIntrinsicID() == Intrinsic::vp_load_ff)
      Ty = Ty->getStructElementType(0);
    return MemoryAccessInfo{Ty, *Ptr, true};
  }
  switch (II->getIntrinsicID()) {
  case Intrinsic::experimental_vector_histogram_add:
  case Intrinsic::experimental_vector_histogram_uadd_sat:
  case Intrinsic::experimental_vector_histogram_umin:
  case Intrinsic::experimental_vector_histogram_umax:
    if (IsRead)
      return MemoryAccessInfo{II->getArgOperand(1)->getType(), 0, *IsRead};
    return std::nullopt;
  case Intrinsic::masked_load:
  case Intrinsic::masked_gather:
  case Intrinsic::masked_expandload:
    return MemoryAccessInfo{II->getType(), 0, true};
  case Intrinsic::masked_store:
  case Intrinsic::masked_scatter:
  case Intrinsic::masked_compressstore:
    return MemoryAccessInfo{II->getArgOperand(0)->getType(), 1, false};
  default:
    return std::nullopt;
  }
}

/// Get the pointer operand of `I` at operand index `Index`.
static const Value *getMemoryPointer(const Instruction &I, unsigned Index) {
  if (const auto *CB = dyn_cast<CallBase>(&I))
    return CB->getArgOperand(Index);
  return I.getOperand(Index);
}

// invariant.group ignores bitcasts and zero-offset GEPs, not address-space
// casts.
static const Value *getInvariantGroupPointer(const Value *V) {
  while (true) {
    if (const auto *Cast = dyn_cast<BitCastOperator>(V))
      V = Cast->getOperand(0);
    else if (const auto *GEP = dyn_cast<GEPOperator>(V);
             GEP && GEP->hasAllZeroIndices())
      V = GEP->getPointerOperand();
    else
      return V;
  }
}

static void copyNonnullMetadata(const LoadInst &OldLI, MDNode *N,
                                LoadInst &NewLI) {
  auto *OldTy = cast<PointerType>(OldLI.getType());
  auto *NewTy = NewLI.getType();

  // Preserve the pointer fact when the pointer type is unchanged.
  if (NewTy == OldTy) {
    NewLI.setMetadata(LLVMContext::MD_nonnull, N);
    return;
  }

  // An integer range must describe the complete pointer representation.
  if (!NewTy->isIntegerTy() ||
      NewTy->getIntegerBitWidth() !=
          OldLI.getDataLayout().getPointerTypeSizeInBits(OldTy))
    return;

  MDBuilder MDB(NewLI.getContext());
  auto *ITy = cast<IntegerType>(NewTy);
  auto *NullInt =
      ConstantExpr::getPtrToInt(ConstantPointerNull::get(OldTy), ITy);
  auto *NonNullInt = ConstantExpr::getAdd(NullInt, ConstantInt::get(ITy, 1));
  NewLI.setMetadata(LLVMContext::MD_range,
                    MDB.createRange(NonNullInt, NullInt));
}

static void copyRangeMetadata(const DataLayout &DL, const LoadInst &OldLI,
                              MDNode *N, LoadInst &NewLI) {
  auto *NewTy = NewLI.getType();
  // Simply copy the metadata if the type did not change.
  if (NewTy == OldLI.getType()) {
    NewLI.setMetadata(LLVMContext::MD_range, N);
    return;
  }

  // Give up unless it is converted to a pointer where there is a single very
  // valuable mapping we can do reliably.
  // FIXME: It would be nice to propagate this in more ways, but the type
  // conversions make it hard.
  if (!NewTy->isPointerTy())
    return;

  unsigned BitWidth = DL.getPointerTypeSizeInBits(NewTy);
  if (BitWidth == OldLI.getType()->getScalarSizeInBits() &&
      !getConstantRangeFromMetadata(*N).contains(APInt(BitWidth, 0))) {
    MDNode *NN = MDNode::get(OldLI.getContext(), {});
    NewLI.setMetadata(LLVMContext::MD_nonnull, NN);
  }
}

void llvm::copyMetadataForMemoryAccess(Instruction &Dest,
                                       const Instruction &Source,
                                       std::optional<int64_t> DestByteOffset) {
  auto DestInfo = getMemoryAccessInfo(Dest);
  // Untyped memory intrinsics are supported only as sources.
  assert(DestInfo && DestInfo->Ty &&
         "expected a typed destination memory access");
  auto SourceInfo = getMemoryAccessInfo(Source, DestInfo->IsRead);
  assert(SourceInfo && SourceInfo->IsRead == DestInfo->IsRead &&
         "expected corresponding memory reads or writes");
  const DataLayout &DL = Source.getDataLayout();
  // Collect the size of the old memory operation if available.
  std::optional<TypeSize> OldSize;
  if (SourceInfo->Ty)
    OldSize = DL.getTypeStoreSize(SourceInfo->Ty);
  else if (auto *Length =
               dyn_cast<ConstantInt>(cast<AnyMemIntrinsic>(Source).getLength()))
    // If Source is not a typed memory operation it must be an intrinsic.
    if (auto Bytes = Length->getValue().tryZExtValue())
      OldSize = TypeSize::getFixed(*Bytes);
  TypeSize NewSize = DL.getTypeStoreSize(DestInfo->Ty);
  // Is the memory `Dest` accesses a subset of the memory `Source` acceses?
  // The caller guarentees this is true if we don't know the size of `Source`
  // or the offset of the `Dest` access in `Source`.
  bool IsSubset = !OldSize || TypeSize::isKnownLE(NewSize, *OldSize);
  if (DestByteOffset)
    IsSubset = IsSubset && *DestByteOffset >= 0 &&
               (!OldSize ||
                uint64_t(*DestByteOffset) <=
                    OldSize->getKnownMinValue() - NewSize.getKnownMinValue());
  bool IsWholeAccess = IsSubset && DestByteOffset == 0 && OldSize == NewSize;
  bool SameElementType = SourceInfo->Ty && SourceInfo->Ty->getScalarType() ==
                                               DestInfo->Ty->getScalarType();
  bool SamePointer = getInvariantGroupPointer(getMemoryPointer(
                         Source, SourceInfo->PointerOperand)) ==
                     getInvariantGroupPointer(
                         getMemoryPointer(Dest, DestInfo->PointerOperand));
  
  // Now loop over all `Source` metadata, propagating what we can.
  SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
  Source.getAllMetadata(MDs);
  for (auto [Kind, MD] : MDs) {
    bool Keep = false;
    switch (Kind) {
    case LLVMContext::MD_dbg:
    case LLVMContext::MD_nontemporal:
    case LLVMContext::MD_annotation:
    case LLVMContext::MD_nosanitize:
      Keep = true;
      break;
    case LLVMContext::MD_mmra:
      Keep = canInstructionHaveMMRAs(Dest);
      break;
    case LLVMContext::MD_access_group:
    case LLVMContext::MD_mem_parallel_loop_access:
      Keep = IsSubset;
      break;
    case LLVMContext::MD_invariant_load:
      Keep = IsSubset && DestInfo->IsRead;
      break;
    case LLVMContext::MD_invariant_group:
      Keep = IsSubset && SamePointer && !isa<CallBase>(Dest);
      break;
    case LLVMContext::MD_noundef:
      // The value loaded remains defined if its element type is unchanged,
      // or every source byte consists of defined value bits (i.e. the source
      // doesn't have padding).
      Keep = IsSubset && isa<LoadInst>(Dest) &&
             (SameElementType ||
              (SourceInfo->Ty && !SourceInfo->Ty->isAggregateType() &&
               DL.typeSizeEqualsStoreSize(SourceInfo->Ty)));
      break;
    case LLVMContext::MD_fpmath:
    case LLVMContext::MD_nofpclass:
      Keep =
          IsSubset && SameElementType &&
          (Kind == LLVMContext::MD_fpmath ? Dest.getType()->isFPOrFPVectorTy()
                                          : isa<LoadInst>(Dest));
      break;
    case LLVMContext::MD_nonnull:
      if (IsWholeAccess && isa<LoadInst>(Source) && isa<LoadInst>(Dest))
        copyNonnullMetadata(cast<LoadInst>(Source), MD, cast<LoadInst>(Dest));
      break;
    case LLVMContext::MD_range:
      // Ranges apply per integer element. Other type changes require the
      // helper's explicit conversion from an integer range to nonnull.
      if (IsSubset && SameElementType && DestInfo->Ty->isIntOrIntVectorTy())
        Keep = true;
      else if (IsWholeAccess && isa<LoadInst>(Source) && isa<LoadInst>(Dest))
        copyRangeMetadata(DL, cast<LoadInst>(Source), MD, cast<LoadInst>(Dest));
      break;
    case LLVMContext::MD_align:
    case LLVMContext::MD_dereferenceable:
    case LLVMContext::MD_dereferenceable_or_null:
      // These describe the loaded pointer value.
      Keep = IsWholeAccess && SourceInfo->Ty == DestInfo->Ty &&
             DestInfo->Ty->isPointerTy() && isa<LoadInst>(Dest);
      break;
    case LLVMContext::MD_DIAssignID:
      Keep = IsWholeAccess && isa<StoreInst>(Source) && isa<StoreInst>(Dest);
      break;
    case LLVMContext::MD_prof:
      Keep =
          IsWholeAccess && ((isa<LoadInst>(Source) && isa<LoadInst>(Dest)) ||
                            (isa<StoreInst>(Source) && isa<StoreInst>(Dest)));
      break;
    case LLVMContext::MD_captures:
      Keep =
          IsWholeAccess && isa<StoreInst>(Dest) && DestInfo->Ty->isPointerTy();
      break;
    default:
      break;
    }
    if (Keep)
      Dest.setMetadata(Kind, MD);
  }

  // Adjust aliasing metadata.
  AAMDNodes AA = Source.getAAMetadata();
  if (!IsSubset) {
    AA.TBAA = nullptr;
    AA.TBAAStruct = nullptr;
    AA.Scope = nullptr;
    AA.NoAlias = nullptr;
  } else if (!IsWholeAccess) {
    if (DestByteOffset)
      AA = AA.adjustForAccess(*DestByteOffset, DestInfo->Ty, DL);
    else
      AA.TBAAStruct = nullptr;
  } else {
    // Prefer !tbaa over !tbaa.struct.
    AAMDNodes Adjusted = AA.adjustForAccess(0, DestInfo->Ty, DL);
    if (!AA.TBAA && Adjusted.TBAA)
      AA = Adjusted;
  }
  Dest.setAAMetadata(AA);
  setMemCacheHint(Dest, DestInfo->PointerOperand,
                  getMemCacheHint(Source, SourceInfo->PointerOperand));
}

void llvm::copyMetadataForScalarizedInstruction(
    Instruction &Dest, const Instruction &Source,
    std::optional<int64_t> DestByteOffset) {
  if (auto DestInfo = getMemoryAccessInfo(Dest))
    if (getMemoryAccessInfo(Source, DestInfo->IsRead)) {
      copyMetadataForMemoryAccess(Dest, Source, DestByteOffset);
      return;
    }
  if (Dest.getType()->getScalarType() == Source.getType()->getScalarType() &&
      Dest.getType()->isFPOrFPVectorTy())
    Dest.copyMetadata(Source, {LLVMContext::MD_fpmath});
  if (!Dest.getDebugLoc())
    Dest.setDebugLoc(Source.getDebugLoc());
}

/// Add all access groups in @p AccGroups to @p List.
template <typename ListT>
static void addToAccessGroupList(ListT &List, MDNode *AccGroups) {
  // Interpret an access group as a list containing itself.
  if (AccGroups->getNumOperands() == 0) {
    assert(isValidAsAccessGroup(AccGroups) && "Node must be an access group");
    List.insert(AccGroups);
    return;
  }

  for (const auto &AccGroupListOp : AccGroups->operands()) {
    auto *Item = cast<MDNode>(AccGroupListOp.get());
    assert(isValidAsAccessGroup(Item) && "List item must be an access group");
    List.insert(Item);
  }
}

MDNode *llvm::uniteAccessGroups(MDNode *AccGroups1, MDNode *AccGroups2) {
  if (!AccGroups1)
    return AccGroups2;
  if (!AccGroups2)
    return AccGroups1;
  if (AccGroups1 == AccGroups2)
    return AccGroups1;

  SmallSetVector<Metadata *, 4> Union;
  addToAccessGroupList(Union, AccGroups1);
  addToAccessGroupList(Union, AccGroups2);

  if (Union.size() == 0)
    return nullptr;
  if (Union.size() == 1)
    return cast<MDNode>(Union.front());

  LLVMContext &Ctx = AccGroups1->getContext();
  return MDNode::get(Ctx, Union.getArrayRef());
}

MDNode *llvm::intersectAccessGroups(const Instruction *Inst1,
                                    const Instruction *Inst2) {
  bool MayAccessMem1 = Inst1->mayReadOrWriteMemory();
  bool MayAccessMem2 = Inst2->mayReadOrWriteMemory();

  if (!MayAccessMem1 && !MayAccessMem2)
    return nullptr;
  if (!MayAccessMem1)
    return Inst2->getMetadata(LLVMContext::MD_access_group);
  if (!MayAccessMem2)
    return Inst1->getMetadata(LLVMContext::MD_access_group);

  MDNode *MD1 = Inst1->getMetadata(LLVMContext::MD_access_group);
  MDNode *MD2 = Inst2->getMetadata(LLVMContext::MD_access_group);
  if (!MD1 || !MD2)
    return nullptr;
  if (MD1 == MD2)
    return MD1;

  // Use set for scalable 'contains' check.
  SmallPtrSet<Metadata *, 4> AccGroupSet2;
  addToAccessGroupList(AccGroupSet2, MD2);

  SmallVector<Metadata *, 4> Intersection;
  if (MD1->getNumOperands() == 0) {
    assert(isValidAsAccessGroup(MD1) && "Node must be an access group");
    if (AccGroupSet2.count(MD1))
      Intersection.push_back(MD1);
  } else {
    for (const MDOperand &Node : MD1->operands()) {
      auto *Item = cast<MDNode>(Node.get());
      assert(isValidAsAccessGroup(Item) && "List item must be an access group");
      if (AccGroupSet2.count(Item))
        Intersection.push_back(Item);
    }
  }

  if (Intersection.size() == 0)
    return nullptr;
  if (Intersection.size() == 1)
    return cast<MDNode>(Intersection.front());

  LLVMContext &Ctx = Inst1->getContext();
  return MDNode::get(Ctx, Intersection);
}

/// Add metadata from \p Inst to \p Metadata, if it can be preserved after
/// vectorization.
void llvm::getMetadataToPropagate(
    Instruction *Inst,
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Metadata) {
  Inst->getAllMetadataOtherThanDebugLoc(Metadata);
  static const unsigned SupportedIDs[] = {
      LLVMContext::MD_tbaa,         LLVMContext::MD_alias_scope,
      LLVMContext::MD_noalias,      LLVMContext::MD_fpmath,
      LLVMContext::MD_nontemporal,  LLVMContext::MD_invariant_load,
      LLVMContext::MD_access_group, LLVMContext::MD_mmra};

  // Remove any unsupported metadata kinds from Metadata.
  for (unsigned Idx = 0; Idx != Metadata.size();) {
    if (is_contained(SupportedIDs, Metadata[Idx].first)) {
      ++Idx;
    } else {
      // Swap element to end and remove it.
      std::swap(Metadata[Idx], Metadata.back());
      Metadata.pop_back();
    }
  }
}

/// \returns \p I after propagating metadata from \p VL.
Instruction *llvm::propagateMetadata(Instruction *Inst, ArrayRef<Value *> VL) {
  if (VL.empty())
    return Inst;

  // Keep cache hints out of getMetadataToPropagate(): VPlan changes operand
  // layouts without remapping cache-hint indices.
  propagateMemCacheHint(*Inst, VL);

  SmallVector<std::pair<unsigned, MDNode *>> Metadata;
  getMetadataToPropagate(cast<Instruction>(VL[0]), Metadata);

  for (auto &[Kind, MD] : Metadata) {
    // Skip MMRA metadata if the instruction cannot have it.
    if (Kind == LLVMContext::MD_mmra && !canInstructionHaveMMRAs(*Inst))
      continue;

    for (int J = 1, E = VL.size(); MD && J != E; ++J) {
      const Instruction *IJ = cast<Instruction>(VL[J]);
      MDNode *IMD = IJ->getMetadata(Kind);

      switch (Kind) {
      case LLVMContext::MD_mmra: {
        MD = MMRAMetadata::combine(Inst->getContext(), MD, IMD);
        break;
      }
      case LLVMContext::MD_tbaa:
        MD = MDNode::getMostGenericTBAA(MD, IMD);
        break;
      case LLVMContext::MD_alias_scope:
        MD = MDNode::getMostGenericAliasScope(MD, IMD);
        break;
      case LLVMContext::MD_fpmath:
        MD = MDNode::getMostGenericFPMath(MD, IMD);
        break;
      case LLVMContext::MD_noalias:
      case LLVMContext::MD_nontemporal:
      case LLVMContext::MD_invariant_load:
        MD = MDNode::intersect(MD, IMD);
        break;
      case LLVMContext::MD_access_group:
        MD = intersectAccessGroups(Inst, IJ);
        break;
      default:
        llvm_unreachable("unhandled metadata");
      }
    }

    Inst->setMetadata(Kind, MD);
  }

  return Inst;
}

/// If AAOnly is set, only intersect alias analysis metadata and preserve other
/// known metadata. Unknown metadata is always dropped.
static void combineMetadata(Instruction *K, const Instruction *J,
                            bool DoesKMove, bool AAOnly = false) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> Metadata;
  K->getAllMetadataOtherThanDebugLoc(Metadata);
  for (const auto &MD : Metadata) {
    unsigned Kind = MD.first;
    MDNode *JMD = J->getMetadata(Kind);
    MDNode *KMD = MD.second;

    // TODO: Assert that this switch is exhaustive for fixed MD kinds.
    switch (Kind) {
    default:
      K->setMetadata(Kind, nullptr); // Remove unknown metadata
      break;
    case LLVMContext::MD_dbg:
      llvm_unreachable("getAllMetadataOtherThanDebugLoc returned a MD_dbg");
    case LLVMContext::MD_DIAssignID:
      if (!AAOnly)
        K->mergeDIAssignID(J);
      break;
    case LLVMContext::MD_tbaa:
      if (DoesKMove)
        K->setMetadata(Kind, MDNode::getMostGenericTBAA(JMD, KMD));
      break;
    case LLVMContext::MD_alias_scope:
      if (DoesKMove)
        K->setMetadata(Kind, MDNode::getMostGenericAliasScope(JMD, KMD));
      break;
    case LLVMContext::MD_noalias:
    case LLVMContext::MD_mem_parallel_loop_access:
      if (DoesKMove)
        K->setMetadata(Kind, MDNode::intersect(JMD, KMD));
      break;
    case LLVMContext::MD_access_group:
      if (DoesKMove)
        K->setMetadata(LLVMContext::MD_access_group,
                       intersectAccessGroups(K, J));
      break;
    case LLVMContext::MD_range:
      if (!AAOnly && (DoesKMove || !K->hasMetadata(LLVMContext::MD_noundef)))
        K->setMetadata(Kind, MDNode::getMostGenericRange(JMD, KMD));
      break;
    case LLVMContext::MD_nofpclass:
      if (!AAOnly && (DoesKMove || !K->hasMetadata(LLVMContext::MD_noundef)))
        K->setMetadata(Kind, MDNode::getMostGenericNoFPClass(JMD, KMD));
      break;
    case LLVMContext::MD_fpmath:
      if (!AAOnly)
        K->setMetadata(Kind, MDNode::getMostGenericFPMath(JMD, KMD));
      break;
    case LLVMContext::MD_invariant_load:
    case LLVMContext::MD_invariant_group:
      // If K moves, only keep the invariant metadata if it is present on
      // both instructions; otherwise the invariant would be asserted on a
      // path (J's) that never promised it. If K does not move, K stays on
      // its original path, so its existing metadata remains valid.
      if (DoesKMove)
        K->setMetadata(Kind, JMD);
      break;
    case LLVMContext::MD_nonnull:
      if (!AAOnly && (DoesKMove || !K->hasMetadata(LLVMContext::MD_noundef)))
        K->setMetadata(Kind, JMD);
      break;
    // Keep empty cases for prof, mmra, memprof, and callsite to prevent them
    // from being removed as unknown metadata. The actual merging is handled
    // separately below.
    case LLVMContext::MD_prof:
    case LLVMContext::MD_mmra:
    case LLVMContext::MD_memprof:
    case LLVMContext::MD_callsite:
      break;
    case LLVMContext::MD_callee_type:
      if (!AAOnly) {
        K->setMetadata(LLVMContext::MD_callee_type,
                       MDNode::getMergedCalleeTypeMetadata(KMD, JMD));
      }
      break;
    case LLVMContext::MD_callees:
      // If K moves, it replaces J on J's path and must allow J's callees as
      // well. If K does not move, its callees remain valid.
      if (!AAOnly && DoesKMove)
        K->setMetadata(Kind, MDNode::getMergedCalleesMetadata(KMD, JMD));
      break;
    case LLVMContext::MD_align:
      if (!AAOnly && (DoesKMove || !K->hasMetadata(LLVMContext::MD_noundef)))
        K->setMetadata(
            Kind, MDNode::getMostGenericAlignmentOrDereferenceable(JMD, KMD));
      break;
    case LLVMContext::MD_dereferenceable:
    case LLVMContext::MD_dereferenceable_or_null:
      if (!AAOnly && DoesKMove)
        K->setMetadata(
            Kind, MDNode::getMostGenericAlignmentOrDereferenceable(JMD, KMD));
      break;
    case LLVMContext::MD_preserve_access_index:
      // Preserve !preserve.access.index in K.
      break;
    case LLVMContext::MD_noundef:
      // If K does move, keep noundef if it is present in both instructions.
      if (!AAOnly && DoesKMove)
        K->setMetadata(Kind, JMD);
      break;
    case LLVMContext::MD_nontemporal:
      // Preserve !nontemporal if it is present on both instructions.
      if (!AAOnly)
        K->setMetadata(Kind, JMD);
      break;
    case LLVMContext::MD_mem_cache_hint:
      // Preserve !mem.cache_hint only if it is present and equivalent on both
      // instructions.
      if (!AAOnly && KMD != JMD)
        K->setMetadata(Kind, nullptr);
      break;
    case LLVMContext::MD_noalias_addrspace:
      if (DoesKMove)
        K->setMetadata(Kind, MDNode::getMostGenericNoaliasAddrspace(JMD, KMD));
      break;
    case LLVMContext::MD_nosanitize:
      // Preserve !nosanitize if both K and J have it.
      K->setMetadata(Kind, JMD);
      break;
    case LLVMContext::MD_captures:
      K->setMetadata(
          Kind, MDNode::fromCaptureComponents(
                    K->getContext(), MDNode::toCaptureComponents(JMD) |
                                         MDNode::toCaptureComponents(KMD)));
      break;
    case LLVMContext::MD_alloc_token:
      if (!AAOnly && KMD != JMD)
        K->setMetadata(Kind, MDNode::getMergedAllocTokenMetadata(KMD, JMD));
      break;
    }
  }

  // Merge MMRAs.
  // This is handled separately because we also want to handle cases where K
  // doesn't have tags but J does.
  auto JMMRA = J->getMetadata(LLVMContext::MD_mmra);
  auto KMMRA = K->getMetadata(LLVMContext::MD_mmra);
  if (JMMRA || KMMRA) {
    K->setMetadata(LLVMContext::MD_mmra,
                   MMRAMetadata::combine(K->getContext(), JMMRA, KMMRA));
  }

  // Merge memprof metadata.
  // Handle separately to support cases where only one instruction has the
  // metadata.
  auto *JMemProf = J->getMetadata(LLVMContext::MD_memprof);
  auto *KMemProf = K->getMetadata(LLVMContext::MD_memprof);
  if (!AAOnly && (JMemProf || KMemProf)) {
    K->setMetadata(LLVMContext::MD_memprof,
                   MDNode::getMergedMemProfMetadata(KMemProf, JMemProf));
  }

  // Merge callsite metadata.
  // Handle separately to support cases where only one instruction has the
  // metadata.
  auto *JCallSite = J->getMetadata(LLVMContext::MD_callsite);
  auto *KCallSite = K->getMetadata(LLVMContext::MD_callsite);
  if (!AAOnly && (JCallSite || KCallSite)) {
    K->setMetadata(LLVMContext::MD_callsite,
                   MDNode::getMergedCallsiteMetadata(KCallSite, JCallSite));
  }

  // Merge prof metadata.
  // Handle separately to support cases where only one instruction has the
  // metadata.
  auto *JProf = J->getMetadata(LLVMContext::MD_prof);
  auto *KProf = K->getMetadata(LLVMContext::MD_prof);
  if (!AAOnly && (JProf || KProf)) {
    // We do not need to merge metadata if we have select instructions and the
    // one we're keeping (K) does not move (explicitly merging would drop the
    // information in this case). Otherwise, explicitly handle the merging.
    if (DoesKMove || (!isa<SelectInst>(J) || !isa<SelectInst>(K)) ||
        ProfcheckDisableMetadataFixes)
      K->setMetadata(LLVMContext::MD_prof,
                     MDNode::getMergedProfMetadata(KProf, JProf, K, J));
  }
}

void llvm::combineMetadataForCSE(Instruction *K, const Instruction *J,
                                 bool DoesKMove) {
  combineMetadata(K, J, DoesKMove);
}

void llvm::combineAAMetadata(Instruction *K, const Instruction *J) {
  combineMetadata(K, J, /*DoesKMove=*/true, /*AAOnly=*/true);
}
