//===- FunctionLocalMetadata.h - Function-Local Metadata Info ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the function-local metadata types and the
// MDFunctionLocalContext class which stores them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FUNCTIONLOCALMETADATA_H
#define LLVM_IR_FUNCTIONLOCALMETADATA_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Metadata.h"
#include <optional>

namespace llvm {

class DILocalScope;
class MDFunctionLocalContext;

struct FLScope;
struct FLSrcLoc;
struct FLInlinedCall;

namespace detail {
template <typename T> struct HasFLIndex {};
template <> struct HasFLIndex<FLScope> {
  using IndexType = uint16_t;
};
template <> struct HasFLIndex<FLSrcLoc> {
  using IndexType = uint32_t;
};
template <> struct HasFLIndex<FLInlinedCall> {
  using IndexType = uint16_t;
};
} // namespace detail

/// Small wrapper class for an FLMD index, which may be "no index".
/// Should be checked to see if value is present before using, otherwise the
/// implicit/explicit index value will be invalid.
/// Internally uses 0 as the "no index" value, and stores the internal index
/// value as the actual index + 1, subtracting to get the real index.
template <typename IndexedType> class FLIndex {
public:
  using IndexType = typename detail::HasFLIndex<IndexedType>::IndexType;

private:
  friend struct DenseMapInfo<FLIndex<IndexedType>>;
  IndexType Index;

public:
  FLIndex() : Index(0) {}
  FLIndex(std::nullopt_t) : Index(0) {}
  FLIndex(IndexType InIndex) : Index(InIndex + 1) {
    assert(InIndex < std::numeric_limits<IndexType>::max() &&
           "Hit maximum index limit!");
  }
  operator bool() const { return Index; }
  IndexType get() const { return Index - 1; }
  bool operator==(const FLIndex &Other) const { return Index == Other.Index; }
  bool operator!=(const FLIndex &Other) const { return Index != Other.Index; }

  IndexType asRaw() const { return Index; }
  static FLIndex fromRaw(IndexType RawIndex) {
    FLIndex Index;
    Index.Index = RawIndex;
    return Index;
  }
  void addOffset(IndexType Offset) {
    if (*this)
      Index += Offset;
  }

  friend inline hash_code hash_value(const FLIndex<IndexedType> &Idx) {
    return hash_value(Idx.asRaw());
  }
};

template <typename IndexedType> struct DenseMapInfo<FLIndex<IndexedType>> {
  static unsigned getHashValue(FLIndex<IndexedType> DL) {
    return hash_value(DL.asRaw());
  }

  static bool isEqual(FLIndex<IndexedType> LHS, FLIndex<IndexedType> RHS) {
    return LHS == RHS;
  }
};

struct FLInlinedCall {
  // The FLDebugLoc class, where the atom group numbers are used, has only 13
  // bits to store AtomGroup - thus there is currently no point in holding
  // MaxAtomGroup numbers larger than 13 bits.
  constexpr static uint16_t MaxAllowedAtomGroup = (1 << 13) - 1;
  FLIndex<FLSrcLoc> SrcLocIdx;
  FLIndex<FLInlinedCall> InlinedAtIdx;
  // Transient field used to track the largest atom in this inlined call context
  // for the purpose of generating new atom groups.
  // NB: We keep the MaxAtomGroup field at 15 bits for padding purposes, but it
  // is capped at 13 bits (see MaxAllowedAtomGroup above); In future we may
  // reclaim these bits for some other purpose.
  uint16_t MaxAtomGroup : 15;
  uint16_t Uniquable : 1;
  MDFunctionLocalContext *InlineeContext;
  FLInlinedCall() = default;
  FLInlinedCall(FLIndex<FLSrcLoc> SrcLocIdx,
                FLIndex<FLInlinedCall> InlinedAtIdx,
                MDFunctionLocalContext *InlineeContext, bool Uniquable,
                uint16_t MaxAtomGroup = 0)
      : SrcLocIdx(SrcLocIdx), InlinedAtIdx(InlinedAtIdx),
        MaxAtomGroup(MaxAtomGroup), Uniquable(Uniquable),
        InlineeContext(InlineeContext) {
    assert(MaxAtomGroup <= MaxAllowedAtomGroup && "Max Atom group too large!");
  }
  MDFunctionLocalContext *getInlinee() const { return InlineeContext; }
  uint16_t getNewAtomGroup() {
    // TODO: This assert exists more for testing purposes, to determine whether
    // this is a limit we actually hit or not. If we hit this frequently, we may
    // need to consider a fix; if we do relax this assert, then we'll need to
    // also relax the assert in mapAtomInstance that new atom groups are
    // strictly greater than old.
    // Suitable fixes include either refusing further atom groups by always
    // returning 0 once the limit is reached, or wrapping back around to 1
    // (which is very unlikely to cause any meaningful collisions).
    assert(MaxAtomGroup < MaxAllowedAtomGroup && "Atom group limit reached!");
    return ++MaxAtomGroup;
  }
  // Equality check that ignores MaxAtomGroup, which is a transient field - it
  // is not serialized, and may be modified in-place.
  bool isEquivalent(const FLInlinedCall &Other) const {
    return SrcLocIdx == Other.SrcLocIdx && InlinedAtIdx == Other.InlinedAtIdx &&
           Uniquable == Other.Uniquable &&
           InlineeContext == Other.InlineeContext;
  }
  /// Returns true if the location of this FLInlinedCall is inlined.
  bool isInlined() const { return InlinedAtIdx; }
  /// Returns the FLInlinedCall that this call was inlined at; only valid if
  /// isInlined().
  FLInlinedCall getInlinedAt(const MDFunctionLocalContext *FLCtx) const;

  // Ignores MaxAtomGroup - as a transient field, we do not want to use it as
  // part of a map key.
  friend inline hash_code hash_value(const FLInlinedCall &IC) {
    return hash_combine(IC.SrcLocIdx, IC.InlinedAtIdx, IC.Uniquable,
                        IC.InlineeContext);
  }
};
template <> struct DenseMapInfo<FLInlinedCall> {
  static unsigned getHashValue(FLInlinedCall IC) { return hash_value(IC); }

  static bool isEqual(FLInlinedCall LHS, FLInlinedCall RHS) {
    return LHS.isEquivalent(RHS);
  }
};

/// Unique FLMD.
/// Stores source location information.
struct FLSrcLoc {
  uint32_t Line;
  uint16_t Column;
  FLIndex<FLScope> ScopeIdx;
  FLSrcLoc() = default;
  FLSrcLoc(uint32_t Line, uint16_t Column, FLIndex<FLScope> ScopeIdx)
      : Line(Line), Column(Column), ScopeIdx(ScopeIdx) {
    assert(ScopeIdx && "SrcLoc needs valid scope!");
  }
  uint64_t asRawInt() const {
    static_assert(sizeof(*this) == sizeof(uint64_t));
    uint64_t Result;
    std::memcpy(&Result, this, sizeof(Result));
    return Result;
  }
  bool operator==(const FLSrcLoc &Other) const {
    return asRawInt() == Other.asRawInt();
  }
  friend inline hash_code hash_value(const FLSrcLoc &SrcLoc) {
    return hash_value(SrcLoc.asRawInt());
  }
};
template <> struct DenseMapInfo<FLSrcLoc> {
  static unsigned getHashValue(FLSrcLoc SrcLoc) { return hash_value(SrcLoc); }

  static bool isEqual(FLSrcLoc LHS, FLSrcLoc RHS) { return LHS == RHS; }
};

/// Unique FLMD, just a wrapper around a DILocalScope.
struct FLScope {
  DILocalScope *Scope;
  FLScope() = default;
  FLScope(DILocalScope *Scope) : Scope(Scope) {}
  operator DILocalScope *() { return Scope; }
  DILocalScope *get() { return Scope; }
  operator const DILocalScope *() const { return Scope; }
  const DILocalScope *get() const { return Scope; }
  bool operator==(const FLScope &Other) const { return Scope == Other.Scope; }
  friend inline hash_code hash_value(const FLScope &Scope) {
    return hash_value(Scope.Scope);
  }
};
template <> struct DenseMapInfo<FLScope> {
  static unsigned getHashValue(FLScope Scope) { return hash_value(Scope); }

  static bool isEqual(FLScope LHS, FLScope RHS) { return LHS == RHS; }
};

/// Storage class for function-local metadata objects.
/// This is a special MDNode variation: the class benefits from being able to
/// appear as an operand in other MDNodes, to be treated as a metadata
/// attachment to functions, and from the generic MDNode enumeration logic.
/// However, it holds MDNode references that are not part of the MDNode
/// operands() array and are not tracking references (meaning it contains
/// forward references that must be resolved separately from the generic MDNode
/// tracking/resolution facilities), does not meaningfully support cloning, and
/// can only ever be created as `distinct`, not unique or temporary.
/// To access all MDNodes referenced from this context, use the
/// `getUsedMDNodes()` function.
class MDFunctionLocalContext : public MDNode {
  MDFunctionLocalContext(LLVMContext &C, StorageType Storage, bool NonDebug)
      : MDNode(C, MDFunctionLocalContextKind, Storage, {}) {
    // NonDebug
    SubclassData1 = NonDebug;
    // MaxAtomGroup (starts at 0 and mutates in-place)
    SubclassData16 = 0;
  }
  ~MDFunctionLocalContext() = default;

  SmallVector<FLScope, 0> Scopes;
  SmallVector<FLSrcLoc, 0> SrcLocs;
  SmallVector<FLInlinedCall, 0> InlinedCalls;

public:
  friend class LLVMContextImpl;
  friend class MDNode;
  constexpr static uint16_t MaxAllowedAtomGroup =
      FLInlinedCall::MaxAllowedAtomGroup;

  static bool classof(const Metadata *MD) {
    return MD->getMetadataID() == MDFunctionLocalContextKind;
  }

  // Populates \p Nodes with the complete set of MDNodes used by this.
  LLVM_ABI void getUsedMDNodes(SmallVectorImpl<MDNode *> &Nodes) const;

  LLVM_ABI static MDFunctionLocalContext *getDistinct(LLVMContext &Context,
                                                      bool NonDebug = false);
  TempMDFunctionLocalContext clone() const { return cloneImpl(); }
  TempMDFunctionLocalContext cloneImpl() const {
    llvm_unreachable("Clone unsupported for MDFunctionLocalContext.");
    return nullptr;
  }

  // If this is true, then this is a special FLContext, used for a function
  // without debug info to store inlined call information. This FLContext
  // contains no Scopes and no SrcLocs, meaning it is never valid to create a
  // non-inlined debug loc within it.
  // Nested InlinedCalls are copied during inlining as normal, and the resulting
  // InlinedCall for the inlining operation has no SrcLoc and no InlinedAt; this
  // requires some special handling for functions like "getInlinedAtLocation"
  // that explicitly traverse the InlinedAt chain. If this function is then
  // inlined into a debug function, the location-less InlinedCall is overwritten
  // instead of transferred.
  // NB: Hopefully this design doesn't raise many problems as-is, because in
  // general we only care about debug locs inlined into a nodebug function when
  // they are inlined again into a debug function - otherwise they are generally
  // ignored. If this leads to problems, then we need a more sophisticated
  // solution.
  bool isNonDebug() const { return SubclassData1; }

  ArrayRef<FLScope> getScopes() const { return Scopes; }
  ArrayRef<FLSrcLoc> getSrcLocs() const { return SrcLocs; }
  ArrayRef<FLInlinedCall> getInlinedCalls() const { return InlinedCalls; }

  FLScope getScope(FLIndex<FLScope> Idx) const {
    assert(Idx && "Attempted to fetch a Scope with an empty index!");
    return Scopes[Idx.get()];
  }
  FLScope getScope(FLIndex<FLScope> Idx,
                   FLIndex<FLInlinedCall> InlinedAt) const {
    assert(Idx && "Attempted to fetch a Scope with an empty index!");
    if (!InlinedAt)
      return Scopes[Idx.get()];
    return InlinedCalls[InlinedAt.get()].getInlinee()->getScope(Idx);
  }
  FLSrcLoc getSrcLoc(FLIndex<FLSrcLoc> Idx) const {
    assert(Idx && "Attempted to fetch a SrcLoc with an empty index!");
    return SrcLocs[Idx.get()];
  }
  FLSrcLoc getSrcLoc(FLIndex<FLSrcLoc> Idx,
                     FLIndex<FLInlinedCall> InlinedAt) const {
    assert(Idx && "Attempted to fetch a SrcLoc with an empty index!");
    if (!InlinedAt)
      return SrcLocs[Idx.get()];
    return InlinedCalls[InlinedAt.get()].getInlinee()->getSrcLoc(Idx);
  }
  FLInlinedCall getInlinedCall(FLIndex<FLInlinedCall> Idx) const {
    assert(Idx && "Attempted to fetch an InlinedCall with an empty index!");
    return InlinedCalls[Idx.get()];
  }

  uint16_t getMaxAtomGroup() const { return SubclassData16; }
  /// Generally this shouldn't be called directly; prefer
  /// updateAtomGroupWaterline, including any relevant InlinedAt.
  /// TODO: Should this be private?
  void setMaxAtomGroup(uint16_t NewAtomGroup) { SubclassData16 = NewAtomGroup; }

  uint16_t getNewAtomGroup(FLIndex<FLInlinedCall> InlinedCallIdx) {
    if (!InlinedCallIdx) {
      // NB: See comment in FLInlinedCall::getNewAtomGroup.
      assert(getMaxAtomGroup() < MaxAllowedAtomGroup &&
             "Atom group limit reached!");
      setMaxAtomGroup(getMaxAtomGroup() + 1);
      return getMaxAtomGroup();
    }
    return InlinedCalls[InlinedCallIdx.get()].getNewAtomGroup();
  }
  void updateAtomGroupWaterline(FLIndex<FLInlinedCall> InlinedCallIdx,
                                uint16_t NewWaterline) {
    assert(NewWaterline <= MaxAllowedAtomGroup &&
           "New atom group exceeds limit!");
    if (!InlinedCallIdx)
      setMaxAtomGroup(std::max(getMaxAtomGroup(), NewWaterline));
    else
      InlinedCalls[InlinedCallIdx.get()].MaxAtomGroup = std::max(
          InlinedCalls[InlinedCallIdx.get()].MaxAtomGroup, NewWaterline);
  }
  uint16_t getAtomGroupWaterline(FLIndex<FLInlinedCall> InlinedCallIdx) {
    if (!InlinedCallIdx)
      return getMaxAtomGroup();
    return InlinedCalls[InlinedCallIdx.get()].MaxAtomGroup;
  }

  /// Inserts a new FLScope \p Scope and returns its index.
  FLIndex<FLScope> insertScope(FLScope Scope) {
    Scopes.push_back(Scope);
    return Scopes.size() - 1;
  }
  /// Returns the existing index of \p Scope, or else inserts it and returns its
  /// new index.
  /// NB: Uses linear search, which is very inefficient, but required if we
  /// may be inserting a duplicate Scope; if performing this operation
  /// repeatedly, FLMDBuilder is significantly more efficient.
  FLIndex<FLScope> getOrInsertScope(FLScope Scope) {
    for (uint16_t Idx = 0; Idx < Scopes.size(); ++Idx)
      if (Scopes[Idx] == Scope)
        return Idx;
    return insertScope(Scope);
  }

  /// Inserts a new FLSrcLoc \p SrcLoc and returns its index.
  FLIndex<FLSrcLoc> insertSrcLoc(FLSrcLoc SrcLoc) {
    SrcLocs.push_back(SrcLoc);
    return SrcLocs.size() - 1;
  }
  /// Returns the existing index of \p SrcLoc, or else inserts it and returns
  /// its new index.
  /// NB: Uses linear search, which is very inefficient, but required if we
  /// may be inserting a duplicate SrcLoc; if performing this operation
  /// repeatedly, FLMDBuilder is significantly more efficient.
  /// An exception to this is if inserting a SrcLoc with DISubprogram scope,
  /// column=0, and line equal to either 0, SP->getLine(), or
  /// SP->getScopeLine(); these are inserted near the front of SrcLocs by
  /// FLMDBuilder, so can generally be retrieved quickly.
  FLIndex<FLSrcLoc> getOrInsertSrcLoc(FLSrcLoc SrcLoc) {
    for (uint32_t Idx = 0; Idx < SrcLocs.size(); ++Idx)
      if (SrcLocs[Idx] == SrcLoc)
        return Idx;
    return insertSrcLoc(SrcLoc);
  }

  /// Returns the existing index of a uniquable \p InlinedCall, or else inserts
  /// it and returns its new index.
  /// NB: Uses linear search for uniquing, which is very inefficient; pass
  /// \p UniquableMayExist as false if passing an InlinedCall which is uniquable
  /// but is known to not already exist.
  FLIndex<FLInlinedCall> insertInlinedCall(FLInlinedCall InlinedCall,
                                           bool UniquableMayExist = true) {
    if (InlinedCall.Uniquable && UniquableMayExist) {
      for (uint16_t Idx = 0; Idx < InlinedCalls.size(); ++Idx) {
        if (InlinedCall.isEquivalent(InlinedCalls[Idx])) {
          InlinedCalls[Idx].MaxAtomGroup = std::max(
              InlinedCall.MaxAtomGroup, InlinedCalls[Idx].MaxAtomGroup);
          return FLIndex<FLInlinedCall>(Idx);
        }
      }
    }
    InlinedCalls.push_back(InlinedCall);
    return InlinedCalls.size() - 1;
  }
};

inline FLInlinedCall
FLInlinedCall::getInlinedAt(const MDFunctionLocalContext *FLCtx) const {
  assert(isInlined() && "getInlinedAt requires the location of this "
                        "FLInlinedCall to be inlined!");
  return FLCtx->getInlinedCall(InlinedAtIdx);
}

/// Builder class used to more efficiently insert new entries into an
/// MDFunctionLocalContext, using maps to perform uniquing instead of linear
/// search.
/// If inserting into an empty MDFunctionLocalContext, also performs initial
/// insertions of the DISubprogram, and of SrcLocs for
/// Line={0, SP->getLine(), SP->getScopeLine()} at top-level scope, as these are
/// likely to be requested by optimizations later on.
class FLMDBuilder {
  MDFunctionLocalContext *FLContext;
  DenseMap<FLScope, FLIndex<FLScope>> ScopeMap;
  DenseMap<FLSrcLoc, FLIndex<FLSrcLoc>> SrcLocMap;

public:
  // Returns the index for a Scope with the given arguments, either returning
  // the index of an existing copy or inserting a new entry and returning its
  // index.
  FLIndex<FLScope> getScope(const DILocalScope *Scope) {
    FLScope InsertedScope(const_cast<DILocalScope *>(Scope));
    auto [ScopeIt, Inserted] = ScopeMap.try_emplace(InsertedScope);
    if (Inserted)
      ScopeIt->second = FLContext->insertScope(InsertedScope);
    return ScopeIt->second;
  }
  // Returns the index for a SrcLoc with the given arguments, either returning
  // the index of an existing copy or inserting a new entry and returning its
  // index.
  FLIndex<FLSrcLoc> getSrcLoc(uint32_t Line, uint16_t Column,
                              FLIndex<FLScope> ScopeIdx) {
    FLSrcLoc InsertedSrcLoc(Line, Column, ScopeIdx);
    auto [SrcLocIt, Inserted] = SrcLocMap.try_emplace(InsertedSrcLoc);
    if (Inserted)
      SrcLocIt->second = FLContext->insertSrcLoc(InsertedSrcLoc);
    return SrcLocIt->second;
  }
  // Returns the index for a SrcLoc with the given arguments, either returning
  // the index of an existing copy or inserting a new entry and returning its
  // index. May also insert \p Scope into the FunctionLocalContext.
  FLIndex<FLSrcLoc> getSrcLocWithScope(uint32_t Line, uint16_t Column,
                                       const DILocalScope *Scope) {
    FLIndex<FLScope> ScopeIdx = getScope(Scope);
    return getSrcLoc(Line, Column, ScopeIdx);
  }

  LLVM_ABI FLMDBuilder(MDFunctionLocalContext *FLContext,
                       const DISubprogram *SP);
};

} // end namespace llvm

#endif // LLVM_IR_FUNCTIONLOCALMETADATA_H
