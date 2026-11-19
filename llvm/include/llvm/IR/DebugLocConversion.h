//===- DebugLocConversion.h - Convert DILocation<->FLDebugLoc ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines data structures and interfaces for converting between
// DILocation-based and FLDebugLoc-based source locations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGLOCCONVERSION_H
#define LLVM_IR_DEBUGLOCCONVERSION_H

#include "llvm/IR/DebugLoc.h"
#include <memory>

namespace llvm {

struct CompleteDebugLocation;
namespace detail {
template <> struct HasFLIndex<CompleteDebugLocation> {
  using IndexType = uint64_t;
};
} // namespace detail

/// Represents a DILocation/DebugLoc without direct use of metadata or a
/// function-local context. Used when converting between formats.
/// There are cases where 2 FLDebugLocs may map to the same DILocation:
/// - In nodebug functions containing inlined debug locations, DILocations do
///   not represent the inline chain into the nodebug function, while
///   FLDebugLocs do.
/// - With uniqued inlined call locations, FLDebugLocs include the inlined
///   function as part of their unique identity, so DILocations that would be
///   merged in uniquing may not be merged as FLDebugLocs.
/// In order to keep these disambiguated, this representation includes a
/// reference to the Function that the location actually appears in (required),
/// and the DISubprogram of the location(s) that are inlined into this (if
/// any), which are used for uniquing.
struct CompleteDebugLocation {
  Function *Fn;
  uint32_t Line;
  uint16_t Column;
  DILocalScope *Scope;
  FLIndex<CompleteDebugLocation> InlinedAt;
  bool IsImplicitCode;
  uint64_t AtomGroup;
  uint8_t AtomRank;
  DILayerLocList *IRLayers;
  bool IsDistinct;
  DISubprogram *InlineeSP;
};

// Persistent per-module storage for all CompleteDebugLocations. Stored in the
// Context and retrieved as needed; may be written to, but never removed.
class DebugLocationStorage {
  SmallVector<CompleteDebugLocation, 0> DebugLocations;
  // For each CompleteDebugLocation, the entry in DILocations/FLDebugLocs at the
  // same index is the equivalent for its representation.
  SmallVector<DILocation *, 0> DILocations;
  SmallVector<FLDebugLoc, 0> FLDebugLocs;
  // We rely on representation-specific maps for uniquing, rather than uniquing
  // by CompleteDebugLocation, since we never create a CompleteDebugLocation
  // "natively", and uniquing directly from the input representation is simpler.
  DenseMap<std::tuple<Function *, FLDebugLoc>, FLIndex<CompleteDebugLocation>>
      FLDLMap;
  DenseMap<std::tuple<Function *, DILocation *, DISubprogram *>,
           FLIndex<CompleteDebugLocation>>
      DILMap;
  // Pair of maps used to track MDFunctionLocalContext attachments, which need
  // to be kept persistent between separate conversions. Nodebug functions never
  // have their FLContext replaced, so we store that attachment directly; other
  // functions have their FLContext replaced iff the DISubprogram is replaced,
  // so we use the subprogram to track attachments.
  DenseMap<Function *, MDFunctionLocalContext *> NodebugFunctionContexts;
  DenseMap<DISubprogram *, MDFunctionLocalContext *> SPContexts;
  // Transient field, should be set when performing one continuous stream of
  // conversions, but needs to be cleared if/when the contents of any
  // FunctionLocalContext may change.
  // TODO: Maybe make this something that gets passed in instead.
  DenseMap<MDFunctionLocalContext *, std::unique_ptr<FLMDBuilder>> Builders;

  FLIndex<CompleteDebugLocation> intern(Function *Fn, DILocation *DL,
                                        DISubprogram *InlineeSP = nullptr);
  FLIndex<CompleteDebugLocation> intern(Function *Fn, FLDebugLoc DL);

  /// Must be called in-order for entries in CompleteDebugLocation, i.e. must
  /// only be called for DebugLocations[i] after calling for all DebugLocations
  /// 0..i-1.
  void convertImpl(FLIndex<CompleteDebugLocation> CDLIdx, FLDebugLoc &New);

  void convertImpl(FLIndex<CompleteDebugLocation> CDLIdx, DILocation *&New);

public:
  // Call after conversions, before any changes might be made to the IR.
  void flush();

  /// Returns the FLContext that, according to this storage, should be attached
  /// to Fn.
  /// This function will not create any new contexts even if they should exist,
  /// so this should be used only after performing conversions.
  MDFunctionLocalContext *getFnContext(Function *Fn) const;

  bool convert(Function *Fn, DILocation *DIL, FLDebugLoc &FLDL);
  bool convert(Function *Fn, FLDebugLoc FLDL, DILocation *&DIL);
};

struct DebugLocConversionInfo {
  using ForeignLocTy = FLDebugLoc;
  using NativeLocTy = DILocation *;

  static MDNode *getNodeFromLoc(Function *Fn, FLDebugLoc Loc);
  static MDNode *getNodeFromLoc(Function *Fn, DILocation *Loc);

  template <typename T> static T getLocFromNode(const MDNode *N);

  using AttachFnTy = void (*)(void *, NativeLocTy);
  template <typename T> static void attach(void *AttachTo, NativeLocTy Loc) {
    static_cast<T *>(AttachTo)->setDebugLoc(Loc);
  }
};

template <typename From, typename To> class DebugLocConversionBase {
protected:
  DebugLocConversionBase(Module *M);
  ~DebugLocConversionBase();
  DebugLocConversionBase(DebugLocConversionBase &&) = delete;
  DebugLocConversionBase &operator=(DebugLocConversionBase &&) = delete;

  using DLInfo = DebugLocConversionInfo;
  struct AttachmentInfo {
    Function *Fn;
    void *AttachTo;
    DLInfo::AttachFnTy LocSetter;
    From Loc;
  };
  SmallVector<AttachmentInfo, 0> Attachments;
  DenseMap<void *, To> AttachmentMap;
  DenseMap<std::pair<Function *, MDNode *>, To> NodeMap;
  SmallVector<std::pair<Instruction *, MDNode *>, 0> LoopInstrs;
  DenseMap<std::pair<Function *, MDNode *>, MDNode *> RemappedLoopNodes;
  Module *M;
  std::shared_ptr<DebugLocationStorage> Storage;

  template <typename T>
  void addAttachment(Function *Fn, T *AttachedTo, From Loc) {
    Attachments.push_back({Fn, AttachedTo, &DLInfo::attach<T>, Loc});
  }
  void addLoopInstr(Instruction *I, MDNode *LoopMD) {
    LoopInstrs.push_back({I, LoopMD});
  }
  bool convertRecordedLocs();
  void flush();
  void replaceLoopNodes();
  To getDebugLoc(const void *AttachedTo) const;
};

// Conversion map used when writing/printing values; acts like an alternative
// metadata store. Makes modifies the module in-place on construction, to ensure
// that metadata is enumerated correctly.
class DebugLocWriteConversionMap
    : public DebugLocConversionBase<DebugLocConversionInfo::NativeLocTy,
                                    DebugLocConversionInfo::ForeignLocTy> {
  SmallVector<std::pair<Function *, MDNode *>, 0> OldFnFLMDAttachments;

public:
  /// Temporarily rewrites the module M on construction, and restores it on
  /// deletion.
  DebugLocWriteConversionMap(Module *M);
  ~DebugLocWriteConversionMap();
  using DebugLocConversionBase::getDebugLoc;
  // Used for obtaining locations after the initial conversion pass; does not
  // modify any existing attachments.
  DLInfo::ForeignLocTy getExtraDebugLoc(Function *Fn, void *AttachedTo,
                                        DLInfo::NativeLocTy Loc);
};

// Conversion map used when reading/parsing values; since we cannot load
// FLMD and MD into the same data type, this map acts as an intermediary, where
// we load the "foreign" data type (which cannot be attached) into this map, and
// when we finish loading metadata this map converts to the native type.
class DebugLocReadConversionMap
    : public DebugLocConversionBase<DebugLocConversionInfo::ForeignLocTy,
                                    DebugLocConversionInfo::NativeLocTy> {
public:
  DebugLocReadConversionMap(Module *M);
  using DebugLocConversionBase::addAttachment;
  using DebugLocConversionBase::addLoopInstr;
  /// Rewrites all attachments and loop metadata to use the native location
  /// type; may only be called once all locations have been fully resolved and
  /// added to this map.
  /// Returns `true` if we failed to convert for any reason.
  bool resolveReferences();
  MDFunctionLocalContext *getFnContext(Function *Fn) const;
};
} // namespace llvm

#endif
