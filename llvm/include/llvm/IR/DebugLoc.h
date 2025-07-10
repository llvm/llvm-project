//===- DebugLoc.h - Debug Location Information ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of light weight data structures used
// to describe and track debug location information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGLOC_H
#define LLVM_IR_DEBUGLOC_H

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/TrackingMDRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

  class LLVMContext;
  class raw_ostream;
  class DILocation;
  class Function;

#if LLVM_ENABLE_DEBUGLOC_TRACKING_COVERAGE
#if LLVM_ENABLE_DEBUGLOC_TRACKING_ORIGIN
  struct DbgLocOrigin {
    static constexpr unsigned long MaxDepth = 16;
    using StackTracesTy =
        SmallVector<std::pair<int, std::array<void *, MaxDepth>>, 0>;
    StackTracesTy StackTraces;
    DbgLocOrigin(bool ShouldCollectTrace);
    void addTrace();
    const StackTracesTy &getOriginStackTraces() const { return StackTraces; };
  };
#else
  struct DbgLocOrigin {
    DbgLocOrigin(bool) {}
  };
#endif
  // Used to represent different "kinds" of DebugLoc, expressing that the
  // instruction it is part of is either normal and should contain a valid
  // DILocation, or otherwise describing the reason why the instruction does
  // not contain a valid DILocation.
  enum class DebugLocKind : uint8_t {
    // The instruction is expected to contain a valid DILocation.
    Normal,
    // The instruction is compiler-generated, i.e. it is not associated with any
    // line in the original source.
    CompilerGenerated,
    // The instruction has intentionally had its source location removed,
    // typically because it was moved outside of its original control-flow and
    // presenting the prior source location would be misleading for debuggers
    // or profilers.
    Dropped,
    // The instruction does not have a known or currently knowable source
    // location, e.g. the attribution is ambiguous in a way that can't be
    // represented, or determining the correct location is complicated and
    // requires future developer effort.
    Unknown,
    // DebugLoc is attached to an instruction that we don't expect to be
    // emitted, and so can omit a valid DILocation; we don't expect to ever try
    // and emit these into the line table, and trying to do so is a sign that
    // something has gone wrong (most likely a DebugLoc leaking from a transient
    // compiler-generated instruction).
    Temporary
  };

  // Extends TrackingMDNodeRef to also store a DebugLocKind and Origin,
  // allowing Debugify to ignore intentionally-empty DebugLocs and display the
  // code responsible for generating unintentionally-empty DebugLocs.
  // Currently we only need to track the Origin of this DILoc when using a
  // DebugLoc that is not annotated (i.e. has DebugLocKind::Normal) and has a
  // null DILocation, so only collect the origin stacktrace in those cases.
  class DILocAndCoverageTracking : public TrackingMDNodeRef,
                                   public DbgLocOrigin {
  public:
    DebugLocKind Kind;
    // Default constructor for empty DebugLocs.
    DILocAndCoverageTracking()
        : TrackingMDNodeRef(nullptr), DbgLocOrigin(true),
          Kind(DebugLocKind::Normal) {}
    // Valid or nullptr MDNode*, no annotative DebugLocKind.
    DILocAndCoverageTracking(const MDNode *Loc)
        : TrackingMDNodeRef(const_cast<MDNode *>(Loc)), DbgLocOrigin(!Loc),
          Kind(DebugLocKind::Normal) {}
    LLVM_ABI DILocAndCoverageTracking(const DILocation *Loc);
    // Explicit DebugLocKind, which always means a nullptr MDNode*.
    DILocAndCoverageTracking(DebugLocKind Kind)
        : TrackingMDNodeRef(nullptr),
          DbgLocOrigin(Kind == DebugLocKind::Normal), Kind(Kind) {}
  };
  template <> struct simplify_type<DILocAndCoverageTracking> {
    using SimpleType = MDNode *;

    static MDNode *getSimplifiedValue(DILocAndCoverageTracking &MD) {
      return MD.get();
    }
  };
  template <> struct simplify_type<const DILocAndCoverageTracking> {
    using SimpleType = MDNode *;

    static MDNode *getSimplifiedValue(const DILocAndCoverageTracking &MD) {
      return MD.get();
    }
  };

  using DebugLocTrackingRef = DILocAndCoverageTracking;
#else
  using DebugLocTrackingRef = TrackingMDNodeRef;
#endif // LLVM_ENABLE_DEBUGLOC_TRACKING_COVERAGE

  /// A debug info location.
  ///
  /// This class is a wrapper around a tracking reference to an \a DILocation
  /// pointer.
  ///
  /// To avoid extra includes, \a DebugLoc doubles the \a DILocation API with a
  /// one based on relatively opaque \a MDNode pointers.
  class DebugLoc {

    DebugLocTrackingRef Loc;

  public:
    DebugLoc() = default;

    /// Construct from an \a DILocation.
    LLVM_ABI DebugLoc(const DILocation *L);

    /// Construct from an \a MDNode.
    ///
    /// Note: if \c N is not an \a DILocation, a verifier check will fail, and
    /// accessors will crash.  However, construction from other nodes is
    /// supported in order to handle forward references when reading textual
    /// IR.
    LLVM_ABI explicit DebugLoc(const MDNode *N);

#if LLVM_ENABLE_DEBUGLOC_TRACKING_COVERAGE
    DebugLoc(DebugLocKind Kind) : Loc(Kind) {}
    DebugLocKind getKind() const { return Loc.Kind; }
#endif

#if LLVM_ENABLE_DEBUGLOC_TRACKING_COVERAGE
    static inline DebugLoc getTemporary() {
      return DebugLoc(DebugLocKind::Temporary);
    }
    static inline DebugLoc getUnknown() {
      return DebugLoc(DebugLocKind::Unknown);
    }
    static inline DebugLoc getCompilerGenerated() {
      return DebugLoc(DebugLocKind::CompilerGenerated);
    }
    static inline DebugLoc getDropped() {
      return DebugLoc(DebugLocKind::Dropped);
    }
#else
    static inline DebugLoc getTemporary() { return DebugLoc(); }
    static inline DebugLoc getUnknown() { return DebugLoc(); }
    static inline DebugLoc getCompilerGenerated() { return DebugLoc(); }
    static inline DebugLoc getDropped() { return DebugLoc(); }
#endif // LLVM_ENABLE_DEBUGLOC_TRACKING_COVERAGE

    /// When two instructions are combined into a single instruction we also
    /// need to combine the original locations into a single location.
    /// When the locations are the same we can use either location.
    /// When they differ, we need a third location which is distinct from
    /// either. If they share a common scope, use this scope and compare the
    /// line/column pair of the locations with the common scope:
    /// * if both match, keep the line and column;
    /// * if only the line number matches, keep the line and set the column as
    /// 0;
    /// * otherwise set line and column as 0.
    /// If they do not share a common scope the location is ambiguous and can't
    /// be represented in a line entry. In this case, set line and column as 0
    /// and use the scope of any location.
    ///
    /// \p LocA \p LocB: The locations to be merged.
    LLVM_ABI static DebugLoc getMergedLocation(DebugLoc LocA, DebugLoc LocB);

    /// Try to combine the vector of locations passed as input in a single one.
    /// This function applies getMergedLocation() repeatedly left-to-right.
    ///
    /// \p Locs: The locations to be merged.
    LLVM_ABI static DebugLoc getMergedLocations(ArrayRef<DebugLoc> Locs);

    /// If this DebugLoc is non-empty, returns this DebugLoc; otherwise, selects
    /// \p Other.
    /// In coverage-tracking builds, this also accounts for whether this or
    /// \p Other have an annotative DebugLocKind applied, such that if both are
    /// empty but exactly one has an annotation, we prefer that annotated
    /// location.
    DebugLoc orElse(DebugLoc Other) const {
      if (*this)
        return *this;
#if LLVM_ENABLE_DEBUGLOC_TRACKING_COVERAGE
      if (Other)
        return Other;
      if (getKind() != DebugLocKind::Normal)
        return *this;
      if (Other.getKind() != DebugLocKind::Normal)
        return Other;
      return *this;
#else
      return Other;
#endif // LLVM_ENABLE_DEBUGLOC_TRACKING_COVERAGE
    }

#if LLVM_ENABLE_DEBUGLOC_TRACKING_ORIGIN
    const DbgLocOrigin::StackTracesTy &getOriginStackTraces() const {
      return Loc.getOriginStackTraces();
    }
    DebugLoc getCopied() const {
      DebugLoc NewDL = *this;
      NewDL.Loc.addTrace();
      return NewDL;
    }
#else
    DebugLoc getCopied() const { return *this; }
#endif

    /// Get the underlying \a DILocation.
    ///
    /// \pre !*this or \c isa<DILocation>(getAsMDNode()).
    /// @{
    LLVM_ABI DILocation *get() const;
    operator DILocation *() const { return get(); }
    DILocation *operator->() const { return get(); }
    DILocation &operator*() const { return *get(); }
    /// @}

    /// Check for null.
    ///
    /// Check for null in a way that is safe with broken debug info.  Unlike
    /// the conversion to \c DILocation, this doesn't require that \c Loc is of
    /// the right type.  Important for cases like \a llvm::StripDebugInfo() and
    /// \a Instruction::hasMetadata().
    explicit operator bool() const { return Loc; }

    /// Check whether this has a trivial destructor.
    bool hasTrivialDestructor() const { return Loc.hasTrivialDestructor(); }

    enum { ReplaceLastInlinedAt = true };
    /// Rebuild the entire inlined-at chain for this instruction so that the top of
    /// the chain now is inlined-at the new call site.
    /// \param   InlinedAt    The new outermost inlined-at in the chain.
    LLVM_ABI static DebugLoc
    appendInlinedAt(const DebugLoc &DL, DILocation *InlinedAt, LLVMContext &Ctx,
                    DenseMap<const MDNode *, MDNode *> &Cache);

    /// Return true if the source locations match, ignoring isImplicitCode and
    /// source atom info.
    bool isSameSourceLocation(const DebugLoc &Other) const {
      if (get() == Other.get())
        return true;
      return ((bool)*this == (bool)Other) && getLine() == Other.getLine() &&
             getCol() == Other.getCol() && getScope() == Other.getScope() &&
             getInlinedAt() == Other.getInlinedAt();
    }

    LLVM_ABI unsigned getLine() const;
    LLVM_ABI unsigned getCol() const;
    LLVM_ABI MDNode *getScope() const;
    LLVM_ABI DILocation *getInlinedAt() const;

    /// Get the fully inlined-at scope for a DebugLoc.
    ///
    /// Gets the inlined-at scope for a DebugLoc.
    LLVM_ABI MDNode *getInlinedAtScope() const;

    /// Rebuild the entire inline-at chain by replacing the subprogram at the
    /// end of the chain with NewSP.
    LLVM_ABI static DebugLoc
    replaceInlinedAtSubprogram(const DebugLoc &DL, DISubprogram &NewSP,
                               LLVMContext &Ctx,
                               DenseMap<const MDNode *, MDNode *> &Cache);

    /// Find the debug info location for the start of the function.
    ///
    /// Walk up the scope chain of given debug loc and find line number info
    /// for the function.
    ///
    /// FIXME: Remove this.  Users should use DILocation/DILocalScope API to
    /// find the subprogram, and then DILocation::get().
    LLVM_ABI DebugLoc getFnDebugLoc() const;

    /// Return \c this as a bar \a MDNode.
    MDNode *getAsMDNode() const { return Loc; }

    /// Check if the DebugLoc corresponds to an implicit code.
    LLVM_ABI bool isImplicitCode() const;
    LLVM_ABI void setImplicitCode(bool ImplicitCode);

    bool operator==(const DebugLoc &DL) const { return Loc == DL.Loc; }
    bool operator!=(const DebugLoc &DL) const { return Loc != DL.Loc; }

    LLVM_ABI void dump() const;

    /// prints source location /path/to/file.exe:line:col @[inlined at]
    LLVM_ABI void print(raw_ostream &OS) const;
  };

} // end namespace llvm

#endif // LLVM_IR_DEBUGLOC_H
