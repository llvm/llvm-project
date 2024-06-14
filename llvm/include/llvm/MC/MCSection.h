//===- MCSection.h - Machine Code Sections ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSection class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTION_H
#define LLVM_MC_MCSECTION_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCFragment.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Support/Alignment.h"
#include <cassert>
#include <utility>

namespace llvm {

class MCAsmInfo;
class MCAssembler;
class MCContext;
class MCExpr;
class MCSymbol;
class raw_ostream;
class Triple;

/// Instances of this class represent a uniqued identifier for a section in the
/// current translation unit.  The MCContext class uniques and creates these.
class MCSection {
public:
  friend MCAssembler;
  static constexpr unsigned NonUniqueID = ~0U;

  enum SectionVariant {
    SV_COFF = 0,
    SV_ELF,
    SV_GOFF,
    SV_MachO,
    SV_Wasm,
    SV_XCOFF,
    SV_SPIRV,
    SV_DXContainer,
  };

  /// Express the state of bundle locked groups while emitting code.
  enum BundleLockStateType {
    NotBundleLocked,
    BundleLocked,
    BundleLockedAlignToEnd
  };

  struct iterator {
    MCFragment *F = nullptr;
    iterator() = default;
    explicit iterator(MCFragment *F) : F(F) {}
    MCFragment &operator*() const { return *F; }
    bool operator==(const iterator &O) const { return F == O.F; }
    bool operator!=(const iterator &O) const { return F != O.F; }
    iterator &operator++() {
      F = F->Next;
      return *this;
    }
    iterator operator++(int) { return iterator(F->Next); }
  };

  struct FragList {
    MCFragment *Head = nullptr;
    MCFragment *Tail = nullptr;
  };

private:
  // At parse time, this holds the fragment list of the current subsection. At
  // layout time, this holds the concatenated fragment lists of all subsections.
  FragList *CurFragList;
  MCSymbol *Begin;
  MCSymbol *End = nullptr;
  /// The alignment requirement of this section.
  Align Alignment;
  /// The section index in the assemblers section list.
  unsigned Ordinal = 0;
  /// The index of this section in the layout order.
  unsigned LayoutOrder = 0;

  /// Keeping track of bundle-locked state.
  BundleLockStateType BundleLockState = NotBundleLocked;

  /// Current nesting depth of bundle_lock directives.
  unsigned BundleLockNestingDepth = 0;

  /// We've seen a bundle_lock directive but not its first instruction
  /// yet.
  bool BundleGroupBeforeFirstInst : 1;

  /// Whether this section has had instructions emitted into it.
  bool HasInstructions : 1;

  bool HasLayout : 1;

  bool IsRegistered : 1;

  MCDummyFragment DummyFragment;

  // Mapping from subsection number to fragment list. At layout time, the
  // subsection 0 list is replaced with concatenated fragments from all
  // subsections.
  SmallVector<std::pair<unsigned, FragList>, 1> Subsections;

  /// State for tracking labels that don't yet have Fragments
  struct PendingLabel {
    MCSymbol* Sym;
    unsigned Subsection;
    PendingLabel(MCSymbol* Sym, unsigned Subsection = 0)
      : Sym(Sym), Subsection(Subsection) {}
  };
  SmallVector<PendingLabel, 2> PendingLabels;

protected:
  // TODO Make Name private when possible.
  StringRef Name;
  SectionVariant Variant;
  SectionKind Kind;

  MCSection(SectionVariant V, StringRef Name, SectionKind K, MCSymbol *Begin);
  ~MCSection();

public:
  MCSection(const MCSection &) = delete;
  MCSection &operator=(const MCSection &) = delete;

  StringRef getName() const { return Name; }
  SectionKind getKind() const { return Kind; }

  SectionVariant getVariant() const { return Variant; }

  MCSymbol *getBeginSymbol() { return Begin; }
  const MCSymbol *getBeginSymbol() const {
    return const_cast<MCSection *>(this)->getBeginSymbol();
  }
  void setBeginSymbol(MCSymbol *Sym) {
    assert(!Begin);
    Begin = Sym;
  }
  MCSymbol *getEndSymbol(MCContext &Ctx);
  bool hasEnded() const;

  Align getAlign() const { return Alignment; }
  void setAlignment(Align Value) { Alignment = Value; }

  /// Makes sure that Alignment is at least MinAlignment.
  void ensureMinAlignment(Align MinAlignment) {
    if (Alignment < MinAlignment)
      Alignment = MinAlignment;
  }

  unsigned getOrdinal() const { return Ordinal; }
  void setOrdinal(unsigned Value) { Ordinal = Value; }

  unsigned getLayoutOrder() const { return LayoutOrder; }
  void setLayoutOrder(unsigned Value) { LayoutOrder = Value; }

  BundleLockStateType getBundleLockState() const { return BundleLockState; }
  void setBundleLockState(BundleLockStateType NewState);
  bool isBundleLocked() const { return BundleLockState != NotBundleLocked; }

  bool isBundleGroupBeforeFirstInst() const {
    return BundleGroupBeforeFirstInst;
  }
  void setBundleGroupBeforeFirstInst(bool IsFirst) {
    BundleGroupBeforeFirstInst = IsFirst;
  }

  bool hasInstructions() const { return HasInstructions; }
  void setHasInstructions(bool Value) { HasInstructions = Value; }

  bool hasLayout() const { return HasLayout; }
  void setHasLayout(bool Value) { HasLayout = Value; }

  bool isRegistered() const { return IsRegistered; }
  void setIsRegistered(bool Value) { IsRegistered = Value; }

  const MCDummyFragment &getDummyFragment() const { return DummyFragment; }
  MCDummyFragment &getDummyFragment() { return DummyFragment; }

  FragList *curFragList() const { return CurFragList; }
  iterator begin() const { return iterator(CurFragList->Head); }
  iterator end() const { return {}; }
  bool empty() const { return !CurFragList->Head; }

  void addFragment(MCFragment &F) {
    // The formal layout order will be finalized in MCAssembler::layout.
    if (CurFragList->Tail) {
      CurFragList->Tail->Next = &F;
      F.setLayoutOrder(CurFragList->Tail->getLayoutOrder() + 1);
    } else {
      CurFragList->Head = &F;
      assert(F.getLayoutOrder() == 0);
    }
    CurFragList->Tail = &F;
  }

  void switchSubsection(unsigned Subsection);

  void dump() const;

  virtual void printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                                    raw_ostream &OS,
                                    const MCExpr *Subsection) const = 0;

  /// Return true if a .align directive should use "optimized nops" to fill
  /// instead of 0s.
  virtual bool useCodeAlign() const = 0;

  /// Check whether this section is "virtual", that is has no actual object
  /// file contents.
  virtual bool isVirtualSection() const = 0;

  virtual StringRef getVirtualSectionKind() const;

  /// Add a pending label for the requested subsection. This label will be
  /// associated with a fragment in flushPendingLabels()
  void addPendingLabel(MCSymbol* label, unsigned Subsection = 0);

  /// Associate all pending labels in a subsection with a fragment.
  void flushPendingLabels(MCFragment *F, uint64_t FOffset = 0,
			  unsigned Subsection = 0);

  /// Associate all pending labels with empty data fragments. One fragment
  /// will be created for each subsection as necessary.
  void flushPendingLabels();
};

} // end namespace llvm

#endif // LLVM_MC_MCSECTION_H
