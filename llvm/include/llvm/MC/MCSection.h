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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <utility>

namespace llvm {

class MCAsmInfo;
class MCAssembler;
class MCContext;
class MCExpr;
class MCFragment;
class MCObjectStreamer;
class MCSymbol;
class MCSection;
class MCSubtargetInfo;
class raw_ostream;
class Triple;

/// Instances of this class represent a uniqued identifier for a section in the
/// current translation unit.  The MCContext class uniques and creates these.
class LLVM_ABI MCSection {
public:
  friend MCAssembler;
  friend MCObjectStreamer;
  friend class MCFragment;
  friend class MCEncodedFragment;
  friend class MCRelaxableFragment;
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
    iterator &operator++();
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

  /// Keeping track of bundle-locked state.
  BundleLockStateType BundleLockState = NotBundleLocked;

  /// Current nesting depth of bundle_lock directives.
  unsigned BundleLockNestingDepth = 0;

  /// We've seen a bundle_lock directive but not its first instruction
  /// yet.
  bool BundleGroupBeforeFirstInst : 1;

  /// Whether this section has had instructions emitted into it.
  bool HasInstructions : 1;

  bool IsRegistered : 1;

  bool IsText : 1;

  bool IsVirtual : 1;

  /// Whether the section contains linker-relaxable fragments. If true, the
  /// offset between two locations may not be fully resolved.
  bool LinkerRelaxable : 1;

  // Mapping from subsection number to fragment list. At layout time, the
  // subsection 0 list is replaced with concatenated fragments from all
  // subsections.
  SmallVector<std::pair<unsigned, FragList>, 1> Subsections;

  // Content and fixup storage for fragments
  SmallVector<char, 0> ContentStorage;
  SmallVector<MCFixup, 0> FixupStorage;
  SmallVector<MCOperand, 0> MCOperandStorage;

protected:
  // TODO Make Name private when possible.
  StringRef Name;
  SectionVariant Variant;

  MCSection(SectionVariant V, StringRef Name, bool IsText, bool IsVirtual,
            MCSymbol *Begin);
  // Protected non-virtual dtor prevents destroy through a base class pointer.
  ~MCSection() {}

public:
  MCSection(const MCSection &) = delete;
  MCSection &operator=(const MCSection &) = delete;

  StringRef getName() const { return Name; }
  bool isText() const { return IsText; }

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

  bool isRegistered() const { return IsRegistered; }
  void setIsRegistered(bool Value) { IsRegistered = Value; }

  bool isLinkerRelaxable() const { return LinkerRelaxable; }
  void setLinkerRelaxable() { LinkerRelaxable = true; }

  MCFragment &getDummyFragment() { return *Subsections[0].second.Head; }

  FragList *curFragList() const { return CurFragList; }
  iterator begin() const { return iterator(CurFragList->Head); }
  iterator end() const { return {}; }

  void dump(DenseMap<const MCFragment *, SmallVector<const MCSymbol *, 0>>
                *FragToSyms = nullptr) const;

  virtual void printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                                    raw_ostream &OS,
                                    uint32_t Subsection) const = 0;

  /// Return true if a .align directive should use "optimized nops" to fill
  /// instead of 0s.
  virtual bool useCodeAlign() const = 0;

  /// Check whether this section is "virtual", that is has no actual object
  /// file contents.
  bool isVirtualSection() const { return IsVirtual; }

  virtual StringRef getVirtualSectionKind() const;
};

// Represents a contiguous piece of code or data within a section. Its size is
// determined by MCAssembler::layout. All subclasses must have trivial
// destructors.
class MCFragment {
  friend class MCAssembler;
  friend class MCObjectStreamer;
  friend class MCSection;

public:
  enum FragmentType : uint8_t {
    FT_Data,
    FT_Relaxable,
    FT_Align,
    FT_Fill,
    FT_LEB,
    FT_Nops,
    FT_Org,
    FT_Dwarf,
    FT_DwarfFrame,
    FT_BoundaryAlign,
    FT_SymbolId,
    FT_CVInlineLines,
    FT_CVDefRange,
    FT_PseudoProbe,
  };

private:
  // The next fragment within the section.
  MCFragment *Next = nullptr;

  /// The data for the section this fragment is in.
  MCSection *Parent = nullptr;

  /// The offset of this fragment in its section.
  uint64_t Offset = 0;

  /// The layout order of this fragment.
  unsigned LayoutOrder = 0;

  FragmentType Kind;

protected:
  /// Used by subclasses for better packing.
  ///
  /// MCEncodedFragment
  bool HasInstructions : 1;
  bool AlignToBundleEnd : 1;
  /// MCDataFragment
  bool LinkerRelaxable : 1;
  /// MCRelaxableFragment: x86-specific
  bool AllowAutoPadding : 1;

  LLVM_ABI MCFragment(FragmentType Kind, bool HasInstructions);

public:
  MCFragment() = delete;
  MCFragment(const MCFragment &) = delete;
  MCFragment &operator=(const MCFragment &) = delete;

  MCFragment *getNext() const { return Next; }

  FragmentType getKind() const { return Kind; }

  MCSection *getParent() const { return Parent; }
  void setParent(MCSection *Value) { Parent = Value; }

  LLVM_ABI const MCSymbol *getAtom() const;

  unsigned getLayoutOrder() const { return LayoutOrder; }
  void setLayoutOrder(unsigned Value) { LayoutOrder = Value; }

  /// Does this fragment have instructions emitted into it? By default
  /// this is false, but specific fragment types may set it to true.
  bool hasInstructions() const { return HasInstructions; }

  bool isLinkerRelaxable() const { return LinkerRelaxable; }
  void setLinkerRelaxable() { LinkerRelaxable = true; }

  LLVM_ABI void dump() const;
};

/// Interface implemented by fragments that contain encoded instructions and/or
/// data.
class MCEncodedFragment : public MCFragment {
  uint8_t BundlePadding = 0;
  uint32_t ContentStart = 0;
  uint32_t ContentEnd = 0;
  uint32_t FixupStart = 0;
  uint32_t FixupEnd = 0;

protected:
  MCEncodedFragment(MCFragment::FragmentType FType, bool HasInstructions)
      : MCFragment(FType, HasInstructions) {}

  /// The MCSubtargetInfo in effect when the instruction was encoded.
  /// It must be non-null for instructions.
  const MCSubtargetInfo *STI = nullptr;

public:
  static bool classof(const MCFragment *F) {
    MCFragment::FragmentType Kind = F->getKind();
    switch (Kind) {
    default:
      return false;
    case MCFragment::FT_Relaxable:
    case MCFragment::FT_Data:
    case MCFragment::FT_Dwarf:
    case MCFragment::FT_DwarfFrame:
    case MCFragment::FT_LEB:
    case MCFragment::FT_PseudoProbe:
    case MCFragment::FT_CVInlineLines:
    case MCFragment::FT_CVDefRange:
      return true;
    }
  }

  /// Should this fragment be placed at the end of an aligned bundle?
  bool alignToBundleEnd() const { return AlignToBundleEnd; }
  void setAlignToBundleEnd(bool V) { AlignToBundleEnd = V; }

  /// Get the padding size that must be inserted before this fragment.
  /// Used for bundling. By default, no padding is inserted.
  /// Note that padding size is restricted to 8 bits. This is an optimization
  /// to reduce the amount of space used for each fragment. In practice, larger
  /// padding should never be required.
  uint8_t getBundlePadding() const { return BundlePadding; }

  /// Set the padding size for this fragment. By default it's a no-op,
  /// and only some fragments have a meaningful implementation.
  void setBundlePadding(uint8_t N) { BundlePadding = N; }

  /// Retrieve the MCSubTargetInfo in effect when the instruction was encoded.
  /// Guaranteed to be non-null if hasInstructions() == true
  const MCSubtargetInfo *getSubtargetInfo() const { return STI; }

  /// Record that the fragment contains instructions with the MCSubtargetInfo in
  /// effect when the instruction was encoded.
  void setHasInstructions(const MCSubtargetInfo &STI) {
    HasInstructions = true;
    this->STI = &STI;
  }

  bool getAllowAutoPadding() const { return AllowAutoPadding; }
  void setAllowAutoPadding(bool V) { AllowAutoPadding = V; }

  // Content-related functions manage parent's storage using ContentStart and
  // ContentSize.
  void clearContents() { ContentEnd = ContentStart; }
  // Get a SmallVector reference. The caller should call doneAppending to update
  // `ContentEnd`.
  SmallVectorImpl<char> &getContentsForAppending() {
    SmallVectorImpl<char> &S = getParent()->ContentStorage;
    if (LLVM_UNLIKELY(ContentEnd != S.size())) {
      // Move the elements to the end. Reserve space to avoid invalidating
      // S.begin()+I for `append`.
      auto Size = ContentEnd - ContentStart;
      auto I = std::exchange(ContentStart, S.size());
      S.reserve(S.size() + Size);
      S.append(S.begin() + I, S.begin() + I + Size);
    }
    return S;
  }
  void doneAppending() { ContentEnd = getParent()->ContentStorage.size(); }
  void appendContents(ArrayRef<char> Contents) {
    getContentsForAppending().append(Contents.begin(), Contents.end());
    doneAppending();
  }
  void appendContents(size_t Num, char Elt) {
    getContentsForAppending().append(Num, Elt);
    doneAppending();
  }
  LLVM_ABI void setContents(ArrayRef<char> Contents);
  MutableArrayRef<char> getContents() {
    return MutableArrayRef(getParent()->ContentStorage)
        .slice(ContentStart, ContentEnd - ContentStart);
  }
  ArrayRef<char> getContents() const {
    return ArrayRef(getParent()->ContentStorage)
        .slice(ContentStart, ContentEnd - ContentStart);
  }

  // Fixup-related functions manage parent's storage using FixupStart and
  // FixupSize.
  void clearFixups() { FixupEnd = FixupStart; }
  LLVM_ABI void addFixup(MCFixup Fixup);
  LLVM_ABI void appendFixups(ArrayRef<MCFixup> Fixups);
  LLVM_ABI void setFixups(ArrayRef<MCFixup> Fixups);
  MutableArrayRef<MCFixup> getFixups() {
    return MutableArrayRef(getParent()->FixupStorage)
        .slice(FixupStart, FixupEnd - FixupStart);
  }
  ArrayRef<MCFixup> getFixups() const {
    return ArrayRef(getParent()->FixupStorage)
        .slice(FixupStart, FixupEnd - FixupStart);
  }

  size_t getSize() const { return ContentEnd - ContentStart; }
};

/// Fragment for data and encoded instructions.
///
class MCDataFragment : public MCEncodedFragment {
public:
  MCDataFragment() : MCEncodedFragment(FT_Data, false) {}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Data;
  }
};

/// A relaxable fragment holds on to its MCInst, since it may need to be
/// relaxed during the assembler layout and relaxation stage.
///
class MCRelaxableFragment : public MCEncodedFragment {
  uint32_t Opcode = 0;
  uint32_t Flags = 0;
  uint32_t OperandStart = 0;
  uint32_t OperandSize = 0;

public:
  MCRelaxableFragment(const MCSubtargetInfo &STI)
      : MCEncodedFragment(FT_Relaxable, true) {
    this->STI = &STI;
  }

  unsigned getOpcode() const { return Opcode; }
  ArrayRef<MCOperand> getOperands() const {
    return MutableArrayRef(getParent()->MCOperandStorage)
        .slice(OperandStart, OperandSize);
  }
  MCInst getInst() const {
    MCInst Inst;
    Inst.setOpcode(Opcode);
    Inst.setFlags(Flags);
    Inst.setOperands(ArrayRef(getParent()->MCOperandStorage)
                         .slice(OperandStart, OperandSize));
    return Inst;
  }
  void setInst(const MCInst &Inst) {
    Opcode = Inst.getOpcode();
    Flags = Inst.getFlags();
    auto &S = getParent()->MCOperandStorage;
    if (Inst.getNumOperands() > OperandSize) {
      OperandStart = S.size();
      S.resize_for_overwrite(S.size() + Inst.getNumOperands());
    }
    OperandSize = Inst.getNumOperands();
    llvm::copy(Inst, S.begin() + OperandStart);
  }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Relaxable;
  }
};

class MCAlignFragment : public MCFragment {
  /// Flag to indicate that (optimal) NOPs should be emitted instead
  /// of using the provided value. The exact interpretation of this flag is
  /// target dependent.
  bool EmitNops : 1;

  /// The alignment to ensure, in bytes.
  Align Alignment;

  /// The size of the integer (in bytes) of \p Value.
  uint8_t FillLen;

  /// The maximum number of bytes to emit; if the alignment
  /// cannot be satisfied in this width then this fragment is ignored.
  unsigned MaxBytesToEmit;

  /// Value to use for filling padding bytes.
  int64_t Fill;

  /// When emitting Nops some subtargets have specific nop encodings.
  const MCSubtargetInfo *STI = nullptr;

public:
  MCAlignFragment(Align Alignment, int64_t Fill, uint8_t FillLen,
                  unsigned MaxBytesToEmit)
      : MCFragment(FT_Align, false), EmitNops(false), Alignment(Alignment),
        FillLen(FillLen), MaxBytesToEmit(MaxBytesToEmit), Fill(Fill) {}

  Align getAlignment() const { return Alignment; }
  int64_t getFill() const { return Fill; }
  uint8_t getFillLen() const { return FillLen; }
  unsigned getMaxBytesToEmit() const { return MaxBytesToEmit; }

  bool hasEmitNops() const { return EmitNops; }
  void setEmitNops(bool Value, const MCSubtargetInfo *STI) {
    EmitNops = Value;
    this->STI = STI;
  }

  const MCSubtargetInfo *getSubtargetInfo() const { return STI; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Align;
  }
};

class MCFillFragment : public MCFragment {
  uint8_t ValueSize;
  /// Value to use for filling bytes.
  uint64_t Value;
  /// The number of bytes to insert.
  const MCExpr &NumValues;
  uint64_t Size = 0;

  /// Source location of the directive that this fragment was created for.
  SMLoc Loc;

public:
  MCFillFragment(uint64_t Value, uint8_t VSize, const MCExpr &NumValues,
                 SMLoc Loc)
      : MCFragment(FT_Fill, false), ValueSize(VSize), Value(Value),
        NumValues(NumValues), Loc(Loc) {}

  uint64_t getValue() const { return Value; }
  uint8_t getValueSize() const { return ValueSize; }
  const MCExpr &getNumValues() const { return NumValues; }
  uint64_t getSize() const { return Size; }
  void setSize(uint64_t Value) { Size = Value; }

  SMLoc getLoc() const { return Loc; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Fill;
  }
};

class MCNopsFragment : public MCFragment {
  /// The number of bytes to insert.
  int64_t Size;
  /// Maximum number of bytes allowed in each NOP instruction.
  int64_t ControlledNopLength;

  /// Source location of the directive that this fragment was created for.
  SMLoc Loc;

  /// When emitting Nops some subtargets have specific nop encodings.
  const MCSubtargetInfo &STI;

public:
  MCNopsFragment(int64_t NumBytes, int64_t ControlledNopLength, SMLoc L,
                 const MCSubtargetInfo &STI)
      : MCFragment(FT_Nops, false), Size(NumBytes),
        ControlledNopLength(ControlledNopLength), Loc(L), STI(STI) {}

  int64_t getNumBytes() const { return Size; }
  int64_t getControlledNopLength() const { return ControlledNopLength; }

  SMLoc getLoc() const { return Loc; }

  const MCSubtargetInfo *getSubtargetInfo() const { return &STI; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Nops;
  }
};

class MCOrgFragment : public MCFragment {
  /// Value to use for filling bytes.
  int8_t Value;

  /// The offset this fragment should start at.
  const MCExpr *Offset;

  /// Source location of the directive that this fragment was created for.
  SMLoc Loc;

public:
  MCOrgFragment(const MCExpr &Offset, int8_t Value, SMLoc Loc)
      : MCFragment(FT_Org, false), Value(Value), Offset(&Offset), Loc(Loc) {}

  const MCExpr &getOffset() const { return *Offset; }

  uint8_t getValue() const { return Value; }

  SMLoc getLoc() const { return Loc; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Org;
  }
};

class MCLEBFragment final : public MCEncodedFragment {
  /// True if this is a sleb128, false if uleb128.
  bool IsSigned;

  /// The value this fragment should contain.
  const MCExpr *Value;

public:
  MCLEBFragment(const MCExpr &Value, bool IsSigned)
      : MCEncodedFragment(FT_LEB, false), IsSigned(IsSigned), Value(&Value) {}

  const MCExpr &getValue() const { return *Value; }
  void setValue(const MCExpr *Expr) { Value = Expr; }

  bool isSigned() const { return IsSigned; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_LEB;
  }
};

class MCDwarfLineAddrFragment : public MCEncodedFragment {
  /// The value of the difference between the two line numbers
  /// between two .loc dwarf directives.
  int64_t LineDelta;

  /// The expression for the difference of the two symbols that
  /// make up the address delta between two .loc dwarf directives.
  const MCExpr *AddrDelta;

public:
  MCDwarfLineAddrFragment(int64_t LineDelta, const MCExpr &AddrDelta)
      : MCEncodedFragment(FT_Dwarf, false), LineDelta(LineDelta),
        AddrDelta(&AddrDelta) {}

  int64_t getLineDelta() const { return LineDelta; }

  const MCExpr &getAddrDelta() const { return *AddrDelta; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Dwarf;
  }
};

class MCDwarfCallFrameFragment : public MCEncodedFragment {
  /// The expression for the difference of the two symbols that
  /// make up the address delta between two .cfi_* dwarf directives.
  const MCExpr *AddrDelta;

public:
  MCDwarfCallFrameFragment(const MCExpr &AddrDelta)
      : MCEncodedFragment(FT_DwarfFrame, false), AddrDelta(&AddrDelta) {}

  const MCExpr &getAddrDelta() const { return *AddrDelta; }
  void setAddrDelta(const MCExpr *E) { AddrDelta = E; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_DwarfFrame;
  }
};

/// Represents a symbol table index fragment.
class MCSymbolIdFragment : public MCFragment {
  const MCSymbol *Sym;

public:
  MCSymbolIdFragment(const MCSymbol *Sym)
      : MCFragment(FT_SymbolId, false), Sym(Sym) {}

  const MCSymbol *getSymbol() { return Sym; }
  const MCSymbol *getSymbol() const { return Sym; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_SymbolId;
  }
};

/// Fragment representing the binary annotations produced by the
/// .cv_inline_linetable directive.
class MCCVInlineLineTableFragment : public MCEncodedFragment {
  unsigned SiteFuncId;
  unsigned StartFileId;
  unsigned StartLineNum;
  const MCSymbol *FnStartSym;
  const MCSymbol *FnEndSym;

  /// CodeViewContext has the real knowledge about this format, so let it access
  /// our members.
  friend class CodeViewContext;

public:
  MCCVInlineLineTableFragment(unsigned SiteFuncId, unsigned StartFileId,
                              unsigned StartLineNum, const MCSymbol *FnStartSym,
                              const MCSymbol *FnEndSym)
      : MCEncodedFragment(FT_CVInlineLines, false), SiteFuncId(SiteFuncId),
        StartFileId(StartFileId), StartLineNum(StartLineNum),
        FnStartSym(FnStartSym), FnEndSym(FnEndSym) {}

  const MCSymbol *getFnStartSym() const { return FnStartSym; }
  const MCSymbol *getFnEndSym() const { return FnEndSym; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_CVInlineLines;
  }
};

/// Fragment representing the .cv_def_range directive.
class MCCVDefRangeFragment : public MCEncodedFragment {
  ArrayRef<std::pair<const MCSymbol *, const MCSymbol *>> Ranges;
  StringRef FixedSizePortion;

  /// CodeViewContext has the real knowledge about this format, so let it access
  /// our members.
  friend class CodeViewContext;

public:
  MCCVDefRangeFragment(
      ArrayRef<std::pair<const MCSymbol *, const MCSymbol *>> Ranges,
      StringRef FixedSizePortion)
      : MCEncodedFragment(FT_CVDefRange, false),
        Ranges(Ranges.begin(), Ranges.end()),
        FixedSizePortion(FixedSizePortion) {}

  ArrayRef<std::pair<const MCSymbol *, const MCSymbol *>> getRanges() const {
    return Ranges;
  }

  StringRef getFixedSizePortion() const { return FixedSizePortion; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_CVDefRange;
  }
};

/// Represents required padding such that a particular other set of fragments
/// does not cross a particular power-of-two boundary. The other fragments must
/// follow this one within the same section.
class MCBoundaryAlignFragment : public MCFragment {
  /// The alignment requirement of the branch to be aligned.
  Align AlignBoundary;
  /// The last fragment in the set of fragments to be aligned.
  const MCFragment *LastFragment = nullptr;
  /// The size of the fragment.  The size is lazily set during relaxation, and
  /// is not meaningful before that.
  uint64_t Size = 0;

  /// When emitting Nops some subtargets have specific nop encodings.
  const MCSubtargetInfo &STI;

public:
  MCBoundaryAlignFragment(Align AlignBoundary, const MCSubtargetInfo &STI)
      : MCFragment(FT_BoundaryAlign, false), AlignBoundary(AlignBoundary),
        STI(STI) {}

  uint64_t getSize() const { return Size; }
  void setSize(uint64_t Value) { Size = Value; }

  Align getAlignment() const { return AlignBoundary; }
  void setAlignment(Align Value) { AlignBoundary = Value; }

  const MCFragment *getLastFragment() const { return LastFragment; }
  void setLastFragment(const MCFragment *F) {
    assert(!F || getParent() == F->getParent());
    LastFragment = F;
  }

  const MCSubtargetInfo *getSubtargetInfo() const { return &STI; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_BoundaryAlign;
  }
};

class MCPseudoProbeAddrFragment : public MCEncodedFragment {
  /// The expression for the difference of the two symbols that
  /// make up the address delta between two .pseudoprobe directives.
  const MCExpr *AddrDelta;

public:
  MCPseudoProbeAddrFragment(const MCExpr *AddrDelta)
      : MCEncodedFragment(FT_PseudoProbe, false), AddrDelta(AddrDelta) {}

  const MCExpr &getAddrDelta() const { return *AddrDelta; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_PseudoProbe;
  }
};

inline MCSection::iterator &MCSection::iterator::operator++() {
  F = F->Next;
  return *this;
}

} // end namespace llvm

#endif // LLVM_MC_MCSECTION_H
