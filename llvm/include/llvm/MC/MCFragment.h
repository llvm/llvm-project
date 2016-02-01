//===- MCFragment.h - Fragment type hierarchy -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCFRAGMENT_H
#define LLVM_MC_MCFRAGMENT_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"

namespace llvm {
class MCSection;
class MCSymbol;
class MCSubtargetInfo;

class MCFragment : public ilist_node_with_parent<MCFragment, MCSection> {
  friend class MCAsmLayout;

  MCFragment(const MCFragment &) = delete;
  void operator=(const MCFragment &) = delete;

public:
  enum FragmentType : uint8_t {
    FT_Align,
    FT_Data,
    FT_CompactEncodedInst,
    FT_Fill,
    FT_Relaxable,
    FT_Org,
    FT_Dwarf,
    FT_DwarfFrame,
    FT_LEB,
    FT_SafeSEH,
    FT_Dummy
  };

private:
  FragmentType Kind;

protected:
  bool HasInstructions;

private:
  /// \brief Should this fragment be aligned to the end of a bundle?
  bool AlignToBundleEnd;

  uint8_t BundlePadding;

  /// LayoutOrder - The layout order of this fragment.
  unsigned LayoutOrder;

  /// The data for the section this fragment is in.
  MCSection *Parent;

  /// Atom - The atom this fragment is in, as represented by it's defining
  /// symbol.
  const MCSymbol *Atom;

  /// \name Assembler Backend Data
  /// @{
  //
  // FIXME: This could all be kept private to the assembler implementation.

  /// Offset - The offset of this fragment in its section. This is ~0 until
  /// initialized.
  uint64_t Offset;

  /// @}

protected:
  MCFragment(FragmentType Kind, bool HasInstructions,
             uint8_t BundlePadding, MCSection *Parent = nullptr);

  ~MCFragment();
private:

  // This is a friend so that the sentinal can be created.
  friend struct ilist_sentinel_traits<MCFragment>;
  MCFragment();

public:
  /// Destroys the current fragment.
  ///
  /// This must be used instead of delete as MCFragment is non-virtual.
  /// This method will dispatch to the appropriate subclass.
  void destroy();

  FragmentType getKind() const { return Kind; }

  MCSection *getParent() const { return Parent; }
  void setParent(MCSection *Value) { Parent = Value; }

  const MCSymbol *getAtom() const { return Atom; }
  void setAtom(const MCSymbol *Value) { Atom = Value; }

  unsigned getLayoutOrder() const { return LayoutOrder; }
  void setLayoutOrder(unsigned Value) { LayoutOrder = Value; }

  /// \brief Does this fragment have instructions emitted into it? By default
  /// this is false, but specific fragment types may set it to true.
  bool hasInstructions() const { return HasInstructions; }

  /// \brief Should this fragment be placed at the end of an aligned bundle?
  bool alignToBundleEnd() const { return AlignToBundleEnd; }
  void setAlignToBundleEnd(bool V) { AlignToBundleEnd = V; }

  /// \brief Get the padding size that must be inserted before this fragment.
  /// Used for bundling. By default, no padding is inserted.
  /// Note that padding size is restricted to 8 bits. This is an optimization
  /// to reduce the amount of space used for each fragment. In practice, larger
  /// padding should never be required.
  uint8_t getBundlePadding() const { return BundlePadding; }

  /// \brief Set the padding size for this fragment. By default it's a no-op,
  /// and only some fragments have a meaningful implementation.
  void setBundlePadding(uint8_t N) { BundlePadding = N; }

  /// \brief Return true if given frgment has FT_Dummy type.
  bool isDummy() const { return Kind == FT_Dummy; }

  void dump();
};

class MCDummyFragment : public MCFragment {
public:
  explicit MCDummyFragment(MCSection *Sec)
      : MCFragment(FT_Dummy, false, 0, Sec){};
  static bool classof(const MCFragment *F) { return F->getKind() == FT_Dummy; }
};

/// Interface implemented by fragments that contain encoded instructions and/or
/// data.
///
class MCEncodedFragment : public MCFragment {
protected:
  MCEncodedFragment(MCFragment::FragmentType FType, bool HasInstructions,
                    MCSection *Sec)
      : MCFragment(FType, HasInstructions, 0, Sec) {}

public:
  static bool classof(const MCFragment *F) {
    MCFragment::FragmentType Kind = F->getKind();
    switch (Kind) {
    default:
      return false;
    case MCFragment::FT_Relaxable:
    case MCFragment::FT_CompactEncodedInst:
    case MCFragment::FT_Data:
      return true;
    }
  }
};

/// Interface implemented by fragments that contain encoded instructions and/or
/// data.
///
template<unsigned ContentsSize>
class MCEncodedFragmentWithContents : public MCEncodedFragment {
  SmallVector<char, ContentsSize> Contents;

protected:
  MCEncodedFragmentWithContents(MCFragment::FragmentType FType,
                                bool HasInstructions,
                                MCSection *Sec)
      : MCEncodedFragment(FType, HasInstructions, Sec) {}

public:
  SmallVectorImpl<char> &getContents() { return Contents; }
  const SmallVectorImpl<char> &getContents() const { return Contents; }
};

/// Interface implemented by fragments that contain encoded instructions and/or
/// data and also have fixups registered.
///
template<unsigned ContentsSize, unsigned FixupsSize>
class MCEncodedFragmentWithFixups :
  public MCEncodedFragmentWithContents<ContentsSize> {

  /// Fixups - The list of fixups in this fragment.
  SmallVector<MCFixup, FixupsSize> Fixups;

protected:
  MCEncodedFragmentWithFixups(MCFragment::FragmentType FType,
                              bool HasInstructions,
                              MCSection *Sec)
      : MCEncodedFragmentWithContents<ContentsSize>(FType, HasInstructions,
                                                    Sec) {}

public:
  typedef SmallVectorImpl<MCFixup>::const_iterator const_fixup_iterator;
  typedef SmallVectorImpl<MCFixup>::iterator fixup_iterator;

  SmallVectorImpl<MCFixup> &getFixups() { return Fixups; }
  const SmallVectorImpl<MCFixup> &getFixups() const { return Fixups; }

  fixup_iterator fixup_begin() { return Fixups.begin(); }
  const_fixup_iterator fixup_begin() const { return Fixups.begin(); }

  fixup_iterator fixup_end() { return Fixups.end(); }
  const_fixup_iterator fixup_end() const { return Fixups.end(); }

  static bool classof(const MCFragment *F) {
    MCFragment::FragmentType Kind = F->getKind();
    return Kind == MCFragment::FT_Relaxable || Kind == MCFragment::FT_Data;
  }
};

/// Fragment for data and encoded instructions.
///
class MCDataFragment : public MCEncodedFragmentWithFixups<32, 4> {
public:
  MCDataFragment(MCSection *Sec = nullptr)
      : MCEncodedFragmentWithFixups<32, 4>(FT_Data, false, Sec) {}

  void setHasInstructions(bool V) { HasInstructions = V; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Data;
  }
};

/// This is a compact (memory-size-wise) fragment for holding an encoded
/// instruction (non-relaxable) that has no fixups registered. When applicable,
/// it can be used instead of MCDataFragment and lead to lower memory
/// consumption.
///
class MCCompactEncodedInstFragment : public MCEncodedFragmentWithContents<4> {
public:
  MCCompactEncodedInstFragment(MCSection *Sec = nullptr)
      : MCEncodedFragmentWithContents(FT_CompactEncodedInst, true, Sec) {
  }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_CompactEncodedInst;
  }
};

/// A relaxable fragment holds on to its MCInst, since it may need to be
/// relaxed during the assembler layout and relaxation stage.
///
class MCRelaxableFragment : public MCEncodedFragmentWithFixups<8, 1> {

  /// Inst - The instruction this is a fragment for.
  MCInst Inst;

  /// STI - The MCSubtargetInfo in effect when the instruction was encoded.
  const MCSubtargetInfo &STI;

public:
  MCRelaxableFragment(const MCInst &Inst, const MCSubtargetInfo &STI,
                      MCSection *Sec = nullptr)
      : MCEncodedFragmentWithFixups(FT_Relaxable, true, Sec),
        Inst(Inst), STI(STI) {}

  const MCInst &getInst() const { return Inst; }
  void setInst(const MCInst &Value) { Inst = Value; }

  const MCSubtargetInfo &getSubtargetInfo() { return STI; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Relaxable;
  }
};

class MCAlignFragment : public MCFragment {

  /// Alignment - The alignment to ensure, in bytes.
  unsigned Alignment;

  /// EmitNops - Flag to indicate that (optimal) NOPs should be emitted instead
  /// of using the provided value. The exact interpretation of this flag is
  /// target dependent.
  bool EmitNops : 1;

  /// Value - Value to use for filling padding bytes.
  int64_t Value;

  /// ValueSize - The size of the integer (in bytes) of \p Value.
  unsigned ValueSize;

  /// MaxBytesToEmit - The maximum number of bytes to emit; if the alignment
  /// cannot be satisfied in this width then this fragment is ignored.
  unsigned MaxBytesToEmit;

public:
  MCAlignFragment(unsigned Alignment, int64_t Value, unsigned ValueSize,
                  unsigned MaxBytesToEmit, MCSection *Sec = nullptr)
      : MCFragment(FT_Align, false, 0, Sec), Alignment(Alignment),
        EmitNops(false), Value(Value),
        ValueSize(ValueSize), MaxBytesToEmit(MaxBytesToEmit) {}

  /// \name Accessors
  /// @{

  unsigned getAlignment() const { return Alignment; }

  int64_t getValue() const { return Value; }

  unsigned getValueSize() const { return ValueSize; }

  unsigned getMaxBytesToEmit() const { return MaxBytesToEmit; }

  bool hasEmitNops() const { return EmitNops; }
  void setEmitNops(bool Value) { EmitNops = Value; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Align;
  }
};

class MCFillFragment : public MCFragment {

  /// Value to use for filling bytes.
  uint8_t Value;

  /// The number of bytes to insert.
  uint64_t Size;

public:
  MCFillFragment(uint8_t Value, uint64_t Size, MCSection *Sec = nullptr)
      : MCFragment(FT_Fill, false, 0, Sec), Value(Value), Size(Size) {}

  uint8_t getValue() const { return Value; }
  uint64_t getSize() const { return Size; }

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Fill;
  }
};

class MCOrgFragment : public MCFragment {

  /// Offset - The offset this fragment should start at.
  const MCExpr *Offset;

  /// Value - Value to use for filling bytes.
  int8_t Value;

public:
  MCOrgFragment(const MCExpr &Offset, int8_t Value, MCSection *Sec = nullptr)
      : MCFragment(FT_Org, false, 0, Sec), Offset(&Offset), Value(Value) {}

  /// \name Accessors
  /// @{

  const MCExpr &getOffset() const { return *Offset; }

  uint8_t getValue() const { return Value; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Org;
  }
};

class MCLEBFragment : public MCFragment {

  /// Value - The value this fragment should contain.
  const MCExpr *Value;

  /// IsSigned - True if this is a sleb128, false if uleb128.
  bool IsSigned;

  SmallString<8> Contents;

public:
  MCLEBFragment(const MCExpr &Value_, bool IsSigned_, MCSection *Sec = nullptr)
      : MCFragment(FT_LEB, false, 0, Sec), Value(&Value_), IsSigned(IsSigned_) {
    Contents.push_back(0);
  }

  /// \name Accessors
  /// @{

  const MCExpr &getValue() const { return *Value; }

  bool isSigned() const { return IsSigned; }

  SmallString<8> &getContents() { return Contents; }
  const SmallString<8> &getContents() const { return Contents; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_LEB;
  }
};

class MCDwarfLineAddrFragment : public MCFragment {

  /// LineDelta - the value of the difference between the two line numbers
  /// between two .loc dwarf directives.
  int64_t LineDelta;

  /// AddrDelta - The expression for the difference of the two symbols that
  /// make up the address delta between two .loc dwarf directives.
  const MCExpr *AddrDelta;

  SmallString<8> Contents;

public:
  MCDwarfLineAddrFragment(int64_t LineDelta, const MCExpr &AddrDelta,
                          MCSection *Sec = nullptr)
      : MCFragment(FT_Dwarf, false, 0, Sec), LineDelta(LineDelta),
        AddrDelta(&AddrDelta) {
    Contents.push_back(0);
  }

  /// \name Accessors
  /// @{

  int64_t getLineDelta() const { return LineDelta; }

  const MCExpr &getAddrDelta() const { return *AddrDelta; }

  SmallString<8> &getContents() { return Contents; }
  const SmallString<8> &getContents() const { return Contents; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_Dwarf;
  }
};

class MCDwarfCallFrameFragment : public MCFragment {

  /// AddrDelta - The expression for the difference of the two symbols that
  /// make up the address delta between two .cfi_* dwarf directives.
  const MCExpr *AddrDelta;

  SmallString<8> Contents;

public:
  MCDwarfCallFrameFragment(const MCExpr &AddrDelta, MCSection *Sec = nullptr)
      : MCFragment(FT_DwarfFrame, false, 0, Sec), AddrDelta(&AddrDelta) {
    Contents.push_back(0);
  }

  /// \name Accessors
  /// @{

  const MCExpr &getAddrDelta() const { return *AddrDelta; }

  SmallString<8> &getContents() { return Contents; }
  const SmallString<8> &getContents() const { return Contents; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_DwarfFrame;
  }
};

class MCSafeSEHFragment : public MCFragment {
  const MCSymbol *Sym;

public:
  MCSafeSEHFragment(const MCSymbol *Sym, MCSection *Sec = nullptr)
      : MCFragment(FT_SafeSEH, false, 0, Sec), Sym(Sym) {}

  /// \name Accessors
  /// @{

  const MCSymbol *getSymbol() { return Sym; }
  const MCSymbol *getSymbol() const { return Sym; }

  /// @}

  static bool classof(const MCFragment *F) {
    return F->getKind() == MCFragment::FT_SafeSEH;
  }
};

} // end namespace llvm

#endif
