//===- MCAssembler.h - Object File Generation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASSEMBLER_H
#define LLVM_MC_MCASSEMBLER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SMLoc.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace llvm {

class MCBoundaryAlignFragment;
class MCCVDefRangeFragment;
class MCCVInlineLineTableFragment;
class MCDwarfCallFrameFragment;
class MCDwarfLineAddrFragment;
class MCEncodedFragment;
class MCFixup;
class MCLEBFragment;
class MCPseudoProbeAddrFragment;
class MCRelaxableFragment;
class MCSymbolRefExpr;
class raw_ostream;
class MCAsmBackend;
class MCContext;
class MCCodeEmitter;
class MCFragment;
class MCObjectWriter;
class MCSection;
class MCValue;

class MCAssembler {
public:
  friend class MCObjectWriter;
  using SectionListType = SmallVector<MCSection *, 0>;
  using const_iterator = pointee_iterator<SectionListType::const_iterator>;

private:
  MCContext &Context;

  std::unique_ptr<MCAsmBackend> Backend;
  std::unique_ptr<MCCodeEmitter> Emitter;
  std::unique_ptr<MCObjectWriter> Writer;

  bool HasLayout = false;
  bool HasFinalLayout = false;
  bool RelaxAll = false;

  SectionListType Sections;

  SmallVector<const MCSymbol *, 0> Symbols;

  mutable SmallVector<std::pair<SMLoc, std::string>, 0> PendingErrors;

  MCDwarfLineTableParams LTParams;

  /// The set of function symbols for which a .thumb_func directive has
  /// been seen.
  //
  // FIXME: We really would like this in target specific code rather than
  // here. Maybe when the relocation stuff moves to target specific,
  // this can go with it? The streamer would need some target specific
  // refactoring too.
  mutable SmallPtrSet<const MCSymbol *, 32> ThumbFuncs;

  /// The bundle alignment size currently set in the assembler.
  ///
  /// By default it's 0, which means bundling is disabled.
  unsigned BundleAlignSize = 0;

  /// Evaluate a fixup to a relocatable expression and the value which should be
  /// placed into the fixup.
  ///
  /// \param F The fragment the fixup is inside.
  /// \param Fixup The fixup to evaluate.
  /// \param Target [out] On return, the relocatable expression the fixup
  /// evaluates to.
  /// \param Value [out] On return, the value of the fixup as currently laid
  /// out.
  /// \param RecordReloc Record relocation if needed.
  /// relocation.
  bool evaluateFixup(const MCFragment &F, MCFixup &Fixup, MCValue &Target,
                     uint64_t &Value, bool RecordReloc,
                     MutableArrayRef<char> Contents) const;

  /// Check whether a fixup can be satisfied, or whether it needs to be relaxed
  /// (increased in size, in order to hold its value correctly).
  bool fixupNeedsRelaxation(const MCRelaxableFragment &, const MCFixup &) const;

  /// Check whether the given fragment needs relaxation.
  bool fragmentNeedsRelaxation(const MCRelaxableFragment &) const;

  void layoutSection(MCSection &Sec);
  /// Perform one layout iteration and return true if any offsets
  /// were adjusted.
  bool relaxOnce();

  /// Perform relaxation on a single fragment.
  bool relaxFragment(MCFragment &F);
  bool relaxInstruction(MCRelaxableFragment &IF);
  bool relaxLEB(MCLEBFragment &IF);
  bool relaxBoundaryAlign(MCBoundaryAlignFragment &BF);
  bool relaxDwarfLineAddr(MCDwarfLineAddrFragment &DF);
  bool relaxDwarfCallFrameFragment(MCDwarfCallFrameFragment &DF);
  bool relaxCVInlineLineTable(MCCVInlineLineTableFragment &DF);
  bool relaxCVDefRange(MCCVDefRangeFragment &DF);
  bool relaxFill(MCFillFragment &F);
  bool relaxPseudoProbeAddr(MCPseudoProbeAddrFragment &DF);

public:
  /// Construct a new assembler instance.
  //
  // FIXME: How are we going to parameterize this? Two obvious options are stay
  // concrete and require clients to pass in a target like object. The other
  // option is to make this abstract, and have targets provide concrete
  // implementations as we do with AsmParser.
  LLVM_ABI MCAssembler(MCContext &Context,
                       std::unique_ptr<MCAsmBackend> Backend,
                       std::unique_ptr<MCCodeEmitter> Emitter,
                       std::unique_ptr<MCObjectWriter> Writer);
  MCAssembler(const MCAssembler &) = delete;
  MCAssembler &operator=(const MCAssembler &) = delete;

  /// Compute the effective fragment size.
  LLVM_ABI uint64_t computeFragmentSize(const MCFragment &F) const;

  LLVM_ABI void layoutBundle(MCFragment *Prev, MCFragment *F) const;

  // Get the offset of the given fragment inside its containing section.
  uint64_t getFragmentOffset(const MCFragment &F) const { return F.Offset; }

  LLVM_ABI uint64_t getSectionAddressSize(const MCSection &Sec) const;
  LLVM_ABI uint64_t getSectionFileSize(const MCSection &Sec) const;

  // Get the offset of the given symbol, as computed in the current
  // layout.
  // \return True on success.
  LLVM_ABI bool getSymbolOffset(const MCSymbol &S, uint64_t &Val) const;

  // Variant that reports a fatal error if the offset is not computable.
  LLVM_ABI uint64_t getSymbolOffset(const MCSymbol &S) const;

  // If this symbol is equivalent to A + Constant, return A.
  LLVM_ABI const MCSymbol *getBaseSymbol(const MCSymbol &Symbol) const;

  /// Emit the section contents to \p OS.
  LLVM_ABI void writeSectionData(raw_ostream &OS,
                                 const MCSection *Section) const;

  /// Check whether a given symbol has been flagged with .thumb_func.
  LLVM_ABI bool isThumbFunc(const MCSymbol *Func) const;

  /// Flag a function symbol as the target of a .thumb_func directive.
  void setIsThumbFunc(const MCSymbol *Func) { ThumbFuncs.insert(Func); }

  /// Reuse an assembler instance
  ///
  LLVM_ABI void reset();

  MCContext &getContext() const { return Context; }

  MCAsmBackend *getBackendPtr() const { return Backend.get(); }

  MCCodeEmitter *getEmitterPtr() const { return Emitter.get(); }

  MCAsmBackend &getBackend() const { return *Backend; }

  MCCodeEmitter &getEmitter() const { return *Emitter; }

  MCObjectWriter &getWriter() const { return *Writer; }

  MCDwarfLineTableParams getDWARFLinetableParams() const { return LTParams; }

  /// Finish - Do final processing and write the object to the output stream.
  /// \p Writer is used for custom object writer (as the MCJIT does),
  /// if not specified it is automatically created from backend.
  LLVM_ABI void Finish();

  // Layout all section and prepare them for emission.
  LLVM_ABI void layout();

  bool hasLayout() const { return HasLayout; }
  bool hasFinalLayout() const { return HasFinalLayout; }
  bool getRelaxAll() const { return RelaxAll; }
  void setRelaxAll(bool Value) { RelaxAll = Value; }

  bool isBundlingEnabled() const { return BundleAlignSize != 0; }

  unsigned getBundleAlignSize() const { return BundleAlignSize; }

  void setBundleAlignSize(unsigned Size) {
    assert((Size == 0 || !(Size & (Size - 1))) &&
           "Expect a power-of-two bundle align size");
    BundleAlignSize = Size;
  }

  const_iterator begin() const { return Sections.begin(); }
  const_iterator end() const { return Sections.end(); }

  SmallVectorImpl<const MCSymbol *> &getSymbols() { return Symbols; }
  iterator_range<pointee_iterator<
      typename SmallVector<const MCSymbol *, 0>::const_iterator>>
  symbols() const {
    return make_pointee_range(Symbols);
  }

  LLVM_ABI bool registerSection(MCSection &Section);
  LLVM_ABI bool registerSymbol(const MCSymbol &Symbol);

  /// Write the necessary bundle padding to \p OS.
  /// Expects a fragment \p F containing instructions and its size \p FSize.
  LLVM_ABI void writeFragmentPadding(raw_ostream &OS,
                                     const MCEncodedFragment &F,
                                     uint64_t FSize) const;

  LLVM_ABI void reportError(SMLoc L, const Twine &Msg) const;
  // Record pending errors during layout iteration, as they may go away once the
  // layout is finalized.
  LLVM_ABI void recordError(SMLoc L, const Twine &Msg) const;
  LLVM_ABI void flushPendingErrors() const;

  LLVM_ABI void dump() const;
};

} // end namespace llvm

#endif // LLVM_MC_MCASSEMBLER_H
