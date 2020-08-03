//===- MCObjectStreamer.h - MCStreamer Object File Interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCOBJECTSTREAMER_H
#define LLVM_MC_MCOBJECTSTREAMER_H

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
class MCAssembler;
class MCCodeEmitter;
class MCSubtargetInfo;
class MCExpr;
class MCFragment;
class MCDataFragment;
class MCAsmBackend;
class raw_ostream;
class raw_pwrite_stream;

/// Streaming object file generation interface.
///
/// This class provides an implementation of the MCStreamer interface which is
/// suitable for use with the assembler backend. Specific object file formats
/// are expected to subclass this interface to implement directives specific
/// to that file format or custom semantics expected by the object writer
/// implementation.
class MCObjectStreamer : public MCStreamer {
  std::unique_ptr<MCAssembler> Assembler;
  MCSection::iterator CurInsertionPoint;
  bool EmitEHFrame;
  bool EmitDebugFrame;
  SmallVector<MCSymbol *, 2> PendingLabels;
  SmallSetVector<MCSection *, 4> PendingLabelSections;
  unsigned CurSubsectionIdx;
  struct PendingMCFixup {
    const MCSymbol *Sym;
    MCFixup Fixup;
    MCDataFragment *DF;
    PendingMCFixup(const MCSymbol *McSym, MCDataFragment *F, MCFixup McFixup)
        : Sym(McSym), Fixup(McFixup), DF(F) {}
  };
  SmallVector<PendingMCFixup, 2> PendingFixups;

  virtual void emitInstToData(const MCInst &Inst, const MCSubtargetInfo&) = 0;
  void emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) override;
  void emitCFIEndProcImpl(MCDwarfFrameInfo &Frame) override;
  MCSymbol *emitCFILabel() override;
  void emitInstructionImpl(const MCInst &Inst, const MCSubtargetInfo &STI);
  void resolvePendingFixups();

protected:
  MCObjectStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> TAB,
                   std::unique_ptr<MCObjectWriter> OW,
                   std::unique_ptr<MCCodeEmitter> Emitter);
  ~MCObjectStreamer();

public:
  /// state management
  void reset() override;

  /// Object streamers require the integrated assembler.
  bool isIntegratedAssemblerRequired() const override { return true; }

  void emitFrames(MCAsmBackend *MAB);
  void emitCFISections(bool EH, bool Debug) override;

  MCFragment *getCurrentFragment() const;

  void insert(MCFragment *F) {
    flushPendingLabels(F);
    MCSection *CurSection = getCurrentSectionOnly();
    CurSection->getFragmentList().insert(CurInsertionPoint, F);
    F->setParent(CurSection);
  }

  /// Get a data fragment to write into, creating a new one if the current
  /// fragment is not a data fragment.
  /// Optionally a \p STI can be passed in so that a new fragment is created
  /// if the Subtarget differs from the current fragment.
  MCDataFragment *getOrCreateDataFragment(const MCSubtargetInfo* STI = nullptr);

protected:
  bool changeSectionImpl(MCSection *Section, const MCExpr *Subsection);

  /// Assign a label to the current Section and Subsection even though a
  /// fragment is not yet present. Use flushPendingLabels(F) to associate
  /// a fragment with this label.
  void addPendingLabel(MCSymbol* label);

  /// If any labels have been emitted but not assigned fragments in the current
  /// Section and Subsection, ensure that they get assigned, either to fragment
  /// F if possible or to a new data fragment. Optionally, one can provide an
  /// offset \p FOffset as a symbol offset within the fragment.
  void flushPendingLabels(MCFragment *F, uint64_t FOffset = 0);

public:
  void visitUsedSymbol(const MCSymbol &Sym) override;

  /// Create a data fragment for any pending labels across all Sections
  /// and Subsections.
  void flushPendingLabels();

  MCAssembler &getAssembler() { return *Assembler; }
  MCAssembler *getAssemblerPtr() override;
  /// \name MCStreamer Interface
  /// @{

  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;
  virtual void emitLabelAtPos(MCSymbol *Symbol, SMLoc Loc, MCFragment *F,
                              uint64_t Offset);
  void emitAssignment(MCSymbol *Symbol, const MCExpr *Value) override;
  void emitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;
  void emitULEB128Value(const MCExpr *Value) override;
  void emitSLEB128Value(const MCExpr *Value) override;
  void emitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override;
  void changeSection(MCSection *Section, const MCExpr *Subsection) override;
  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;

  /// Emit an instruction to a special fragment, because this instruction
  /// can change its size during relaxation.
  virtual void emitInstToFragment(const MCInst &Inst, const MCSubtargetInfo &);

  void emitBundleAlignMode(unsigned AlignPow2) override;
  void emitBundleLock(bool AlignToEnd) override;
  void emitBundleUnlock() override;
  void emitBytes(StringRef Data) override;
  void emitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                            unsigned ValueSize = 1,
                            unsigned MaxBytesToEmit = 0) override;
  void emitCodeAlignment(unsigned ByteAlignment,
                         unsigned MaxBytesToEmit = 0) override;
  void emitValueToOffset(const MCExpr *Offset, unsigned char Value,
                         SMLoc Loc) override;
  void emitDwarfLocDirective(unsigned FileNo, unsigned Line, unsigned Column,
                             unsigned Flags, unsigned Isa,
                             unsigned Discriminator,
                             StringRef FileName) override;
  void emitDwarfAdvanceLineAddr(int64_t LineDelta, const MCSymbol *LastLabel,
                                const MCSymbol *Label, unsigned PointerSize);
  void emitDwarfAdvanceFrameAddr(const MCSymbol *LastLabel,
                                 const MCSymbol *Label);
  void emitCVLocDirective(unsigned FunctionId, unsigned FileNo, unsigned Line,
                          unsigned Column, bool PrologueEnd, bool IsStmt,
                          StringRef FileName, SMLoc Loc) override;
  void emitCVLinetableDirective(unsigned FunctionId, const MCSymbol *Begin,
                                const MCSymbol *End) override;
  void emitCVInlineLinetableDirective(unsigned PrimaryFunctionId,
                                      unsigned SourceFileId,
                                      unsigned SourceLineNum,
                                      const MCSymbol *FnStartSym,
                                      const MCSymbol *FnEndSym) override;
  void emitCVDefRangeDirective(
      ArrayRef<std::pair<const MCSymbol *, const MCSymbol *>> Ranges,
      StringRef FixedSizePortion) override;
  void emitCVStringTableDirective() override;
  void emitCVFileChecksumsDirective() override;
  void emitCVFileChecksumOffsetDirective(unsigned FileNo) override;
  void emitDTPRel32Value(const MCExpr *Value) override;
  void emitDTPRel64Value(const MCExpr *Value) override;
  void emitTPRel32Value(const MCExpr *Value) override;
  void emitTPRel64Value(const MCExpr *Value) override;
  void emitGPRel32Value(const MCExpr *Value) override;
  void emitGPRel64Value(const MCExpr *Value) override;
  Optional<std::pair<bool, std::string>>
  emitRelocDirective(const MCExpr &Offset, StringRef Name, const MCExpr *Expr,
                     SMLoc Loc, const MCSubtargetInfo &STI) override;
  using MCStreamer::emitFill;
  void emitFill(const MCExpr &NumBytes, uint64_t FillValue,
                SMLoc Loc = SMLoc()) override;
  void emitFill(const MCExpr &NumValues, int64_t Size, int64_t Expr,
                SMLoc Loc = SMLoc()) override;
  void emitNops(int64_t NumBytes, int64_t ControlledNopLength,
                SMLoc Loc) override;
  void emitFileDirective(StringRef Filename) override;

  void emitAddrsig() override;
  void emitAddrsigSym(const MCSymbol *Sym) override;

  void finishImpl() override;

  /// Emit the absolute difference between two symbols if possible.
  ///
  /// Emit the absolute difference between \c Hi and \c Lo, as long as we can
  /// compute it.  Currently, that requires that both symbols are in the same
  /// data fragment and that the target has not specified that diff expressions
  /// require relocations to be emitted. Otherwise, do nothing and return
  /// \c false.
  ///
  /// \pre Offset of \c Hi is greater than the offset \c Lo.
  void emitAbsoluteSymbolDiff(const MCSymbol *Hi, const MCSymbol *Lo,
                              unsigned Size) override;

  void emitAbsoluteSymbolDiffAsULEB128(const MCSymbol *Hi,
                                       const MCSymbol *Lo) override;

  bool mayHaveInstructions(MCSection &Sec) const override;
};

} // end namespace llvm

#endif
