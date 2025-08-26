//===- lib/MC/MCObjectStreamer.cpp - Object File MCStreamer Interface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCCodeView.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSFrame.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
using namespace llvm;

MCObjectStreamer::MCObjectStreamer(MCContext &Context,
                                   std::unique_ptr<MCAsmBackend> TAB,
                                   std::unique_ptr<MCObjectWriter> OW,
                                   std::unique_ptr<MCCodeEmitter> Emitter)
    : MCStreamer(Context),
      Assembler(std::make_unique<MCAssembler>(
          Context, std::move(TAB), std::move(Emitter), std::move(OW))),
      EmitEHFrame(true), EmitDebugFrame(false), EmitSFrame(false) {
  assert(Assembler->getBackendPtr() && Assembler->getEmitterPtr());
  IsObj = true;
  setAllowAutoPadding(Assembler->getBackend().allowAutoPadding());
  if (Context.getTargetOptions() && Context.getTargetOptions()->MCRelaxAll)
    Assembler->setRelaxAll(true);
}

MCObjectStreamer::~MCObjectStreamer() = default;

MCAssembler *MCObjectStreamer::getAssemblerPtr() {
  if (getUseAssemblerInfoForParsing())
    return Assembler.get();
  return nullptr;
}

constexpr size_t FragBlockSize = 16384;
// Ensure the new fragment can at least store a few bytes.
constexpr size_t NewFragHeadroom = 8;

static_assert(NewFragHeadroom >= alignof(MCFragment));
static_assert(FragBlockSize >= sizeof(MCFragment) + NewFragHeadroom);

MCFragment *MCObjectStreamer::allocFragSpace(size_t Headroom) {
  auto Size = std::max(FragBlockSize, sizeof(MCFragment) + Headroom);
  FragSpace = Size - sizeof(MCFragment);
  auto Block = std::unique_ptr<uint8_t[]>(new uint8_t[Size]);
  auto *F = reinterpret_cast<MCFragment *>(Block.get());
  FragStorage.push_back(std::move(Block));
  return F;
}

void MCObjectStreamer::newFragment() {
  MCFragment *F;
  if (LLVM_LIKELY(sizeof(MCFragment) + NewFragHeadroom <= FragSpace)) {
    auto End = reinterpret_cast<size_t>(getCurFragEnd());
    F = reinterpret_cast<MCFragment *>(
        alignToPowerOf2(End, alignof(MCFragment)));
    FragSpace -= size_t(F) - End + sizeof(MCFragment);
  } else {
    F = allocFragSpace(0);
  }
  new (F) MCFragment();
  addFragment(F);
}

void MCObjectStreamer::ensureHeadroom(size_t Headroom) {
  if (Headroom <= FragSpace)
    return;
  auto *F = allocFragSpace(Headroom);
  new (F) MCFragment();
  addFragment(F);
}

void MCObjectStreamer::addSpecialFragment(MCFragment *Frag) {
  assert(Frag->getKind() != MCFragment::FT_Data &&
         "Frag should have a variable-size tail");
  // Frag is not connected to FragSpace. Before modifying CurFrag with
  // addFragment(Frag), allocate an empty fragment to maintain FragSpace
  // connectivity, potentially reusing CurFrag's associated space.
  MCFragment *F;
  if (LLVM_LIKELY(sizeof(MCFragment) + NewFragHeadroom <= FragSpace)) {
    auto End = reinterpret_cast<size_t>(getCurFragEnd());
    F = reinterpret_cast<MCFragment *>(
        alignToPowerOf2(End, alignof(MCFragment)));
    FragSpace -= size_t(F) - End + sizeof(MCFragment);
  } else {
    F = allocFragSpace(0);
  }
  new (F) MCFragment();

  addFragment(Frag);
  addFragment(F);
}

void MCObjectStreamer::appendContents(ArrayRef<char> Contents) {
  ensureHeadroom(Contents.size());
  assert(FragSpace >= Contents.size());
  llvm::copy(Contents, getCurFragEnd());
  CurFrag->FixedSize += Contents.size();
  FragSpace -= Contents.size();
}

void MCObjectStreamer::appendContents(size_t Num, uint8_t Elt) {
  ensureHeadroom(Num);
  MutableArrayRef<uint8_t> Data(getCurFragEnd(), Num);
  llvm::fill(Data, Elt);
  CurFrag->FixedSize += Num;
  FragSpace -= Num;
}

void MCObjectStreamer::addFixup(const MCExpr *Value, MCFixupKind Kind) {
  CurFrag->addFixup(MCFixup::create(getCurFragSize(), Value, Kind));
}

// As a compile-time optimization, avoid allocating and evaluating an MCExpr
// tree for (Hi - Lo) when Hi and Lo are offsets into the same fragment's fixed
// part.
static std::optional<uint64_t> absoluteSymbolDiff(const MCSymbol *Hi,
                                                  const MCSymbol *Lo) {
  assert(Hi && Lo);
  if (Lo == Hi)
    return 0;
  if (Hi->isVariable() || Lo->isVariable())
    return std::nullopt;
  auto *LoF = Lo->getFragment();
  if (!LoF || Hi->getFragment() != LoF || LoF->isLinkerRelaxable())
    return std::nullopt;
  // If either symbol resides in the variable part, bail out.
  auto Fixed = LoF->getFixedSize();
  if (Lo->getOffset() > Fixed || Hi->getOffset() > Fixed)
    return std::nullopt;

  return Hi->getOffset() - Lo->getOffset();
}

void MCObjectStreamer::emitAbsoluteSymbolDiff(const MCSymbol *Hi,
                                              const MCSymbol *Lo,
                                              unsigned Size) {
  if (std::optional<uint64_t> Diff = absoluteSymbolDiff(Hi, Lo))
    emitIntValue(*Diff, Size);
  else
    MCStreamer::emitAbsoluteSymbolDiff(Hi, Lo, Size);
}

void MCObjectStreamer::emitAbsoluteSymbolDiffAsULEB128(const MCSymbol *Hi,
                                                       const MCSymbol *Lo) {
  if (std::optional<uint64_t> Diff = absoluteSymbolDiff(Hi, Lo))
    emitULEB128IntValue(*Diff);
  else
    MCStreamer::emitAbsoluteSymbolDiffAsULEB128(Hi, Lo);
}

void MCObjectStreamer::reset() {
  if (Assembler) {
    Assembler->reset();
    if (getContext().getTargetOptions())
      Assembler->setRelaxAll(getContext().getTargetOptions()->MCRelaxAll);
  }
  EmitEHFrame = true;
  EmitDebugFrame = false;
  FragStorage.clear();
  FragSpace = 0;
  SpecialFragAllocator.Reset();
  MCStreamer::reset();
}

void MCObjectStreamer::emitFrames(MCAsmBackend *MAB) {
  if (!getNumFrameInfos())
    return;

  if (EmitEHFrame)
    MCDwarfFrameEmitter::Emit(*this, MAB, true);

  if (EmitDebugFrame)
    MCDwarfFrameEmitter::Emit(*this, MAB, false);

  if (EmitSFrame || (getContext().getTargetOptions() &&
                     getContext().getTargetOptions()->EmitSFrameUnwind))
    MCSFrameEmitter::emit(*this);
}

void MCObjectStreamer::visitUsedSymbol(const MCSymbol &Sym) {
  Assembler->registerSymbol(Sym);
}

void MCObjectStreamer::emitCFISections(bool EH, bool Debug, bool SFrame) {
  MCStreamer::emitCFISections(EH, Debug, SFrame);
  EmitEHFrame = EH;
  EmitDebugFrame = Debug;
  EmitSFrame = SFrame;
}

void MCObjectStreamer::emitValueImpl(const MCExpr *Value, unsigned Size,
                                     SMLoc Loc) {
  MCStreamer::emitValueImpl(Value, Size, Loc);

  MCDwarfLineEntry::make(this, getCurrentSectionOnly());

  // Avoid fixups when possible.
  int64_t AbsValue;
  if (Value->evaluateAsAbsolute(AbsValue, getAssemblerPtr())) {
    if (!isUIntN(8 * Size, AbsValue) && !isIntN(8 * Size, AbsValue)) {
      getContext().reportError(
          Loc, "value evaluated as " + Twine(AbsValue) + " is out of range.");
      return;
    }
    emitIntValue(AbsValue, Size);
    return;
  }
  ensureHeadroom(Size);
  addFixup(Value, MCFixup::getDataKindForSize(Size));
  appendContents(Size, 0);
}

MCSymbol *MCObjectStreamer::emitCFILabel() {
  MCSymbol *Label = getContext().createTempSymbol("cfi");
  emitLabel(Label);
  return Label;
}

void MCObjectStreamer::emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) {
  // We need to create a local symbol to avoid relocations.
  Frame.Begin = getContext().createTempSymbol();
  emitLabel(Frame.Begin);
}

void MCObjectStreamer::emitCFIEndProcImpl(MCDwarfFrameInfo &Frame) {
  Frame.End = getContext().createTempSymbol();
  emitLabel(Frame.End);
}

void MCObjectStreamer::emitLabel(MCSymbol *Symbol, SMLoc Loc) {
  MCStreamer::emitLabel(Symbol, Loc);
  // If Symbol is a non-redefiniable variable, emitLabel has reported an error.
  // Bail out.
  if (Symbol->isVariable())
    return;

  getAssembler().registerSymbol(*Symbol);

  // Set the fragment and offset. This function might be called by
  // changeSection, when the section stack top hasn't been changed to the new
  // section.
  MCFragment *F = CurFrag;
  Symbol->setFragment(F);
  Symbol->setOffset(F->getFixedSize());

  emitPendingAssignments(Symbol);
}

void MCObjectStreamer::emitPendingAssignments(MCSymbol *Symbol) {
  auto Assignments = pendingAssignments.find(Symbol);
  if (Assignments != pendingAssignments.end()) {
    for (const PendingAssignment &A : Assignments->second)
      emitAssignment(A.Symbol, A.Value);

    pendingAssignments.erase(Assignments);
  }
}

// Emit a label at a previously emitted fragment/offset position. This must be
// within the currently-active section.
void MCObjectStreamer::emitLabelAtPos(MCSymbol *Symbol, SMLoc Loc,
                                      MCFragment &F, uint64_t Offset) {
  assert(F.getParent() == getCurrentSectionOnly());
  MCStreamer::emitLabel(Symbol, Loc);
  getAssembler().registerSymbol(*Symbol);
  Symbol->setFragment(&F);
  Symbol->setOffset(Offset);
}

void MCObjectStreamer::emitULEB128Value(const MCExpr *Value) {
  int64_t IntValue;
  if (Value->evaluateAsAbsolute(IntValue, getAssembler())) {
    emitULEB128IntValue(IntValue);
    return;
  }
  auto *F = getCurrentFragment();
  F->makeLEB(false, Value);
  newFragment();
}

void MCObjectStreamer::emitSLEB128Value(const MCExpr *Value) {
  int64_t IntValue;
  if (Value->evaluateAsAbsolute(IntValue, getAssembler())) {
    emitSLEB128IntValue(IntValue);
    return;
  }
  auto *F = getCurrentFragment();
  F->makeLEB(true, Value);
  newFragment();
}

void MCObjectStreamer::emitWeakReference(MCSymbol *Alias,
                                         const MCSymbol *Target) {
  reportFatalUsageError("this file format doesn't support weak aliases");
}

void MCObjectStreamer::changeSection(MCSection *Section, uint32_t Subsection) {
  assert(Section && "Cannot switch to a null section!");
  getContext().clearDwarfLocSeen();

  // Register the section and create an initial fragment for subsection 0
  // if `Subsection` is non-zero.
  bool NewSec = getAssembler().registerSection(*Section);
  MCFragment *F0 = nullptr;
  if (NewSec && Subsection) {
    changeSection(Section, 0);
    F0 = CurFrag;
  }

  // To maintain connectivity between CurFrag and FragSpace when CurFrag is
  // modified, allocate an empty fragment and append it to the fragment list.
  // (Subsections[I].second.Tail is not connected to FragSpace.)
  MCFragment *F;
  if (LLVM_LIKELY(sizeof(MCFragment) + NewFragHeadroom <= FragSpace)) {
    auto End = reinterpret_cast<size_t>(getCurFragEnd());
    F = reinterpret_cast<MCFragment *>(
        alignToPowerOf2(End, alignof(MCFragment)));
    FragSpace -= size_t(F) - End + sizeof(MCFragment);
  } else {
    F = allocFragSpace(0);
  }
  new (F) MCFragment();
  F->setParent(Section);

  auto &Subsections = Section->Subsections;
  size_t I = 0, E = Subsections.size();
  while (I != E && Subsections[I].first < Subsection)
    ++I;
  // If the subsection number is not in the sorted Subsections list, create a
  // new fragment list.
  if (I == E || Subsections[I].first != Subsection) {
    Subsections.insert(Subsections.begin() + I,
                       {Subsection, MCSection::FragList{F, F}});
    Section->CurFragList = &Subsections[I].second;
    CurFrag = F;
  } else {
    Section->CurFragList = &Subsections[I].second;
    CurFrag = Subsections[I].second.Tail;
    // Ensure CurFrag is associated with FragSpace.
    addFragment(F);
  }

  // Define the section symbol at subsection 0's initial fragment if required.
  if (!NewSec)
    return;
  if (auto *Sym = Section->getBeginSymbol()) {
    Sym->setFragment(Subsection ? F0 : CurFrag);
    getAssembler().registerSymbol(*Sym);
  }
}

void MCObjectStreamer::emitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  getAssembler().registerSymbol(*Symbol);
  MCStreamer::emitAssignment(Symbol, Value);
  emitPendingAssignments(Symbol);
}

void MCObjectStreamer::emitConditionalAssignment(MCSymbol *Symbol,
                                                 const MCExpr *Value) {
  const MCSymbol *Target = &cast<MCSymbolRefExpr>(*Value).getSymbol();

  // If the symbol already exists, emit the assignment. Otherwise, emit it
  // later only if the symbol is also emitted.
  if (Target->isRegistered())
    emitAssignment(Symbol, Value);
  else
    pendingAssignments[Target].push_back({Symbol, Value});
}

bool MCObjectStreamer::mayHaveInstructions(MCSection &Sec) const {
  return Sec.hasInstructions();
}

void MCObjectStreamer::emitInstruction(const MCInst &Inst,
                                       const MCSubtargetInfo &STI) {
  MCStreamer::emitInstruction(Inst, STI);

  MCSection *Sec = getCurrentSectionOnly();
  Sec->setHasInstructions(true);

  // Now that a machine instruction has been assembled into this section, make
  // a line entry for any .loc directive that has been seen.
  MCDwarfLineEntry::make(this, getCurrentSectionOnly());

  // If this instruction doesn't need relaxation, just emit it as data.
  MCAssembler &Assembler = getAssembler();
  MCAsmBackend &Backend = Assembler.getBackend();
  if (!(Backend.mayNeedRelaxation(Inst.getOpcode(), Inst.getOperands(), STI) ||
        Backend.allowEnhancedRelaxation())) {
    emitInstToData(Inst, STI);
    return;
  }

  // Otherwise, relax and emit it as data if RelaxAll is specified.
  if (Assembler.getRelaxAll()) {
    MCInst Relaxed = Inst;
    while (Backend.mayNeedRelaxation(Relaxed.getOpcode(), Relaxed.getOperands(),
                                     STI))
      Backend.relaxInstruction(Relaxed, STI);
    emitInstToData(Relaxed, STI);
    return;
  }

  emitInstToFragment(Inst, STI);
}

void MCObjectStreamer::emitInstToData(const MCInst &Inst,
                                      const MCSubtargetInfo &STI) {
  MCFragment *F = getCurrentFragment();

  // Append the instruction to the data fragment.
  size_t CodeOffset = getCurFragSize();
  SmallString<16> Content;
  SmallVector<MCFixup, 1> Fixups;
  getAssembler().getEmitter().encodeInstruction(Inst, Content, Fixups, STI);
  appendContents(Content);
  if (CurFrag != F) {
    F = CurFrag;
    CodeOffset = 0;
  }
  F->setHasInstructions(STI);

  if (Fixups.empty())
    return;
  bool MarkedLinkerRelaxable = false;
  for (auto &Fixup : Fixups) {
    Fixup.setOffset(Fixup.getOffset() + CodeOffset);
    if (!Fixup.isLinkerRelaxable() || MarkedLinkerRelaxable)
      continue;
    MarkedLinkerRelaxable = true;
    // Set the fragment's order within the subsection for use by
    // MCAssembler::relaxAlign.
    auto *Sec = F->getParent();
    if (!Sec->isLinkerRelaxable())
      Sec->setFirstLinkerRelaxable(F->getLayoutOrder());
    // Do not add data after a linker-relaxable instruction. The difference
    // between a new label and a label at or before the linker-relaxable
    // instruction cannot be resolved at assemble-time.
    F->setLinkerRelaxable();
    newFragment();
  }
  F->appendFixups(Fixups);
}

void MCObjectStreamer::emitInstToFragment(const MCInst &Inst,
                                          const MCSubtargetInfo &STI) {
  auto *F = getCurrentFragment();
  SmallVector<char, 16> Data;
  SmallVector<MCFixup, 1> Fixups;
  getAssembler().getEmitter().encodeInstruction(Inst, Data, Fixups, STI);

  F->Kind = MCFragment::FT_Relaxable;
  F->setHasInstructions(STI);

  F->setVarContents(Data);
  F->setInst(Inst);

  bool MarkedLinkerRelaxable = false;
  for (auto &Fixup : Fixups) {
    if (!Fixup.isLinkerRelaxable() || MarkedLinkerRelaxable)
      continue;
    MarkedLinkerRelaxable = true;
    auto *Sec = F->getParent();
    if (!Sec->isLinkerRelaxable())
      Sec->setFirstLinkerRelaxable(F->getLayoutOrder());
    F->setLinkerRelaxable();
  }
  F->setVarFixups(Fixups);

  newFragment();
}

void MCObjectStreamer::emitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                             unsigned Column, unsigned Flags,
                                             unsigned Isa,
                                             unsigned Discriminator,
                                             StringRef FileName,
                                             StringRef Comment) {
  // In case we see two .loc directives in a row, make sure the
  // first one gets a line entry.
  MCDwarfLineEntry::make(this, getCurrentSectionOnly());

  this->MCStreamer::emitDwarfLocDirective(FileNo, Line, Column, Flags, Isa,
                                          Discriminator, FileName, Comment);
}

static const MCExpr *buildSymbolDiff(MCObjectStreamer &OS, const MCSymbol *A,
                                     const MCSymbol *B, SMLoc Loc) {
  MCContext &Context = OS.getContext();
  const MCExpr *ARef = MCSymbolRefExpr::create(A, Context);
  const MCExpr *BRef = MCSymbolRefExpr::create(B, Context);
  const MCExpr *AddrDelta =
      MCBinaryExpr::create(MCBinaryExpr::Sub, ARef, BRef, Context, Loc);
  return AddrDelta;
}

static void emitDwarfSetLineAddr(MCObjectStreamer &OS,
                                 MCDwarfLineTableParams Params,
                                 int64_t LineDelta, const MCSymbol *Label,
                                 int PointerSize) {
  // emit the sequence to set the address
  OS.emitIntValue(dwarf::DW_LNS_extended_op, 1);
  OS.emitULEB128IntValue(PointerSize + 1);
  OS.emitIntValue(dwarf::DW_LNE_set_address, 1);
  OS.emitSymbolValue(Label, PointerSize);

  // emit the sequence for the LineDelta (from 1) and a zero address delta.
  MCDwarfLineAddr::Emit(&OS, Params, LineDelta, 0);
}

void MCObjectStreamer::emitDwarfAdvanceLineAddr(int64_t LineDelta,
                                                const MCSymbol *LastLabel,
                                                const MCSymbol *Label,
                                                unsigned PointerSize) {
  if (!LastLabel) {
    emitDwarfSetLineAddr(*this, Assembler->getDWARFLinetableParams(), LineDelta,
                         Label, PointerSize);
    return;
  }

  // If the two labels are within the same fragment, then the address-offset is
  // already a fixed constant and is not relaxable. Emit the advance-line-addr
  // data immediately to save time and memory.
  if (auto OptAddrDelta = absoluteSymbolDiff(Label, LastLabel)) {
    SmallString<16> Tmp;
    MCDwarfLineAddr::encode(getContext(), Assembler->getDWARFLinetableParams(),
                            LineDelta, *OptAddrDelta, Tmp);
    emitBytes(Tmp);
    return;
  }

  auto *F = getCurrentFragment();
  F->Kind = MCFragment::FT_Dwarf;
  F->setDwarfAddrDelta(buildSymbolDiff(*this, Label, LastLabel, SMLoc()));
  F->setDwarfLineDelta(LineDelta);
  newFragment();
}

void MCObjectStreamer::emitDwarfLineEndEntry(MCSection *Section,
                                             MCSymbol *LastLabel,
                                             MCSymbol *EndLabel) {
  // Emit a DW_LNE_end_sequence into the line table. When EndLabel is null, it
  // means we should emit the entry for the end of the section and therefore we
  // use the section end label for the reference label. After having the
  // appropriate reference label, we emit the address delta and use INT64_MAX as
  // the line delta which is the signal that this is actually a
  // DW_LNE_end_sequence.
  if (!EndLabel)
    EndLabel = endSection(Section);

  // Switch back the dwarf line section, in case endSection had to switch the
  // section.
  MCContext &Ctx = getContext();
  switchSection(Ctx.getObjectFileInfo()->getDwarfLineSection());

  const MCAsmInfo *AsmInfo = Ctx.getAsmInfo();
  emitDwarfAdvanceLineAddr(INT64_MAX, LastLabel, EndLabel,
                           AsmInfo->getCodePointerSize());
}

void MCObjectStreamer::emitDwarfAdvanceFrameAddr(const MCSymbol *LastLabel,
                                                 const MCSymbol *Label,
                                                 SMLoc Loc) {
  auto *F = getCurrentFragment();
  F->Kind = MCFragment::FT_DwarfFrame;
  F->setDwarfAddrDelta(buildSymbolDiff(*this, Label, LastLabel, Loc));
  newFragment();
}

void MCObjectStreamer::emitCVLocDirective(unsigned FunctionId, unsigned FileNo,
                                          unsigned Line, unsigned Column,
                                          bool PrologueEnd, bool IsStmt,
                                          StringRef FileName, SMLoc Loc) {
  // Validate the directive.
  if (!checkCVLocSection(FunctionId, FileNo, Loc))
    return;

  // Emit a label at the current position and record it in the CodeViewContext.
  MCSymbol *LineSym = getContext().createTempSymbol();
  emitLabel(LineSym);
  getContext().getCVContext().recordCVLoc(getContext(), LineSym, FunctionId,
                                          FileNo, Line, Column, PrologueEnd,
                                          IsStmt);
}

void MCObjectStreamer::emitCVLinetableDirective(unsigned FunctionId,
                                                const MCSymbol *Begin,
                                                const MCSymbol *End) {
  getContext().getCVContext().emitLineTableForFunction(*this, FunctionId, Begin,
                                                       End);
  this->MCStreamer::emitCVLinetableDirective(FunctionId, Begin, End);
}

void MCObjectStreamer::emitCVInlineLinetableDirective(
    unsigned PrimaryFunctionId, unsigned SourceFileId, unsigned SourceLineNum,
    const MCSymbol *FnStartSym, const MCSymbol *FnEndSym) {
  getContext().getCVContext().emitInlineLineTableForFunction(
      *this, PrimaryFunctionId, SourceFileId, SourceLineNum, FnStartSym,
      FnEndSym);
  this->MCStreamer::emitCVInlineLinetableDirective(
      PrimaryFunctionId, SourceFileId, SourceLineNum, FnStartSym, FnEndSym);
}

void MCObjectStreamer::emitCVDefRangeDirective(
    ArrayRef<std::pair<const MCSymbol *, const MCSymbol *>> Ranges,
    StringRef FixedSizePortion) {
  getContext().getCVContext().emitDefRange(*this, Ranges, FixedSizePortion);
  // Attach labels that were pending before we created the defrange fragment to
  // the beginning of the new fragment.
  this->MCStreamer::emitCVDefRangeDirective(Ranges, FixedSizePortion);
}

void MCObjectStreamer::emitCVStringTableDirective() {
  getContext().getCVContext().emitStringTable(*this);
}
void MCObjectStreamer::emitCVFileChecksumsDirective() {
  getContext().getCVContext().emitFileChecksums(*this);
}

void MCObjectStreamer::emitCVFileChecksumOffsetDirective(unsigned FileNo) {
  getContext().getCVContext().emitFileChecksumOffset(*this, FileNo);
}

void MCObjectStreamer::emitBytes(StringRef Data) {
  MCDwarfLineEntry::make(this, getCurrentSectionOnly());
  appendContents(ArrayRef(Data.data(), Data.size()));
}

void MCObjectStreamer::emitValueToAlignment(Align Alignment, int64_t Fill,
                                            uint8_t FillLen,
                                            unsigned MaxBytesToEmit) {
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = Alignment.value();
  MCFragment *F = getCurrentFragment();
  F->makeAlign(Alignment, Fill, FillLen, MaxBytesToEmit);
  newFragment();

  // Update the maximum alignment on the current section if necessary.
  F->getParent()->ensureMinAlignment(Alignment);
}

void MCObjectStreamer::emitCodeAlignment(Align Alignment,
                                         const MCSubtargetInfo *STI,
                                         unsigned MaxBytesToEmit) {
  auto *F = getCurrentFragment();
  emitValueToAlignment(Alignment, 0, 1, MaxBytesToEmit);
  F->u.align.EmitNops = true;
  F->STI = STI;
}

void MCObjectStreamer::emitValueToOffset(const MCExpr *Offset,
                                         unsigned char Value,
                                         SMLoc Loc) {
  newSpecialFragment<MCOrgFragment>(*Offset, Value, Loc);
}

void MCObjectStreamer::emitRelocDirective(const MCExpr &Offset, StringRef Name,
                                          const MCExpr *Expr, SMLoc Loc) {
  std::optional<MCFixupKind> MaybeKind =
      Assembler->getBackend().getFixupKind(Name);
  if (!MaybeKind) {
    getContext().reportError(Loc, "unknown relocation name");
    return;
  }

  MCFixupKind Kind = *MaybeKind;
  if (Expr)
    visitUsedExpr(*Expr);
  else
    Expr =
        MCSymbolRefExpr::create(getContext().createTempSymbol(), getContext());

  auto *O = &Offset;
  int64_t Val;
  if (Offset.evaluateAsAbsolute(Val, nullptr)) {
    auto *SecSym = getCurrentSectionOnly()->getBeginSymbol();
    O = MCBinaryExpr::createAdd(MCSymbolRefExpr::create(SecSym, getContext()),
                                O, getContext(), Loc);
  }
  getAssembler().addRelocDirective({*O, Expr, Kind});
}

void MCObjectStreamer::emitFill(const MCExpr &NumBytes, uint64_t FillValue,
                                SMLoc Loc) {
  assert(getCurrentSectionOnly() && "need a section");
  newSpecialFragment<MCFillFragment>(FillValue, 1, NumBytes, Loc);
}

void MCObjectStreamer::emitFill(const MCExpr &NumValues, int64_t Size,
                                int64_t Expr, SMLoc Loc) {
  int64_t IntNumValues;
  // Do additional checking now if we can resolve the value.
  if (NumValues.evaluateAsAbsolute(IntNumValues, getAssembler())) {
    if (IntNumValues < 0) {
      getContext().getSourceManager()->PrintMessage(
          Loc, SourceMgr::DK_Warning,
          "'.fill' directive with negative repeat count has no effect");
      return;
    }
    // Emit now if we can for better errors.
    int64_t NonZeroSize = Size > 4 ? 4 : Size;
    Expr &= ~0ULL >> (64 - NonZeroSize * 8);
    for (uint64_t i = 0, e = IntNumValues; i != e; ++i) {
      emitIntValue(Expr, NonZeroSize);
      if (NonZeroSize < Size)
        emitIntValue(0, Size - NonZeroSize);
    }
    return;
  }

  // Otherwise emit as fragment.
  assert(getCurrentSectionOnly() && "need a section");
  newSpecialFragment<MCFillFragment>(Expr, Size, NumValues, Loc);
}

void MCObjectStreamer::emitNops(int64_t NumBytes, int64_t ControlledNopLength,
                                SMLoc Loc, const MCSubtargetInfo &STI) {
  assert(getCurrentSectionOnly() && "need a section");
  newSpecialFragment<MCNopsFragment>(NumBytes, ControlledNopLength, Loc, STI);
}

void MCObjectStreamer::emitFileDirective(StringRef Filename) {
  MCAssembler &Asm = getAssembler();
  Asm.getWriter().addFileName(Filename);
}

void MCObjectStreamer::emitFileDirective(StringRef Filename,
                                         StringRef CompilerVersion,
                                         StringRef TimeStamp,
                                         StringRef Description) {
  MCObjectWriter &W = getAssembler().getWriter();
  W.addFileName(Filename);
  if (CompilerVersion.size())
    W.setCompilerVersion(CompilerVersion);
  // TODO: add TimeStamp and Description to .file symbol table entry
  // with the integrated assembler.
}

void MCObjectStreamer::emitAddrsig() {
  getAssembler().getWriter().emitAddrsigSection();
}

void MCObjectStreamer::emitAddrsigSym(const MCSymbol *Sym) {
  getAssembler().getWriter().addAddrsigSymbol(Sym);
}

void MCObjectStreamer::finishImpl() {
  getContext().RemapDebugPaths();

  // If we are generating dwarf for assembly source files dump out the sections.
  if (getContext().getGenDwarfForAssembly())
    MCGenDwarfInfo::Emit(this);

  // Dump out the dwarf file & directory tables and line tables.
  MCDwarfLineTable::emit(this, getAssembler().getDWARFLinetableParams());

  // Emit pseudo probes for the current module.
  MCPseudoProbeTable::emit(this);

  getAssembler().Finish();
}
