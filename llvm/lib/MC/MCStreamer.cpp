//===- lib/MC/MCStreamer.cpp - Streaming Machine Code Output --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeView.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCWin64EH.h"
#include "llvm/MC/MCWinEH.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>

using namespace llvm;

MCTargetStreamer::MCTargetStreamer(MCStreamer &S) : Streamer(S) {
  S.setTargetStreamer(this);
}

// Pin the vtables to this file.
MCTargetStreamer::~MCTargetStreamer() = default;

void MCTargetStreamer::emitLabel(MCSymbol *Symbol) {}

void MCTargetStreamer::finish() {}

void MCTargetStreamer::emitAssignment(MCSymbol *Symbol, const MCExpr *Value) {}

MCStreamer::MCStreamer(MCContext &Ctx)
    : Context(Ctx), CurrentWinFrameInfo(nullptr) {
  SectionStack.push_back(std::pair<MCSectionSubPair, MCSectionSubPair>());
}

MCStreamer::~MCStreamer() {
  for (unsigned i = 0; i < getNumWinFrameInfos(); ++i)
    delete WinFrameInfos[i];
}

void MCStreamer::reset() {
  DwarfFrameInfos.clear();
  for (unsigned i = 0; i < getNumWinFrameInfos(); ++i)
    delete WinFrameInfos[i];
  WinFrameInfos.clear();
  CurrentWinFrameInfo = nullptr;
  SymbolOrdering.clear();
  SectionStack.clear();
  SectionStack.push_back(std::pair<MCSectionSubPair, MCSectionSubPair>());
}

raw_ostream &MCStreamer::GetCommentOS() {
  // By default, discard comments.
  return nulls();
}

void MCStreamer::emitRawComment(const Twine &T, bool TabPrefix) {}

void MCStreamer::addExplicitComment(const Twine &T) {}
void MCStreamer::emitExplicitComments() {}

void MCStreamer::generateCompactUnwindEncodings(MCAsmBackend *MAB) {
  for (auto &FI : DwarfFrameInfos)
    FI.CompactUnwindEncoding =
        (MAB ? MAB->generateCompactUnwindEncoding(FI.Instructions) : 0);
}

/// EmitIntValue - Special case of EmitValue that avoids the client having to
/// pass in a MCExpr for constant integers.
void MCStreamer::EmitIntValue(uint64_t Value, unsigned Size) {
  assert(1 <= Size && Size <= 8 && "Invalid size");
  assert((isUIntN(8 * Size, Value) || isIntN(8 * Size, Value)) &&
         "Invalid size");
  char buf[8];
  const bool isLittleEndian = Context.getAsmInfo()->isLittleEndian();
  for (unsigned i = 0; i != Size; ++i) {
    unsigned index = isLittleEndian ? i : (Size - i - 1);
    buf[i] = uint8_t(Value >> (index * 8));
  }
  EmitBytes(StringRef(buf, Size));
}

/// EmitULEB128Value - Special case of EmitULEB128Value that avoids the
/// client having to pass in a MCExpr for constant integers.
void MCStreamer::EmitULEB128IntValue(uint64_t Value, unsigned Padding) {
  SmallString<128> Tmp;
  raw_svector_ostream OSE(Tmp);
  encodeULEB128(Value, OSE, Padding);
  EmitBytes(OSE.str());
}

/// EmitSLEB128Value - Special case of EmitSLEB128Value that avoids the
/// client having to pass in a MCExpr for constant integers.
void MCStreamer::EmitSLEB128IntValue(int64_t Value) {
  SmallString<128> Tmp;
  raw_svector_ostream OSE(Tmp);
  encodeSLEB128(Value, OSE);
  EmitBytes(OSE.str());
}

void MCStreamer::EmitValue(const MCExpr *Value, unsigned Size, SMLoc Loc) {
  EmitValueImpl(Value, Size, Loc);
}

void MCStreamer::EmitSymbolValue(const MCSymbol *Sym, unsigned Size,
                                 bool IsSectionRelative) {
  assert((!IsSectionRelative || Size == 4) &&
         "SectionRelative value requires 4-bytes");

  if (!IsSectionRelative)
    EmitValueImpl(MCSymbolRefExpr::create(Sym, getContext()), Size);
  else
    EmitCOFFSecRel32(Sym, /*Offset=*/0);
}

void MCStreamer::EmitDTPRel64Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

void MCStreamer::EmitDTPRel32Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

void MCStreamer::EmitTPRel64Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

void MCStreamer::EmitTPRel32Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

void MCStreamer::EmitGPRel64Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

void MCStreamer::EmitGPRel32Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

/// Emit NumBytes bytes worth of the value specified by FillValue.
/// This implements directives such as '.space'.
void MCStreamer::emitFill(uint64_t NumBytes, uint8_t FillValue) {
  for (uint64_t i = 0, e = NumBytes; i != e; ++i)
    EmitIntValue(FillValue, 1);
}

void MCStreamer::emitFill(uint64_t NumValues, int64_t Size, int64_t Expr) {
  int64_t NonZeroSize = Size > 4 ? 4 : Size;
  Expr &= ~0ULL >> (64 - NonZeroSize * 8);
  for (uint64_t i = 0, e = NumValues; i != e; ++i) {
    EmitIntValue(Expr, NonZeroSize);
    if (NonZeroSize < Size)
      EmitIntValue(0, Size - NonZeroSize);
  }
}

/// The implementation in this class just redirects to emitFill.
void MCStreamer::EmitZeros(uint64_t NumBytes) {
  emitFill(NumBytes, 0);
}

unsigned MCStreamer::EmitDwarfFileDirective(unsigned FileNo,
                                            StringRef Directory,
                                            StringRef Filename, unsigned CUID) {
  return getContext().getDwarfFile(Directory, Filename, FileNo, CUID);
}

void MCStreamer::EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                       unsigned Column, unsigned Flags,
                                       unsigned Isa,
                                       unsigned Discriminator,
                                       StringRef FileName) {
  getContext().setCurrentDwarfLoc(FileNo, Line, Column, Flags, Isa,
                                  Discriminator);
}

MCSymbol *MCStreamer::getDwarfLineTableSymbol(unsigned CUID) {
  MCDwarfLineTable &Table = getContext().getMCDwarfLineTable(CUID);
  if (!Table.getLabel()) {
    StringRef Prefix = Context.getAsmInfo()->getPrivateGlobalPrefix();
    Table.setLabel(
        Context.getOrCreateSymbol(Prefix + "line_table_start" + Twine(CUID)));
  }
  return Table.getLabel();
}

MCDwarfFrameInfo *MCStreamer::getCurrentDwarfFrameInfo() {
  if (DwarfFrameInfos.empty())
    return nullptr;
  return &DwarfFrameInfos.back();
}

bool MCStreamer::hasUnfinishedDwarfFrameInfo() {
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  return CurFrame && !CurFrame->End;
}

void MCStreamer::EnsureValidDwarfFrame() {
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  if (!CurFrame || CurFrame->End)
    report_fatal_error("No open frame");
}

bool MCStreamer::EmitCVFileDirective(unsigned FileNo, StringRef Filename,
                                     StringRef Checksum,
                                     unsigned ChecksumKind) {
  return getContext().getCVContext().addFile(*this, FileNo, Filename, Checksum,
                                             ChecksumKind);
}

bool MCStreamer::EmitCVFuncIdDirective(unsigned FunctionId) {
  return getContext().getCVContext().recordFunctionId(FunctionId);
}

bool MCStreamer::EmitCVInlineSiteIdDirective(unsigned FunctionId,
                                             unsigned IAFunc, unsigned IAFile,
                                             unsigned IALine, unsigned IACol,
                                             SMLoc Loc) {
  if (getContext().getCVContext().getCVFunctionInfo(IAFunc) == nullptr) {
    getContext().reportError(Loc, "parent function id not introduced by "
                                  ".cv_func_id or .cv_inline_site_id");
    return true;
  }

  return getContext().getCVContext().recordInlinedCallSiteId(
      FunctionId, IAFunc, IAFile, IALine, IACol);
}

void MCStreamer::EmitCVLocDirective(unsigned FunctionId, unsigned FileNo,
                                    unsigned Line, unsigned Column,
                                    bool PrologueEnd, bool IsStmt,
                                    StringRef FileName, SMLoc Loc) {
  CodeViewContext &CVC = getContext().getCVContext();
  MCCVFunctionInfo *FI = CVC.getCVFunctionInfo(FunctionId);
  if (!FI)
    return getContext().reportError(
        Loc, "function id not introduced by .cv_func_id or .cv_inline_site_id");

  // Track the section
  if (FI->Section == nullptr)
    FI->Section = getCurrentSectionOnly();
  else if (FI->Section != getCurrentSectionOnly())
    return getContext().reportError(
        Loc,
        "all .cv_loc directives for a function must be in the same section");

  CVC.setCurrentCVLoc(FunctionId, FileNo, Line, Column, PrologueEnd, IsStmt);
}

void MCStreamer::EmitCVLinetableDirective(unsigned FunctionId,
                                          const MCSymbol *Begin,
                                          const MCSymbol *End) {}

void MCStreamer::EmitCVInlineLinetableDirective(unsigned PrimaryFunctionId,
                                                unsigned SourceFileId,
                                                unsigned SourceLineNum,
                                                const MCSymbol *FnStartSym,
                                                const MCSymbol *FnEndSym) {}

void MCStreamer::EmitCVDefRangeDirective(
    ArrayRef<std::pair<const MCSymbol *, const MCSymbol *>> Ranges,
    StringRef FixedSizePortion) {}

void MCStreamer::EmitEHSymAttributes(const MCSymbol *Symbol,
                                     MCSymbol *EHSymbol) {
}

void MCStreamer::InitSections(bool NoExecStack) {
  SwitchSection(getContext().getObjectFileInfo()->getTextSection());
}

void MCStreamer::AssignFragment(MCSymbol *Symbol, MCFragment *Fragment) {
  assert(Fragment);
  Symbol->setFragment(Fragment);

  // As we emit symbols into a section, track the order so that they can
  // be sorted upon later. Zero is reserved to mean 'unemitted'.
  SymbolOrdering[Symbol] = 1 + SymbolOrdering.size();
}

void MCStreamer::EmitLabel(MCSymbol *Symbol, SMLoc Loc) {
  Symbol->redefineIfPossible();

  if (!Symbol->isUndefined() || Symbol->isVariable())
    return getContext().reportError(Loc, "invalid symbol redefinition");

  assert(!Symbol->isVariable() && "Cannot emit a variable symbol!");
  assert(getCurrentSectionOnly() && "Cannot emit before setting section!");
  assert(!Symbol->getFragment() && "Unexpected fragment on symbol data!");
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  Symbol->setFragment(&getCurrentSectionOnly()->getDummyFragment());

  MCTargetStreamer *TS = getTargetStreamer();
  if (TS)
    TS->emitLabel(Symbol);
}

void MCStreamer::EmitCFISections(bool EH, bool Debug) {
  assert(EH || Debug);
}

void MCStreamer::EmitCFIStartProc(bool IsSimple) {
  if (hasUnfinishedDwarfFrameInfo())
    report_fatal_error("Starting a frame before finishing the previous one!");

  MCDwarfFrameInfo Frame;
  Frame.IsSimple = IsSimple;
  EmitCFIStartProcImpl(Frame);

  const MCAsmInfo* MAI = Context.getAsmInfo();
  if (MAI) {
    for (const MCCFIInstruction& Inst : MAI->getInitialFrameState()) {
      if (Inst.getOperation() == MCCFIInstruction::OpDefCfa ||
          Inst.getOperation() == MCCFIInstruction::OpDefCfaRegister) {
        Frame.CurrentCfaRegister = Inst.getRegister();
      }
    }
  }

  DwarfFrameInfos.push_back(Frame);
}

void MCStreamer::EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame) {
}

void MCStreamer::EmitCFIEndProc() {
  EnsureValidDwarfFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  EmitCFIEndProcImpl(*CurFrame);
}

void MCStreamer::EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame) {
  // Put a dummy non-null value in Frame.End to mark that this frame has been
  // closed.
  Frame.End = (MCSymbol *) 1;
}

MCSymbol *MCStreamer::EmitCFILabel() {
  MCSymbol *Label = getContext().createTempSymbol("cfi", true);
  EmitLabel(Label);
  return Label;
}

MCSymbol *MCStreamer::EmitCFICommon() {
  EnsureValidDwarfFrame();
  return EmitCFILabel();
}

void MCStreamer::EmitCFIDefCfa(int64_t Register, int64_t Offset) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createDefCfa(Label, Register, Offset);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
  CurFrame->CurrentCfaRegister = static_cast<unsigned>(Register);
}

void MCStreamer::EmitCFIDefCfaOffset(int64_t Offset) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createDefCfaOffset(Label, Offset);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIAdjustCfaOffset(int64_t Adjustment) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createAdjustCfaOffset(Label, Adjustment);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIDefCfaRegister(int64_t Register) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createDefCfaRegister(Label, Register);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
  CurFrame->CurrentCfaRegister = static_cast<unsigned>(Register);
}

void MCStreamer::EmitCFIOffset(int64_t Register, int64_t Offset) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createOffset(Label, Register, Offset);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRelOffset(int64_t Register, int64_t Offset) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createRelOffset(Label, Register, Offset);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIPersonality(const MCSymbol *Sym,
                                    unsigned Encoding) {
  EnsureValidDwarfFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Personality = Sym;
  CurFrame->PersonalityEncoding = Encoding;
}

void MCStreamer::EmitCFILsda(const MCSymbol *Sym, unsigned Encoding) {
  EnsureValidDwarfFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Lsda = Sym;
  CurFrame->LsdaEncoding = Encoding;
}

void MCStreamer::EmitCFIRememberState() {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction = MCCFIInstruction::createRememberState(Label);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRestoreState() {
  // FIXME: Error if there is no matching cfi_remember_state.
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction = MCCFIInstruction::createRestoreState(Label);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFISameValue(int64_t Register) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createSameValue(Label, Register);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRestore(int64_t Register) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createRestore(Label, Register);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIEscape(StringRef Values) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction = MCCFIInstruction::createEscape(Label, Values);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIGnuArgsSize(int64_t Size) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction = 
    MCCFIInstruction::createGnuArgsSize(Label, Size);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFISignalFrame() {
  EnsureValidDwarfFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->IsSignalFrame = true;
}

void MCStreamer::EmitCFIUndefined(int64_t Register) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createUndefined(Label, Register);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRegister(int64_t Register1, int64_t Register2) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createRegister(Label, Register1, Register2);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIWindowSave() {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createWindowSave(Label);
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIReturnColumn(int64_t Register) {
  EnsureValidDwarfFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentDwarfFrameInfo();
  CurFrame->RAReg = Register;
}

void MCStreamer::EnsureValidWinFrameInfo() {
  const MCAsmInfo *MAI = Context.getAsmInfo();
  if (!MAI->usesWindowsCFI())
    report_fatal_error(".seh_* directives are not supported on this target");
  if (!CurrentWinFrameInfo || CurrentWinFrameInfo->End)
    report_fatal_error("No open Win64 EH frame function!");
}

void MCStreamer::EmitWinCFIStartProc(const MCSymbol *Symbol) {
  const MCAsmInfo *MAI = Context.getAsmInfo();
  if (!MAI->usesWindowsCFI())
    report_fatal_error(".seh_* directives are not supported on this target");
  if (CurrentWinFrameInfo && !CurrentWinFrameInfo->End)
    report_fatal_error("Starting a function before ending the previous one!");

  MCSymbol *StartProc = EmitCFILabel();

  WinFrameInfos.push_back(new WinEH::FrameInfo(Symbol, StartProc));
  CurrentWinFrameInfo = WinFrameInfos.back();
  CurrentWinFrameInfo->TextSection = getCurrentSectionOnly();
}

void MCStreamer::EmitWinCFIEndProc() {
  EnsureValidWinFrameInfo();
  if (CurrentWinFrameInfo->ChainedParent)
    report_fatal_error("Not all chained regions terminated!");

  MCSymbol *Label = EmitCFILabel();
  CurrentWinFrameInfo->End = Label;
}

void MCStreamer::EmitWinCFIStartChained() {
  EnsureValidWinFrameInfo();

  MCSymbol *StartProc = EmitCFILabel();

  WinFrameInfos.push_back(new WinEH::FrameInfo(CurrentWinFrameInfo->Function,
                                               StartProc, CurrentWinFrameInfo));
  CurrentWinFrameInfo = WinFrameInfos.back();
  CurrentWinFrameInfo->TextSection = getCurrentSectionOnly();
}

void MCStreamer::EmitWinCFIEndChained() {
  EnsureValidWinFrameInfo();
  if (!CurrentWinFrameInfo->ChainedParent)
    report_fatal_error("End of a chained region outside a chained region!");

  MCSymbol *Label = EmitCFILabel();

  CurrentWinFrameInfo->End = Label;
  CurrentWinFrameInfo =
      const_cast<WinEH::FrameInfo *>(CurrentWinFrameInfo->ChainedParent);
}

void MCStreamer::EmitWinEHHandler(const MCSymbol *Sym, bool Unwind,
                                  bool Except) {
  EnsureValidWinFrameInfo();
  if (CurrentWinFrameInfo->ChainedParent)
    report_fatal_error("Chained unwind areas can't have handlers!");
  CurrentWinFrameInfo->ExceptionHandler = Sym;
  if (!Except && !Unwind)
    report_fatal_error("Don't know what kind of handler this is!");
  if (Unwind)
    CurrentWinFrameInfo->HandlesUnwind = true;
  if (Except)
    CurrentWinFrameInfo->HandlesExceptions = true;
}

void MCStreamer::EmitWinEHHandlerData() {
  EnsureValidWinFrameInfo();
  if (CurrentWinFrameInfo->ChainedParent)
    report_fatal_error("Chained unwind areas can't have handlers!");
}

static MCSection *getWinCFISection(MCContext &Context, unsigned *NextWinCFIID,
                                   MCSection *MainCFISec,
                                   const MCSection *TextSec) {
  // If this is the main .text section, use the main unwind info section.
  if (TextSec == Context.getObjectFileInfo()->getTextSection())
    return MainCFISec;

  const auto *TextSecCOFF = cast<MCSectionCOFF>(TextSec);
  unsigned UniqueID = TextSecCOFF->getOrAssignWinCFISectionID(NextWinCFIID);

  // If this section is COMDAT, this unwind section should be COMDAT associative
  // with its group.
  const MCSymbol *KeySym = nullptr;
  if (TextSecCOFF->getCharacteristics() & COFF::IMAGE_SCN_LNK_COMDAT)
    KeySym = TextSecCOFF->getCOMDATSymbol();

  return Context.getAssociativeCOFFSection(cast<MCSectionCOFF>(MainCFISec),
                                           KeySym, UniqueID);
}

MCSection *MCStreamer::getAssociatedPDataSection(const MCSection *TextSec) {
  return getWinCFISection(getContext(), &NextWinCFIID,
                          getContext().getObjectFileInfo()->getPDataSection(),
                          TextSec);
}

MCSection *MCStreamer::getAssociatedXDataSection(const MCSection *TextSec) {
  return getWinCFISection(getContext(), &NextWinCFIID,
                          getContext().getObjectFileInfo()->getXDataSection(),
                          TextSec);
}

void MCStreamer::EmitSyntaxDirective() {}

void MCStreamer::EmitWinCFIPushReg(unsigned Register) {
  EnsureValidWinFrameInfo();

  MCSymbol *Label = EmitCFILabel();

  WinEH::Instruction Inst = Win64EH::Instruction::PushNonVol(Label, Register);
  CurrentWinFrameInfo->Instructions.push_back(Inst);
}

void MCStreamer::EmitWinCFISetFrame(unsigned Register, unsigned Offset) {
  EnsureValidWinFrameInfo();
  if (CurrentWinFrameInfo->LastFrameInst >= 0)
    report_fatal_error("Frame register and offset already specified!");
  if (Offset & 0x0F)
    report_fatal_error("Misaligned frame pointer offset!");
  if (Offset > 240)
    report_fatal_error("Frame offset must be less than or equal to 240!");

  MCSymbol *Label = EmitCFILabel();

  WinEH::Instruction Inst =
      Win64EH::Instruction::SetFPReg(Label, Register, Offset);
  CurrentWinFrameInfo->LastFrameInst = CurrentWinFrameInfo->Instructions.size();
  CurrentWinFrameInfo->Instructions.push_back(Inst);
}

void MCStreamer::EmitWinCFIAllocStack(unsigned Size) {
  EnsureValidWinFrameInfo();
  if (Size == 0)
    report_fatal_error("Allocation size must be non-zero!");
  if (Size & 7)
    report_fatal_error("Misaligned stack allocation!");

  MCSymbol *Label = EmitCFILabel();

  WinEH::Instruction Inst = Win64EH::Instruction::Alloc(Label, Size);
  CurrentWinFrameInfo->Instructions.push_back(Inst);
}

void MCStreamer::EmitWinCFISaveReg(unsigned Register, unsigned Offset) {
  EnsureValidWinFrameInfo();
  if (Offset & 7)
    report_fatal_error("Misaligned saved register offset!");

  MCSymbol *Label = EmitCFILabel();

  WinEH::Instruction Inst =
      Win64EH::Instruction::SaveNonVol(Label, Register, Offset);
  CurrentWinFrameInfo->Instructions.push_back(Inst);
}

void MCStreamer::EmitWinCFISaveXMM(unsigned Register, unsigned Offset) {
  EnsureValidWinFrameInfo();
  if (Offset & 0x0F)
    report_fatal_error("Misaligned saved vector register offset!");

  MCSymbol *Label = EmitCFILabel();

  WinEH::Instruction Inst =
      Win64EH::Instruction::SaveXMM(Label, Register, Offset);
  CurrentWinFrameInfo->Instructions.push_back(Inst);
}

void MCStreamer::EmitWinCFIPushFrame(bool Code) {
  EnsureValidWinFrameInfo();
  if (!CurrentWinFrameInfo->Instructions.empty())
    report_fatal_error("If present, PushMachFrame must be the first UOP");

  MCSymbol *Label = EmitCFILabel();

  WinEH::Instruction Inst = Win64EH::Instruction::PushMachFrame(Label, Code);
  CurrentWinFrameInfo->Instructions.push_back(Inst);
}

void MCStreamer::EmitWinCFIEndProlog() {
  EnsureValidWinFrameInfo();

  MCSymbol *Label = EmitCFILabel();

  CurrentWinFrameInfo->PrologEnd = Label;
}

void MCStreamer::EmitCOFFSafeSEH(MCSymbol const *Symbol) {
}

void MCStreamer::EmitCOFFSectionIndex(MCSymbol const *Symbol) {
}

void MCStreamer::EmitCOFFSecRel32(MCSymbol const *Symbol, uint64_t Offset) {}

/// EmitRawText - If this file is backed by an assembly streamer, this dumps
/// the specified string in the output .s file.  This capability is
/// indicated by the hasRawTextSupport() predicate.
void MCStreamer::EmitRawTextImpl(StringRef String) {
  errs() << "EmitRawText called on an MCStreamer that doesn't support it, "
  " something must not be fully mc'ized\n";
  abort();
}

void MCStreamer::EmitRawText(const Twine &T) {
  SmallString<128> Str;
  EmitRawTextImpl(T.toStringRef(Str));
}

void MCStreamer::EmitWindowsUnwindTables() {
}

void MCStreamer::Finish() {
  if (!DwarfFrameInfos.empty() && !DwarfFrameInfos.back().End)
    report_fatal_error("Unfinished frame!");

  MCTargetStreamer *TS = getTargetStreamer();
  if (TS)
    TS->finish();

  FinishImpl();
}

void MCStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  visitUsedExpr(*Value);
  Symbol->setVariableValue(Value);

  MCTargetStreamer *TS = getTargetStreamer();
  if (TS)
    TS->emitAssignment(Symbol, Value);
}

void MCTargetStreamer::prettyPrintAsm(MCInstPrinter &InstPrinter, raw_ostream &OS,
                              const MCInst &Inst, const MCSubtargetInfo &STI) {
  InstPrinter.printInst(&Inst, OS, "", STI);
}

void MCStreamer::visitUsedSymbol(const MCSymbol &Sym) {
}

void MCStreamer::visitUsedExpr(const MCExpr &Expr) {
  switch (Expr.getKind()) {
  case MCExpr::Target:
    cast<MCTargetExpr>(Expr).visitUsedExpr(*this);
    break;

  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr &BE = cast<MCBinaryExpr>(Expr);
    visitUsedExpr(*BE.getLHS());
    visitUsedExpr(*BE.getRHS());
    break;
  }

  case MCExpr::SymbolRef:
    visitUsedSymbol(cast<MCSymbolRefExpr>(Expr).getSymbol());
    break;

  case MCExpr::Unary:
    visitUsedExpr(*cast<MCUnaryExpr>(Expr).getSubExpr());
    break;
  }
}

void MCStreamer::EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                                 bool) {
  // Scan for values.
  for (unsigned i = Inst.getNumOperands(); i--;)
    if (Inst.getOperand(i).isExpr())
      visitUsedExpr(*Inst.getOperand(i).getExpr());
}

void MCStreamer::emitAbsoluteSymbolDiff(const MCSymbol *Hi, const MCSymbol *Lo,
                                        unsigned Size) {
  // Get the Hi-Lo expression.
  const MCExpr *Diff =
      MCBinaryExpr::createSub(MCSymbolRefExpr::create(Hi, Context),
                              MCSymbolRefExpr::create(Lo, Context), Context);

  const MCAsmInfo *MAI = Context.getAsmInfo();
  if (!MAI->doesSetDirectiveSuppressReloc()) {
    EmitValue(Diff, Size);
    return;
  }

  // Otherwise, emit with .set (aka assignment).
  MCSymbol *SetLabel = Context.createTempSymbol("set", true);
  EmitAssignment(SetLabel, Diff);
  EmitSymbolValue(SetLabel, Size);
}

void MCStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {}
void MCStreamer::EmitThumbFunc(MCSymbol *Func) {}
void MCStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {}
void MCStreamer::BeginCOFFSymbolDef(const MCSymbol *Symbol) {
  llvm_unreachable("this directive only supported on COFF targets");
}
void MCStreamer::EndCOFFSymbolDef() {
  llvm_unreachable("this directive only supported on COFF targets");
}
void MCStreamer::EmitFileDirective(StringRef Filename) {}
void MCStreamer::EmitCOFFSymbolStorageClass(int StorageClass) {
  llvm_unreachable("this directive only supported on COFF targets");
}
void MCStreamer::EmitCOFFSymbolType(int Type) {
  llvm_unreachable("this directive only supported on COFF targets");
}
void MCStreamer::emitELFSize(MCSymbol *Symbol, const MCExpr *Value) {}
void MCStreamer::emitELFSymverDirective(MCSymbol *Alias,
                                        const MCSymbol *Aliasee) {}
void MCStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {}
void MCStreamer::EmitTBSSSymbol(MCSection *Section, MCSymbol *Symbol,
                                uint64_t Size, unsigned ByteAlignment) {}
void MCStreamer::ChangeSection(MCSection *, const MCExpr *) {}
void MCStreamer::EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) {}
void MCStreamer::EmitBytes(StringRef Data) {}
void MCStreamer::EmitBinaryData(StringRef Data) { EmitBytes(Data); }
void MCStreamer::EmitValueImpl(const MCExpr *Value, unsigned Size, SMLoc Loc) {
  visitUsedExpr(*Value);
}
void MCStreamer::EmitULEB128Value(const MCExpr *Value) {}
void MCStreamer::EmitSLEB128Value(const MCExpr *Value) {}
void MCStreamer::emitFill(const MCExpr &NumBytes, uint64_t Value, SMLoc Loc) {}
void MCStreamer::emitFill(const MCExpr &NumValues, int64_t Size, int64_t Expr,
                          SMLoc Loc) {}
void MCStreamer::EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                                      unsigned ValueSize,
                                      unsigned MaxBytesToEmit) {}
void MCStreamer::EmitCodeAlignment(unsigned ByteAlignment,
                                   unsigned MaxBytesToEmit) {}
void MCStreamer::emitValueToOffset(const MCExpr *Offset, unsigned char Value,
                                   SMLoc Loc) {}
void MCStreamer::EmitBundleAlignMode(unsigned AlignPow2) {}
void MCStreamer::EmitBundleLock(bool AlignToEnd) {}
void MCStreamer::FinishImpl() {}
void MCStreamer::EmitBundleUnlock() {}

void MCStreamer::SwitchSection(MCSection *Section, const MCExpr *Subsection) {
  assert(Section && "Cannot switch to a null section!");
  MCSectionSubPair curSection = SectionStack.back().first;
  SectionStack.back().second = curSection;
  if (MCSectionSubPair(Section, Subsection) != curSection) {
    ChangeSection(Section, Subsection);
    SectionStack.back().first = MCSectionSubPair(Section, Subsection);
    assert(!Section->hasEnded() && "Section already ended");
    MCSymbol *Sym = Section->getBeginSymbol();
    if (Sym && !Sym->isInSection())
      EmitLabel(Sym);
  }
}

MCSymbol *MCStreamer::endSection(MCSection *Section) {
  // TODO: keep track of the last subsection so that this symbol appears in the
  // correct place.
  MCSymbol *Sym = Section->getEndSymbol(Context);
  if (Sym->isInSection())
    return Sym;

  SwitchSection(Section);
  EmitLabel(Sym);
  return Sym;
}
