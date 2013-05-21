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
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace llvm;

MCStreamer::MCStreamer(StreamerKind Kind, MCContext &Ctx)
    : Kind(Kind), Context(Ctx), EmitEHFrame(true), EmitDebugFrame(false),
      CurrentW64UnwindInfo(0), LastSymbol(0), AutoInitSections(false) {
  SectionStack.push_back(std::pair<MCSectionSubPair, MCSectionSubPair>());
}

MCStreamer::~MCStreamer() {
  for (unsigned i = 0; i < getNumW64UnwindInfos(); ++i)
    delete W64UnwindInfos[i];
}

void MCStreamer::reset() {
  for (unsigned i = 0; i < getNumW64UnwindInfos(); ++i)
    delete W64UnwindInfos[i];
  W64UnwindInfos.clear();
  EmitEHFrame = true;
  EmitDebugFrame = false;
  CurrentW64UnwindInfo = 0;
  LastSymbol = 0;
  SectionStack.clear();
  SectionStack.push_back(std::pair<MCSectionSubPair, MCSectionSubPair>());
}

const MCExpr *MCStreamer::BuildSymbolDiff(MCContext &Context,
                                          const MCSymbol *A,
                                          const MCSymbol *B) {
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  const MCExpr *ARef =
    MCSymbolRefExpr::Create(A, Variant, Context);
  const MCExpr *BRef =
    MCSymbolRefExpr::Create(B, Variant, Context);
  const MCExpr *AddrDelta =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, ARef, BRef, Context);
  return AddrDelta;
}

const MCExpr *MCStreamer::ForceExpAbs(const MCExpr* Expr) {
  if (Context.getAsmInfo().hasAggressiveSymbolFolding() ||
      isa<MCSymbolRefExpr>(Expr))
    return Expr;

  MCSymbol *ABS = Context.CreateTempSymbol();
  EmitAssignment(ABS, Expr);
  return MCSymbolRefExpr::Create(ABS, Context);
}

raw_ostream &MCStreamer::GetCommentOS() {
  // By default, discard comments.
  return nulls();
}

void MCStreamer::EmitDwarfSetLineAddr(int64_t LineDelta,
                                      const MCSymbol *Label, int PointerSize) {
  // emit the sequence to set the address
  EmitIntValue(dwarf::DW_LNS_extended_op, 1);
  EmitULEB128IntValue(PointerSize + 1);
  EmitIntValue(dwarf::DW_LNE_set_address, 1);
  EmitSymbolValue(Label, PointerSize);

  // emit the sequence for the LineDelta (from 1) and a zero address delta.
  MCDwarfLineAddr::Emit(this, LineDelta, 0);
}

/// EmitIntValue - Special case of EmitValue that avoids the client having to
/// pass in a MCExpr for constant integers.
void MCStreamer::EmitIntValue(uint64_t Value, unsigned Size,
                              unsigned AddrSpace) {
  assert(Size <= 8 && "Invalid size");
  assert((isUIntN(8 * Size, Value) || isIntN(8 * Size, Value)) &&
         "Invalid size");
  char buf[8];
  const bool isLittleEndian = Context.getAsmInfo().isLittleEndian();
  for (unsigned i = 0; i != Size; ++i) {
    unsigned index = isLittleEndian ? i : (Size - i - 1);
    buf[i] = uint8_t(Value >> (index * 8));
  }
  EmitBytes(StringRef(buf, Size), AddrSpace);
}

/// EmitULEB128Value - Special case of EmitULEB128Value that avoids the
/// client having to pass in a MCExpr for constant integers.
void MCStreamer::EmitULEB128IntValue(uint64_t Value, unsigned Padding,
                                     unsigned AddrSpace) {
  SmallString<128> Tmp;
  raw_svector_ostream OSE(Tmp);
  encodeULEB128(Value, OSE, Padding);
  EmitBytes(OSE.str(), AddrSpace);
}

/// EmitSLEB128Value - Special case of EmitSLEB128Value that avoids the
/// client having to pass in a MCExpr for constant integers.
void MCStreamer::EmitSLEB128IntValue(int64_t Value, unsigned AddrSpace) {
  SmallString<128> Tmp;
  raw_svector_ostream OSE(Tmp);
  encodeSLEB128(Value, OSE);
  EmitBytes(OSE.str(), AddrSpace);
}

void MCStreamer::EmitAbsValue(const MCExpr *Value, unsigned Size,
                              unsigned AddrSpace) {
  const MCExpr *ABS = ForceExpAbs(Value);
  EmitValue(ABS, Size, AddrSpace);
}


void MCStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                           unsigned AddrSpace) {
  EmitValueImpl(Value, Size, AddrSpace);
}

void MCStreamer::EmitSymbolValue(const MCSymbol *Sym, unsigned Size,
                                  unsigned AddrSpace) {
  EmitValueImpl(MCSymbolRefExpr::Create(Sym, getContext()), Size,
                AddrSpace);
}

void MCStreamer::EmitGPRel64Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

void MCStreamer::EmitGPRel32Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

/// EmitFill - Emit NumBytes bytes worth of the value specified by
/// FillValue.  This implements directives such as '.space'.
void MCStreamer::EmitFill(uint64_t NumBytes, uint8_t FillValue,
                          unsigned AddrSpace) {
  const MCExpr *E = MCConstantExpr::Create(FillValue, getContext());
  for (uint64_t i = 0, e = NumBytes; i != e; ++i)
    EmitValue(E, 1, AddrSpace);
}

bool MCStreamer::EmitDwarfFileDirective(unsigned FileNo,
                                        StringRef Directory,
                                        StringRef Filename, unsigned CUID) {
  return getContext().GetDwarfFile(Directory, Filename, FileNo, CUID) == 0;
}

void MCStreamer::EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                       unsigned Column, unsigned Flags,
                                       unsigned Isa,
                                       unsigned Discriminator,
                                       StringRef FileName) {
  getContext().setCurrentDwarfLoc(FileNo, Line, Column, Flags, Isa,
                                  Discriminator);
}

MCDwarfFrameInfo *MCStreamer::getCurrentFrameInfo() {
  if (FrameInfos.empty())
    return 0;
  return &FrameInfos.back();
}

void MCStreamer::EnsureValidFrame() {
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  if (!CurFrame || CurFrame->End)
    report_fatal_error("No open frame");
}

void MCStreamer::EmitEHSymAttributes(const MCSymbol *Symbol,
                                     MCSymbol *EHSymbol) {
}

void MCStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(!Symbol->isVariable() && "Cannot emit a variable symbol!");
  assert(getCurrentSection().first && "Cannot emit before setting section!");
  Symbol->setSection(*getCurrentSection().first);
  LastSymbol = Symbol;
}

void MCStreamer::EmitDebugLabel(MCSymbol *Symbol) {
  assert(!Symbol->isVariable() && "Cannot emit a variable symbol!");
  assert(getCurrentSection().first && "Cannot emit before setting section!");
  Symbol->setSection(*getCurrentSection().first);
  LastSymbol = Symbol;
}

void MCStreamer::EmitCompactUnwindEncoding(uint32_t CompactUnwindEncoding) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->CompactUnwindEncoding = CompactUnwindEncoding;
}

void MCStreamer::EmitCFISections(bool EH, bool Debug) {
  assert(EH || Debug);
  EmitEHFrame = EH;
  EmitDebugFrame = Debug;
}

void MCStreamer::EmitCFIStartProc() {
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  if (CurFrame && !CurFrame->End)
    report_fatal_error("Starting a frame before finishing the previous one!");

  MCDwarfFrameInfo Frame;
  EmitCFIStartProcImpl(Frame);

  FrameInfos.push_back(Frame);
}

void MCStreamer::EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame) {
}

void MCStreamer::RecordProcStart(MCDwarfFrameInfo &Frame) {
  Frame.Function = LastSymbol;
  // If the function is externally visible, we need to create a local
  // symbol to avoid relocations.
  StringRef Prefix = getContext().getAsmInfo().getPrivateGlobalPrefix();
  if (LastSymbol && LastSymbol->getName().startswith(Prefix)) {
    Frame.Begin = LastSymbol;
  } else {
    Frame.Begin = getContext().CreateTempSymbol();
    EmitLabel(Frame.Begin);
  }
}

void MCStreamer::EmitCFIEndProc() {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  EmitCFIEndProcImpl(*CurFrame);
}

void MCStreamer::EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame) {
}

void MCStreamer::RecordProcEnd(MCDwarfFrameInfo &Frame) {
  Frame.End = getContext().CreateTempSymbol();
  EmitLabel(Frame.End);
}

MCSymbol *MCStreamer::EmitCFICommon() {
  EnsureValidFrame();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  return Label;
}

void MCStreamer::EmitCFIDefCfa(int64_t Register, int64_t Offset) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createDefCfa(Label, Register, Offset);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIDefCfaOffset(int64_t Offset) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createDefCfaOffset(Label, Offset);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIAdjustCfaOffset(int64_t Adjustment) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createAdjustCfaOffset(Label, Adjustment);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIDefCfaRegister(int64_t Register) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createDefCfaRegister(Label, Register);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIOffset(int64_t Register, int64_t Offset) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createOffset(Label, Register, Offset);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRelOffset(int64_t Register, int64_t Offset) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createRelOffset(Label, Register, Offset);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIPersonality(const MCSymbol *Sym,
                                    unsigned Encoding) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Personality = Sym;
  CurFrame->PersonalityEncoding = Encoding;
}

void MCStreamer::EmitCFILsda(const MCSymbol *Sym, unsigned Encoding) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Lsda = Sym;
  CurFrame->LsdaEncoding = Encoding;
}

void MCStreamer::EmitCFIRememberState() {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction = MCCFIInstruction::createRememberState(Label);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRestoreState() {
  // FIXME: Error if there is no matching cfi_remember_state.
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction = MCCFIInstruction::createRestoreState(Label);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFISameValue(int64_t Register) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createSameValue(Label, Register);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRestore(int64_t Register) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createRestore(Label, Register);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIEscape(StringRef Values) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction = MCCFIInstruction::createEscape(Label, Values);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFISignalFrame() {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->IsSignalFrame = true;
}

void MCStreamer::EmitCFIUndefined(int64_t Register) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createUndefined(Label, Register);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRegister(int64_t Register1, int64_t Register2) {
  MCSymbol *Label = EmitCFICommon();
  MCCFIInstruction Instruction =
    MCCFIInstruction::createRegister(Label, Register1, Register2);
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::setCurrentW64UnwindInfo(MCWin64EHUnwindInfo *Frame) {
  W64UnwindInfos.push_back(Frame);
  CurrentW64UnwindInfo = W64UnwindInfos.back();
}

void MCStreamer::EnsureValidW64UnwindInfo() {
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (!CurFrame || CurFrame->End)
    report_fatal_error("No open Win64 EH frame function!");
}

void MCStreamer::EmitWin64EHStartProc(const MCSymbol *Symbol) {
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame && !CurFrame->End)
    report_fatal_error("Starting a function before ending the previous one!");
  MCWin64EHUnwindInfo *Frame = new MCWin64EHUnwindInfo;
  Frame->Begin = getContext().CreateTempSymbol();
  Frame->Function = Symbol;
  EmitLabel(Frame->Begin);
  setCurrentW64UnwindInfo(Frame);
}

void MCStreamer::EmitWin64EHEndProc() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame->ChainedParent)
    report_fatal_error("Not all chained regions terminated!");
  CurFrame->End = getContext().CreateTempSymbol();
  EmitLabel(CurFrame->End);
}

void MCStreamer::EmitWin64EHStartChained() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *Frame = new MCWin64EHUnwindInfo;
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  Frame->Begin = getContext().CreateTempSymbol();
  Frame->Function = CurFrame->Function;
  Frame->ChainedParent = CurFrame;
  EmitLabel(Frame->Begin);
  setCurrentW64UnwindInfo(Frame);
}

void MCStreamer::EmitWin64EHEndChained() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (!CurFrame->ChainedParent)
    report_fatal_error("End of a chained region outside a chained region!");
  CurFrame->End = getContext().CreateTempSymbol();
  EmitLabel(CurFrame->End);
  CurrentW64UnwindInfo = CurFrame->ChainedParent;
}

void MCStreamer::EmitWin64EHHandler(const MCSymbol *Sym, bool Unwind,
                                    bool Except) {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame->ChainedParent)
    report_fatal_error("Chained unwind areas can't have handlers!");
  CurFrame->ExceptionHandler = Sym;
  if (!Except && !Unwind)
    report_fatal_error("Don't know what kind of handler this is!");
  if (Unwind)
    CurFrame->HandlesUnwind = true;
  if (Except)
    CurFrame->HandlesExceptions = true;
}

void MCStreamer::EmitWin64EHHandlerData() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame->ChainedParent)
    report_fatal_error("Chained unwind areas can't have handlers!");
}

void MCStreamer::EmitWin64EHPushReg(unsigned Register) {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCSymbol *Label = getContext().CreateTempSymbol();
  MCWin64EHInstruction Inst(Win64EH::UOP_PushNonVol, Label, Register);
  EmitLabel(Label);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHSetFrame(unsigned Register, unsigned Offset) {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame->LastFrameInst >= 0)
    report_fatal_error("Frame register and offset already specified!");
  if (Offset & 0x0F)
    report_fatal_error("Misaligned frame pointer offset!");
  MCWin64EHInstruction Inst(Win64EH::UOP_SetFPReg, 0, Register, Offset);
  CurFrame->LastFrameInst = CurFrame->Instructions.size();
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHAllocStack(unsigned Size) {
  EnsureValidW64UnwindInfo();
  if (Size & 7)
    report_fatal_error("Misaligned stack allocation!");
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCSymbol *Label = getContext().CreateTempSymbol();
  MCWin64EHInstruction Inst(Label, Size);
  EmitLabel(Label);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHSaveReg(unsigned Register, unsigned Offset) {
  EnsureValidW64UnwindInfo();
  if (Offset & 7)
    report_fatal_error("Misaligned saved register offset!");
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCSymbol *Label = getContext().CreateTempSymbol();
  MCWin64EHInstruction Inst(
     Offset > 512*1024-8 ? Win64EH::UOP_SaveNonVolBig : Win64EH::UOP_SaveNonVol,
                            Label, Register, Offset);
  EmitLabel(Label);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHSaveXMM(unsigned Register, unsigned Offset) {
  EnsureValidW64UnwindInfo();
  if (Offset & 0x0F)
    report_fatal_error("Misaligned saved vector register offset!");
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCSymbol *Label = getContext().CreateTempSymbol();
  MCWin64EHInstruction Inst(
    Offset > 512*1024-16 ? Win64EH::UOP_SaveXMM128Big : Win64EH::UOP_SaveXMM128,
                            Label, Register, Offset);
  EmitLabel(Label);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHPushFrame(bool Code) {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame->Instructions.size() > 0)
    report_fatal_error("If present, PushMachFrame must be the first UOP");
  MCSymbol *Label = getContext().CreateTempSymbol();
  MCWin64EHInstruction Inst(Win64EH::UOP_PushMachFrame, Label, Code);
  EmitLabel(Label);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHEndProlog() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  CurFrame->PrologEnd = getContext().CreateTempSymbol();
  EmitLabel(CurFrame->PrologEnd);
}

void MCStreamer::EmitCOFFSecRel32(MCSymbol const *Symbol) {
  llvm_unreachable("This file format doesn't support this directive");
}

void MCStreamer::EmitFnStart() {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitFnEnd() {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitCantUnwind() {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitHandlerData() {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitPersonality(const MCSymbol *Personality) {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitSetFP(unsigned FpReg, unsigned SpReg, int64_t Offset) {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitPad(int64_t Offset) {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitRegSave(const SmallVectorImpl<unsigned> &RegList, bool) {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitTCEntry(const MCSymbol &S) {
  llvm_unreachable("Unsupported method");
}

/// EmitRawText - If this file is backed by an assembly streamer, this dumps
/// the specified string in the output .s file.  This capability is
/// indicated by the hasRawTextSupport() predicate.
void MCStreamer::EmitRawText(StringRef String) {
  errs() << "EmitRawText called on an MCStreamer that doesn't support it, "
  " something must not be fully mc'ized\n";
  abort();
}

void MCStreamer::EmitRawText(const Twine &T) {
  SmallString<128> Str;
  T.toVector(Str);
  EmitRawText(Str.str());
}

void MCStreamer::EmitFrames(bool usingCFI) {
  if (!getNumFrameInfos())
    return;

  if (EmitEHFrame)
    MCDwarfFrameEmitter::Emit(*this, usingCFI, true);

  if (EmitDebugFrame)
    MCDwarfFrameEmitter::Emit(*this, usingCFI, false);
}

void MCStreamer::EmitW64Tables() {
  if (!getNumW64UnwindInfos())
    return;

  MCWin64EHUnwindEmitter::Emit(*this);
}

void MCStreamer::Finish() {
  if (!FrameInfos.empty() && !FrameInfos.back().End)
    report_fatal_error("Unfinished frame!");

  FinishImpl();
}

MCSymbolData &MCStreamer::getOrCreateSymbolData(MCSymbol *Symbol) {
  report_fatal_error("Not supported!");
  return *(static_cast<MCSymbolData*>(0));
}
