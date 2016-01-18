//===-- CodeGen/AsmPrinter/WinException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Win64 exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "WinException.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCWin64EH.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

WinException::WinException(AsmPrinter *A) : EHStreamer(A) {
  // MSVC's EH tables are always composed of 32-bit words.  All known 64-bit
  // platforms use an imagerel32 relocation to refer to symbols.
  useImageRel32 = (A->getDataLayout().getPointerSizeInBits() == 64);
}

WinException::~WinException() {}

/// endModule - Emit all exception information that should come after the
/// content.
void WinException::endModule() {
  auto &OS = *Asm->OutStreamer;
  const Module *M = MMI->getModule();
  for (const Function &F : *M)
    if (F.hasFnAttribute("safeseh"))
      OS.EmitCOFFSafeSEH(Asm->getSymbol(&F));
}

void WinException::beginFunction(const MachineFunction *MF) {
  shouldEmitMoves = shouldEmitPersonality = shouldEmitLSDA = false;

  // If any landing pads survive, we need an EH table.
  bool hasLandingPads = !MMI->getLandingPads().empty();
  bool hasEHFunclets = MMI->hasEHFunclets();

  const Function *F = MF->getFunction();

  shouldEmitMoves = Asm->needsSEHMoves();

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  unsigned PerEncoding = TLOF.getPersonalityEncoding();
  const Function *Per = nullptr;
  if (F->hasPersonalityFn())
    Per = dyn_cast<Function>(F->getPersonalityFn()->stripPointerCasts());

  bool forceEmitPersonality =
    F->hasPersonalityFn() && !isNoOpWithoutInvoke(classifyEHPersonality(Per)) &&
    F->needsUnwindTableEntry();

  shouldEmitPersonality =
      forceEmitPersonality || ((hasLandingPads || hasEHFunclets) &&
                               PerEncoding != dwarf::DW_EH_PE_omit && Per);

  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  shouldEmitLSDA = shouldEmitPersonality &&
    LSDAEncoding != dwarf::DW_EH_PE_omit;

  // If we're not using CFI, we don't want the CFI or the personality, but we
  // might want EH tables if we had EH pads.
  if (!Asm->MAI->usesWindowsCFI()) {
    shouldEmitLSDA = hasEHFunclets;
    shouldEmitPersonality = false;
    return;
  }

  beginFunclet(MF->front(), Asm->CurrentFnSym);
}

/// endFunction - Gather and emit post-function exception information.
///
void WinException::endFunction(const MachineFunction *MF) {
  if (!shouldEmitPersonality && !shouldEmitMoves && !shouldEmitLSDA)
    return;

  const Function *F = MF->getFunction();
  EHPersonality Per = EHPersonality::Unknown;
  if (F->hasPersonalityFn())
    Per = classifyEHPersonality(F->getPersonalityFn());

  // Get rid of any dead landing pads if we're not using funclets. In funclet
  // schemes, the landing pad is not actually reachable. It only exists so
  // that we can emit the right table data.
  if (!isFuncletEHPersonality(Per))
    MMI->TidyLandingPads();

  endFunclet();

  // endFunclet will emit the necessary .xdata tables for x64 SEH.
  if (Per == EHPersonality::MSVC_Win64SEH && MMI->hasEHFunclets())
    return;

  if (shouldEmitPersonality || shouldEmitLSDA) {
    Asm->OutStreamer->PushSection();

    // Just switch sections to the right xdata section. This use of CurrentFnSym
    // assumes that we only emit the LSDA when ending the parent function.
    MCSection *XData = WinEH::UnwindEmitter::getXDataSection(Asm->CurrentFnSym,
                                                             Asm->OutContext);
    Asm->OutStreamer->SwitchSection(XData);

    // Emit the tables appropriate to the personality function in use. If we
    // don't recognize the personality, assume it uses an Itanium-style LSDA.
    if (Per == EHPersonality::MSVC_Win64SEH)
      emitCSpecificHandlerTable(MF);
    else if (Per == EHPersonality::MSVC_X86SEH)
      emitExceptHandlerTable(MF);
    else if (Per == EHPersonality::MSVC_CXX)
      emitCXXFrameHandler3Table(MF);
    else if (Per == EHPersonality::CoreCLR)
      emitCLRExceptionTable(MF);
    else
      emitExceptionTable();

    Asm->OutStreamer->PopSection();
  }
}

/// Retreive the MCSymbol for a GlobalValue or MachineBasicBlock.
static MCSymbol *getMCSymbolForMBB(AsmPrinter *Asm,
                                   const MachineBasicBlock *MBB) {
  if (!MBB)
    return nullptr;

  assert(MBB->isEHFuncletEntry());

  // Give catches and cleanups a name based off of their parent function and
  // their funclet entry block's number.
  const MachineFunction *MF = MBB->getParent();
  const Function *F = MF->getFunction();
  StringRef FuncLinkageName = GlobalValue::getRealLinkageName(F->getName());
  MCContext &Ctx = MF->getContext();
  StringRef HandlerPrefix = MBB->isCleanupFuncletEntry() ? "dtor" : "catch";
  return Ctx.getOrCreateSymbol("?" + HandlerPrefix + "$" +
                               Twine(MBB->getNumber()) + "@?0?" +
                               FuncLinkageName + "@4HA");
}

void WinException::beginFunclet(const MachineBasicBlock &MBB,
                                MCSymbol *Sym) {
  CurrentFuncletEntry = &MBB;

  const Function *F = Asm->MF->getFunction();
  // If a symbol was not provided for the funclet, invent one.
  if (!Sym) {
    Sym = getMCSymbolForMBB(Asm, &MBB);

    // Describe our funclet symbol as a function with internal linkage.
    Asm->OutStreamer->BeginCOFFSymbolDef(Sym);
    Asm->OutStreamer->EmitCOFFSymbolStorageClass(COFF::IMAGE_SYM_CLASS_STATIC);
    Asm->OutStreamer->EmitCOFFSymbolType(COFF::IMAGE_SYM_DTYPE_FUNCTION
                                         << COFF::SCT_COMPLEX_TYPE_SHIFT);
    Asm->OutStreamer->EndCOFFSymbolDef();

    // We want our funclet's entry point to be aligned such that no nops will be
    // present after the label.
    Asm->EmitAlignment(std::max(Asm->MF->getAlignment(), MBB.getAlignment()),
                       F);

    // Now that we've emitted the alignment directive, point at our funclet.
    Asm->OutStreamer->EmitLabel(Sym);
  }

  // Mark 'Sym' as starting our funclet.
  if (shouldEmitMoves || shouldEmitPersonality)
    Asm->OutStreamer->EmitWinCFIStartProc(Sym);

  if (shouldEmitPersonality) {
    const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
    const Function *PerFn = nullptr;

    // Determine which personality routine we are using for this funclet.
    if (F->hasPersonalityFn())
      PerFn = dyn_cast<Function>(F->getPersonalityFn()->stripPointerCasts());
    const MCSymbol *PersHandlerSym =
        TLOF.getCFIPersonalitySymbol(PerFn, *Asm->Mang, Asm->TM, MMI);

    // Classify the personality routine so that we may reason about it.
    EHPersonality Per = EHPersonality::Unknown;
    if (F->hasPersonalityFn())
      Per = classifyEHPersonality(F->getPersonalityFn());

    // Do not emit a .seh_handler directive if it is a C++ cleanup funclet.
    if (Per != EHPersonality::MSVC_CXX ||
        !CurrentFuncletEntry->isCleanupFuncletEntry())
      Asm->OutStreamer->EmitWinEHHandler(PersHandlerSym, true, true);
  }
}

void WinException::endFunclet() {
  // No funclet to process?  Great, we have nothing to do.
  if (!CurrentFuncletEntry)
    return;

  if (shouldEmitMoves || shouldEmitPersonality) {
    const Function *F = Asm->MF->getFunction();
    EHPersonality Per = EHPersonality::Unknown;
    if (F->hasPersonalityFn())
      Per = classifyEHPersonality(F->getPersonalityFn());

    // The .seh_handlerdata directive implicitly switches section, push the
    // current section so that we may return to it.
    Asm->OutStreamer->PushSection();

    // Emit an UNWIND_INFO struct describing the prologue.
    Asm->OutStreamer->EmitWinEHHandlerData();

    if (Per == EHPersonality::MSVC_CXX && shouldEmitPersonality &&
        !CurrentFuncletEntry->isCleanupFuncletEntry()) {
      // If this is a C++ catch funclet (or the parent function),
      // emit a reference to the LSDA for the parent function.
      StringRef FuncLinkageName = GlobalValue::getRealLinkageName(F->getName());
      MCSymbol *FuncInfoXData = Asm->OutContext.getOrCreateSymbol(
          Twine("$cppxdata$", FuncLinkageName));
      Asm->OutStreamer->EmitValue(create32bitRef(FuncInfoXData), 4);
    } else if (Per == EHPersonality::MSVC_Win64SEH && MMI->hasEHFunclets() &&
               !CurrentFuncletEntry->isEHFuncletEntry()) {
      // If this is the parent function in Win64 SEH, emit the LSDA immediately
      // following .seh_handlerdata.
      emitCSpecificHandlerTable(Asm->MF);
    }

    // Switch back to the previous section now that we are done writing to
    // .xdata.
    Asm->OutStreamer->PopSection();

    // Emit a .seh_endproc directive to mark the end of the function.
    Asm->OutStreamer->EmitWinCFIEndProc();
  }

  // Let's make sure we don't try to end the same funclet twice.
  CurrentFuncletEntry = nullptr;
}

const MCExpr *WinException::create32bitRef(const MCSymbol *Value) {
  if (!Value)
    return MCConstantExpr::create(0, Asm->OutContext);
  return MCSymbolRefExpr::create(Value, useImageRel32
                                            ? MCSymbolRefExpr::VK_COFF_IMGREL32
                                            : MCSymbolRefExpr::VK_None,
                                 Asm->OutContext);
}

const MCExpr *WinException::create32bitRef(const GlobalValue *GV) {
  if (!GV)
    return MCConstantExpr::create(0, Asm->OutContext);
  return create32bitRef(Asm->getSymbol(GV));
}

const MCExpr *WinException::getLabelPlusOne(const MCSymbol *Label) {
  return MCBinaryExpr::createAdd(create32bitRef(Label),
                                 MCConstantExpr::create(1, Asm->OutContext),
                                 Asm->OutContext);
}

const MCExpr *WinException::getOffset(const MCSymbol *OffsetOf,
                                      const MCSymbol *OffsetFrom) {
  return MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(OffsetOf, Asm->OutContext),
      MCSymbolRefExpr::create(OffsetFrom, Asm->OutContext), Asm->OutContext);
}

const MCExpr *WinException::getOffsetPlusOne(const MCSymbol *OffsetOf,
                                             const MCSymbol *OffsetFrom) {
  return MCBinaryExpr::createAdd(getOffset(OffsetOf, OffsetFrom),
                                 MCConstantExpr::create(1, Asm->OutContext),
                                 Asm->OutContext);
}

int WinException::getFrameIndexOffset(int FrameIndex, WinEHFuncInfo &FuncInfo) {
  const TargetFrameLowering &TFI = *Asm->MF->getSubtarget().getFrameLowering();
  unsigned UnusedReg;
  if (Asm->MAI->usesWindowsCFI())
    return TFI.getFrameIndexReferenceFromSP(*Asm->MF, FrameIndex, UnusedReg);
  // For 32-bit, offsets should be relative to the end of the EH registration
  // node. For 64-bit, it's relative to SP at the end of the prologue.
  assert(FuncInfo.EHRegNodeEndOffset != INT_MAX);
  int Offset = TFI.getFrameIndexReference(*Asm->MF, FrameIndex, UnusedReg);
  Offset += FuncInfo.EHRegNodeEndOffset;
  return Offset;
}

namespace {

/// Top-level state used to represent unwind to caller
const int NullState = -1;

struct InvokeStateChange {
  /// EH Label immediately after the last invoke in the previous state, or
  /// nullptr if the previous state was the null state.
  const MCSymbol *PreviousEndLabel;

  /// EH label immediately before the first invoke in the new state, or nullptr
  /// if the new state is the null state.
  const MCSymbol *NewStartLabel;

  /// State of the invoke following NewStartLabel, or NullState to indicate
  /// the presence of calls which may unwind to caller.
  int NewState;
};

/// Iterator that reports all the invoke state changes in a range of machine
/// basic blocks.  Changes to the null state are reported whenever a call that
/// may unwind to caller is encountered.  The MBB range is expected to be an
/// entire function or funclet, and the start and end of the range are treated
/// as being in the NullState even if there's not an unwind-to-caller call
/// before the first invoke or after the last one (i.e., the first state change
/// reported is the first change to something other than NullState, and a
/// change back to NullState is always reported at the end of iteration).
class InvokeStateChangeIterator {
  InvokeStateChangeIterator(WinEHFuncInfo &EHInfo,
                            MachineFunction::const_iterator MFI,
                            MachineFunction::const_iterator MFE,
                            MachineBasicBlock::const_iterator MBBI)
      : EHInfo(EHInfo), MFI(MFI), MFE(MFE), MBBI(MBBI) {
    LastStateChange.PreviousEndLabel = nullptr;
    LastStateChange.NewStartLabel = nullptr;
    LastStateChange.NewState = NullState;
    scan();
  }

public:
  static iterator_range<InvokeStateChangeIterator>
  range(WinEHFuncInfo &EHInfo, const MachineFunction &MF) {
    // Reject empty MFs to simplify bookkeeping by ensuring that we can get the
    // end of the last block.
    assert(!MF.empty());
    auto FuncBegin = MF.begin();
    auto FuncEnd = MF.end();
    auto BlockBegin = FuncBegin->begin();
    auto BlockEnd = MF.back().end();
    return make_range(
        InvokeStateChangeIterator(EHInfo, FuncBegin, FuncEnd, BlockBegin),
        InvokeStateChangeIterator(EHInfo, FuncEnd, FuncEnd, BlockEnd));
  }
  static iterator_range<InvokeStateChangeIterator>
  range(WinEHFuncInfo &EHInfo, MachineFunction::const_iterator Begin,
        MachineFunction::const_iterator End) {
    // Reject empty ranges to simplify bookkeeping by ensuring that we can get
    // the end of the last block.
    assert(Begin != End);
    auto BlockBegin = Begin->begin();
    auto BlockEnd = std::prev(End)->end();
    return make_range(InvokeStateChangeIterator(EHInfo, Begin, End, BlockBegin),
                      InvokeStateChangeIterator(EHInfo, End, End, BlockEnd));
  }

  // Iterator methods.
  bool operator==(const InvokeStateChangeIterator &O) const {
    // Must be visiting same block.
    if (MFI != O.MFI)
      return false;
    // Must be visiting same isntr.
    if (MBBI != O.MBBI)
      return false;
    // At end of block/instr iteration, we can still have two distinct states:
    // one to report the final EndLabel, and another indicating the end of the
    // state change iteration.  Check for CurrentEndLabel equality to
    // distinguish these.
    return CurrentEndLabel == O.CurrentEndLabel;
  }

  bool operator!=(const InvokeStateChangeIterator &O) const {
    return !operator==(O);
  }
  InvokeStateChange &operator*() { return LastStateChange; }
  InvokeStateChange *operator->() { return &LastStateChange; }
  InvokeStateChangeIterator &operator++() { return scan(); }

private:
  InvokeStateChangeIterator &scan();

  WinEHFuncInfo &EHInfo;
  const MCSymbol *CurrentEndLabel = nullptr;
  MachineFunction::const_iterator MFI;
  MachineFunction::const_iterator MFE;
  MachineBasicBlock::const_iterator MBBI;
  InvokeStateChange LastStateChange;
  bool VisitingInvoke = false;
};

} // end anonymous namespace

InvokeStateChangeIterator &InvokeStateChangeIterator::scan() {
  bool IsNewBlock = false;
  for (; MFI != MFE; ++MFI, IsNewBlock = true) {
    if (IsNewBlock)
      MBBI = MFI->begin();
    for (auto MBBE = MFI->end(); MBBI != MBBE; ++MBBI) {
      const MachineInstr &MI = *MBBI;
      if (!VisitingInvoke && LastStateChange.NewState != NullState &&
          MI.isCall() && !EHStreamer::callToNoUnwindFunction(&MI)) {
        // Indicate a change of state to the null state.  We don't have
        // start/end EH labels handy but the caller won't expect them for
        // null state regions.
        LastStateChange.PreviousEndLabel = CurrentEndLabel;
        LastStateChange.NewStartLabel = nullptr;
        LastStateChange.NewState = NullState;
        CurrentEndLabel = nullptr;
        // Don't re-visit this instr on the next scan
        ++MBBI;
        return *this;
      }

      // All other state changes are at EH labels before/after invokes.
      if (!MI.isEHLabel())
        continue;
      MCSymbol *Label = MI.getOperand(0).getMCSymbol();
      if (Label == CurrentEndLabel) {
        VisitingInvoke = false;
        continue;
      }
      auto InvokeMapIter = EHInfo.InvokeToStateMap.find(Label);
      // Ignore EH labels that aren't the ones inserted before an invoke
      if (InvokeMapIter == EHInfo.InvokeToStateMap.end())
        continue;
      auto &StateAndEnd = InvokeMapIter->second;
      int NewState = StateAndEnd.first;
      // Ignore EH labels explicitly annotated with the null state (which
      // can happen for invokes that unwind to a chain of endpads the last
      // of which unwinds to caller).  We'll see the subsequent invoke and
      // report a transition to the null state same as we do for calls.
      if (NewState == NullState)
        continue;
      // Keep track of the fact that we're between EH start/end labels so
      // we know not to treat the inoke we'll see as unwinding to caller.
      VisitingInvoke = true;
      if (NewState == LastStateChange.NewState) {
        // The state isn't actually changing here.  Record the new end and
        // keep going.
        CurrentEndLabel = StateAndEnd.second;
        continue;
      }
      // Found a state change to report
      LastStateChange.PreviousEndLabel = CurrentEndLabel;
      LastStateChange.NewStartLabel = Label;
      LastStateChange.NewState = NewState;
      // Start keeping track of the new current end
      CurrentEndLabel = StateAndEnd.second;
      // Don't re-visit this instr on the next scan
      ++MBBI;
      return *this;
    }
  }
  // Iteration hit the end of the block range.
  if (LastStateChange.NewState != NullState) {
    // Report the end of the last new state
    LastStateChange.PreviousEndLabel = CurrentEndLabel;
    LastStateChange.NewStartLabel = nullptr;
    LastStateChange.NewState = NullState;
    // Leave CurrentEndLabel non-null to distinguish this state from end.
    assert(CurrentEndLabel != nullptr);
    return *this;
  }
  // We've reported all state changes and hit the end state.
  CurrentEndLabel = nullptr;
  return *this;
}

/// Emit the language-specific data that __C_specific_handler expects.  This
/// handler lives in the x64 Microsoft C runtime and allows catching or cleaning
/// up after faults with __try, __except, and __finally.  The typeinfo values
/// are not really RTTI data, but pointers to filter functions that return an
/// integer (1, 0, or -1) indicating how to handle the exception. For __finally
/// blocks and other cleanups, the landing pad label is zero, and the filter
/// function is actually a cleanup handler with the same prototype.  A catch-all
/// entry is modeled with a null filter function field and a non-zero landing
/// pad label.
///
/// Possible filter function return values:
///   EXCEPTION_EXECUTE_HANDLER (1):
///     Jump to the landing pad label after cleanups.
///   EXCEPTION_CONTINUE_SEARCH (0):
///     Continue searching this table or continue unwinding.
///   EXCEPTION_CONTINUE_EXECUTION (-1):
///     Resume execution at the trapping PC.
///
/// Inferred table structure:
///   struct Table {
///     int NumEntries;
///     struct Entry {
///       imagerel32 LabelStart;
///       imagerel32 LabelEnd;
///       imagerel32 FilterOrFinally;  // One means catch-all.
///       imagerel32 LabelLPad;        // Zero means __finally.
///     } Entries[NumEntries];
///   };
void WinException::emitCSpecificHandlerTable(const MachineFunction *MF) {
  auto &OS = *Asm->OutStreamer;
  MCContext &Ctx = Asm->OutContext;

  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(MF->getFunction());
  // Use the assembler to compute the number of table entries through label
  // difference and division.
  MCSymbol *TableBegin =
      Ctx.createTempSymbol("lsda_begin", /*AlwaysAddSuffix=*/true);
  MCSymbol *TableEnd =
      Ctx.createTempSymbol("lsda_end", /*AlwaysAddSuffix=*/true);
  const MCExpr *LabelDiff = getOffset(TableEnd, TableBegin);
  const MCExpr *EntrySize = MCConstantExpr::create(16, Ctx);
  const MCExpr *EntryCount = MCBinaryExpr::createDiv(LabelDiff, EntrySize, Ctx);
  OS.EmitValue(EntryCount, 4);

  OS.EmitLabel(TableBegin);

  // Iterate over all the invoke try ranges. Unlike MSVC, LLVM currently only
  // models exceptions from invokes. LLVM also allows arbitrary reordering of
  // the code, so our tables end up looking a bit different. Rather than
  // trying to match MSVC's tables exactly, we emit a denormalized table.  For
  // each range of invokes in the same state, we emit table entries for all
  // the actions that would be taken in that state. This means our tables are
  // slightly bigger, which is OK.
  const MCSymbol *LastStartLabel = nullptr;
  int LastEHState = -1;
  // Break out before we enter into a finally funclet.
  // FIXME: We need to emit separate EH tables for cleanups.
  MachineFunction::const_iterator End = MF->end();
  MachineFunction::const_iterator Stop = std::next(MF->begin());
  while (Stop != End && !Stop->isEHFuncletEntry())
    ++Stop;
  for (const auto &StateChange :
       InvokeStateChangeIterator::range(FuncInfo, MF->begin(), Stop)) {
    // Emit all the actions for the state we just transitioned out of
    // if it was not the null state
    if (LastEHState != -1)
      emitSEHActionsForRange(FuncInfo, LastStartLabel,
                             StateChange.PreviousEndLabel, LastEHState);
    LastStartLabel = StateChange.NewStartLabel;
    LastEHState = StateChange.NewState;
  }

  OS.EmitLabel(TableEnd);
}

void WinException::emitSEHActionsForRange(WinEHFuncInfo &FuncInfo,
                                          const MCSymbol *BeginLabel,
                                          const MCSymbol *EndLabel, int State) {
  auto &OS = *Asm->OutStreamer;
  MCContext &Ctx = Asm->OutContext;

  assert(BeginLabel && EndLabel);
  while (State != -1) {
    SEHUnwindMapEntry &UME = FuncInfo.SEHUnwindMap[State];
    const MCExpr *FilterOrFinally;
    const MCExpr *ExceptOrNull;
    auto *Handler = UME.Handler.get<MachineBasicBlock *>();
    if (UME.IsFinally) {
      FilterOrFinally = create32bitRef(getMCSymbolForMBB(Asm, Handler));
      ExceptOrNull = MCConstantExpr::create(0, Ctx);
    } else {
      // For an except, the filter can be 1 (catch-all) or a function
      // label.
      FilterOrFinally = UME.Filter ? create32bitRef(UME.Filter)
                                   : MCConstantExpr::create(1, Ctx);
      ExceptOrNull = create32bitRef(Handler->getSymbol());
    }

    OS.EmitValue(getLabelPlusOne(BeginLabel), 4);
    OS.EmitValue(getLabelPlusOne(EndLabel), 4);
    OS.EmitValue(FilterOrFinally, 4);
    OS.EmitValue(ExceptOrNull, 4);

    assert(UME.ToState < State && "states should decrease");
    State = UME.ToState;
  }
}

void WinException::emitCXXFrameHandler3Table(const MachineFunction *MF) {
  const Function *F = MF->getFunction();
  auto &OS = *Asm->OutStreamer;
  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(F);

  StringRef FuncLinkageName = GlobalValue::getRealLinkageName(F->getName());

  SmallVector<std::pair<const MCExpr *, int>, 4> IPToStateTable;
  MCSymbol *FuncInfoXData = nullptr;
  if (shouldEmitPersonality) {
    // If we're 64-bit, emit a pointer to the C++ EH data, and build a map from
    // IPs to state numbers.
    FuncInfoXData =
        Asm->OutContext.getOrCreateSymbol(Twine("$cppxdata$", FuncLinkageName));
    computeIP2StateTable(MF, FuncInfo, IPToStateTable);
  } else {
    FuncInfoXData = Asm->OutContext.getOrCreateLSDASymbol(FuncLinkageName);
  }

  int UnwindHelpOffset = 0;
  if (Asm->MAI->usesWindowsCFI())
    UnwindHelpOffset =
        getFrameIndexOffset(FuncInfo.UnwindHelpFrameIdx, FuncInfo);

  MCSymbol *UnwindMapXData = nullptr;
  MCSymbol *TryBlockMapXData = nullptr;
  MCSymbol *IPToStateXData = nullptr;
  if (!FuncInfo.CxxUnwindMap.empty())
    UnwindMapXData = Asm->OutContext.getOrCreateSymbol(
        Twine("$stateUnwindMap$", FuncLinkageName));
  if (!FuncInfo.TryBlockMap.empty())
    TryBlockMapXData =
        Asm->OutContext.getOrCreateSymbol(Twine("$tryMap$", FuncLinkageName));
  if (!IPToStateTable.empty())
    IPToStateXData =
        Asm->OutContext.getOrCreateSymbol(Twine("$ip2state$", FuncLinkageName));

  // FuncInfo {
  //   uint32_t           MagicNumber
  //   int32_t            MaxState;
  //   UnwindMapEntry    *UnwindMap;
  //   uint32_t           NumTryBlocks;
  //   TryBlockMapEntry  *TryBlockMap;
  //   uint32_t           IPMapEntries; // always 0 for x86
  //   IPToStateMapEntry *IPToStateMap; // always 0 for x86
  //   uint32_t           UnwindHelp;   // non-x86 only
  //   ESTypeList        *ESTypeList;
  //   int32_t            EHFlags;
  // }
  // EHFlags & 1 -> Synchronous exceptions only, no async exceptions.
  // EHFlags & 2 -> ???
  // EHFlags & 4 -> The function is noexcept(true), unwinding can't continue.
  OS.EmitValueToAlignment(4);
  OS.EmitLabel(FuncInfoXData);
  OS.EmitIntValue(0x19930522, 4);                      // MagicNumber
  OS.EmitIntValue(FuncInfo.CxxUnwindMap.size(), 4);       // MaxState
  OS.EmitValue(create32bitRef(UnwindMapXData), 4);     // UnwindMap
  OS.EmitIntValue(FuncInfo.TryBlockMap.size(), 4);     // NumTryBlocks
  OS.EmitValue(create32bitRef(TryBlockMapXData), 4);   // TryBlockMap
  OS.EmitIntValue(IPToStateTable.size(), 4);           // IPMapEntries
  OS.EmitValue(create32bitRef(IPToStateXData), 4);     // IPToStateMap
  if (Asm->MAI->usesWindowsCFI())
    OS.EmitIntValue(UnwindHelpOffset, 4);              // UnwindHelp
  OS.EmitIntValue(0, 4);                               // ESTypeList
  OS.EmitIntValue(1, 4);                               // EHFlags

  // UnwindMapEntry {
  //   int32_t ToState;
  //   void  (*Action)();
  // };
  if (UnwindMapXData) {
    OS.EmitLabel(UnwindMapXData);
    for (const CxxUnwindMapEntry &UME : FuncInfo.CxxUnwindMap) {
      MCSymbol *CleanupSym =
          getMCSymbolForMBB(Asm, UME.Cleanup.dyn_cast<MachineBasicBlock *>());
      OS.EmitIntValue(UME.ToState, 4);             // ToState
      OS.EmitValue(create32bitRef(CleanupSym), 4); // Action
    }
  }

  // TryBlockMap {
  //   int32_t      TryLow;
  //   int32_t      TryHigh;
  //   int32_t      CatchHigh;
  //   int32_t      NumCatches;
  //   HandlerType *HandlerArray;
  // };
  if (TryBlockMapXData) {
    OS.EmitLabel(TryBlockMapXData);
    SmallVector<MCSymbol *, 1> HandlerMaps;
    for (size_t I = 0, E = FuncInfo.TryBlockMap.size(); I != E; ++I) {
      WinEHTryBlockMapEntry &TBME = FuncInfo.TryBlockMap[I];

      MCSymbol *HandlerMapXData = nullptr;
      if (!TBME.HandlerArray.empty())
        HandlerMapXData =
            Asm->OutContext.getOrCreateSymbol(Twine("$handlerMap$")
                                                  .concat(Twine(I))
                                                  .concat("$")
                                                  .concat(FuncLinkageName));
      HandlerMaps.push_back(HandlerMapXData);

      // TBMEs should form intervals.
      assert(0 <= TBME.TryLow && "bad trymap interval");
      assert(TBME.TryLow <= TBME.TryHigh && "bad trymap interval");
      assert(TBME.TryHigh < TBME.CatchHigh && "bad trymap interval");
      assert(TBME.CatchHigh < int(FuncInfo.CxxUnwindMap.size()) &&
             "bad trymap interval");

      OS.EmitIntValue(TBME.TryLow, 4);                    // TryLow
      OS.EmitIntValue(TBME.TryHigh, 4);                   // TryHigh
      OS.EmitIntValue(TBME.CatchHigh, 4);                 // CatchHigh
      OS.EmitIntValue(TBME.HandlerArray.size(), 4);       // NumCatches
      OS.EmitValue(create32bitRef(HandlerMapXData), 4);   // HandlerArray
    }

    // All funclets use the same parent frame offset currently.
    unsigned ParentFrameOffset = 0;
    if (shouldEmitPersonality) {
      const TargetFrameLowering *TFI = MF->getSubtarget().getFrameLowering();
      ParentFrameOffset = TFI->getWinEHParentFrameOffset(*MF);
    }

    for (size_t I = 0, E = FuncInfo.TryBlockMap.size(); I != E; ++I) {
      WinEHTryBlockMapEntry &TBME = FuncInfo.TryBlockMap[I];
      MCSymbol *HandlerMapXData = HandlerMaps[I];
      if (!HandlerMapXData)
        continue;
      // HandlerType {
      //   int32_t         Adjectives;
      //   TypeDescriptor *Type;
      //   int32_t         CatchObjOffset;
      //   void          (*Handler)();
      //   int32_t         ParentFrameOffset; // x64 only
      // };
      OS.EmitLabel(HandlerMapXData);
      for (const WinEHHandlerType &HT : TBME.HandlerArray) {
        // Get the frame escape label with the offset of the catch object. If
        // the index is INT_MAX, then there is no catch object, and we should
        // emit an offset of zero, indicating that no copy will occur.
        const MCExpr *FrameAllocOffsetRef = nullptr;
        if (HT.CatchObj.FrameIndex != INT_MAX) {
          int Offset = getFrameIndexOffset(HT.CatchObj.FrameIndex, FuncInfo);
          FrameAllocOffsetRef = MCConstantExpr::create(Offset, Asm->OutContext);
        } else {
          FrameAllocOffsetRef = MCConstantExpr::create(0, Asm->OutContext);
        }

        MCSymbol *HandlerSym =
            getMCSymbolForMBB(Asm, HT.Handler.dyn_cast<MachineBasicBlock *>());

        OS.EmitIntValue(HT.Adjectives, 4);                  // Adjectives
        OS.EmitValue(create32bitRef(HT.TypeDescriptor), 4); // Type
        OS.EmitValue(FrameAllocOffsetRef, 4);               // CatchObjOffset
        OS.EmitValue(create32bitRef(HandlerSym), 4);        // Handler
        if (shouldEmitPersonality)
          OS.EmitIntValue(ParentFrameOffset, 4); // ParentFrameOffset
      }
    }
  }

  // IPToStateMapEntry {
  //   void   *IP;
  //   int32_t State;
  // };
  if (IPToStateXData) {
    OS.EmitLabel(IPToStateXData);
    for (auto &IPStatePair : IPToStateTable) {
      OS.EmitValue(IPStatePair.first, 4);     // IP
      OS.EmitIntValue(IPStatePair.second, 4); // State
    }
  }
}

void WinException::computeIP2StateTable(
    const MachineFunction *MF, WinEHFuncInfo &FuncInfo,
    SmallVectorImpl<std::pair<const MCExpr *, int>> &IPToStateTable) {
  // Indicate that all calls from the prologue to the first invoke unwind to
  // caller. We handle this as a special case since other ranges starting at end
  // labels need to use LtmpN+1.
  MCSymbol *StartLabel = Asm->getFunctionBegin();
  assert(StartLabel && "need local function start label");
  IPToStateTable.push_back(std::make_pair(create32bitRef(StartLabel), -1));

  // FIXME: Do we need to emit entries for funclet base states?
  for (const auto &StateChange :
       InvokeStateChangeIterator::range(FuncInfo, *MF)) {
    // Compute the label to report as the start of this entry; use the EH start
    // label for the invoke if we have one, otherwise (this is a call which may
    // unwind to our caller and does not have an EH start label, so) use the
    // previous end label.
    const MCSymbol *ChangeLabel = StateChange.NewStartLabel;
    if (!ChangeLabel)
      ChangeLabel = StateChange.PreviousEndLabel;
    // Emit an entry indicating that PCs after 'Label' have this EH state.
    IPToStateTable.push_back(
        std::make_pair(getLabelPlusOne(ChangeLabel), StateChange.NewState));
  }
}

void WinException::emitEHRegistrationOffsetLabel(const WinEHFuncInfo &FuncInfo,
                                                 StringRef FLinkageName) {
  // Outlined helpers called by the EH runtime need to know the offset of the EH
  // registration in order to recover the parent frame pointer. Now that we know
  // we've code generated the parent, we can emit the label assignment that
  // those helpers use to get the offset of the registration node.
  assert(FuncInfo.EHRegNodeEscapeIndex != INT_MAX &&
         "no EH reg node localescape index");
  MCSymbol *ParentFrameOffset =
      Asm->OutContext.getOrCreateParentFrameOffsetSymbol(FLinkageName);
  MCSymbol *RegistrationOffsetSym = Asm->OutContext.getOrCreateFrameAllocSymbol(
      FLinkageName, FuncInfo.EHRegNodeEscapeIndex);
  const MCExpr *RegistrationOffsetSymRef =
      MCSymbolRefExpr::create(RegistrationOffsetSym, Asm->OutContext);
  Asm->OutStreamer->EmitAssignment(ParentFrameOffset, RegistrationOffsetSymRef);
}

/// Emit the language-specific data that _except_handler3 and 4 expect. This is
/// functionally equivalent to the __C_specific_handler table, except it is
/// indexed by state number instead of IP.
void WinException::emitExceptHandlerTable(const MachineFunction *MF) {
  MCStreamer &OS = *Asm->OutStreamer;
  const Function *F = MF->getFunction();
  StringRef FLinkageName = GlobalValue::getRealLinkageName(F->getName());

  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(F);
  emitEHRegistrationOffsetLabel(FuncInfo, FLinkageName);

  // Emit the __ehtable label that we use for llvm.x86.seh.lsda.
  MCSymbol *LSDALabel = Asm->OutContext.getOrCreateLSDASymbol(FLinkageName);
  OS.EmitValueToAlignment(4);
  OS.EmitLabel(LSDALabel);

  const Function *Per =
      dyn_cast<Function>(F->getPersonalityFn()->stripPointerCasts());
  StringRef PerName = Per->getName();
  int BaseState = -1;
  if (PerName == "_except_handler4") {
    // The LSDA for _except_handler4 starts with this struct, followed by the
    // scope table:
    //
    // struct EH4ScopeTable {
    //   int32_t GSCookieOffset;
    //   int32_t GSCookieXOROffset;
    //   int32_t EHCookieOffset;
    //   int32_t EHCookieXOROffset;
    //   ScopeTableEntry ScopeRecord[];
    // };
    //
    // Only the EHCookieOffset field appears to vary, and it appears to be the
    // offset from the final saved SP value to the retaddr.
    OS.EmitIntValue(-2, 4);
    OS.EmitIntValue(0, 4);
    // FIXME: Calculate.
    OS.EmitIntValue(9999, 4);
    OS.EmitIntValue(0, 4);
    BaseState = -2;
  }

  assert(!FuncInfo.SEHUnwindMap.empty());
  for (SEHUnwindMapEntry &UME : FuncInfo.SEHUnwindMap) {
    MCSymbol *ExceptOrFinally =
        UME.Handler.get<MachineBasicBlock *>()->getSymbol();
    // -1 is usually the base state for "unwind to caller", but for
    // _except_handler4 it's -2. Do that replacement here if necessary.
    int ToState = UME.ToState == -1 ? BaseState : UME.ToState;
    OS.EmitIntValue(ToState, 4);                      // ToState
    OS.EmitValue(create32bitRef(UME.Filter), 4);      // Filter
    OS.EmitValue(create32bitRef(ExceptOrFinally), 4); // Except/Finally
  }
}

static int getRank(WinEHFuncInfo &FuncInfo, int State) {
  int Rank = 0;
  while (State != -1) {
    ++Rank;
    State = FuncInfo.ClrEHUnwindMap[State].Parent;
  }
  return Rank;
}

static int getAncestor(WinEHFuncInfo &FuncInfo, int Left, int Right) {
  int LeftRank = getRank(FuncInfo, Left);
  int RightRank = getRank(FuncInfo, Right);

  while (LeftRank < RightRank) {
    Right = FuncInfo.ClrEHUnwindMap[Right].Parent;
    --RightRank;
  }

  while (RightRank < LeftRank) {
    Left = FuncInfo.ClrEHUnwindMap[Left].Parent;
    --LeftRank;
  }

  while (Left != Right) {
    Left = FuncInfo.ClrEHUnwindMap[Left].Parent;
    Right = FuncInfo.ClrEHUnwindMap[Right].Parent;
  }

  return Left;
}

void WinException::emitCLRExceptionTable(const MachineFunction *MF) {
  // CLR EH "states" are really just IDs that identify handlers/funclets;
  // states, handlers, and funclets all have 1:1 mappings between them, and a
  // handler/funclet's "state" is its index in the ClrEHUnwindMap.
  MCStreamer &OS = *Asm->OutStreamer;
  const Function *F = MF->getFunction();
  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(F);
  MCSymbol *FuncBeginSym = Asm->getFunctionBegin();
  MCSymbol *FuncEndSym = Asm->getFunctionEnd();

  // A ClrClause describes a protected region.
  struct ClrClause {
    const MCSymbol *StartLabel; // Start of protected region
    const MCSymbol *EndLabel;   // End of protected region
    int State;          // Index of handler protecting the protected region
    int EnclosingState; // Index of funclet enclosing the protected region
  };
  SmallVector<ClrClause, 8> Clauses;

  // Build a map from handler MBBs to their corresponding states (i.e. their
  // indices in the ClrEHUnwindMap).
  int NumStates = FuncInfo.ClrEHUnwindMap.size();
  assert(NumStates > 0 && "Don't need exception table!");
  DenseMap<const MachineBasicBlock *, int> HandlerStates;
  for (int State = 0; State < NumStates; ++State) {
    MachineBasicBlock *HandlerBlock =
        FuncInfo.ClrEHUnwindMap[State].Handler.get<MachineBasicBlock *>();
    HandlerStates[HandlerBlock] = State;
    // Use this loop through all handlers to verify our assumption (used in
    // the MinEnclosingState computation) that ancestors have lower state
    // numbers than their descendants.
    assert(FuncInfo.ClrEHUnwindMap[State].Parent < State &&
           "ill-formed state numbering");
  }
  // Map the main function to the NullState.
  HandlerStates[&MF->front()] = NullState;

  // Write out a sentinel indicating the end of the standard (Windows) xdata
  // and the start of the additional (CLR) info.
  OS.EmitIntValue(0xffffffff, 4);
  // Write out the number of funclets
  OS.EmitIntValue(NumStates, 4);

  // Walk the machine blocks/instrs, computing and emitting a few things:
  // 1. Emit a list of the offsets to each handler entry, in lexical order.
  // 2. Compute a map (EndSymbolMap) from each funclet to the symbol at its end.
  // 3. Compute the list of ClrClauses, in the required order (inner before
  //    outer, earlier before later; the order by which a forward scan with
  //    early termination will find the innermost enclosing clause covering
  //    a given address).
  // 4. A map (MinClauseMap) from each handler index to the index of the
  //    outermost funclet/function which contains a try clause targeting the
  //    key handler.  This will be used to determine IsDuplicate-ness when
  //    emitting ClrClauses.  The NullState value is used to indicate that the
  //    top-level function contains a try clause targeting the key handler.
  // HandlerStack is a stack of (PendingStartLabel, PendingState) pairs for
  // try regions we entered before entering the PendingState try but which
  // we haven't yet exited.
  SmallVector<std::pair<const MCSymbol *, int>, 4> HandlerStack;
  // EndSymbolMap and MinClauseMap are maps described above.
  std::unique_ptr<MCSymbol *[]> EndSymbolMap(new MCSymbol *[NumStates]);
  SmallVector<int, 4> MinClauseMap((size_t)NumStates, NumStates);

  // Visit the root function and each funclet.

  for (MachineFunction::const_iterator FuncletStart = MF->begin(),
                                       FuncletEnd = MF->begin(),
                                       End = MF->end();
       FuncletStart != End; FuncletStart = FuncletEnd) {
    int FuncletState = HandlerStates[&*FuncletStart];
    // Find the end of the funclet
    MCSymbol *EndSymbol = FuncEndSym;
    while (++FuncletEnd != End) {
      if (FuncletEnd->isEHFuncletEntry()) {
        EndSymbol = getMCSymbolForMBB(Asm, &*FuncletEnd);
        break;
      }
    }
    // Emit the function/funclet end and, if this is a funclet (and not the
    // root function), record it in the EndSymbolMap.
    OS.EmitValue(getOffset(EndSymbol, FuncBeginSym), 4);
    if (FuncletState != NullState) {
      // Record the end of the handler.
      EndSymbolMap[FuncletState] = EndSymbol;
    }

    // Walk the state changes in this function/funclet and compute its clauses.
    // Funclets always start in the null state.
    const MCSymbol *CurrentStartLabel = nullptr;
    int CurrentState = NullState;
    assert(HandlerStack.empty());
    for (const auto &StateChange :
         InvokeStateChangeIterator::range(FuncInfo, FuncletStart, FuncletEnd)) {
      // Close any try regions we're not still under
      int AncestorState =
          getAncestor(FuncInfo, CurrentState, StateChange.NewState);
      while (CurrentState != AncestorState) {
        assert(CurrentState != NullState && "Failed to find ancestor!");
        // Close the pending clause
        Clauses.push_back({CurrentStartLabel, StateChange.PreviousEndLabel,
                           CurrentState, FuncletState});
        // Now the parent handler is current
        CurrentState = FuncInfo.ClrEHUnwindMap[CurrentState].Parent;
        // Pop the new start label from the handler stack if we've exited all
        // descendants of the corresponding handler.
        if (HandlerStack.back().second == CurrentState)
          CurrentStartLabel = HandlerStack.pop_back_val().first;
      }

      if (StateChange.NewState != CurrentState) {
        // For each clause we're starting, update the MinClauseMap so we can
        // know which is the topmost funclet containing a clause targeting
        // it.
        for (int EnteredState = StateChange.NewState;
             EnteredState != CurrentState;
             EnteredState = FuncInfo.ClrEHUnwindMap[EnteredState].Parent) {
          int &MinEnclosingState = MinClauseMap[EnteredState];
          if (FuncletState < MinEnclosingState)
            MinEnclosingState = FuncletState;
        }
        // Save the previous current start/label on the stack and update to
        // the newly-current start/state.
        HandlerStack.emplace_back(CurrentStartLabel, CurrentState);
        CurrentStartLabel = StateChange.NewStartLabel;
        CurrentState = StateChange.NewState;
      }
    }
    assert(HandlerStack.empty());
  }

  // Now emit the clause info, starting with the number of clauses.
  OS.EmitIntValue(Clauses.size(), 4);
  for (ClrClause &Clause : Clauses) {
    // Emit a CORINFO_EH_CLAUSE :
    /*
      struct CORINFO_EH_CLAUSE
      {
          CORINFO_EH_CLAUSE_FLAGS Flags;         // actually a CorExceptionFlag
          DWORD                   TryOffset;
          DWORD                   TryLength;     // actually TryEndOffset
          DWORD                   HandlerOffset;
          DWORD                   HandlerLength; // actually HandlerEndOffset
          union
          {
              DWORD               ClassToken;   // use for catch clauses
              DWORD               FilterOffset; // use for filter clauses
          };
      };

      enum CORINFO_EH_CLAUSE_FLAGS
      {
          CORINFO_EH_CLAUSE_NONE    = 0,
          CORINFO_EH_CLAUSE_FILTER  = 0x0001, // This clause is for a filter
          CORINFO_EH_CLAUSE_FINALLY = 0x0002, // This clause is a finally clause
          CORINFO_EH_CLAUSE_FAULT   = 0x0004, // This clause is a fault clause
      };
      typedef enum CorExceptionFlag
      {
          COR_ILEXCEPTION_CLAUSE_NONE,
          COR_ILEXCEPTION_CLAUSE_FILTER  = 0x0001, // This is a filter clause
          COR_ILEXCEPTION_CLAUSE_FINALLY = 0x0002, // This is a finally clause
          COR_ILEXCEPTION_CLAUSE_FAULT = 0x0004,   // This is a fault clause
          COR_ILEXCEPTION_CLAUSE_DUPLICATED = 0x0008, // duplicated clause. This
                                                      // clause was duplicated
                                                      // to a funclet which was
                                                      // pulled out of line
      } CorExceptionFlag;
    */
    // Add 1 to the start/end of the EH clause; the IP associated with a
    // call when the runtime does its scan is the IP of the next instruction
    // (the one to which control will return after the call), so we need
    // to add 1 to the end of the clause to cover that offset.  We also add
    // 1 to the start of the clause to make sure that the ranges reported
    // for all clauses are disjoint.  Note that we'll need some additional
    // logic when machine traps are supported, since in that case the IP
    // that the runtime uses is the offset of the faulting instruction
    // itself; if such an instruction immediately follows a call but the
    // two belong to different clauses, we'll need to insert a nop between
    // them so the runtime can distinguish the point to which the call will
    // return from the point at which the fault occurs.

    const MCExpr *ClauseBegin =
        getOffsetPlusOne(Clause.StartLabel, FuncBeginSym);
    const MCExpr *ClauseEnd = getOffsetPlusOne(Clause.EndLabel, FuncBeginSym);

    ClrEHUnwindMapEntry &Entry = FuncInfo.ClrEHUnwindMap[Clause.State];
    MachineBasicBlock *HandlerBlock = Entry.Handler.get<MachineBasicBlock *>();
    MCSymbol *BeginSym = getMCSymbolForMBB(Asm, HandlerBlock);
    const MCExpr *HandlerBegin = getOffset(BeginSym, FuncBeginSym);
    MCSymbol *EndSym = EndSymbolMap[Clause.State];
    const MCExpr *HandlerEnd = getOffset(EndSym, FuncBeginSym);

    uint32_t Flags = 0;
    switch (Entry.HandlerType) {
    case ClrHandlerType::Catch:
      // Leaving bits 0-2 clear indicates catch.
      break;
    case ClrHandlerType::Filter:
      Flags |= 1;
      break;
    case ClrHandlerType::Finally:
      Flags |= 2;
      break;
    case ClrHandlerType::Fault:
      Flags |= 4;
      break;
    }
    if (Clause.EnclosingState != MinClauseMap[Clause.State]) {
      // This is a "duplicate" clause; the handler needs to be entered from a
      // frame above the one holding the invoke.
      assert(Clause.EnclosingState > MinClauseMap[Clause.State]);
      Flags |= 8;
    }
    OS.EmitIntValue(Flags, 4);

    // Write the clause start/end
    OS.EmitValue(ClauseBegin, 4);
    OS.EmitValue(ClauseEnd, 4);

    // Write out the handler start/end
    OS.EmitValue(HandlerBegin, 4);
    OS.EmitValue(HandlerEnd, 4);

    // Write out the type token or filter offset
    assert(Entry.HandlerType != ClrHandlerType::Filter && "NYI: filters");
    OS.EmitIntValue(Entry.TypeToken, 4);
  }
}
