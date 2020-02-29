//===- lib/MC/MCWin64EH.cpp - MCWin64EH implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCWin64EH.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Win64EH.h"

using namespace llvm;

// NOTE: All relocations generated here are 4-byte image-relative.

static uint8_t CountOfUnwindCodes(std::vector<WinEH::Instruction> &Insns) {
  uint8_t Count = 0;
  for (const auto &I : Insns) {
    switch (static_cast<Win64EH::UnwindOpcodes>(I.Operation)) {
    default:
      llvm_unreachable("Unsupported unwind code");
    case Win64EH::UOP_PushNonVol:
    case Win64EH::UOP_AllocSmall:
    case Win64EH::UOP_SetFPReg:
    case Win64EH::UOP_PushMachFrame:
      Count += 1;
      break;
    case Win64EH::UOP_SaveNonVol:
    case Win64EH::UOP_SaveXMM128:
      Count += 2;
      break;
    case Win64EH::UOP_SaveNonVolBig:
    case Win64EH::UOP_SaveXMM128Big:
      Count += 3;
      break;
    case Win64EH::UOP_AllocLarge:
      Count += (I.Offset > 512 * 1024 - 8) ? 3 : 2;
      break;
    }
  }
  return Count;
}

static void EmitAbsDifference(MCStreamer &Streamer, const MCSymbol *LHS,
                              const MCSymbol *RHS) {
  MCContext &Context = Streamer.getContext();
  const MCExpr *Diff =
      MCBinaryExpr::createSub(MCSymbolRefExpr::create(LHS, Context),
                              MCSymbolRefExpr::create(RHS, Context), Context);
  Streamer.emitValue(Diff, 1);
}

static void EmitUnwindCode(MCStreamer &streamer, const MCSymbol *begin,
                           WinEH::Instruction &inst) {
  uint8_t b2;
  uint16_t w;
  b2 = (inst.Operation & 0x0F);
  switch (static_cast<Win64EH::UnwindOpcodes>(inst.Operation)) {
  default:
    llvm_unreachable("Unsupported unwind code");
  case Win64EH::UOP_PushNonVol:
    EmitAbsDifference(streamer, inst.Label, begin);
    b2 |= (inst.Register & 0x0F) << 4;
    streamer.emitInt8(b2);
    break;
  case Win64EH::UOP_AllocLarge:
    EmitAbsDifference(streamer, inst.Label, begin);
    if (inst.Offset > 512 * 1024 - 8) {
      b2 |= 0x10;
      streamer.emitInt8(b2);
      w = inst.Offset & 0xFFF8;
      streamer.emitInt16(w);
      w = inst.Offset >> 16;
    } else {
      streamer.emitInt8(b2);
      w = inst.Offset >> 3;
    }
    streamer.emitInt16(w);
    break;
  case Win64EH::UOP_AllocSmall:
    b2 |= (((inst.Offset - 8) >> 3) & 0x0F) << 4;
    EmitAbsDifference(streamer, inst.Label, begin);
    streamer.emitInt8(b2);
    break;
  case Win64EH::UOP_SetFPReg:
    EmitAbsDifference(streamer, inst.Label, begin);
    streamer.emitInt8(b2);
    break;
  case Win64EH::UOP_SaveNonVol:
  case Win64EH::UOP_SaveXMM128:
    b2 |= (inst.Register & 0x0F) << 4;
    EmitAbsDifference(streamer, inst.Label, begin);
    streamer.emitInt8(b2);
    w = inst.Offset >> 3;
    if (inst.Operation == Win64EH::UOP_SaveXMM128)
      w >>= 1;
    streamer.emitInt16(w);
    break;
  case Win64EH::UOP_SaveNonVolBig:
  case Win64EH::UOP_SaveXMM128Big:
    b2 |= (inst.Register & 0x0F) << 4;
    EmitAbsDifference(streamer, inst.Label, begin);
    streamer.emitInt8(b2);
    if (inst.Operation == Win64EH::UOP_SaveXMM128Big)
      w = inst.Offset & 0xFFF0;
    else
      w = inst.Offset & 0xFFF8;
    streamer.emitInt16(w);
    w = inst.Offset >> 16;
    streamer.emitInt16(w);
    break;
  case Win64EH::UOP_PushMachFrame:
    if (inst.Offset == 1)
      b2 |= 0x10;
    EmitAbsDifference(streamer, inst.Label, begin);
    streamer.emitInt8(b2);
    break;
  }
}

static void EmitSymbolRefWithOfs(MCStreamer &streamer,
                                 const MCSymbol *Base,
                                 const MCSymbol *Other) {
  MCContext &Context = streamer.getContext();
  const MCSymbolRefExpr *BaseRef = MCSymbolRefExpr::create(Base, Context);
  const MCSymbolRefExpr *OtherRef = MCSymbolRefExpr::create(Other, Context);
  const MCExpr *Ofs = MCBinaryExpr::createSub(OtherRef, BaseRef, Context);
  const MCSymbolRefExpr *BaseRefRel = MCSymbolRefExpr::create(Base,
                                              MCSymbolRefExpr::VK_COFF_IMGREL32,
                                              Context);
  streamer.emitValue(MCBinaryExpr::createAdd(BaseRefRel, Ofs, Context), 4);
}

static void EmitRuntimeFunction(MCStreamer &streamer,
                                const WinEH::FrameInfo *info) {
  MCContext &context = streamer.getContext();

  streamer.emitValueToAlignment(4);
  EmitSymbolRefWithOfs(streamer, info->Function, info->Begin);
  EmitSymbolRefWithOfs(streamer, info->Function, info->End);
  streamer.emitValue(MCSymbolRefExpr::create(info->Symbol,
                                             MCSymbolRefExpr::VK_COFF_IMGREL32,
                                             context), 4);
}

static void EmitUnwindInfo(MCStreamer &streamer, WinEH::FrameInfo *info) {
  // If this UNWIND_INFO already has a symbol, it's already been emitted.
  if (info->Symbol)
    return;

  MCContext &context = streamer.getContext();
  MCSymbol *Label = context.createTempSymbol();

  streamer.emitValueToAlignment(4);
  streamer.emitLabel(Label);
  info->Symbol = Label;

  // Upper 3 bits are the version number (currently 1).
  uint8_t flags = 0x01;
  if (info->ChainedParent)
    flags |= Win64EH::UNW_ChainInfo << 3;
  else {
    if (info->HandlesUnwind)
      flags |= Win64EH::UNW_TerminateHandler << 3;
    if (info->HandlesExceptions)
      flags |= Win64EH::UNW_ExceptionHandler << 3;
  }
  streamer.emitInt8(flags);

  if (info->PrologEnd)
    EmitAbsDifference(streamer, info->PrologEnd, info->Begin);
  else
    streamer.emitInt8(0);

  uint8_t numCodes = CountOfUnwindCodes(info->Instructions);
  streamer.emitInt8(numCodes);

  uint8_t frame = 0;
  if (info->LastFrameInst >= 0) {
    WinEH::Instruction &frameInst = info->Instructions[info->LastFrameInst];
    assert(frameInst.Operation == Win64EH::UOP_SetFPReg);
    frame = (frameInst.Register & 0x0F) | (frameInst.Offset & 0xF0);
  }
  streamer.emitInt8(frame);

  // Emit unwind instructions (in reverse order).
  uint8_t numInst = info->Instructions.size();
  for (uint8_t c = 0; c < numInst; ++c) {
    WinEH::Instruction inst = info->Instructions.back();
    info->Instructions.pop_back();
    EmitUnwindCode(streamer, info->Begin, inst);
  }

  // For alignment purposes, the instruction array will always have an even
  // number of entries, with the final entry potentially unused (in which case
  // the array will be one longer than indicated by the count of unwind codes
  // field).
  if (numCodes & 1) {
    streamer.emitInt16(0);
  }

  if (flags & (Win64EH::UNW_ChainInfo << 3))
    EmitRuntimeFunction(streamer, info->ChainedParent);
  else if (flags &
           ((Win64EH::UNW_TerminateHandler|Win64EH::UNW_ExceptionHandler) << 3))
    streamer.emitValue(MCSymbolRefExpr::create(info->ExceptionHandler,
                                              MCSymbolRefExpr::VK_COFF_IMGREL32,
                                              context), 4);
  else if (numCodes == 0) {
    // The minimum size of an UNWIND_INFO struct is 8 bytes. If we're not
    // a chained unwind info, if there is no handler, and if there are fewer
    // than 2 slots used in the unwind code array, we have to pad to 8 bytes.
    streamer.emitInt32(0);
  }
}

void llvm::Win64EH::UnwindEmitter::Emit(MCStreamer &Streamer) const {
  // Emit the unwind info structs first.
  for (const auto &CFI : Streamer.getWinFrameInfos()) {
    MCSection *XData = Streamer.getAssociatedXDataSection(CFI->TextSection);
    Streamer.SwitchSection(XData);
    ::EmitUnwindInfo(Streamer, CFI.get());
  }

  // Now emit RUNTIME_FUNCTION entries.
  for (const auto &CFI : Streamer.getWinFrameInfos()) {
    MCSection *PData = Streamer.getAssociatedPDataSection(CFI->TextSection);
    Streamer.SwitchSection(PData);
    EmitRuntimeFunction(Streamer, CFI.get());
  }
}

void llvm::Win64EH::UnwindEmitter::EmitUnwindInfo(
    MCStreamer &Streamer, WinEH::FrameInfo *info) const {
  // Switch sections (the static function above is meant to be called from
  // here and from Emit().
  MCSection *XData = Streamer.getAssociatedXDataSection(info->TextSection);
  Streamer.SwitchSection(XData);

  ::EmitUnwindInfo(Streamer, info);
}

static int64_t GetAbsDifference(MCStreamer &Streamer, const MCSymbol *LHS,
                                const MCSymbol *RHS) {
  MCContext &Context = Streamer.getContext();
  const MCExpr *Diff =
      MCBinaryExpr::createSub(MCSymbolRefExpr::create(LHS, Context),
                              MCSymbolRefExpr::create(RHS, Context), Context);
  MCObjectStreamer *OS = (MCObjectStreamer *)(&Streamer);
  // It should normally be possible to calculate the length of a function
  // at this point, but it might not be possible in the presence of certain
  // unusual constructs, like an inline asm with an alignment directive.
  int64_t value;
  if (!Diff->evaluateAsAbsolute(value, OS->getAssembler()))
    report_fatal_error("Failed to evaluate function length in SEH unwind info");
  return value;
}

static uint32_t
ARM64CountOfUnwindCodes(const std::vector<WinEH::Instruction> &Insns) {
  uint32_t Count = 0;
  for (const auto &I : Insns) {
    switch (static_cast<Win64EH::UnwindOpcodes>(I.Operation)) {
    default:
      llvm_unreachable("Unsupported ARM64 unwind code");
    case Win64EH::UOP_AllocSmall:
      Count += 1;
      break;
    case Win64EH::UOP_AllocMedium:
      Count += 2;
      break;
    case Win64EH::UOP_AllocLarge:
      Count += 4;
      break;
    case Win64EH::UOP_SaveFPLRX:
      Count += 1;
      break;
    case Win64EH::UOP_SaveFPLR:
      Count += 1;
      break;
    case Win64EH::UOP_SaveReg:
      Count += 2;
      break;
    case Win64EH::UOP_SaveRegP:
      Count += 2;
      break;
    case Win64EH::UOP_SaveRegPX:
      Count += 2;
      break;
    case Win64EH::UOP_SaveRegX:
      Count += 2;
      break;
    case Win64EH::UOP_SaveFReg:
      Count += 2;
      break;
    case Win64EH::UOP_SaveFRegP:
      Count += 2;
      break;
    case Win64EH::UOP_SaveFRegX:
      Count += 2;
      break;
    case Win64EH::UOP_SaveFRegPX:
      Count += 2;
      break;
    case Win64EH::UOP_SetFP:
      Count += 1;
      break;
    case Win64EH::UOP_AddFP:
      Count += 2;
      break;
    case Win64EH::UOP_Nop:
      Count += 1;
      break;
    case Win64EH::UOP_End:
      Count += 1;
      break;
    }
  }
  return Count;
}

// Unwind opcode encodings and restrictions are documented at
// https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling
static void ARM64EmitUnwindCode(MCStreamer &streamer, const MCSymbol *begin,
                                WinEH::Instruction &inst) {
  uint8_t b, reg;
  switch (static_cast<Win64EH::UnwindOpcodes>(inst.Operation)) {
  default:
    llvm_unreachable("Unsupported ARM64 unwind code");
  case Win64EH::UOP_AllocSmall:
    b = (inst.Offset >> 4) & 0x1F;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_AllocMedium: {
    uint16_t hw = (inst.Offset >> 4) & 0x7FF;
    b = 0xC0;
    b |= (hw >> 8);
    streamer.emitInt8(b);
    b = hw & 0xFF;
    streamer.emitInt8(b);
    break;
  }
  case Win64EH::UOP_AllocLarge: {
    uint32_t w;
    b = 0xE0;
    streamer.emitInt8(b);
    w = inst.Offset >> 4;
    b = (w & 0x00FF0000) >> 16;
    streamer.emitInt8(b);
    b = (w & 0x0000FF00) >> 8;
    streamer.emitInt8(b);
    b = w & 0x000000FF;
    streamer.emitInt8(b);
    break;
  }
  case Win64EH::UOP_SetFP:
    b = 0xE1;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_AddFP:
    b = 0xE2;
    streamer.emitInt8(b);
    b = (inst.Offset >> 3);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_Nop:
    b = 0xE3;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveFPLRX:
    b = 0x80;
    b |= ((inst.Offset - 1) >> 3) & 0x3F;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveFPLR:
    b = 0x40;
    b |= (inst.Offset >> 3) & 0x3F;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveReg:
    assert(inst.Register >= 19 && "Saved reg must be >= 19");
    reg = inst.Register - 19;
    b = 0xD0 | ((reg & 0xC) >> 2);
    streamer.emitInt8(b);
    b = ((reg & 0x3) << 6) | (inst.Offset >> 3);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveRegX:
    assert(inst.Register >= 19 && "Saved reg must be >= 19");
    reg = inst.Register - 19;
    b = 0xD4 | ((reg & 0x8) >> 3);
    streamer.emitInt8(b);
    b = ((reg & 0x7) << 5) | ((inst.Offset >> 3) - 1);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveRegP:
    assert(inst.Register >= 19 && "Saved registers must be >= 19");
    reg = inst.Register - 19;
    b = 0xC8 | ((reg & 0xC) >> 2);
    streamer.emitInt8(b);
    b = ((reg & 0x3) << 6) | (inst.Offset >> 3);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveRegPX:
    assert(inst.Register >= 19 && "Saved registers must be >= 19");
    reg = inst.Register - 19;
    b = 0xCC | ((reg & 0xC) >> 2);
    streamer.emitInt8(b);
    b = ((reg & 0x3) << 6) | ((inst.Offset >> 3) - 1);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveFReg:
    assert(inst.Register >= 8 && "Saved dreg must be >= 8");
    reg = inst.Register - 8;
    b = 0xDC | ((reg & 0x4) >> 2);
    streamer.emitInt8(b);
    b = ((reg & 0x3) << 6) | (inst.Offset >> 3);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveFRegX:
    assert(inst.Register >= 8 && "Saved dreg must be >= 8");
    reg = inst.Register - 8;
    b = 0xDE;
    streamer.emitInt8(b);
    b = ((reg & 0x7) << 5) | ((inst.Offset >> 3) - 1);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveFRegP:
    assert(inst.Register >= 8 && "Saved dregs must be >= 8");
    reg = inst.Register - 8;
    b = 0xD8 | ((reg & 0x4) >> 2);
    streamer.emitInt8(b);
    b = ((reg & 0x3) << 6) | (inst.Offset >> 3);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_SaveFRegPX:
    assert(inst.Register >= 8 && "Saved dregs must be >= 8");
    reg = inst.Register - 8;
    b = 0xDA | ((reg & 0x4) >> 2);
    streamer.emitInt8(b);
    b = ((reg & 0x3) << 6) | ((inst.Offset >> 3) - 1);
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_End:
    b = 0xE4;
    streamer.emitInt8(b);
    break;
  }
}

// Returns the epilog symbol of an epilog with the exact same unwind code
// sequence, if it exists.  Otherwise, returns nulltpr.
// EpilogInstrs - Unwind codes for the current epilog.
// Epilogs - Epilogs that potentialy match the current epilog.
static MCSymbol*
FindMatchingEpilog(const std::vector<WinEH::Instruction>& EpilogInstrs,
                   const std::vector<MCSymbol *>& Epilogs,
                   const WinEH::FrameInfo *info) {
  for (auto *EpilogStart : Epilogs) {
    auto InstrsIter = info->EpilogMap.find(EpilogStart);
    assert(InstrsIter != info->EpilogMap.end() &&
           "Epilog not found in EpilogMap");
    const auto &Instrs = InstrsIter->second;

    if (Instrs.size() != EpilogInstrs.size())
      continue;

    bool Match = true;
    for (unsigned i = 0; i < Instrs.size(); ++i)
      if (Instrs[i].Operation != EpilogInstrs[i].Operation ||
          Instrs[i].Offset != EpilogInstrs[i].Offset ||
          Instrs[i].Register != EpilogInstrs[i].Register) {
         Match = false;
         break;
      }

    if (Match)
      return EpilogStart;
  }
  return nullptr;
}

// Populate the .xdata section.  The format of .xdata on ARM64 is documented at
// https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling
static void ARM64EmitUnwindInfo(MCStreamer &streamer, WinEH::FrameInfo *info) {
  // If this UNWIND_INFO already has a symbol, it's already been emitted.
  if (info->Symbol)
    return;

  MCContext &context = streamer.getContext();
  MCSymbol *Label = context.createTempSymbol();

  streamer.emitValueToAlignment(4);
  streamer.emitLabel(Label);
  info->Symbol = Label;

  int64_t RawFuncLength;
  if (!info->FuncletOrFuncEnd) {
    // FIXME: This is very wrong; we emit SEH data which covers zero bytes
    // of code. But otherwise test/MC/AArch64/seh.s crashes.
    RawFuncLength = 0;
  } else {
    // FIXME: GetAbsDifference tries to compute the length of the function
    // immediately, before the whole file is emitted, but in general
    // that's impossible: the size in bytes of certain assembler directives
    // like .align and .fill is not known until the whole file is parsed and
    // relaxations are applied. Currently, GetAbsDifference fails with a fatal
    // error in that case. (We mostly don't hit this because inline assembly
    // specifying those directives is rare, and we don't normally try to
    // align loops on AArch64.)
    //
    // There are two potential approaches to delaying the computation. One,
    // we could emit something like ".word (endfunc-beginfunc)/4+0x10800000",
    // as long as we have some conservative estimate we could use to prove
    // that we don't need to split the unwind data. Emitting the constant
    // is straightforward, but there's no existing code for estimating the
    // size of the function.
    //
    // The other approach would be to use a dedicated, relaxable fragment,
    // which could grow to accommodate splitting the unwind data if
    // necessary. This is more straightforward, since it automatically works
    // without any new infrastructure, and it's consistent with how we handle
    // relaxation in other contexts.  But it would require some refactoring
    // to move parts of the pdata/xdata emission into the implementation of
    // a fragment. We could probably continue to encode the unwind codes
    // here, but we'd have to emit the pdata, the xdata header, and the
    // epilogue scopes later, since they depend on whether the we need to
    // split the unwind data.
    RawFuncLength = GetAbsDifference(streamer, info->FuncletOrFuncEnd,
                                     info->Begin);
  }
  if (RawFuncLength > 0xFFFFF)
    report_fatal_error("SEH unwind data splitting not yet implemented");
  uint32_t FuncLength = (uint32_t)RawFuncLength / 4;
  uint32_t PrologCodeBytes = ARM64CountOfUnwindCodes(info->Instructions);
  uint32_t TotalCodeBytes = PrologCodeBytes;

  // Process epilogs.
  MapVector<MCSymbol *, uint32_t> EpilogInfo;
  // Epilogs processed so far.
  std::vector<MCSymbol *> AddedEpilogs;

  for (auto &I : info->EpilogMap) {
    MCSymbol *EpilogStart = I.first;
    auto &EpilogInstrs = I.second;
    uint32_t CodeBytes = ARM64CountOfUnwindCodes(EpilogInstrs);

    MCSymbol* MatchingEpilog =
      FindMatchingEpilog(EpilogInstrs, AddedEpilogs, info);
    if (MatchingEpilog) {
      assert(EpilogInfo.find(MatchingEpilog) != EpilogInfo.end() &&
             "Duplicate epilog not found");
      EpilogInfo[EpilogStart] = EpilogInfo.lookup(MatchingEpilog);
      // Clear the unwind codes in the EpilogMap, so that they don't get output
      // in the logic below.
      EpilogInstrs.clear();
    } else {
      EpilogInfo[EpilogStart] = TotalCodeBytes;
      TotalCodeBytes += CodeBytes;
      AddedEpilogs.push_back(EpilogStart);
    }
  }

  // Code Words, Epilog count, E, X, Vers, Function Length
  uint32_t row1 = 0x0;
  uint32_t CodeWords = TotalCodeBytes / 4;
  uint32_t CodeWordsMod = TotalCodeBytes % 4;
  if (CodeWordsMod)
    CodeWords++;
  uint32_t EpilogCount = info->EpilogMap.size();
  bool ExtensionWord = EpilogCount > 31 || TotalCodeBytes > 124;
  if (!ExtensionWord) {
    row1 |= (EpilogCount & 0x1F) << 22;
    row1 |= (CodeWords & 0x1F) << 27;
  }
  // E is always 0 right now, TODO: packed epilog setup
  if (info->HandlesExceptions) // X
    row1 |= 1 << 20;
  row1 |= FuncLength & 0x3FFFF;
  streamer.emitInt32(row1);

  // Extended Code Words, Extended Epilog Count
  if (ExtensionWord) {
    // FIXME: We should be able to split unwind info into multiple sections.
    // FIXME: We should share epilog codes across epilogs, where possible,
    // which would make this issue show up less frequently.
    if (CodeWords > 0xFF || EpilogCount > 0xFFFF)
      report_fatal_error("SEH unwind data splitting not yet implemented");
    uint32_t row2 = 0x0;
    row2 |= (CodeWords & 0xFF) << 16;
    row2 |= (EpilogCount & 0xFFFF);
    streamer.emitInt32(row2);
  }

  // Epilog Start Index, Epilog Start Offset
  for (auto &I : EpilogInfo) {
    MCSymbol *EpilogStart = I.first;
    uint32_t EpilogIndex = I.second;
    uint32_t EpilogOffset =
        (uint32_t)GetAbsDifference(streamer, EpilogStart, info->Begin);
    if (EpilogOffset)
      EpilogOffset /= 4;
    uint32_t row3 = EpilogOffset;
    row3 |= (EpilogIndex & 0x3FF) << 22;
    streamer.emitInt32(row3);
  }

  // Emit prolog unwind instructions (in reverse order).
  uint8_t numInst = info->Instructions.size();
  for (uint8_t c = 0; c < numInst; ++c) {
    WinEH::Instruction inst = info->Instructions.back();
    info->Instructions.pop_back();
    ARM64EmitUnwindCode(streamer, info->Begin, inst);
  }

  // Emit epilog unwind instructions
  for (auto &I : info->EpilogMap) {
    auto &EpilogInstrs = I.second;
    for (uint32_t i = 0; i < EpilogInstrs.size(); i++) {
      WinEH::Instruction inst = EpilogInstrs[i];
      ARM64EmitUnwindCode(streamer, info->Begin, inst);
    }
  }

  int32_t BytesMod = CodeWords * 4 - TotalCodeBytes;
  assert(BytesMod >= 0);
  for (int i = 0; i < BytesMod; i++)
    streamer.emitInt8(0xE3);

  if (info->HandlesExceptions)
    streamer.emitValue(
        MCSymbolRefExpr::create(info->ExceptionHandler,
                                MCSymbolRefExpr::VK_COFF_IMGREL32, context),
        4);
}

static void ARM64EmitRuntimeFunction(MCStreamer &streamer,
                                     const WinEH::FrameInfo *info) {
  MCContext &context = streamer.getContext();

  streamer.emitValueToAlignment(4);
  EmitSymbolRefWithOfs(streamer, info->Function, info->Begin);
  streamer.emitValue(MCSymbolRefExpr::create(info->Symbol,
                                             MCSymbolRefExpr::VK_COFF_IMGREL32,
                                             context),
                     4);
}

void llvm::Win64EH::ARM64UnwindEmitter::Emit(MCStreamer &Streamer) const {
  // Emit the unwind info structs first.
  for (const auto &CFI : Streamer.getWinFrameInfos()) {
    MCSection *XData = Streamer.getAssociatedXDataSection(CFI->TextSection);
    Streamer.SwitchSection(XData);
    ARM64EmitUnwindInfo(Streamer, CFI.get());
  }

  // Now emit RUNTIME_FUNCTION entries.
  for (const auto &CFI : Streamer.getWinFrameInfos()) {
    MCSection *PData = Streamer.getAssociatedPDataSection(CFI->TextSection);
    Streamer.SwitchSection(PData);
    ARM64EmitRuntimeFunction(Streamer, CFI.get());
  }
}

void llvm::Win64EH::ARM64UnwindEmitter::EmitUnwindInfo(
    MCStreamer &Streamer, WinEH::FrameInfo *info) const {
  // Switch sections (the static function above is meant to be called from
  // here and from Emit().
  MCSection *XData = Streamer.getAssociatedXDataSection(info->TextSection);
  Streamer.SwitchSection(XData);
  ARM64EmitUnwindInfo(Streamer, info);
}
