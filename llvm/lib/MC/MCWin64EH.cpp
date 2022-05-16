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
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Win64EH.h"
namespace llvm {
class MCSection;
}

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
  EmitSymbolRefWithOfs(streamer, info->Begin, info->Begin);
  EmitSymbolRefWithOfs(streamer, info->Begin, info->End);
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

void llvm::Win64EH::UnwindEmitter::EmitUnwindInfo(MCStreamer &Streamer,
                                                  WinEH::FrameInfo *info,
                                                  bool HandlerData) const {
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

static uint32_t ARM64CountOfUnwindCodes(ArrayRef<WinEH::Instruction> Insns) {
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
    case Win64EH::UOP_SaveR19R20X:
      Count += 1;
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
    case Win64EH::UOP_SaveLRPair:
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
    case Win64EH::UOP_SaveNext:
      Count += 1;
      break;
    case Win64EH::UOP_TrapFrame:
      Count += 1;
      break;
    case Win64EH::UOP_PushMachFrame:
      Count += 1;
      break;
    case Win64EH::UOP_Context:
      Count += 1;
      break;
    case Win64EH::UOP_ClearUnwoundToCall:
      Count += 1;
      break;
    }
  }
  return Count;
}

// Unwind opcode encodings and restrictions are documented at
// https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling
static void ARM64EmitUnwindCode(MCStreamer &streamer, const MCSymbol *begin,
                                const WinEH::Instruction &inst) {
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
  case Win64EH::UOP_SaveR19R20X:
    b = 0x20;
    b |= (inst.Offset >> 3) & 0x1F;
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
  case Win64EH::UOP_SaveLRPair:
    assert(inst.Register >= 19 && "Saved reg must be >= 19");
    reg = inst.Register - 19;
    assert((reg % 2) == 0 && "Saved reg must be 19+2*X");
    reg /= 2;
    b = 0xD6 | ((reg & 0x7) >> 2);
    streamer.emitInt8(b);
    b = ((reg & 0x3) << 6) | (inst.Offset >> 3);
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
  case Win64EH::UOP_SaveNext:
    b = 0xE6;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_TrapFrame:
    b = 0xE8;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_PushMachFrame:
    b = 0xE9;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_Context:
    b = 0xEA;
    streamer.emitInt8(b);
    break;
  case Win64EH::UOP_ClearUnwoundToCall:
    b = 0xEC;
    streamer.emitInt8(b);
    break;
  }
}

// Returns the epilog symbol of an epilog with the exact same unwind code
// sequence, if it exists.  Otherwise, returns nullptr.
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
      if (Instrs[i] != EpilogInstrs[i]) {
        Match = false;
        break;
      }

    if (Match)
      return EpilogStart;
  }
  return nullptr;
}

static void simplifyOpcodes(std::vector<WinEH::Instruction> &Instructions,
                            bool Reverse) {
  unsigned PrevOffset = -1;
  unsigned PrevRegister = -1;

  auto VisitInstruction = [&](WinEH::Instruction &Inst) {
    // Convert 2-byte opcodes into equivalent 1-byte ones.
    if (Inst.Operation == Win64EH::UOP_SaveRegP && Inst.Register == 29) {
      Inst.Operation = Win64EH::UOP_SaveFPLR;
      Inst.Register = -1;
    } else if (Inst.Operation == Win64EH::UOP_SaveRegPX &&
               Inst.Register == 29) {
      Inst.Operation = Win64EH::UOP_SaveFPLRX;
      Inst.Register = -1;
    } else if (Inst.Operation == Win64EH::UOP_SaveRegPX &&
               Inst.Register == 19 && Inst.Offset <= 248) {
      Inst.Operation = Win64EH::UOP_SaveR19R20X;
      Inst.Register = -1;
    } else if (Inst.Operation == Win64EH::UOP_AddFP && Inst.Offset == 0) {
      Inst.Operation = Win64EH::UOP_SetFP;
    } else if (Inst.Operation == Win64EH::UOP_SaveRegP &&
               Inst.Register == PrevRegister + 2 &&
               Inst.Offset == PrevOffset + 16) {
      Inst.Operation = Win64EH::UOP_SaveNext;
      Inst.Register = -1;
      Inst.Offset = 0;
      // Intentionally not creating UOP_SaveNext for float register pairs,
      // as current versions of Windows (up to at least 20.04) is buggy
      // regarding SaveNext for float pairs.
    }
    // Update info about the previous instruction, for detecting if
    // the next one can be made a UOP_SaveNext
    if (Inst.Operation == Win64EH::UOP_SaveR19R20X) {
      PrevOffset = 0;
      PrevRegister = 19;
    } else if (Inst.Operation == Win64EH::UOP_SaveRegPX) {
      PrevOffset = 0;
      PrevRegister = Inst.Register;
    } else if (Inst.Operation == Win64EH::UOP_SaveRegP) {
      PrevOffset = Inst.Offset;
      PrevRegister = Inst.Register;
    } else if (Inst.Operation == Win64EH::UOP_SaveNext) {
      PrevRegister += 2;
      PrevOffset += 16;
    } else {
      PrevRegister = -1;
      PrevOffset = -1;
    }
  };

  // Iterate over instructions in a forward order (for prologues),
  // backwards for epilogues (i.e. always reverse compared to how the
  // opcodes are stored).
  if (Reverse) {
    for (auto It = Instructions.rbegin(); It != Instructions.rend(); It++)
      VisitInstruction(*It);
  } else {
    for (WinEH::Instruction &Inst : Instructions)
      VisitInstruction(Inst);
  }
}

static int checkPackedEpilog(MCStreamer &streamer, WinEH::FrameInfo *info,
                             int PrologCodeBytes) {
  // Can only pack if there's one single epilog
  if (info->EpilogMap.size() != 1)
    return -1;

  const std::vector<WinEH::Instruction> &Epilog =
      info->EpilogMap.begin()->second;

  // Check that the epilog actually is at the very end of the function,
  // otherwise it can't be packed.
  uint32_t DistanceFromEnd = (uint32_t)GetAbsDifference(
      streamer, info->FuncletOrFuncEnd, info->EpilogMap.begin()->first);
  if (DistanceFromEnd / 4 != Epilog.size())
    return -1;

  int RetVal = -1;
  // Even if we don't end up sharing opcodes with the prolog, we can still
  // write the offset as a packed offset, if the single epilog is located at
  // the end of the function and the offset (pointing after the prolog) fits
  // as a packed offset.
  if (PrologCodeBytes <= 31 &&
      PrologCodeBytes + ARM64CountOfUnwindCodes(Epilog) <= 124)
    RetVal = PrologCodeBytes;

  // Can pack if the epilog is a subset of the prolog but not vice versa
  if (Epilog.size() > info->Instructions.size())
    return RetVal;

  // Check that the epilog actually is a perfect match for the end (backwrds)
  // of the prolog.
  for (int I = Epilog.size() - 1; I >= 0; I--) {
    if (info->Instructions[I] != Epilog[Epilog.size() - 1 - I])
      return RetVal;
  }

  int Offset = Epilog.size() == info->Instructions.size()
                   ? 0
                   : ARM64CountOfUnwindCodes(ArrayRef<WinEH::Instruction>(
                         &info->Instructions[Epilog.size()],
                         info->Instructions.size() - Epilog.size()));

  // Check that the offset and prolog size fits in the first word; it's
  // unclear whether the epilog count in the extension word can be taken
  // as packed epilog offset.
  if (Offset > 31 || PrologCodeBytes > 124)
    return RetVal;

  // As we choose to express the epilog as part of the prolog, remove the
  // epilog from the map, so we don't try to emit its opcodes.
  info->EpilogMap.clear();
  return Offset;
}

static bool tryPackedUnwind(WinEH::FrameInfo *info, uint32_t FuncLength,
                            int PackedEpilogOffset) {
  if (PackedEpilogOffset == 0) {
    // Fully symmetric prolog and epilog, should be ok for packed format.
    // For CR=3, the corresponding synthesized epilog actually lacks the
    // SetFP opcode, but unwinding should work just fine despite that
    // (if at the SetFP opcode, the unwinder considers it as part of the
    // function body and just unwinds the full prolog instead).
  } else if (PackedEpilogOffset == 1) {
    // One single case of differences between prolog and epilog is allowed:
    // The epilog can lack a single SetFP that is the last opcode in the
    // prolog, for the CR=3 case.
    if (info->Instructions.back().Operation != Win64EH::UOP_SetFP)
      return false;
  } else {
    // Too much difference between prolog and epilog.
    return false;
  }
  unsigned RegI = 0, RegF = 0;
  int Predecrement = 0;
  enum {
    Start,
    Start2,
    IntRegs,
    FloatRegs,
    InputArgs,
    StackAdjust,
    FrameRecord,
    End
  } Location = Start;
  bool StandaloneLR = false, FPLRPair = false;
  int StackOffset = 0;
  int Nops = 0;
  // Iterate over the prolog and check that all opcodes exactly match
  // the canonical order and form. A more lax check could verify that
  // all saved registers are in the expected locations, but not enforce
  // the order - that would work fine when unwinding from within
  // functions, but not be exactly right if unwinding happens within
  // prologs/epilogs.
  for (const WinEH::Instruction &Inst : info->Instructions) {
    switch (Inst.Operation) {
    case Win64EH::UOP_End:
      if (Location != Start)
        return false;
      Location = Start2;
      break;
    case Win64EH::UOP_SaveR19R20X:
      if (Location != Start2)
        return false;
      Predecrement = Inst.Offset;
      RegI = 2;
      Location = IntRegs;
      break;
    case Win64EH::UOP_SaveRegX:
      if (Location != Start2)
        return false;
      Predecrement = Inst.Offset;
      if (Inst.Register == 19)
        RegI += 1;
      else if (Inst.Register == 30)
        StandaloneLR = true;
      else
        return false;
      // Odd register; can't be any further int registers.
      Location = FloatRegs;
      break;
    case Win64EH::UOP_SaveRegPX:
      // Can't have this in a canonical prologue. Either this has been
      // canonicalized into SaveR19R20X or SaveFPLRX, or it's an unsupported
      // register pair.
      // It can't be canonicalized into SaveR19R20X if the offset is
      // larger than 248 bytes, but even with the maximum case with
      // RegI=10/RegF=8/CR=1/H=1, we end up with SavSZ = 216, which should
      // fit into SaveR19R20X.
      // The unwinding opcodes can't describe the otherwise seemingly valid
      // case for RegI=1 CR=1, that would start with a
      // "stp x19, lr, [sp, #-...]!" as that fits neither SaveRegPX nor
      // SaveLRPair.
      return false;
    case Win64EH::UOP_SaveRegP:
      if (Location != IntRegs || Inst.Offset != 8 * RegI ||
          Inst.Register != 19 + RegI)
        return false;
      RegI += 2;
      break;
    case Win64EH::UOP_SaveReg:
      if (Location != IntRegs || Inst.Offset != 8 * RegI)
        return false;
      if (Inst.Register == 19 + RegI)
        RegI += 1;
      else if (Inst.Register == 30)
        StandaloneLR = true;
      else
        return false;
      // Odd register; can't be any further int registers.
      Location = FloatRegs;
      break;
    case Win64EH::UOP_SaveLRPair:
      if (Location != IntRegs || Inst.Offset != 8 * RegI ||
          Inst.Register != 19 + RegI)
        return false;
      RegI += 1;
      StandaloneLR = true;
      Location = FloatRegs;
      break;
    case Win64EH::UOP_SaveFRegX:
      // Packed unwind can't handle prologs that only save one single
      // float register.
      return false;
    case Win64EH::UOP_SaveFReg:
      if (Location != FloatRegs || RegF == 0 || Inst.Register != 8 + RegF ||
          Inst.Offset != 8 * (RegI + (StandaloneLR ? 1 : 0) + RegF))
        return false;
      RegF += 1;
      Location = InputArgs;
      break;
    case Win64EH::UOP_SaveFRegPX:
      if (Location != Start2 || Inst.Register != 8)
        return false;
      Predecrement = Inst.Offset;
      RegF = 2;
      Location = FloatRegs;
      break;
    case Win64EH::UOP_SaveFRegP:
      if ((Location != IntRegs && Location != FloatRegs) ||
          Inst.Register != 8 + RegF ||
          Inst.Offset != 8 * (RegI + (StandaloneLR ? 1 : 0) + RegF))
        return false;
      RegF += 2;
      Location = FloatRegs;
      break;
    case Win64EH::UOP_SaveNext:
      if (Location == IntRegs)
        RegI += 2;
      else if (Location == FloatRegs)
        RegF += 2;
      else
        return false;
      break;
    case Win64EH::UOP_Nop:
      if (Location != IntRegs && Location != FloatRegs && Location != InputArgs)
        return false;
      Location = InputArgs;
      Nops++;
      break;
    case Win64EH::UOP_AllocSmall:
    case Win64EH::UOP_AllocMedium:
      if (Location != Start2 && Location != IntRegs && Location != FloatRegs &&
          Location != InputArgs && Location != StackAdjust)
        return false;
      // Can have either a single decrement, or a pair of decrements with
      // 4080 and another decrement.
      if (StackOffset == 0)
        StackOffset = Inst.Offset;
      else if (StackOffset != 4080)
        return false;
      else
        StackOffset += Inst.Offset;
      Location = StackAdjust;
      break;
    case Win64EH::UOP_SaveFPLRX:
      // Not allowing FPLRX after StackAdjust; if a StackAdjust is used, it
      // should be followed by a FPLR instead.
      if (Location != Start2 && Location != IntRegs && Location != FloatRegs &&
          Location != InputArgs)
        return false;
      StackOffset = Inst.Offset;
      Location = FrameRecord;
      FPLRPair = true;
      break;
    case Win64EH::UOP_SaveFPLR:
      // This can only follow after a StackAdjust
      if (Location != StackAdjust || Inst.Offset != 0)
        return false;
      Location = FrameRecord;
      FPLRPair = true;
      break;
    case Win64EH::UOP_SetFP:
      if (Location != FrameRecord)
        return false;
      Location = End;
      break;
    }
  }
  if (RegI > 10 || RegF > 8)
    return false;
  if (StandaloneLR && FPLRPair)
    return false;
  if (FPLRPair && Location != End)
    return false;
  if (Nops != 0 && Nops != 4)
    return false;
  int H = Nops == 4;
  int IntSZ = 8 * RegI;
  if (StandaloneLR)
    IntSZ += 8;
  int FpSZ = 8 * RegF; // RegF not yet decremented
  int SavSZ = (IntSZ + FpSZ + 8 * 8 * H + 0xF) & ~0xF;
  if (Predecrement != SavSZ)
    return false;
  if (FPLRPair && StackOffset < 16)
    return false;
  if (StackOffset % 16)
    return false;
  uint32_t FrameSize = (StackOffset + SavSZ) / 16;
  if (FrameSize > 0x1FF)
    return false;
  assert(RegF != 1 && "One single float reg not allowed");
  if (RegF > 0)
    RegF--; // Convert from actual number of registers, to value stored
  assert(FuncLength <= 0x7FF && "FuncLength should have been checked earlier");
  int Flag = 0x01; // Function segments not supported yet
  int CR = FPLRPair ? 3 : StandaloneLR ? 1 : 0;
  info->PackedInfo |= Flag << 0;
  info->PackedInfo |= (FuncLength & 0x7FF) << 2;
  info->PackedInfo |= (RegF & 0x7) << 13;
  info->PackedInfo |= (RegI & 0xF) << 16;
  info->PackedInfo |= (H & 0x1) << 20;
  info->PackedInfo |= (CR & 0x3) << 21;
  info->PackedInfo |= (FrameSize & 0x1FF) << 23;
  return true;
}

// Populate the .xdata section.  The format of .xdata on ARM64 is documented at
// https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling
static void ARM64EmitUnwindInfo(MCStreamer &streamer, WinEH::FrameInfo *info,
                                bool TryPacked = true) {
  // If this UNWIND_INFO already has a symbol, it's already been emitted.
  if (info->Symbol)
    return;
  // If there's no unwind info here (not even a terminating UOP_End), the
  // unwind info is considered bogus and skipped. If this was done in
  // response to an explicit .seh_handlerdata, the associated trailing
  // handler data is left orphaned in the xdata section.
  if (info->empty()) {
    info->EmitAttempted = true;
    return;
  }
  if (info->EmitAttempted) {
    // If we tried to emit unwind info before (due to an explicit
    // .seh_handlerdata directive), but skipped it (because there was no
    // valid information to emit at the time), and it later got valid unwind
    // opcodes, we can't emit it here, because the trailing handler data
    // was already emitted elsewhere in the xdata section.
    streamer.getContext().reportError(
        SMLoc(), "Earlier .seh_handlerdata for " + info->Function->getName() +
                     " skipped due to no unwind info at the time "
                     "(.seh_handlerdata too early?), but the function later "
                     "did get unwind info that can't be emitted");
    return;
  }

  simplifyOpcodes(info->Instructions, false);
  for (auto &I : info->EpilogMap)
    simplifyOpcodes(I.second, true);

  MCContext &context = streamer.getContext();
  MCSymbol *Label = context.createTempSymbol();

  streamer.emitValueToAlignment(4);
  streamer.emitLabel(Label);
  info->Symbol = Label;

  int64_t RawFuncLength;
  if (!info->FuncletOrFuncEnd) {
    report_fatal_error("FuncletOrFuncEnd not set");
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

  int PackedEpilogOffset = checkPackedEpilog(streamer, info, PrologCodeBytes);

  if (PackedEpilogOffset >= 0 &&
      uint32_t(PackedEpilogOffset) < PrologCodeBytes &&
      !info->HandlesExceptions && FuncLength <= 0x7ff && TryPacked) {
    // Matching prolog/epilog and no exception handlers; check if the
    // prolog matches the patterns that can be described by the packed
    // format.

    // info->Symbol was already set even if we didn't actually write any
    // unwind info there. Keep using that as indicator that this unwind
    // info has been generated already.

    if (tryPackedUnwind(info, FuncLength, PackedEpilogOffset))
      return;
  }

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
  uint32_t EpilogCount =
      PackedEpilogOffset >= 0 ? PackedEpilogOffset : info->EpilogMap.size();
  bool ExtensionWord = EpilogCount > 31 || TotalCodeBytes > 124;
  if (!ExtensionWord) {
    row1 |= (EpilogCount & 0x1F) << 22;
    row1 |= (CodeWords & 0x1F) << 27;
  }
  if (info->HandlesExceptions) // X
    row1 |= 1 << 20;
  if (PackedEpilogOffset >= 0) // E
    row1 |= 1 << 21;
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

  if (PackedEpilogOffset < 0) {
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
    for (const WinEH::Instruction &inst : EpilogInstrs)
      ARM64EmitUnwindCode(streamer, info->Begin, inst);
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
  EmitSymbolRefWithOfs(streamer, info->Begin, info->Begin);
  if (info->PackedInfo)
    streamer.emitInt32(info->PackedInfo);
  else
    streamer.emitValue(
        MCSymbolRefExpr::create(info->Symbol, MCSymbolRefExpr::VK_COFF_IMGREL32,
                                context),
        4);
}

void llvm::Win64EH::ARM64UnwindEmitter::Emit(MCStreamer &Streamer) const {
  // Emit the unwind info structs first.
  for (const auto &CFI : Streamer.getWinFrameInfos()) {
    WinEH::FrameInfo *Info = CFI.get();
    if (Info->empty())
      continue;
    MCSection *XData = Streamer.getAssociatedXDataSection(CFI->TextSection);
    Streamer.SwitchSection(XData);
    ARM64EmitUnwindInfo(Streamer, Info);
  }

  // Now emit RUNTIME_FUNCTION entries.
  for (const auto &CFI : Streamer.getWinFrameInfos()) {
    WinEH::FrameInfo *Info = CFI.get();
    // ARM64EmitUnwindInfo above clears the info struct, so we can't check
    // empty here. But if a Symbol is set, we should create the corresponding
    // pdata entry.
    if (!Info->Symbol)
      continue;
    MCSection *PData = Streamer.getAssociatedPDataSection(CFI->TextSection);
    Streamer.SwitchSection(PData);
    ARM64EmitRuntimeFunction(Streamer, Info);
  }
}

void llvm::Win64EH::ARM64UnwindEmitter::EmitUnwindInfo(MCStreamer &Streamer,
                                                       WinEH::FrameInfo *info,
                                                       bool HandlerData) const {
  // Called if there's an .seh_handlerdata directive before the end of the
  // function. This forces writing the xdata record already here - and
  // in this case, the function isn't actually ended already, but the xdata
  // record needs to know the function length. In these cases, if the funclet
  // end hasn't been marked yet, the xdata function length won't cover the
  // whole function, only up to this point.
  if (!info->FuncletOrFuncEnd) {
    Streamer.SwitchSection(info->TextSection);
    info->FuncletOrFuncEnd = Streamer.emitCFILabel();
  }
  // Switch sections (the static function above is meant to be called from
  // here and from Emit().
  MCSection *XData = Streamer.getAssociatedXDataSection(info->TextSection);
  Streamer.SwitchSection(XData);
  ARM64EmitUnwindInfo(Streamer, info, /* TryPacked = */ !HandlerData);
}
