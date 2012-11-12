//===- lib/MC/MCWin64EH.cpp - MCWin64EH implementation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCWin64EH.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/Twine.h"

namespace llvm {

// NOTE: All relocations generated here are 4-byte image-relative.

static uint8_t CountOfUnwindCodes(std::vector<MCWin64EHInstruction> &instArray){
  uint8_t count = 0;
  for (std::vector<MCWin64EHInstruction>::const_iterator I = instArray.begin(),
       E = instArray.end(); I != E; ++I) {
    switch (I->getOperation()) {
    case Win64EH::UOP_PushNonVol:
    case Win64EH::UOP_AllocSmall:
    case Win64EH::UOP_SetFPReg:
    case Win64EH::UOP_PushMachFrame:
      count += 1;
      break;
    case Win64EH::UOP_SaveNonVol:
    case Win64EH::UOP_SaveXMM128:
      count += 2;
      break;
    case Win64EH::UOP_SaveNonVolBig:
    case Win64EH::UOP_SaveXMM128Big:
      count += 3;
      break;
    case Win64EH::UOP_AllocLarge:
      if (I->getSize() > 512*1024-8)
        count += 3;
      else
        count += 2;
      break;
    }
  }
  return count;
}

static void EmitAbsDifference(MCStreamer &streamer, MCSymbol *lhs,
                              MCSymbol *rhs) {
  MCContext &context = streamer.getContext();
  const MCExpr *diff = MCBinaryExpr::CreateSub(MCSymbolRefExpr::Create(
                                                                  lhs, context),
                                               MCSymbolRefExpr::Create(
                                                                  rhs, context),
                                               context);
  streamer.EmitAbsValue(diff, 1);

}

static void EmitUnwindCode(MCStreamer &streamer, MCSymbol *begin,
                           MCWin64EHInstruction &inst) {
  uint8_t b1, b2;
  uint16_t w;
  b2 = (inst.getOperation() & 0x0F);
  switch (inst.getOperation()) {
  case Win64EH::UOP_PushNonVol:
    EmitAbsDifference(streamer, inst.getLabel(), begin);
    b2 |= (inst.getRegister() & 0x0F) << 4;
    streamer.EmitIntValue(b2, 1);
    break;
  case Win64EH::UOP_AllocLarge:
    EmitAbsDifference(streamer, inst.getLabel(), begin);
    if (inst.getSize() > 512*1024-8) {
      b2 |= 0x10;
      streamer.EmitIntValue(b2, 1);
      w = inst.getSize() & 0xFFF8;
      streamer.EmitIntValue(w, 2);
      w = inst.getSize() >> 16;
    } else {
      streamer.EmitIntValue(b2, 1);
      w = inst.getSize() >> 3;
    }
    streamer.EmitIntValue(w, 2);
    break;
  case Win64EH::UOP_AllocSmall:
    b2 |= (((inst.getSize()-8) >> 3) & 0x0F) << 4;
    EmitAbsDifference(streamer, inst.getLabel(), begin);
    streamer.EmitIntValue(b2, 1);
    break;
  case Win64EH::UOP_SetFPReg:
    b1 = inst.getOffset() & 0xF0;
    streamer.EmitIntValue(b1, 1);
    streamer.EmitIntValue(b2, 1);
    break;
  case Win64EH::UOP_SaveNonVol:
  case Win64EH::UOP_SaveXMM128:
    b2 |= (inst.getRegister() & 0x0F) << 4;
    EmitAbsDifference(streamer, inst.getLabel(), begin);
    streamer.EmitIntValue(b2, 1);
    w = inst.getOffset() >> 3;
    if (inst.getOperation() == Win64EH::UOP_SaveXMM128)
      w >>= 1;
    streamer.EmitIntValue(w, 2);
    break;
  case Win64EH::UOP_SaveNonVolBig:
  case Win64EH::UOP_SaveXMM128Big:
    b2 |= (inst.getRegister() & 0x0F) << 4;
    EmitAbsDifference(streamer, inst.getLabel(), begin);
    streamer.EmitIntValue(b2, 1);
    if (inst.getOperation() == Win64EH::UOP_SaveXMM128Big)
      w = inst.getOffset() & 0xFFF0;
    else
      w = inst.getOffset() & 0xFFF8;
    streamer.EmitIntValue(w, 2);
    w = inst.getOffset() >> 16;
    streamer.EmitIntValue(w, 2);
    break;
  case Win64EH::UOP_PushMachFrame:
    if (inst.isPushCodeFrame())
      b2 |= 0x10;
    EmitAbsDifference(streamer, inst.getLabel(), begin);
    streamer.EmitIntValue(b2, 1);
    break;
  }
}

static void EmitRuntimeFunction(MCStreamer &streamer,
                                const MCWin64EHUnwindInfo *info) {
  MCContext &context = streamer.getContext();

  streamer.EmitValueToAlignment(4);
  streamer.EmitValue(MCSymbolRefExpr::Create(info->Begin, context), 4);
  streamer.EmitValue(MCSymbolRefExpr::Create(info->End, context), 4);
  streamer.EmitValue(MCSymbolRefExpr::Create(info->Symbol, context), 4);
}

static void EmitUnwindInfo(MCStreamer &streamer, MCWin64EHUnwindInfo *info) {
  // If this UNWIND_INFO already has a symbol, it's already been emitted.
  if (info->Symbol) return;

  MCContext &context = streamer.getContext();
  streamer.EmitValueToAlignment(4);
  // Upper 3 bits are the version number (currently 1).
  uint8_t flags = 0x01;
  info->Symbol = context.CreateTempSymbol();
  streamer.EmitLabel(info->Symbol);

  if (info->ChainedParent)
    flags |= Win64EH::UNW_ChainInfo << 3;
  else {
    if (info->HandlesUnwind)
      flags |= Win64EH::UNW_TerminateHandler << 3;
    if (info->HandlesExceptions)
      flags |= Win64EH::UNW_ExceptionHandler << 3;
  }
  streamer.EmitIntValue(flags, 1);

  if (info->PrologEnd)
    EmitAbsDifference(streamer, info->PrologEnd, info->Begin);
  else
    streamer.EmitIntValue(0, 1);

  uint8_t numCodes = CountOfUnwindCodes(info->Instructions);
  streamer.EmitIntValue(numCodes, 1);

  uint8_t frame = 0;
  if (info->LastFrameInst >= 0) {
    MCWin64EHInstruction &frameInst = info->Instructions[info->LastFrameInst];
    assert(frameInst.getOperation() == Win64EH::UOP_SetFPReg);
    frame = (frameInst.getRegister() & 0x0F) |
            (frameInst.getOffset() & 0xF0);
  }
  streamer.EmitIntValue(frame, 1);

  // Emit unwind instructions (in reverse order).
  uint8_t numInst = info->Instructions.size();
  for (uint8_t c = 0; c < numInst; ++c) {
    MCWin64EHInstruction inst = info->Instructions.back();
    info->Instructions.pop_back();
    EmitUnwindCode(streamer, info->Begin, inst);
  }

  if (flags & (Win64EH::UNW_ChainInfo << 3))
    EmitRuntimeFunction(streamer, info->ChainedParent);
  else if (flags &
           ((Win64EH::UNW_TerminateHandler|Win64EH::UNW_ExceptionHandler) << 3))
    streamer.EmitValue(MCSymbolRefExpr::Create(info->ExceptionHandler, context),
                       4);
  else if (numCodes < 2) {
    // The minimum size of an UNWIND_INFO struct is 8 bytes. If we're not
    // a chained unwind info, if there is no handler, and if there are fewer
    // than 2 slots used in the unwind code array, we have to pad to 8 bytes.
    if (numCodes == 1)
      streamer.EmitIntValue(0, 2);
    else
      streamer.EmitIntValue(0, 4);
  }
}

StringRef MCWin64EHUnwindEmitter::GetSectionSuffix(const MCSymbol *func) {
  if (!func || !func->isInSection()) return "";
  const MCSection *section = &func->getSection();
  const MCSectionCOFF *COFFSection;
  if ((COFFSection = dyn_cast<MCSectionCOFF>(section))) {
    StringRef name = COFFSection->getSectionName();
    size_t dollar = name.find('$');
    size_t dot = name.find('.', 1);
    if (dollar == StringRef::npos && dot == StringRef::npos)
      return "";
    if (dot == StringRef::npos)
      return name.substr(dollar);
    if (dollar == StringRef::npos || dot < dollar)
      return name.substr(dot);
    return name.substr(dollar);
  }
  return "";
}

static const MCSection *getWin64EHTableSection(StringRef suffix,
                                               MCContext &context) {
  if (suffix == "")
    return context.getObjectFileInfo()->getXDataSection();

  return context.getCOFFSection((".xdata"+suffix).str(),
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getDataRel());
}

static const MCSection *getWin64EHFuncTableSection(StringRef suffix,
                                                   MCContext &context) {
  if (suffix == "")
    return context.getObjectFileInfo()->getPDataSection();
  return context.getCOFFSection((".pdata"+suffix).str(),
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getDataRel());
}

void MCWin64EHUnwindEmitter::EmitUnwindInfo(MCStreamer &streamer,
                                            MCWin64EHUnwindInfo *info) {
  // Switch sections (the static function above is meant to be called from
  // here and from Emit().
  MCContext &context = streamer.getContext();
  const MCSection *xdataSect =
    getWin64EHTableSection(GetSectionSuffix(info->Function), context);
  streamer.SwitchSection(xdataSect);

  llvm::EmitUnwindInfo(streamer, info);
}

void MCWin64EHUnwindEmitter::Emit(MCStreamer &streamer) {
  MCContext &context = streamer.getContext();
  // Emit the unwind info structs first.
  for (unsigned i = 0; i < streamer.getNumW64UnwindInfos(); ++i) {
    MCWin64EHUnwindInfo &info = streamer.getW64UnwindInfo(i);
    const MCSection *xdataSect =
      getWin64EHTableSection(GetSectionSuffix(info.Function), context);
    streamer.SwitchSection(xdataSect);
    llvm::EmitUnwindInfo(streamer, &info);
  }
  // Now emit RUNTIME_FUNCTION entries.
  for (unsigned i = 0; i < streamer.getNumW64UnwindInfos(); ++i) {
    MCWin64EHUnwindInfo &info = streamer.getW64UnwindInfo(i);
    const MCSection *pdataSect =
      getWin64EHFuncTableSection(GetSectionSuffix(info.Function), context);
    streamer.SwitchSection(pdataSect);
    EmitRuntimeFunction(streamer, &info);
  }
}

} // End of namespace llvm

