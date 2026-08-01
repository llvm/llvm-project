//==-- SystemZTargetStreamer.cpp - SystemZ Target Streamer Methods ----------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines SystemZ-specific target streamer classes.
/// These are for implementing support for target-specific assembly directives.
///
//===----------------------------------------------------------------------===//

#include "SystemZTargetStreamer.h"
#include "SystemZHLASMAsmStreamer.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCGOFFStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/Support/ConvertEBCDIC.h"

using namespace llvm;

void SystemZTargetStreamer::emitConstantPools() {
  // Emit EXRL target instructions.
  if (EXRLTargets2Sym.empty())
    return;
  // Switch to the .text section.
  const MCObjectFileInfo &OFI = *Streamer.getContext().getObjectFileInfo();
  Streamer.switchSection(OFI.getTextSection());
  for (auto &I : EXRLTargets2Sym) {
    Streamer.emitLabel(I.second);
    const MCInstSTIPair &MCI_STI = I.first;
    Streamer.emitInstruction(MCI_STI.first, *MCI_STI.second);
  }
  EXRLTargets2Sym.clear();
}

static void emitPPA1Flags(MCStreamer &OutStreamer, bool VarArg,
                          bool StackProtector, bool FPRMask, bool VRMask,
                          bool EHBlock, bool HasArgAreaLength, bool HasName) {
  enum class PPA1Flag1 : uint8_t {
    DSA64Bit = (0x80 >> 0),
    VarArg = (0x80 >> 7),
    LLVM_MARK_AS_BITMASK_ENUM(DSA64Bit)
  };
  enum class PPA1Flag2 : uint8_t {
    ExternalProcedure = (0x80 >> 0),
    STACKPROTECTOR = (0x80 >> 3),
    LLVM_MARK_AS_BITMASK_ENUM(ExternalProcedure)
  };
  enum class PPA1Flag3 : uint8_t {
    HasArgAreaLength = (0x80 >> 1),
    FPRMask = (0x80 >> 2),
    LLVM_MARK_AS_BITMASK_ENUM(HasArgAreaLength)
  };
  enum class PPA1Flag4 : uint8_t {
    EPMOffsetPresent = (0x80 >> 0),
    VRMask = (0x80 >> 2),
    EHBlock = (0x80 >> 3),
    ProcedureNamePresent = (0x80 >> 7),
    LLVM_MARK_AS_BITMASK_ENUM(EPMOffsetPresent)
  };

  // Declare optional section flags that can be modified.
  auto Flags1 = PPA1Flag1(0);
  auto Flags2 = PPA1Flag2::ExternalProcedure;
  auto Flags3 = PPA1Flag3(0);
  auto Flags4 = PPA1Flag4::EPMOffsetPresent;

  Flags1 |= PPA1Flag1::DSA64Bit;

  if (VarArg)
    Flags1 |= PPA1Flag1::VarArg;

  if (StackProtector)
    Flags2 |= PPA1Flag2::STACKPROTECTOR;

  if (HasArgAreaLength)
    Flags3 |= PPA1Flag3::HasArgAreaLength; // Add emit ArgAreaLength flag.

  // SavedGPRMask, SavedFPRMask, and SavedVRMask are precomputed in.
  if (FPRMask)
    Flags3 |= PPA1Flag3::FPRMask; // Add emit FPR mask flag.

  if (VRMask)
    Flags4 |= PPA1Flag4::VRMask; // Add emit VR mask flag.

  if (EHBlock)
    Flags4 |= PPA1Flag4::EHBlock; // Add optional EH block.

  if (HasName)
    Flags4 |= PPA1Flag4::ProcedureNamePresent; // Add optional name block.

  OutStreamer.AddComment("PPA1 Flags 1");
  OutStreamer.AddComment("  Bit 0: 1 = 64-bit DSA");
  if ((Flags1 & PPA1Flag1::VarArg) == PPA1Flag1::VarArg)
    OutStreamer.AddComment("  Bit 7: 1 = Vararg function");
  OutStreamer.emitInt8(static_cast<uint8_t>(Flags1)); // Flags 1.

  OutStreamer.AddComment("PPA1 Flags 2");
  if ((Flags2 & PPA1Flag2::ExternalProcedure) == PPA1Flag2::ExternalProcedure)
    OutStreamer.AddComment("  Bit 0: 1 = External procedure");
  if ((Flags2 & PPA1Flag2::STACKPROTECTOR) == PPA1Flag2::STACKPROTECTOR)
    OutStreamer.AddComment("  Bit 3: 1 = STACKPROTECT is enabled");
  else
    OutStreamer.AddComment("  Bit 3: 0 = STACKPROTECT is not enabled");
  OutStreamer.emitInt8(static_cast<uint8_t>(Flags2)); // Flags 2.

  OutStreamer.AddComment("PPA1 Flags 3");
  if ((Flags3 & PPA1Flag3::HasArgAreaLength) == PPA1Flag3::HasArgAreaLength)
    OutStreamer.AddComment(
        "  Bit 1: 1 = Argument Area Length is in optional area");
  if ((Flags3 & PPA1Flag3::FPRMask) == PPA1Flag3::FPRMask)
    OutStreamer.AddComment("  Bit 2: 1 = FP Reg Mask is in optional area");
  OutStreamer.emitInt8(
      static_cast<uint8_t>(Flags3)); // Flags 3 (optional sections).

  OutStreamer.AddComment("PPA1 Flags 4");
  if ((Flags4 & PPA1Flag4::VRMask) == PPA1Flag4::VRMask)
    OutStreamer.AddComment("  Bit 2: 1 = Vector Reg Mask is in optional area");
  if ((Flags4 & PPA1Flag4::EHBlock) == PPA1Flag4::EHBlock)
    OutStreamer.AddComment("  Bit 3: 1 = C++ EH block");
  if ((Flags4 & PPA1Flag4::ProcedureNamePresent) ==
      PPA1Flag4::ProcedureNamePresent)
    OutStreamer.AddComment("  Bit 7: 1 = Name Length and Name");
  OutStreamer.emitInt8(static_cast<uint8_t>(
      Flags4)); // Flags 4 (optional sections, always emit these).
}

static void emitPPA1Name(MCStreamer &OutStreamer, StringRef OutName) {
  size_t NameSize = OutName.size();
  uint16_t OutSize;
  if (NameSize < UINT16_MAX) {
    OutSize = static_cast<uint16_t>(NameSize);
  } else {
    OutName = OutName.substr(0, UINT16_MAX);
    OutSize = UINT16_MAX;
  }
  // Emit padding to ensure that the next optional field word-aligned.
  uint8_t ExtraZeros = 4 - ((2 + OutSize) % 4);

  SmallString<512> OutnameConv;
  ConverterEBCDIC::convertToEBCDIC(OutName, OutnameConv);
  OutName = OutnameConv.str();

  OutStreamer.AddComment("Length of Name");
  OutStreamer.emitInt16(OutSize);
  OutStreamer.AddComment("Name of Function");
  OutStreamer.emitBytes(OutName);
  OutStreamer.emitZeros(ExtraZeros);
}

void SystemZTargetzOSStreamer::emitPPA1(PPA1Info &Info) {
  assert(PPA2Sym != nullptr && "PPA2 Symbol not defined");
  MCStreamer &OutStreamer = getStreamer();
  MCContext &OutContext = OutStreamer.getContext();

  // Optional Argument Area Length.
  // Note: This represents the length of the argument area that we reserve
  //       in our stack for setting up arguments for calls to other
  //       routines. If this optional field is not set, LE will reserve
  //       128 bytes for the argument area. This optional field is
  //       created if greater than 128 bytes is required - to guarantee
  //       the required space is reserved on stack extension in the new
  //       extension.  This optional field is also created if the
  //       routine has alloca(). This may reduce stack space
  //       if alloca() call causes a stack extension.
  bool HasArgAreaLength = (Info.AllocaReg != 0) || (Info.CallFrameSize > 128);

  // The personality function is present if at least one of the displacements is
  // larger than zero.
  bool HasPersonalityFn = Info.PersonalityADADisp > 0 || Info.GCCEHADADisp > 0;

  // Emit PPA1 section.
  OutStreamer.AddComment("PPA1");
  OutStreamer.emitLabel(Info.PPA1);
  OutStreamer.AddComment("Version");
  OutStreamer.emitInt8(0x02); // Version.
  OutStreamer.AddComment("LE Signature X'CE'");
  OutStreamer.emitInt8(0xCE); // CEL signature.
  OutStreamer.AddComment("Saved GPR Mask");
  OutStreamer.emitInt16(Info.SavedGPRMask);
  OutStreamer.AddComment("Offset to PPA2");
  OutStreamer.emitAbsoluteSymbolDiff(PPA2Sym, Info.PPA1, 4);

  emitPPA1Flags(OutStreamer, Info.IsVarArg, Info.HasStackProtector,
                Info.SavedFPRMask != 0, Info.SavedVRMask != 0, HasPersonalityFn,
                HasArgAreaLength, Info.Name.size() > 0);

  OutStreamer.AddComment("Length/4 of Parms");
  OutStreamer.emitInt16(
      static_cast<uint16_t>(Info.SizeOfFnParams / 4)); // Parms/4.

  OutStreamer.AddComment("Length/2 of Prolog ");
  if (Info.EndOfProlog)
    OutStreamer.emitValue(
        createWordDiffExpr(OutContext, Info.EndOfProlog, Info.Fn), 1);
  else
    OutStreamer.emitInt8(0);

  OutStreamer.AddComment("Alloca Reg + Offset/2 to SP Update");
  OutStreamer.AddComment(
      Twine("  Bit 0-3: Register R").concat(utostr(Info.AllocaReg)).str());
  OutStreamer.AddComment("  Bit 4-8: Offset ");
  const MCExpr *AllocaRegExpr =
      MCConstantExpr::create(Info.AllocaReg << 4, OutContext);
  if (Info.StackUpdate)
    OutStreamer.emitValue(
        MCBinaryExpr::createOr(
            createWordDiffExpr(OutContext, Info.StackUpdate, Info.Fn),
            AllocaRegExpr, OutContext),
        1);
  else
    OutStreamer.emitValue(AllocaRegExpr, 1);

  OutStreamer.AddComment("Length of Code");
  OutStreamer.emitAbsoluteSymbolDiff(Info.FnEnd, Info.EPMarker, 4);

  if (HasArgAreaLength) {
    OutStreamer.AddComment("Argument Area Length");
    OutStreamer.emitInt32(Info.CallFrameSize);
  }

  // Emit saved FPR mask and offset to FPR save area (0x20 of flags 3).
  if (Info.SavedFPRMask) {
    OutStreamer.AddComment("FPR mask");
    OutStreamer.emitInt16(Info.SavedFPRMask);
    OutStreamer.AddComment("AR mask");
    OutStreamer.emitInt16(0); // AR Mask, unused currently.
    OutStreamer.AddComment("FPR Save Area Locator");
    uint64_t FPRSaveAreaOffset = Info.OffsetFPR;
    assert(FPRSaveAreaOffset < 0x10000000 && "Offset out of range");
    FPRSaveAreaOffset &= 0x0FFFFFFF; // Lose top 4 bits.
    OutStreamer.AddComment(
        Twine("  Bit 0-3: Register R").concat(utostr(Info.FrameReg)));
    OutStreamer.AddComment(
        Twine("  Bit 4-31: Offset ").concat(utostr(FPRSaveAreaOffset)));
    OutStreamer.emitInt32(FPRSaveAreaOffset |
                          (Info.FrameReg << 28)); // Offset to FPR save area
                                                  // with register to add
                                                  // value to (alloca reg).
  }

  // Emit saved VR mask to VR save area.
  if (Info.SavedVRMask) {
    OutStreamer.AddComment("VR mask");
    OutStreamer.emitInt8(Info.SavedVRMask);
    OutStreamer.emitInt8(0);  // Reserved.
    OutStreamer.emitInt16(0); // Also reserved.
    uint64_t VRSaveAreaOffset = Info.OffsetVR;
    assert(VRSaveAreaOffset < 0x10000000 && "Offset out of range");
    VRSaveAreaOffset &= 0x0FFFFFFF; // Lose top 4 bits.
    OutStreamer.AddComment("VR Save Area Locator");
    OutStreamer.AddComment(
        Twine("  Bit 0-3: Register R").concat(utostr(Info.FrameReg)));
    OutStreamer.AddComment(
        Twine("  Bit 4-31: Offset ").concat(utostr(VRSaveAreaOffset)));
    OutStreamer.emitInt32(VRSaveAreaOffset | (Info.FrameReg << 28));
  }

  // Emit C++ EH information block.
  if (HasPersonalityFn) {
    OutStreamer.AddComment("Version");
    OutStreamer.emitInt32(1);
    OutStreamer.AddComment("Flags");
    OutStreamer.emitInt32(0); // LSDA field is a WAS offset
    OutStreamer.AddComment("Personality routine");
    OutStreamer.emitInt64(Info.PersonalityADADisp);
    OutStreamer.AddComment("LSDA location");
    OutStreamer.emitInt64(Info.GCCEHADADisp);
  }

  // Emit name length and name optional section (0x01 of flags 4)
  if (Info.Name.size() > 0)
    emitPPA1Name(OutStreamer, Info.Name);

  // Emit offset to entry point optional section (0x80 of flags 4).
  OutStreamer.emitAbsoluteSymbolDiff(Info.EPMarker, Info.PPA1, 4);
}

void SystemZTargetzOSStreamer::emitConstantPools() {
  // Emit EXRL target instructions (base class prolog).
  SystemZTargetStreamer::emitConstantPools();

  // Emit deferred PPA1 blocks into the text section.
  if (DeferredPPA1.empty())
    return;
  const MCObjectFileInfo &OFI = *getStreamer().getContext().getObjectFileInfo();
  getStreamer().switchSection(OFI.getTextSection());
  for (auto &Info : DeferredPPA1)
    emitPPA1(Info);
}

SystemZHLASMAsmStreamer &SystemZTargetHLASMStreamer::getHLASMStreamer() {
  return static_cast<SystemZHLASMAsmStreamer &>(getStreamer());
}

// HLASM statements can only perform a single operation at a time
const MCExpr *SystemZTargetHLASMStreamer::createWordDiffExpr(
    MCContext &Ctx, const MCSymbol *Hi, const MCSymbol *Lo) {
  assert(Hi && Lo && "Symbols required to calculate expression");
  MCSymbol *Temp = Ctx.createTempSymbol();
  OS << Temp->getName() << " EQU ";
  const MCBinaryExpr *TempExpr = MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(Hi, Ctx), MCSymbolRefExpr::create(Lo, Ctx), Ctx);
  Ctx.getAsmInfo().printExpr(OS, *TempExpr);
  OS << "\n";
  return MCBinaryExpr::createLShr(MCSymbolRefExpr::create(Temp, Ctx),
                                  MCConstantExpr::create(1, Ctx), Ctx);
}

const MCExpr *SystemZTargetGOFFStreamer::createWordDiffExpr(
    MCContext &Ctx, const MCSymbol *Hi, const MCSymbol *Lo) {
  assert(Hi && Lo && "Symbols required to calculate expression");
  return MCBinaryExpr::createLShr(
      MCBinaryExpr::createSub(MCSymbolRefExpr::create(Hi, Ctx),
                              MCSymbolRefExpr::create(Lo, Ctx), Ctx),
      MCConstantExpr::create(1, Ctx), Ctx);
}
