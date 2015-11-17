//===-- MipsAsmBackend.cpp - Mips Asm Backend  ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MipsAsmBackend class.
//
//===----------------------------------------------------------------------===//
//

#include "MCTargetDesc/MipsFixupKinds.h"
#include "MCTargetDesc/MipsAsmBackend.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Prepare value for the target space for it
static unsigned adjustFixupValue(const MCFixup &Fixup, uint64_t Value,
                                 MCContext *Ctx = nullptr) {

  unsigned Kind = Fixup.getKind();

  // Add/subtract and shift
  switch (Kind) {
  default:
    return 0;
  case FK_Data_2:
  case FK_GPRel_4:
  case FK_Data_4:
  case FK_Data_8:
  case Mips::fixup_Mips_LO16:
  case Mips::fixup_Mips_GPREL16:
  case Mips::fixup_Mips_GPOFF_HI:
  case Mips::fixup_Mips_GPOFF_LO:
  case Mips::fixup_Mips_GOT_PAGE:
  case Mips::fixup_Mips_GOT_OFST:
  case Mips::fixup_Mips_GOT_DISP:
  case Mips::fixup_Mips_GOT_LO16:
  case Mips::fixup_Mips_CALL_LO16:
  case Mips::fixup_MICROMIPS_LO16:
  case Mips::fixup_MICROMIPS_GOT_PAGE:
  case Mips::fixup_MICROMIPS_GOT_OFST:
  case Mips::fixup_MICROMIPS_GOT_DISP:
  case Mips::fixup_MIPS_PCLO16:
    break;
  case Mips::fixup_Mips_PC16:
    // The displacement is then divided by 4 to give us an 18 bit
    // address range. Forcing a signed division because Value can be negative.
    Value = (int64_t)Value / 4;
    // We now check if Value can be encoded as a 16-bit signed immediate.
    if (!isInt<16>(Value) && Ctx) {
      Ctx->reportError(Fixup.getLoc(), "out of range PC16 fixup");
      return 0;
    }
    break;
  case Mips::fixup_MIPS_PC19_S2:
    // Forcing a signed division because Value can be negative.
    Value = (int64_t)Value / 4;
    // We now check if Value can be encoded as a 19-bit signed immediate.
    if (!isInt<19>(Value) && Ctx) {
      Ctx->reportError(Fixup.getLoc(), "out of range PC19 fixup");
      return 0;
    }
    break;
  case Mips::fixup_Mips_26:
    // So far we are only using this type for jumps.
    // The displacement is then divided by 4 to give us an 28 bit
    // address range.
    Value >>= 2;
    break;
  case Mips::fixup_Mips_HI16:
  case Mips::fixup_Mips_GOT_Local:
  case Mips::fixup_Mips_GOT_HI16:
  case Mips::fixup_Mips_CALL_HI16:
  case Mips::fixup_MICROMIPS_HI16:
  case Mips::fixup_MIPS_PCHI16:
    // Get the 2nd 16-bits. Also add 1 if bit 15 is 1.
    Value = ((Value + 0x8000) >> 16) & 0xffff;
    break;
  case Mips::fixup_Mips_HIGHER:
    // Get the 3rd 16-bits.
    Value = ((Value + 0x80008000LL) >> 32) & 0xffff;
    break;
  case Mips::fixup_Mips_HIGHEST:
    // Get the 4th 16-bits.
    Value = ((Value + 0x800080008000LL) >> 48) & 0xffff;
    break;
  case Mips::fixup_MICROMIPS_26_S1:
    Value >>= 1;
    break;
  case Mips::fixup_MICROMIPS_PC7_S1:
    Value -= 4;
    // Forcing a signed division because Value can be negative.
    Value = (int64_t) Value / 2;
    // We now check if Value can be encoded as a 7-bit signed immediate.
    if (!isInt<7>(Value) && Ctx) {
      Ctx->reportError(Fixup.getLoc(), "out of range PC7 fixup");
      return 0;
    }
    break;
  case Mips::fixup_MICROMIPS_PC10_S1:
    Value -= 2;
    // Forcing a signed division because Value can be negative.
    Value = (int64_t) Value / 2;
    // We now check if Value can be encoded as a 10-bit signed immediate.
    if (!isInt<10>(Value) && Ctx) {
      Ctx->reportError(Fixup.getLoc(), "out of range PC10 fixup");
      return 0;
    }
    break;
  case Mips::fixup_MICROMIPS_PC16_S1:
    Value -= 4;
    // Forcing a signed division because Value can be negative.
    Value = (int64_t)Value / 2;
    // We now check if Value can be encoded as a 16-bit signed immediate.
    if (!isInt<16>(Value) && Ctx) {
      Ctx->reportError(Fixup.getLoc(), "out of range PC16 fixup");
      return 0;
    }
    break;
  case Mips::fixup_MIPS_PC18_S3:
    // Forcing a signed division because Value can be negative.
    Value = (int64_t)Value / 8;
    // We now check if Value can be encoded as a 18-bit signed immediate.
    if (!isInt<18>(Value) && Ctx) {
      Ctx->reportError(Fixup.getLoc(), "out of range PC18 fixup");
      return 0;
    }
    break;
  case Mips::fixup_MIPS_PC21_S2:
    // Forcing a signed division because Value can be negative.
    Value = (int64_t) Value / 4;
    // We now check if Value can be encoded as a 21-bit signed immediate.
    if (!isInt<21>(Value) && Ctx) {
      Ctx->reportError(Fixup.getLoc(), "out of range PC21 fixup");
      return 0;
    }
    break;
  case Mips::fixup_MIPS_PC26_S2:
    // Forcing a signed division because Value can be negative.
    Value = (int64_t) Value / 4;
    // We now check if Value can be encoded as a 26-bit signed immediate.
    if (!isInt<26>(Value) && Ctx) {
      Ctx->reportError(Fixup.getLoc(), "out of range PC26 fixup");
      return 0;
    }
    break;
  }

  return Value;
}

MCObjectWriter *
MipsAsmBackend::createObjectWriter(raw_pwrite_stream &OS) const {
  return createMipsELFObjectWriter(OS,
    MCELFObjectTargetWriter::getOSABI(OSType), IsLittle, Is64Bit);
}

// Little-endian fixup data byte ordering:
//   mips32r2:   a | b | x | x
//   microMIPS:  x | x | a | b

static bool needsMMLEByteOrder(unsigned Kind) {
  return Kind != Mips::fixup_MICROMIPS_PC10_S1 &&
         Kind >= Mips::fixup_MICROMIPS_26_S1 &&
         Kind < Mips::LastTargetFixupKind;
}

// Calculate index for microMIPS specific little endian byte order
static unsigned calculateMMLEIndex(unsigned i) {
  assert(i <= 3 && "Index out of range!");

  return (1 - i / 2) * 2 + i % 2;
}

/// ApplyFixup - Apply the \p Value for given \p Fixup into the provided
/// data fragment, at the offset specified by the fixup and following the
/// fixup kind as appropriate.
void MipsAsmBackend::applyFixup(const MCFixup &Fixup, char *Data,
                                unsigned DataSize, uint64_t Value,
                                bool IsPCRel) const {
  MCFixupKind Kind = Fixup.getKind();
  Value = adjustFixupValue(Fixup, Value);

  if (!Value)
    return; // Doesn't change encoding.

  // Where do we start in the object
  unsigned Offset = Fixup.getOffset();
  // Number of bytes we need to fixup
  unsigned NumBytes = (getFixupKindInfo(Kind).TargetSize + 7) / 8;
  // Used to point to big endian bytes
  unsigned FullSize;

  switch ((unsigned)Kind) {
  case FK_Data_2:
  case Mips::fixup_Mips_16:
  case Mips::fixup_MICROMIPS_PC10_S1:
    FullSize = 2;
    break;
  case FK_Data_8:
  case Mips::fixup_Mips_64:
    FullSize = 8;
    break;
  case FK_Data_4:
  default:
    FullSize = 4;
    break;
  }

  // Grab current value, if any, from bits.
  uint64_t CurVal = 0;

  bool microMipsLEByteOrder = needsMMLEByteOrder((unsigned) Kind);

  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned Idx = IsLittle ? (microMipsLEByteOrder ? calculateMMLEIndex(i)
                                                    : i)
                            : (FullSize - 1 - i);
    CurVal |= (uint64_t)((uint8_t)Data[Offset + Idx]) << (i*8);
  }

  uint64_t Mask = ((uint64_t)(-1) >>
                    (64 - getFixupKindInfo(Kind).TargetSize));
  CurVal |= Value & Mask;

  // Write out the fixed up bytes back to the code/data bits.
  for (unsigned i = 0; i != NumBytes; ++i) {
    unsigned Idx = IsLittle ? (microMipsLEByteOrder ? calculateMMLEIndex(i)
                                                    : i)
                            : (FullSize - 1 - i);
    Data[Offset + Idx] = (uint8_t)((CurVal >> (i*8)) & 0xff);
  }
}

bool MipsAsmBackend::getFixupKind(StringRef Name, MCFixupKind &MappedKind) const {
  if (Name == "R_MIPS_NONE") {
    MappedKind = (MCFixupKind)Mips::fixup_Mips_NONE;
    return true;
  }
  if (Name == "R_MIPS_32") {
    MappedKind = FK_Data_4;
    return true;
  }
  return MCAsmBackend::getFixupKind(Name, MappedKind);
}

const MCFixupKindInfo &MipsAsmBackend::
getFixupKindInfo(MCFixupKind Kind) const {
  const static MCFixupKindInfo LittleEndianInfos[Mips::NumTargetFixupKinds] = {
    // This table *must* be in same the order of fixup_* kinds in
    // MipsFixupKinds.h.
    //
    // name                    offset  bits  flags
    { "fixup_Mips_NONE",         0,      0,   0 },
    { "fixup_Mips_16",           0,     16,   0 },
    { "fixup_Mips_32",           0,     32,   0 },
    { "fixup_Mips_REL32",        0,     32,   0 },
    { "fixup_Mips_26",           0,     26,   0 },
    { "fixup_Mips_HI16",         0,     16,   0 },
    { "fixup_Mips_LO16",         0,     16,   0 },
    { "fixup_Mips_GPREL16",      0,     16,   0 },
    { "fixup_Mips_LITERAL",      0,     16,   0 },
    { "fixup_Mips_GOT_Global",   0,     16,   0 },
    { "fixup_Mips_GOT_Local",    0,     16,   0 },
    { "fixup_Mips_PC16",         0,     16,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_Mips_CALL16",       0,     16,   0 },
    { "fixup_Mips_GPREL32",      0,     32,   0 },
    { "fixup_Mips_SHIFT5",       6,      5,   0 },
    { "fixup_Mips_SHIFT6",       6,      5,   0 },
    { "fixup_Mips_64",           0,     64,   0 },
    { "fixup_Mips_TLSGD",        0,     16,   0 },
    { "fixup_Mips_GOTTPREL",     0,     16,   0 },
    { "fixup_Mips_TPREL_HI",     0,     16,   0 },
    { "fixup_Mips_TPREL_LO",     0,     16,   0 },
    { "fixup_Mips_TLSLDM",       0,     16,   0 },
    { "fixup_Mips_DTPREL_HI",    0,     16,   0 },
    { "fixup_Mips_DTPREL_LO",    0,     16,   0 },
    { "fixup_Mips_Branch_PCRel", 0,     16,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_Mips_GPOFF_HI",     0,     16,   0 },
    { "fixup_Mips_GPOFF_LO",     0,     16,   0 },
    { "fixup_Mips_GOT_PAGE",     0,     16,   0 },
    { "fixup_Mips_GOT_OFST",     0,     16,   0 },
    { "fixup_Mips_GOT_DISP",     0,     16,   0 },
    { "fixup_Mips_HIGHER",       0,     16,   0 },
    { "fixup_Mips_HIGHEST",      0,     16,   0 },
    { "fixup_Mips_GOT_HI16",     0,     16,   0 },
    { "fixup_Mips_GOT_LO16",     0,     16,   0 },
    { "fixup_Mips_CALL_HI16",    0,     16,   0 },
    { "fixup_Mips_CALL_LO16",    0,     16,   0 },
    { "fixup_Mips_PC18_S3",      0,     18,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PC19_S2",      0,     19,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PC21_S2",      0,     21,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PC26_S2",      0,     26,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PCHI16",       0,     16,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PCLO16",       0,     16,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MICROMIPS_26_S1",   0,     26,   0 },
    { "fixup_MICROMIPS_HI16",    0,     16,   0 },
    { "fixup_MICROMIPS_LO16",    0,     16,   0 },
    { "fixup_MICROMIPS_GOT16",   0,     16,   0 },
    { "fixup_MICROMIPS_PC7_S1",  0,      7,   MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MICROMIPS_PC10_S1", 0,     10,   MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MICROMIPS_PC16_S1", 0,     16,   MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MICROMIPS_CALL16",  0,     16,   0 },
    { "fixup_MICROMIPS_GOT_DISP",        0,     16,   0 },
    { "fixup_MICROMIPS_GOT_PAGE",        0,     16,   0 },
    { "fixup_MICROMIPS_GOT_OFST",        0,     16,   0 },
    { "fixup_MICROMIPS_TLS_GD",          0,     16,   0 },
    { "fixup_MICROMIPS_TLS_LDM",         0,     16,   0 },
    { "fixup_MICROMIPS_TLS_DTPREL_HI16", 0,     16,   0 },
    { "fixup_MICROMIPS_TLS_DTPREL_LO16", 0,     16,   0 },
    { "fixup_MICROMIPS_TLS_TPREL_HI16",  0,     16,   0 },
    { "fixup_MICROMIPS_TLS_TPREL_LO16",  0,     16,   0 }
  };

  const static MCFixupKindInfo BigEndianInfos[Mips::NumTargetFixupKinds] = {
    // This table *must* be in same the order of fixup_* kinds in
    // MipsFixupKinds.h.
    //
    // name                    offset  bits  flags
    { "fixup_Mips_NONE",         0,      0,   0 },
    { "fixup_Mips_16",          16,     16,   0 },
    { "fixup_Mips_32",           0,     32,   0 },
    { "fixup_Mips_REL32",        0,     32,   0 },
    { "fixup_Mips_26",           6,     26,   0 },
    { "fixup_Mips_HI16",        16,     16,   0 },
    { "fixup_Mips_LO16",        16,     16,   0 },
    { "fixup_Mips_GPREL16",     16,     16,   0 },
    { "fixup_Mips_LITERAL",     16,     16,   0 },
    { "fixup_Mips_GOT_Global",  16,     16,   0 },
    { "fixup_Mips_GOT_Local",   16,     16,   0 },
    { "fixup_Mips_PC16",        16,     16,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_Mips_CALL16",      16,     16,   0 },
    { "fixup_Mips_GPREL32",      0,     32,   0 },
    { "fixup_Mips_SHIFT5",      21,      5,   0 },
    { "fixup_Mips_SHIFT6",      21,      5,   0 },
    { "fixup_Mips_64",           0,     64,   0 },
    { "fixup_Mips_TLSGD",       16,     16,   0 },
    { "fixup_Mips_GOTTPREL",    16,     16,   0 },
    { "fixup_Mips_TPREL_HI",    16,     16,   0 },
    { "fixup_Mips_TPREL_LO",    16,     16,   0 },
    { "fixup_Mips_TLSLDM",      16,     16,   0 },
    { "fixup_Mips_DTPREL_HI",   16,     16,   0 },
    { "fixup_Mips_DTPREL_LO",   16,     16,   0 },
    { "fixup_Mips_Branch_PCRel",16,     16,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_Mips_GPOFF_HI",    16,     16,   0 },
    { "fixup_Mips_GPOFF_LO",    16,     16,   0 },
    { "fixup_Mips_GOT_PAGE",    16,     16,   0 },
    { "fixup_Mips_GOT_OFST",    16,     16,   0 },
    { "fixup_Mips_GOT_DISP",    16,     16,   0 },
    { "fixup_Mips_HIGHER",      16,     16,   0 },
    { "fixup_Mips_HIGHEST",     16,     16,   0 },
    { "fixup_Mips_GOT_HI16",    16,     16,   0 },
    { "fixup_Mips_GOT_LO16",    16,     16,   0 },
    { "fixup_Mips_CALL_HI16",   16,     16,   0 },
    { "fixup_Mips_CALL_LO16",   16,     16,   0 },
    { "fixup_Mips_PC18_S3",     14,     18,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PC19_S2",     13,     19,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PC21_S2",     11,     21,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PC26_S2",      6,     26,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PCHI16",      16,     16,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MIPS_PCLO16",      16,     16,  MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MICROMIPS_26_S1",   6,     26,   0 },
    { "fixup_MICROMIPS_HI16",   16,     16,   0 },
    { "fixup_MICROMIPS_LO16",   16,     16,   0 },
    { "fixup_MICROMIPS_GOT16",  16,     16,   0 },
    { "fixup_MICROMIPS_PC7_S1",  9,      7,   MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MICROMIPS_PC10_S1", 6,     10,   MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MICROMIPS_PC16_S1",16,     16,   MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MICROMIPS_CALL16", 16,     16,   0 },
    { "fixup_MICROMIPS_GOT_DISP",        16,     16,   0 },
    { "fixup_MICROMIPS_GOT_PAGE",        16,     16,   0 },
    { "fixup_MICROMIPS_GOT_OFST",        16,     16,   0 },
    { "fixup_MICROMIPS_TLS_GD",          16,     16,   0 },
    { "fixup_MICROMIPS_TLS_LDM",         16,     16,   0 },
    { "fixup_MICROMIPS_TLS_DTPREL_HI16", 16,     16,   0 },
    { "fixup_MICROMIPS_TLS_DTPREL_LO16", 16,     16,   0 },
    { "fixup_MICROMIPS_TLS_TPREL_HI16",  16,     16,   0 },
    { "fixup_MICROMIPS_TLS_TPREL_LO16",  16,     16,   0 }
  };

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
          "Invalid kind!");

  if (IsLittle)
    return LittleEndianInfos[Kind - FirstTargetFixupKind];
  return BigEndianInfos[Kind - FirstTargetFixupKind];
}

/// WriteNopData - Write an (optimal) nop sequence of Count bytes
/// to the given output. If the target cannot generate such a sequence,
/// it should return an error.
///
/// \return - True on success.
bool MipsAsmBackend::writeNopData(uint64_t Count, MCObjectWriter *OW) const {
  // Check for a less than instruction size number of bytes
  // FIXME: 16 bit instructions are not handled yet here.
  // We shouldn't be using a hard coded number for instruction size.

  // If the count is not 4-byte aligned, we must be writing data into the text
  // section (otherwise we have unaligned instructions, and thus have far
  // bigger problems), so just write zeros instead.
  OW->WriteZeros(Count);
  return true;
}

/// processFixupValue - Target hook to process the literal value of a fixup
/// if necessary.
void MipsAsmBackend::processFixupValue(const MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       const MCFixup &Fixup,
                                       const MCFragment *DF,
                                       const MCValue &Target,
                                       uint64_t &Value,
                                       bool &IsResolved) {
  // At this point we'll ignore the value returned by adjustFixupValue as
  // we are only checking if the fixup can be applied correctly. We have
  // access to MCContext from here which allows us to report a fatal error
  // with *possibly* a source code location.
  (void)adjustFixupValue(Fixup, Value, &Asm.getContext());
}

// MCAsmBackend
MCAsmBackend *llvm::createMipsAsmBackendEL32(const Target &T,
                                             const MCRegisterInfo &MRI,
                                             const Triple &TT, StringRef CPU) {
  return new MipsAsmBackend(T, TT.getOS(), /*IsLittle*/ true,
                            /*Is64Bit*/ false);
}

MCAsmBackend *llvm::createMipsAsmBackendEB32(const Target &T,
                                             const MCRegisterInfo &MRI,
                                             const Triple &TT, StringRef CPU) {
  return new MipsAsmBackend(T, TT.getOS(), /*IsLittle*/ false,
                            /*Is64Bit*/ false);
}

MCAsmBackend *llvm::createMipsAsmBackendEL64(const Target &T,
                                             const MCRegisterInfo &MRI,
                                             const Triple &TT, StringRef CPU) {
  return new MipsAsmBackend(T, TT.getOS(), /*IsLittle*/ true, /*Is64Bit*/ true);
}

MCAsmBackend *llvm::createMipsAsmBackendEB64(const Target &T,
                                             const MCRegisterInfo &MRI,
                                             const Triple &TT, StringRef CPU) {
  return new MipsAsmBackend(T, TT.getOS(), /*IsLittle*/ false,
                            /*Is64Bit*/ true);
}
