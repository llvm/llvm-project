//===- ARMDisassembler.cpp - Disassembler for ARM/Thumb ISA -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the ARM Disassembler.
// It contains code to implement the public interfaces of ARMDisassembler and
// ThumbDisassembler, both of which are instances of MCDisassembler.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-disassembler"

#include "ARMDisassembler.h"
#include "ARMDisassemblerCore.h"

#include "llvm/MC/EDInstInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

//#define DEBUG(X) do { X; } while (0)

/// ARMGenDecoderTables.inc - ARMDecoderTables.inc is tblgen'ed from
/// ARMDecoderEmitter.cpp TableGen backend.  It contains:
///
/// o Mappings from opcode to ARM/Thumb instruction format
///
/// o static uint16_t decodeInstruction(uint32_t insn) - the decoding function
/// for an ARM instruction.
///
/// o static uint16_t decodeThumbInstruction(field_t insn) - the decoding
/// function for a Thumb instruction.
///
#include "../ARMGenDecoderTables.inc"

#include "../ARMGenEDInfo.inc"

using namespace llvm;

/// showBitVector - Use the raw_ostream to log a diagnostic message describing
/// the inidividual bits of the instruction.
///
static inline void showBitVector(raw_ostream &os, const uint32_t &insn) {
  // Split the bit position markers into more than one lines to fit 80 columns.
  os << " 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11"
     << " 10  9  8  7  6  5  4  3  2  1  0 \n";
  os << "---------------------------------------------------------------"
     << "----------------------------------\n";
  os << '|';
  for (unsigned i = 32; i != 0; --i) {
    if (insn >> (i - 1) & 0x01)
      os << " 1";
    else
      os << " 0";
    os << (i%4 == 1 ? '|' : ':');
  }
  os << '\n';
  // Split the bit position markers into more than one lines to fit 80 columns.
  os << "---------------------------------------------------------------"
     << "----------------------------------\n";
  os << '\n';
}

/// decodeARMInstruction is a decorator function which tries special cases of
/// instruction matching before calling the auto-generated decoder function.
static unsigned decodeARMInstruction(uint32_t &insn) {
  if (slice(insn, 31, 28) == 15)
    goto AutoGenedDecoder;

  // Special case processing, if any, goes here....

  // LLVM combines the offset mode of A8.6.197 & A8.6.198 into STRB.
  // The insufficient encoding information of the combined instruction confuses
  // the decoder wrt BFC/BFI.  Therefore, we try to recover here.
  // For BFC, Inst{27-21} = 0b0111110 & Inst{6-0} = 0b0011111.
  // For BFI, Inst{27-21} = 0b0111110 & Inst{6-4} = 0b001 & Inst{3-0} =! 0b1111.
  if (slice(insn, 27, 21) == 0x3e && slice(insn, 6, 4) == 1) {
    if (slice(insn, 3, 0) == 15)
      return ARM::BFC;
    else
      return ARM::BFI;
  }

  // Ditto for STRBT, which is a super-instruction for A8.6.199 Encoding A1 & A2.
  // As a result, the decoder fails to deocode USAT properly.
  if (slice(insn, 27, 21) == 0x37 && slice(insn, 5, 4) == 1)
    return ARM::USAT;

  // Ditto for ADDSrs, which is a super-instruction for A8.6.7 & A8.6.8.
  // As a result, the decoder fails to decode UMULL properly.
  if (slice(insn, 27, 21) == 0x04 && slice(insn, 7, 4) == 9) {
    return ARM::UMULL;
  }

  // Ditto for STR_PRE, which is a super-instruction for A8.6.194 & A8.6.195.
  // As a result, the decoder fails to decode SBFX properly.
  if (slice(insn, 27, 21) == 0x3d && slice(insn, 6, 4) == 5)
    return ARM::SBFX;

  // And STRB_PRE, which is a super-instruction for A8.6.197 & A8.6.198.
  // As a result, the decoder fails to decode UBFX properly.
  if (slice(insn, 27, 21) == 0x3f && slice(insn, 6, 4) == 5)
    return ARM::UBFX;

  // Ditto for STRT, which is a super-instruction for A8.6.210 Encoding A1 & A2.
  // As a result, the decoder fails to deocode SSAT properly.
  if (slice(insn, 27, 21) == 0x35 && slice(insn, 5, 4) == 1)
    return ARM::SSAT;

  // Ditto for RSCrs, which is a super-instruction for A8.6.146 & A8.6.147.
  // As a result, the decoder fails to decode STRHT/LDRHT/LDRSHT/LDRSBT.
  if (slice(insn, 27, 24) == 0) {
    switch (slice(insn, 21, 20)) {
    case 2:
      switch (slice(insn, 7, 4)) {
      case 11:
        return ARM::STRHT;
      default:
        break; // fallthrough
      }
      break;
    case 3:
      switch (slice(insn, 7, 4)) {
      case 11:
        return ARM::LDRHT;
      case 13:
        return ARM::LDRSBT;
      case 15:
        return ARM::LDRSHT;
      default:
        break; // fallthrough
      }
      break;
    default:
      break;   // fallthrough
    }
  }

  // Ditto for SBCrs, which is a super-instruction for A8.6.152 & A8.6.153.
  // As a result, the decoder fails to decode STRH_Post/LDRD_POST/STRD_POST
  // properly.
  if (slice(insn, 27, 25) == 0 && slice(insn, 20, 20) == 0) {
    unsigned PW = slice(insn, 24, 24) << 1 | slice(insn, 21, 21);
    switch (slice(insn, 7, 4)) {
    case 11:
      switch (PW) {
      case 2: // Offset
        return ARM::STRH;
      case 3: // Pre-indexed
        return ARM::STRH_PRE;
      case 0: // Post-indexed
        return ARM::STRH_POST;
      default:
        break; // fallthrough
      }
      break;
    case 13:
      switch (PW) {
      case 2: // Offset
        return ARM::LDRD;
      case 3: // Pre-indexed
        return ARM::LDRD_PRE;
      case 0: // Post-indexed
        return ARM::LDRD_POST;
      default:
        break; // fallthrough
      }
      break;
    case 15:
      switch (PW) {
      case 2: // Offset
        return ARM::STRD;
      case 3: // Pre-indexed
        return ARM::STRD_PRE;
      case 0: // Post-indexed
        return ARM::STRD_POST;
      default:
        break; // fallthrough
      }
      break;
    default:
      break; // fallthrough
    }
  }

  // Ditto for SBCSSrs, which is a super-instruction for A8.6.152 & A8.6.153.
  // As a result, the decoder fails to decode LDRH_POST/LDRSB_POST/LDRSH_POST
  // properly.
  if (slice(insn, 27, 25) == 0 && slice(insn, 20, 20) == 1) {
    unsigned PW = slice(insn, 24, 24) << 1 | slice(insn, 21, 21);
    switch (slice(insn, 7, 4)) {
    case 11:
      switch (PW) {
      case 2: // Offset
        return ARM::LDRH;
      case 3: // Pre-indexed
        return ARM::LDRH_PRE;
      case 0: // Post-indexed
        return ARM::LDRH_POST;
      default:
        break; // fallthrough
      }
      break;
    case 13:
      switch (PW) {
      case 2: // Offset
        return ARM::LDRSB;
      case 3: // Pre-indexed
        return ARM::LDRSB_PRE;
      case 0: // Post-indexed
        return ARM::LDRSB_POST;
      default:
        break; // fallthrough
      }
      break;
    case 15:
      switch (PW) {
      case 2: // Offset
        return ARM::LDRSH;
      case 3: // Pre-indexed
        return ARM::LDRSH_PRE;
      case 0: // Post-indexed
        return ARM::LDRSH_POST;
      default:
        break; // fallthrough
      }
      break;
    default:
      break; // fallthrough
    }
  }

AutoGenedDecoder:
  // Calling the auto-generated decoder function.
  return decodeInstruction(insn);
}

// Helper function for special case handling of LDR (literal) and friends.
// See, for example, A6.3.7 Load word: Table A6-18 Load word.
// See A8.6.57 T3, T4 & A8.6.60 T2 and friends for why we morphed the opcode
// before returning it.
static unsigned T2Morph2LoadLiteral(unsigned Opcode) {
  switch (Opcode) {
  default:
    return Opcode; // Return unmorphed opcode.

  case ARM::t2LDRDi8:
    return ARM::t2LDRDpci;

  case ARM::t2LDR_POST:   case ARM::t2LDR_PRE:
  case ARM::t2LDRi12:     case ARM::t2LDRi8:
  case ARM::t2LDRs:       case ARM::t2LDRT:
    return ARM::t2LDRpci;

  case ARM::t2LDRB_POST:  case ARM::t2LDRB_PRE:
  case ARM::t2LDRBi12:    case ARM::t2LDRBi8:
  case ARM::t2LDRBs:      case ARM::t2LDRBT:
    return ARM::t2LDRBpci;

  case ARM::t2LDRH_POST:  case ARM::t2LDRH_PRE:
  case ARM::t2LDRHi12:    case ARM::t2LDRHi8:
  case ARM::t2LDRHs:      case ARM::t2LDRHT:
    return ARM::t2LDRHpci;

  case ARM::t2LDRSB_POST:  case ARM::t2LDRSB_PRE:
  case ARM::t2LDRSBi12:    case ARM::t2LDRSBi8:
  case ARM::t2LDRSBs:      case ARM::t2LDRSBT:
    return ARM::t2LDRSBpci;

  case ARM::t2LDRSH_POST:  case ARM::t2LDRSH_PRE:
  case ARM::t2LDRSHi12:    case ARM::t2LDRSHi8:
  case ARM::t2LDRSHs:      case ARM::t2LDRSHT:
    return ARM::t2LDRSHpci;
  }
}

/// decodeThumbSideEffect is a decorator function which can potentially twiddle
/// the instruction or morph the returned opcode under Thumb2.
///
/// First it checks whether the insn is a NEON or VFP instr; if true, bit
/// twiddling could be performed on insn to turn it into an ARM NEON/VFP
/// equivalent instruction and decodeInstruction is called with the transformed
/// insn.
///
/// Next, there is special handling for Load byte/halfword/word instruction by
/// checking whether Rn=0b1111 and call T2Morph2LoadLiteral() on the decoded
/// Thumb2 instruction.  See comments below for further details.
///
/// Finally, one last check is made to see whether the insn is a NEON/VFP and
/// decodeInstruction(insn) is invoked on the original insn.
///
/// Otherwise, decodeThumbInstruction is called with the original insn.
static unsigned decodeThumbSideEffect(bool IsThumb2, unsigned &insn) {
  if (IsThumb2) {
    uint16_t op1 = slice(insn, 28, 27);
    uint16_t op2 = slice(insn, 26, 20);

    // A6.3 32-bit Thumb instruction encoding
    // Table A6-9 32-bit Thumb instruction encoding

    // The coprocessor instructions of interest are transformed to their ARM
    // equivalents.

    // --------- Transform Begin Marker ---------
    if ((op1 == 1 || op1 == 3) && slice(op2, 6, 4) == 7) {
      // A7.4 Advanced SIMD data-processing instructions
      // U bit of Thumb corresponds to Inst{24} of ARM.
      uint16_t U = slice(op1, 1, 1);

      // Inst{28-24} of ARM = {1,0,0,1,U};
      uint16_t bits28_24 = 9 << 1 | U;
      DEBUG(showBitVector(errs(), insn));
      setSlice(insn, 28, 24, bits28_24);
      return decodeInstruction(insn);
    }

    if (op1 == 3 && slice(op2, 6, 4) == 1 && slice(op2, 0, 0) == 0) {
      // A7.7 Advanced SIMD element or structure load/store instructions
      // Inst{27-24} of Thumb = 0b1001
      // Inst{27-24} of ARM   = 0b0100
      DEBUG(showBitVector(errs(), insn));
      setSlice(insn, 27, 24, 4);
      return decodeInstruction(insn);
    }
    // --------- Transform End Marker ---------

    // See, for example, A6.3.7 Load word: Table A6-18 Load word.
    // See A8.6.57 T3, T4 & A8.6.60 T2 and friends for why we morphed the opcode
    // before returning it to our caller.
    if (op1 == 3 && slice(op2, 6, 5) == 0 && slice(op2, 0, 0) == 1
        && slice(insn, 19, 16) == 15)
      return T2Morph2LoadLiteral(decodeThumbInstruction(insn));

    // One last check for NEON/VFP instructions.
    if ((op1 == 1 || op1 == 3) && slice(op2, 6, 6) == 1)
      return decodeInstruction(insn);

    // Fall through.
  }

  return decodeThumbInstruction(insn);
}

static inline bool Thumb2PreloadOpcodeNoPCI(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case ARM::t2PLDi12:   case ARM::t2PLDi8:
  case ARM::t2PLDr:     case ARM::t2PLDs:
  case ARM::t2PLDWi12:  case ARM::t2PLDWi8:
  case ARM::t2PLDWr:    case ARM::t2PLDWs:
  case ARM::t2PLIi12:   case ARM::t2PLIi8:
  case ARM::t2PLIr:     case ARM::t2PLIs:
    return true;
  }
}

static inline unsigned T2Morph2Preload2PCI(unsigned Opcode) {
  switch (Opcode) {
  default:
    return 0;
  case ARM::t2PLDi12:   case ARM::t2PLDi8:
  case ARM::t2PLDr:     case ARM::t2PLDs:
    return ARM::t2PLDpci;
  case ARM::t2PLDWi12:  case ARM::t2PLDWi8:
  case ARM::t2PLDWr:    case ARM::t2PLDWs:
    return ARM::t2PLDWpci;
  case ARM::t2PLIi12:   case ARM::t2PLIi8:
  case ARM::t2PLIr:     case ARM::t2PLIs:
    return ARM::t2PLIpci;
  }
}

//
// Public interface for the disassembler
//

bool ARMDisassembler::getInstruction(MCInst &MI,
                                     uint64_t &Size,
                                     const MemoryObject &Region,
                                     uint64_t Address,
                                     raw_ostream &os) const {
  // The machine instruction.
  uint32_t insn;
  uint8_t bytes[4];

  // We want to read exactly 4 bytes of data.
  if (Region.readBytes(Address, 4, (uint8_t*)bytes, NULL) == -1)
    return false;

  // Encoded as a small-endian 32-bit word in the stream.
  insn = (bytes[3] << 24) |
         (bytes[2] << 16) |
         (bytes[1] <<  8) |
         (bytes[0] <<  0);

  unsigned Opcode = decodeARMInstruction(insn);
  ARMFormat Format = ARMFormats[Opcode];
  Size = 4;

  DEBUG({
      errs() << "Opcode=" << Opcode << " Name=" << ARMUtils::OpcodeName(Opcode)
             << " Format=" << stringForARMFormat(Format) << '(' << (int)Format
             << ")\n";
      showBitVector(errs(), insn);
    });

  ARMBasicMCBuilder *Builder = CreateMCBuilder(Opcode, Format);
  if (!Builder)
    return false;

  if (!Builder->Build(MI, insn))
    return false;

  delete Builder;

  return true;
}

bool ThumbDisassembler::getInstruction(MCInst &MI,
                                       uint64_t &Size,
                                       const MemoryObject &Region,
                                       uint64_t Address,
                                       raw_ostream &os) const {
  // The Thumb instruction stream is a sequence of halhwords.

  // This represents the first halfword as well as the machine instruction
  // passed to decodeThumbInstruction().  For 16-bit Thumb instruction, the top
  // halfword of insn is 0x00 0x00; otherwise, the first halfword is moved to
  // the top half followed by the second halfword.
  unsigned insn = 0;
  // Possible second halfword.
  uint16_t insn1 = 0;

  // A6.1 Thumb instruction set encoding
  //
  // If bits [15:11] of the halfword being decoded take any of the following
  // values, the halfword is the first halfword of a 32-bit instruction:
  // o 0b11101
  // o 0b11110
  // o 0b11111.
  //
  // Otherwise, the halfword is a 16-bit instruction.

  // Read 2 bytes of data first.
  uint8_t bytes[2];
  if (Region.readBytes(Address, 2, (uint8_t*)bytes, NULL) == -1)
    return false;

  // Encoded as a small-endian 16-bit halfword in the stream.
  insn = (bytes[1] << 8) | bytes[0];
  unsigned bits15_11 = slice(insn, 15, 11);
  bool IsThumb2 = false;

  // 32-bit instructions if the bits [15:11] of the halfword matches
  // { 0b11101 /* 0x1D */, 0b11110 /* 0x1E */, ob11111 /* 0x1F */ }.
  if (bits15_11 == 0x1D || bits15_11 == 0x1E || bits15_11 == 0x1F) {
    IsThumb2 = true;
    if (Region.readBytes(Address + 2, 2, (uint8_t*)bytes, NULL) == -1)
      return false;
    // Encoded as a small-endian 16-bit halfword in the stream.
    insn1 = (bytes[1] << 8) | bytes[0];
    insn = (insn << 16 | insn1);
  }

  // The insn could potentially be bit-twiddled in order to be decoded as an ARM
  // NEON/VFP opcode.  In such case, the modified insn is later disassembled as
  // an ARM NEON/VFP instruction.
  //
  // This is a short term solution for lack of encoding bits specified for the
  // Thumb2 NEON/VFP instructions.  The long term solution could be adding some
  // infrastructure to have each instruction support more than one encodings.
  // Which encoding is used would be based on which subtarget the compiler/
  // disassembler is working with at the time.  This would allow the sharing of
  // the NEON patterns between ARM and Thumb2, as well as potential greater
  // sharing between the regular ARM instructions and the 32-bit wide Thumb2
  // instructions as well.
  unsigned Opcode = decodeThumbSideEffect(IsThumb2, insn);

  // A8.6.117/119/120/121.
  // PLD/PLDW/PLI instructions with Rn==15 is transformed to the pci variant.
  if (Thumb2PreloadOpcodeNoPCI(Opcode) && slice(insn, 19, 16) == 15)
    Opcode = T2Morph2Preload2PCI(Opcode);

  ARMFormat Format = ARMFormats[Opcode];
  Size = IsThumb2 ? 4 : 2;

  DEBUG({
      errs() << "Opcode=" << Opcode << " Name=" << ARMUtils::OpcodeName(Opcode)
             << " Format=" << stringForARMFormat(Format) << '(' << (int)Format
             << ")\n";
      showBitVector(errs(), insn);
    });

  ARMBasicMCBuilder *Builder = CreateMCBuilder(Opcode, Format);
  if (!Builder)
    return false;

  Builder->SetSession(const_cast<Session *>(&SO));

  if (!Builder->Build(MI, insn))
    return false;

  delete Builder;

  return true;
}

// A8.6.50
// Valid return values are {1, 2, 3, 4}, with 0 signifying an error condition.
static unsigned short CountITSize(unsigned ITMask) {
  // First count the trailing zeros of the IT mask.
  unsigned TZ = CountTrailingZeros_32(ITMask);
  if (TZ > 3) {
    DEBUG(errs() << "Encoding error: IT Mask '0000'");
    return 0;
  }
  return (4 - TZ);
}

/// Init ITState.  Note that at least one bit is always 1 in mask.
bool Session::InitIT(unsigned short bits7_0) {
  ITCounter = CountITSize(slice(bits7_0, 3, 0));
  if (ITCounter == 0)
    return false;

  // A8.6.50 IT
  unsigned short FirstCond = slice(bits7_0, 7, 4);
  if (FirstCond == 0xF) {
    DEBUG(errs() << "Encoding error: IT FirstCond '1111'");
    return false;
  }
  if (FirstCond == 0xE && ITCounter != 1) {
    DEBUG(errs() << "Encoding error: IT FirstCond '1110' && Mask != '1000'");
    return false;
  }

  ITState = bits7_0;

  return true;
}

/// Update ITState if necessary.
void Session::UpdateIT() {
  assert(ITCounter);
  --ITCounter;
  if (ITCounter == 0)
    ITState = 0;
  else {
    unsigned short NewITState4_0 = slice(ITState, 4, 0) << 1;
    setSlice(ITState, 4, 0, NewITState4_0);
  }
}

static MCDisassembler *createARMDisassembler(const Target &T) {
  return new ARMDisassembler;
}

static MCDisassembler *createThumbDisassembler(const Target &T) {
  return new ThumbDisassembler;
}

extern "C" void LLVMInitializeARMDisassembler() { 
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheARMTarget, 
                                         createARMDisassembler);
  TargetRegistry::RegisterMCDisassembler(TheThumbTarget,
                                         createThumbDisassembler);
}

EDInstInfo *ARMDisassembler::getEDInfo() const {
  return instInfoARM;
}

EDInstInfo *ThumbDisassembler::getEDInfo() const {
  return instInfoARM;
}
