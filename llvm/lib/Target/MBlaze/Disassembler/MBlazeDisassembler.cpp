//===-- MBlazeDisassembler.cpp - Disassembler for MicroBlaze  -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the MBlaze Disassembler. It contains code to translate
// the data produced by the decoder into MCInsts.
//
//===----------------------------------------------------------------------===//

#include "MBlaze.h"
#include "MBlazeDisassembler.h"

#include "llvm/MC/EDInstInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

// #include "MBlazeGenDecoderTables.inc"
// #include "MBlazeGenRegisterNames.inc"
#include "MBlazeGenEDInfo.inc"

namespace llvm {
extern const MCInstrDesc MBlazeInsts[];
}

using namespace llvm;

const uint16_t UNSUPPORTED = -1;

static const uint16_t mblazeBinary2Opcode[] = {
  MBlaze::ADD,   MBlaze::RSUB,   MBlaze::ADDC,   MBlaze::RSUBC,   //00,01,02,03
  MBlaze::ADDK,  MBlaze::RSUBK,  MBlaze::ADDKC,  MBlaze::RSUBKC,  //04,05,06,07
  MBlaze::ADDI,  MBlaze::RSUBI,  MBlaze::ADDIC,  MBlaze::RSUBIC,  //08,09,0A,0B
  MBlaze::ADDIK, MBlaze::RSUBIK, MBlaze::ADDIKC, MBlaze::RSUBIKC, //0C,0D,0E,0F

  MBlaze::MUL,   MBlaze::BSRL,   MBlaze::IDIV,   MBlaze::GETD,    //10,11,12,13
  UNSUPPORTED,   UNSUPPORTED,    MBlaze::FADD,   UNSUPPORTED,     //14,15,16,17
  MBlaze::MULI,  MBlaze::BSRLI,  UNSUPPORTED,    MBlaze::GET,     //18,19,1A,1B
  UNSUPPORTED,   UNSUPPORTED,    UNSUPPORTED,    UNSUPPORTED,     //1C,1D,1E,1F

  MBlaze::OR,    MBlaze::AND,    MBlaze::XOR,    MBlaze::ANDN,    //20,21,22,23
  MBlaze::SEXT8, MBlaze::MFS,    MBlaze::BR,     MBlaze::BEQ,     //24,25,26,27
  MBlaze::ORI,   MBlaze::ANDI,   MBlaze::XORI,   MBlaze::ANDNI,   //28,29,2A,2B
  MBlaze::IMM,   MBlaze::RTSD,   MBlaze::BRI,    MBlaze::BEQI,    //2C,2D,2E,2F

  MBlaze::LBU,   MBlaze::LHU,    MBlaze::LW,     UNSUPPORTED,     //30,31,32,33
  MBlaze::SB,    MBlaze::SH,     MBlaze::SW,     UNSUPPORTED,     //34,35,36,37
  MBlaze::LBUI,  MBlaze::LHUI,   MBlaze::LWI,    UNSUPPORTED,     //38,39,3A,3B
  MBlaze::SBI,   MBlaze::SHI,    MBlaze::SWI,    UNSUPPORTED,     //3C,3D,3E,3F
};

static unsigned getRD(uint32_t insn) {
  if (!isMBlazeRegister((insn>>21)&0x1F))
    return UNSUPPORTED;
  return getMBlazeRegisterFromNumbering((insn>>21)&0x1F);
}

static unsigned getRA(uint32_t insn) {
  if (!getMBlazeRegisterFromNumbering((insn>>16)&0x1F))
    return UNSUPPORTED;
  return getMBlazeRegisterFromNumbering((insn>>16)&0x1F);
}

static unsigned getRB(uint32_t insn) {
  if (!getMBlazeRegisterFromNumbering((insn>>11)&0x1F))
    return UNSUPPORTED;
  return getMBlazeRegisterFromNumbering((insn>>11)&0x1F);
}

static int64_t getRS(uint32_t insn) {
  if (!isSpecialMBlazeRegister(insn&0x3FFF))
    return UNSUPPORTED;
  return getSpecialMBlazeRegisterFromNumbering(insn&0x3FFF);
}

static int64_t getIMM(uint32_t insn) {
    int16_t val = (insn & 0xFFFF);
    return val;
}

static int64_t getSHT(uint32_t insn) {
    int16_t val = (insn & 0x1F);
    return val;
}

static unsigned getFLAGS(int32_t insn) {
    return (insn & 0x7FF);
}

static int64_t getFSL(uint32_t insn) {
    int16_t val = (insn & 0xF);
    return val;
}

static unsigned decodeMUL(uint32_t insn) {
    switch (getFLAGS(insn)) {
    default: return UNSUPPORTED;
    case 0:  return MBlaze::MUL;
    case 1:  return MBlaze::MULH;
    case 2:  return MBlaze::MULHSU;
    case 3:  return MBlaze::MULHU;
    }
}

static unsigned decodeSEXT(uint32_t insn) {
    switch (insn&0x7FF) {
    default:   return UNSUPPORTED;
    case 0x60: return MBlaze::SEXT8;
    case 0x68: return MBlaze::WIC;
    case 0x64: return MBlaze::WDC;
    case 0x66: return MBlaze::WDCC;
    case 0x74: return MBlaze::WDCF;
    case 0x61: return MBlaze::SEXT16;
    case 0x41: return MBlaze::SRL;
    case 0x21: return MBlaze::SRC;
    case 0x01: return MBlaze::SRA;
    case 0xE0: return MBlaze::CLZ;
    }
}

static unsigned decodeBEQ(uint32_t insn) {
    switch ((insn>>21)&0x1F) {
    default:    return UNSUPPORTED;
    case 0x00:  return MBlaze::BEQ;
    case 0x10:  return MBlaze::BEQD;
    case 0x05:  return MBlaze::BGE;
    case 0x15:  return MBlaze::BGED;
    case 0x04:  return MBlaze::BGT;
    case 0x14:  return MBlaze::BGTD;
    case 0x03:  return MBlaze::BLE;
    case 0x13:  return MBlaze::BLED;
    case 0x02:  return MBlaze::BLT;
    case 0x12:  return MBlaze::BLTD;
    case 0x01:  return MBlaze::BNE;
    case 0x11:  return MBlaze::BNED;
    }
}

static unsigned decodeBEQI(uint32_t insn) {
    switch ((insn>>21)&0x1F) {
    default:    return UNSUPPORTED;
    case 0x00:  return MBlaze::BEQI;
    case 0x10:  return MBlaze::BEQID;
    case 0x05:  return MBlaze::BGEI;
    case 0x15:  return MBlaze::BGEID;
    case 0x04:  return MBlaze::BGTI;
    case 0x14:  return MBlaze::BGTID;
    case 0x03:  return MBlaze::BLEI;
    case 0x13:  return MBlaze::BLEID;
    case 0x02:  return MBlaze::BLTI;
    case 0x12:  return MBlaze::BLTID;
    case 0x01:  return MBlaze::BNEI;
    case 0x11:  return MBlaze::BNEID;
    }
}

static unsigned decodeBR(uint32_t insn) {
    switch ((insn>>16)&0x1F) {
    default:   return UNSUPPORTED;
    case 0x00: return MBlaze::BR;
    case 0x08: return MBlaze::BRA;
    case 0x0C: return MBlaze::BRK;
    case 0x10: return MBlaze::BRD;
    case 0x14: return MBlaze::BRLD;
    case 0x18: return MBlaze::BRAD;
    case 0x1C: return MBlaze::BRALD;
    }
}

static unsigned decodeBRI(uint32_t insn) {
    switch (insn&0x3FFFFFF) {
    default:        break;
    case 0x0020004: return MBlaze::IDMEMBAR;
    case 0x0220004: return MBlaze::DMEMBAR;
    case 0x0420004: return MBlaze::IMEMBAR;
    }

    switch ((insn>>16)&0x1F) {
    default:   return UNSUPPORTED;
    case 0x00: return MBlaze::BRI;
    case 0x08: return MBlaze::BRAI;
    case 0x0C: return MBlaze::BRKI;
    case 0x10: return MBlaze::BRID;
    case 0x14: return MBlaze::BRLID;
    case 0x18: return MBlaze::BRAID;
    case 0x1C: return MBlaze::BRALID;
    }
}

static unsigned decodeBSRL(uint32_t insn) {
    switch ((insn>>9)&0x3) {
    default:  return UNSUPPORTED;
    case 0x2: return MBlaze::BSLL;
    case 0x1: return MBlaze::BSRA;
    case 0x0: return MBlaze::BSRL;
    }
}

static unsigned decodeBSRLI(uint32_t insn) {
    switch ((insn>>9)&0x3) {
    default:  return UNSUPPORTED;
    case 0x2: return MBlaze::BSLLI;
    case 0x1: return MBlaze::BSRAI;
    case 0x0: return MBlaze::BSRLI;
    }
}

static unsigned decodeRSUBK(uint32_t insn) {
    switch (getFLAGS(insn)) {
    default:  return UNSUPPORTED;
    case 0x0: return MBlaze::RSUBK;
    case 0x1: return MBlaze::CMP;
    case 0x3: return MBlaze::CMPU;
    }
}

static unsigned decodeFADD(uint32_t insn) {
    switch (getFLAGS(insn)) {
    default:    return UNSUPPORTED;
    case 0x000: return MBlaze::FADD;
    case 0x080: return MBlaze::FRSUB;
    case 0x100: return MBlaze::FMUL;
    case 0x180: return MBlaze::FDIV;
    case 0x200: return MBlaze::FCMP_UN;
    case 0x210: return MBlaze::FCMP_LT;
    case 0x220: return MBlaze::FCMP_EQ;
    case 0x230: return MBlaze::FCMP_LE;
    case 0x240: return MBlaze::FCMP_GT;
    case 0x250: return MBlaze::FCMP_NE;
    case 0x260: return MBlaze::FCMP_GE;
    case 0x280: return MBlaze::FLT;
    case 0x300: return MBlaze::FINT;
    case 0x380: return MBlaze::FSQRT;
    }
}

static unsigned decodeGET(uint32_t insn) {
    switch ((insn>>10)&0x3F) {
    default:   return UNSUPPORTED;
    case 0x00: return MBlaze::GET;
    case 0x01: return MBlaze::EGET;
    case 0x02: return MBlaze::AGET;
    case 0x03: return MBlaze::EAGET;
    case 0x04: return MBlaze::TGET;
    case 0x05: return MBlaze::TEGET;
    case 0x06: return MBlaze::TAGET;
    case 0x07: return MBlaze::TEAGET;
    case 0x08: return MBlaze::CGET;
    case 0x09: return MBlaze::ECGET;
    case 0x0A: return MBlaze::CAGET;
    case 0x0B: return MBlaze::ECAGET;
    case 0x0C: return MBlaze::TCGET;
    case 0x0D: return MBlaze::TECGET;
    case 0x0E: return MBlaze::TCAGET;
    case 0x0F: return MBlaze::TECAGET;
    case 0x10: return MBlaze::NGET;
    case 0x11: return MBlaze::NEGET;
    case 0x12: return MBlaze::NAGET;
    case 0x13: return MBlaze::NEAGET;
    case 0x14: return MBlaze::TNGET;
    case 0x15: return MBlaze::TNEGET;
    case 0x16: return MBlaze::TNAGET;
    case 0x17: return MBlaze::TNEAGET;
    case 0x18: return MBlaze::NCGET;
    case 0x19: return MBlaze::NECGET;
    case 0x1A: return MBlaze::NCAGET;
    case 0x1B: return MBlaze::NECAGET;
    case 0x1C: return MBlaze::TNCGET;
    case 0x1D: return MBlaze::TNECGET;
    case 0x1E: return MBlaze::TNCAGET;
    case 0x1F: return MBlaze::TNECAGET;
    case 0x20: return MBlaze::PUT;
    case 0x22: return MBlaze::APUT;
    case 0x24: return MBlaze::TPUT;
    case 0x26: return MBlaze::TAPUT;
    case 0x28: return MBlaze::CPUT;
    case 0x2A: return MBlaze::CAPUT;
    case 0x2C: return MBlaze::TCPUT;
    case 0x2E: return MBlaze::TCAPUT;
    case 0x30: return MBlaze::NPUT;
    case 0x32: return MBlaze::NAPUT;
    case 0x34: return MBlaze::TNPUT;
    case 0x36: return MBlaze::TNAPUT;
    case 0x38: return MBlaze::NCPUT;
    case 0x3A: return MBlaze::NCAPUT;
    case 0x3C: return MBlaze::TNCPUT;
    case 0x3E: return MBlaze::TNCAPUT;
    }
}

static unsigned decodeGETD(uint32_t insn) {
    switch ((insn>>5)&0x3F) {
    default:   return UNSUPPORTED;
    case 0x00: return MBlaze::GETD;
    case 0x01: return MBlaze::EGETD;
    case 0x02: return MBlaze::AGETD;
    case 0x03: return MBlaze::EAGETD;
    case 0x04: return MBlaze::TGETD;
    case 0x05: return MBlaze::TEGETD;
    case 0x06: return MBlaze::TAGETD;
    case 0x07: return MBlaze::TEAGETD;
    case 0x08: return MBlaze::CGETD;
    case 0x09: return MBlaze::ECGETD;
    case 0x0A: return MBlaze::CAGETD;
    case 0x0B: return MBlaze::ECAGETD;
    case 0x0C: return MBlaze::TCGETD;
    case 0x0D: return MBlaze::TECGETD;
    case 0x0E: return MBlaze::TCAGETD;
    case 0x0F: return MBlaze::TECAGETD;
    case 0x10: return MBlaze::NGETD;
    case 0x11: return MBlaze::NEGETD;
    case 0x12: return MBlaze::NAGETD;
    case 0x13: return MBlaze::NEAGETD;
    case 0x14: return MBlaze::TNGETD;
    case 0x15: return MBlaze::TNEGETD;
    case 0x16: return MBlaze::TNAGETD;
    case 0x17: return MBlaze::TNEAGETD;
    case 0x18: return MBlaze::NCGETD;
    case 0x19: return MBlaze::NECGETD;
    case 0x1A: return MBlaze::NCAGETD;
    case 0x1B: return MBlaze::NECAGETD;
    case 0x1C: return MBlaze::TNCGETD;
    case 0x1D: return MBlaze::TNECGETD;
    case 0x1E: return MBlaze::TNCAGETD;
    case 0x1F: return MBlaze::TNECAGETD;
    case 0x20: return MBlaze::PUTD;
    case 0x22: return MBlaze::APUTD;
    case 0x24: return MBlaze::TPUTD;
    case 0x26: return MBlaze::TAPUTD;
    case 0x28: return MBlaze::CPUTD;
    case 0x2A: return MBlaze::CAPUTD;
    case 0x2C: return MBlaze::TCPUTD;
    case 0x2E: return MBlaze::TCAPUTD;
    case 0x30: return MBlaze::NPUTD;
    case 0x32: return MBlaze::NAPUTD;
    case 0x34: return MBlaze::TNPUTD;
    case 0x36: return MBlaze::TNAPUTD;
    case 0x38: return MBlaze::NCPUTD;
    case 0x3A: return MBlaze::NCAPUTD;
    case 0x3C: return MBlaze::TNCPUTD;
    case 0x3E: return MBlaze::TNCAPUTD;
    }
}

static unsigned decodeIDIV(uint32_t insn) {
    switch (insn&0x3) {
    default:  return UNSUPPORTED;
    case 0x0: return MBlaze::IDIV;
    case 0x2: return MBlaze::IDIVU;
    }
}

static unsigned decodeLBU(uint32_t insn) {
    switch ((insn>>9)&0x1) {
    default:  return UNSUPPORTED;
    case 0x0: return MBlaze::LBU;
    case 0x1: return MBlaze::LBUR;
    }
}

static unsigned decodeLHU(uint32_t insn) {
    switch ((insn>>9)&0x1) {
    default:  return UNSUPPORTED;
    case 0x0: return MBlaze::LHU;
    case 0x1: return MBlaze::LHUR;
    }
}

static unsigned decodeLW(uint32_t insn) {
    switch ((insn>>9)&0x3) {
    default:  return UNSUPPORTED;
    case 0x0: return MBlaze::LW;
    case 0x1: return MBlaze::LWR;
    case 0x2: return MBlaze::LWX;
    }
}

static unsigned decodeSB(uint32_t insn) {
    switch ((insn>>9)&0x1) {
    default:  return UNSUPPORTED;
    case 0x0: return MBlaze::SB;
    case 0x1: return MBlaze::SBR;
    }
}

static unsigned decodeSH(uint32_t insn) {
    switch ((insn>>9)&0x1) {
    default:  return UNSUPPORTED;
    case 0x0: return MBlaze::SH;
    case 0x1: return MBlaze::SHR;
    }
}

static unsigned decodeSW(uint32_t insn) {
    switch ((insn>>9)&0x3) {
    default:  return UNSUPPORTED;
    case 0x0: return MBlaze::SW;
    case 0x1: return MBlaze::SWR;
    case 0x2: return MBlaze::SWX;
    }
}

static unsigned decodeMFS(uint32_t insn) {
    switch ((insn>>15)&0x1) {
    default:   return UNSUPPORTED;
    case 0x0:
      switch ((insn>>16)&0x1) {
      default:   return UNSUPPORTED;
      case 0x0: return MBlaze::MSRSET;
      case 0x1: return MBlaze::MSRCLR;
      }
    case 0x1:
      switch ((insn>>14)&0x1) {
      default:   return UNSUPPORTED;
      case 0x0: return MBlaze::MFS;
      case 0x1: return MBlaze::MTS;
      }
    }
}

static unsigned decodeOR(uint32_t insn) {
    switch (getFLAGS(insn)) {
    default:    return UNSUPPORTED;
    case 0x000: return MBlaze::OR;
    case 0x400: return MBlaze::PCMPBF;
    }
}

static unsigned decodeXOR(uint32_t insn) {
    switch (getFLAGS(insn)) {
    default:    return UNSUPPORTED;
    case 0x000: return MBlaze::XOR;
    case 0x400: return MBlaze::PCMPEQ;
    }
}

static unsigned decodeANDN(uint32_t insn) {
    switch (getFLAGS(insn)) {
    default:    return UNSUPPORTED;
    case 0x000: return MBlaze::ANDN;
    case 0x400: return MBlaze::PCMPNE;
    }
}

static unsigned decodeRTSD(uint32_t insn) {
    switch ((insn>>21)&0x1F) {
    default:   return UNSUPPORTED;
    case 0x10: return MBlaze::RTSD;
    case 0x11: return MBlaze::RTID;
    case 0x12: return MBlaze::RTBD;
    case 0x14: return MBlaze::RTED;
    }
}

static unsigned getOPCODE(uint32_t insn) {
  unsigned opcode = mblazeBinary2Opcode[ (insn>>26)&0x3F ];
  switch (opcode) {
  case MBlaze::MUL:     return decodeMUL(insn);
  case MBlaze::SEXT8:   return decodeSEXT(insn);
  case MBlaze::BEQ:     return decodeBEQ(insn);
  case MBlaze::BEQI:    return decodeBEQI(insn);
  case MBlaze::BR:      return decodeBR(insn);
  case MBlaze::BRI:     return decodeBRI(insn);
  case MBlaze::BSRL:    return decodeBSRL(insn);
  case MBlaze::BSRLI:   return decodeBSRLI(insn);
  case MBlaze::RSUBK:   return decodeRSUBK(insn);
  case MBlaze::FADD:    return decodeFADD(insn);
  case MBlaze::GET:     return decodeGET(insn);
  case MBlaze::GETD:    return decodeGETD(insn);
  case MBlaze::IDIV:    return decodeIDIV(insn);
  case MBlaze::LBU:     return decodeLBU(insn);
  case MBlaze::LHU:     return decodeLHU(insn);
  case MBlaze::LW:      return decodeLW(insn);
  case MBlaze::SB:      return decodeSB(insn);
  case MBlaze::SH:      return decodeSH(insn);
  case MBlaze::SW:      return decodeSW(insn);
  case MBlaze::MFS:     return decodeMFS(insn);
  case MBlaze::OR:      return decodeOR(insn);
  case MBlaze::XOR:     return decodeXOR(insn);
  case MBlaze::ANDN:    return decodeANDN(insn);
  case MBlaze::RTSD:    return decodeRTSD(insn);
  default:              return opcode;
  }
}

const EDInstInfo *MBlazeDisassembler::getEDInfo() const {
  return instInfoMBlaze;
}

//
// Public interface for the disassembler
//

MCDisassembler::DecodeStatus MBlazeDisassembler::getInstruction(MCInst &instr,
                                        uint64_t &size,
                                        const MemoryObject &region,
                                        uint64_t address,
                                        raw_ostream &vStream,
                                        raw_ostream &cStream) const {
  // The machine instruction.
  uint32_t insn;
  uint64_t read;
  uint8_t bytes[4];

  // By default we consume 1 byte on failure
  size = 1;

  // We want to read exactly 4 bytes of data.
  if (region.readBytes(address, 4, (uint8_t*)bytes, &read) == -1 || read < 4)
    return Fail;

  // Encoded as a big-endian 32-bit word in the stream.
  insn = (bytes[0]<<24) | (bytes[1]<<16) | (bytes[2]<< 8) | (bytes[3]<<0);

  // Get the MCInst opcode from the binary instruction and make sure
  // that it is a valid instruction.
  unsigned opcode = getOPCODE(insn);
  if (opcode == UNSUPPORTED)
    return Fail;

  instr.setOpcode(opcode);

  unsigned RD = getRD(insn);
  unsigned RA = getRA(insn);
  unsigned RB = getRB(insn);
  unsigned RS = getRS(insn);

  uint64_t tsFlags = MBlazeInsts[opcode].TSFlags;
  switch ((tsFlags & MBlazeII::FormMask)) {
  default: 
    return Fail;

  case MBlazeII::FC:
    break;

  case MBlazeII::FRRRR:
    if (RD == UNSUPPORTED || RA == UNSUPPORTED || RB == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateReg(RB));
    instr.addOperand(MCOperand::CreateReg(RA));
    break;

  case MBlazeII::FRRR:
    if (RD == UNSUPPORTED || RA == UNSUPPORTED || RB == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateReg(RA));
    instr.addOperand(MCOperand::CreateReg(RB));
    break;

  case MBlazeII::FRR:
    if (RD == UNSUPPORTED || RA == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateReg(RA));
    break;

  case MBlazeII::FRI:
    switch (opcode) {
    default: 
      return Fail;
    case MBlaze::MFS:
      if (RD == UNSUPPORTED)
        return Fail;
      instr.addOperand(MCOperand::CreateReg(RD));
      instr.addOperand(MCOperand::CreateImm(insn&0x3FFF));
      break;
    case MBlaze::MTS:
      if (RA == UNSUPPORTED)
        return Fail;
      instr.addOperand(MCOperand::CreateImm(insn&0x3FFF));
      instr.addOperand(MCOperand::CreateReg(RA));
      break;
    case MBlaze::MSRSET:
    case MBlaze::MSRCLR:
      if (RD == UNSUPPORTED)
        return Fail;
      instr.addOperand(MCOperand::CreateReg(RD));
      instr.addOperand(MCOperand::CreateImm(insn&0x7FFF));
      break;
    }
    break;

  case MBlazeII::FRRI:
    if (RD == UNSUPPORTED || RA == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateReg(RA));
    switch (opcode) {
    default:
      instr.addOperand(MCOperand::CreateImm(getIMM(insn)));
      break;
    case MBlaze::BSRLI:
    case MBlaze::BSRAI:
    case MBlaze::BSLLI:
      instr.addOperand(MCOperand::CreateImm(insn&0x1F));
      break;
    }
    break;

  case MBlazeII::FCRR:
    if (RA == UNSUPPORTED || RB == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RA));
    instr.addOperand(MCOperand::CreateReg(RB));
    break;

  case MBlazeII::FCRI:
    if (RA == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RA));
    instr.addOperand(MCOperand::CreateImm(getIMM(insn)));
    break;

  case MBlazeII::FRCR:
    if (RD == UNSUPPORTED || RB == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateReg(RB));
    break;

  case MBlazeII::FRCI:
    if (RD == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateImm(getIMM(insn)));
    break;

  case MBlazeII::FCCR:
    if (RB == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RB));
    break;

  case MBlazeII::FCCI:
    instr.addOperand(MCOperand::CreateImm(getIMM(insn)));
    break;

  case MBlazeII::FRRCI:
    if (RD == UNSUPPORTED || RA == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateReg(RA));
    instr.addOperand(MCOperand::CreateImm(getSHT(insn)));
    break;

  case MBlazeII::FRRC:
    if (RD == UNSUPPORTED || RA == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateReg(RA));
    break;

  case MBlazeII::FRCX:
    if (RD == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateImm(getFSL(insn)));
    break;

  case MBlazeII::FRCS:
    if (RD == UNSUPPORTED || RS == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateReg(RS));
    break;

  case MBlazeII::FCRCS:
    if (RS == UNSUPPORTED || RA == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RS));
    instr.addOperand(MCOperand::CreateReg(RA));
    break;

  case MBlazeII::FCRCX:
    if (RA == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RA));
    instr.addOperand(MCOperand::CreateImm(getFSL(insn)));
    break;

  case MBlazeII::FCX:
    instr.addOperand(MCOperand::CreateImm(getFSL(insn)));
    break;

  case MBlazeII::FCR:
    if (RB == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RB));
    break;

  case MBlazeII::FRIR:
    if (RD == UNSUPPORTED || RA == UNSUPPORTED)
      return Fail;
    instr.addOperand(MCOperand::CreateReg(RD));
    instr.addOperand(MCOperand::CreateImm(getIMM(insn)));
    instr.addOperand(MCOperand::CreateReg(RA));
    break;
  }

  // We always consume 4 bytes of data on success
  size = 4;

  return Success;
}

static MCDisassembler *createMBlazeDisassembler(const Target &T,
                                                const MCSubtargetInfo &STI) {
  return new MBlazeDisassembler(STI);
}

extern "C" void LLVMInitializeMBlazeDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheMBlazeTarget,
                                         createMBlazeDisassembler);
}
