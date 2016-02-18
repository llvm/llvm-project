//===-- AMDGPUDisassembler.cpp - Disassembler for AMDGPU ISA --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This file contains definition for AMDGPU ISA disassembler
//
//===----------------------------------------------------------------------===//

// ToDo: What to do with instruction suffixes (v_mov_b32 vs v_mov_b32_e32)?

#include "AMDGPUDisassembler.h"
#include "AMDGPU.h"
#include "AMDGPURegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"


using namespace llvm;

#define DEBUG_TYPE "amdgpu-disassembler"

typedef llvm::MCDisassembler::DecodeStatus DecodeStatus;


static DecodeStatus DecodeVGPR_32RegisterClass(MCInst &Inst, unsigned Imm,
                                               uint64_t Addr, const void *Decoder) {
  const AMDGPUDisassembler *Dis = 
    static_cast<const AMDGPUDisassembler *>(Decoder);
  return Dis->DecodeVGPR_32RegisterClass(Inst, Imm, Addr);
}

static DecodeStatus DecodeVS_32RegisterClass(MCInst &Inst, unsigned Imm,
                                             uint64_t Addr, const void *Decoder) {
  const AMDGPUDisassembler *Dis = 
    static_cast<const AMDGPUDisassembler *>(Decoder);
  return Dis->DecodeVS_32RegisterClass(Inst, Imm, Addr);
}

static DecodeStatus DecodeVS_64RegisterClass(MCInst &Inst, unsigned Imm, 
                                             uint64_t Addr, const void *Decoder) {
  // ToDo
  return MCDisassembler::Fail;
}

static DecodeStatus DecodeVReg_64RegisterClass(MCInst &Inst, unsigned Imm, 
                                               uint64_t Addr, const void *Decoder) {
  // ToDo
  return MCDisassembler::Fail;
}

static DecodeStatus DecodeVReg_96RegisterClass(MCInst &Inst, unsigned Imm, 
                                               uint64_t Addr, const void *Decoder) {
  // ToDo
  return MCDisassembler::Fail;
}

static DecodeStatus DecodeVReg_128RegisterClass(MCInst &Inst, unsigned Imm, 
                                                uint64_t Addr, const void *Decoder) {
  // ToDo
  return MCDisassembler::Fail;
}

static DecodeStatus DecodeSReg_32RegisterClass(MCInst &Inst, unsigned Imm, 
                                               uint64_t Addr, const void *Decoder) {
  // ToDo
  return MCDisassembler::Fail;
}

static DecodeStatus DecodeSReg_64RegisterClass(MCInst &Inst, unsigned Imm, 
                                               uint64_t Addr, const void *Decoder) {
  // ToDo
  return MCDisassembler::Fail;
}

static DecodeStatus DecodeSReg_128RegisterClass(MCInst &Inst, unsigned Imm, 
                                                uint64_t Addr, const void *Decoder) {
  // ToDo
  return MCDisassembler::Fail;
}

static DecodeStatus DecodeSReg_256RegisterClass(MCInst &Inst, unsigned Imm, 
                                                uint64_t Addr, const void *Decoder) {
  // ToDo
  return MCDisassembler::Fail;
}


#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"
#undef GET_SUBTARGETINFO_ENUM

#include "AMDGPUGenDisassemblerTables.inc"

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

DecodeStatus AMDGPUDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                ArrayRef<uint8_t> Bytes,
                                                uint64_t Address, 
                                                raw_ostream &WS,
                                                raw_ostream &CS) const {
  CommentStream = &CS;

  // ToDo: AMDGPUDisassembler supports only VI ISA.
  assert(AMDGPU::isVI(STI) && "Can disassemble only VI ISA.");

  // Try decode 32-bit instruction
  if (Bytes.size() < 4) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  uint32_t Insn =
      (Bytes[3] << 24) | (Bytes[2] << 16) | (Bytes[1] << 8) | (Bytes[0] << 0);

  // Calling the auto-generated decoder function.
  DecodeStatus Result =
      decodeInstruction(DecoderTableVI32, MI, Insn, Address, this, STI);
  if (Result != MCDisassembler::Success) {
      Size = 0;
      return MCDisassembler::Fail;
  }
  Size = 4;

  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::DecodeLitFloat(unsigned Imm, uint32_t& F) const {
  // ToDo: case 248: 1/(2*PI) - is allowed only on VI
  // ToDo: AMDGPUInstPrinter does not support 1/(2*PI). It consider 1/(2*PI) as 
  // literal constant.
  switch(Imm) {
  case 240: F = FloatToBits(0.5f); return MCDisassembler::Success;
  case 241: F = FloatToBits(-0.5f); return MCDisassembler::Success;
  case 242: F = FloatToBits(1.0f); return MCDisassembler::Success;
  case 243: F = FloatToBits(-1.0f); return MCDisassembler::Success;
  case 244: F = FloatToBits(2.0f); return MCDisassembler::Success;
  case 245: F = FloatToBits(-2.0f); return MCDisassembler::Success;
  case 246: F = FloatToBits(4.0f); return MCDisassembler::Success;
  case 247: F = FloatToBits(-4.0f); return MCDisassembler::Success;
  case 248: F = 0x3e22f983; return MCDisassembler::Success; // 1/(2*PI)
  default: return MCDisassembler::Fail;
  }
}

DecodeStatus AMDGPUDisassembler::DecodeLitInteger(unsigned Imm, 
                                                  int64_t& I) const {
  if ((Imm >= 128) && (Imm <= 192)) {
    I = Imm - 128;
    return MCDisassembler::Success;
  } else if ((Imm >= 193) && (Imm <= 208)) {
    I = 192 - Imm;
    return MCDisassembler::Success;
  }
  return MCDisassembler::Fail;
}

DecodeStatus AMDGPUDisassembler::DecodeVgprRegister(unsigned Val, 
                                                    unsigned& RegID) const {
  if (Val > 255) {
    return MCDisassembler::Fail;
  }
  RegID = AMDGPUMCRegisterClasses[AMDGPU::VGPR_32RegClassID].getRegister(Val);
  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::DecodeSgprRegister(unsigned Val, 
                                                    unsigned& RegID) const {
  // ToDo: SI/CI have 104 SGPRs, VI - 102
  if (Val > 101) {
    return MCDisassembler::Fail;
  }
  RegID = AMDGPUMCRegisterClasses[AMDGPU::SGPR_32RegClassID].getRegister(Val);
  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::DecodeSrcRegister(unsigned Val, 
                                                   unsigned& RegID) const {
  // ToDo: deal with out-of range registers  
  using namespace AMDGPU;
  if (Val <= 101) {
    return DecodeSgprRegister(Val, RegID);
  } else if ((Val >= 256) && (Val <= 511)) {
    return DecodeVgprRegister(Val - 256, RegID);
  } else {
    switch(Val) {
    case 102: RegID = getMCReg(FLAT_SCR_LO, STI); return MCDisassembler::Success;
    case 103: RegID = getMCReg(FLAT_SCR_HI, STI); return MCDisassembler::Success;
    // ToDo: no support for xnack_mask_lo/_hi register
    case 104:
    case 105: return MCDisassembler::Fail;
    case 106: RegID = getMCReg(VCC_LO, STI); return MCDisassembler::Success;
    case 107: RegID = getMCReg(VCC_HI, STI); return MCDisassembler::Success;
    // ToDo: no support for tba_lo/_hi register
    case 108:
    case 109: return MCDisassembler::Fail;
    // ToDo: no support for tma_lo/_hi register
    case 110:
    case 111: return MCDisassembler::Fail;
    // ToDo: no support for ttmp[0:11] register
    case 112:
    case 113:
    case 114:
    case 115:
    case 116:
    case 117:
    case 118:
    case 119:
    case 120:
    case 121:
    case 122:
    case 123: return MCDisassembler::Fail;
    case 124: RegID = getMCReg(M0, STI); return MCDisassembler::Success;
    case 126: RegID = getMCReg(EXEC_LO, STI); return MCDisassembler::Success;
    case 127: RegID = getMCReg(EXEC_HI, STI); return MCDisassembler::Success;
    // ToDo: no support for vccz register
    case 251: return MCDisassembler::Fail;
    // ToDo: no support for execz register
    case 252: return MCDisassembler::Fail;
    case 253: RegID = getMCReg(SCC, STI); return MCDisassembler::Success;
    default: return MCDisassembler::Fail;
    }
  }
  return MCDisassembler::Fail;
}

DecodeStatus AMDGPUDisassembler::DecodeVGPR_32RegisterClass(llvm::MCInst &Inst, 
                                                            unsigned Imm, 
                                                            uint64_t Addr) const {
  unsigned RegID;
  if (DecodeVgprRegister(Imm, RegID) == MCDisassembler::Success) {
    Inst.addOperand(MCOperand::createReg(RegID));
    return MCDisassembler::Success;
  }
  return MCDisassembler::Fail;
}

DecodeStatus AMDGPUDisassembler::DecodeVS_32RegisterClass(MCInst &Inst, 
                                                          unsigned Imm, 
                                                          uint64_t Addr) const {
  // ToDo: different opcodes allow different formats og this operands
  if ((Imm >= 128) && (Imm <= 208)) {
    // immediate integer
    int64_t Val;
    if (DecodeLitInteger(Imm, Val) == MCDisassembler::Success) {
      Inst.addOperand(MCOperand::createImm(Val));
      return MCDisassembler::Success;
    }
  } else if ((Imm >= 240) && (Imm <= 248)) {
    // immediate float
    uint32_t Val;
    if (DecodeLitFloat(Imm, Val) == MCDisassembler::Success) {
      Inst.addOperand(MCOperand::createImm(Val));
      return MCDisassembler::Success;
    }
  } else if (Imm == 254) {
    // LDS direct
    // ToDo: implement LDS direct read
  } else if (Imm == 255) {
    // literal constant
  } else if ((Imm == 125) || 
             ((Imm >= 209) && (Imm <= 239)) || 
             (Imm == 249) || 
             (Imm == 250) || 
             (Imm >= 512)) {
    // reserved
    return MCDisassembler::Fail;
  } else {
    // register
    unsigned RegID;
    if (DecodeSrcRegister(Imm, RegID) == MCDisassembler::Success) {
      Inst.addOperand(MCOperand::createReg(RegID));
      return MCDisassembler::Success;
    }
  }
  return MCDisassembler::Fail;
}

static MCDisassembler *createAMDGPUDisassembler(const Target &T,
                                                const MCSubtargetInfo &STI,
                                                MCContext &Ctx) {
  return new AMDGPUDisassembler(STI, Ctx);
}

extern "C" void LLVMInitializeAMDGPUDisassembler() {
  TargetRegistry::RegisterMCDisassembler(TheGCNTarget, createAMDGPUDisassembler);
}
