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
  const AMDGPUDisassembler *Dis =
    static_cast<const AMDGPUDisassembler *>(Decoder);
  return Dis->DecodeVS_64RegisterClass(Inst, Imm, Addr);
}

static DecodeStatus DecodeVReg_64RegisterClass(MCInst &Inst, unsigned Imm,
                                               uint64_t Addr, const void *Decoder) {
  const AMDGPUDisassembler *Dis =
    static_cast<const AMDGPUDisassembler *>(Decoder);
  return Dis->DecodeVReg_64RegisterClass(Inst, Imm, Addr);
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

  HasLiteral = false;
  this->Bytes = Bytes;

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
  if (HasLiteral == true) {
    Size = 8;
    HasLiteral = false;
  } else {
    Size = 4;
  }

  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::DecodeImmedFloat(unsigned Imm, uint32_t &F) const {
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

DecodeStatus AMDGPUDisassembler::DecodeImmedDouble(unsigned Imm, uint64_t &D) const {
  switch(Imm) {
  case 240: D = DoubleToBits(0.5); return MCDisassembler::Success;
  case 241: D = DoubleToBits(-0.5); return MCDisassembler::Success;
  case 242: D = DoubleToBits(1.0); return MCDisassembler::Success;
  case 243: D = DoubleToBits(-1.0); return MCDisassembler::Success;
  case 244: D = DoubleToBits(2.0); return MCDisassembler::Success;
  case 245: D = DoubleToBits(-2.0); return MCDisassembler::Success;
  case 246: D = DoubleToBits(4.0); return MCDisassembler::Success;
  case 247: D = DoubleToBits(-4.0); return MCDisassembler::Success;
  case 248: D = 0x3fc45f306dc9c882; return MCDisassembler::Success; // 1/(2*PI)
  default: return MCDisassembler::Fail;
  }
}

DecodeStatus AMDGPUDisassembler::DecodeImmedInteger(unsigned Imm,
                                                    int64_t &I) const {
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
                                                    unsigned &RegID,
                                                    unsigned Size) const {
  if (Val > (256 - Size / 32)) {
    return MCDisassembler::Fail;
  }
  unsigned RegClassID;
  switch (Size) {
  case 32: RegClassID = AMDGPU::VGPR_32RegClassID; break;
  case 64: RegClassID = AMDGPU::VReg_64RegClassID; break;
  case 96: RegClassID = AMDGPU::VReg_96RegClassID; break;
  case 128: RegClassID = AMDGPU::VReg_128RegClassID; break;
  case 256: RegClassID = AMDGPU::VReg_256RegClassID; break;
  case 512: RegClassID = AMDGPU::VReg_512RegClassID; break;
  default:
    return MCDisassembler::Fail;
  }

  RegID = AMDGPUMCRegisterClasses[RegClassID].getRegister(Val);
  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::DecodeSgprRegister(unsigned Val,
                                                    unsigned &RegID,
                                                    unsigned Size) const {
  // ToDo: SI/CI have 104 SGPRs, VI - 102
  unsigned RegClassID;

  switch (Size) {
  case 32:
    if (Val > 101) {
      return MCDisassembler::Fail;
    }
    RegClassID = AMDGPU::SGPR_32RegClassID;
    break;
  case 64:
    if ((Val % 2 != 0) || (Val > 100)) {
      return MCDisassembler::Fail;
    }
    Val /= 2;
    RegClassID = AMDGPU::SGPR_64RegClassID;
    break;
  case 128:
    // ToDo: unclear if s[100:104] is available on VI. Can we use VCC as SGPR in
    // this bundle?
    if ((Val % 4 != 0) || (Val > 96)) {
      return MCDisassembler::Fail;
    }
    Val /= 4;
    RegClassID = AMDGPU::SReg_128RegClassID;
    break;
  case 256:
    // ToDo: unclear if s[96:104] is available on VI. Can we use VCC as SGPR in
    // this bundle?
    if ((Val % 4 != 0) || (Val > 92)) {
      return MCDisassembler::Fail;
    }
    Val /= 4;
    RegClassID = AMDGPU::SReg_256RegClassID;
    break;
  case 512:
    // ToDo: unclear if s[88:104] is available on VI. Can we use VCC as SGPR in
    // this bundle?
    if ((Val % 4 != 0) || (Val > 84)) {
      return MCDisassembler::Fail;
    }
    Val /= 4;
    RegClassID = AMDGPU::SReg_512RegClassID;
    break;
  default:
    return MCDisassembler::Fail;
  }

  RegID = AMDGPUMCRegisterClasses[RegClassID].getRegister(Val);
  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::DecodeSrc32Register(unsigned Val,
                                                     unsigned &RegID) const {
  // ToDo: deal with out-of range registers
  using namespace AMDGPU;
  if (Val <= 101) {
    return DecodeSgprRegister(Val, RegID, 32);
  } else if ((Val >= 256) && (Val <= 511)) {
    return DecodeVgprRegister(Val - 256, RegID, 32);
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

DecodeStatus AMDGPUDisassembler::DecodeSrc64Register(unsigned Val,
                                                     unsigned &RegID) const {
  // ToDo: deal with out-of range registers
  using namespace AMDGPU;
  if (Val <= 101) {
    return DecodeSgprRegister(Val, RegID, 64);
  } else if ((Val >= 256) && (Val <= 511)) {
    return DecodeVgprRegister(Val - 256, RegID, 64);
  } else {
    switch(Val) {
    case 102: RegID = getMCReg(FLAT_SCR, STI); return MCDisassembler::Success;
    case 106: RegID = getMCReg(VCC, STI); return MCDisassembler::Success;
    case 126: RegID = getMCReg(EXEC, STI); return MCDisassembler::Success;
    default: return MCDisassembler::Fail;
    }
  }
  return MCDisassembler::Fail;
}

DecodeStatus AMDGPUDisassembler::DecodeLiteralConstant(MCInst &Inst,
                                                       uint64_t &Literal) const {
  // For now all literal constants are supposed to be unsigned integer
  // ToDo: deal with signed/unsigned 64-bit integer constants
  // ToDo: deal with float/double constants
  if (Bytes.size() < 8) {
    return MCDisassembler::Fail;
  }
  Literal =
    0 | (Bytes[7] << 24) | (Bytes[6] << 16) | (Bytes[5] << 8) | (Bytes[4] << 0);
  return MCDisassembler::Success;
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

DecodeStatus AMDGPUDisassembler::DecodeVSRegisterClass(MCInst &Inst,
                                                       unsigned Imm,
                                                       uint64_t Addr,
                                                       bool Is32) const {
  // ToDo: different opcodes allow different formats of this operands
  if ((Imm >= 128) && (Imm <= 208)) {
    // immediate integer
    int64_t Val;
    if (DecodeImmedInteger(Imm, Val) == MCDisassembler::Success) {
      Inst.addOperand(MCOperand::createImm(Val));
      return MCDisassembler::Success;
    }
  } else if ((Imm >= 240) && (Imm <= 248)) {
    // immediate float/double
    uint64_t Val;
    DecodeStatus status;
    if (Is32) {
      uint32_t Val32;
      status = DecodeImmedFloat(Imm, Val32);
      Val = static_cast<uint64_t>(Val32);
    } else {
      status = DecodeImmedDouble(Imm, Val);
    }
    if (status == MCDisassembler::Success) {
      Inst.addOperand(MCOperand::createImm(Val));
      return MCDisassembler::Success;
    }
  } else if (Imm == 254) {
    // LDS direct
    // ToDo: implement LDS direct read
  } else if (Imm == 255) {
    // literal constant
    HasLiteral = true;
    uint64_t Literal;
    if (DecodeLiteralConstant(Inst, Literal) == MCDisassembler::Success) {
      Inst.addOperand(MCOperand::createImm(Literal));
      return MCDisassembler::Success;
    }
    return MCDisassembler::Fail;
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
    DecodeStatus status = Is32 ? DecodeSrc32Register(Imm, RegID)
                               : DecodeSrc64Register(Imm, RegID);
    if (status == MCDisassembler::Success) {
      Inst.addOperand(MCOperand::createReg(RegID));
      return MCDisassembler::Success;
    }
  }
  return MCDisassembler::Fail;
}

DecodeStatus AMDGPUDisassembler::DecodeVS_32RegisterClass(MCInst &Inst,
                                                          unsigned Imm,
                                                          uint64_t Addr) const {
  return DecodeVSRegisterClass(Inst, Imm, Addr, true);
}

DecodeStatus AMDGPUDisassembler::DecodeVS_64RegisterClass(MCInst &Inst,
                                                          unsigned Imm,
                                                          uint64_t Addr) const {
  return DecodeVSRegisterClass(Inst, Imm, Addr, false);
}

DecodeStatus AMDGPUDisassembler::DecodeVReg_64RegisterClass(llvm::MCInst &Inst,
                                                            unsigned Imm,
                                                            uint64_t Addr) const {
  unsigned RegID;
  if (DecodeVgprRegister(Imm, RegID, 64) == MCDisassembler::Success) {
    Inst.addOperand(MCOperand::createReg(RegID));
    return MCDisassembler::Success;
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
