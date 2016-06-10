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
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"


using namespace llvm;

#define DEBUG_TYPE "amdgpu-disassembler"

typedef llvm::MCDisassembler::DecodeStatus DecodeStatus;


inline static MCDisassembler::DecodeStatus
addOperand(MCInst &Inst, const MCOperand& Opnd) {
  Inst.addOperand(Opnd);
  return Opnd.isValid() ?
    MCDisassembler::Success :
    MCDisassembler::SoftFail;
}

#define DECODE_OPERAND2(RegClass, DecName) \
static DecodeStatus Decode##RegClass##RegisterClass(MCInst &Inst, \
                                                    unsigned Imm, \
                                                    uint64_t /*Addr*/, \
                                                    const void *Decoder) { \
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder); \
  return addOperand(Inst, DAsm->decodeOperand_##DecName(Imm)); \
}

#define DECODE_OPERAND(RegClass) DECODE_OPERAND2(RegClass, RegClass)

DECODE_OPERAND(VGPR_32)
DECODE_OPERAND(VS_32)
DECODE_OPERAND(VS_64)

DECODE_OPERAND(VReg_64)
DECODE_OPERAND(VReg_96)
DECODE_OPERAND(VReg_128)

DECODE_OPERAND(SReg_32)
DECODE_OPERAND(SReg_32_XM0)
DECODE_OPERAND(SReg_64)
DECODE_OPERAND(SReg_128)
DECODE_OPERAND(SReg_256)
DECODE_OPERAND(SReg_512)

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"
#undef GET_SUBTARGETINFO_ENUM

#include "AMDGPUGenDisassemblerTables.inc"

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

template <typename T> static inline T eatBytes(ArrayRef<uint8_t>& Bytes) {
  assert(Bytes.size() >= sizeof(T));
  const auto Res = support::endian::read<T, support::endianness::little>(Bytes.data());
  Bytes = Bytes.slice(sizeof(T));
  return Res;
}

DecodeStatus AMDGPUDisassembler::tryDecodeInst(const uint8_t* Table,
                                               MCInst &MI,
                                               uint64_t Inst,
                                               uint64_t Address) const {
  assert(MI.getOpcode() == 0);
  assert(MI.getNumOperands() == 0);
  MCInst TmpInst;
  const auto SavedBytes = Bytes;
  if (decodeInstruction(Table, TmpInst, Inst, Address, this, STI)) {
    MI = TmpInst;
    return MCDisassembler::Success;
  }
  Bytes = SavedBytes;
  return MCDisassembler::Fail;
}

DecodeStatus AMDGPUDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                ArrayRef<uint8_t> Bytes_,
                                                uint64_t Address,
                                                raw_ostream &WS,
                                                raw_ostream &CS) const {
  CommentStream = &CS;

  // ToDo: AMDGPUDisassembler supports only VI ISA.
  assert(AMDGPU::isVI(STI) && "Can disassemble only VI ISA.");

  const unsigned MaxInstBytesNum = (std::min)((size_t)8, Bytes_.size());
  Bytes = Bytes_.slice(0, MaxInstBytesNum);

  DecodeStatus Res = MCDisassembler::Fail;
  do {
    // ToDo: better to switch encoding length using some bit predicate
    // but it is unknown yet, so try all we can

    // Try to decode DPP and SDWA first to solve conflict with VOP1 and VOP2
    // encodings
    if (Bytes.size() >= 8) {
      const uint64_t QW = eatBytes<uint64_t>(Bytes);
      Res = tryDecodeInst(DecoderTableDPP64, MI, QW, Address);
      if (Res) break;

      Res = tryDecodeInst(DecoderTableSDWA64, MI, QW, Address);
      if (Res) break;
    }

    // Reinitialize Bytes as DPP64 could have eaten too much
    Bytes = Bytes_.slice(0, MaxInstBytesNum);

    // Try decode 32-bit instruction
    if (Bytes.size() < 4) break;
    const uint32_t DW = eatBytes<uint32_t>(Bytes);
    Res = tryDecodeInst(DecoderTableVI32, MI, DW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableAMDGPU32, MI, DW, Address);
    if (Res) break;

    if (Bytes.size() < 4) break;
    const uint64_t QW = ((uint64_t)eatBytes<uint32_t>(Bytes) << 32) | DW;
    Res = tryDecodeInst(DecoderTableVI64, MI, QW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableAMDGPU64, MI, QW, Address);
  } while (false);

  Size = Res ? (MaxInstBytesNum - Bytes.size()) : 0;
  return Res;
}

const char* AMDGPUDisassembler::getRegClassName(unsigned RegClassID) const {
  return getContext().getRegisterInfo()->
    getRegClassName(&AMDGPUMCRegisterClasses[RegClassID]);
}

inline
MCOperand AMDGPUDisassembler::errOperand(unsigned V,
                                         const Twine& ErrMsg) const {
  *CommentStream << "Error: " + ErrMsg;

  // ToDo: add support for error operands to MCInst.h
  // return MCOperand::createError(V);
  return MCOperand();
}

inline
MCOperand AMDGPUDisassembler::createRegOperand(unsigned int RegId) const {
  return MCOperand::createReg(RegId);
}

inline
MCOperand AMDGPUDisassembler::createRegOperand(unsigned RegClassID,
                                               unsigned Val) const {
  const auto& RegCl = AMDGPUMCRegisterClasses[RegClassID];
  if (Val >= RegCl.getNumRegs())
    return errOperand(Val, Twine(getRegClassName(RegClassID)) +
                           ": unknown register " + Twine(Val));
  return createRegOperand(RegCl.getRegister(Val));
}

inline
MCOperand AMDGPUDisassembler::createSRegOperand(unsigned SRegClassID,
                                                unsigned Val) const {
  // ToDo: SI/CI have 104 SGPRs, VI - 102
  // Valery: here we accepting as much as we can, let assembler sort it out
  int shift = 0;
  switch (SRegClassID) {
  case AMDGPU::SGPR_32RegClassID:
  case AMDGPU::TTMP_32RegClassID:
    break;
  case AMDGPU::SGPR_64RegClassID:
  case AMDGPU::TTMP_64RegClassID:
    shift = 1;
    break;
  case AMDGPU::SGPR_128RegClassID:
  case AMDGPU::TTMP_128RegClassID:
  // ToDo: unclear if s[100:104] is available on VI. Can we use VCC as SGPR in
  // this bundle?
  case AMDGPU::SReg_256RegClassID:
  // ToDo: unclear if s[96:104] is available on VI. Can we use VCC as SGPR in
  // this bundle?
  case AMDGPU::SReg_512RegClassID:
    shift = 2;
    break;
  // ToDo: unclear if s[88:104] is available on VI. Can we use VCC as SGPR in
  // this bundle?
  default:
    assert(false);
    break;
  }
  if (Val % (1 << shift))
    *CommentStream << "Warning: " << getRegClassName(SRegClassID)
                   << ": scalar reg isn't aligned " << Val;
  return createRegOperand(SRegClassID, Val >> shift);
}

MCOperand AMDGPUDisassembler::decodeOperand_VS_32(unsigned Val) const {
  return decodeSrcOp(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VS_64(unsigned Val) const {
  return decodeSrcOp(OPW64, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VGPR_32(unsigned Val) const {
  return createRegOperand(AMDGPU::VGPR_32RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_64(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_64RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_96(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_96RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_128(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_128RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_32(unsigned Val) const {
  // table-gen generated disassembler doesn't care about operand types
  // leaving only registry class so SSrc_32 operand turns into SReg_32
  // and therefore we accept immediates and literals here as well
  return decodeSrcOp(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_32_XM0(unsigned Val) const {
  // SReg_32_XM0 is SReg_32 without M0
  return decodeOperand_SReg_32(Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_64(unsigned Val) const {
  // see decodeOperand_SReg_32 comment
  return decodeSrcOp(OPW64, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_128(unsigned Val) const {
  return decodeSrcOp(OPW128, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_256(unsigned Val) const {
  return createSRegOperand(AMDGPU::SReg_256RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_512(unsigned Val) const {
  return createSRegOperand(AMDGPU::SReg_512RegClassID, Val);
}


MCOperand AMDGPUDisassembler::decodeLiteralConstant() const {
  // For now all literal constants are supposed to be unsigned integer
  // ToDo: deal with signed/unsigned 64-bit integer constants
  // ToDo: deal with float/double constants
  if (Bytes.size() < 4)
    return errOperand(0, "cannot read literal, inst bytes left " +
                         Twine(Bytes.size()));
  return MCOperand::createImm(eatBytes<uint32_t>(Bytes));
}

MCOperand AMDGPUDisassembler::decodeIntImmed(unsigned Imm) {
  using namespace AMDGPU::EncValues;
  assert(Imm >= INLINE_INTEGER_C_MIN && Imm <= INLINE_INTEGER_C_MAX);
  return MCOperand::createImm((Imm <= INLINE_INTEGER_C_POSITIVE_MAX) ?
    (static_cast<int64_t>(Imm) - INLINE_INTEGER_C_MIN) :
    (INLINE_INTEGER_C_POSITIVE_MAX - static_cast<int64_t>(Imm)));
      // Cast prevents negative overflow.
}

MCOperand AMDGPUDisassembler::decodeFPImmed(bool Is32, unsigned Imm) {
  assert(Imm >= AMDGPU::EncValues::INLINE_FLOATING_C_MIN
      && Imm <= AMDGPU::EncValues::INLINE_FLOATING_C_MAX);
  // ToDo: case 248: 1/(2*PI) - is allowed only on VI
  // ToDo: AMDGPUInstPrinter does not support 1/(2*PI). It consider 1/(2*PI) as
  // literal constant.
  float V = 0.0f;
  switch (Imm) {
  case 240: V =  0.5f; break;
  case 241: V = -0.5f; break;
  case 242: V =  1.0f; break;
  case 243: V = -1.0f; break;
  case 244: V =  2.0f; break;
  case 245: V = -2.0f; break;
  case 246: V =  4.0f; break;
  case 247: V = -4.0f; break;
  case 248: return MCOperand::createImm(Is32 ?         // 1/(2*PI)
                                          0x3e22f983 :
                                          0x3fc45f306dc9c882);
  default: break;
  }
  return MCOperand::createImm(Is32? FloatToBits(V) : DoubleToBits(V));
}

unsigned AMDGPUDisassembler::getVgprClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;
  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32: return VGPR_32RegClassID;
  case OPW64: return VReg_64RegClassID;
  case OPW128: return VReg_128RegClassID;
  }
}

unsigned AMDGPUDisassembler::getSgprClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;
  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32: return SGPR_32RegClassID;
  case OPW64: return SGPR_64RegClassID;
  case OPW128: return SGPR_128RegClassID;
  }
}

unsigned AMDGPUDisassembler::getTtmpClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;
  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32: return TTMP_32RegClassID;
  case OPW64: return TTMP_64RegClassID;
  case OPW128: return TTMP_128RegClassID;
  }
}

MCOperand AMDGPUDisassembler::decodeSrcOp(const OpWidthTy Width, unsigned Val) const {
  using namespace AMDGPU::EncValues;
  assert(Val < 512); // enum9

  if (VGPR_MIN <= Val && Val <= VGPR_MAX) {
    return createRegOperand(getVgprClassId(Width), Val - VGPR_MIN);
  }
  if (Val <= SGPR_MAX) {
    assert(SGPR_MIN == 0); // "SGPR_MIN <= Val" is always true and causes compilation warning.
    return createSRegOperand(getSgprClassId(Width), Val - SGPR_MIN);
  }
  if (TTMP_MIN <= Val && Val <= TTMP_MAX) {
    return createSRegOperand(getTtmpClassId(Width), Val - TTMP_MIN);
  }

  assert(Width == OPW32 || Width == OPW64);
  const bool Is32 = (Width == OPW32);

  if (INLINE_INTEGER_C_MIN <= Val && Val <= INLINE_INTEGER_C_MAX)
    return decodeIntImmed(Val);

  if (INLINE_FLOATING_C_MIN <= Val && Val <= INLINE_FLOATING_C_MAX)
    return decodeFPImmed(Is32, Val);

  if (Val == LITERAL_CONST)
    return decodeLiteralConstant();

  return Is32 ? decodeSpecialReg32(Val) : decodeSpecialReg64(Val);
}

MCOperand AMDGPUDisassembler::decodeSpecialReg32(unsigned Val) const {
  using namespace AMDGPU;
  switch (Val) {
  case 102: return createRegOperand(getMCReg(FLAT_SCR_LO, STI));
  case 103: return createRegOperand(getMCReg(FLAT_SCR_HI, STI));
    // ToDo: no support for xnack_mask_lo/_hi register
  case 104:
  case 105: break;
  case 106: return createRegOperand(VCC_LO);
  case 107: return createRegOperand(VCC_HI);
  case 108: return createRegOperand(TBA_LO);
  case 109: return createRegOperand(TBA_HI);
  case 110: return createRegOperand(TMA_LO);
  case 111: return createRegOperand(TMA_HI);
  case 124: return createRegOperand(M0);
  case 126: return createRegOperand(EXEC_LO);
  case 127: return createRegOperand(EXEC_HI);
    // ToDo: no support for vccz register
  case 251: break;
    // ToDo: no support for execz register
  case 252: break;
  case 253: return createRegOperand(SCC);
  default: break;
  }
  return errOperand(Val, "unknown operand encoding " + Twine(Val));
}

MCOperand AMDGPUDisassembler::decodeSpecialReg64(unsigned Val) const {
  using namespace AMDGPU;
  switch (Val) {
  case 102: return createRegOperand(getMCReg(FLAT_SCR, STI));
  case 106: return createRegOperand(VCC);
  case 108: return createRegOperand(TBA);
  case 110: return createRegOperand(TMA);
  case 126: return createRegOperand(EXEC);
  default: break;
  }
  return errOperand(Val, "unknown operand encoding " + Twine(Val));
}

static MCDisassembler *createAMDGPUDisassembler(const Target &T,
                                                const MCSubtargetInfo &STI,
                                                MCContext &Ctx) {
  return new AMDGPUDisassembler(STI, Ctx);
}

extern "C" void LLVMInitializeAMDGPUDisassembler() {
  TargetRegistry::RegisterMCDisassembler(TheGCNTarget, createAMDGPUDisassembler);
}
