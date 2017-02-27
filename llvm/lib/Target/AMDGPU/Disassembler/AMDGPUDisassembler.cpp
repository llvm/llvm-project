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
#include "llvm/Support/ELF.h"
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

static DecodeStatus decodeSoppBrTarget(MCInst &Inst, unsigned Imm,
                                       uint64_t Addr, const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);

  APInt SignedOffset(18, Imm * 4, true);
  int64_t Offset = (SignedOffset.sext(64) + 4 + Addr).getSExtValue();

  if (DAsm->tryAddingSymbolicOperand(Inst, Offset, Addr, true, 2, 2))
    return MCDisassembler::Success;
  return addOperand(Inst, MCOperand::createImm(Imm));
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
DECODE_OPERAND(SReg_32_XM0_XEXEC)
DECODE_OPERAND(SReg_64)
DECODE_OPERAND(SReg_64_XEXEC)
DECODE_OPERAND(SReg_128)
DECODE_OPERAND(SReg_256)
DECODE_OPERAND(SReg_512)


static DecodeStatus decodeOperand_VSrc16(MCInst &Inst,
                                         unsigned Imm,
                                         uint64_t Addr,
                                         const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrc16(Imm));
}

static DecodeStatus decodeOperand_VSrcV216(MCInst &Inst,
                                         unsigned Imm,
                                         uint64_t Addr,
                                         const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrcV216(Imm));
}

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
  if (!STI.getFeatureBits()[AMDGPU::FeatureGCN3Encoding])
    report_fatal_error("Disassembly not yet supported for subtarget");

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
    llvm_unreachable("unhandled register class");
  }

  if (Val % (1 << shift)) {
    *CommentStream << "Warning: " << getRegClassName(SRegClassID)
                   << ": scalar reg isn't aligned " << Val;
  }

  return createRegOperand(SRegClassID, Val >> shift);
}

MCOperand AMDGPUDisassembler::decodeOperand_VS_32(unsigned Val) const {
  return decodeSrcOp(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VS_64(unsigned Val) const {
  return decodeSrcOp(OPW64, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VSrc16(unsigned Val) const {
  return decodeSrcOp(OPW16, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VSrcV216(unsigned Val) const {
  return decodeSrcOp(OPWV216, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VGPR_32(unsigned Val) const {
  // Some instructions have operand restrictions beyond what the encoding
  // allows. Some ordinarily VSrc_32 operands are VGPR_32, so clear the extra
  // high bit.
  Val &= 255;

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

MCOperand AMDGPUDisassembler::decodeOperand_SReg_32_XM0_XEXEC(
  unsigned Val) const {
  // SReg_32_XM0 is SReg_32 without M0 or EXEC_LO/EXEC_HI
  return decodeOperand_SReg_32(Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_64(unsigned Val) const {
  return decodeSrcOp(OPW64, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_64_XEXEC(unsigned Val) const {
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

static int64_t getInlineImmVal32(unsigned Imm) {
  switch (Imm) {
  case 240:
    return FloatToBits(0.5f);
  case 241:
    return FloatToBits(-0.5f);
  case 242:
    return FloatToBits(1.0f);
  case 243:
    return FloatToBits(-1.0f);
  case 244:
    return FloatToBits(2.0f);
  case 245:
    return FloatToBits(-2.0f);
  case 246:
    return FloatToBits(4.0f);
  case 247:
    return FloatToBits(-4.0f);
  case 248: // 1 / (2 * PI)
    return 0x3e22f983;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

static int64_t getInlineImmVal64(unsigned Imm) {
  switch (Imm) {
  case 240:
    return DoubleToBits(0.5);
  case 241:
    return DoubleToBits(-0.5);
  case 242:
    return DoubleToBits(1.0);
  case 243:
    return DoubleToBits(-1.0);
  case 244:
    return DoubleToBits(2.0);
  case 245:
    return DoubleToBits(-2.0);
  case 246:
    return DoubleToBits(4.0);
  case 247:
    return DoubleToBits(-4.0);
  case 248: // 1 / (2 * PI)
    return 0x3fc45f306dc9c882;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

static int64_t getInlineImmVal16(unsigned Imm) {
  switch (Imm) {
  case 240:
    return 0x3800;
  case 241:
    return 0xB800;
  case 242:
    return 0x3C00;
  case 243:
    return 0xBC00;
  case 244:
    return 0x4000;
  case 245:
    return 0xC000;
  case 246:
    return 0x4400;
  case 247:
    return 0xC400;
  case 248: // 1 / (2 * PI)
    return 0x3118;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

MCOperand AMDGPUDisassembler::decodeFPImmed(OpWidthTy Width, unsigned Imm) {
  assert(Imm >= AMDGPU::EncValues::INLINE_FLOATING_C_MIN
      && Imm <= AMDGPU::EncValues::INLINE_FLOATING_C_MAX);

  // ToDo: case 248: 1/(2*PI) - is allowed only on VI
  switch (Width) {
  case OPW32:
    return MCOperand::createImm(getInlineImmVal32(Imm));
  case OPW64:
    return MCOperand::createImm(getInlineImmVal64(Imm));
  case OPW16:
  case OPWV216:
    return MCOperand::createImm(getInlineImmVal16(Imm));
  default:
    llvm_unreachable("implement me");
  }
}

unsigned AMDGPUDisassembler::getVgprClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;
  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32:
  case OPW16:
  case OPWV216:
    return VGPR_32RegClassID;
  case OPW64: return VReg_64RegClassID;
  case OPW128: return VReg_128RegClassID;
  }
}

unsigned AMDGPUDisassembler::getSgprClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;
  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32:
  case OPW16:
  case OPWV216:
    return SGPR_32RegClassID;
  case OPW64: return SGPR_64RegClassID;
  case OPW128: return SGPR_128RegClassID;
  }
}

unsigned AMDGPUDisassembler::getTtmpClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;
  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32:
  case OPW16:
  case OPWV216:
    return TTMP_32RegClassID;
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

  assert(Width == OPW16 || Width == OPW32 || Width == OPW64);

  if (INLINE_INTEGER_C_MIN <= Val && Val <= INLINE_INTEGER_C_MAX)
    return decodeIntImmed(Val);

  if (INLINE_FLOATING_C_MIN <= Val && Val <= INLINE_FLOATING_C_MAX)
    return decodeFPImmed(Width, Val);

  if (Val == LITERAL_CONST)
    return decodeLiteralConstant();

  switch (Width) {
  case OPW32:
  case OPW16:
  case OPWV216:
    return decodeSpecialReg32(Val);
  case OPW64:
    return decodeSpecialReg64(Val);
  default:
    llvm_unreachable("unexpected immediate type");
  }
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
  case 235: return createRegOperand(SRC_SHARED_BASE);
  case 236: return createRegOperand(SRC_SHARED_LIMIT);
  case 237: return createRegOperand(SRC_PRIVATE_BASE);
  case 238: return createRegOperand(SRC_PRIVATE_LIMIT);
    // TODO: SRC_POPS_EXITING_WAVE_ID
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

//===----------------------------------------------------------------------===//
// AMDGPUSymbolizer
//===----------------------------------------------------------------------===//

// Try to find symbol name for specified label
bool AMDGPUSymbolizer::tryAddingSymbolicOperand(MCInst &Inst,
                                raw_ostream &/*cStream*/, int64_t Value,
                                uint64_t /*Address*/, bool IsBranch,
                                uint64_t /*Offset*/, uint64_t /*InstSize*/) {
  typedef std::tuple<uint64_t, StringRef, uint8_t> SymbolInfoTy;
  typedef std::vector<SymbolInfoTy> SectionSymbolsTy;

  if (!IsBranch) {
    return false;
  }

  auto *Symbols = static_cast<SectionSymbolsTy *>(DisInfo);
  auto Result = std::find_if(Symbols->begin(), Symbols->end(),
                             [Value](const SymbolInfoTy& Val) {
                                return std::get<0>(Val) == static_cast<uint64_t>(Value)
                                    && std::get<2>(Val) == ELF::STT_NOTYPE;
                             });
  if (Result != Symbols->end()) {
    auto *Sym = Ctx.getOrCreateSymbol(std::get<1>(*Result));
    const auto *Add = MCSymbolRefExpr::create(Sym, Ctx);
    Inst.addOperand(MCOperand::createExpr(Add));
    return true;
  }
  return false;
}

void AMDGPUSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                                       int64_t Value,
                                                       uint64_t Address) {
  llvm_unreachable("unimplemented");
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

static MCSymbolizer *createAMDGPUSymbolizer(const Triple &/*TT*/,
                              LLVMOpInfoCallback /*GetOpInfo*/,
                              LLVMSymbolLookupCallback /*SymbolLookUp*/,
                              void *DisInfo,
                              MCContext *Ctx,
                              std::unique_ptr<MCRelocationInfo> &&RelInfo) {
  return new AMDGPUSymbolizer(*Ctx, std::move(RelInfo), DisInfo);
}

static MCDisassembler *createAMDGPUDisassembler(const Target &T,
                                                const MCSubtargetInfo &STI,
                                                MCContext &Ctx) {
  return new AMDGPUDisassembler(STI, Ctx);
}

extern "C" void LLVMInitializeAMDGPUDisassembler() {
  TargetRegistry::RegisterMCDisassembler(getTheGCNTarget(),
                                         createAMDGPUDisassembler);
  TargetRegistry::RegisterMCSymbolizer(getTheGCNTarget(),
                                       createAMDGPUSymbolizer);
}
