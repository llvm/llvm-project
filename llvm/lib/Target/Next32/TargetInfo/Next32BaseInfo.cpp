//===-- Next32BaseInfo.cpp - Next32 Helpers Function ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Next32 helpers APIs and constants
//
//===----------------------------------------------------------------------===//

#include "Next32BaseInfo.h"
#include "Next32.h"
#include "Next32ISelLowering.h"
#include "llvm/CodeGen/RuntimeLibcallUtil.h"

using namespace llvm;

struct CondNameEntry {
  Next32Constants::CondCode cond;
  Next32Constants::CondCode reverse;
  const char *name;
};

static const CondNameEntry gCodeNameTranslation[] = {
    {Next32Constants::E, Next32Constants::NE, ".e"},
    {Next32Constants::NE, Next32Constants::E, ".ne"},
    {Next32Constants::BE, Next32Constants::A, ".be"},
    {Next32Constants::AE, Next32Constants::B, ".ae"},
    {Next32Constants::B, Next32Constants::AE, ".b"},
    {Next32Constants::A, Next32Constants::BE, ".a"},
    {Next32Constants::GE, Next32Constants::L, ".ge"},
    {Next32Constants::LE, Next32Constants::G, ".le"},
    {Next32Constants::G, Next32Constants::LE, ".g"},
    {Next32Constants::L, Next32Constants::GE, ".l"}};

static const char *gNext32RRIAttributeNames[] = {
    /* Default */ "",
    /* ReadOnly */ ".ro",
    /* ReadNone */ ".rn",
    /* WriterParallel*/ ".wp"};

static const char *gFunctionAttributeNames[Next32Constants::ATT_SIZE] = {
    "next32-noreturn", // ATT_NORETURN
    "next32-inline",   // ATT_INLINE
};

StringRef Next32Helpers::GetRRIAttributeMnemonic(unsigned int AttributeValue) {
  return gNext32RRIAttributeNames[AttributeValue];
}

StringRef
Next32Helpers::GetFunctionAttrName(Next32Constants::Next32Attributes Attr) {
  assert(Attr < Next32Constants::ATT_SIZE);
  return gFunctionAttributeNames[Attr];
}

StringRef Next32Helpers::GetParallelMnemonic() { return ".p"; }

StringRef Next32Helpers::GetWriterMnemonic() { return "writer"; }

StringRef Next32Helpers::GetFeederMnemonic() { return "feeder"; }

StringRef Next32Helpers::GetCondCodeString(Next32Constants::CondCode Cond) {
  for (size_t i = 0; i < sizeof(gCodeNameTranslation) / sizeof(CondNameEntry);
       i++) {
    if (gCodeNameTranslation[i].cond == Cond)
      return gCodeNameTranslation[i].name;
  }
  return "";
}

Next32Constants::CondCode
Next32Helpers::GetCondCodeFromString(StringRef CondStr) {
  for (size_t i = 0; i < sizeof(gCodeNameTranslation) / sizeof(CondNameEntry);
       i++) {
    if (CondStr.compare_insensitive(gCodeNameTranslation[i].name))
      return gCodeNameTranslation[i].cond;
  }
  return Next32Constants::NoCondition;
}

Next32Constants::CondCode Next32Helpers::ISDCCToNext32CC(ISD::CondCode ISDCC) {
  switch (ISDCC) {
  case ISD::SETUEQ:
  case ISD::SETEQ:
    return Next32Constants::E;
  case ISD::SETUNE:
  case ISD::SETNE:
    return Next32Constants::NE;
  case ISD::SETULE:
    return Next32Constants::BE;
  case ISD::SETUGE:
    return Next32Constants::AE;
  case ISD::SETULT:
    return Next32Constants::B;
  case ISD::SETUGT:
    return Next32Constants::A;
  case ISD::SETGE:
    return Next32Constants::GE;
  case ISD::SETLE:
    return Next32Constants::LE;
  case ISD::SETGT:
    return Next32Constants::G;
  case ISD::SETLT:
    return Next32Constants::L;
  default:
    break;
  }
  llvm_unreachable("Unsupported ISD::CondCode");
  return Next32Constants::NoCondition;
}

Next32Constants::CondCode
Next32Helpers::GetReverseNext32CC(Next32Constants::CondCode Cond) {
  for (size_t i = 0; i < sizeof(gCodeNameTranslation) / sizeof(CondNameEntry);
       i++) {
    if (gCodeNameTranslation[i].cond == Cond)
      return gCodeNameTranslation[i].reverse;
  }
  return Next32Constants::NoCondition;
}

bool Next32Helpers::IsPseudoMemOpcode(unsigned int Next32ISDOpcode) {
  return (IsPseudoReadOpcode(Next32ISDOpcode) ||
          IsPseudoWriteOpcode(Next32ISDOpcode) ||
          IsPseudoAtomicOpcode(Next32ISDOpcode));
}

bool Next32Helpers::IsPseudoReadOpcode(unsigned int Next32ISDOpcode) {
  switch (Next32ISDOpcode) {
  case Next32ISD::G_VMEM_READ_1:
  case Next32ISD::G_VMEM_READ_2:
  case Next32ISD::G_VMEM_READ_4:
  case Next32ISD::G_VMEM_READ_8:
  case Next32ISD::G_VMEM_READ_16:
  case Next32ISD::G_MEM_READ_1:
  case Next32ISD::G_MEM_READ_2:
  case Next32ISD::G_MEM_READ_4:
  case Next32ISD::G_MEM_READ_8:
  case Next32ISD::G_MEM_READ_16:
    return true;
  }
  return false;
}

bool Next32Helpers::IsPseudoWriteOpcode(unsigned int Next32ISDOpcode) {
  return Next32ISDOpcode == Next32ISD::G_MEM_WRITE ||
         Next32ISDOpcode == Next32ISD::G_VMEM_WRITE;
}

bool Next32Helpers::IsPseudoAtomicOpcode(unsigned int Next32ISDOpcode) {
  switch (Next32ISDOpcode) {
  case Next32ISD::G_MEM_FAOP_S:
  case Next32ISD::G_MEM_FAOP_D:
  case Next32ISD::G_MEM_CAS_S:
  case Next32ISD::G_MEM_CAS_D:
    return true;
  }

  return false;
}

unsigned Next32Helpers::GetNext32VariadicPosition() { return 1; }

unsigned Next32Helpers::BitsToSizeFieldValue(unsigned Bits) {
  // In case of _ExtInt types, we need to set the bits to the next power of 2.
  Bits = std::max(PowerOf2Ceil(Bits), (uint64_t)8);

  switch (Bits) {
  case 1:
    return llvm::Next32Constants::InstructionSize::InstructionSize8;
  case 8:
    return llvm::Next32Constants::InstructionSize::InstructionSize8;
  case 16:
    return llvm::Next32Constants::InstructionSize::InstructionSize16;
  case 32:
    return llvm::Next32Constants::InstructionSize::InstructionSize32;
  case 64:
    return llvm::Next32Constants::InstructionSize::InstructionSize64;
  case 128:
    return llvm::Next32Constants::InstructionSize::InstructionSize128;
  case 256:
    return llvm::Next32Constants::InstructionSize::InstructionSize256;
  case 512:
    return llvm::Next32Constants::InstructionSize::InstructionSize512;
  case 1024:
    return llvm::Next32Constants::InstructionSize::InstructionSize1024;
  }
  llvm_unreachable("Invalid bit size");
}

unsigned Next32Helpers::BytesToLog2AlignValue(unsigned Bytes) {
  Bytes = std::min(Bytes, (unsigned)128);
  return Log2_32(Bytes);
}

unsigned Next32Helpers::Log2AlignValueToBytes(unsigned SizeAlignValue) {
  return 1 << SizeAlignValue;
}

unsigned Next32Helpers::SizeFieldValueToBits(unsigned SizeFieldValue) {
  switch (SizeFieldValue) {
  case llvm::Next32Constants::InstructionSize::InstructionSize8:
    return 8;
  case llvm::Next32Constants::InstructionSize::InstructionSize16:
    return 16;
  case llvm::Next32Constants::InstructionSize::InstructionSize32:
    return 32;
  case llvm::Next32Constants::InstructionSize::InstructionSize64:
    return 64;
  case llvm::Next32Constants::InstructionSize::InstructionSize128:
    return 128;
  case llvm::Next32Constants::InstructionSize::InstructionSize256:
    return 256;
  case llvm::Next32Constants::InstructionSize::InstructionSize512:
    return 512;
  case llvm::Next32Constants::InstructionSize::InstructionSize1024:
    return 1024;
  }
  llvm_unreachable("Invalid SizeFieldValue");
}

unsigned Next32Helpers::CountToLog2VecElemFieldValue(unsigned Count) {
  assert(isPowerOf2_32(Count) && "Vector element count must be power of 2");
  assert(Count <= 128 && "Vector element count is too large");
  return Log2_32(Count);
}

unsigned
Next32Helpers::Log2VecElemFieldValueToCount(unsigned VecElemFieldValue) {
  return 1 << VecElemFieldValue;
}

unsigned Next32Helpers::GetInstAddressSpace(unsigned AddrSpaceValue) {
  switch (AddrSpaceValue) {
  case llvm::AddressSpace::ADDRESS_SPACE_GENERIC:
    return Next32Constants::InstCodeAddressSpace::GENERIC;
  case llvm::AddressSpace::ADDRESS_SPACE_TLS:
    return Next32Constants::InstCodeAddressSpace::TLS;
  case llvm::AddressSpace::ADDRESS_SPACE_GLOBAL:
    return Next32Constants::InstCodeAddressSpace::GLOBAL;
  case llvm::AddressSpace::ADDRESS_SPACE_CONST:
    return Next32Constants::InstCodeAddressSpace::CONST;
  case llvm::AddressSpace::ADDRESS_SPACE_LOCAL:
    return Next32Constants::InstCodeAddressSpace::LOCAL;
  }
  llvm_unreachable("Invalid AddressSpaceValue");
}

unsigned Next32Helpers::MemNodeTypeToMemOps(MemSDNode *Mem) {
  switch (Mem->getOpcode()) {
  case ISD::LOAD:
    return Mem->isNonTemporal() ? Next32::MEMREAD_ONCE : Next32::MEMREAD;
  case ISD::STORE:
    return Mem->isNonTemporal() ? Next32::MEMWRITE_ONCE : Next32::MEMWRITE;
  case ISD::ATOMIC_LOAD:
    return Next32::MEMREAD_ATOMIC;
  case ISD::ATOMIC_STORE:
    return Next32::MEMWRITE_ATOMIC;
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS:
    return Next32::MEMCAS;
  case ISD::ATOMIC_SWAP:
    return Next32::MEMFA_SET;
  case ISD::ATOMIC_LOAD_ADD:
    return Next32::MEMFA_ADD;
  case ISD::ATOMIC_LOAD_SUB:
    return Next32::MEMFA_SUB;
  case ISD::ATOMIC_LOAD_AND:
    return Next32::MEMFA_AND;
  case ISD::ATOMIC_LOAD_OR:
    return Next32::MEMFA_OR;
  case ISD::ATOMIC_LOAD_XOR:
    return Next32::MEMFA_XOR;
  case ISD::ATOMIC_LOAD_NAND:
    return Next32::MEMFA_NAND;
  case ISD::ATOMIC_LOAD_MIN:
    return Next32::MEMFA_SMIN;
  case ISD::ATOMIC_LOAD_MAX:
    return Next32::MEMFA_SMAX;
  case ISD::ATOMIC_LOAD_UMIN:
    return Next32::MEMFA_UMIN;
  case ISD::ATOMIC_LOAD_UMAX:
    return Next32::MEMFA_UMAX;
  case ISD::ATOMIC_LOAD_FADD:
    return Next32::MEMFA_FADD;
  case ISD::ATOMIC_LOAD_FSUB:
    return Next32::MEMFA_FSUB;
  }
  llvm_unreachable("Invalid memory node type");
}

static const char *gNext32LibCallsFunction[] = {
    /* TLS_BASE   */ "__next32_tls_base",
};

bool Next32Helpers::IsValidVectorTy(EVT VT) {
  assert(VT.isVector() && "Called on a non-vector type?");

  if (!VT.isSimple())
    return false;

  const unsigned NumElems = VT.getVectorNumElements();
  const bool ValidNumElems = 1 < NumElems && NumElems <= 16;

  const unsigned ElemSizeBits = VT.getVectorElementType().getSizeInBits();
  const bool ValidElemSize = 8 <= ElemSizeBits && ElemSizeBits <= 64;

  const unsigned BitWidth = VT.getFixedSizeInBits();
  const bool ValidBitWidth = 8 <= BitWidth && BitWidth <= 512;

  return ValidNumElems && ValidElemSize && ValidBitWidth;
}

const char *
Next32Helpers::GetLibcallFunctionName(Next32Constants::Next32LibCalls Func) {
  return gNext32LibCallsFunction[Func];
}

static const RTLIB::Libcall gNext32RNFunctions[] = {
    RTLIB::SHL_I16,
    RTLIB::SHL_I32,
    RTLIB::SHL_I64,
    RTLIB::SHL_I128,
    RTLIB::SRL_I16,
    RTLIB::SRL_I32,
    RTLIB::SRL_I64,
    RTLIB::SRL_I128,
    RTLIB::SRA_I16,
    RTLIB::SRA_I32,
    RTLIB::SRA_I64,
    RTLIB::SRA_I128,
    RTLIB::MUL_I8,
    RTLIB::MUL_I16,
    RTLIB::MUL_I32,
    RTLIB::MUL_I64,
    RTLIB::MUL_I128,
    RTLIB::MULO_I32,
    RTLIB::MULO_I64,
    RTLIB::MULO_I128,
    RTLIB::SDIV_I8,
    RTLIB::SDIV_I16,
    RTLIB::SDIV_I32,
    RTLIB::SDIV_I64,
    RTLIB::SDIV_I128,
    RTLIB::UDIV_I8,
    RTLIB::UDIV_I16,
    RTLIB::UDIV_I32,
    RTLIB::UDIV_I64,
    RTLIB::UDIV_I128,
    RTLIB::UDIVREM_I32,
    RTLIB::UDIVREM_I64,
    RTLIB::SDIVREM_I32,
    RTLIB::SDIVREM_I64,
    RTLIB::SREM_I8,
    RTLIB::SREM_I16,
    RTLIB::SREM_I32,
    RTLIB::SREM_I64,
    RTLIB::SREM_I128,
    RTLIB::UREM_I8,
    RTLIB::UREM_I16,
    RTLIB::UREM_I32,
    RTLIB::UREM_I64,
    RTLIB::UREM_I128,
    RTLIB::SDIVREM_I32,
    RTLIB::UDIVREM_I32,
    RTLIB::SDIVREM_I64,
    RTLIB::UDIVREM_I64,
    RTLIB::NEG_I32,
    RTLIB::NEG_I64,
    RTLIB::ADD_F32,
    RTLIB::ADD_F64,
    RTLIB::ADD_F80,
    RTLIB::ADD_F128,
    RTLIB::ADD_PPCF128,
    RTLIB::SUB_F32,
    RTLIB::SUB_F64,
    RTLIB::SUB_F80,
    RTLIB::SUB_F128,
    RTLIB::SUB_PPCF128,
    RTLIB::MUL_F32,
    RTLIB::MUL_F64,
    RTLIB::MUL_F80,
    RTLIB::MUL_F128,
    RTLIB::MUL_PPCF128,
    RTLIB::DIV_F32,
    RTLIB::DIV_F64,
    RTLIB::DIV_F80,
    RTLIB::DIV_F128,
    RTLIB::DIV_PPCF128,
    RTLIB::REM_F32,
    RTLIB::REM_F64,
    RTLIB::REM_F80,
    RTLIB::REM_F128,
    RTLIB::REM_PPCF128,
    RTLIB::FMA_F32,
    RTLIB::FMA_F64,
    RTLIB::FMA_F80,
    RTLIB::FMA_F128,
    RTLIB::FMA_PPCF128,
    RTLIB::POWI_F32,
    RTLIB::POWI_F64,
    RTLIB::POWI_F80,
    RTLIB::POWI_F128,
    RTLIB::POWI_PPCF128,
    RTLIB::SQRT_F32,
    RTLIB::SQRT_F64,
    RTLIB::SQRT_F80,
    RTLIB::SQRT_F128,
    RTLIB::SQRT_PPCF128,
    RTLIB::LOG_F32,
    RTLIB::LOG_F64,
    RTLIB::LOG_F80,
    RTLIB::LOG_F128,
    RTLIB::LOG_PPCF128,
    RTLIB::LOG2_F32,
    RTLIB::LOG2_F64,
    RTLIB::LOG2_F80,
    RTLIB::LOG2_F128,
    RTLIB::LOG2_PPCF128,
    RTLIB::LOG10_F32,
    RTLIB::LOG10_F64,
    RTLIB::LOG10_F80,
    RTLIB::LOG10_F128,
    RTLIB::LOG10_PPCF128,
    RTLIB::EXP_F32,
    RTLIB::EXP_F64,
    RTLIB::EXP_F80,
    RTLIB::EXP_F128,
    RTLIB::EXP_PPCF128,
    RTLIB::EXP2_F32,
    RTLIB::EXP2_F64,
    RTLIB::EXP2_F80,
    RTLIB::EXP2_F128,
    RTLIB::EXP2_PPCF128,
    RTLIB::SIN_F32,
    RTLIB::SIN_F64,
    RTLIB::SIN_F80,
    RTLIB::SIN_F128,
    RTLIB::SIN_PPCF128,
    RTLIB::COS_F32,
    RTLIB::COS_F64,
    RTLIB::COS_F80,
    RTLIB::COS_F128,
    RTLIB::COS_PPCF128,
    RTLIB::POW_F32,
    RTLIB::POW_F64,
    RTLIB::POW_F80,
    RTLIB::POW_F128,
    RTLIB::POW_PPCF128,
    RTLIB::CEIL_F32,
    RTLIB::CEIL_F64,
    RTLIB::CEIL_F80,
    RTLIB::CEIL_F128,
    RTLIB::CEIL_PPCF128,
    RTLIB::TRUNC_F32,
    RTLIB::TRUNC_F64,
    RTLIB::TRUNC_F80,
    RTLIB::TRUNC_F128,
    RTLIB::TRUNC_PPCF128,
    RTLIB::RINT_F32,
    RTLIB::RINT_F64,
    RTLIB::RINT_F80,
    RTLIB::RINT_F128,
    RTLIB::RINT_PPCF128,
    RTLIB::NEARBYINT_F32,
    RTLIB::NEARBYINT_F64,
    RTLIB::NEARBYINT_F80,
    RTLIB::NEARBYINT_F128,
    RTLIB::NEARBYINT_PPCF128,
    RTLIB::ROUND_F32,
    RTLIB::ROUND_F64,
    RTLIB::ROUND_F80,
    RTLIB::ROUND_F128,
    RTLIB::ROUND_PPCF128,
    RTLIB::FLOOR_F32,
    RTLIB::FLOOR_F64,
    RTLIB::FLOOR_F80,
    RTLIB::FLOOR_F128,
    RTLIB::FLOOR_PPCF128,
    RTLIB::COPYSIGN_F32,
    RTLIB::COPYSIGN_F64,
    RTLIB::COPYSIGN_F80,
    RTLIB::COPYSIGN_F128,
    RTLIB::COPYSIGN_PPCF128,
    RTLIB::FMIN_F32,
    RTLIB::FMIN_F64,
    RTLIB::FMIN_F80,
    RTLIB::FMIN_F128,
    RTLIB::FMIN_PPCF128,
    RTLIB::FMAX_F32,
    RTLIB::FMAX_F64,
    RTLIB::FMAX_F80,
    RTLIB::FMAX_F128,
    RTLIB::FMAX_PPCF128,
    RTLIB::FPEXT_F32_PPCF128,
    RTLIB::FPEXT_F64_PPCF128,
    RTLIB::FPEXT_F64_F128,
    RTLIB::FPEXT_F32_F128,
    RTLIB::FPEXT_F32_F64,
    RTLIB::FPEXT_F16_F32,
    RTLIB::FPROUND_F32_F16,
    RTLIB::FPROUND_F64_F16,
    RTLIB::FPROUND_F80_F16,
    RTLIB::FPROUND_F128_F16,
    RTLIB::FPROUND_PPCF128_F16,
    RTLIB::FPROUND_F64_F32,
    RTLIB::FPROUND_F80_F32,
    RTLIB::FPROUND_F128_F32,
    RTLIB::FPROUND_PPCF128_F32,
    RTLIB::FPROUND_F80_F64,
    RTLIB::FPROUND_F128_F64,
    RTLIB::FPROUND_PPCF128_F64,
    RTLIB::FPTOSINT_F32_I32,
    RTLIB::FPTOSINT_F32_I64,
    RTLIB::FPTOSINT_F32_I128,
    RTLIB::FPTOSINT_F64_I32,
    RTLIB::FPTOSINT_F64_I64,
    RTLIB::FPTOSINT_F64_I128,
    RTLIB::FPTOSINT_F80_I32,
    RTLIB::FPTOSINT_F80_I64,
    RTLIB::FPTOSINT_F80_I128,
    RTLIB::FPTOSINT_F128_I32,
    RTLIB::FPTOSINT_F128_I64,
    RTLIB::FPTOSINT_F128_I128,
    RTLIB::FPTOSINT_PPCF128_I32,
    RTLIB::FPTOSINT_PPCF128_I64,
    RTLIB::FPTOSINT_PPCF128_I128,
    RTLIB::FPTOUINT_F32_I32,
    RTLIB::FPTOUINT_F32_I64,
    RTLIB::FPTOUINT_F32_I128,
    RTLIB::FPTOUINT_F64_I32,
    RTLIB::FPTOUINT_F64_I64,
    RTLIB::FPTOUINT_F64_I128,
    RTLIB::FPTOUINT_F80_I32,
    RTLIB::FPTOUINT_F80_I64,
    RTLIB::FPTOUINT_F80_I128,
    RTLIB::FPTOUINT_F128_I32,
    RTLIB::FPTOUINT_F128_I64,
    RTLIB::FPTOUINT_F128_I128,
    RTLIB::FPTOUINT_PPCF128_I32,
    RTLIB::FPTOUINT_PPCF128_I64,
    RTLIB::FPTOUINT_PPCF128_I128,
    RTLIB::SINTTOFP_I32_F32,
    RTLIB::SINTTOFP_I32_F64,
    RTLIB::SINTTOFP_I32_F80,
    RTLIB::SINTTOFP_I32_F128,
    RTLIB::SINTTOFP_I32_PPCF128,
    RTLIB::SINTTOFP_I64_F32,
    RTLIB::SINTTOFP_I64_F64,
    RTLIB::SINTTOFP_I64_F80,
    RTLIB::SINTTOFP_I64_F128,
    RTLIB::SINTTOFP_I64_PPCF128,
    RTLIB::SINTTOFP_I128_F32,
    RTLIB::SINTTOFP_I128_F64,
    RTLIB::SINTTOFP_I128_F80,
    RTLIB::SINTTOFP_I128_F128,
    RTLIB::SINTTOFP_I128_PPCF128,
    RTLIB::UINTTOFP_I32_F32,
    RTLIB::UINTTOFP_I32_F64,
    RTLIB::UINTTOFP_I32_F80,
    RTLIB::UINTTOFP_I32_F128,
    RTLIB::UINTTOFP_I32_PPCF128,
    RTLIB::UINTTOFP_I64_F32,
    RTLIB::UINTTOFP_I64_F64,
    RTLIB::UINTTOFP_I64_F80,
    RTLIB::UINTTOFP_I64_F128,
    RTLIB::UINTTOFP_I64_PPCF128,
    RTLIB::UINTTOFP_I128_F32,
    RTLIB::UINTTOFP_I128_F64,
    RTLIB::UINTTOFP_I128_F80,
    RTLIB::UINTTOFP_I128_F128,
    RTLIB::UINTTOFP_I128_PPCF128,
    RTLIB::OEQ_F32,
    RTLIB::OEQ_F64,
    RTLIB::OEQ_F128,
    RTLIB::OEQ_PPCF128,
    RTLIB::UNE_F32,
    RTLIB::UNE_F64,
    RTLIB::UNE_F128,
    RTLIB::UNE_PPCF128,
    RTLIB::OGE_F32,
    RTLIB::OGE_F64,
    RTLIB::OGE_F128,
    RTLIB::OGE_PPCF128,
    RTLIB::OLT_F32,
    RTLIB::OLT_F64,
    RTLIB::OLT_F128,
    RTLIB::OLT_PPCF128,
    RTLIB::OLE_F32,
    RTLIB::OLE_F64,
    RTLIB::OLE_F128,
    RTLIB::OLE_PPCF128,
    RTLIB::OGT_F32,
    RTLIB::OGT_F64,
    RTLIB::OGT_F128,
    RTLIB::OGT_PPCF128,
    RTLIB::UO_F32,
    RTLIB::UO_F64,
    RTLIB::UO_F128,
    RTLIB::UO_PPCF128,
};

MachineBasicBlock::iterator
Next32Helpers::FindArgumentFeedersEnd(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator Start) {
  bool VisitedFeeder = false;
  MachineBasicBlock::iterator I = Start;
  for (; I != MBB.end(); I++) {
    if (I->getOpcode() == Next32::FEEDER || I->getOpcode() == Next32::FEEDERP) {
      VisitedFeeder = true;
      continue;
    }
    if (VisitedFeeder)
      break;
  }
  return I;
}

Next32Constants::RRIAttribute
Next32Helpers::GetFunctionAttribute(StringRef FuncName,
                                    const TargetLoweringBase *TLB) {
  for (unsigned int i = 0;
       i < sizeof(gNext32RNFunctions) / sizeof(RTLIB::Libcall); ++i)
    if (TLB->getLibcallName(gNext32RNFunctions[i]) == FuncName)
      return Next32Constants::RRIAttribute::ReadNone;
  return Next32Constants::RRIAttribute::Default;
}
