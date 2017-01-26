//===-- AMDGPUBaseInfo.cpp - AMDGPU Base encoding information--------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "AMDGPUBaseInfo.h"
#include "AMDGPU.h"
#include "SIDefines.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/SubtargetFeature.h"

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"
#undef GET_SUBTARGETINFO_ENUM

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"
#undef GET_REGINFO_ENUM

#define GET_INSTRINFO_NAMED_OPS
#define GET_INSTRINFO_ENUM
#include "AMDGPUGenInstrInfo.inc"
#undef GET_INSTRINFO_NAMED_OPS
#undef GET_INSTRINFO_ENUM

namespace {

/// \returns Bit mask for given bit \p Shift and bit \p Width.
unsigned getBitMask(unsigned Shift, unsigned Width) {
  return ((1 << Width) - 1) << Shift;
}

/// \brief Packs \p Src into \p Dst for given bit \p Shift and bit \p Width.
///
/// \returns Packed \p Dst.
unsigned packBits(unsigned Src, unsigned Dst, unsigned Shift, unsigned Width) {
  Dst &= ~(1 << Shift) & ~getBitMask(Shift, Width);
  Dst |= (Src << Shift) & getBitMask(Shift, Width);
  return Dst;
}

/// \brief Unpacks bits from \p Src for given bit \p Shift and bit \p Width.
///
/// \returns Unpacked bits.
unsigned unpackBits(unsigned Src, unsigned Shift, unsigned Width) {
  return (Src & getBitMask(Shift, Width)) >> Shift;
}

/// \returns Vmcnt bit shift.
unsigned getVmcntBitShift() { return 0; }

/// \returns Vmcnt bit width.
unsigned getVmcntBitWidth() { return 4; }

/// \returns Expcnt bit shift.
unsigned getExpcntBitShift() { return 4; }

/// \returns Expcnt bit width.
unsigned getExpcntBitWidth() { return 3; }

/// \returns Lgkmcnt bit shift.
unsigned getLgkmcntBitShift() { return 8; }

/// \returns Lgkmcnt bit width.
unsigned getLgkmcntBitWidth() { return 4; }

} // anonymous namespace

namespace llvm {
namespace AMDGPU {

IsaVersion getIsaVersion(const FeatureBitset &Features) {

  if (Features.test(FeatureISAVersion7_0_0))
    return {7, 0, 0};

  if (Features.test(FeatureISAVersion7_0_1))
    return {7, 0, 1};

  if (Features.test(FeatureISAVersion7_0_2))
    return {7, 0, 2};

  if (Features.test(FeatureISAVersion8_0_0))
    return {8, 0, 0};

  if (Features.test(FeatureISAVersion8_0_1))
    return {8, 0, 1};

  if (Features.test(FeatureISAVersion8_0_2))
    return {8, 0, 2};

  if (Features.test(FeatureISAVersion8_0_3))
    return {8, 0, 3};

  if (Features.test(FeatureISAVersion8_0_4))
    return {8, 0, 4};

  if (Features.test(FeatureISAVersion8_1_0))
    return {8, 1, 0};

  return {0, 0, 0};
}

void initDefaultAMDKernelCodeT(amd_kernel_code_t &Header,
                               const FeatureBitset &Features) {

  IsaVersion ISA = getIsaVersion(Features);

  memset(&Header, 0, sizeof(Header));

  Header.amd_kernel_code_version_major = 1;
  Header.amd_kernel_code_version_minor = 0;
  Header.amd_machine_kind = 1; // AMD_MACHINE_KIND_AMDGPU
  Header.amd_machine_version_major = ISA.Major;
  Header.amd_machine_version_minor = ISA.Minor;
  Header.amd_machine_version_stepping = ISA.Stepping;
  Header.kernel_code_entry_byte_offset = sizeof(Header);
  // wavefront_size is specified as a power of 2: 2^6 = 64 threads.
  Header.wavefront_size = 6;

  // If the code object does not support indirect functions, then the value must
  // be 0xffffffff.
  Header.call_convention = -1;

  // These alignment values are specified in powers of two, so alignment =
  // 2^n.  The minimum alignment is 2^4 = 16.
  Header.kernarg_segment_alignment = 4;
  Header.group_segment_alignment = 4;
  Header.private_segment_alignment = 4;
}

MCSection *getHSATextSection(MCContext &Ctx) {
  return Ctx.getELFSection(".hsatext", ELF::SHT_PROGBITS,
                           ELF::SHF_ALLOC | ELF::SHF_WRITE |
                           ELF::SHF_EXECINSTR |
                           ELF::SHF_AMDGPU_HSA_AGENT |
                           ELF::SHF_AMDGPU_HSA_CODE);
}

MCSection *getHSADataGlobalAgentSection(MCContext &Ctx) {
  return Ctx.getELFSection(".hsadata_global_agent", ELF::SHT_PROGBITS,
                           ELF::SHF_ALLOC | ELF::SHF_WRITE |
                           ELF::SHF_AMDGPU_HSA_GLOBAL |
                           ELF::SHF_AMDGPU_HSA_AGENT);
}

MCSection *getHSADataGlobalProgramSection(MCContext &Ctx) {
  return  Ctx.getELFSection(".hsadata_global_program", ELF::SHT_PROGBITS,
                            ELF::SHF_ALLOC | ELF::SHF_WRITE |
                            ELF::SHF_AMDGPU_HSA_GLOBAL);
}

MCSection *getHSARodataReadonlyAgentSection(MCContext &Ctx) {
  return Ctx.getELFSection(".hsarodata_readonly_agent", ELF::SHT_PROGBITS,
                           ELF::SHF_ALLOC | ELF::SHF_AMDGPU_HSA_READONLY |
                           ELF::SHF_AMDGPU_HSA_AGENT);
}

bool isGroupSegment(const GlobalValue *GV) {
  return GV->getType()->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
}

bool isGlobalSegment(const GlobalValue *GV) {
  return GV->getType()->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
}

bool isReadOnlySegment(const GlobalValue *GV) {
  return GV->getType()->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS;
}

bool shouldEmitConstantsToTextSection(const Triple &TT) {
  return TT.getOS() != Triple::AMDHSA;
}

int getIntegerAttribute(const Function &F, StringRef Name, int Default) {
  Attribute A = F.getFnAttribute(Name);
  int Result = Default;

  if (A.isStringAttribute()) {
    StringRef Str = A.getValueAsString();
    if (Str.getAsInteger(0, Result)) {
      LLVMContext &Ctx = F.getContext();
      Ctx.emitError("can't parse integer attribute " + Name);
    }
  }

  return Result;
}

std::pair<int, int> getIntegerPairAttribute(const Function &F,
                                            StringRef Name,
                                            std::pair<int, int> Default,
                                            bool OnlyFirstRequired) {
  Attribute A = F.getFnAttribute(Name);
  if (!A.isStringAttribute())
    return Default;

  LLVMContext &Ctx = F.getContext();
  std::pair<int, int> Ints = Default;
  std::pair<StringRef, StringRef> Strs = A.getValueAsString().split(',');
  if (Strs.first.trim().getAsInteger(0, Ints.first)) {
    Ctx.emitError("can't parse first integer attribute " + Name);
    return Default;
  }
  if (Strs.second.trim().getAsInteger(0, Ints.second)) {
    if (!OnlyFirstRequired || Strs.second.trim().size()) {
      Ctx.emitError("can't parse second integer attribute " + Name);
      return Default;
    }
  }

  return Ints;
}

unsigned getWaitcntBitMask(IsaVersion Version) {
  unsigned Vmcnt = getBitMask(getVmcntBitShift(), getVmcntBitWidth());
  unsigned Expcnt = getBitMask(getExpcntBitShift(), getExpcntBitWidth());
  unsigned Lgkmcnt = getBitMask(getLgkmcntBitShift(), getLgkmcntBitWidth());
  return Vmcnt | Expcnt | Lgkmcnt;
}

unsigned getVmcntBitMask(IsaVersion Version) {
  return (1 << getVmcntBitWidth()) - 1;
}

unsigned getExpcntBitMask(IsaVersion Version) {
  return (1 << getExpcntBitWidth()) - 1;
}

unsigned getLgkmcntBitMask(IsaVersion Version) {
  return (1 << getLgkmcntBitWidth()) - 1;
}

unsigned decodeVmcnt(IsaVersion Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getVmcntBitShift(), getVmcntBitWidth());
}

unsigned decodeExpcnt(IsaVersion Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getExpcntBitShift(), getExpcntBitWidth());
}

unsigned decodeLgkmcnt(IsaVersion Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getLgkmcntBitShift(), getLgkmcntBitWidth());
}

void decodeWaitcnt(IsaVersion Version, unsigned Waitcnt,
                   unsigned &Vmcnt, unsigned &Expcnt, unsigned &Lgkmcnt) {
  Vmcnt = decodeVmcnt(Version, Waitcnt);
  Expcnt = decodeExpcnt(Version, Waitcnt);
  Lgkmcnt = decodeLgkmcnt(Version, Waitcnt);
}

unsigned encodeVmcnt(IsaVersion Version, unsigned Waitcnt, unsigned Vmcnt) {
  return packBits(Vmcnt, Waitcnt, getVmcntBitShift(), getVmcntBitWidth());
}

unsigned encodeExpcnt(IsaVersion Version, unsigned Waitcnt, unsigned Expcnt) {
  return packBits(Expcnt, Waitcnt, getExpcntBitShift(), getExpcntBitWidth());
}

unsigned encodeLgkmcnt(IsaVersion Version, unsigned Waitcnt, unsigned Lgkmcnt) {
  return packBits(Lgkmcnt, Waitcnt, getLgkmcntBitShift(), getLgkmcntBitWidth());
}

unsigned encodeWaitcnt(IsaVersion Version,
                       unsigned Vmcnt, unsigned Expcnt, unsigned Lgkmcnt) {
  unsigned Waitcnt = getWaitcntBitMask(Version);
  Waitcnt = encodeVmcnt(Version, Waitcnt, Vmcnt);
  Waitcnt = encodeExpcnt(Version, Waitcnt, Expcnt);
  Waitcnt = encodeLgkmcnt(Version, Waitcnt, Lgkmcnt);
  return Waitcnt;
}

unsigned getInitialPSInputAddr(const Function &F) {
  return getIntegerAttribute(F, "InitialPSInputAddr", 0);
}

bool isShader(CallingConv::ID cc) {
  switch(cc) {
    case CallingConv::AMDGPU_VS:
    case CallingConv::AMDGPU_GS:
    case CallingConv::AMDGPU_PS:
    case CallingConv::AMDGPU_CS:
      return true;
    default:
      return false;
  }
}

bool isCompute(CallingConv::ID cc) {
  return !isShader(cc) || cc == CallingConv::AMDGPU_CS;
}

bool isSI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureSouthernIslands];
}

bool isCI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureSeaIslands];
}

bool isVI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureVolcanicIslands];
}

unsigned getMCReg(unsigned Reg, const MCSubtargetInfo &STI) {

  switch(Reg) {
  default: break;
  case AMDGPU::FLAT_SCR:
    assert(!isSI(STI));
    return isCI(STI) ? AMDGPU::FLAT_SCR_ci : AMDGPU::FLAT_SCR_vi;

  case AMDGPU::FLAT_SCR_LO:
    assert(!isSI(STI));
    return isCI(STI) ? AMDGPU::FLAT_SCR_LO_ci : AMDGPU::FLAT_SCR_LO_vi;

  case AMDGPU::FLAT_SCR_HI:
    assert(!isSI(STI));
    return isCI(STI) ? AMDGPU::FLAT_SCR_HI_ci : AMDGPU::FLAT_SCR_HI_vi;
  }
  return Reg;
}

bool isSISrcOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;
  return OpType >= AMDGPU::OPERAND_SRC_FIRST &&
         OpType <= AMDGPU::OPERAND_SRC_LAST;
}

bool isSISrcFPOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;
  switch (OpType) {
  case AMDGPU::OPERAND_REG_IMM_FP32:
  case AMDGPU::OPERAND_REG_IMM_FP64:
  case AMDGPU::OPERAND_REG_IMM_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_FP64:
  case AMDGPU::OPERAND_REG_INLINE_C_FP16:
    return true;
  default:
    return false;
  }
}

bool isSISrcInlinableOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;
  return OpType >= AMDGPU::OPERAND_REG_INLINE_C_FIRST &&
         OpType <= AMDGPU::OPERAND_REG_INLINE_C_LAST;
}

// Avoid using MCRegisterClass::getSize, since that function will go away
// (move from MC* level to Target* level). Return size in bits.
unsigned getRegBitWidth(unsigned RCID) {
  switch (RCID) {
  case AMDGPU::SGPR_32RegClassID:
  case AMDGPU::VGPR_32RegClassID:
  case AMDGPU::VS_32RegClassID:
  case AMDGPU::SReg_32RegClassID:
  case AMDGPU::SReg_32_XM0RegClassID:
    return 32;
  case AMDGPU::SGPR_64RegClassID:
  case AMDGPU::VS_64RegClassID:
  case AMDGPU::SReg_64RegClassID:
  case AMDGPU::VReg_64RegClassID:
    return 64;
  case AMDGPU::VReg_96RegClassID:
    return 96;
  case AMDGPU::SGPR_128RegClassID:
  case AMDGPU::SReg_128RegClassID:
  case AMDGPU::VReg_128RegClassID:
    return 128;
  case AMDGPU::SReg_256RegClassID:
  case AMDGPU::VReg_256RegClassID:
    return 256;
  case AMDGPU::SReg_512RegClassID:
  case AMDGPU::VReg_512RegClassID:
    return 512;
  default:
    llvm_unreachable("Unexpected register class");
  }
}

unsigned getRegBitWidth(const MCRegisterClass &RC) {
  return getRegBitWidth(RC.getID());
}

unsigned getRegOperandSize(const MCRegisterInfo *MRI, const MCInstrDesc &Desc,
                           unsigned OpNo) {
  unsigned RCID = Desc.OpInfo[OpNo].RegClass;
  return getRegBitWidth(MRI->getRegClass(RCID)) / 8;
}

bool isInlinableLiteral64(int64_t Literal, bool HasInv2Pi) {
  if (Literal >= -16 && Literal <= 64)
    return true;

  uint64_t Val = static_cast<uint64_t>(Literal);
  return (Val == DoubleToBits(0.0)) ||
         (Val == DoubleToBits(1.0)) ||
         (Val == DoubleToBits(-1.0)) ||
         (Val == DoubleToBits(0.5)) ||
         (Val == DoubleToBits(-0.5)) ||
         (Val == DoubleToBits(2.0)) ||
         (Val == DoubleToBits(-2.0)) ||
         (Val == DoubleToBits(4.0)) ||
         (Val == DoubleToBits(-4.0)) ||
         (Val == 0x3fc45f306dc9c882 && HasInv2Pi);
}

bool isInlinableLiteral32(int32_t Literal, bool HasInv2Pi) {
  if (Literal >= -16 && Literal <= 64)
    return true;

  // The actual type of the operand does not seem to matter as long
  // as the bits match one of the inline immediate values.  For example:
  //
  // -nan has the hexadecimal encoding of 0xfffffffe which is -2 in decimal,
  // so it is a legal inline immediate.
  //
  // 1065353216 has the hexadecimal encoding 0x3f800000 which is 1.0f in
  // floating-point, so it is a legal inline immediate.

  uint32_t Val = static_cast<uint32_t>(Literal);
  return (Val == FloatToBits(0.0f)) ||
         (Val == FloatToBits(1.0f)) ||
         (Val == FloatToBits(-1.0f)) ||
         (Val == FloatToBits(0.5f)) ||
         (Val == FloatToBits(-0.5f)) ||
         (Val == FloatToBits(2.0f)) ||
         (Val == FloatToBits(-2.0f)) ||
         (Val == FloatToBits(4.0f)) ||
         (Val == FloatToBits(-4.0f)) ||
         (Val == 0x3e22f983 && HasInv2Pi);
}

bool isInlinableLiteral16(int16_t Literal, bool HasInv2Pi) {
  if (!HasInv2Pi)
    return false;

  if (Literal >= -16 && Literal <= 64)
    return true;

  uint16_t Val = static_cast<uint16_t>(Literal);
  return Val == 0x3C00 || // 1.0
         Val == 0xBC00 || // -1.0
         Val == 0x3800 || // 0.5
         Val == 0xB800 || // -0.5
         Val == 0x4000 || // 2.0
         Val == 0xC000 || // -2.0
         Val == 0x4400 || // 4.0
         Val == 0xC400 || // -4.0
         Val == 0x3118;   // 1/2pi
}

} // End namespace AMDGPU
} // End namespace llvm
