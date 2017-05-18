//===- AMDGPUBaseInfo.cpp - AMDGPU Base encoding information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUBaseInfo.h"
#include "SIDefines.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"


#define GET_INSTRINFO_NAMED_OPS
#include "AMDGPUGenInstrInfo.inc"
#undef GET_INSTRINFO_NAMED_OPS

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

/// \returns Vmcnt bit shift (lower bits).
unsigned getVmcntBitShiftLo() { return 0; }

/// \returns Vmcnt bit width (lower bits).
unsigned getVmcntBitWidthLo() { return 4; }

/// \returns Expcnt bit shift.
unsigned getExpcntBitShift() { return 4; }

/// \returns Expcnt bit width.
unsigned getExpcntBitWidth() { return 3; }

/// \returns Lgkmcnt bit shift.
unsigned getLgkmcntBitShift() { return 8; }

/// \returns Lgkmcnt bit width.
unsigned getLgkmcntBitWidth() { return 4; }

/// \returns Vmcnt bit shift (higher bits).
unsigned getVmcntBitShiftHi() { return 14; }

/// \returns Vmcnt bit width (higher bits).
unsigned getVmcntBitWidthHi() { return 2; }

} // end namespace anonymous

namespace llvm {

static cl::opt<bool> EnablePackedInlinableLiterals(
    "enable-packed-inlinable-literals",
    cl::desc("Enable packed inlinable literals (v2f16, v2i16)"),
    cl::init(false));

namespace AMDGPU {

namespace IsaInfo {

IsaVersion getIsaVersion(const FeatureBitset &Features) {
  // CI.
  if (Features.test(FeatureISAVersion7_0_0))
    return {7, 0, 0};
  if (Features.test(FeatureISAVersion7_0_1))
    return {7, 0, 1};
  if (Features.test(FeatureISAVersion7_0_2))
    return {7, 0, 2};

  // VI.
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

  // GFX9.
  if (Features.test(FeatureISAVersion9_0_0))
    return {9, 0, 0};
  if (Features.test(FeatureISAVersion9_0_1))
    return {9, 0, 1};

  if (!Features.test(FeatureGCN) || Features.test(FeatureSouthernIslands))
    return {0, 0, 0};
  return {7, 0, 0};
}

unsigned getWavefrontSize(const FeatureBitset &Features) {
  if (Features.test(FeatureWavefrontSize16))
    return 16;
  if (Features.test(FeatureWavefrontSize32))
    return 32;

  return 64;
}

unsigned getLocalMemorySize(const FeatureBitset &Features) {
  if (Features.test(FeatureLocalMemorySize32768))
    return 32768;
  if (Features.test(FeatureLocalMemorySize65536))
    return 65536;

  return 0;
}

unsigned getEUsPerCU(const FeatureBitset &Features) {
  return 4;
}

unsigned getMaxWorkGroupsPerCU(const FeatureBitset &Features,
                               unsigned FlatWorkGroupSize) {
  if (!Features.test(FeatureGCN))
    return 8;
  unsigned N = getWavesPerWorkGroup(Features, FlatWorkGroupSize);
  if (N == 1)
    return 40;
  N = 40 / N;
  return std::min(N, 16u);
}

unsigned getMaxWavesPerCU(const FeatureBitset &Features) {
  return getMaxWavesPerEU(Features) * getEUsPerCU(Features);
}

unsigned getMaxWavesPerCU(const FeatureBitset &Features,
                          unsigned FlatWorkGroupSize) {
  return getWavesPerWorkGroup(Features, FlatWorkGroupSize);
}

unsigned getMinWavesPerEU(const FeatureBitset &Features) {
  return 1;
}

unsigned getMaxWavesPerEU(const FeatureBitset &Features) {
  if (!Features.test(FeatureGCN))
    return 8;
  // FIXME: Need to take scratch memory into account.
  return 10;
}

unsigned getMaxWavesPerEU(const FeatureBitset &Features,
                          unsigned FlatWorkGroupSize) {
  return alignTo(getMaxWavesPerCU(Features, FlatWorkGroupSize),
                 getEUsPerCU(Features)) / getEUsPerCU(Features);
}

unsigned getMinFlatWorkGroupSize(const FeatureBitset &Features) {
  return 1;
}

unsigned getMaxFlatWorkGroupSize(const FeatureBitset &Features) {
  return 2048;
}

unsigned getWavesPerWorkGroup(const FeatureBitset &Features,
                              unsigned FlatWorkGroupSize) {
  return alignTo(FlatWorkGroupSize, getWavefrontSize(Features)) /
                 getWavefrontSize(Features);
}

unsigned getSGPRAllocGranule(const FeatureBitset &Features) {
  IsaVersion Version = getIsaVersion(Features);
  if (Version.Major >= 8)
    return 16;
  return 8;
}

unsigned getSGPREncodingGranule(const FeatureBitset &Features) {
  return 8;
}

unsigned getTotalNumSGPRs(const FeatureBitset &Features) {
  IsaVersion Version = getIsaVersion(Features);
  if (Version.Major >= 8)
    return 800;
  return 512;
}

unsigned getAddressableNumSGPRs(const FeatureBitset &Features) {
  if (Features.test(FeatureSGPRInitBug))
    return FIXED_NUM_SGPRS_FOR_INIT_BUG;

  IsaVersion Version = getIsaVersion(Features);
  if (Version.Major >= 8)
    return 102;
  return 104;
}

unsigned getMinNumSGPRs(const FeatureBitset &Features, unsigned WavesPerEU) {
  assert(WavesPerEU != 0);

  if (WavesPerEU >= getMaxWavesPerEU(Features))
    return 0;
  unsigned MinNumSGPRs =
      alignDown(getTotalNumSGPRs(Features) / (WavesPerEU + 1),
                getSGPRAllocGranule(Features)) + 1;
  return std::min(MinNumSGPRs, getAddressableNumSGPRs(Features));
}

unsigned getMaxNumSGPRs(const FeatureBitset &Features, unsigned WavesPerEU,
                        bool Addressable) {
  assert(WavesPerEU != 0);

  IsaVersion Version = getIsaVersion(Features);
  unsigned MaxNumSGPRs = alignDown(getTotalNumSGPRs(Features) / WavesPerEU,
                                   getSGPRAllocGranule(Features));
  unsigned AddressableNumSGPRs = getAddressableNumSGPRs(Features);
  if (Version.Major >= 8 && !Addressable)
    AddressableNumSGPRs = 112;
  return std::min(MaxNumSGPRs, AddressableNumSGPRs);
}

unsigned getVGPRAllocGranule(const FeatureBitset &Features) {
  return 4;
}

unsigned getVGPREncodingGranule(const FeatureBitset &Features) {
  return getVGPRAllocGranule(Features);
}

unsigned getTotalNumVGPRs(const FeatureBitset &Features) {
  return 256;
}

unsigned getAddressableNumVGPRs(const FeatureBitset &Features) {
  return getTotalNumVGPRs(Features);
}

unsigned getMinNumVGPRs(const FeatureBitset &Features, unsigned WavesPerEU) {
  assert(WavesPerEU != 0);

  if (WavesPerEU >= getMaxWavesPerEU(Features))
    return 0;
  unsigned MinNumVGPRs =
      alignDown(getTotalNumVGPRs(Features) / (WavesPerEU + 1),
                getVGPRAllocGranule(Features)) + 1;
  return std::min(MinNumVGPRs, getAddressableNumVGPRs(Features));
}

unsigned getMaxNumVGPRs(const FeatureBitset &Features, unsigned WavesPerEU) {
  assert(WavesPerEU != 0);

  unsigned MaxNumVGPRs = alignDown(getTotalNumVGPRs(Features) / WavesPerEU,
                                   getVGPRAllocGranule(Features));
  unsigned AddressableNumVGPRs = getAddressableNumVGPRs(Features);
  return std::min(MaxNumVGPRs, AddressableNumVGPRs);
}

} // end namespace IsaInfo

void initDefaultAMDKernelCodeT(amd_kernel_code_t &Header,
                               const FeatureBitset &Features) {
  IsaInfo::IsaVersion ISA = IsaInfo::getIsaVersion(Features);

  memset(&Header, 0, sizeof(Header));

  Header.amd_kernel_code_version_major = 1;
  Header.amd_kernel_code_version_minor = 1;
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

bool isGroupSegment(const GlobalValue *GV, AMDGPUAS AS) {
  return GV->getType()->getAddressSpace() == AS.LOCAL_ADDRESS;
}

bool isGlobalSegment(const GlobalValue *GV, AMDGPUAS AS) {
  return GV->getType()->getAddressSpace() == AS.GLOBAL_ADDRESS;
}

bool isReadOnlySegment(const GlobalValue *GV, AMDGPUAS AS) {
  return GV->getType()->getAddressSpace() == AS.CONSTANT_ADDRESS;
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
    if (!OnlyFirstRequired || !Strs.second.trim().empty()) {
      Ctx.emitError("can't parse second integer attribute " + Name);
      return Default;
    }
  }

  return Ints;
}

unsigned getVmcntBitMask(const IsaInfo::IsaVersion &Version) {
  unsigned VmcntLo = (1 << getVmcntBitWidthLo()) - 1;
  if (Version.Major < 9)
    return VmcntLo;

  unsigned VmcntHi = ((1 << getVmcntBitWidthHi()) - 1) << getVmcntBitWidthLo();
  return VmcntLo | VmcntHi;
}

unsigned getExpcntBitMask(const IsaInfo::IsaVersion &Version) {
  return (1 << getExpcntBitWidth()) - 1;
}

unsigned getLgkmcntBitMask(const IsaInfo::IsaVersion &Version) {
  return (1 << getLgkmcntBitWidth()) - 1;
}

unsigned getWaitcntBitMask(const IsaInfo::IsaVersion &Version) {
  unsigned VmcntLo = getBitMask(getVmcntBitShiftLo(), getVmcntBitWidthLo());
  unsigned Expcnt = getBitMask(getExpcntBitShift(), getExpcntBitWidth());
  unsigned Lgkmcnt = getBitMask(getLgkmcntBitShift(), getLgkmcntBitWidth());
  unsigned Waitcnt = VmcntLo | Expcnt | Lgkmcnt;
  if (Version.Major < 9)
    return Waitcnt;

  unsigned VmcntHi = getBitMask(getVmcntBitShiftHi(), getVmcntBitWidthHi());
  return Waitcnt | VmcntHi;
}

unsigned decodeVmcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt) {
  unsigned VmcntLo =
      unpackBits(Waitcnt, getVmcntBitShiftLo(), getVmcntBitWidthLo());
  if (Version.Major < 9)
    return VmcntLo;

  unsigned VmcntHi =
      unpackBits(Waitcnt, getVmcntBitShiftHi(), getVmcntBitWidthHi());
  VmcntHi <<= getVmcntBitWidthLo();
  return VmcntLo | VmcntHi;
}

unsigned decodeExpcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getExpcntBitShift(), getExpcntBitWidth());
}

unsigned decodeLgkmcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getLgkmcntBitShift(), getLgkmcntBitWidth());
}

void decodeWaitcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt,
                   unsigned &Vmcnt, unsigned &Expcnt, unsigned &Lgkmcnt) {
  Vmcnt = decodeVmcnt(Version, Waitcnt);
  Expcnt = decodeExpcnt(Version, Waitcnt);
  Lgkmcnt = decodeLgkmcnt(Version, Waitcnt);
}

unsigned encodeVmcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt,
                     unsigned Vmcnt) {
  Waitcnt =
      packBits(Vmcnt, Waitcnt, getVmcntBitShiftLo(), getVmcntBitWidthLo());
  if (Version.Major < 9)
    return Waitcnt;

  Vmcnt >>= getVmcntBitWidthLo();
  return packBits(Vmcnt, Waitcnt, getVmcntBitShiftHi(), getVmcntBitWidthHi());
}

unsigned encodeExpcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt,
                      unsigned Expcnt) {
  return packBits(Expcnt, Waitcnt, getExpcntBitShift(), getExpcntBitWidth());
}

unsigned encodeLgkmcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt,
                       unsigned Lgkmcnt) {
  return packBits(Lgkmcnt, Waitcnt, getLgkmcntBitShift(), getLgkmcntBitWidth());
}

unsigned encodeWaitcnt(const IsaInfo::IsaVersion &Version,
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
    case CallingConv::AMDGPU_HS:
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

bool isEntryFunctionCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::AMDGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
  case CallingConv::AMDGPU_CS:
  case CallingConv::AMDGPU_HS:
    return true;
  default:
    return false;
  }
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

unsigned mc2PseudoReg(unsigned Reg) {
  switch (Reg) {
  case AMDGPU::FLAT_SCR_ci:
  case AMDGPU::FLAT_SCR_vi:
    return FLAT_SCR;

  case AMDGPU::FLAT_SCR_LO_ci:
  case AMDGPU::FLAT_SCR_LO_vi:
    return AMDGPU::FLAT_SCR_LO;

  case AMDGPU::FLAT_SCR_HI_ci:
  case AMDGPU::FLAT_SCR_HI_vi:
    return AMDGPU::FLAT_SCR_HI;

  default:
    return Reg;
  }
}

bool isSISrcOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  assert(OpNo < Desc.NumOperands);
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;
  return OpType >= AMDGPU::OPERAND_SRC_FIRST &&
         OpType <= AMDGPU::OPERAND_SRC_LAST;
}

bool isSISrcFPOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  assert(OpNo < Desc.NumOperands);
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;
  switch (OpType) {
  case AMDGPU::OPERAND_REG_IMM_FP32:
  case AMDGPU::OPERAND_REG_IMM_FP64:
  case AMDGPU::OPERAND_REG_IMM_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_FP64:
  case AMDGPU::OPERAND_REG_INLINE_C_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
    return true;
  default:
    return false;
  }
}

bool isSISrcInlinableOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  assert(OpNo < Desc.NumOperands);
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
  assert(OpNo < Desc.NumOperands);
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

bool isInlinableLiteralV216(int32_t Literal, bool HasInv2Pi) {
  assert(HasInv2Pi);

  if (!EnablePackedInlinableLiterals)
    return false;

  int16_t Lo16 = static_cast<int16_t>(Literal);
  int16_t Hi16 = static_cast<int16_t>(Literal >> 16);
  return Lo16 == Hi16 && isInlinableLiteral16(Lo16, HasInv2Pi);
}

bool isUniformMMO(const MachineMemOperand *MMO) {
  const Value *Ptr = MMO->getValue();
  // UndefValue means this is a load of a kernel input.  These are uniform.
  // Sometimes LDS instructions have constant pointers.
  // If Ptr is null, then that means this mem operand contains a
  // PseudoSourceValue like GOT.
  if (!Ptr || isa<UndefValue>(Ptr) || isa<Argument>(Ptr) ||
      isa<Constant>(Ptr) || isa<GlobalValue>(Ptr))
    return true;

  const Instruction *I = dyn_cast<Instruction>(Ptr);
  return I && I->getMetadata("amdgpu.uniform");
}

int64_t getSMRDEncodedOffset(const MCSubtargetInfo &ST, int64_t ByteOffset) {
  if (isSI(ST) || isCI(ST))
    return ByteOffset >> 2;

  return ByteOffset;
}

bool isLegalSMRDImmOffset(const MCSubtargetInfo &ST, int64_t ByteOffset) {
  int64_t EncodedOffset = getSMRDEncodedOffset(ST, ByteOffset);
  return isSI(ST) || isCI(ST) ? isUInt<8>(EncodedOffset) :
                                isUInt<20>(EncodedOffset);
}
} // end namespace AMDGPU

} // end namespace llvm

const unsigned AMDGPUAS::MAX_COMMON_ADDRESS;
const unsigned AMDGPUAS::GLOBAL_ADDRESS;
const unsigned AMDGPUAS::LOCAL_ADDRESS;
const unsigned AMDGPUAS::PARAM_D_ADDRESS;
const unsigned AMDGPUAS::PARAM_I_ADDRESS;
const unsigned AMDGPUAS::CONSTANT_BUFFER_0;
const unsigned AMDGPUAS::CONSTANT_BUFFER_1;
const unsigned AMDGPUAS::CONSTANT_BUFFER_2;
const unsigned AMDGPUAS::CONSTANT_BUFFER_3;
const unsigned AMDGPUAS::CONSTANT_BUFFER_4;
const unsigned AMDGPUAS::CONSTANT_BUFFER_5;
const unsigned AMDGPUAS::CONSTANT_BUFFER_6;
const unsigned AMDGPUAS::CONSTANT_BUFFER_7;
const unsigned AMDGPUAS::CONSTANT_BUFFER_8;
const unsigned AMDGPUAS::CONSTANT_BUFFER_9;
const unsigned AMDGPUAS::CONSTANT_BUFFER_10;
const unsigned AMDGPUAS::CONSTANT_BUFFER_11;
const unsigned AMDGPUAS::CONSTANT_BUFFER_12;
const unsigned AMDGPUAS::CONSTANT_BUFFER_13;
const unsigned AMDGPUAS::CONSTANT_BUFFER_14;
const unsigned AMDGPUAS::CONSTANT_BUFFER_15;
const unsigned AMDGPUAS::UNKNOWN_ADDRESS_SPACE;

namespace llvm {
namespace AMDGPU {

AMDGPUAS getAMDGPUAS(Triple T) {
  auto Env = T.getEnvironmentName();
  AMDGPUAS AS;
  if (Env == "amdgiz" || Env == "amdgizcl") {
    AS.FLAT_ADDRESS     = 0;
    AS.PRIVATE_ADDRESS  = 5;
    AS.REGION_ADDRESS   = 4;
  }
  else {
    AS.FLAT_ADDRESS     = 4;
    AS.PRIVATE_ADDRESS  = 0;
    AS.REGION_ADDRESS   = 5;
   }
  return AS;
}

AMDGPUAS getAMDGPUAS(const TargetMachine &M) {
  return getAMDGPUAS(M.getTargetTriple());
}

AMDGPUAS getAMDGPUAS(const Module &M) {
  return getAMDGPUAS(Triple(M.getTargetTriple()));
}
} // namespace AMDGPU
} // namespace llvm
