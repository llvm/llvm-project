//===- AMDGPUBaseInfo.h - Top level definitions for AMDGPU ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBASEINFO_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBASEINFO_H

#include "AMDGPU.h"
#include "AMDKernelCodeT.h"
#include "SIDefines.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <string>
#include <utility>

namespace llvm {

class Argument;
class FeatureBitset;
class Function;
class GlobalValue;
class MCContext;
class MCRegisterClass;
class MCRegisterInfo;
class MCSection;
class MCSubtargetInfo;
class MachineMemOperand;
class Triple;

namespace AMDGPU {

#define GET_MIMGBaseOpcode_DECL
#define GET_MIMGDim_DECL
#define GET_MIMGEncoding_DECL
#include "AMDGPUGenSearchableTables.inc"

namespace IsaInfo {

enum {
  // The closed Vulkan driver sets 96, which limits the wave count to 8 but
  // doesn't spill SGPRs as much as when 80 is set.
  FIXED_NUM_SGPRS_FOR_INIT_BUG = 96,
  TRAP_NUM_SGPRS = 16
};

/// Instruction set architecture version.
struct IsaVersion {
  unsigned Major;
  unsigned Minor;
  unsigned Stepping;
};

/// \returns Isa version for given subtarget \p Features.
IsaVersion getIsaVersion(const FeatureBitset &Features);

/// Streams isa version string for given subtarget \p STI into \p Stream.
void streamIsaVersion(const MCSubtargetInfo *STI, raw_ostream &Stream);

/// \returns True if given subtarget \p STI supports code object version 3,
/// false otherwise.
bool hasCodeObjectV3(const MCSubtargetInfo *STI);

/// \returns Wavefront size for given subtarget \p Features.
unsigned getWavefrontSize(const FeatureBitset &Features);

/// \returns Local memory size in bytes for given subtarget \p Features.
unsigned getLocalMemorySize(const FeatureBitset &Features);

/// \returns Number of execution units per compute unit for given subtarget \p
/// Features.
unsigned getEUsPerCU(const FeatureBitset &Features);

/// \returns Maximum number of work groups per compute unit for given subtarget
/// \p Features and limited by given \p FlatWorkGroupSize.
unsigned getMaxWorkGroupsPerCU(const FeatureBitset &Features,
                               unsigned FlatWorkGroupSize);

/// \returns Maximum number of waves per compute unit for given subtarget \p
/// Features without any kind of limitation.
unsigned getMaxWavesPerCU(const FeatureBitset &Features);

/// \returns Maximum number of waves per compute unit for given subtarget \p
/// Features and limited by given \p FlatWorkGroupSize.
unsigned getMaxWavesPerCU(const FeatureBitset &Features,
                          unsigned FlatWorkGroupSize);

/// \returns Minimum number of waves per execution unit for given subtarget \p
/// Features.
unsigned getMinWavesPerEU(const FeatureBitset &Features);

/// \returns Maximum number of waves per execution unit for given subtarget \p
/// Features without any kind of limitation.
unsigned getMaxWavesPerEU(const FeatureBitset &Features);

/// \returns Maximum number of waves per execution unit for given subtarget \p
/// Features and limited by given \p FlatWorkGroupSize.
unsigned getMaxWavesPerEU(const FeatureBitset &Features,
                          unsigned FlatWorkGroupSize);

/// \returns Minimum flat work group size for given subtarget \p Features.
unsigned getMinFlatWorkGroupSize(const FeatureBitset &Features);

/// \returns Maximum flat work group size for given subtarget \p Features.
unsigned getMaxFlatWorkGroupSize(const FeatureBitset &Features);

/// \returns Number of waves per work group for given subtarget \p Features and
/// limited by given \p FlatWorkGroupSize.
unsigned getWavesPerWorkGroup(const FeatureBitset &Features,
                              unsigned FlatWorkGroupSize);

/// \returns SGPR allocation granularity for given subtarget \p Features.
unsigned getSGPRAllocGranule(const FeatureBitset &Features);

/// \returns SGPR encoding granularity for given subtarget \p Features.
unsigned getSGPREncodingGranule(const FeatureBitset &Features);

/// \returns Total number of SGPRs for given subtarget \p Features.
unsigned getTotalNumSGPRs(const FeatureBitset &Features);

/// \returns Addressable number of SGPRs for given subtarget \p Features.
unsigned getAddressableNumSGPRs(const FeatureBitset &Features);

/// \returns Minimum number of SGPRs that meets the given number of waves per
/// execution unit requirement for given subtarget \p Features.
unsigned getMinNumSGPRs(const FeatureBitset &Features, unsigned WavesPerEU);

/// \returns Maximum number of SGPRs that meets the given number of waves per
/// execution unit requirement for given subtarget \p Features.
unsigned getMaxNumSGPRs(const FeatureBitset &Features, unsigned WavesPerEU,
                        bool Addressable);

/// \returns Number of extra SGPRs implicitly required by given subtarget \p
/// Features when the given special registers are used.
unsigned getNumExtraSGPRs(const FeatureBitset &Features, bool VCCUsed,
                          bool FlatScrUsed, bool XNACKUsed);

/// \returns Number of extra SGPRs implicitly required by given subtarget \p
/// Features when the given special registers are used. XNACK is inferred from
/// \p Features.
unsigned getNumExtraSGPRs(const FeatureBitset &Features, bool VCCUsed,
                          bool FlatScrUsed);

/// \returns Number of SGPR blocks needed for given subtarget \p Features when
/// \p NumSGPRs are used. \p NumSGPRs should already include any special
/// register counts.
unsigned getNumSGPRBlocks(const FeatureBitset &Features, unsigned NumSGPRs);

/// \returns VGPR allocation granularity for given subtarget \p Features.
unsigned getVGPRAllocGranule(const FeatureBitset &Features);

/// \returns VGPR encoding granularity for given subtarget \p Features.
unsigned getVGPREncodingGranule(const FeatureBitset &Features);

/// \returns Total number of VGPRs for given subtarget \p Features.
unsigned getTotalNumVGPRs(const FeatureBitset &Features);

/// \returns Addressable number of VGPRs for given subtarget \p Features.
unsigned getAddressableNumVGPRs(const FeatureBitset &Features);

/// \returns Minimum number of VGPRs that meets given number of waves per
/// execution unit requirement for given subtarget \p Features.
unsigned getMinNumVGPRs(const FeatureBitset &Features, unsigned WavesPerEU);

/// \returns Maximum number of VGPRs that meets given number of waves per
/// execution unit requirement for given subtarget \p Features.
unsigned getMaxNumVGPRs(const FeatureBitset &Features, unsigned WavesPerEU);

/// \returns Number of VGPR blocks needed for given subtarget \p Features when
/// \p NumVGPRs are used.
unsigned getNumVGPRBlocks(const FeatureBitset &Features, unsigned NumSGPRs);

} // end namespace IsaInfo

LLVM_READONLY
int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx);

struct MIMGBaseOpcodeInfo {
  MIMGBaseOpcode BaseOpcode;
  bool Store;
  bool Atomic;
  bool AtomicX2;
  bool Sampler;

  uint8_t NumExtraArgs;
  bool Gradients;
  bool Coordinates;
  bool LodOrClampOrMip;
  bool HasD16;
};

LLVM_READONLY
const MIMGBaseOpcodeInfo *getMIMGBaseOpcodeInfo(unsigned BaseOpcode);

struct MIMGDimInfo {
  MIMGDim Dim;
  uint8_t NumCoords;
  uint8_t NumGradients;
  bool DA;
};

LLVM_READONLY
const MIMGDimInfo *getMIMGDimInfo(unsigned Dim);

LLVM_READONLY
int getMIMGOpcode(unsigned BaseOpcode, unsigned MIMGEncoding,
                  unsigned VDataDwords, unsigned VAddrDwords);

LLVM_READONLY
int getMaskedMIMGOp(unsigned Opc, unsigned NewChannels);

LLVM_READONLY
int getMCOpcode(uint16_t Opcode, unsigned Gen);

void initDefaultAMDKernelCodeT(amd_kernel_code_t &Header,
                               const FeatureBitset &Features);

amdhsa::kernel_descriptor_t getDefaultAmdhsaKernelDescriptor();

bool isGroupSegment(const GlobalValue *GV);
bool isGlobalSegment(const GlobalValue *GV);
bool isReadOnlySegment(const GlobalValue *GV);

/// \returns True if constants should be emitted to .text section for given
/// target triple \p TT, false otherwise.
bool shouldEmitConstantsToTextSection(const Triple &TT);

/// \returns Integer value requested using \p F's \p Name attribute.
///
/// \returns \p Default if attribute is not present.
///
/// \returns \p Default and emits error if requested value cannot be converted
/// to integer.
int getIntegerAttribute(const Function &F, StringRef Name, int Default);

/// \returns A pair of integer values requested using \p F's \p Name attribute
/// in "first[,second]" format ("second" is optional unless \p OnlyFirstRequired
/// is false).
///
/// \returns \p Default if attribute is not present.
///
/// \returns \p Default and emits error if one of the requested values cannot be
/// converted to integer, or \p OnlyFirstRequired is false and "second" value is
/// not present.
std::pair<int, int> getIntegerPairAttribute(const Function &F,
                                            StringRef Name,
                                            std::pair<int, int> Default,
                                            bool OnlyFirstRequired = false);

/// \returns Vmcnt bit mask for given isa \p Version.
unsigned getVmcntBitMask(const IsaInfo::IsaVersion &Version);

/// \returns Expcnt bit mask for given isa \p Version.
unsigned getExpcntBitMask(const IsaInfo::IsaVersion &Version);

/// \returns Lgkmcnt bit mask for given isa \p Version.
unsigned getLgkmcntBitMask(const IsaInfo::IsaVersion &Version);

/// \returns Waitcnt bit mask for given isa \p Version.
unsigned getWaitcntBitMask(const IsaInfo::IsaVersion &Version);

/// \returns Decoded Vmcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeVmcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt);

/// \returns Decoded Expcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeExpcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt);

/// \returns Decoded Lgkmcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeLgkmcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt);

/// Decodes Vmcnt, Expcnt and Lgkmcnt from given \p Waitcnt for given isa
/// \p Version, and writes decoded values into \p Vmcnt, \p Expcnt and
/// \p Lgkmcnt respectively.
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are decoded as follows:
///     \p Vmcnt = \p Waitcnt[3:0]                      (pre-gfx9 only)
///     \p Vmcnt = \p Waitcnt[3:0] | \p Waitcnt[15:14]  (gfx9+ only)
///     \p Expcnt = \p Waitcnt[6:4]
///     \p Lgkmcnt = \p Waitcnt[11:8]
void decodeWaitcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt,
                   unsigned &Vmcnt, unsigned &Expcnt, unsigned &Lgkmcnt);

/// \returns \p Waitcnt with encoded \p Vmcnt for given isa \p Version.
unsigned encodeVmcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt,
                     unsigned Vmcnt);

/// \returns \p Waitcnt with encoded \p Expcnt for given isa \p Version.
unsigned encodeExpcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt,
                      unsigned Expcnt);

/// \returns \p Waitcnt with encoded \p Lgkmcnt for given isa \p Version.
unsigned encodeLgkmcnt(const IsaInfo::IsaVersion &Version, unsigned Waitcnt,
                       unsigned Lgkmcnt);

/// Encodes \p Vmcnt, \p Expcnt and \p Lgkmcnt into Waitcnt for given isa
/// \p Version.
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are encoded as follows:
///     Waitcnt[3:0]   = \p Vmcnt       (pre-gfx9 only)
///     Waitcnt[3:0]   = \p Vmcnt[3:0]  (gfx9+ only)
///     Waitcnt[6:4]   = \p Expcnt
///     Waitcnt[11:8]  = \p Lgkmcnt
///     Waitcnt[15:14] = \p Vmcnt[5:4]  (gfx9+ only)
///
/// \returns Waitcnt with encoded \p Vmcnt, \p Expcnt and \p Lgkmcnt for given
/// isa \p Version.
unsigned encodeWaitcnt(const IsaInfo::IsaVersion &Version,
                       unsigned Vmcnt, unsigned Expcnt, unsigned Lgkmcnt);

unsigned getInitialPSInputAddr(const Function &F);

LLVM_READNONE
bool isShader(CallingConv::ID CC);

LLVM_READNONE
bool isCompute(CallingConv::ID CC);

LLVM_READNONE
bool isEntryFunctionCC(CallingConv::ID CC);

// FIXME: Remove this when calling conventions cleaned up
LLVM_READNONE
inline bool isKernel(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::AMDGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  default:
    return false;
  }
}

bool hasXNACK(const MCSubtargetInfo &STI);
bool hasMIMG_R128(const MCSubtargetInfo &STI);
bool hasPackedD16(const MCSubtargetInfo &STI);

bool isSI(const MCSubtargetInfo &STI);
bool isCI(const MCSubtargetInfo &STI);
bool isVI(const MCSubtargetInfo &STI);
bool isGFX9(const MCSubtargetInfo &STI);

/// Is Reg - scalar register
bool isSGPR(unsigned Reg, const MCRegisterInfo* TRI);

/// Is there any intersection between registers
bool isRegIntersect(unsigned Reg0, unsigned Reg1, const MCRegisterInfo* TRI);

/// If \p Reg is a pseudo reg, return the correct hardware register given
/// \p STI otherwise return \p Reg.
unsigned getMCReg(unsigned Reg, const MCSubtargetInfo &STI);

/// Convert hardware register \p Reg to a pseudo register
LLVM_READNONE
unsigned mc2PseudoReg(unsigned Reg);

/// Can this operand also contain immediate values?
bool isSISrcOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// Is this floating-point operand?
bool isSISrcFPOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// Does this opearnd support only inlinable literals?
bool isSISrcInlinableOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// Get the size in bits of a register from the register class \p RC.
unsigned getRegBitWidth(unsigned RCID);

/// Get the size in bits of a register from the register class \p RC.
unsigned getRegBitWidth(const MCRegisterClass &RC);

/// Get size of register operand
unsigned getRegOperandSize(const MCRegisterInfo *MRI, const MCInstrDesc &Desc,
                           unsigned OpNo);

LLVM_READNONE
inline unsigned getOperandSize(const MCOperandInfo &OpInfo) {
  switch (OpInfo.OperandType) {
  case AMDGPU::OPERAND_REG_IMM_INT32:
  case AMDGPU::OPERAND_REG_IMM_FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_INT32:
  case AMDGPU::OPERAND_REG_INLINE_C_FP32:
    return 4;

  case AMDGPU::OPERAND_REG_IMM_INT64:
  case AMDGPU::OPERAND_REG_IMM_FP64:
  case AMDGPU::OPERAND_REG_INLINE_C_INT64:
  case AMDGPU::OPERAND_REG_INLINE_C_FP64:
    return 8;

  case AMDGPU::OPERAND_REG_IMM_INT16:
  case AMDGPU::OPERAND_REG_IMM_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
    return 2;

  default:
    llvm_unreachable("unhandled operand type");
  }
}

LLVM_READNONE
inline unsigned getOperandSize(const MCInstrDesc &Desc, unsigned OpNo) {
  return getOperandSize(Desc.OpInfo[OpNo]);
}

/// Is this literal inlinable
LLVM_READNONE
bool isInlinableLiteral64(int64_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableLiteral32(int32_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableLiteral16(int16_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableLiteralV216(int32_t Literal, bool HasInv2Pi);

bool isArgPassedInSGPR(const Argument *Arg);

/// \returns The encoding that will be used for \p ByteOffset in the SMRD
/// offset field.
int64_t getSMRDEncodedOffset(const MCSubtargetInfo &ST, int64_t ByteOffset);

/// \returns true if this offset is small enough to fit in the SMRD
/// offset field.  \p ByteOffset should be the offset in bytes and
/// not the encoded offset.
bool isLegalSMRDImmOffset(const MCSubtargetInfo &ST, int64_t ByteOffset);

/// \returns true if the intrinsic is divergent
bool isIntrinsicSourceOfDivergence(unsigned IntrID);

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBASEINFO_H
