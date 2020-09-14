//===- AMDGPUBaseInfo.h - Top level definitions for AMDGPU ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBASEINFO_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBASEINFO_H

#include "AMDGPU.h"
#include "AMDKernelCodeT.h"
#include "SIDefines.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetParser.h"
#include <cstdint>
#include <string>
#include <utility>

namespace llvm {

class Argument;
class Function;
class GCNSubtarget;
class GlobalValue;
class MCRegisterClass;
class MCRegisterInfo;
class MCSubtargetInfo;
class StringRef;
class Triple;

namespace AMDGPU {

struct GcnBufferFormatInfo {
  unsigned Format;
  unsigned BitsPerComp;
  unsigned NumComponents;
  unsigned NumFormat;
  unsigned DataFormat;
};

#define GET_MIMGBaseOpcode_DECL
#define GET_MIMGDim_DECL
#define GET_MIMGEncoding_DECL
#define GET_MIMGLZMapping_DECL
#define GET_MIMGMIPMapping_DECL
#include "AMDGPUGenSearchableTables.inc"

namespace IsaInfo {

enum {
  // The closed Vulkan driver sets 96, which limits the wave count to 8 but
  // doesn't spill SGPRs as much as when 80 is set.
  FIXED_NUM_SGPRS_FOR_INIT_BUG = 96,
  TRAP_NUM_SGPRS = 16
};

/// Streams isa version string for given subtarget \p STI into \p Stream.
void streamIsaVersion(const MCSubtargetInfo *STI, raw_ostream &Stream);

/// \returns True if given subtarget \p STI supports code object version 3,
/// false otherwise.
bool hasCodeObjectV3(const MCSubtargetInfo *STI);

/// \returns Wavefront size for given subtarget \p STI.
unsigned getWavefrontSize(const MCSubtargetInfo *STI);

/// \returns Local memory size in bytes for given subtarget \p STI.
unsigned getLocalMemorySize(const MCSubtargetInfo *STI);

/// \returns Number of execution units per compute unit for given subtarget \p
/// STI.
unsigned getEUsPerCU(const MCSubtargetInfo *STI);

/// \returns Maximum number of work groups per compute unit for given subtarget
/// \p STI and limited by given \p FlatWorkGroupSize.
unsigned getMaxWorkGroupsPerCU(const MCSubtargetInfo *STI,
                               unsigned FlatWorkGroupSize);

/// \returns Minimum number of waves per execution unit for given subtarget \p
/// STI.
unsigned getMinWavesPerEU(const MCSubtargetInfo *STI);

/// \returns Maximum number of waves per execution unit for given subtarget \p
/// STI without any kind of limitation.
unsigned getMaxWavesPerEU(const MCSubtargetInfo *STI);

/// \returns Number of waves per execution unit required to support the given \p
/// FlatWorkGroupSize.
unsigned getWavesPerEUForWorkGroup(const MCSubtargetInfo *STI,
                                   unsigned FlatWorkGroupSize);

/// \returns Minimum flat work group size for given subtarget \p STI.
unsigned getMinFlatWorkGroupSize(const MCSubtargetInfo *STI);

/// \returns Maximum flat work group size for given subtarget \p STI.
unsigned getMaxFlatWorkGroupSize(const MCSubtargetInfo *STI);

/// \returns Number of waves per work group for given subtarget \p STI and
/// \p FlatWorkGroupSize.
unsigned getWavesPerWorkGroup(const MCSubtargetInfo *STI,
                              unsigned FlatWorkGroupSize);

/// \returns SGPR allocation granularity for given subtarget \p STI.
unsigned getSGPRAllocGranule(const MCSubtargetInfo *STI);

/// \returns SGPR encoding granularity for given subtarget \p STI.
unsigned getSGPREncodingGranule(const MCSubtargetInfo *STI);

/// \returns Total number of SGPRs for given subtarget \p STI.
unsigned getTotalNumSGPRs(const MCSubtargetInfo *STI);

/// \returns Addressable number of SGPRs for given subtarget \p STI.
unsigned getAddressableNumSGPRs(const MCSubtargetInfo *STI);

/// \returns Minimum number of SGPRs that meets the given number of waves per
/// execution unit requirement for given subtarget \p STI.
unsigned getMinNumSGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU);

/// \returns Maximum number of SGPRs that meets the given number of waves per
/// execution unit requirement for given subtarget \p STI.
unsigned getMaxNumSGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU,
                        bool Addressable);

/// \returns Number of extra SGPRs implicitly required by given subtarget \p
/// STI when the given special registers are used.
unsigned getNumExtraSGPRs(const MCSubtargetInfo *STI, bool VCCUsed,
                          bool FlatScrUsed, bool XNACKUsed);

/// \returns Number of extra SGPRs implicitly required by given subtarget \p
/// STI when the given special registers are used. XNACK is inferred from
/// \p STI.
unsigned getNumExtraSGPRs(const MCSubtargetInfo *STI, bool VCCUsed,
                          bool FlatScrUsed);

/// \returns Number of SGPR blocks needed for given subtarget \p STI when
/// \p NumSGPRs are used. \p NumSGPRs should already include any special
/// register counts.
unsigned getNumSGPRBlocks(const MCSubtargetInfo *STI, unsigned NumSGPRs);

/// \returns VGPR allocation granularity for given subtarget \p STI.
///
/// For subtargets which support it, \p EnableWavefrontSize32 should match
/// the ENABLE_WAVEFRONT_SIZE32 kernel descriptor field.
unsigned getVGPRAllocGranule(const MCSubtargetInfo *STI,
                             Optional<bool> EnableWavefrontSize32 = None);

/// \returns VGPR encoding granularity for given subtarget \p STI.
///
/// For subtargets which support it, \p EnableWavefrontSize32 should match
/// the ENABLE_WAVEFRONT_SIZE32 kernel descriptor field.
unsigned getVGPREncodingGranule(const MCSubtargetInfo *STI,
                                Optional<bool> EnableWavefrontSize32 = None);

/// \returns Total number of VGPRs for given subtarget \p STI.
unsigned getTotalNumVGPRs(const MCSubtargetInfo *STI);

/// \returns Addressable number of VGPRs for given subtarget \p STI.
unsigned getAddressableNumVGPRs(const MCSubtargetInfo *STI);

/// \returns Minimum number of VGPRs that meets given number of waves per
/// execution unit requirement for given subtarget \p STI.
unsigned getMinNumVGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU);

/// \returns Maximum number of VGPRs that meets given number of waves per
/// execution unit requirement for given subtarget \p STI.
unsigned getMaxNumVGPRs(const MCSubtargetInfo *STI, unsigned WavesPerEU);

/// \returns Number of VGPR blocks needed for given subtarget \p STI when
/// \p NumVGPRs are used.
///
/// For subtargets which support it, \p EnableWavefrontSize32 should match the
/// ENABLE_WAVEFRONT_SIZE32 kernel descriptor field.
unsigned getNumVGPRBlocks(const MCSubtargetInfo *STI, unsigned NumSGPRs,
                          Optional<bool> EnableWavefrontSize32 = None);

} // end namespace IsaInfo

LLVM_READONLY
int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx);

LLVM_READONLY
int getSOPPWithRelaxation(uint16_t Opcode);

struct MIMGBaseOpcodeInfo {
  MIMGBaseOpcode BaseOpcode;
  bool Store;
  bool Atomic;
  bool AtomicX2;
  bool Sampler;
  bool Gather4;

  uint8_t NumExtraArgs;
  bool Gradients;
  bool G16;
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
  uint8_t Encoding;
  const char *AsmSuffix;
};

LLVM_READONLY
const MIMGDimInfo *getMIMGDimInfo(unsigned DimEnum);

LLVM_READONLY
const MIMGDimInfo *getMIMGDimInfoByEncoding(uint8_t DimEnc);

LLVM_READONLY
const MIMGDimInfo *getMIMGDimInfoByAsmSuffix(StringRef AsmSuffix);

struct MIMGLZMappingInfo {
  MIMGBaseOpcode L;
  MIMGBaseOpcode LZ;
};

struct MIMGMIPMappingInfo {
  MIMGBaseOpcode MIP;
  MIMGBaseOpcode NONMIP;
};

struct MIMGG16MappingInfo {
  MIMGBaseOpcode G;
  MIMGBaseOpcode G16;
};

LLVM_READONLY
const MIMGLZMappingInfo *getMIMGLZMappingInfo(unsigned L);

LLVM_READONLY
const MIMGMIPMappingInfo *getMIMGMIPMappingInfo(unsigned MIP);

LLVM_READONLY
const MIMGG16MappingInfo *getMIMGG16MappingInfo(unsigned G);

LLVM_READONLY
int getMIMGOpcode(unsigned BaseOpcode, unsigned MIMGEncoding,
                  unsigned VDataDwords, unsigned VAddrDwords);

LLVM_READONLY
int getMaskedMIMGOp(unsigned Opc, unsigned NewChannels);

struct MIMGInfo {
  uint16_t Opcode;
  uint16_t BaseOpcode;
  uint8_t MIMGEncoding;
  uint8_t VDataDwords;
  uint8_t VAddrDwords;
};

LLVM_READONLY
const MIMGInfo *getMIMGInfo(unsigned Opc);

LLVM_READONLY
int getMTBUFBaseOpcode(unsigned Opc);

LLVM_READONLY
int getMTBUFOpcode(unsigned BaseOpc, unsigned Elements);

LLVM_READONLY
int getMTBUFElements(unsigned Opc);

LLVM_READONLY
bool getMTBUFHasVAddr(unsigned Opc);

LLVM_READONLY
bool getMTBUFHasSrsrc(unsigned Opc);

LLVM_READONLY
bool getMTBUFHasSoffset(unsigned Opc);

LLVM_READONLY
int getMUBUFBaseOpcode(unsigned Opc);

LLVM_READONLY
int getMUBUFOpcode(unsigned BaseOpc, unsigned Elements);

LLVM_READONLY
int getMUBUFElements(unsigned Opc);

LLVM_READONLY
bool getMUBUFHasVAddr(unsigned Opc);

LLVM_READONLY
bool getMUBUFHasSrsrc(unsigned Opc);

LLVM_READONLY
bool getMUBUFHasSoffset(unsigned Opc);

LLVM_READONLY
bool getSMEMIsBuffer(unsigned Opc);

LLVM_READONLY
const GcnBufferFormatInfo *getGcnBufferFormatInfo(uint8_t BitsPerComp,
                                                  uint8_t NumComponents,
                                                  uint8_t NumFormat,
                                                  const MCSubtargetInfo &STI);
LLVM_READONLY
const GcnBufferFormatInfo *getGcnBufferFormatInfo(uint8_t Format,
                                                  const MCSubtargetInfo &STI);

LLVM_READONLY
int getMCOpcode(uint16_t Opcode, unsigned Gen);

void initDefaultAMDKernelCodeT(amd_kernel_code_t &Header,
                               const MCSubtargetInfo *STI);

amdhsa::kernel_descriptor_t getDefaultAmdhsaKernelDescriptor(
    const MCSubtargetInfo *STI);

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

/// Represents the counter values to wait for in an s_waitcnt instruction.
///
/// Large values (including the maximum possible integer) can be used to
/// represent "don't care" waits.
struct Waitcnt {
  unsigned VmCnt = ~0u;
  unsigned ExpCnt = ~0u;
  unsigned LgkmCnt = ~0u;
  unsigned VsCnt = ~0u;

  Waitcnt() {}
  Waitcnt(unsigned VmCnt, unsigned ExpCnt, unsigned LgkmCnt, unsigned VsCnt)
      : VmCnt(VmCnt), ExpCnt(ExpCnt), LgkmCnt(LgkmCnt), VsCnt(VsCnt) {}

  static Waitcnt allZero(const IsaVersion &Version) {
    return Waitcnt(0, 0, 0, Version.Major >= 10 ? 0 : ~0u);
  }
  static Waitcnt allZeroExceptVsCnt() { return Waitcnt(0, 0, 0, ~0u); }

  bool hasWait() const {
    return VmCnt != ~0u || ExpCnt != ~0u || LgkmCnt != ~0u || VsCnt != ~0u;
  }

  bool dominates(const Waitcnt &Other) const {
    return VmCnt <= Other.VmCnt && ExpCnt <= Other.ExpCnt &&
           LgkmCnt <= Other.LgkmCnt && VsCnt <= Other.VsCnt;
  }

  Waitcnt combined(const Waitcnt &Other) const {
    return Waitcnt(std::min(VmCnt, Other.VmCnt), std::min(ExpCnt, Other.ExpCnt),
                   std::min(LgkmCnt, Other.LgkmCnt),
                   std::min(VsCnt, Other.VsCnt));
  }
};

/// \returns Vmcnt bit mask for given isa \p Version.
unsigned getVmcntBitMask(const IsaVersion &Version);

/// \returns Expcnt bit mask for given isa \p Version.
unsigned getExpcntBitMask(const IsaVersion &Version);

/// \returns Lgkmcnt bit mask for given isa \p Version.
unsigned getLgkmcntBitMask(const IsaVersion &Version);

/// \returns Waitcnt bit mask for given isa \p Version.
unsigned getWaitcntBitMask(const IsaVersion &Version);

/// \returns Decoded Vmcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeVmcnt(const IsaVersion &Version, unsigned Waitcnt);

/// \returns Decoded Expcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeExpcnt(const IsaVersion &Version, unsigned Waitcnt);

/// \returns Decoded Lgkmcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeLgkmcnt(const IsaVersion &Version, unsigned Waitcnt);

/// Decodes Vmcnt, Expcnt and Lgkmcnt from given \p Waitcnt for given isa
/// \p Version, and writes decoded values into \p Vmcnt, \p Expcnt and
/// \p Lgkmcnt respectively.
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are decoded as follows:
///     \p Vmcnt = \p Waitcnt[3:0]                      (pre-gfx9 only)
///     \p Vmcnt = \p Waitcnt[3:0] | \p Waitcnt[15:14]  (gfx9+ only)
///     \p Expcnt = \p Waitcnt[6:4]
///     \p Lgkmcnt = \p Waitcnt[11:8]                   (pre-gfx10 only)
///     \p Lgkmcnt = \p Waitcnt[13:8]                   (gfx10+ only)
void decodeWaitcnt(const IsaVersion &Version, unsigned Waitcnt,
                   unsigned &Vmcnt, unsigned &Expcnt, unsigned &Lgkmcnt);

Waitcnt decodeWaitcnt(const IsaVersion &Version, unsigned Encoded);

/// \returns \p Waitcnt with encoded \p Vmcnt for given isa \p Version.
unsigned encodeVmcnt(const IsaVersion &Version, unsigned Waitcnt,
                     unsigned Vmcnt);

/// \returns \p Waitcnt with encoded \p Expcnt for given isa \p Version.
unsigned encodeExpcnt(const IsaVersion &Version, unsigned Waitcnt,
                      unsigned Expcnt);

/// \returns \p Waitcnt with encoded \p Lgkmcnt for given isa \p Version.
unsigned encodeLgkmcnt(const IsaVersion &Version, unsigned Waitcnt,
                       unsigned Lgkmcnt);

/// Encodes \p Vmcnt, \p Expcnt and \p Lgkmcnt into Waitcnt for given isa
/// \p Version.
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are encoded as follows:
///     Waitcnt[3:0]   = \p Vmcnt       (pre-gfx9 only)
///     Waitcnt[3:0]   = \p Vmcnt[3:0]  (gfx9+ only)
///     Waitcnt[6:4]   = \p Expcnt
///     Waitcnt[11:8]  = \p Lgkmcnt     (pre-gfx10 only)
///     Waitcnt[13:8]  = \p Lgkmcnt     (gfx10+ only)
///     Waitcnt[15:14] = \p Vmcnt[5:4]  (gfx9+ only)
///
/// \returns Waitcnt with encoded \p Vmcnt, \p Expcnt and \p Lgkmcnt for given
/// isa \p Version.
unsigned encodeWaitcnt(const IsaVersion &Version,
                       unsigned Vmcnt, unsigned Expcnt, unsigned Lgkmcnt);

unsigned encodeWaitcnt(const IsaVersion &Version, const Waitcnt &Decoded);

namespace Hwreg {

LLVM_READONLY
int64_t getHwregId(const StringRef Name);

LLVM_READNONE
bool isValidHwreg(int64_t Id, const MCSubtargetInfo &STI);

LLVM_READNONE
bool isValidHwreg(int64_t Id);

LLVM_READNONE
bool isValidHwregOffset(int64_t Offset);

LLVM_READNONE
bool isValidHwregWidth(int64_t Width);

LLVM_READNONE
uint64_t encodeHwreg(uint64_t Id, uint64_t Offset, uint64_t Width);

LLVM_READNONE
StringRef getHwreg(unsigned Id, const MCSubtargetInfo &STI);

void decodeHwreg(unsigned Val, unsigned &Id, unsigned &Offset, unsigned &Width);

} // namespace Hwreg

namespace SendMsg {

LLVM_READONLY
int64_t getMsgId(const StringRef Name);

LLVM_READONLY
int64_t getMsgOpId(int64_t MsgId, const StringRef Name);

LLVM_READNONE
StringRef getMsgName(int64_t MsgId);

LLVM_READNONE
StringRef getMsgOpName(int64_t MsgId, int64_t OpId);

LLVM_READNONE
bool isValidMsgId(int64_t MsgId, const MCSubtargetInfo &STI, bool Strict = true);

LLVM_READNONE
bool isValidMsgOp(int64_t MsgId, int64_t OpId, bool Strict = true);

LLVM_READNONE
bool isValidMsgStream(int64_t MsgId, int64_t OpId, int64_t StreamId, bool Strict = true);

LLVM_READNONE
bool msgRequiresOp(int64_t MsgId);

LLVM_READNONE
bool msgSupportsStream(int64_t MsgId, int64_t OpId);

void decodeMsg(unsigned Val,
               uint16_t &MsgId,
               uint16_t &OpId,
               uint16_t &StreamId);

LLVM_READNONE
uint64_t encodeMsg(uint64_t MsgId,
                   uint64_t OpId,
                   uint64_t StreamId);

} // namespace SendMsg


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
bool hasSRAMECC(const MCSubtargetInfo &STI);
bool hasMIMG_R128(const MCSubtargetInfo &STI);
bool hasGFX10A16(const MCSubtargetInfo &STI);
bool hasG16(const MCSubtargetInfo &STI);
bool hasPackedD16(const MCSubtargetInfo &STI);

bool isSI(const MCSubtargetInfo &STI);
bool isCI(const MCSubtargetInfo &STI);
bool isVI(const MCSubtargetInfo &STI);
bool isGFX9(const MCSubtargetInfo &STI);
bool isGFX10(const MCSubtargetInfo &STI);
bool isGCN3Encoding(const MCSubtargetInfo &STI);
bool isGFX10_BEncoding(const MCSubtargetInfo &STI);
bool hasGFX10_3Insts(const MCSubtargetInfo &STI);

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
  case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
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
  case AMDGPU::OPERAND_REG_INLINE_AC_INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2FP16:
  case AMDGPU::OPERAND_REG_IMM_V2INT16:
  case AMDGPU::OPERAND_REG_IMM_V2FP16:
    return 2;

  default:
    llvm_unreachable("unhandled operand type");
  }
}

LLVM_READNONE
inline unsigned getOperandSize(const MCInstrDesc &Desc, unsigned OpNo) {
  return getOperandSize(Desc.OpInfo[OpNo]);
}

/// Is this literal inlinable, and not one of the values intended for floating
/// point values.
LLVM_READNONE
inline bool isInlinableIntLiteral(int64_t Literal) {
  return Literal >= -16 && Literal <= 64;
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

LLVM_READNONE
bool isInlinableIntLiteralV216(int32_t Literal);

LLVM_READNONE
bool isFoldableLiteralV216(int32_t Literal, bool HasInv2Pi);

bool isArgPassedInSGPR(const Argument *Arg);

LLVM_READONLY
bool isLegalSMRDEncodedUnsignedOffset(const MCSubtargetInfo &ST,
                                      int64_t EncodedOffset);

LLVM_READONLY
bool isLegalSMRDEncodedSignedOffset(const MCSubtargetInfo &ST,
                                    int64_t EncodedOffset,
                                    bool IsBuffer);

/// Convert \p ByteOffset to dwords if the subtarget uses dword SMRD immediate
/// offsets.
uint64_t convertSMRDOffsetUnits(const MCSubtargetInfo &ST, uint64_t ByteOffset);

/// \returns The encoding that will be used for \p ByteOffset in the
/// SMRD offset field, or None if it won't fit. On GFX9 and GFX10
/// S_LOAD instructions have a signed offset, on other subtargets it is
/// unsigned. S_BUFFER has an unsigned offset for all subtargets.
Optional<int64_t> getSMRDEncodedOffset(const MCSubtargetInfo &ST,
                                       int64_t ByteOffset, bool IsBuffer);

/// \return The encoding that can be used for a 32-bit literal offset in an SMRD
/// instruction. This is only useful on CI.s
Optional<int64_t> getSMRDEncodedLiteralOffset32(const MCSubtargetInfo &ST,
                                                int64_t ByteOffset);

/// \returns true if this offset is small enough to fit in the SMRD
/// offset field.  \p ByteOffset should be the offset in bytes and
/// not the encoded offset.
bool isLegalSMRDImmOffset(const MCSubtargetInfo &ST, int64_t ByteOffset);

bool splitMUBUFOffset(uint32_t Imm, uint32_t &SOffset, uint32_t &ImmOffset,
                      const GCNSubtarget *Subtarget,
                      Align Alignment = Align(4));

/// \returns true if the intrinsic is divergent
bool isIntrinsicSourceOfDivergence(unsigned IntrID);

// Track defaults for fields in the MODE registser.
struct SIModeRegisterDefaults {
  /// Floating point opcodes that support exception flag gathering quiet and
  /// propagate signaling NaN inputs per IEEE 754-2008. Min_dx10 and max_dx10
  /// become IEEE 754- 2008 compliant due to signaling NaN propagation and
  /// quieting.
  bool IEEE : 1;

  /// Used by the vector ALU to force DX10-style treatment of NaNs: when set,
  /// clamp NaN to zero; otherwise, pass NaN through.
  bool DX10Clamp : 1;

  /// If this is set, neither input or output denormals are flushed for most f32
  /// instructions.
  bool FP32InputDenormals : 1;
  bool FP32OutputDenormals : 1;

  /// If this is set, neither input or output denormals are flushed for both f64
  /// and f16/v2f16 instructions.
  bool FP64FP16InputDenormals : 1;
  bool FP64FP16OutputDenormals : 1;

  SIModeRegisterDefaults() :
    IEEE(true),
    DX10Clamp(true),
    FP32InputDenormals(true),
    FP32OutputDenormals(true),
    FP64FP16InputDenormals(true),
    FP64FP16OutputDenormals(true) {}

  SIModeRegisterDefaults(const Function &F);

  static SIModeRegisterDefaults getDefaultForCallingConv(CallingConv::ID CC) {
    const bool IsCompute = AMDGPU::isCompute(CC);

    SIModeRegisterDefaults Mode;
    Mode.IEEE = IsCompute;
    return Mode;
  }

  bool operator ==(const SIModeRegisterDefaults Other) const {
    return IEEE == Other.IEEE && DX10Clamp == Other.DX10Clamp &&
           FP32InputDenormals == Other.FP32InputDenormals &&
           FP32OutputDenormals == Other.FP32OutputDenormals &&
           FP64FP16InputDenormals == Other.FP64FP16InputDenormals &&
           FP64FP16OutputDenormals == Other.FP64FP16OutputDenormals;
  }

  bool allFP32Denormals() const {
    return FP32InputDenormals && FP32OutputDenormals;
  }

  bool allFP64FP16Denormals() const {
    return FP64FP16InputDenormals && FP64FP16OutputDenormals;
  }

  /// Get the encoding value for the FP_DENORM bits of the mode register for the
  /// FP32 denormal mode.
  uint32_t fpDenormModeSPValue() const {
    if (FP32InputDenormals && FP32OutputDenormals)
      return FP_DENORM_FLUSH_NONE;
    if (FP32InputDenormals)
      return FP_DENORM_FLUSH_OUT;
    if (FP32OutputDenormals)
      return FP_DENORM_FLUSH_IN;
    return FP_DENORM_FLUSH_IN_FLUSH_OUT;
  }

  /// Get the encoding value for the FP_DENORM bits of the mode register for the
  /// FP64/FP16 denormal mode.
  uint32_t fpDenormModeDPValue() const {
    if (FP64FP16InputDenormals && FP64FP16OutputDenormals)
      return FP_DENORM_FLUSH_NONE;
    if (FP64FP16InputDenormals)
      return FP_DENORM_FLUSH_OUT;
    if (FP64FP16OutputDenormals)
      return FP_DENORM_FLUSH_IN;
    return FP_DENORM_FLUSH_IN_FLUSH_OUT;
  }

  /// Returns true if a flag is compatible if it's enabled in the callee, but
  /// disabled in the caller.
  static bool oneWayCompatible(bool CallerMode, bool CalleeMode) {
    return CallerMode == CalleeMode || (!CallerMode && CalleeMode);
  }

  // FIXME: Inlining should be OK for dx10-clamp, since the caller's mode should
  // be able to override.
  bool isInlineCompatible(SIModeRegisterDefaults CalleeMode) const {
    if (DX10Clamp != CalleeMode.DX10Clamp)
      return false;
    if (IEEE != CalleeMode.IEEE)
      return false;

    // Allow inlining denormals enabled into denormals flushed functions.
    return oneWayCompatible(FP64FP16InputDenormals, CalleeMode.FP64FP16InputDenormals) &&
           oneWayCompatible(FP64FP16OutputDenormals, CalleeMode.FP64FP16OutputDenormals) &&
           oneWayCompatible(FP32InputDenormals, CalleeMode.FP32InputDenormals) &&
           oneWayCompatible(FP32OutputDenormals, CalleeMode.FP32OutputDenormals);
  }
};

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBASEINFO_H
