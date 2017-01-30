//===-- AMDGPUBaseInfo.h - Top level definitions for AMDGPU -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBASEINFO_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPUBASEINFO_H

#include "AMDKernelCodeT.h"
#include "llvm/IR/CallingConv.h"

#include "SIDefines.h"

#define GET_INSTRINFO_OPERAND_ENUM
#include "AMDGPUGenInstrInfo.inc"
#undef GET_INSTRINFO_OPERAND_ENUM

namespace llvm {

class FeatureBitset;
class Function;
class GlobalValue;
class MachineMemOperand;
class MCContext;
class MCInstrDesc;
class MCRegisterClass;
class MCRegisterInfo;
class MCSection;
class MCSubtargetInfo;

namespace AMDGPU {

LLVM_READONLY
int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx);

struct IsaVersion {
  unsigned Major;
  unsigned Minor;
  unsigned Stepping;
};

IsaVersion getIsaVersion(const FeatureBitset &Features);
void initDefaultAMDKernelCodeT(amd_kernel_code_t &Header,
                               const FeatureBitset &Features);
MCSection *getHSATextSection(MCContext &Ctx);

MCSection *getHSADataGlobalAgentSection(MCContext &Ctx);

MCSection *getHSADataGlobalProgramSection(MCContext &Ctx);

MCSection *getHSARodataReadonlyAgentSection(MCContext &Ctx);

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

/// \returns Waitcnt bit mask for given isa \p Version.
unsigned getWaitcntBitMask(IsaVersion Version);

/// \returns Vmcnt bit mask for given isa \p Version.
unsigned getVmcntBitMask(IsaVersion Version);

/// \returns Expcnt bit mask for given isa \p Version.
unsigned getExpcntBitMask(IsaVersion Version);

/// \returns Lgkmcnt bit mask for given isa \p Version.
unsigned getLgkmcntBitMask(IsaVersion Version);

/// \returns Decoded Vmcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeVmcnt(IsaVersion Version, unsigned Waitcnt);

/// \returns Decoded Expcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeExpcnt(IsaVersion Version, unsigned Waitcnt);

/// \returns Decoded Lgkmcnt from given \p Waitcnt for given isa \p Version.
unsigned decodeLgkmcnt(IsaVersion Version, unsigned Waitcnt);

/// \brief Decodes Vmcnt, Expcnt and Lgkmcnt from given \p Waitcnt for given isa
/// \p Version, and writes decoded values into \p Vmcnt, \p Expcnt and
/// \p Lgkmcnt respectively.
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are decoded as follows:
///     \p Vmcnt = \p Waitcnt[3:0]
///     \p Expcnt = \p Waitcnt[6:4]
///     \p Lgkmcnt = \p Waitcnt[11:8]
void decodeWaitcnt(IsaVersion Version, unsigned Waitcnt,
                   unsigned &Vmcnt, unsigned &Expcnt, unsigned &Lgkmcnt);

/// \returns \p Waitcnt with encoded \p Vmcnt for given isa \p Version.
unsigned encodeVmcnt(IsaVersion Version, unsigned Waitcnt, unsigned Vmcnt);

/// \returns \p Waitcnt with encoded \p Expcnt for given isa \p Version.
unsigned encodeExpcnt(IsaVersion Version, unsigned Waitcnt, unsigned Expcnt);

/// \returns \p Waitcnt with encoded \p Lgkmcnt for given isa \p Version.
unsigned encodeLgkmcnt(IsaVersion Version, unsigned Waitcnt, unsigned Lgkmcnt);

/// \brief Encodes \p Vmcnt, \p Expcnt and \p Lgkmcnt into Waitcnt for given isa
/// \p Version.
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are encoded as follows:
///     Waitcnt[3:0]  = \p Vmcnt
///     Waitcnt[6:4]  = \p Expcnt
///     Waitcnt[11:8] = \p Lgkmcnt
///
/// \returns Waitcnt with encoded \p Vmcnt, \p Expcnt and \p Lgkmcnt for given
/// isa \p Version.
unsigned encodeWaitcnt(IsaVersion Version,
                       unsigned Vmcnt, unsigned Expcnt, unsigned Lgkmcnt);

unsigned getInitialPSInputAddr(const Function &F);

bool isShader(CallingConv::ID cc);
bool isCompute(CallingConv::ID cc);

bool isSI(const MCSubtargetInfo &STI);
bool isCI(const MCSubtargetInfo &STI);
bool isVI(const MCSubtargetInfo &STI);

/// If \p Reg is a pseudo reg, return the correct hardware register given
/// \p STI otherwise return \p Reg.
unsigned getMCReg(unsigned Reg, const MCSubtargetInfo &STI);

/// \brief Can this operand also contain immediate values?
bool isSISrcOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// \brief Is this floating-point operand?
bool isSISrcFPOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// \brief Does this opearnd support only inlinable literals?
bool isSISrcInlinableOperand(const MCInstrDesc &Desc, unsigned OpNo);

/// \brief Get the size in bits of a register from the register class \p RC.
unsigned getRegBitWidth(unsigned RCID);

/// \brief Get the size in bits of a register from the register class \p RC.
unsigned getRegBitWidth(const MCRegisterClass &RC);

/// \brief Get size of register operand
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
    return 2;

  default:
    llvm_unreachable("unhandled operand type");
  }
}

LLVM_READNONE
inline unsigned getOperandSize(const MCInstrDesc &Desc, unsigned OpNo) {
  return getOperandSize(Desc.OpInfo[OpNo]);
}

/// \brief Is this literal inlinable
LLVM_READNONE
bool isInlinableLiteral64(int64_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableLiteral32(int32_t Literal, bool HasInv2Pi);

LLVM_READNONE
bool isInlinableLiteral16(int16_t Literal, bool HasInv2Pi);

bool isUniformMMO(const MachineMemOperand *MMO);

/// \returns The encoding that will be used for \p ByteOffset in the SMRD
/// offset field.
int64_t getSMRDEncodedOffset(const MCSubtargetInfo &ST, int64_t ByteOffset);

/// \returns true if this offset is small enough to fit in the SMRD
/// offset field.  \p ByteOffset should be the offset in bytes and
/// not the encoded offset.
bool isLegalSMRDImmOffset(const MCSubtargetInfo &ST, int64_t ByteOffset);

} // end namespace AMDGPU
} // end namespace llvm

#endif
