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

namespace llvm {

class FeatureBitset;
class Function;
class GlobalValue;
class MCContext;
class MCInstrDesc;
class MCRegisterInfo;
class MCSection;
class MCSubtargetInfo;

namespace AMDGPU {

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

/// \brief Get size of register operand
unsigned getRegOperandSize(const MCRegisterInfo *MRI, const MCInstrDesc &Desc,
                           unsigned OpNo);

/// \brief Is this literal inlinable
bool isInlinableLiteral64(int64_t Literal, bool IsVI);
bool isInlinableLiteral32(int32_t Literal, bool IsVI);

} // end namespace AMDGPU
} // end namespace llvm

#endif
