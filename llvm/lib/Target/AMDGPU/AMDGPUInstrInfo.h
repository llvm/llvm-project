//===-- AMDGPUInstrInfo.h - AMDGPU Instruction Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Contains the definition of a TargetInstrInfo class that is common
/// to all AMD GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUINSTRINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUINSTRINFO_H

#include "Utils/AMDGPUBaseInfo.h"

namespace llvm {

class MachineMemOperand;
class MachineInstr;

namespace AMDGPU {

bool isUniformMMO(const MachineMemOperand *MMO);

/// Return the intrinsic ID for opcodes with the G_AMDGPU_INTRIN_ prefix.
///
/// These opcodes have an Intrinsic::ID operand similar to a GIntrinsic. But
/// they are not actual instances of GIntrinsics, so we cannot use
/// GIntrinsic::getIntrinsicID() on them.
Intrinsic::ID getIntrinsicID(const MachineInstr &I);

struct RsrcIntrinsic {
  unsigned Intr;
  uint8_t RsrcArg;
  bool IsImage;
};
const RsrcIntrinsic *lookupRsrcIntrinsic(unsigned Intr);

#define GET_BufferIntrinsicKind_DECL
#include "AMDGPUGenSearchableTables.inc"

/// The argument layout of a buffer intrinsic. Argument positions are indices
/// into the IR call's argument list, with NoArg for arguments the intrinsic
/// doesn't have. Use getMIROperandIdx() / getSDOperandIdx() to convert them to
/// operand indices. See AMDGPUBufferIntrinsic in IntrinsicsAMDGPU.td.
struct BufferIntrinsicInfo {
  /// The position of absent arguments, (uint8_t)-1.
  static constexpr uint8_t NoArg = 255;

  unsigned Intr;
  BufferIntrinsicKind Kind;
  uint8_t RsrcArg;
  uint8_t VDataArg;
  uint8_t SizeArg;
  uint8_t VIndexArg;
  uint8_t VOffsetArg;
  uint8_t SOffsetArg;
  uint8_t ImmOffsetArg;
  uint8_t FormatArg;
  uint8_t AuxArg;
  /// The data is converted according to a format. This includes the typed
  /// (tbuffer) intrinsics.
  bool IsFormat;
  /// Loads to LDS that are asynchronous.
  bool IsAsync;

private:
  static std::optional<unsigned> getArg(uint8_t Arg) {
    if (Arg == NoArg)
      return std::nullopt;
    return Arg;
  }

public:
  bool isStruct() const { return VIndexArg != NoArg; }
  bool isTyped() const { return FormatArg != NoArg; }

  /// The stored value for stores and atomics.
  std::optional<unsigned> getVDataArg() const { return getArg(VDataArg); }
  /// The comparison value for compare-and-swap atomics.
  std::optional<unsigned> getCmpArg() const {
    if (Kind != BufferKindAtomicCmpSwap)
      return std::nullopt;
    return VDataArg + 1;
  }
  /// The LDS destination of loads to LDS.
  std::optional<unsigned> getLDSPtrArg() const {
    if (Kind != BufferKindLoadToLDS)
      return std::nullopt;
    return RsrcArg + 1;
  }
  /// The access size of loads to LDS and prefetches.
  std::optional<unsigned> getSizeArg() const { return getArg(SizeArg); }
  std::optional<unsigned> getVIndexArg() const { return getArg(VIndexArg); }
  /// The voffset, which for scalar loads and prefetches is their only offset.
  unsigned getVOffsetArg() const { return VOffsetArg; }
  std::optional<unsigned> getSOffsetArg() const { return getArg(SOffsetArg); }
  /// The immediate offset of loads to LDS.
  std::optional<unsigned> getImmOffsetArg() const {
    return getArg(ImmOffsetArg);
  }
  std::optional<unsigned> getFormatArg() const { return getArg(FormatArg); }
  /// The auxiliary (cache policy) argument.
  std::optional<unsigned> getAuxArg() const { return getArg(AuxArg); }
};
const BufferIntrinsicInfo *lookupBufferIntrinsic(unsigned Intr);

/// Convert IR argument index \p Arg of the intrinsic called by \p MI, a
/// G_INTRINSIC* instruction, into an operand index of \p MI.
unsigned getMIROperandIdx(const MachineInstr &MI, unsigned Arg);

/// Convert IR argument index \p Arg of an intrinsic into an operand index of
/// the INTRINSIC_* SelectionDAG node calling it. \p HasChain is false for
/// INTRINSIC_WO_CHAIN.
constexpr unsigned getSDOperandIdx(unsigned Arg, bool HasChain = true) {
  // The chain, if any, and the intrinsic ID precede the arguments.
  return Arg + (HasChain ? 2 : 1);
}

struct ImageDimIntrinsicInfo {
  unsigned Intr;
  unsigned BaseOpcode;
  unsigned AtomicNoRetBaseOpcode;
  MIMGDim Dim;

  uint8_t NumOffsetArgs;
  uint8_t NumBiasArgs;
  uint8_t NumZCompareArgs;
  uint8_t NumGradients;
  uint8_t NumDmask;
  uint8_t NumData;
  uint8_t NumVAddrs;
  uint8_t NumArgs;

  uint8_t DMaskIndex;
  uint8_t VAddrStart;
  uint8_t OffsetIndex;
  uint8_t BiasIndex;
  uint8_t ZCompareIndex;
  uint8_t GradientStart;
  uint8_t CoordStart;
  uint8_t LodIndex;
  uint8_t MipIndex;
  uint8_t VAddrEnd;
  uint8_t RsrcIndex;
  uint8_t SampIndex;
  uint8_t UnormIndex;
  uint8_t TexFailCtrlIndex;
  uint8_t CachePolicyIndex;

  uint8_t BiasTyArg;
  uint8_t GradientTyArg;
  uint8_t CoordTyArg;
};
const ImageDimIntrinsicInfo *getImageDimIntrinsicInfo(unsigned Intr);

const ImageDimIntrinsicInfo *
getImageDimIntrinsicByBaseOpcode(unsigned BaseOpcode, unsigned Dim);

} // end AMDGPU namespace
} // End llvm namespace

#endif
