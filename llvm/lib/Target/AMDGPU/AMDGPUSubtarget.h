//=====-- AMDGPUSubtarget.h - Define Subtarget for AMDGPU -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// Base class for AMDGPU specific classes of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSUBTARGET_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSUBTARGET_H

#include "llvm/ADT/Triple.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"

namespace llvm {

enum AMDGPUDwarfFlavour : unsigned;
class Function;
class Instruction;
class MachineFunction;
class TargetMachine;

class AMDGPUSubtarget {
public:
  enum Generation {
    INVALID = 0,
    R600 = 1,
    R700 = 2,
    EVERGREEN = 3,
    NORTHERN_ISLANDS = 4,
    SOUTHERN_ISLANDS = 5,
    SEA_ISLANDS = 6,
    VOLCANIC_ISLANDS = 7,
    GFX9 = 8,
    GFX10 = 9
  };

private:
  Triple TargetTriple;

protected:
  bool GCN3Encoding;
  bool Has16BitInsts;
  bool HasMadMixInsts;
  bool HasMadMacF32Insts;
  bool HasDsSrc2Insts;
  bool HasSDWA;
  bool HasVOP3PInsts;
  bool HasMulI24;
  bool HasMulU24;
  bool HasSMulHi;
  bool HasInv2PiInlineImm;
  bool HasFminFmaxLegacy;
  bool EnablePromoteAlloca;
  bool HasTrigReducedRange;
  unsigned MaxWavesPerEU;
  unsigned LocalMemorySize;
  char WavefrontSizeLog2;

public:
  AMDGPUSubtarget(const Triple &TT);

  static const AMDGPUSubtarget &get(const MachineFunction &MF);
  static const AMDGPUSubtarget &get(const TargetMachine &TM,
                                    const Function &F);

  /// \returns Default range flat work group size for a calling convention.
  std::pair<unsigned, unsigned> getDefaultFlatWorkGroupSize(CallingConv::ID CC) const;

  /// \returns Subtarget's default pair of minimum/maximum flat work group sizes
  /// for function \p F, or minimum/maximum flat work group sizes explicitly
  /// requested using "amdgpu-flat-work-group-size" attribute attached to
  /// function \p F.
  ///
  /// \returns Subtarget's default values if explicitly requested values cannot
  /// be converted to integer, or violate subtarget's specifications.
  std::pair<unsigned, unsigned> getFlatWorkGroupSizes(const Function &F) const;

  /// \returns Subtarget's default pair of minimum/maximum number of waves per
  /// execution unit for function \p F, or minimum/maximum number of waves per
  /// execution unit explicitly requested using "amdgpu-waves-per-eu" attribute
  /// attached to function \p F.
  ///
  /// \returns Subtarget's default values if explicitly requested values cannot
  /// be converted to integer, violate subtarget's specifications, or are not
  /// compatible with minimum/maximum number of waves limited by flat work group
  /// size, register usage, and/or lds usage.
  std::pair<unsigned, unsigned> getWavesPerEU(const Function &F) const {
    // Default/requested minimum/maximum flat work group sizes.
    std::pair<unsigned, unsigned> FlatWorkGroupSizes = getFlatWorkGroupSizes(F);
    return getWavesPerEU(F, FlatWorkGroupSizes);
  }

  /// Overload which uses the specified values for the flat work group sizes,
  /// rather than querying the function itself. \p FlatWorkGroupSizes Should
  /// correspond to the function's value for getFlatWorkGroupSizes.
  std::pair<unsigned, unsigned>
  getWavesPerEU(const Function &F,
                std::pair<unsigned, unsigned> FlatWorkGroupSizes) const;

  /// Return the amount of LDS that can be used that will not restrict the
  /// occupancy lower than WaveCount.
  unsigned getMaxLocalMemSizeWithWaveCount(unsigned WaveCount,
                                           const Function &) const;

  /// Inverse of getMaxLocalMemWithWaveCount. Return the maximum wavecount if
  /// the given LDS memory size is the only constraint.
  unsigned getOccupancyWithLocalMemSize(uint32_t Bytes, const Function &) const;

  unsigned getOccupancyWithLocalMemSize(const MachineFunction &MF) const;

  bool isAmdHsaOS() const {
    return TargetTriple.getOS() == Triple::AMDHSA;
  }

  bool isAmdPalOS() const {
    return TargetTriple.getOS() == Triple::AMDPAL;
  }

  bool isMesa3DOS() const {
    return TargetTriple.getOS() == Triple::Mesa3D;
  }

  bool isMesaKernel(const Function &F) const;

  bool isAmdHsaOrMesa(const Function &F) const {
    return isAmdHsaOS() || isMesaKernel(F);
  }

  bool isGCN() const {
    return TargetTriple.getArch() == Triple::amdgcn;
  }

  bool isGCN3Encoding() const {
    return GCN3Encoding;
  }

  bool has16BitInsts() const {
    return Has16BitInsts;
  }

  bool hasMadMixInsts() const {
    return HasMadMixInsts;
  }

  bool hasMadMacF32Insts() const {
    return HasMadMacF32Insts || !isGCN();
  }

  bool hasDsSrc2Insts() const {
    return HasDsSrc2Insts;
  }

  bool hasSDWA() const {
    return HasSDWA;
  }

  bool hasVOP3PInsts() const {
    return HasVOP3PInsts;
  }

  bool hasMulI24() const {
    return HasMulI24;
  }

  bool hasMulU24() const {
    return HasMulU24;
  }

  bool hasSMulHi() const {
    return HasSMulHi;
  }

  bool hasInv2PiInlineImm() const {
    return HasInv2PiInlineImm;
  }

  bool hasFminFmaxLegacy() const {
    return HasFminFmaxLegacy;
  }

  bool hasTrigReducedRange() const {
    return HasTrigReducedRange;
  }

  bool isPromoteAllocaEnabled() const {
    return EnablePromoteAlloca;
  }

  unsigned getWavefrontSize() const {
    return 1 << WavefrontSizeLog2;
  }

  unsigned getWavefrontSizeLog2() const {
    return WavefrontSizeLog2;
  }

  unsigned getLocalMemorySize() const {
    return LocalMemorySize;
  }

  Align getAlignmentForImplicitArgPtr() const {
    return isAmdHsaOS() ? Align(8) : Align(4);
  }

  /// Returns the offset in bytes from the start of the input buffer
  ///        of the first explicit kernel argument.
  unsigned getExplicitKernelArgOffset(const Function &F) const {
    return isAmdHsaOrMesa(F) ? 0 : 36;
  }

  /// \returns Maximum number of work groups per compute unit supported by the
  /// subtarget and limited by given \p FlatWorkGroupSize.
  virtual unsigned getMaxWorkGroupsPerCU(unsigned FlatWorkGroupSize) const = 0;

  /// \returns Minimum flat work group size supported by the subtarget.
  virtual unsigned getMinFlatWorkGroupSize() const = 0;

  /// \returns Maximum flat work group size supported by the subtarget.
  virtual unsigned getMaxFlatWorkGroupSize() const = 0;

  /// \returns Number of waves per execution unit required to support the given
  /// \p FlatWorkGroupSize.
  virtual unsigned
  getWavesPerEUForWorkGroup(unsigned FlatWorkGroupSize) const = 0;

  /// \returns Minimum number of waves per execution unit supported by the
  /// subtarget.
  virtual unsigned getMinWavesPerEU() const = 0;

  /// \returns Maximum number of waves per execution unit supported by the
  /// subtarget without any kind of limitation.
  unsigned getMaxWavesPerEU() const { return MaxWavesPerEU; }

  /// Return the maximum workitem ID value in the function, for the given (0, 1,
  /// 2) dimension.
  unsigned getMaxWorkitemID(const Function &Kernel, unsigned Dimension) const;

  /// Creates value range metadata on an workitemid.* intrinsic call or load.
  bool makeLIDRangeMetadata(Instruction *I) const;

  /// \returns Number of bytes of arguments that are passed to a shader or
  /// kernel in addition to the explicit ones declared for the function.
  unsigned getImplicitArgNumBytes(const Function &F) const;
  uint64_t getExplicitKernArgSize(const Function &F, Align &MaxAlign) const;
  unsigned getKernArgSegmentSize(const Function &F, Align &MaxAlign) const;

  /// \returns Corresponding DWARF register number mapping flavour for the
  /// \p WavefrontSize.
  AMDGPUDwarfFlavour getAMDGPUDwarfFlavour() const;

  virtual ~AMDGPUSubtarget() {}
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSUBTARGET_H
