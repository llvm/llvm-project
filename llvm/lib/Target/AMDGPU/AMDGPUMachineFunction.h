//===-- AMDGPUMachineFunctionInfo.h -------------------------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEFUNCTION_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEFUNCTION_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class AMDGPUMachineFunction : public MachineFunctionInfo {
  /// A map to keep track of local memory objects and their offsets within the
  /// local memory space.
  SmallDenseMap<const GlobalValue *, unsigned, 4> LocalMemoryObjects;

  uint64_t KernArgSize;
  unsigned MaxKernArgAlign;

  /// Number of bytes in the LDS that are being used.
  unsigned LDSSize;

  // FIXME: This should probably be removed.
  /// Start of implicit kernel args
  unsigned ABIArgOffset;

  // Kernels + shaders. i.e. functions called by the driver and not not called
  // by other functions.
  bool IsEntryFunction;

  bool NoSignedZerosFPMath;

public:
  AMDGPUMachineFunction(const MachineFunction &MF);

  uint64_t allocateKernArg(uint64_t Size, unsigned Align) {
    assert(isPowerOf2_32(Align));
    KernArgSize = alignTo(KernArgSize, Align);

    uint64_t Result = KernArgSize;
    KernArgSize += Size;

    MaxKernArgAlign = std::max(Align, MaxKernArgAlign);
    return Result;
  }

  uint64_t getKernArgSize() const {
    return KernArgSize;
  }

  unsigned getMaxKernArgAlign() const {
    return MaxKernArgAlign;
  }

  void setABIArgOffset(unsigned NewOffset) {
    ABIArgOffset = NewOffset;
  }

  unsigned getABIArgOffset() const {
    return ABIArgOffset;
  }

  unsigned getLDSSize() const {
    return LDSSize;
  }

  bool isEntryFunction() const {
    return IsEntryFunction;
  }

  bool hasNoSignedZerosFPMath() const {
    return NoSignedZerosFPMath;
  }

  unsigned allocateLDSGlobal(const DataLayout &DL, const GlobalValue &GV);
};

}
#endif
