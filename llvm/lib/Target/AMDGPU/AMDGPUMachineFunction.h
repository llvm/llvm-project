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
#include <map>

namespace llvm {

class AMDGPUMachineFunction : public MachineFunctionInfo {
  uint64_t KernArgSize;
  unsigned MaxKernArgAlign;

  virtual void anchor();

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

  /// A map to keep track of local memory objects and their offsets within
  /// the local memory space.
  std::map<const GlobalValue *, unsigned> LocalMemoryObjects;
  /// Number of bytes in the LDS that are being used.
  unsigned LDSSize;

  /// Start of implicit kernel args
  unsigned ABIArgOffset;

  bool isKernel() const {
    // FIXME: Assume everything is a kernel until function calls are supported.
    return true;
  }

  unsigned ScratchSize;
  bool IsKernel;
};

}
#endif
