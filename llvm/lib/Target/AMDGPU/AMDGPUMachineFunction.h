//===-- AMDGPUMachineFunctionInfo.h -------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEFUNCTION_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEFUNCTION_H

#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class GCNSubtarget;

class AMDGPUMachineFunction : public MachineFunctionInfo {
  /// A map to keep track of local memory objects and their offsets within the
  /// local memory space.
  SmallDenseMap<const GlobalValue *, unsigned, 4> LocalMemoryObjects;

protected:
  uint64_t ExplicitKernArgSize = 0; // Cache for this.
  Align MaxKernArgAlign;        // Cache for this.

  /// Number of bytes in the LDS that are being used.
  unsigned LDSSize = 0;

  /// Number of bytes in the LDS allocated statically. This field is only used
  /// in the instruction selector and not part of the machine function info.
  unsigned StaticLDSSize = 0;

  /// Align for dynamic shared memory if any. Dynamic shared memory is
  /// allocated directly after the static one, i.e., LDSSize. Need to pad
  /// LDSSize to ensure that dynamic one is aligned accordingly.
  /// The maximal alignment is updated during IR translation or lowering
  /// stages.
  Align DynLDSAlign;

  // State of MODE register, assumed FP mode.
  AMDGPU::SIModeRegisterDefaults Mode;

  // Kernels + shaders. i.e. functions called by the hardware and not called
  // by other functions.
  bool IsEntryFunction = false;

  // Entry points called by other functions instead of directly by the hardware.
  bool IsModuleEntryFunction = false;

  bool NoSignedZerosFPMath = false;

  // Function may be memory bound.
  bool MemoryBound = false;

  // Kernel may need limited waves per EU for better performance.
  bool WaveLimiter = false;

public:
  AMDGPUMachineFunction(const MachineFunction &MF);

  uint64_t getExplicitKernArgSize() const {
    return ExplicitKernArgSize;
  }

  unsigned getMaxKernArgAlign() const { return MaxKernArgAlign.value(); }

  unsigned getLDSSize() const {
    return LDSSize;
  }

  AMDGPU::SIModeRegisterDefaults getMode() const {
    return Mode;
  }

  bool isEntryFunction() const {
    return IsEntryFunction;
  }

  bool isModuleEntryFunction() const { return IsModuleEntryFunction; }

  bool hasNoSignedZerosFPMath() const {
    return NoSignedZerosFPMath;
  }

  bool isMemoryBound() const {
    return MemoryBound;
  }

  bool needsWaveLimiter() const {
    return WaveLimiter;
  }

  unsigned allocateLDSGlobal(const DataLayout &DL, const GlobalVariable &GV);
  void allocateModuleLDSGlobal(const Module *M);

  Align getDynLDSAlign() const { return DynLDSAlign; }

  void setDynLDSAlign(const DataLayout &DL, const GlobalVariable &GV);
};

}
#endif
