//===-- R600MachineFunctionInfo.h - R600 Machine Function Info ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_R600_AMDGPUMACHINEFUNCTION_H
#define LLVM_LIB_TARGET_R600_AMDGPUMACHINEFUNCTION_H

#include "llvm/CodeGen/MachineFunction.h"
#include <map>

namespace llvm {

class AMDGPUMachineFunction : public MachineFunctionInfo {
  virtual void anchor();
  unsigned ShaderType;

public:
  AMDGPUMachineFunction(const MachineFunction &MF);
  /// A map to keep track of local memory objects and their offsets within
  /// the local memory space.
  std::map<const GlobalValue *, unsigned> LocalMemoryObjects;
  /// Number of bytes in the LDS that are being used.
  unsigned LDSSize;

  /// Start of implicit kernel args
  unsigned ABIArgOffset;

  unsigned getShaderType() const {
    return ShaderType;
  }

  unsigned ScratchSize;
  bool IsKernel;
};

}
#endif
