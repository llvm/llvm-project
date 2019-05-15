//===-- TargetInfo/AMDGPUTargetInfo.cpp - TargetInfo for AMDGPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/AMDGPUTargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

/// The target which supports all AMD GPUs.  This will eventually
///         be deprecated and there will be a R600 target and a GCN target.
Target &llvm::getTheAMDGPUTarget() {
  static Target TheAMDGPUTarget;
  return TheAMDGPUTarget;
}
/// The target for GCN GPUs
Target &llvm::getTheGCNTarget() {
  static Target TheGCNTarget;
  return TheGCNTarget;
}

/// Extern function to initialize the targets for the AMDGPU backend
extern "C" void LLVMInitializeAMDGPUTargetInfo() {
  RegisterTarget<Triple::r600, false> R600(getTheAMDGPUTarget(), "r600",
                                           "AMD GPUs HD2XXX-HD6XXX", "AMDGPU");
  RegisterTarget<Triple::amdgcn, false> GCN(getTheGCNTarget(), "amdgcn",
                                            "AMD GCN GPUs", "AMDGPU");
}
