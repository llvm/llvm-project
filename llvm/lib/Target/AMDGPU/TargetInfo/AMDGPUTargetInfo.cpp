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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

/// The target for R600 GPUs.
Target &llvm::getTheR600Target() {
  static Target TheAMDGPUTarget;
  return TheAMDGPUTarget;
}

/// The target for GCN GPUs.
Target &llvm::getTheGCNTarget() {
  static Target TheGCNTarget;
  return TheGCNTarget;
}

/// The target for GCN GPUs, registered under the legacy "amdgcn" name. This
/// should be removed when all tool users of "amdgcn" are migrated.
Target &llvm::getTheGCNLegacyTarget() {
  static Target TheGCNLegacyTarget;
  return TheGCNLegacyTarget;
}

/// Extern function to initialize the targets for the AMDGPU backend
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeAMDGPUTargetInfo() {
  RegisterTarget<Triple::r600, false> R600(getTheR600Target(), "r600",
                                           "AMD GPUs HD2XXX-HD6XXX", "AMDGPU");
  Target &GCNTgt = getTheGCNTarget();
  RegisterTarget<Triple::amdgpu, false> GCN(GCNTgt, "amdgpu", "AMDGPU gfx6+",
                                            "AMDGPU");

  // Register the legacy "amdgcn" name for use with -march. It must not take
  // possession of the Triple::amdgpu tag, so it uses a match function that
  // never matches a triple. This hack is copied from AArch64's handling of
  // "arm64".
  TargetRegistry::RegisterTarget(getTheGCNLegacyTarget(), "amdgcn",
                                 "legacy name for amdgpu", "AMDGPU",
                                 [](Triple::ArchType) { return false; });
}
