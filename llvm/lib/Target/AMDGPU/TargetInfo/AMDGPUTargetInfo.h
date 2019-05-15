//===-- TargetInfo/AMDGPUTargetInfo.h - TargetInfo for AMDGPU ---*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_AMDGPU_TARGETINFO_AMDGPUTARGETINFO_H
#define LLVM_LIB_TARGET_AMDGPU_TARGETINFO_AMDGPUTARGETINFO_H

namespace llvm {

class Target;

/// The target which supports all AMD GPUs.  This will eventually
///         be deprecated and there will be a R600 target and a GCN target.
Target &getTheAMDGPUTarget();

/// The target for GCN GPUs
Target &getTheGCNTarget();

}

#endif // LLVM_LIB_TARGET_AMDGPU_TARGETINFO_AMDGPUTARGETINFO_H
