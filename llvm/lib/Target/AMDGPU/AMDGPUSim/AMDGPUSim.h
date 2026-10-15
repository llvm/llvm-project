//===- AMDGPUSim/AMDGPUSim.h - AMDGPU static simulator ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Provides hardware model construction for supported AMDGPU subtargets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_AMDGPUSIM_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_AMDGPUSIM_H

#include "HWModel.h"
#include "Simulator.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace llvm {
namespace AMDGPUSim {

inline std::unique_ptr<HWModel> createHWModel(GPUTarget Target) {
  switch (Target) {
  case GPUTarget::GFX1250:
    return std::make_unique<GFX1250HWModel>();
  }
  llvm_unreachable("unknown GPU target");
}

} // namespace AMDGPUSim
} // namespace llvm

#endif
