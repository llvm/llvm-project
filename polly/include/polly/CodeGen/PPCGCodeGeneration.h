//===--- polly/PPCGCodeGeneration.h - Polly Accelerator Code Generation. --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Take a scop created by ScopInfo and map it to GPU code using the ppcg
// GPU mapping strategy.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_PPCGCODEGENERATION_H
#define POLLY_PPCGCODEGENERATION_H

/// The GPU Architecture to target.
enum GPUArch { NVPTX64, SPIR32, SPIR64 };

/// The GPU Runtime implementation to use.
enum GPURuntime { CUDA, OpenCL };

namespace polly {
extern bool PollyManagedMemory;
}

#endif // POLLY_PPCGCODEGENERATION_H
