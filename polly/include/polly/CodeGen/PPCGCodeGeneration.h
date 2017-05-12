//===--- polly/PPCGCodeGeneration.h - Polly Accelerator Code Generation. --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
enum GPUArch { NVPTX64 };

/// The GPU Runtime implementation to use.
enum GPURuntime { CUDA, OpenCL };

#endif // POLLY_PPCGCODEGENERATION_H
