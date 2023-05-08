//===-- X86EncodingOptimization.h - X86 Encoding optimization ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the X86 encoding optimization
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86ENCODINGOPTIMIZATION_H
#define LLVM_LIB_TARGET_X86_X86ENCODINGOPTIMIZATION_H
namespace llvm {
class MCInst;
namespace X86 {
bool optimizeInstFromVEX3ToVEX2(MCInst &MI);
bool optimizeShiftRotateWithImmediateOne(MCInst &MI);
} // namespace X86
} // namespace llvm
#endif
