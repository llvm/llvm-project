//===-- X86TargetParser - Parser for X86 features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise X86 hardware features.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_X86TARGETPARSERCOMMON_H
#define LLVM_SUPPORT_X86TARGETPARSERCOMMON_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {
class StringRef;

namespace X86 {

enum CPUKind {
  CK_None,
#define PROC(ENUM, STRING, IS64BIT) CK_##ENUM,
#include "llvm/Support/X86TargetParser.def"
};

/// Parse \p CPU string into a CPUKind. Will only accept 64-bit capable CPUs if
/// \p Only64Bit is true.
CPUKind parseArchX86(StringRef CPU, bool Only64Bit = false);

/// Provide a list of valid CPU names. If \p Only64Bit is true, the list will
/// only contain 64-bit capable CPUs.
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values,
                          bool ArchIs32Bit);

} // namespace X86
} // namespace llvm

#endif
