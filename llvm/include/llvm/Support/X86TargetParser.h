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

// This should be kept in sync with libcc/compiler-rt as its included by clang
// as a proxy for what's in libgcc/compiler-rt.
enum ProcessorVendors : unsigned {
  VENDOR_DUMMY,
#define X86_VENDOR(ENUM, STRING) \
  ENUM,
#include "llvm/Support/X86TargetParser.def"
  VENDOR_OTHER
};

// This should be kept in sync with libcc/compiler-rt as its included by clang
// as a proxy for what's in libgcc/compiler-rt.
enum ProcessorTypes : unsigned {
  CPU_TYPE_DUMMY,
#define X86_CPU_TYPE(ARCHNAME, ENUM) \
  ENUM,
#include "llvm/Support/X86TargetParser.def"
  CPU_TYPE_MAX
};

// This should be kept in sync with libcc/compiler-rt as its included by clang
// as a proxy for what's in libgcc/compiler-rt.
enum ProcessorSubtypes : unsigned {
  CPU_SUBTYPE_DUMMY,
#define X86_CPU_SUBTYPE(ARCHNAME, ENUM) \
  ENUM,
#include "llvm/Support/X86TargetParser.def"
  CPU_SUBTYPE_MAX
};

// This should be kept in sync with libcc/compiler-rt as it should be used
// by clang as a proxy for what's in libgcc/compiler-rt.
enum ProcessorFeatures {
#define X86_FEATURE(VAL, ENUM) \
  ENUM = VAL,
#include "llvm/Support/X86TargetParser.def"
  CPU_FEATURE_MAX
};

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
