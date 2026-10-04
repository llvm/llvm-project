//===- AMDGPUEmitPrintf.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility function to lower a printf call into a series of device
// library calls on the AMDGPU target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_AMDGPUEMITPRINTF_H
#define LLVM_TRANSFORMS_UTILS_AMDGPUEMITPRINTF_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm {

LLVM_ABI Value *emitAMDGPUPrintfCall(IRBuilder<> &Builder,
                                     ArrayRef<Value *> Args, bool isBuffered);

/// Marks the printf arguments that the format string \p Fmt specifies as
/// strings, i.e. with a "%s" specifier. Argument 0 is the format string itself,
/// and each '*' in a specifier consumes an argument of its own.
LLVM_ABI void locateAMDGPUPrintfCStrings(SparseBitVector<8> &BV, StringRef Fmt);

/// Returns the ID that buffered printf stores in place of the constant format
/// string \p Fmt: the low 64 bits of its MD5 hash.
LLVM_ABI uint64_t getAMDGPUPrintfFormatHash(StringRef Fmt);

/// Returns the llvm.printf.fmts entry that maps the ID of the constant format
/// string \p Fmt back to the string.
LLVM_ABI std::string getAMDGPUPrintfFormatMetadata(StringRef Fmt);

/// The llvm.printf.fmts entry added when a module only uses buffered printf
/// with non-constant format strings.
inline constexpr StringLiteral AMDGPUPrintfNonConstFormatMetadata =
    "0:0:ffffffff,\"Non const format string\"";

/// Splits the constant string argument \p Str, including its terminating NUL,
/// into the little-endian 32-bit words that buffered printf stores for it,
/// padded to a multiple of 8 bytes.
LLVM_ABI void packAMDGPUPrintfConstantString(StringRef Str,
                                             SmallVectorImpl<uint32_t> &Words);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_AMDGPUEMITPRINTF_H
