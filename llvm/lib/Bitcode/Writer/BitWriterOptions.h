//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_BITCODE_WRITER_BITWRITEROPTIONS_H
#define LLVM_LIB_BITCODE_WRITER_BITWRITEROPTIONS_H

#include "llvm/IR/LLVMContext.h"

#define OPTIONS_STRUCT_DECL
#include "BitWriterOptions.inc"

namespace llvm {
inline const BitWriterOptions &getBitWriterOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<BitWriterOptions>();
}
} // namespace llvm

#endif // LLVM_LIB_BITCODE_WRITER_BITWRITEROPTIONS_H
