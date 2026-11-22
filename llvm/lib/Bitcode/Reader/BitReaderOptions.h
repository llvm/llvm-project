//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_BITCODE_READER_BITREADEROPTIONS_H
#define LLVM_LIB_BITCODE_READER_BITREADEROPTIONS_H

#include "llvm/IR/LLVMContext.h"

#define OPTIONS_STRUCT_DECL
#include "BitReaderOptions.inc"

namespace llvm {
inline const BitReaderOptions &getBitReaderOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<BitReaderOptions>();
}
} // namespace llvm

#endif // LLVM_LIB_BITCODE_READER_BITREADEROPTIONS_H
