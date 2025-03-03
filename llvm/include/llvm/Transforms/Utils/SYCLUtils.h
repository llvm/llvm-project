//===------------ SYCLUtils.h - SYCL utility functions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for SYCL.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_SYCLUTILS_H
#define LLVM_TRANSFORMS_UTILS_SYCLUTILS_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>

namespace llvm {

class raw_ostream;

using SYCLStringTable = SmallVector<SmallVector<SmallString<64>>>;

void writeSYCLStringTable(const SYCLStringTable &Table, raw_ostream &OS);

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SYCLUTILS_H
