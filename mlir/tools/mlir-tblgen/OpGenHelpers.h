//===- OpGenHelpers.h - MLIR operation generator helpers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers used in the op generators.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_OPGENHELPERS_H_
#define MLIR_TOOLS_MLIRTBLGEN_OPGENHELPERS_H_

#include "llvm/TableGen/Record.h"
#include <vector>

namespace mlir {
namespace tblgen {

/// Returns all the op definitions filtered by the user. The filtering is via
/// command-line option "op-include-regex" and "op-exclude-regex".
std::vector<llvm::Record *>
getRequestedOpDefinitions(const llvm::RecordKeeper &recordKeeper);

/// Checks whether `str` is a Python keyword or would shadow builtin function.
/// Regenerate using python -c"print(set(sorted(__import__('keyword').kwlist)))"
bool isPythonReserved(llvm::StringRef str);

} // namespace tblgen
} // namespace mlir

#endif //  MLIR_TOOLS_MLIRTBLGEN_OPGENHELPERS_H_
