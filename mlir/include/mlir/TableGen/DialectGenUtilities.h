//===- DialectGenUtilities.h - Utilities for dialect generation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MLIRTBLGEN_DIALECTGENUTILITIES_H_
#define MLIR_MLIRTBLGEN_DIALECTGENUTILITIES_H_

#include "mlir/Support/LLVM.h"

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace tblgen {
class Dialect;

/// Find the dialect selected by the user to generate for. Returns std::nullopt
/// if no dialect was found, or if more than one potential dialect was found.
std::optional<Dialect>
findDialectToGenerate(ArrayRef<Dialect> dialects,
                      const llvm::cl::opt<std::string> &selectedDialect);
} // namespace tblgen
} // namespace mlir

#endif // MLIR_MLIRTBLGEN_DIALECTGENUTILITIES_H_
