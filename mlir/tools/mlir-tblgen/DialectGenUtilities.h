//===- DialectGenUtilities.h - Utilities for dialect generation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_DIALECTGENUTILITIES_H_
#define MLIR_TOOLS_MLIRTBLGEN_DIALECTGENUTILITIES_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace tblgen {
class Dialect;

/// Find the dialect selected by the user to generate for. Returns std::nullopt
/// if no dialect was found, or if more than one potential dialect was found.
std::optional<Dialect> findDialectToGenerate(ArrayRef<Dialect> dialects);

/// Returns the name of the generated function that registers the attributes
/// (if `valueType` is "Attribute") or types (if `valueType` is "Type") of
/// `dialect`. The name contains `registrationName`, or the dialect class name
/// if empty.
std::string getAttrOrTypeRegistrationHookName(const Dialect &dialect,
                                              StringRef valueType,
                                              StringRef registrationName = {});
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_DIALECTGENUTILITIES_H_
