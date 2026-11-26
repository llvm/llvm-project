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

/// The kinds of definitions that are registered with their dialect by
/// generated registration hooks.
enum class RegistrationKind { Operation, Attribute, Type };

/// Returns the name of the generated function that registers the `kind`s of
/// `dialect` in partition or shard `part`, or all of them if `part` is empty.
/// The name contains `registrationName`, or the dialect class name if empty.
std::string getRegistrationHookName(const Dialect &dialect,
                                    RegistrationKind kind,
                                    StringRef registrationName = {},
                                    StringRef part = {});

/// Emit the declarations of the hook that registers the `kind`s of `dialect`
/// (see `getRegistrationHookName`) and of the hooks of `parts`, which it calls
/// if the registration is split into several parts (e.g. partitions).
/// `partsDescription` names the parts in the documentation of the hook. The
/// dialect class is forward declared, so that the declarations do not require
/// the dialect's declaration. Must be emitted in the namespace of `dialect`.
void emitRegistrationHookDecls(raw_ostream &os, const Dialect &dialect,
                               RegistrationKind kind,
                               StringRef registrationName,
                               ArrayRef<std::string> parts,
                               StringRef partsDescription);

/// Emit the definition of the hook that registers `classNamesToRegister`, the
/// `kind`s of part `part`, with `dialect`. If `registrationNameSuffix` is not
/// empty, it replaces the dialect class name in the names of the hooks (see
/// `getRegistrationHookName`). Without `parts`, `part` must be empty and this
/// is the hook of the dialect. Otherwise, the first of `parts` also defines the
/// hook of the dialect, which calls the hooks of all parts. Must be emitted in
/// the namespace of `dialect`.
void emitRegistrationHookDefs(raw_ostream &os, const Dialect &dialect,
                              RegistrationKind kind,
                              StringRef registrationNameSuffix, StringRef part,
                              ArrayRef<std::string> parts,
                              ArrayRef<std::string> classNamesToRegister);
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_DIALECTGENUTILITIES_H_
