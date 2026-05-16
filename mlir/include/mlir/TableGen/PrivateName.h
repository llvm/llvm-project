//===- PrivateName.h - Private name obfuscation for ODS ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public API for ODS-driven obfuscation of dialect namespaces, op mnemonics,
// AttrDef/TypeDef mnemonics, and pass arguments / names. Which dialects are
// private is configured by the build via `--mlir-private-dialects`. Whether
// passes are private is a single global toggle controlled by the build via
// `--mlir-private-passes` (a pass's privacy depends on the tool consuming
// it, just like a dialect's). Op argument attribute keys are intentionally
// not obfuscated. For private passes, the description / per-option /
// per-statistic descriptions are also emitted as empty strings when
// obfuscation is enabled.
//
// All of the configuration / plumbing (the cl::opts, the obfuscator-
// subprocess invocation, and the name cache) lives in
// `mlir/tools/mlir-tblgen/PrivateName.cpp` so that callers of this header
// only see the pure query API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PRIVATENAME_H_
#define MLIR_TABLEGEN_PRIVATENAME_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {
namespace tblgen {

/// Returns true if private-name obfuscation is enabled, i.e., a non-empty
/// obfuscator command has been configured via `--mlir-private-name-obfuscator`.
bool obfuscatePrivateNamesEnabled();

/// Mark the dialect with the given namespace as private. Called by the
/// mlir-tblgen driver once per value parsed from `--mlir-private-dialects`.
void addPrivateDialect(StringRef dialectName);

/// Returns true if the dialect with the given namespace has been listed as
/// private via `--mlir-private-dialects` on the mlir-tblgen command line.
/// Privacy of a dialect is a build-level concern (it depends on which tool
/// the dialect is being compiled into), so it is not tracked in ODS.
bool isDialectPrivate(StringRef dialectName);

/// Returns true if `--mlir-private-passes` was set on the mlir-tblgen
/// command line, i.e., all passes should be treated as private. Privacy of
/// a pass is a build-level concern (it depends on which tool the pass is
/// being compiled into), so it is not tracked in ODS.
bool arePassesPrivate();

/// Returns the obfuscated form of `name`. The returned StringRef is stable
/// for the lifetime of the process. The configured obfuscator is invoked at
/// most once per distinct name.
StringRef obfuscatePrivateName(StringRef name);

/// Returns either `name` (when not private or obfuscation is disabled) or
/// `obfuscatePrivateName(name)`.
inline StringRef maybeObfuscate(StringRef name, bool isPrivate) {
  if (!isPrivate || !obfuscatePrivateNamesEnabled())
    return name;
  return obfuscatePrivateName(name);
}

/// For a dotted name "dialect.mnemonic", obfuscates the dialect prefix and
/// the mnemonic suffix independently and rejoins them with a dot. This keeps
/// runtime parsing of the dialect prefix in `OperationName` working. Names
/// without a '.' are obfuscated as-is.
std::string maybeObfuscateDotted(StringRef name, bool isPrivate);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PRIVATENAME_H_
