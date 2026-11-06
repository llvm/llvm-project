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

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/TableGen/Record.h"
#include <string>
#include <vector>

namespace mlir {
namespace tblgen {
/// Returns all the op definitions filtered by the user. The filtering is via
/// command-line options "op-include-regex", "op-exclude-regex" and
/// "op-partition". Verifies "op-partition" and the op partitions of the
/// dialects with ops.
std::vector<const llvm::Record *>
getRequestedOpDefinitions(const llvm::RecordKeeper &records);

/// Returns the name of the private member function of a dialect class that
/// registers the ops of the dialect in op partition `partition`, or all its ops
/// if `partition` is empty.
std::string getOpRegistrationHookName(StringRef partition = "");

/// Checks whether `str` is a Python keyword or would shadow builtin function.
/// Regenerate using python -c"print(set(sorted(__import__('keyword').kwlist)))"
bool isPythonReserved(llvm::StringRef str);

/// Shard the op definitions into the number of shards set by "op-shard-count".
void shardOpDefinitions(
    ArrayRef<const llvm::Record *> defs,
    SmallVectorImpl<ArrayRef<const llvm::Record *>> &shardedDefs);

/// Returns the op partition set by "op-partition", or an empty string if all
/// ops are requested.
StringRef getRequestedOpPartition();

/// Returns whether "op-partition" requests `partition`.
/// Returns true if "op-partition" is not set.
bool isRequestedOpPartition(StringRef partition);

/// Verifies that the op partitions of `dialect` have unique registration hooks.
void verifyDialectOpPartitions(const Dialect &dialect);

/// Returns the dialects that have ops in `records`, in order of first
/// appearance.
std::vector<Dialect> getOpDialects(const llvm::RecordKeeper &records);
} // namespace tblgen
} // namespace mlir

#endif //  MLIR_TOOLS_MLIRTBLGEN_OPGENHELPERS_H_
