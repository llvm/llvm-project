//===- PartitionHelpers.h - Helpers for generating by partition -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers for the generators that generate the definitions
// of a dialect one partition (see `Partition` in DialectBase.td) at a time,
// such as its ops, attributes or types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_PARTITIONHELPERS_H_
#define MLIR_TOOLS_MLIRTBLGEN_PARTITIONHELPERS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <string>

namespace mlir {
namespace tblgen {
/// Returns the name of the partition of `def`, or an empty string if it is in
/// no partition. `def` must have a field `partition` of type `Partition` (see
/// DialectBase.td), which may be unset, like `Op` and `AttrOrTypeDef`.
StringRef getPartitionName(const llvm::Record &def);

/// Returns the suffix of the names of the generated functions of `partition`,
/// such as its registration hook.
std::string getPartitionFunctionSuffix(StringRef partition);

/// Verifies the partitions of `defs`: every def must be in a partition,
/// `selectedPartition` must have defs and different partitions must not map to
/// the same suffix (see `getPartitionFunctionSuffix`). `kind` (e.g. "op") and
/// `option` (e.g. "-op-partition") are used in diagnostics.
void verifyPartitions(ArrayRef<const llvm::Record *> defs,
                      StringRef selectedPartition, StringRef kind,
                      StringRef option);

/// Removes the defs of partitions other than `selectedPartition` from `defs`,
/// assigning them to `otherPartitionDefs` if non-null. `getRecord(def)` returns
/// the record of a def.
template <typename ContainerT, typename GetRecordFn = llvm::identity>
void selectPartition(
    StringRef selectedPartition, ContainerT &defs,
    llvm::type_identity_t<ContainerT> *otherPartitionDefs = nullptr,
    GetRecordFn getRecord = {}) {
  auto otherPartitionsBegin =
      std::stable_partition(defs.begin(), defs.end(), [&](const auto &def) {
        return getPartitionName(*getRecord(def)) == selectedPartition;
      });
  if (otherPartitionDefs)
    otherPartitionDefs->assign(otherPartitionsBegin, defs.end());
  defs.erase(otherPartitionsBegin, defs.end());
}

/// Returns the names of the partitions of `defs`, sorted by name.
SmallVector<StringRef> getPartitionNames(ArrayRef<const llvm::Record *> defs);
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_PARTITIONHELPERS_H_
