//===- IndexRemap.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CVPACK_INDEXREMAP_H
#define LLVM_DEBUGINFO_CVPACK_INDEXREMAP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <optional>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

/// Maps the pack indices that an object uses to its local indices, as the
/// sorted ranges of its range map.
class IndexRemap {
public:
  LLVM_ABI void add(uint64_t From, uint64_t Count, uint64_t To);
  /// Sort the ranges. Fails if a pack index is mapped more than once.
  LLVM_ABI Error finalize();
  LLVM_ABI std::optional<uint32_t> lookup(uint32_t Index) const;

private:
  struct Range {
    uint32_t From, Count, To;
  };
  std::vector<Range> Ranges;
};

/// Rewrite in place every non-simple type index of a type record, or of a
/// symbol record if IsSymbol, through Map. Fails for an index that Map does
/// not hold.
LLVM_ABI Error remapTypeIndices(MutableArrayRef<uint8_t> Record, bool IsSymbol,
                                const IndexRemap &Map);

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CVPACK_INDEXREMAP_H
