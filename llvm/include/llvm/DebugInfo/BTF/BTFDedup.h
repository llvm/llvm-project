//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// BTF type deduplication algorithm.
///
/// Implements the 5-pass deduplication algorithm from libbpf:
///   1. String deduplication
///   2. Primitive/struct/enum type dedup with DFS equivalence checking
///   3. Reference type dedup (PTR, TYPEDEF, etc.)
///   4. Type compaction (remove duplicates, assign new IDs)
///   5. Type ID remapping
///
/// The algorithm achieves ~137x type reduction on a full Linux kernel
/// (3.6M types -> 26K types).
///
/// Reference: https://nakryiko.com/posts/btf-dedup/
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_BTF_BTFDEDUP_H
#define LLVM_DEBUGINFO_BTF_BTFDEDUP_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

namespace llvm {

class BTFBuilder;

namespace BTF {

/// Deduplicate types in a BTFBuilder in-place.
///
/// After deduplication, structurally equivalent types are merged and
/// all type IDs are remapped to point to canonical representatives.
/// The resulting BTFBuilder can be serialized to produce a compact
/// .BTF section.
LLVM_ABI Error dedup(BTFBuilder &Builder);

} // namespace BTF
} // namespace llvm

#endif // LLVM_DEBUGINFO_BTF_BTFDEDUP_H
