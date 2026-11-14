//===- MetadataUtils.h - Metadata utilities ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_METADATAUTILS_H
#define LLVM_ANALYSIS_METADATAUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace llvm {
class Instruction;
class MDNode;
class Value;

/// Copy metadata valid for a replacement memory access. The destination must
/// access the same bytes, a subset, or an expanded range in the same object.
/// The destination must be a typed memory access; memory-transfer and
/// memory-set intrinsics are supported only as sources. DestByteOffset is the
/// destination's offset within the source access. A missing offset requires the
/// caller to establish that the new access is a subset. Access types and
/// pointer operands are inferred for loads, stores, and masked/VP memory
/// intrinsics. For a source memory intrinsic, the source or destination pointer
/// is selected to match the new read or write. The caller must establish range
/// containment when the intrinsic's length is unknown. Pointer and value facts
/// are retained only when valid for the destination; alias tags are adjusted
/// for its range. Masks and execution conditions are not inspected.
LLVM_ABI void
copyMetadataForMemoryAccess(Instruction &Dest, const Instruction &Source,
                            std::optional<int64_t> DestByteOffset = 0);

/// Copy metadata valid after splitting an instruction into smaller elements.
/// DestByteOffset describes the selected memory fragment, if applicable.
LLVM_ABI void copyMetadataForScalarizedInstruction(
    Instruction &Dest, const Instruction &Source,
    std::optional<int64_t> DestByteOffset = std::nullopt);

/// Combine the metadata of two instructions so that K can replace J. This
/// specifically handles the case of CSE-like transformations. Some
/// metadata can only be kept if K dominates J. For this to be correct,
/// K cannot be hoisted.
///
/// Unknown metadata is removed.
LLVM_ABI void combineMetadataForCSE(Instruction *K, const Instruction *J,
                                    bool DoesKMove);

/// Combine metadata of two instructions, where instruction J is a memory
/// access that has been merged into K. This will intersect alias-analysis
/// metadata, while preserving other known metadata.
LLVM_ABI void combineAAMetadata(Instruction *K, const Instruction *J);

/// Compute the union of two access-group lists.
///
/// If the list contains just one access group, it is returned directly. If the
/// list is empty, returns nullptr.
LLVM_ABI MDNode *uniteAccessGroups(MDNode *AccGroups1, MDNode *AccGroups2);

/// Compute the access-group list of access groups that @p Inst1 and @p Inst2
/// are both in. If either instruction does not access memory at all, it is
/// considered to be in every list.
///
/// If the list contains just one access group, it is returned directly. If the
/// list is empty, returns nullptr.
LLVM_ABI MDNode *intersectAccessGroups(const Instruction *Inst1,
                                       const Instruction *Inst2);

/// Collect metadata that can be preserved when vectorizing an instruction.
LLVM_ABI void getMetadataToPropagate(
    Instruction *Inst,
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Metadata);

/// Propagate metadata that can be preserved after vectorization from \p VL to
/// \p I, retaining only information valid for all sources and the destination.
/// Return \p I.
LLVM_ABI Instruction *propagateMetadata(Instruction *I, ArrayRef<Value *> VL);

} // namespace llvm

#endif // LLVM_ANALYSIS_METADATAUTILS_H
