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
class DataLayout;
class LoadInst;
class Instruction;
class MDNode;
class Value;

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

/// Copy the metadata from the source instruction to the destination (the
/// replacement for the source instruction).
LLVM_ABI void copyMetadataForLoad(LoadInst &Dest, const LoadInst &Source);

/// Copy a nonnull metadata node to a new load instruction.
///
/// This handles mapping it to range metadata if the new load is an integer
/// load instead of a pointer load.
LLVM_ABI void copyNonnullMetadata(const LoadInst &OldLI, MDNode *N,
                                  LoadInst &NewLI);

/// Copy a range metadata node to a new load instruction.
///
/// This handles mapping it to nonnull metadata if the new load is a pointer
/// load instead of an integer load and the range doesn't cover null.
LLVM_ABI void copyRangeMetadata(const DataLayout &DL, const LoadInst &OldLI,
                                MDNode *N, LoadInst &NewLI);

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

/// Add metadata from \p Inst to \p Metadata, if it can be preserved after
/// vectorization. It can be preserved after vectorization if the kind is one of
/// [MD_tbaa, MD_alias_scope, MD_noalias, MD_fpmath, MD_nontemporal,
/// MD_access_group, MD_mmra].
LLVM_ABI void getMetadataToPropagate(
    Instruction *Inst,
    SmallVectorImpl<std::pair<unsigned, MDNode *>> &Metadata);

/// Propagate metadata that can be preserved after vectorization from \p VL to
/// \p I, retaining only information valid for all sources and the destination.
/// Return \p I.
LLVM_ABI Instruction *propagateMetadata(Instruction *I, ArrayRef<Value *> VL);

} // namespace llvm

#endif // LLVM_ANALYSIS_METADATAUTILS_H
