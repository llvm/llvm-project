//===--------------------- Support.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Helper functions used by various pipeline components.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_SUPPORT_H
#define LLVM_TOOLS_LLVM_MCA_SUPPORT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSchedule.h"

namespace mca {

/// Populates vector Masks with processor resource masks.
///
/// The number of bits set in a mask depends on the processor resource type.
/// Each processor resource mask has at least one bit set. For groups, the
/// number of bits set in the mask is equal to the cardinality of the group plus
/// one. Excluding the most significant bit, the remaining bits in the mask
/// identify processor resources that are part of the group.
///
/// Example:
///
///  ResourceA  -- Mask: 0b001
///  ResourceB  -- Mask: 0b010
///  ResourceAB -- Mask: 0b100 U (ResourceA::Mask | ResourceB::Mask) == 0b111
///
/// ResourceAB is a processor resource group containing ResourceA and ResourceB.
/// Each resource mask uniquely identifies a resource; both ResourceA and
/// ResourceB only have one bit set.
/// ResourceAB is a group; excluding the most significant bit in the mask, the
/// remaining bits identify the composition of the group.
///
/// Resource masks are used by the ResourceManager to solve set membership
/// problems with simple bit manipulation operations.
void computeProcResourceMasks(const llvm::MCSchedModel &SM,
                              llvm::SmallVectorImpl<uint64_t> &Masks);

/// Compute the reciprocal block throughput from a set of processor resource
/// cycles. The reciprocal block throughput is computed as the MAX between:
///  - NumMicroOps / DispatchWidth
///  - ProcResourceCycles / #ProcResourceUnits  (for every consumed resource).
double computeBlockRThroughput(const llvm::MCSchedModel &SM,
                               unsigned DispatchWidth, unsigned NumMicroOps,
                               llvm::ArrayRef<unsigned> ProcResourceUsage);
} // namespace mca

#endif
