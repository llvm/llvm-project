//===- CodeLayout.h - Code layout/placement algorithms  ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Declares methods and data structures for code layout algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CODELAYOUT_H
#define LLVM_TRANSFORMS_UTILS_CODELAYOUT_H

#include "llvm/ADT/DenseMap.h"

#include <vector>

namespace llvm {

/// Find a layout of nodes (basic blocks) of a given CFG optimizing jump
/// locality and thus processor I-cache utilization. This is achieved via
/// increasing the number of fall-through jumps and co-locating frequently
/// executed nodes together.
/// The nodes are assumed to be indexed by integers from [0, |V|) so that the
/// current order is the identity permutation.
/// \p NodeSizes: The sizes of the nodes (in bytes).
/// \p NodeCounts: The execution counts of the nodes in the profile.
/// \p EdgeCounts: The execution counts of every edge (jump) in the profile. The
///    map also defines the edges in CFG and should include 0-count edges.
/// \returns The best block order found.
std::vector<uint64_t> applyExtTspLayout(
    const std::vector<uint64_t> &NodeSizes,
    const std::vector<uint64_t> &NodeCounts,
    const DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> &EdgeCounts);

/// Estimate the "quality" of a given node order in CFG. The higher the score,
/// the better the order is. The score is designed to reflect the locality of
/// the given order, which is anti-correlated with the number of I-cache misses
/// in a typical execution of the function.
double calcExtTspScore(
    const std::vector<uint64_t> &Order, const std::vector<uint64_t> &NodeSizes,
    const std::vector<uint64_t> &NodeCounts,
    const DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> &EdgeCounts);

/// Estimate the "quality" of the current node order in CFG.
double calcExtTspScore(
    const std::vector<uint64_t> &NodeSizes,
    const std::vector<uint64_t> &NodeCounts,
    const DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> &EdgeCounts);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CODELAYOUT_H
