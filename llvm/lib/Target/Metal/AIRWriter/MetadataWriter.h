//===- MetadataWriter.h - AIR metadata-block writer -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_METADATAWRITER_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_METADATAWRITER_H

#include "ValueEnumerator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Metadata.h"
#include <vector>

namespace llvm {
class Module;
class MDNode;
class MDString;
class ValueAsMetadata;
namespace metal {

// Metadata ID assignment: strings first, then values, then nodes.
struct MetadataEnumerator {
  std::vector<StringRef> strings;
  DenseMap<const MDString *, unsigned> stringMap;

  struct ValEntry {
    unsigned typeIdx;
    unsigned valueID;
  };
  std::vector<ValEntry> values;
  DenseMap<const ValueAsMetadata *, unsigned> valueMap;

  std::vector<const MDNode *> nodes;
  DenseMap<const MDNode *, unsigned> nodeMap;

  // Also walks instruction-attached metadata across every function so
  // alias.scope / noalias / tbaa MDNodes get IDs.
  void collect(Module &M, ValueEnumerator &E);
  void visitNode(const MDNode *N, ValueEnumerator &E);
  uint64_t operandID(const Metadata *Op) const;

  // Global metadata ID of an MDNode (matches reader's MetadataList index):
  // strings first, then values, then nodes.
  unsigned globalMDID(const MDNode *N) const {
    return strings.size() + values.size() + nodeMap.lookup(N);
  }
};

} // namespace metal
} // namespace llvm

#endif
