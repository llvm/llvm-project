//===- OutlinedHashTree.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This defines the OutlinedHashTree class. It contains sequences of stable
// hash values of instructions that have been outlined. This OutlinedHashTree
// can be used to track the outlined instruction sequences across modules.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CODEGENDATA_OUTLINEDHASHTREE_H
#define LLVM_CODEGENDATA_OUTLINEDHASHTREE_H

#include "llvm/ADT/StableHashing.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>
#include <vector>

namespace llvm {

/// A HashNode is an entry in an OutlinedHashTree, holding a hash value
/// and a collection of Successors (other HashNodes). If a HashNode has
/// a positive terminal value (Terminals > 0), it signifies the end of
/// a hash sequence with that occurrence count.
struct HashNode {
  /// The hash value of the node.
  stable_hash Hash;
  /// The number of terminals in the sequence ending at this node.
  unsigned Terminals;
  /// The successors of this node.
  std::unordered_map<stable_hash, std::unique_ptr<HashNode>> Successors;
};

/// HashNodeStable is the serialized, stable, and compact representation
/// of a HashNode.
struct HashNodeStable {
  llvm::yaml::Hex64 Hash;
  unsigned Terminals;
  std::vector<unsigned> SuccessorIds;
};

class OutlinedHashTree {

  using EdgeCallbackFn =
      std::function<void(const HashNode *, const HashNode *)>;
  using NodeCallbackFn = std::function<void(const HashNode *)>;

  using HashSequence = std::vector<stable_hash>;
  using HashSequencePair = std::pair<std::vector<stable_hash>, unsigned>;

public:
  /// Walks every edge and node in the OutlinedHashTree and calls CallbackEdge
  /// for the edges and CallbackNode for the nodes with the stable_hash for
  /// the source and the stable_hash of the sink for an edge. These generic
  /// callbacks can be used to traverse a OutlinedHashTree for the purpose of
  /// print debugging or serializing it.
  void walkGraph(NodeCallbackFn CallbackNode,
                 EdgeCallbackFn CallbackEdge = nullptr,
                 bool SortedWalk = false) const;

  /// Release all hash nodes except the root hash node.
  void clear() {
    assert(getRoot()->Hash == 0 && getRoot()->Terminals == 0);
    getRoot()->Successors.clear();
  }

  /// \returns true if the hash tree has only the root node.
  bool empty() { return size() == 1; }

  /// \returns the size of a OutlinedHashTree by traversing it. If
  /// \p GetTerminalCountOnly is true, it only counts the terminal nodes
  /// (meaning it returns the the number of hash sequences in the
  /// OutlinedHashTree).
  size_t size(bool GetTerminalCountOnly = false) const;

  /// \returns the depth of a OutlinedHashTree by traversing it.
  size_t depth() const;

  /// \returns the root hash node of a OutlinedHashTree.
  const HashNode *getRoot() const { return Root.get(); }
  HashNode *getRoot() { return Root.get(); }

  /// Inserts a \p Sequence into the this tree. The last node in the sequence
  /// will increase Terminals.
  void insert(const HashSequencePair &SequencePair);

  /// Merge a \p OtherTree into this Tree.
  void merge(const OutlinedHashTree *OtherTree);

  /// \returns the matching count if \p Sequence exists in the OutlinedHashTree.
  unsigned find(const HashSequence &Sequence) const;

  OutlinedHashTree() { Root = std::make_unique<HashNode>(); }

private:
  std::unique_ptr<HashNode> Root;
};

} // namespace llvm

#endif
