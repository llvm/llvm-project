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

#ifndef LLVM_CGDATA_OUTLINEDHASHTREE_H
#define LLVM_CGDATA_OUTLINEDHASHTREE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace llvm {

/// A HashNode is an entry in an OutlinedHashTree, holding a hash value
/// and a collection of Successors (other HashNodes). If a HashNode has
/// a positive terminal value (Terminals > 0), it signifies the end of
/// a hash sequence with that occurrence count.
struct HashNode {
  /// The hash value of the node.
  stable_hash Hash = 0;
  /// The number of terminals in the sequence ending at this node.
  std::optional<unsigned> Terminals;
  /// The successors of this node.
  DenseMap<stable_hash, std::unique_ptr<HashNode>> Successors;
};

/// A read-only outlined hash tree: an in-memory HashNode trie, or a tree read
/// in place from a mapped blob. Either way it supports only the consume path.
/// Building, mutating, or traversing the whole tree is done through
/// MutableOutlinedHashTree, which takes over a tree's storage.
class OutlinedHashTree {
protected:
  using HashSequence = SmallVector<stable_hash>;
  using HashSequencePair = std::pair<HashSequence, unsigned>;

  // Serializes the tree, and constructs it when (de)serializing / converting a
  // read-in-place tree to a mutable one.
  friend struct OutlinedHashTreeRecord;

public:
  /// A lightweight, copyable, read-only cursor over a single node in the tree.
  /// It works for both the eager in-memory representation and the read-in-place
  /// representation, so a consumer can walk the tree without knowing which is
  /// in use.
  class HashNodeCursor {
  public:
    /// \returns a cursor over the successor reached by following \p H from this
    /// node in \p Tree, or std::nullopt if there is none.
    LLVM_ABI std::optional<HashNodeCursor>
    getSuccessor(const OutlinedHashTree &Tree, stable_hash H) const;

    /// \returns the terminal count at this node in \p Tree, or std::nullopt if
    /// the node is not the end of any sequence.
    LLVM_ABI std::optional<unsigned>
    getTerminals(const OutlinedHashTree &Tree) const;

  private:
    friend class OutlinedHashTree;
    HashNodeCursor(const HashNode *N) : Node(N) {}
    HashNodeCursor(uint64_t BlockOffset) : BlockOffset(BlockOffset) {}

    // A single node handle: an in-memory HashNode pointer or a byte offset
    // into the buffer region; which is live is selected by the tree's
    // representation.
    union {
      const HashNode *Node;
      uint64_t BlockOffset;
    };
  };

  /// Construct a tree read in place from a mapped blob: its block region begins
  /// at \p BlockBase (the root block at its offset 0) inside \p Buffer, which
  /// is kept alive for the tree's lifetime.
  OutlinedHashTree(std::shared_ptr<MemoryBuffer> Buffer,
                   const unsigned char *BlockBase, uint32_t NumNodes)
      : Buffer(std::move(Buffer)), BlockBase(BlockBase), NumNodes(NumNodes) {}

  virtual ~OutlinedHashTree() = default;

  /// \returns a read cursor positioned at the root, valid for both the
  /// in-memory and read-in-place representations. This is the entry point for
  /// the in-place consume path and never materializes the tree.
  LLVM_ABI HashNodeCursor getRootCursor() const;

  /// \returns the matching count if \p Sequence exists in the OutlinedHashTree.
  LLVM_ABI std::optional<unsigned> find(const HashSequence &Sequence) const;

  /// \returns true if the hash tree has only the root node.
  bool empty() const {
    return isReadInPlace() ? NumNodes <= 1 : Root.Successors.empty();
  }

  /// \returns true if this tree is read in place from a mapped blob instead of
  /// being materialized into HashNode objects.
  bool isReadInPlace() const { return (bool)Buffer; }

protected:
  /// An empty in-memory tree, to be populated by MutableOutlinedHashTree.
  OutlinedHashTree() = default;

  HashNode Root;

private:
  std::shared_ptr<MemoryBuffer> Buffer;
  const unsigned char *BlockBase = nullptr;
  uint32_t NumNodes = 0;
};

/// A mutable, always-in-memory OutlinedHashTree, adding building, mutation, and
/// whole-tree traversal on top of the read-only base. Turn a read-only tree
/// mutable by handing its storage to the move constructor: an in-memory tree's
/// trie is moved; a read-in-place tree is materialized from its blob.
class MutableOutlinedHashTree : public OutlinedHashTree {
private:
  using EdgeCallbackFn =
      std::function<void(const HashNode *, const HashNode *)>;
  using NodeCallbackFn = std::function<void(const HashNode *)>;

public:
  MutableOutlinedHashTree() = default;

  /// \returns the root hash node of the OutlinedHashTree.
  const HashNode *getRoot() const { return &Root; }
  HashNode *getRoot() { return &Root; }

  /// Inserts a \p Sequence into this tree. The last node in the sequence
  /// increments its Terminals.
  LLVM_ABI void insert(const HashSequencePair &SequencePair);

  /// Merge \p OtherTree into this tree.
  LLVM_ABI void merge(const MutableOutlinedHashTree *OtherTree);

  /// Release all hash nodes except the root hash node.
  void clear() {
    assert(Root.Hash == 0 && !Root.Terminals);
    Root.Successors.clear();
  }

  /// Walks every edge and node in the OutlinedHashTree and calls CallbackEdge
  /// for the edges and CallbackNode for the nodes with the stable_hash for
  /// the source and the stable_hash of the sink for an edge. These generic
  /// callbacks can be used to traverse a OutlinedHashTree for the purpose of
  /// print debugging or serializing it.
  LLVM_ABI void walkGraph(NodeCallbackFn CallbackNode,
                          EdgeCallbackFn CallbackEdge = nullptr,
                          bool SortedWalk = false) const;

  /// \returns the size of a OutlinedHashTree by traversing it. If
  /// \p GetTerminalCountOnly is true, it only counts the terminal nodes
  /// (meaning it returns the the number of hash sequences in the
  /// OutlinedHashTree).
  LLVM_ABI size_t size(bool GetTerminalCountOnly = false) const;

  /// \returns the depth of a OutlinedHashTree by traversing it.
  LLVM_ABI size_t depth() const;
};

} // namespace llvm

#endif
