//===- OutlinedHashTreeRecord.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This defines the OutlinedHashTreeRecord class. This class holds the outlined
// hash tree for both serialization and deserialization processes. It utilizes
// two data formats for serialization: raw binary data and YAML.
// These two formats can be used interchangeably.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CGDATA_OUTLINEDHASHTREERECORD_H
#define LLVM_CGDATA_OUTLINEDHASHTREERECORD_H

#include "llvm/CGData/OutlinedHashTree.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// HashNodeStable is the serialized, stable, and compact representation
/// of a HashNode.
struct HashNodeStable {
  llvm::yaml::Hex64 Hash;
  unsigned Terminals;
  std::vector<unsigned> SuccessorIds;
};

using IdHashNodeStableMapTy = std::map<unsigned, HashNodeStable>;
using IdHashNodeMapTy = DenseMap<unsigned, HashNode *>;
using HashNodeIdMapTy = DenseMap<const HashNode *, unsigned>;

struct OutlinedHashTreeRecord {
  std::unique_ptr<MutableOutlinedHashTree> HashTree;

  OutlinedHashTreeRecord()
      : HashTree(std::make_unique<MutableOutlinedHashTree>()) {}
  /// Take ownership of the already-in-memory \p Tree.
  OutlinedHashTreeRecord(std::unique_ptr<MutableOutlinedHashTree> Tree)
      : HashTree(std::move(Tree)) {}

  /// Serialize the outlined hash tree to a raw_ostream.
  LLVM_ABI void serialize(raw_ostream &OS) const;
  /// Deserialize the outlined hash tree from a raw_ostream.
  LLVM_ABI void deserialize(const unsigned char *&Ptr);
  /// Lazily set up this record's tree to read in place from \p Buffer at
  /// \p Offset instead of deserializing it in memory.
  LLVM_ABI void lazyDeserialize(std::shared_ptr<MemoryBuffer> Buffer,
                                uint64_t Offset);

  /// \returns a read-only tree that reads its nodes in place from \p Buffer
  /// starting at \p BlobOffset, keeping \p Buffer alive for the tree's
  /// lifetime.
  LLVM_ABI static std::unique_ptr<OutlinedHashTree>
  createReadInPlace(std::shared_ptr<MemoryBuffer> Buffer, uint64_t BlobOffset);

  /// \returns a mutable, fully in-memory version of \p Tree: a read-in-place
  /// tree is materialized from its blob; a tree already in memory is taken
  /// over as-is.
  LLVM_ABI static std::unique_ptr<MutableOutlinedHashTree>
  createInMemory(std::unique_ptr<OutlinedHashTree> Tree);

  /// \returns the block offset of the successor reached by following \p H, or
  /// std::nullopt if there is none. In-place readers over a single serialized
  /// node block at \p Block (the block region base + the node's block offset).
  LLVM_ABI static std::optional<uint64_t>
  readSuccessor(const unsigned char *Block, stable_hash H);

  /// \returns the terminal count stored at \p Block (0 if not a terminal).
  LLVM_ABI static unsigned readTerminals(const unsigned char *Block);

  /// Serialize the outlined hash tree to a YAML stream.
  LLVM_ABI void serializeYAML(yaml::Output &YOS) const;
  /// Deserialize the outlined hash tree from a YAML stream.
  LLVM_ABI void deserializeYAML(yaml::Input &YIS);

  /// Merge the other outlined hash tree into this one.
  void merge(const OutlinedHashTreeRecord &Other) {
    HashTree->merge(Other.HashTree.get());
  }

  /// Release the held outlined hash tree as a read-only tree.
  std::unique_ptr<OutlinedHashTree> releaseTree() {
    if (ReadInPlaceHashTree)
      return std::move(ReadInPlaceHashTree);
    return std::move(HashTree);
  }

  /// \returns true if the outlined hash tree is empty.
  bool empty() const { return tree().empty(); }

  /// Print the outlined hash tree in a YAML format.
  void print(raw_ostream &OS = llvm::errs()) const {
    yaml::Output YOS(OS);
    serializeYAML(YOS);
  }

private:
  /// \returns a mutable, in-memory tree built from the block region beginning
  /// at \p BlockBase (the root block at its offset 0).
  static std::unique_ptr<MutableOutlinedHashTree>
  createInMemory(const unsigned char *BlockBase);

  /// \returns the held tree, whichever representation is in use.
  const OutlinedHashTree &tree() const {
    return ReadInPlaceHashTree ? *ReadInPlaceHashTree : *HashTree;
  }

  /// Controls how a node's successors are ordered when converting to stable
  /// data.
  enum class SuccessorOrder {
    /// Sort by successor id, for stable, deterministic textual output.
    ById,
    /// Sort by child hash, as required by the binary format so the in-place
    /// reader can binary-search a node's successors.
    ByHash,
  };

  /// Convert the outlined hash tree to stable data, ordering each node's
  /// successors per \p Order.
  void convertToStableData(IdHashNodeStableMapTy &IdNodeStableMap,
                           SuccessorOrder Order = SuccessorOrder::ById) const;

  /// Convert the stable data back to the outlined hash tree.
  void convertFromStableData(const IdHashNodeStableMapTy &IdNodeStableMap);

  /// A read-only tree that reads its nodes in place from a mapped buffer, set
  /// by lazyDeserialize(). Mutually exclusive with the in-memory HashTree;
  /// releaseTree() hands out whichever of the two is set.
  std::unique_ptr<OutlinedHashTree> ReadInPlaceHashTree;
};

} // end namespace llvm

#endif // LLVM_CGDATA_OUTLINEDHASHTREERECORD_H
