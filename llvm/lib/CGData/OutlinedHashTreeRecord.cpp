//===-- OutlinedHashTreeRecord.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the OutlinedHashTreeRecord class. This class holds the outlined
// hash tree for both serialization and deserialization processes. It utilizes
// two data formats for serialization: raw binary data and YAML.
// These two formats can be used interchangeably.
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/OutlinedHashTreeRecord.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "outlined-hash-tree"

using namespace llvm;
using namespace llvm::support;

namespace {
/// On-disk binary layout of the serialized outlined hash tree. The structure is
/// as follows:
/// - Number of nodes
/// - Padding to keep the block region that follows 8-byte aligned
/// - Total size of the block region, for advancing past the blob when merging
/// - Block region: one self-describing block per node, the root node first,
///   each node identified by its block's byte offset within the region (the
///   root is at offset 0):
///   - Number of successors
///   - Terminal count
///   - Successor entries, sorted by child hash so readSuccessor() can
///     binary-search them:
///     - Child hash (a node's hash is stored only here, in its parent's
///       entry; the root's is implicitly 0)
///     - Byte offset of the child's block within the block region
/// - Padding to align the blob to 8 bytes, so concatenated blobs stay aligned
struct OutlinedHashTreeFormat {
  static constexpr unsigned BlobHeaderSize = 16;
  static constexpr unsigned BlockRegionSizeFieldOffset = 8;
  static constexpr unsigned NumSuccFieldOffset = 0;
  static constexpr unsigned TerminalsFieldOffset = 4;
  static constexpr unsigned BlockHeaderSize = 8;
  static constexpr unsigned EntryHashFieldOffset = 0;
  static constexpr unsigned EntryChildOffsetFieldOffset = 8;
  static constexpr unsigned EntrySize = 16;
  static constexpr unsigned BlobAlign = 8;
};
} // namespace

using Fmt = OutlinedHashTreeFormat;

namespace llvm {
namespace yaml {

template <> struct MappingTraits<HashNodeStable> {
  static void mapping(IO &io, HashNodeStable &res) {
    io.mapRequired("Hash", res.Hash);
    io.mapRequired("Terminals", res.Terminals);
    io.mapRequired("SuccessorIds", res.SuccessorIds);
  }
};

template <> struct CustomMappingTraits<IdHashNodeStableMapTy> {
  static void inputOne(IO &io, StringRef Key, IdHashNodeStableMapTy &V) {
    HashNodeStable NodeStable;
    io.mapRequired(Key, NodeStable);
    unsigned Id;
    if (Key.getAsInteger(0, Id)) {
      io.setError("Id not an integer");
      return;
    }
    V.insert({Id, NodeStable});
  }

  static void output(IO &io, IdHashNodeStableMapTy &V) {
    for (auto Iter = V.begin(); Iter != V.end(); ++Iter)
      io.mapRequired(utostr(Iter->first), Iter->second);
  }
};

} // namespace yaml
} // namespace llvm

void OutlinedHashTreeRecord::serialize(raw_ostream &OS) const {
  IdHashNodeStableMapTy IdNodeStableMap;
  convertToStableData(IdNodeStableMap, SuccessorOrder::ByHash);
  support::endian::Writer Writer(OS, endianness::little);

  // Compute the offsets for each node to avoid writeback.
  uint32_t NumNodes = static_cast<uint32_t>(IdNodeStableMap.size());
  std::vector<uint64_t> BlockOffset(NumNodes);
  uint64_t Offset = 0;
  for (const auto &[Id, NodeStable] : IdNodeStableMap) {
    BlockOffset[Id] = Offset;
    Offset += Fmt::BlockHeaderSize +
              uint64_t(Fmt::EntrySize) * NodeStable.SuccessorIds.size();
  }
  uint64_t BlockRegionSize = Offset;

  Writer.write<uint32_t>(NumNodes);
  Writer.write<uint32_t>(0);
  Writer.write<uint64_t>(BlockRegionSize);
  for (const auto &[Id, NodeStable] : IdNodeStableMap) {
    Writer.write<uint32_t>(
        static_cast<uint32_t>(NodeStable.SuccessorIds.size()));
    Writer.write<uint32_t>(NodeStable.Terminals);
    for (unsigned SuccId : NodeStable.SuccessorIds) {
      Writer.write<uint64_t>(IdNodeStableMap.at(SuccId).Hash);
      Writer.write<uint64_t>(BlockOffset[SuccId]);
    }
  }

  // Pad the blob to OutlinedHashTreeFormat::BlobAlign so that concatenated
  // blobs stay aligned and deserialize() can advance from one to the next.
  uint64_t BlobSize = Fmt::BlobHeaderSize + BlockRegionSize;
  for (uint64_t I = BlobSize, E = alignTo(BlobSize, Fmt::BlobAlign); I < E; ++I)
    Writer.write<uint8_t>(0);
}

void OutlinedHashTreeRecord::deserialize(const unsigned char *&Ptr) {
  const unsigned char *BlobStart = Ptr;
  uint64_t BlockRegionSize = endian::read<uint64_t, unaligned>(
      BlobStart + Fmt::BlockRegionSizeFieldOffset, endianness::little);
  const unsigned char *BlockBase = BlobStart + Fmt::BlobHeaderSize;
  HashTree = createInMemory(BlockBase);

  // Advance Ptr past the blob, including the alignment padding serialize()
  // appended, so a concatenated merge loop lands on the next blob / the end.
  size_t ContentSize = (BlockBase + BlockRegionSize) - BlobStart;
  Ptr = BlobStart + alignTo(ContentSize, Fmt::BlobAlign);
}

std::unique_ptr<OutlinedHashTree>
OutlinedHashTreeRecord::createReadInPlace(std::shared_ptr<MemoryBuffer> Buffer,
                                          uint64_t BlobOffset) {
  const auto *BlobStart =
      reinterpret_cast<const unsigned char *>(Buffer->getBufferStart()) +
      BlobOffset;
  uint32_t NumNodes =
      endian::read<uint32_t, unaligned>(BlobStart, endianness::little);
  const unsigned char *BlockBase = BlobStart + Fmt::BlobHeaderSize;
  return std::make_unique<OutlinedHashTree>(std::move(Buffer), BlockBase,
                                            NumNodes);
}

void OutlinedHashTreeRecord::lazyDeserialize(
    std::shared_ptr<MemoryBuffer> Buffer, uint64_t Offset) {
  // The read-in-place tree becomes the record's only tree; drop the empty
  // in-memory tree the constructor created.
  HashTree.reset();
  ReadInPlaceHashTree = createReadInPlace(std::move(Buffer), Offset);
}

static uint32_t readU32(const unsigned char *P) {
  return endian::read<uint32_t, unaligned>(P, endianness::little);
}
static uint64_t readU64(const unsigned char *P) {
  return endian::read<uint64_t, unaligned>(P, endianness::little);
}

std::optional<uint64_t>
OutlinedHashTreeRecord::readSuccessor(const unsigned char *Block,
                                      stable_hash H) {
  const unsigned char *Entries = Block + Fmt::BlockHeaderSize;
  uint32_t Lo = 0, Hi = readU32(Block + Fmt::NumSuccFieldOffset);
  while (Lo < Hi) {
    uint32_t Mid = Lo + (Hi - Lo) / 2;
    const unsigned char *E = Entries + Fmt::EntrySize * Mid;
    stable_hash ChildHash = readU64(E + Fmt::EntryHashFieldOffset);
    if (ChildHash < H)
      Lo = Mid + 1;
    else if (ChildHash > H)
      Hi = Mid;
    else
      return readU64(E + Fmt::EntryChildOffsetFieldOffset);
  }
  return std::nullopt;
}

unsigned OutlinedHashTreeRecord::readTerminals(const unsigned char *Block) {
  return readU32(Block + Fmt::TerminalsFieldOffset);
}

std::unique_ptr<MutableOutlinedHashTree>
OutlinedHashTreeRecord::createInMemory(const unsigned char *BlockBase) {
  auto Tree = std::make_unique<MutableOutlinedHashTree>();
  HashNode &Root = *Tree->getRoot();
  // Iterative to avoid deep-chain stack overflow, in production the tree could
  // have very long single-child chains.
  SmallVector<std::pair<HashNode *, uint64_t>> Stack;
  Stack.emplace_back(&Root, uint64_t(0));
  while (!Stack.empty()) {
    auto [Node, Off] = Stack.pop_back_val();
    const unsigned char *Block = BlockBase + Off;
    uint32_t NumSucc = readU32(Block + Fmt::NumSuccFieldOffset);
    if (unsigned T = readTerminals(Block))
      Node->Terminals = T;
    const unsigned char *Entries = Block + Fmt::BlockHeaderSize;
    for (uint32_t K = 0; K < NumSucc; ++K) {
      const unsigned char *E = Entries + Fmt::EntrySize * K;
      stable_hash ChildHash = readU64(E + Fmt::EntryHashFieldOffset);
      uint64_t ChildOff = readU64(E + Fmt::EntryChildOffsetFieldOffset);
      auto Child = std::make_unique<HashNode>();
      Child->Hash = ChildHash;
      HashNode *ChildPtr = Child.get();
      Node->Successors.try_emplace(ChildHash, std::move(Child));
      Stack.emplace_back(ChildPtr, ChildOff);
    }
  }
  return Tree;
}

std::unique_ptr<MutableOutlinedHashTree>
OutlinedHashTreeRecord::createInMemory(std::unique_ptr<OutlinedHashTree> Tree) {
  if (Tree->isReadInPlace())
    return createInMemory(Tree->BlockBase);
  // An in-memory tree is already a MutableOutlinedHashTree; move its nodes into
  // a fresh one so the caller gets the concrete mutable type back.
  auto InMemory = std::make_unique<MutableOutlinedHashTree>();
  *InMemory->getRoot() = std::move(Tree->Root);
  return InMemory;
}

void OutlinedHashTreeRecord::serializeYAML(yaml::Output &YOS) const {
  IdHashNodeStableMapTy IdNodeStableMap;
  convertToStableData(IdNodeStableMap);

  YOS << IdNodeStableMap;
}

void OutlinedHashTreeRecord::deserializeYAML(yaml::Input &YIS) {
  IdHashNodeStableMapTy IdNodeStableMap;

  YIS >> IdNodeStableMap;
  YIS.nextDocument();

  convertFromStableData(IdNodeStableMap);
}

void OutlinedHashTreeRecord::convertToStableData(
    IdHashNodeStableMapTy &IdNodeStableMap, SuccessorOrder Order) const {
  // Build NodeIdMap
  HashNodeIdMapTy NodeIdMap;
  HashTree->walkGraph(
      [&NodeIdMap](const HashNode *Current) {
        size_t Index = NodeIdMap.size();
        NodeIdMap[Current] = Index;
        assert((Index + 1 == NodeIdMap.size()) &&
               "Duplicate key in NodeIdMap: 'Current' should be unique.");
      },
      /*EdgeCallbackFn=*/nullptr, /*SortedWork=*/true);

  // Convert NodeIdMap to NodeStableMap
  for (auto &P : NodeIdMap) {
    auto *Node = P.first;
    auto Id = P.second;
    HashNodeStable NodeStable;
    NodeStable.Hash = Node->Hash;
    NodeStable.Terminals = Node->Terminals.value_or(0);
    for (auto &P : Node->Successors)
      NodeStable.SuccessorIds.push_back(NodeIdMap[P.second.get()]);
    IdNodeStableMap[Id] = NodeStable;
  }

  // Sort each node's successors per the requested order. ById gives stable
  // textual output; ByHash lets the in-place reader binary-search a
  // node's successors by child hash.
  for (auto &P : IdNodeStableMap) {
    switch (Order) {
    case SuccessorOrder::ById:
      llvm::sort(P.second.SuccessorIds);
      break;
    case SuccessorOrder::ByHash:
      llvm::sort(P.second.SuccessorIds, [&](unsigned A, unsigned B) {
        return static_cast<uint64_t>(IdNodeStableMap.at(A).Hash) <
               static_cast<uint64_t>(IdNodeStableMap.at(B).Hash);
      });
      break;
    }
  }
}

void OutlinedHashTreeRecord::convertFromStableData(
    const IdHashNodeStableMapTy &IdNodeStableMap) {
  IdHashNodeMapTy IdNodeMap;
  // Initialize the root node at 0.
  IdNodeMap[0] = HashTree->getRoot();
  assert(IdNodeMap[0]->Successors.empty());

  for (auto &P : IdNodeStableMap) {
    auto Id = P.first;
    const HashNodeStable &NodeStable = P.second;
    assert(IdNodeMap.count(Id));
    HashNode *Curr = IdNodeMap[Id];
    Curr->Hash = NodeStable.Hash;
    if (NodeStable.Terminals)
      Curr->Terminals = NodeStable.Terminals;
    auto &Successors = Curr->Successors;
    assert(Successors.empty());
    for (auto SuccessorId : NodeStable.SuccessorIds) {
      auto Sucessor = std::make_unique<HashNode>();
      IdNodeMap[SuccessorId] = Sucessor.get();
      auto Hash = IdNodeStableMap.at(SuccessorId).Hash;
      Successors[Hash] = std::move(Sucessor);
    }
  }
}
