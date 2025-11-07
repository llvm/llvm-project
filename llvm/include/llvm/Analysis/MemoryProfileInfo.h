//===- llvm/Analysis/MemoryProfileInfo.h - memory profile info ---*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities to analyze memory profile information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORYPROFILEINFO_H
#define LLVM_ANALYSIS_MEMORYPROFILEINFO_H

#include "llvm/IR/Metadata.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/Compiler.h"
#include <map>

namespace llvm {

class OptimizationRemarkEmitter;

namespace memprof {

/// Whether the alloc memeprof metadata will include context size info for all
/// MIBs.
LLVM_ABI bool metadataIncludesAllContextSizeInfo();

/// Whether the alloc memprof metadata may include context size info for some
/// MIBs (but possibly not all).
LLVM_ABI bool metadataMayIncludeContextSizeInfo();

/// Whether we need to record the context size info in the alloc trie used to
/// build metadata.
LLVM_ABI bool recordContextSizeInfoForAnalysis();

/// Build callstack metadata from the provided list of call stack ids. Returns
/// the resulting metadata node.
LLVM_ABI MDNode *buildCallstackMetadata(ArrayRef<uint64_t> CallStack,
                                        LLVMContext &Ctx);

/// Build metadata from the provided list of full stack id and profiled size, to
/// use when reporting of hinted sizes is enabled.
LLVM_ABI MDNode *
buildContextSizeMetadata(ArrayRef<ContextTotalSize> ContextSizeInfo,
                         LLVMContext &Ctx);

/// Returns the stack node from an MIB metadata node.
LLVM_ABI MDNode *getMIBStackNode(const MDNode *MIB);

/// Returns the allocation type from an MIB metadata node.
LLVM_ABI AllocationType getMIBAllocType(const MDNode *MIB);

/// Returns the string to use in attributes with the given type.
LLVM_ABI std::string getAllocTypeAttributeString(AllocationType Type);

/// True if the AllocTypes bitmask contains just a single type.
LLVM_ABI bool hasSingleAllocType(uint8_t AllocTypes);

/// Removes any existing "ambiguous" memprof attribute. Called before we apply a
/// specific allocation type such as "cold", "notcold", or "hot".
LLVM_ABI void removeAnyExistingAmbiguousAttribute(CallBase *CB);

/// Adds an "ambiguous" memprof attribute to call with a matched allocation
/// profile but that we haven't yet been able to disambiguate.
LLVM_ABI void addAmbiguousAttribute(CallBase *CB);

/// Class to build a trie of call stack contexts for a particular profiled
/// allocation call, along with their associated allocation types.
/// The allocation will be at the root of the trie, which is then used to
/// compute the minimum lists of context ids needed to associate a call context
/// with a single allocation type.
class CallStackTrie {
private:
  struct CallStackTrieNode {
    // Allocation types for call context sharing the context prefix at this
    // node.
    uint8_t AllocTypes;
    // If the user has requested reporting of hinted sizes, keep track of the
    // associated full stack id and profiled sizes. Can have more than one
    // after trimming (e.g. when building from metadata). This is only placed on
    // the last (root-most) trie node for each allocation context.
    std::vector<ContextTotalSize> ContextSizeInfo;
    // Map of caller stack id to the corresponding child Trie node.
    std::map<uint64_t, CallStackTrieNode *> Callers;
    CallStackTrieNode(AllocationType Type)
        : AllocTypes(static_cast<uint8_t>(Type)) {}
    void addAllocType(AllocationType AllocType) {
      AllocTypes |= static_cast<uint8_t>(AllocType);
    }
    void removeAllocType(AllocationType AllocType) {
      AllocTypes &= ~static_cast<uint8_t>(AllocType);
    }
    bool hasAllocType(AllocationType AllocType) const {
      return AllocTypes & static_cast<uint8_t>(AllocType);
    }
  };

  // The node for the allocation at the root.
  CallStackTrieNode *Alloc = nullptr;
  // The allocation's leaf stack id.
  uint64_t AllocStackId = 0;

  // If the client provides a remarks emitter object, we will emit remarks on
  // allocations for which we apply non-context sensitive allocation hints.
  OptimizationRemarkEmitter *ORE;

  // The maximum size of a cold allocation context, from the profile summary.
  uint64_t MaxColdSize;

  // Tracks whether we have built the Trie from existing MD_memprof metadata. We
  // apply different heuristics for determining whether to discard non-cold
  // contexts when rebuilding as we have lost information available during the
  // original profile match.
  bool BuiltFromExistingMetadata = false;

  void deleteTrieNode(CallStackTrieNode *Node) {
    if (!Node)
      return;
    for (auto C : Node->Callers)
      deleteTrieNode(C.second);
    delete Node;
  }

  // Recursively build up a complete list of context size information from the
  // trie nodes reached form the given Node, for hint size reporting.
  void collectContextSizeInfo(CallStackTrieNode *Node,
                              std::vector<ContextTotalSize> &ContextSizeInfo);

  // Recursively convert hot allocation types to notcold, since we don't
  // actually do any cloning for hot contexts, to facilitate more aggressive
  // pruning of contexts.
  void convertHotToNotCold(CallStackTrieNode *Node);

  // Recursive helper to trim contexts and create metadata nodes.
  bool buildMIBNodes(CallStackTrieNode *Node, LLVMContext &Ctx,
                     std::vector<uint64_t> &MIBCallStack,
                     std::vector<Metadata *> &MIBNodes,
                     bool CalleeHasAmbiguousCallerContext, uint64_t &TotalBytes,
                     uint64_t &ColdBytes);

public:
  CallStackTrie(OptimizationRemarkEmitter *ORE = nullptr,
                uint64_t MaxColdSize = 0)
      : ORE(ORE), MaxColdSize(MaxColdSize) {}
  ~CallStackTrie() { deleteTrieNode(Alloc); }

  bool empty() const { return Alloc == nullptr; }

  /// Add a call stack context with the given allocation type to the Trie.
  /// The context is represented by the list of stack ids (computed during
  /// matching via a debug location hash), expected to be in order from the
  /// allocation call down to the bottom of the call stack (i.e. callee to
  /// caller order).
  LLVM_ABI void
  addCallStack(AllocationType AllocType, ArrayRef<uint64_t> StackIds,
               std::vector<ContextTotalSize> ContextSizeInfo = {});

  /// Add the call stack context along with its allocation type from the MIB
  /// metadata to the Trie.
  LLVM_ABI void addCallStack(MDNode *MIB);

  /// Build and attach the minimal necessary MIB metadata. If the alloc has a
  /// single allocation type, add a function attribute instead. The reason for
  /// adding an attribute in this case is that it matches how the behavior for
  /// allocation calls will be communicated to lib call simplification after
  /// cloning or another optimization to distinguish the allocation types,
  /// which is lower overhead and more direct than maintaining this metadata.
  /// Returns true if memprof metadata attached, false if not (attribute added).
  LLVM_ABI bool buildAndAttachMIBMetadata(CallBase *CI);

  /// Add an attribute for the given allocation type to the call instruction.
  /// If hinted by reporting is enabled, a message is emitted with the given
  /// descriptor used to identify the category of single allocation type.
  LLVM_ABI void addSingleAllocTypeAttribute(CallBase *CI, AllocationType AT,
                                            StringRef Descriptor);
};

/// Helper class to iterate through stack ids in both metadata (memprof MIB and
/// callsite) and the corresponding ThinLTO summary data structures
/// (CallsiteInfo and MIBInfo). This simplifies implementation of client code
/// which doesn't need to worry about whether we are operating with IR (Regular
/// LTO), or summary (ThinLTO).
template <class NodeT, class IteratorT> class CallStack {
public:
  CallStack(const NodeT *N = nullptr) : N(N) {}

  // Implement minimum required methods for range-based for loop.
  // The default implementation assumes we are operating on ThinLTO data
  // structures, which have a vector of StackIdIndices. There are specialized
  // versions provided to iterate through metadata.
  struct CallStackIterator {
    const NodeT *N = nullptr;
    IteratorT Iter;
    CallStackIterator(const NodeT *N, bool End);
    uint64_t operator*();
    bool operator==(const CallStackIterator &rhs) { return Iter == rhs.Iter; }
    bool operator!=(const CallStackIterator &rhs) { return !(*this == rhs); }
    void operator++() { ++Iter; }
  };

  bool empty() const { return N == nullptr; }

  CallStackIterator begin() const;
  CallStackIterator end() const { return CallStackIterator(N, /*End*/ true); }
  CallStackIterator beginAfterSharedPrefix(const CallStack &Other);
  uint64_t back() const;

private:
  const NodeT *N = nullptr;
};

template <class NodeT, class IteratorT>
CallStack<NodeT, IteratorT>::CallStackIterator::CallStackIterator(
    const NodeT *N, bool End)
    : N(N) {
  if (!N) {
    Iter = nullptr;
    return;
  }
  Iter = End ? N->StackIdIndices.end() : N->StackIdIndices.begin();
}

template <class NodeT, class IteratorT>
uint64_t CallStack<NodeT, IteratorT>::CallStackIterator::operator*() {
  assert(Iter != N->StackIdIndices.end());
  return *Iter;
}

template <class NodeT, class IteratorT>
uint64_t CallStack<NodeT, IteratorT>::back() const {
  assert(N);
  return N->StackIdIndices.back();
}

template <class NodeT, class IteratorT>
typename CallStack<NodeT, IteratorT>::CallStackIterator
CallStack<NodeT, IteratorT>::begin() const {
  return CallStackIterator(N, /*End*/ false);
}

template <class NodeT, class IteratorT>
typename CallStack<NodeT, IteratorT>::CallStackIterator
CallStack<NodeT, IteratorT>::beginAfterSharedPrefix(const CallStack &Other) {
  CallStackIterator Cur = begin();
  for (CallStackIterator OtherCur = Other.begin();
       Cur != end() && OtherCur != Other.end(); ++Cur, ++OtherCur)
    assert(*Cur == *OtherCur);
  return Cur;
}

/// Specializations for iterating through IR metadata stack contexts.
template <>
LLVM_ABI
CallStack<MDNode, MDNode::op_iterator>::CallStackIterator::CallStackIterator(
    const MDNode *N, bool End);
template <>
LLVM_ABI uint64_t
CallStack<MDNode, MDNode::op_iterator>::CallStackIterator::operator*();
template <>
LLVM_ABI uint64_t CallStack<MDNode, MDNode::op_iterator>::back() const;

} // end namespace memprof
} // end namespace llvm

#endif
