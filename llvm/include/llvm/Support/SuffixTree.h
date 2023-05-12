//===- llvm/ADT/SuffixTree.h - Tree for substrings --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Suffix Tree class and Suffix Tree Node struct.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_SUPPORT_SUFFIXTREE_H
#define LLVM_SUPPORT_SUFFIXTREE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace llvm {

/// A node in a suffix tree which represents a substring or suffix.
///
/// Each node has either no children or at least two children, with the root
/// being a exception in the empty tree.
///
/// Children are represented as a map between unsigned integers and nodes. If
/// a node N has a child M on unsigned integer k, then the mapping represented
/// by N is a proper prefix of the mapping represented by M. Note that this,
/// although similar to a trie is somewhat different: each node stores a full
/// substring of the full mapping rather than a single character state.
///
/// Each internal node contains a pointer to the internal node representing
/// the same string, but with the first character chopped off. This is stored
/// in \p Link. Each leaf node stores the start index of its respective
/// suffix in \p SuffixIdx.
struct SuffixTreeNode {
public:
  /// Represents an undefined index in the suffix tree.
  static const unsigned EmptyIdx = -1;
  enum class NodeKind { ST_Leaf, ST_Internal };

private:
  const NodeKind Kind;
  /// The start index of this node's substring in the main string.
  unsigned StartIdx = EmptyIdx;

  /// The length of the string formed by concatenating the edge labels from
  /// the root to this node.
  unsigned ConcatLen = 0;

public:
  NodeKind getKind() const { return Kind; }

  /// \return the start index of this node's substring in the entire string.
  virtual unsigned getStartIdx() const { return StartIdx; }

  /// \returns the end index of this node.
  virtual unsigned getEndIdx() const = 0;

  /// Advance this node's StartIdx by \p Inc.
  void incrementStartIdx(unsigned Inc) { StartIdx += Inc; }

  /// Set the length of the string from the root to this node to \p Len.
  void setConcatLen(unsigned Len) { ConcatLen = Len; }

  /// \returns the length of the string from the root to this node.
  unsigned getConcatLen() const { return ConcatLen; }

  SuffixTreeNode(NodeKind Kind, unsigned StartIdx)
      : Kind(Kind), StartIdx(StartIdx) {}
  virtual ~SuffixTreeNode() = default;
};

struct SuffixTreeInternalNode : SuffixTreeNode {
private:
  /// The end index of this node's substring in the main string.
  ///
  /// Every leaf node must have its \p EndIdx incremented at the end of every
  /// step in the construction algorithm. To avoid having to update O(N)
  /// nodes individually at the end of every step, the end index is stored
  /// as a pointer.
  unsigned EndIdx = EmptyIdx;

  /// A pointer to the internal node representing the same sequence with the
  /// first character chopped off.
  ///
  /// This acts as a shortcut in Ukkonen's algorithm. One of the things that
  /// Ukkonen's algorithm does to achieve linear-time construction is
  /// keep track of which node the next insert should be at. This makes each
  /// insert O(1), and there are a total of O(N) inserts. The suffix link
  /// helps with inserting children of internal nodes.
  ///
  /// Say we add a child to an internal node with associated mapping S. The
  /// next insertion must be at the node representing S - its first character.
  /// This is given by the way that we iteratively build the tree in Ukkonen's
  /// algorithm. The main idea is to look at the suffixes of each prefix in the
  /// string, starting with the longest suffix of the prefix, and ending with
  /// the shortest. Therefore, if we keep pointers between such nodes, we can
  /// move to the next insertion point in O(1) time. If we don't, then we'd
  /// have to query from the root, which takes O(N) time. This would make the
  /// construction algorithm O(N^2) rather than O(N).
  SuffixTreeInternalNode *Link = nullptr;

public:
  static bool classof(const SuffixTreeNode *N) {
    return N->getKind() == NodeKind::ST_Internal;
  }

  /// \returns true if this node is the root of its owning \p SuffixTree.
  bool isRoot() const { return getStartIdx() == EmptyIdx; }

  /// \returns the end index of this node's substring in the entire string.
  unsigned getEndIdx() const override { return EndIdx; }

  /// Sets \p Link to \p L. Assumes \p L is not null.
  void setLink(SuffixTreeInternalNode *L) {
    assert(L && "Cannot set a null link?");
    Link = L;
  }

  /// \returns the pointer to the Link node.
  SuffixTreeInternalNode *getLink() const {
    return Link;
  }

  /// The children of this node.
  ///
  /// A child existing on an unsigned integer implies that from the mapping
  /// represented by the current node, there is a way to reach another
  /// mapping by tacking that character on the end of the current string.
  DenseMap<unsigned, SuffixTreeNode *> Children;

  SuffixTreeInternalNode(unsigned StartIdx, unsigned EndIdx,
                         SuffixTreeInternalNode *Link)
      : SuffixTreeNode(NodeKind::ST_Internal, StartIdx), EndIdx(EndIdx),
        Link(Link) {}

  virtual ~SuffixTreeInternalNode() = default;
};

struct SuffixTreeLeafNode : SuffixTreeNode {
private:
  /// The start index of the suffix represented by this leaf.
  unsigned SuffixIdx = EmptyIdx;

  /// The end index of this node's substring in the main string.
  ///
  /// Every leaf node must have its \p EndIdx incremented at the end of every
  /// step in the construction algorithm. To avoid having to update O(N)
  /// nodes individually at the end of every step, the end index is stored
  /// as a pointer.
  unsigned *EndIdx = nullptr;

public:
  static bool classof(const SuffixTreeNode *N) {
    return N->getKind() == NodeKind::ST_Leaf;
  }

  /// \returns the end index of this node's substring in the entire string.
  unsigned getEndIdx() const override {
    assert(EndIdx && "EndIdx is empty?");
    return *EndIdx;
  }

  /// \returns the start index of the suffix represented by this leaf.
  unsigned getSuffixIdx() const { return SuffixIdx; }
  /// Sets the start index of the suffix represented by this leaf to \p Idx.
  void setSuffixIdx(unsigned Idx) { SuffixIdx = Idx; }
  SuffixTreeLeafNode(unsigned StartIdx, unsigned *EndIdx)
      : SuffixTreeNode(NodeKind::ST_Leaf, StartIdx), EndIdx(EndIdx) {}

  virtual ~SuffixTreeLeafNode() = default;
};

/// A data structure for fast substring queries.
///
/// Suffix trees represent the suffixes of their input strings in their leaves.
/// A suffix tree is a type of compressed trie structure where each node
/// represents an entire substring rather than a single character. Each leaf
/// of the tree is a suffix.
///
/// A suffix tree can be seen as a type of state machine where each state is a
/// substring of the full string. The tree is structured so that, for a string
/// of length N, there are exactly N leaves in the tree. This structure allows
/// us to quickly find repeated substrings of the input string.
///
/// In this implementation, a "string" is a vector of unsigned integers.
/// These integers may result from hashing some data type. A suffix tree can
/// contain 1 or many strings, which can then be queried as one large string.
///
/// The suffix tree is implemented using Ukkonen's algorithm for linear-time
/// suffix tree construction. Ukkonen's algorithm is explained in more detail
/// in the paper by Esko Ukkonen "On-line construction of suffix trees. The
/// paper is available at
///
/// https://www.cs.helsinki.fi/u/ukkonen/SuffixT1withFigs.pdf
class SuffixTree {
public:
  /// Each element is an integer representing an instruction in the module.
  ArrayRef<unsigned> Str;

  /// A repeated substring in the tree.
  struct RepeatedSubstring {
    /// The length of the string.
    unsigned Length;

    /// The start indices of each occurrence.
    SmallVector<unsigned> StartIndices;
  };

private:
  /// Maintains internal nodes in the tree.
  SpecificBumpPtrAllocator<SuffixTreeInternalNode> InternalNodeAllocator;
  /// Maintains leaf nodes in the tree.
  SpecificBumpPtrAllocator<SuffixTreeLeafNode> LeafNodeAllocator;

  /// The root of the suffix tree.
  ///
  /// The root represents the empty string. It is maintained by the
  /// \p NodeAllocator like every other node in the tree.
  SuffixTreeInternalNode *Root = nullptr;

  /// The end index of each leaf in the tree.
  unsigned LeafEndIdx = -1;

  /// Helper struct which keeps track of the next insertion point in
  /// Ukkonen's algorithm.
  struct ActiveState {
    /// The next node to insert at.
    SuffixTreeInternalNode *Node = nullptr;

    /// The index of the first character in the substring currently being added.
    unsigned Idx = SuffixTreeNode::EmptyIdx;

    /// The length of the substring we have to add at the current step.
    unsigned Len = 0;
  };

  /// The point the next insertion will take place at in the
  /// construction algorithm.
  ActiveState Active;

  /// Allocate a leaf node and add it to the tree.
  ///
  /// \param Parent The parent of this node.
  /// \param StartIdx The start index of this node's associated string.
  /// \param Edge The label on the edge leaving \p Parent to this node.
  ///
  /// \returns A pointer to the allocated leaf node.
  SuffixTreeNode *insertLeaf(SuffixTreeInternalNode &Parent, unsigned StartIdx,
                             unsigned Edge);

  /// Allocate an internal node and add it to the tree.
  ///
  /// \param Parent The parent of this node. Only null when allocating the root.
  /// \param StartIdx The start index of this node's associated string.
  /// \param EndIdx The end index of this node's associated string.
  /// \param Edge The label on the edge leaving \p Parent to this node.
  ///
  /// \returns A pointer to the allocated internal node.
  SuffixTreeInternalNode *insertInternalNode(SuffixTreeInternalNode *Parent,
                                             unsigned StartIdx, unsigned EndIdx,
                                             unsigned Edge);

  /// Allocate the root node and add it to the tree.
  ///
  /// \returns A pointer to the root.
  SuffixTreeInternalNode *insertRoot();

  /// Set the suffix indices of the leaves to the start indices of their
  /// respective suffixes.
  void setSuffixIndices();

  /// Construct the suffix tree for the prefix of the input ending at
  /// \p EndIdx.
  ///
  /// Used to construct the full suffix tree iteratively. At the end of each
  /// step, the constructed suffix tree is either a valid suffix tree, or a
  /// suffix tree with implicit suffixes. At the end of the final step, the
  /// suffix tree is a valid tree.
  ///
  /// \param EndIdx The end index of the current prefix in the main string.
  /// \param SuffixesToAdd The number of suffixes that must be added
  /// to complete the suffix tree at the current phase.
  ///
  /// \returns The number of suffixes that have not been added at the end of
  /// this step.
  unsigned extend(unsigned EndIdx, unsigned SuffixesToAdd);

public:
  /// Construct a suffix tree from a sequence of unsigned integers.
  ///
  /// \param Str The string to construct the suffix tree for.
  SuffixTree(const ArrayRef<unsigned> &Str);

  /// Iterator for finding all repeated substrings in the suffix tree.
  struct RepeatedSubstringIterator {
  private:
    /// The current node we're visiting.
    SuffixTreeNode *N = nullptr;

    /// The repeated substring associated with this node.
    RepeatedSubstring RS;

    /// The nodes left to visit.
    SmallVector<SuffixTreeInternalNode *> InternalNodesToVisit;

    /// The minimum length of a repeated substring to find.
    /// Since we're outlining, we want at least two instructions in the range.
    /// FIXME: This may not be true for targets like X86 which support many
    /// instruction lengths.
    const unsigned MinLength = 2;

    /// Move the iterator to the next repeated substring.
    void advance() {
      // Clear the current state. If we're at the end of the range, then this
      // is the state we want to be in.
      RS = RepeatedSubstring();
      N = nullptr;

      // Each leaf node represents a repeat of a string.
      SmallVector<SuffixTreeLeafNode *> LeafChildren;

      // Continue visiting nodes until we find one which repeats more than once.
      while (!InternalNodesToVisit.empty()) {
        LeafChildren.clear();
        auto *Curr = InternalNodesToVisit.back();
        InternalNodesToVisit.pop_back();

        // Keep track of the length of the string associated with the node. If
        // it's too short, we'll quit.
        unsigned Length = Curr->getConcatLen();

        // Iterate over each child, saving internal nodes for visiting, and
        // leaf nodes in LeafChildren. Internal nodes represent individual
        // strings, which may repeat.
        for (auto &ChildPair : Curr->Children) {
          // Save all of this node's children for processing.
          if (auto *InternalChild =
                  dyn_cast<SuffixTreeInternalNode>(ChildPair.second))
            InternalNodesToVisit.push_back(InternalChild);

          // It's not an internal node, so it must be a leaf. If we have a
          // long enough string, then save the leaf children.
          else if (Length >= MinLength)
            LeafChildren.push_back(cast<SuffixTreeLeafNode>(ChildPair.second));
        }

        // The root never represents a repeated substring. If we're looking at
        // that, then skip it.
        if (Curr->isRoot())
          continue;

        // Do we have any repeated substrings?
        if (LeafChildren.size() >= 2) {
          // Yes. Update the state to reflect this, and then bail out.
          N = Curr;
          RS.Length = Length;
          for (SuffixTreeLeafNode *Leaf : LeafChildren)
            RS.StartIndices.push_back(Leaf->getSuffixIdx());
          break;
        }
      }
      // At this point, either NewRS is an empty RepeatedSubstring, or it was
      // set in the above loop. Similarly, N is either nullptr, or the node
      // associated with NewRS.
    }

  public:
    /// Return the current repeated substring.
    RepeatedSubstring &operator*() { return RS; }

    RepeatedSubstringIterator &operator++() {
      advance();
      return *this;
    }

    RepeatedSubstringIterator operator++(int I) {
      RepeatedSubstringIterator It(*this);
      advance();
      return It;
    }

    bool operator==(const RepeatedSubstringIterator &Other) const {
      return N == Other.N;
    }
    bool operator!=(const RepeatedSubstringIterator &Other) const {
      return !(*this == Other);
    }

    RepeatedSubstringIterator(SuffixTreeInternalNode *N) : N(N) {
      // Do we have a non-null node?
      if (!N)
        return;
      // Yes. At the first step, we need to visit all of N's children.
      // Note: This means that we visit N last.
      InternalNodesToVisit.push_back(N);
      advance();
    }
  };

  typedef RepeatedSubstringIterator iterator;
  iterator begin() { return iterator(Root); }
  iterator end() { return iterator(nullptr); }
};

} // namespace llvm

#endif // LLVM_SUPPORT_SUFFIXTREE_H
