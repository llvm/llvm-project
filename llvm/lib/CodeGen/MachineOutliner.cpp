//===---- MachineOutliner.cpp - Outline instructions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Replaces repeated sequences of instructions with function calls.
///
/// This works by placing every instruction from every basic block in a
/// suffix tree, and repeatedly querying that tree for repeated sequences of
/// instructions. If a sequence of instructions appears often, then it ought
/// to be beneficial to pull out into a function.
///
/// This was originally presented at the 2016 LLVM Developers' Meeting in the
/// talk "Reducing Code Size Using Outlining". For a high-level overview of
/// how this pass works, the talk is available on YouTube at
///
/// https://www.youtube.com/watch?v=yorld-WSOeU
///
/// The slides for the talk are available at
///
/// http://www.llvm.org/devmtg/2016-11/Slides/Paquette-Outliner.pdf
///
/// The talk provides an overview of how the outliner finds candidates and
/// ultimately outlines them. It describes how the main data structure for this
/// pass, the suffix tree, is queried and purged for candidates. It also gives
/// a simplified suffix tree construction algorithm for suffix trees based off
/// of the algorithm actually used here, Ukkonen's algorithm.
///
/// For the original RFC for this pass, please see
///
/// http://lists.llvm.org/pipermail/llvm-dev/2016-August/104170.html
///
/// For more information on the suffix tree data structure, please see
/// https://www.cs.helsinki.fi/u/ukkonen/SuffixT1withFigs.pdf
///
//===----------------------------------------------------------------------===//
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <functional>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#define DEBUG_TYPE "machine-outliner"

using namespace llvm;

STATISTIC(NumOutlined, "Number of candidates outlined");
STATISTIC(FunctionsCreated, "Number of functions created");

namespace {

/// Represents an undefined index in the suffix tree.
const size_t EmptyIdx = -1;

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

  /// The children of this node.
  ///
  /// A child existing on an unsigned integer implies that from the mapping
  /// represented by the current node, there is a way to reach another
  /// mapping by tacking that character on the end of the current string.
  DenseMap<unsigned, SuffixTreeNode *> Children;

  /// A flag set to false if the node has been pruned from the tree.
  bool IsInTree = true;

  /// The start index of this node's substring in the main string.
  size_t StartIdx = EmptyIdx;

  /// The end index of this node's substring in the main string.
  ///
  /// Every leaf node must have its \p EndIdx incremented at the end of every
  /// step in the construction algorithm. To avoid having to update O(N)
  /// nodes individually at the end of every step, the end index is stored
  /// as a pointer.
  size_t *EndIdx = nullptr;

  /// For leaves, the start index of the suffix represented by this node.
  ///
  /// For all other nodes, this is ignored.
  size_t SuffixIdx = EmptyIdx;

  /// \brief For internal nodes, a pointer to the internal node representing
  /// the same sequence with the first character chopped off.
  ///
  /// This has two major purposes in the suffix tree. The first is as a
  /// shortcut in Ukkonen's construction algorithm. One of the things that
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
  ///
  /// The suffix link is also used during the tree pruning process to let us
  /// quickly throw out a bunch of potential overlaps. Say we have a sequence
  /// S we want to outline. Then each of its suffixes contribute to at least
  /// one overlapping case. Therefore, we can follow the suffix links
  /// starting at the node associated with S to the root and "delete" those
  /// nodes, save for the root. For each candidate, this removes
  /// O(|candidate|) overlaps from the search space. We don't actually
  /// completely invalidate these nodes though; doing that is far too
  /// aggressive. Consider the following pathological string:
  ///
  /// 1 2 3 1 2 3 2 3 2 3 2 3 2 3 2 3 2 3
  ///
  /// If we, for the sake of example, outlined 1 2 3, then we would throw
  /// out all instances of 2 3. This isn't desirable. To get around this,
  /// when we visit a link node, we decrement its occurrence count by the
  /// number of sequences we outlined in the current step. In the pathological
  /// example, the 2 3 node would have an occurrence count of 8, while the
  /// 1 2 3 node would have an occurrence count of 2. Thus, the 2 3 node
  /// would survive to the next round allowing us to outline the extra
  /// instances of 2 3.
  SuffixTreeNode *Link = nullptr;

  /// The parent of this node. Every node except for the root has a parent.
  SuffixTreeNode *Parent = nullptr;

  /// The number of times this node's string appears in the tree.
  ///
  /// This is equal to the number of leaf children of the string. It represents
  /// the number of suffixes that the node's string is a prefix of.
  size_t OccurrenceCount = 0;

  /// Returns true if this node is a leaf.
  bool isLeaf() const { return SuffixIdx != EmptyIdx; }

  /// Returns true if this node is the root of its owning \p SuffixTree.
  bool isRoot() const { return StartIdx == EmptyIdx; }

  /// Return the number of elements in the substring associated with this node.
  size_t size() const {

    // Is it the root? If so, it's the empty string so return 0.
    if (isRoot())
      return 0;

    assert(*EndIdx != EmptyIdx && "EndIdx is undefined!");

    // Size = the number of elements in the string.
    // For example, [0 1 2 3] has length 4, not 3. 3-0 = 3, so we have 3-0+1.
    return *EndIdx - StartIdx + 1;
  }

  SuffixTreeNode(size_t StartIdx, size_t *EndIdx, SuffixTreeNode *Link,
                 SuffixTreeNode *Parent)
      : StartIdx(StartIdx), EndIdx(EndIdx), Link(Link), Parent(Parent) {}

  SuffixTreeNode() {}
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
private:
  /// Each element is an integer representing an instruction in the module.
  ArrayRef<unsigned> Str;

  /// Maintains each node in the tree.
  SpecificBumpPtrAllocator<SuffixTreeNode> NodeAllocator;

  /// The root of the suffix tree.
  ///
  /// The root represents the empty string. It is maintained by the
  /// \p NodeAllocator like every other node in the tree.
  SuffixTreeNode *Root = nullptr;

  /// Stores each leaf in the tree for better pruning.
  std::vector<SuffixTreeNode *> LeafVector;

  /// Maintains the end indices of the internal nodes in the tree.
  ///
  /// Each internal node is guaranteed to never have its end index change
  /// during the construction algorithm; however, leaves must be updated at
  /// every step. Therefore, we need to store leaf end indices by reference
  /// to avoid updating O(N) leaves at every step of construction. Thus,
  /// every internal node must be allocated its own end index.
  BumpPtrAllocator InternalEndIdxAllocator;

  /// The end index of each leaf in the tree.
  size_t LeafEndIdx = -1;

  /// \brief Helper struct which keeps track of the next insertion point in
  /// Ukkonen's algorithm.
  struct ActiveState {
    /// The next node to insert at.
    SuffixTreeNode *Node;

    /// The index of the first character in the substring currently being added.
    size_t Idx = EmptyIdx;

    /// The length of the substring we have to add at the current step.
    size_t Len = 0;
  };

  /// \brief The point the next insertion will take place at in the
  /// construction algorithm.
  ActiveState Active;

  /// Allocate a leaf node and add it to the tree.
  ///
  /// \param Parent The parent of this node.
  /// \param StartIdx The start index of this node's associated string.
  /// \param Edge The label on the edge leaving \p Parent to this node.
  ///
  /// \returns A pointer to the allocated leaf node.
  SuffixTreeNode *insertLeaf(SuffixTreeNode &Parent, size_t StartIdx,
                             unsigned Edge) {

    assert(StartIdx <= LeafEndIdx && "String can't start after it ends!");

    SuffixTreeNode *N = new (NodeAllocator.Allocate()) SuffixTreeNode(StartIdx, 
                                                                   &LeafEndIdx,
                                                                       nullptr,
                                                                      &Parent);
    Parent.Children[Edge] = N;

    return N;
  }

  /// Allocate an internal node and add it to the tree.
  ///
  /// \param Parent The parent of this node. Only null when allocating the root.
  /// \param StartIdx The start index of this node's associated string.
  /// \param EndIdx The end index of this node's associated string.
  /// \param Edge The label on the edge leaving \p Parent to this node.
  ///
  /// \returns A pointer to the allocated internal node.
  SuffixTreeNode *insertInternalNode(SuffixTreeNode *Parent, size_t StartIdx,
                                     size_t EndIdx, unsigned Edge) {

    assert(StartIdx <= EndIdx && "String can't start after it ends!");
    assert(!(!Parent && StartIdx != EmptyIdx) &&
    "Non-root internal nodes must have parents!");

    size_t *E = new (InternalEndIdxAllocator) size_t(EndIdx);
    SuffixTreeNode *N = new (NodeAllocator.Allocate()) SuffixTreeNode(StartIdx,
                                                                      E,
                                                                      Root,
                                                                      Parent);
    if (Parent)
      Parent->Children[Edge] = N;

    return N;
  }

  /// \brief Set the suffix indices of the leaves to the start indices of their
  /// respective suffixes. Also stores each leaf in \p LeafVector at its
  /// respective suffix index.
  ///
  /// \param[in] CurrNode The node currently being visited.
  /// \param CurrIdx The current index of the string being visited.
  void setSuffixIndices(SuffixTreeNode &CurrNode, size_t CurrIdx) {

    bool IsLeaf = CurrNode.Children.size() == 0 && !CurrNode.isRoot();

    // Traverse the tree depth-first.
    for (auto &ChildPair : CurrNode.Children) {
      assert(ChildPair.second && "Node had a null child!");
      setSuffixIndices(*ChildPair.second,
                       CurrIdx + ChildPair.second->size());
    }

    // Is this node a leaf?
    if (IsLeaf) {
      // If yes, give it a suffix index and bump its parent's occurrence count.
      CurrNode.SuffixIdx = Str.size() - CurrIdx;
      assert(CurrNode.Parent && "CurrNode had no parent!");
      CurrNode.Parent->OccurrenceCount++;

      // Store the leaf in the leaf vector for pruning later.
      LeafVector[CurrNode.SuffixIdx] = &CurrNode;
    }
  }

  /// \brief Construct the suffix tree for the prefix of the input ending at
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
  unsigned extend(size_t EndIdx, size_t SuffixesToAdd) {
    SuffixTreeNode *NeedsLink = nullptr;

    while (SuffixesToAdd > 0) {
    
      // Are we waiting to add anything other than just the last character?
      if (Active.Len == 0) {
        // If not, then say the active index is the end index.
        Active.Idx = EndIdx;
      }

      assert(Active.Idx <= EndIdx && "Start index can't be after end index!");

      // The first character in the current substring we're looking at.
      unsigned FirstChar = Str[Active.Idx];

      // Have we inserted anything starting with FirstChar at the current node?
      if (Active.Node->Children.count(FirstChar) == 0) {
        // If not, then we can just insert a leaf and move too the next step.
        insertLeaf(*Active.Node, EndIdx, FirstChar);

        // The active node is an internal node, and we visited it, so it must
        // need a link if it doesn't have one.
        if (NeedsLink) {
          NeedsLink->Link = Active.Node;
          NeedsLink = nullptr;
        }
      } else {
        // There's a match with FirstChar, so look for the point in the tree to
        // insert a new node.
        SuffixTreeNode *NextNode = Active.Node->Children[FirstChar];

        size_t SubstringLen = NextNode->size();

        // Is the current suffix we're trying to insert longer than the size of
        // the child we want to move to?
        if (Active.Len >= SubstringLen) {
          // If yes, then consume the characters we've seen and move to the next
          // node.
          Active.Idx += SubstringLen;
          Active.Len -= SubstringLen;
          Active.Node = NextNode;
          continue;
        }

        // Otherwise, the suffix we're trying to insert must be contained in the
        // next node we want to move to.
        unsigned LastChar = Str[EndIdx];

        // Is the string we're trying to insert a substring of the next node?
        if (Str[NextNode->StartIdx + Active.Len] == LastChar) {
          // If yes, then we're done for this step. Remember our insertion point
          // and move to the next end index. At this point, we have an implicit
          // suffix tree.
          if (NeedsLink && !Active.Node->isRoot()) {
            NeedsLink->Link = Active.Node;
            NeedsLink = nullptr;
          }

          Active.Len++;
          break;
        }

        // The string we're trying to insert isn't a substring of the next node,
        // but matches up to a point. Split the node.
        //
        // For example, say we ended our search at a node n and we're trying to
        // insert ABD. Then we'll create a new node s for AB, reduce n to just
        // representing C, and insert a new leaf node l to represent d. This
        // allows us to ensure that if n was a leaf, it remains a leaf.
        //
        //   | ABC  ---split--->  | AB
        //   n                    s
        //                     C / \ D
        //                      n   l

        // The node s from the diagram
        SuffixTreeNode *SplitNode =
            insertInternalNode(Active.Node,
                               NextNode->StartIdx,
                               NextNode->StartIdx + Active.Len - 1,
                               FirstChar);

        // Insert the new node representing the new substring into the tree as
        // a child of the split node. This is the node l from the diagram.
        insertLeaf(*SplitNode, EndIdx, LastChar);

        // Make the old node a child of the split node and update its start
        // index. This is the node n from the diagram.
        NextNode->StartIdx += Active.Len;
        NextNode->Parent = SplitNode;
        SplitNode->Children[Str[NextNode->StartIdx]] = NextNode;

        // SplitNode is an internal node, update the suffix link.
        if (NeedsLink)
          NeedsLink->Link = SplitNode;

        NeedsLink = SplitNode;
      }

      // We've added something new to the tree, so there's one less suffix to
      // add.
      SuffixesToAdd--;

      if (Active.Node->isRoot()) {
        if (Active.Len > 0) {
          Active.Len--;
          Active.Idx = EndIdx - SuffixesToAdd + 1;
        }
      } else {
        // Start the next phase at the next smallest suffix.
        Active.Node = Active.Node->Link;
      }
    }

    return SuffixesToAdd;
  }

  /// \brief Return the start index and length of a string which maximizes a
  /// benefit function by traversing the tree depth-first.
  ///
  /// Helper function for \p bestRepeatedSubstring.
  ///
  /// \param CurrNode The node currently being visited.
  /// \param CurrLen Length of the current string.
  /// \param[out] BestLen Length of the most beneficial substring.
  /// \param[out] MaxBenefit Benefit of the most beneficial substring.
  /// \param[out] BestStartIdx Start index of the most beneficial substring.
  /// \param BenefitFn The function the query should return a maximum string
  /// for.
  void findBest(SuffixTreeNode &CurrNode, size_t CurrLen, size_t &BestLen,
                size_t &MaxBenefit, size_t &BestStartIdx,
                const std::function<unsigned(SuffixTreeNode &, size_t CurrLen)>
                &BenefitFn) {

    if (!CurrNode.IsInTree)
      return;

    // Can we traverse further down the tree?
    if (!CurrNode.isLeaf()) {
      // If yes, continue the traversal.
      for (auto &ChildPair : CurrNode.Children) {
        if (ChildPair.second && ChildPair.second->IsInTree)
          findBest(*ChildPair.second, CurrLen + ChildPair.second->size(),
                   BestLen, MaxBenefit, BestStartIdx, BenefitFn);
      }
    } else {
      // We hit a leaf.
      size_t StringLen = CurrLen - CurrNode.size();
      unsigned Benefit = BenefitFn(CurrNode, StringLen);

      // Did we do better than in the last step?
      if (Benefit <= MaxBenefit)
        return;

      // We did better, so update the best string.
      MaxBenefit = Benefit;
      BestStartIdx = CurrNode.SuffixIdx;
      BestLen = StringLen;
    }
  }

public:

  /// \brief Return a substring of the tree with maximum benefit if such a
  /// substring exists.
  ///
  /// Clears the input vector and fills it with a maximum substring or empty.
  ///
  /// \param[in,out] Best The most beneficial substring in the tree. Empty
  /// if it does not exist.
  /// \param BenefitFn The function the query should return a maximum string
  /// for.
  void bestRepeatedSubstring(std::vector<unsigned> &Best,
                 const std::function<unsigned(SuffixTreeNode &, size_t CurrLen)>
                 &BenefitFn) {
    Best.clear();
    size_t Length = 0;   // Becomes the length of the best substring.
    size_t Benefit = 0;  // Becomes the benefit of the best substring.
    size_t StartIdx = 0; // Becomes the start index of the best substring.
    findBest(*Root, 0, Length, Benefit, StartIdx, BenefitFn);

    for (size_t Idx = 0; Idx < Length; Idx++)
      Best.push_back(Str[Idx + StartIdx]);
  }

  /// Perform a depth-first search for \p QueryString on the suffix tree.
  ///
  /// \param QueryString The string to search for.
  /// \param CurrIdx The current index in \p QueryString that is being matched
  /// against.
  /// \param CurrNode The suffix tree node being searched in.
  ///
  /// \returns A \p SuffixTreeNode that \p QueryString appears in if such a
  /// node exists, and \p nullptr otherwise.
  SuffixTreeNode *findString(const std::vector<unsigned> &QueryString,
                             size_t &CurrIdx, SuffixTreeNode *CurrNode) {

    // The search ended at a nonexistent or pruned node. Quit.
    if (!CurrNode || !CurrNode->IsInTree)
      return nullptr;

    unsigned Edge = QueryString[CurrIdx]; // The edge we want to move on.
    SuffixTreeNode *NextNode = CurrNode->Children[Edge]; // Next node in query.

    if (CurrNode->isRoot()) {
      // If we're at the root we have to check if there's a child, and move to
      // that child. Don't consume the character since \p Root represents the
      // empty string.
      if (NextNode && NextNode->IsInTree)
        return findString(QueryString, CurrIdx, NextNode);
      return nullptr;
    }

    size_t StrIdx = CurrNode->StartIdx;
    size_t MaxIdx = QueryString.size();
    bool ContinueSearching = false;

    // Match as far as possible into the string. If there's a mismatch, quit.
    for (; CurrIdx < MaxIdx; CurrIdx++, StrIdx++) {
      Edge = QueryString[CurrIdx];

      // We matched perfectly, but still have a remainder to search.
      if (StrIdx > *(CurrNode->EndIdx)) {
        ContinueSearching = true;
        break;
      }

      if (Edge != Str[StrIdx])
        return nullptr;
    }

    NextNode = CurrNode->Children[Edge];

    // Move to the node which matches what we're looking for and continue
    // searching.
    if (ContinueSearching)
      return findString(QueryString, CurrIdx, NextNode);

    // We matched perfectly so we're done.
    return CurrNode;
  }

  /// \brief Remove a node from a tree and all nodes representing proper
  /// suffixes of that node's string.
  ///
  /// This is used in the outlining algorithm to reduce the number of
  /// overlapping candidates
  ///
  /// \param N The suffix tree node to start pruning from.
  /// \param Len The length of the string to be pruned.
  ///
  /// \returns True if this candidate didn't overlap with a previously chosen
  /// candidate.
  bool prune(SuffixTreeNode *N, size_t Len) {

    bool NoOverlap = true;
    std::vector<unsigned> IndicesToPrune;

    // Look at each of N's children.
    for (auto &ChildPair : N->Children) {
      SuffixTreeNode *M = ChildPair.second;

      // Is this a leaf child?
      if (M && M->IsInTree && M->isLeaf()) {
        // Save each leaf child's suffix indices and remove them from the tree.
        IndicesToPrune.push_back(M->SuffixIdx);
        M->IsInTree = false;
      }
    }

    // Remove each suffix we have to prune from the tree. Each of these will be
    // I + some offset for I in IndicesToPrune and some offset < Len.
    unsigned Offset = 1;
    for (unsigned CurrentSuffix = 1; CurrentSuffix < Len; CurrentSuffix++) {
      for (unsigned I : IndicesToPrune) {

        unsigned PruneIdx = I + Offset;

        // Is this index actually in the string?
        if (PruneIdx < LeafVector.size()) {
          // If yes, we have to try and prune it.
          // Was the current leaf already pruned by another candidate?
          if (LeafVector[PruneIdx]->IsInTree) {
            // If not, prune it.
            LeafVector[PruneIdx]->IsInTree = false;
          } else {
            // If yes, signify that we've found an overlap, but keep pruning.
            NoOverlap = false;
          }

          // Update the parent of the current leaf's occurrence count.
          SuffixTreeNode *Parent = LeafVector[PruneIdx]->Parent;

          // Is the parent still in the tree?
          if (Parent->OccurrenceCount > 0) {
            Parent->OccurrenceCount--;
            Parent->IsInTree = (Parent->OccurrenceCount > 1);
          }
        }
      }

      // Move to the next character in the string.
      Offset++;
    }

    // We know we can never outline anything which starts one index back from
    // the indices we want to outline. This is because our minimum outlining
    // length is always 2.
    for (unsigned I : IndicesToPrune) {
      if (I > 0) {

        unsigned PruneIdx = I-1;
        SuffixTreeNode *Parent = LeafVector[PruneIdx]->Parent;

        // Was the leaf one index back from I already pruned?
        if (LeafVector[PruneIdx]->IsInTree) {
          // If not, prune it.
          LeafVector[PruneIdx]->IsInTree = false;
        } else {
          // If yes, signify that we've found an overlap, but keep pruning.
          NoOverlap = false;
        }

        // Update the parent of the current leaf's occurrence count.
        if (Parent->OccurrenceCount > 0) {
          Parent->OccurrenceCount--;
          Parent->IsInTree = (Parent->OccurrenceCount > 1);
        }
      }
    }

    // Finally, remove N from the tree and set its occurrence count to 0.
    N->IsInTree = false;
    N->OccurrenceCount = 0;

    return NoOverlap;
  }

  /// \brief Find each occurrence of of a string in \p QueryString and prune
  /// their nodes.
  ///
  /// \param QueryString The string to search for.
  /// \param[out] Occurrences The start indices of each occurrence.
  ///
  /// \returns Whether or not the occurrence overlaps with a previous candidate.
  bool findOccurrencesAndPrune(const std::vector<unsigned> &QueryString,
                               std::vector<size_t> &Occurrences) {
    size_t Dummy = 0;
    SuffixTreeNode *N = findString(QueryString, Dummy, Root);

    if (!N || !N->IsInTree)
      return false;

    // If this is an internal node, occurrences are the number of leaf children
    // of the node.
    for (auto &ChildPair : N->Children) {
      SuffixTreeNode *M = ChildPair.second;

      // Is it a leaf? If so, we have an occurrence.
      if (M && M->IsInTree && M->isLeaf())
        Occurrences.push_back(M->SuffixIdx);
    }

    // If we're in a leaf, then this node is the only occurrence.
    if (N->isLeaf())
      Occurrences.push_back(N->SuffixIdx);

    return prune(N, QueryString.size());
  }

  /// Construct a suffix tree from a sequence of unsigned integers.
  ///
  /// \param Str The string to construct the suffix tree for.
  SuffixTree(const std::vector<unsigned> &Str) : Str(Str) {
    Root = insertInternalNode(nullptr, EmptyIdx, EmptyIdx, 0);
    Root->IsInTree = true;
    Active.Node = Root;
    LeafVector = std::vector<SuffixTreeNode*>(Str.size());

    // Keep track of the number of suffixes we have to add of the current
    // prefix.
    size_t SuffixesToAdd = 0;
    Active.Node = Root;

    // Construct the suffix tree iteratively on each prefix of the string.
    // PfxEndIdx is the end index of the current prefix.
    // End is one past the last element in the string.
    for (size_t PfxEndIdx = 0, End = Str.size(); PfxEndIdx < End; PfxEndIdx++) {
      SuffixesToAdd++;
      LeafEndIdx = PfxEndIdx; // Extend each of the leaves.
      SuffixesToAdd = extend(PfxEndIdx, SuffixesToAdd);
    }

    // Set the suffix indices of each leaf.
    assert(Root && "Root node can't be nullptr!");
    setSuffixIndices(*Root, 0);
  }
};

/// \brief An individual sequence of instructions to be replaced with a call to
/// an outlined function.
struct Candidate {

  /// Set to false if the candidate overlapped with another candidate.
  bool InCandidateList = true;

  /// The start index of this \p Candidate.
  size_t StartIdx;

  /// The number of instructions in this \p Candidate.
  size_t Len;

  /// The index of this \p Candidate's \p OutlinedFunction in the list of
  /// \p OutlinedFunctions.
  size_t FunctionIdx;

  Candidate(size_t StartIdx, size_t Len, size_t FunctionIdx)
      : StartIdx(StartIdx), Len(Len), FunctionIdx(FunctionIdx) {}

  Candidate() {}

  /// \brief Used to ensure that \p Candidates are outlined in an order that
  /// preserves the start and end indices of other \p Candidates.
  bool operator<(const Candidate &RHS) const { return StartIdx > RHS.StartIdx; }
};

/// \brief The information necessary to create an outlined function for some
/// class of candidate.
struct OutlinedFunction {

  /// The actual outlined function created.
  /// This is initialized after we go through and create the actual function.
  MachineFunction *MF = nullptr;

  /// A number assigned to this function which appears at the end of its name.
  size_t Name;

  /// The number of times that this function has appeared.
  size_t OccurrenceCount = 0;

  /// \brief The sequence of integers corresponding to the instructions in this
  /// function.
  std::vector<unsigned> Sequence;

  /// The number of instructions this function would save.
  unsigned Benefit = 0;

  OutlinedFunction(size_t Name, size_t OccurrenceCount,
                   const std::vector<unsigned> &Sequence,
                   unsigned Benefit)
      : Name(Name), OccurrenceCount(OccurrenceCount), Sequence(Sequence),
        Benefit(Benefit)
        {}
};

/// \brief Maps \p MachineInstrs to unsigned integers and stores the mappings.
struct InstructionMapper {

  /// \brief The next available integer to assign to a \p MachineInstr that
  /// cannot be outlined.
  ///
  /// Set to -3 for compatability with \p DenseMapInfo<unsigned>.
  unsigned IllegalInstrNumber = -3;

  /// \brief The next available integer to assign to a \p MachineInstr that can
  /// be outlined.
  unsigned LegalInstrNumber = 0;

  /// Correspondence from \p MachineInstrs to unsigned integers.
  DenseMap<MachineInstr *, unsigned, MachineInstrExpressionTrait>
      InstructionIntegerMap;

  /// Corresponcence from unsigned integers to \p MachineInstrs.
  /// Inverse of \p InstructionIntegerMap.
  DenseMap<unsigned, MachineInstr *> IntegerInstructionMap;

  /// The vector of unsigned integers that the module is mapped to.
  std::vector<unsigned> UnsignedVec;

  /// \brief Stores the location of the instruction associated with the integer
  /// at index i in \p UnsignedVec for each index i.
  std::vector<MachineBasicBlock::iterator> InstrList;

  /// \brief Maps \p *It to a legal integer.
  ///
  /// Updates \p InstrList, \p UnsignedVec, \p InstructionIntegerMap,
  /// \p IntegerInstructionMap, and \p LegalInstrNumber.
  ///
  /// \returns The integer that \p *It was mapped to.
  unsigned mapToLegalUnsigned(MachineBasicBlock::iterator &It) {

    // Get the integer for this instruction or give it the current
    // LegalInstrNumber.
    InstrList.push_back(It);
    MachineInstr &MI = *It;
    bool WasInserted;
    DenseMap<MachineInstr *, unsigned, MachineInstrExpressionTrait>::iterator
    ResultIt;
    std::tie(ResultIt, WasInserted) =
    InstructionIntegerMap.insert(std::make_pair(&MI, LegalInstrNumber));
    unsigned MINumber = ResultIt->second;

    // There was an insertion.
    if (WasInserted) {
      LegalInstrNumber++;
      IntegerInstructionMap.insert(std::make_pair(MINumber, &MI));
    }

    UnsignedVec.push_back(MINumber);

    // Make sure we don't overflow or use any integers reserved by the DenseMap.
    if (LegalInstrNumber >= IllegalInstrNumber)
      report_fatal_error("Instruction mapping overflow!");

    assert(LegalInstrNumber != DenseMapInfo<unsigned>::getEmptyKey()
          && "Tried to assign DenseMap tombstone or empty key to instruction.");
    assert(LegalInstrNumber != DenseMapInfo<unsigned>::getTombstoneKey()
          && "Tried to assign DenseMap tombstone or empty key to instruction.");

    return MINumber;
  }

  /// Maps \p *It to an illegal integer.
  ///
  /// Updates \p InstrList, \p UnsignedVec, and \p IllegalInstrNumber.
  ///
  /// \returns The integer that \p *It was mapped to.
  unsigned mapToIllegalUnsigned(MachineBasicBlock::iterator &It) {
    unsigned MINumber = IllegalInstrNumber;

    InstrList.push_back(It);
    UnsignedVec.push_back(IllegalInstrNumber);
    IllegalInstrNumber--;

    assert(LegalInstrNumber < IllegalInstrNumber &&
           "Instruction mapping overflow!");

    assert(IllegalInstrNumber !=
      DenseMapInfo<unsigned>::getEmptyKey() &&
      "IllegalInstrNumber cannot be DenseMap tombstone or empty key!");

    assert(IllegalInstrNumber !=
      DenseMapInfo<unsigned>::getTombstoneKey() &&
      "IllegalInstrNumber cannot be DenseMap tombstone or empty key!");

    return MINumber;
  }

  /// \brief Transforms a \p MachineBasicBlock into a \p vector of \p unsigneds
  /// and appends it to \p UnsignedVec and \p InstrList.
  ///
  /// Two instructions are assigned the same integer if they are identical.
  /// If an instruction is deemed unsafe to outline, then it will be assigned an
  /// unique integer. The resulting mapping is placed into a suffix tree and
  /// queried for candidates.
  ///
  /// \param MBB The \p MachineBasicBlock to be translated into integers.
  /// \param TRI \p TargetRegisterInfo for the module.
  /// \param TII \p TargetInstrInfo for the module.
  void convertToUnsignedVec(MachineBasicBlock &MBB,
                            const TargetRegisterInfo &TRI,
                            const TargetInstrInfo &TII) {
    for (MachineBasicBlock::iterator It = MBB.begin(), Et = MBB.end(); It != Et;
         It++) {

      // Keep track of where this instruction is in the module.
      switch(TII.getOutliningType(*It)) {
        case TargetInstrInfo::MachineOutlinerInstrType::Illegal:
          mapToIllegalUnsigned(It);
          break;

        case TargetInstrInfo::MachineOutlinerInstrType::Legal:
          mapToLegalUnsigned(It);
          break;

        case TargetInstrInfo::MachineOutlinerInstrType::Invisible:
          break;
      }
    }

    // After we're done every insertion, uniquely terminate this part of the
    // "string". This makes sure we won't match across basic block or function
    // boundaries since the "end" is encoded uniquely and thus appears in no
    // repeated substring.
    InstrList.push_back(MBB.end());
    UnsignedVec.push_back(IllegalInstrNumber);
    IllegalInstrNumber--;
  }

  InstructionMapper() {
    // Make sure that the implementation of DenseMapInfo<unsigned> hasn't
    // changed.
    assert(DenseMapInfo<unsigned>::getEmptyKey() == (unsigned)-1 &&
                "DenseMapInfo<unsigned>'s empty key isn't -1!");
    assert(DenseMapInfo<unsigned>::getTombstoneKey() == (unsigned)-2 &&
                "DenseMapInfo<unsigned>'s tombstone key isn't -2!");
  }
};

/// \brief An interprocedural pass which finds repeated sequences of
/// instructions and replaces them with calls to functions.
///
/// Each instruction is mapped to an unsigned integer and placed in a string.
/// The resulting mapping is then placed in a \p SuffixTree. The \p SuffixTree
/// is then repeatedly queried for repeated sequences of instructions. Each
/// non-overlapping repeated sequence is then placed in its own
/// \p MachineFunction and each instance is then replaced with a call to that
/// function.
struct MachineOutliner : public ModulePass {

  static char ID;

  StringRef getPassName() const override { return "Machine Outliner"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfo>();
    AU.addPreserved<MachineModuleInfo>();
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }

  MachineOutliner() : ModulePass(ID) {
    initializeMachineOutlinerPass(*PassRegistry::getPassRegistry());
  }

  /// \brief Replace the sequences of instructions represented by the
  /// \p Candidates in \p CandidateList with calls to \p MachineFunctions
  /// described in \p FunctionList.
  ///
  /// \param M The module we are outlining from.
  /// \param CandidateList A list of candidates to be outlined.
  /// \param FunctionList A list of functions to be inserted into the module.
  /// \param Mapper Contains the instruction mappings for the module.
  bool outline(Module &M, const ArrayRef<Candidate> &CandidateList,
               std::vector<OutlinedFunction> &FunctionList,
               InstructionMapper &Mapper);

  /// Creates a function for \p OF and inserts it into the module.
  MachineFunction *createOutlinedFunction(Module &M, const OutlinedFunction &OF,
                                          InstructionMapper &Mapper);

  /// Find potential outlining candidates and store them in \p CandidateList.
  ///
  /// For each type of potential candidate, also build an \p OutlinedFunction
  /// struct containing the information to build the function for that
  /// candidate.
  ///
  /// \param[out] CandidateList Filled with outlining candidates for the module.
  /// \param[out] FunctionList Filled with functions corresponding to each type
  /// of \p Candidate.
  /// \param ST The suffix tree for the module.
  /// \param TII TargetInstrInfo for the module.
  ///
  /// \returns The length of the longest candidate found. 0 if there are none.
  unsigned buildCandidateList(std::vector<Candidate> &CandidateList,
                              std::vector<OutlinedFunction> &FunctionList,
                              SuffixTree &ST, const TargetInstrInfo &TII);

  /// \brief Remove any overlapping candidates that weren't handled by the
  /// suffix tree's pruning method.
  ///
  /// Pruning from the suffix tree doesn't necessarily remove all overlaps.
  /// If a short candidate is chosen for outlining, then a longer candidate
  /// which has that short candidate as a suffix is chosen, the tree's pruning
  /// method will not find it. Thus, we need to prune before outlining as well.
  ///
  /// \param[in,out] CandidateList A list of outlining candidates.
  /// \param[in,out] FunctionList A list of functions to be outlined.
  /// \param MaxCandidateLen The length of the longest candidate.
  /// \param TII TargetInstrInfo for the module.
  void pruneOverlaps(std::vector<Candidate> &CandidateList,
                     std::vector<OutlinedFunction> &FunctionList,
                     unsigned MaxCandidateLen,
                     const TargetInstrInfo &TII);

  /// Construct a suffix tree on the instructions in \p M and outline repeated
  /// strings from that tree.
  bool runOnModule(Module &M) override;
};

} // Anonymous namespace.

char MachineOutliner::ID = 0;

namespace llvm {
ModulePass *createMachineOutlinerPass() { return new MachineOutliner(); }
}

INITIALIZE_PASS(MachineOutliner, "machine-outliner",
                "Machine Function Outliner", false, false)

void MachineOutliner::pruneOverlaps(std::vector<Candidate> &CandidateList,
                                    std::vector<OutlinedFunction> &FunctionList,
                                    unsigned MaxCandidateLen,
                                    const TargetInstrInfo &TII) {

  // Check for overlaps in the range. This is O(n^2) worst case, but we can
  // alleviate that somewhat by bounding our search space using the start
  // index of our first candidate and the maximum distance an overlapping
  // candidate could have from the first candidate.
  for (auto It = CandidateList.begin(), Et = CandidateList.end(); It != Et;
       It++) {
    Candidate &C1 = *It;
    OutlinedFunction &F1 = FunctionList[C1.FunctionIdx];

    // If we removed this candidate, skip it.
    if (!C1.InCandidateList)
      continue;

    // If the candidate's function isn't good to outline anymore, then
    // remove the candidate and skip it.
    if (F1.OccurrenceCount < 2 || F1.Benefit < 1) {
      C1.InCandidateList = false;
      continue;
    }

    // The minimum start index of any candidate that could overlap with this
    // one.
    unsigned FarthestPossibleIdx = 0;

    // Either the index is 0, or it's at most MaxCandidateLen indices away.
    if (C1.StartIdx > MaxCandidateLen)
      FarthestPossibleIdx = C1.StartIdx - MaxCandidateLen;

    // Compare against the other candidates in the list.
    // This is at most MaxCandidateLen/2 other candidates.
    // This is because each candidate has to be at least 2 indices away.
    // = O(n * MaxCandidateLen/2) comparisons
    //
    // On average, the maximum length of a candidate is quite small; a fraction
    // of the total module length in terms of instructions. If the maximum
    // candidate length is large, then there are fewer possible candidates to
    // compare against in the first place.
    for (auto Sit = It + 1; Sit != Et; Sit++) {
      Candidate &C2 = *Sit;
      OutlinedFunction &F2 = FunctionList[C2.FunctionIdx];

      // Is this candidate too far away to overlap?
      // NOTE: This will be true in
      //    O(max(FarthestPossibleIdx/2, #Candidates remaining)) steps
      // for every candidate.
      if (C2.StartIdx < FarthestPossibleIdx)
        break;

      // Did we already remove this candidate in a previous step?
      if (!C2.InCandidateList)
        continue;

      // Is the function beneficial to outline?
      if (F2.OccurrenceCount < 2 || F2.Benefit < 1) {
        // If not, remove this candidate and move to the next one.
        C2.InCandidateList = false;
        continue;
      }

      size_t C2End = C2.StartIdx + C2.Len - 1;

      // Do C1 and C2 overlap?
      //
      // Not overlapping:
      // High indices... [C1End ... C1Start][C2End ... C2Start] ...Low indices
      //
      // We sorted our candidate list so C2Start <= C1Start. We know that
      // C2End > C2Start since each candidate has length >= 2. Therefore, all we
      // have to check is C2End < C2Start to see if we overlap.
      if (C2End < C1.StartIdx)
        continue;

      // C2 overlaps with C1. Because we pruned the tree already, the only way
      // this can happen is if C1 is a proper suffix of C2. Thus, we must have
      // found C1 first during our query, so it must have benefit greater or
      // equal to C2. Greedily pick C1 as the candidate to keep and toss out C2.
      DEBUG (
            size_t C1End = C1.StartIdx + C1.Len - 1;
            dbgs() << "- Found an overlap to purge.\n";
            dbgs() << "--- C1 :[" << C1.StartIdx << ", " << C1End << "]\n";
            dbgs() << "--- C2 :[" << C2.StartIdx << ", " << C2End << "]\n";
            );

      // Update the function's occurrence count and benefit to reflec that C2
      // is being removed.
      F2.OccurrenceCount--;
      F2.Benefit = TII.getOutliningBenefit(F2.Sequence.size(),
                                           F2.OccurrenceCount
                                           );

      // Mark C2 as not in the list.
      C2.InCandidateList = false;

      DEBUG (
            dbgs() << "- Removed C2. \n";
            dbgs() << "--- Num fns left for C2: " << F2.OccurrenceCount << "\n";
            dbgs() << "--- C2's benefit: " << F2.Benefit << "\n";
            );
    }
  }
}

unsigned
MachineOutliner::buildCandidateList(std::vector<Candidate> &CandidateList,
                                    std::vector<OutlinedFunction> &FunctionList,
                                    SuffixTree &ST,
                                    const TargetInstrInfo &TII) {

  std::vector<unsigned> CandidateSequence; // Current outlining candidate.
  unsigned MaxCandidateLen = 0; // Length of the longest candidate.

  // Function for maximizing query in the suffix tree.
  // This allows us to define more fine-grained types of things to outline in
  // the target without putting target-specific info in the suffix tree.
  auto BenefitFn = [&TII](const SuffixTreeNode &Curr, size_t StringLen) {

    // Any leaf whose parent is the root only has one occurrence.
    if (Curr.Parent->isRoot())
      return 0u;

    // Anything with length < 2 will never be beneficial on any target.
    if (StringLen < 2)
      return 0u;

    size_t Occurrences = Curr.Parent->OccurrenceCount;

    // Anything with fewer than 2 occurrences will never be beneficial on any
    // target.
    if (Occurrences < 2)
      return 0u;

    return TII.getOutliningBenefit(StringLen, Occurrences);
  };

  // Repeatedly query the suffix tree for the substring that maximizes
  // BenefitFn. Find the occurrences of that string, prune the tree, and store
  // each occurrence as a candidate.
  for (ST.bestRepeatedSubstring(CandidateSequence, BenefitFn);
       CandidateSequence.size() > 1;
       ST.bestRepeatedSubstring(CandidateSequence, BenefitFn)) {

    std::vector<size_t> Occurrences;

    bool GotNonOverlappingCandidate =
        ST.findOccurrencesAndPrune(CandidateSequence, Occurrences);

    // Is the candidate we found known to overlap with something we already
    // outlined?
    if (!GotNonOverlappingCandidate)
      continue;

    // Is this candidate the longest so far?
    if (CandidateSequence.size() > MaxCandidateLen)
      MaxCandidateLen = CandidateSequence.size();

    // Keep track of the benefit of outlining this candidate in its
    // OutlinedFunction.
    unsigned FnBenefit = TII.getOutliningBenefit(CandidateSequence.size(),
                                                 Occurrences.size()
                                                 );

    assert(FnBenefit > 0 && "Function cannot be unbeneficial!");

    // Save an OutlinedFunction for this candidate.
    FunctionList.emplace_back(
        FunctionList.size(), // Number of this function.
        Occurrences.size(),  // Number of occurrences.
        CandidateSequence,   // Sequence to outline.
        FnBenefit            // Instructions saved by outlining this function.
        );

    // Save each of the occurrences of the candidate so we can outline them.
    for (size_t &Occ : Occurrences)
      CandidateList.emplace_back(
          Occ,                      // Starting idx in that MBB.
          CandidateSequence.size(), // Candidate length.
          FunctionList.size() - 1   // Idx of the corresponding function.
          );

    FunctionsCreated++;
  }

  // Sort the candidates in decending order. This will simplify the outlining
  // process when we have to remove the candidates from the mapping by
  // allowing us to cut them out without keeping track of an offset.
  std::stable_sort(CandidateList.begin(), CandidateList.end());

  return MaxCandidateLen;
}

MachineFunction *
MachineOutliner::createOutlinedFunction(Module &M, const OutlinedFunction &OF,
  InstructionMapper &Mapper) {

  // Create the function name. This should be unique. For now, just hash the
  // module name and include it in the function name plus the number of this
  // function.
  std::ostringstream NameStream;
  NameStream << "OUTLINED_FUNCTION" << "_" << OF.Name;

  // Create the function using an IR-level function.
  LLVMContext &C = M.getContext();
  Function *F = dyn_cast<Function>(
      M.getOrInsertFunction(NameStream.str(), Type::getVoidTy(C), nullptr));
  assert(F && "Function was null!");

  // NOTE: If this is linkonceodr, then we can take advantage of linker deduping
  // which gives us better results when we outline from linkonceodr functions.
  F->setLinkage(GlobalValue::PrivateLinkage);
  F->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  BasicBlock *EntryBB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(EntryBB);
  Builder.CreateRetVoid();

  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfo>();
  MachineFunction &MF = MMI.getMachineFunction(*F);
  MachineBasicBlock &MBB = *MF.CreateMachineBasicBlock();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetInstrInfo &TII = *STI.getInstrInfo();

  // Insert the new function into the module.
  MF.insert(MF.begin(), &MBB);

  TII.insertOutlinerPrologue(MBB, MF);

  // Copy over the instructions for the function using the integer mappings in
  // its sequence.
  for (unsigned Str : OF.Sequence) {
    MachineInstr *NewMI =
        MF.CloneMachineInstr(Mapper.IntegerInstructionMap.find(Str)->second);
    NewMI->dropMemRefs();

    // Don't keep debug information for outlined instructions.
    // FIXME: This means outlined functions are currently undebuggable.
    NewMI->setDebugLoc(DebugLoc());
    MBB.insert(MBB.end(), NewMI);
  }

  TII.insertOutlinerEpilogue(MBB, MF);

  return &MF;
}

bool MachineOutliner::outline(Module &M,
                              const ArrayRef<Candidate> &CandidateList,
                              std::vector<OutlinedFunction> &FunctionList,
                              InstructionMapper &Mapper) {

  bool OutlinedSomething = false;

  // Replace the candidates with calls to their respective outlined functions.
  for (const Candidate &C : CandidateList) {

    // Was the candidate removed during pruneOverlaps?
    if (!C.InCandidateList)
      continue;

    // If not, then look at its OutlinedFunction.
    OutlinedFunction &OF = FunctionList[C.FunctionIdx];

    // Was its OutlinedFunction made unbeneficial during pruneOverlaps?
    if (OF.OccurrenceCount < 2 || OF.Benefit < 1)
      continue;

    // If not, then outline it.
    assert(C.StartIdx < Mapper.InstrList.size() && "Candidate out of bounds!");
    MachineBasicBlock *MBB = (*Mapper.InstrList[C.StartIdx]).getParent();
    MachineBasicBlock::iterator StartIt = Mapper.InstrList[C.StartIdx];
    unsigned EndIdx = C.StartIdx + C.Len - 1;

    assert(EndIdx < Mapper.InstrList.size() && "Candidate out of bounds!");
    MachineBasicBlock::iterator EndIt = Mapper.InstrList[EndIdx];
    assert(EndIt != MBB->end() && "EndIt out of bounds!");

    EndIt++; // Erase needs one past the end index.

    // Does this candidate have a function yet?
    if (!OF.MF)
      OF.MF = createOutlinedFunction(M, OF, Mapper);

    MachineFunction *MF = OF.MF;
    const TargetSubtargetInfo &STI = MF->getSubtarget();
    const TargetInstrInfo &TII = *STI.getInstrInfo();

    // Insert a call to the new function and erase the old sequence.
    TII.insertOutlinedCall(M, *MBB, StartIt, *MF);
    StartIt = Mapper.InstrList[C.StartIdx];
    MBB->erase(StartIt, EndIt);

    OutlinedSomething = true;

    // Statistics.
    NumOutlined++;
  }

  DEBUG (
    dbgs() << "OutlinedSomething = " << OutlinedSomething << "\n";
  );

  return OutlinedSomething;
}

bool MachineOutliner::runOnModule(Module &M) {

  // Is there anything in the module at all?
  if (M.empty())
    return false;

  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfo>();
  const TargetSubtargetInfo &STI = MMI.getMachineFunction(*M.begin())
                                      .getSubtarget();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  const TargetInstrInfo *TII = STI.getInstrInfo();

  InstructionMapper Mapper;

  // Build instruction mappings for each function in the module.
  for (Function &F : M) {
    MachineFunction &MF = MMI.getMachineFunction(F);

    // Is the function empty? Safe to outline from?
    if (F.empty() || !TII->isFunctionSafeToOutlineFrom(MF))
      continue;

    // If it is, look at each MachineBasicBlock in the function.
    for (MachineBasicBlock &MBB : MF) {

      // Is there anything in MBB?
      if (MBB.empty())
        continue;

      // If yes, map it.
      Mapper.convertToUnsignedVec(MBB, *TRI, *TII);
    }
  }

  // Construct a suffix tree, use it to find candidates, and then outline them.
  SuffixTree ST(Mapper.UnsignedVec);
  std::vector<Candidate> CandidateList;
  std::vector<OutlinedFunction> FunctionList;

  unsigned MaxCandidateLen =
      buildCandidateList(CandidateList, FunctionList, ST, *TII);

  pruneOverlaps(CandidateList, FunctionList, MaxCandidateLen, *TII);
  return outline(M, CandidateList, FunctionList, Mapper);
}
