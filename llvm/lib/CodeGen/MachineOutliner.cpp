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
/// The MachineOutliner communicates with a given target using hooks defined in
/// TargetInstrInfo.h. The target supplies the outliner with information on how
/// a specific sequence of instructions should be outlined. This information
/// is used to deduce the number of instructions necessary to
///
/// * Create an outlined function
/// * Call that outlined function
///
/// Targets must implement
///   * getOutliningCandidateInfo
///   * insertOutlinerEpilogue
///   * insertOutlinedCall
///   * insertOutlinerPrologue
///   * isFunctionSafeToOutlineFrom
///
/// in order to make use of the MachineOutliner.
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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#define DEBUG_TYPE "machine-outliner"

using namespace llvm;
using namespace ore;

STATISTIC(NumOutlined, "Number of candidates outlined");
STATISTIC(FunctionsCreated, "Number of functions created");

// Set to true if the user wants the outliner to run on linkonceodr linkage
// functions. This is false by default because the linker can dedupe linkonceodr
// functions. Since the outliner is confined to a single module (modulo LTO),
// this is off by default. It should, however, be the default behaviour in
// LTO.
static cl::opt<bool> EnableLinkOnceODROutlining(
    "enable-linkonceodr-outlining",
    cl::Hidden,
    cl::desc("Enable the machine outliner on linkonceodr functions"),
    cl::init(false));

namespace {

/// An individual sequence of instructions to be replaced with a call to
/// an outlined function.
struct Candidate {
private:
  /// The start index of this \p Candidate in the instruction list.
  unsigned StartIdx;

  /// The number of instructions in this \p Candidate.
  unsigned Len;

  /// The MachineFunction containing this \p Candidate.
  MachineFunction *MF = nullptr;

public:
  /// Set to false if the candidate overlapped with another candidate.
  bool InCandidateList = true;

  /// The index of this \p Candidate's \p OutlinedFunction in the list of
  /// \p OutlinedFunctions.
  unsigned FunctionIdx;

  /// Contains all target-specific information for this \p Candidate.
  TargetInstrInfo::MachineOutlinerInfo MInfo;

  /// If there is a DISubprogram associated with the function that this
  /// Candidate lives in, return it.
  DISubprogram *getSubprogramOrNull() const {
    assert(MF && "Candidate has no MF!");
    if (DISubprogram *SP = MF->getFunction().getSubprogram())
      return SP;
    return nullptr;
  }

  /// Return the number of instructions in this Candidate.
  unsigned getLength() const { return Len; }

  /// Return the start index of this candidate.
  unsigned getStartIdx() const { return StartIdx; }

  // Return the end index of this candidate.
  unsigned getEndIdx() const { return StartIdx + Len - 1; }

  /// The number of instructions that would be saved by outlining every
  /// candidate of this type.
  ///
  /// This is a fixed value which is not updated during the candidate pruning
  /// process. It is only used for deciding which candidate to keep if two
  /// candidates overlap. The true benefit is stored in the OutlinedFunction
  /// for some given candidate.
  unsigned Benefit = 0;

  Candidate(unsigned StartIdx, unsigned Len, unsigned FunctionIdx,
            MachineFunction *MF)
      : StartIdx(StartIdx), Len(Len), MF(MF), FunctionIdx(FunctionIdx) {}

  Candidate() {}

  /// Used to ensure that \p Candidates are outlined in an order that
  /// preserves the start and end indices of other \p Candidates.
  bool operator<(const Candidate &RHS) const {
    return getStartIdx() > RHS.getStartIdx();
  }
};

/// The information necessary to create an outlined function for some
/// class of candidate.
struct OutlinedFunction {

private:
  /// The number of candidates for this \p OutlinedFunction.
  unsigned OccurrenceCount = 0;

public:
  std::vector<std::shared_ptr<Candidate>> Candidates;

  /// The actual outlined function created.
  /// This is initialized after we go through and create the actual function.
  MachineFunction *MF = nullptr;

  /// A number assigned to this function which appears at the end of its name.
  unsigned Name;

  /// The sequence of integers corresponding to the instructions in this
  /// function.
  std::vector<unsigned> Sequence;

  /// Contains all target-specific information for this \p OutlinedFunction.
  TargetInstrInfo::MachineOutlinerInfo MInfo;

  /// If there is a DISubprogram for any Candidate for this outlined function,
  /// then return it. Otherwise, return nullptr.
  DISubprogram *getSubprogramOrNull() const {
    for (const auto &C : Candidates)
      if (DISubprogram *SP = C->getSubprogramOrNull())
        return SP;
    return nullptr;
  }

  /// Return the number of candidates for this \p OutlinedFunction.
  unsigned getOccurrenceCount() { return OccurrenceCount; }

  /// Decrement the occurrence count of this OutlinedFunction and return the
  /// new count.
  unsigned decrement() {
    assert(OccurrenceCount > 0 && "Can't decrement an empty function!");
    OccurrenceCount--;
    return getOccurrenceCount();
  }

  /// Return the number of bytes it would take to outline this
  /// function.
  unsigned getOutliningCost() {
    return (OccurrenceCount * MInfo.CallOverhead) + MInfo.SequenceSize +
           MInfo.FrameOverhead;
  }

  /// Return the size in bytes of the unoutlined sequences.
  unsigned getNotOutlinedCost() {
    return OccurrenceCount * MInfo.SequenceSize;
  }

  /// Return the number of instructions that would be saved by outlining
  /// this function.
  unsigned getBenefit() {
    unsigned NotOutlinedCost = getNotOutlinedCost();
    unsigned OutlinedCost = getOutliningCost();
    return (NotOutlinedCost < OutlinedCost) ? 0
                                            : NotOutlinedCost - OutlinedCost;
  }

  OutlinedFunction(unsigned Name, unsigned OccurrenceCount,
                   const std::vector<unsigned> &Sequence,
                   TargetInstrInfo::MachineOutlinerInfo &MInfo)
      : OccurrenceCount(OccurrenceCount), Name(Name), Sequence(Sequence),
        MInfo(MInfo) {}
};

/// Represents an undefined index in the suffix tree.
const unsigned EmptyIdx = -1;

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
  unsigned StartIdx = EmptyIdx;

  /// The end index of this node's substring in the main string.
  ///
  /// Every leaf node must have its \p EndIdx incremented at the end of every
  /// step in the construction algorithm. To avoid having to update O(N)
  /// nodes individually at the end of every step, the end index is stored
  /// as a pointer.
  unsigned *EndIdx = nullptr;

  /// For leaves, the start index of the suffix represented by this node.
  ///
  /// For all other nodes, this is ignored.
  unsigned SuffixIdx = EmptyIdx;

  /// For internal nodes, a pointer to the internal node representing
  /// the same sequence with the first character chopped off.
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
  SuffixTreeNode *Link = nullptr;

  /// The parent of this node. Every node except for the root has a parent.
  SuffixTreeNode *Parent = nullptr;

  /// The number of times this node's string appears in the tree.
  ///
  /// This is equal to the number of leaf children of the string. It represents
  /// the number of suffixes that the node's string is a prefix of.
  unsigned OccurrenceCount = 0;

  /// The length of the string formed by concatenating the edge labels from the
  /// root to this node.
  unsigned ConcatLen = 0;

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

  SuffixTreeNode(unsigned StartIdx, unsigned *EndIdx, SuffixTreeNode *Link,
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
public:
  /// Stores each leaf node in the tree.
  ///
  /// This is used for finding outlining candidates.
  std::vector<SuffixTreeNode *> LeafVector;

  /// Each element is an integer representing an instruction in the module.
  ArrayRef<unsigned> Str;

private:
  /// Maintains each node in the tree.
  SpecificBumpPtrAllocator<SuffixTreeNode> NodeAllocator;

  /// The root of the suffix tree.
  ///
  /// The root represents the empty string. It is maintained by the
  /// \p NodeAllocator like every other node in the tree.
  SuffixTreeNode *Root = nullptr;

  /// Maintains the end indices of the internal nodes in the tree.
  ///
  /// Each internal node is guaranteed to never have its end index change
  /// during the construction algorithm; however, leaves must be updated at
  /// every step. Therefore, we need to store leaf end indices by reference
  /// to avoid updating O(N) leaves at every step of construction. Thus,
  /// every internal node must be allocated its own end index.
  BumpPtrAllocator InternalEndIdxAllocator;

  /// The end index of each leaf in the tree.
  unsigned LeafEndIdx = -1;

  /// Helper struct which keeps track of the next insertion point in
  /// Ukkonen's algorithm.
  struct ActiveState {
    /// The next node to insert at.
    SuffixTreeNode *Node;

    /// The index of the first character in the substring currently being added.
    unsigned Idx = EmptyIdx;

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
  SuffixTreeNode *insertLeaf(SuffixTreeNode &Parent, unsigned StartIdx,
                             unsigned Edge) {

    assert(StartIdx <= LeafEndIdx && "String can't start after it ends!");

    SuffixTreeNode *N = new (NodeAllocator.Allocate())
        SuffixTreeNode(StartIdx, &LeafEndIdx, nullptr, &Parent);
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
  SuffixTreeNode *insertInternalNode(SuffixTreeNode *Parent, unsigned StartIdx,
                                     unsigned EndIdx, unsigned Edge) {

    assert(StartIdx <= EndIdx && "String can't start after it ends!");
    assert(!(!Parent && StartIdx != EmptyIdx) &&
           "Non-root internal nodes must have parents!");

    unsigned *E = new (InternalEndIdxAllocator) unsigned(EndIdx);
    SuffixTreeNode *N = new (NodeAllocator.Allocate())
        SuffixTreeNode(StartIdx, E, Root, Parent);
    if (Parent)
      Parent->Children[Edge] = N;

    return N;
  }

  /// Set the suffix indices of the leaves to the start indices of their
  /// respective suffixes. Also stores each leaf in \p LeafVector at its
  /// respective suffix index.
  ///
  /// \param[in] CurrNode The node currently being visited.
  /// \param CurrIdx The current index of the string being visited.
  void setSuffixIndices(SuffixTreeNode &CurrNode, unsigned CurrIdx) {

    bool IsLeaf = CurrNode.Children.size() == 0 && !CurrNode.isRoot();

    // Store the length of the concatenation of all strings from the root to
    // this node.
    if (!CurrNode.isRoot()) {
      if (CurrNode.ConcatLen == 0)
        CurrNode.ConcatLen = CurrNode.size();

      if (CurrNode.Parent)
        CurrNode.ConcatLen += CurrNode.Parent->ConcatLen;
    }

    // Traverse the tree depth-first.
    for (auto &ChildPair : CurrNode.Children) {
      assert(ChildPair.second && "Node had a null child!");
      setSuffixIndices(*ChildPair.second, CurrIdx + ChildPair.second->size());
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
  unsigned extend(unsigned EndIdx, unsigned SuffixesToAdd) {
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

        unsigned SubstringLen = NextNode->size();

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
            insertInternalNode(Active.Node, NextNode->StartIdx,
                               NextNode->StartIdx + Active.Len - 1, FirstChar);

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

public:
  /// Construct a suffix tree from a sequence of unsigned integers.
  ///
  /// \param Str The string to construct the suffix tree for.
  SuffixTree(const std::vector<unsigned> &Str) : Str(Str) {
    Root = insertInternalNode(nullptr, EmptyIdx, EmptyIdx, 0);
    Root->IsInTree = true;
    Active.Node = Root;
    LeafVector = std::vector<SuffixTreeNode *>(Str.size());

    // Keep track of the number of suffixes we have to add of the current
    // prefix.
    unsigned SuffixesToAdd = 0;
    Active.Node = Root;

    // Construct the suffix tree iteratively on each prefix of the string.
    // PfxEndIdx is the end index of the current prefix.
    // End is one past the last element in the string.
    for (unsigned PfxEndIdx = 0, End = Str.size(); PfxEndIdx < End;
         PfxEndIdx++) {
      SuffixesToAdd++;
      LeafEndIdx = PfxEndIdx; // Extend each of the leaves.
      SuffixesToAdd = extend(PfxEndIdx, SuffixesToAdd);
    }

    // Set the suffix indices of each leaf.
    assert(Root && "Root node can't be nullptr!");
    setSuffixIndices(*Root, 0);
  }
};

/// Maps \p MachineInstrs to unsigned integers and stores the mappings.
struct InstructionMapper {

  /// The next available integer to assign to a \p MachineInstr that
  /// cannot be outlined.
  ///
  /// Set to -3 for compatability with \p DenseMapInfo<unsigned>.
  unsigned IllegalInstrNumber = -3;

  /// The next available integer to assign to a \p MachineInstr that can
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

  /// Stores the location of the instruction associated with the integer
  /// at index i in \p UnsignedVec for each index i.
  std::vector<MachineBasicBlock::iterator> InstrList;

  /// Maps \p *It to a legal integer.
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

    assert(LegalInstrNumber != DenseMapInfo<unsigned>::getEmptyKey() &&
           "Tried to assign DenseMap tombstone or empty key to instruction.");
    assert(LegalInstrNumber != DenseMapInfo<unsigned>::getTombstoneKey() &&
           "Tried to assign DenseMap tombstone or empty key to instruction.");

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

    assert(IllegalInstrNumber != DenseMapInfo<unsigned>::getEmptyKey() &&
           "IllegalInstrNumber cannot be DenseMap tombstone or empty key!");

    assert(IllegalInstrNumber != DenseMapInfo<unsigned>::getTombstoneKey() &&
           "IllegalInstrNumber cannot be DenseMap tombstone or empty key!");

    return MINumber;
  }

  /// Transforms a \p MachineBasicBlock into a \p vector of \p unsigneds
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
    unsigned Flags = TII.getMachineOutlinerMBBFlags(MBB);

    for (MachineBasicBlock::iterator It = MBB.begin(), Et = MBB.end(); It != Et;
         It++) {

      // Keep track of where this instruction is in the module.
      switch (TII.getOutliningType(It, Flags)) {
      case TargetInstrInfo::MachineOutlinerInstrType::Illegal:
        mapToIllegalUnsigned(It);
        break;

      case TargetInstrInfo::MachineOutlinerInstrType::Legal:
        mapToLegalUnsigned(It);
        break;

      case TargetInstrInfo::MachineOutlinerInstrType::LegalTerminator:
        mapToLegalUnsigned(It);
        InstrList.push_back(It);
        UnsignedVec.push_back(IllegalInstrNumber);
        IllegalInstrNumber--;
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

/// An interprocedural pass which finds repeated sequences of
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

  /// Set to true if the outliner should consider functions with
  /// linkonceodr linkage.
  bool OutlineFromLinkOnceODRs = false;

  // Collection of IR functions created by the outliner.
  std::vector<Function *> CreatedIRFunctions;

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

  /// Find all repeated substrings that satisfy the outlining cost model.
  ///
  /// If a substring appears at least twice, then it must be represented by
  /// an internal node which appears in at least two suffixes. Each suffix is
  /// represented by a leaf node. To do this, we visit each internal node in
  /// the tree, using the leaf children of each internal node. If an internal
  /// node represents a beneficial substring, then we use each of its leaf
  /// children to find the locations of its substring.
  ///
  /// \param ST A suffix tree to query.
  /// \param TII TargetInstrInfo for the target.
  /// \param Mapper Contains outlining mapping information.
  /// \param[out] CandidateList Filled with candidates representing each
  /// beneficial substring.
  /// \param[out] FunctionList Filled with a list of \p OutlinedFunctions each
  /// type of candidate.
  ///
  /// \returns The length of the longest candidate found.
  unsigned
  findCandidates(SuffixTree &ST, const TargetInstrInfo &TII,
                 InstructionMapper &Mapper,
                 std::vector<std::shared_ptr<Candidate>> &CandidateList,
                 std::vector<OutlinedFunction> &FunctionList);

  /// Replace the sequences of instructions represented by the
  /// \p Candidates in \p CandidateList with calls to \p MachineFunctions
  /// described in \p FunctionList.
  ///
  /// \param M The module we are outlining from.
  /// \param CandidateList A list of candidates to be outlined.
  /// \param FunctionList A list of functions to be inserted into the module.
  /// \param Mapper Contains the instruction mappings for the module.
  bool outline(Module &M,
               const ArrayRef<std::shared_ptr<Candidate>> &CandidateList,
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
  unsigned
  buildCandidateList(std::vector<std::shared_ptr<Candidate>> &CandidateList,
                     std::vector<OutlinedFunction> &FunctionList,
                     SuffixTree &ST, InstructionMapper &Mapper,
                     const TargetInstrInfo &TII);

  /// Helper function for pruneOverlaps.
  /// Removes \p C from the candidate list, and updates its \p OutlinedFunction.
  void prune(Candidate &C, std::vector<OutlinedFunction> &FunctionList);

  /// Remove any overlapping candidates that weren't handled by the
  /// suffix tree's pruning method.
  ///
  /// Pruning from the suffix tree doesn't necessarily remove all overlaps.
  /// If a short candidate is chosen for outlining, then a longer candidate
  /// which has that short candidate as a suffix is chosen, the tree's pruning
  /// method will not find it. Thus, we need to prune before outlining as well.
  ///
  /// \param[in,out] CandidateList A list of outlining candidates.
  /// \param[in,out] FunctionList A list of functions to be outlined.
  /// \param Mapper Contains instruction mapping info for outlining.
  /// \param MaxCandidateLen The length of the longest candidate.
  /// \param TII TargetInstrInfo for the module.
  void pruneOverlaps(std::vector<std::shared_ptr<Candidate>> &CandidateList,
                     std::vector<OutlinedFunction> &FunctionList,
                     InstructionMapper &Mapper, unsigned MaxCandidateLen,
                     const TargetInstrInfo &TII);

  /// Construct a suffix tree on the instructions in \p M and outline repeated
  /// strings from that tree.
  bool runOnModule(Module &M) override;
};

} // Anonymous namespace.

char MachineOutliner::ID = 0;

namespace llvm {
ModulePass *createMachineOutlinerPass() {
  return new MachineOutliner();
}

} // namespace llvm

INITIALIZE_PASS(MachineOutliner, DEBUG_TYPE, "Machine Function Outliner", false,
                false)

unsigned MachineOutliner::findCandidates(
    SuffixTree &ST, const TargetInstrInfo &TII, InstructionMapper &Mapper,
    std::vector<std::shared_ptr<Candidate>> &CandidateList,
    std::vector<OutlinedFunction> &FunctionList) {
  CandidateList.clear();
  FunctionList.clear();
  unsigned MaxLen = 0;

  // FIXME: Visit internal nodes instead of leaves.
  for (SuffixTreeNode *Leaf : ST.LeafVector) {
    assert(Leaf && "Leaves in LeafVector cannot be null!");
    if (!Leaf->IsInTree)
      continue;

    assert(Leaf->Parent && "All leaves must have parents!");
    SuffixTreeNode &Parent = *(Leaf->Parent);

    // If it doesn't appear enough, or we already outlined from it, skip it.
    if (Parent.OccurrenceCount < 2 || Parent.isRoot() || !Parent.IsInTree)
      continue;

    // Figure out if this candidate is beneficial.
    unsigned StringLen = Leaf->ConcatLen - (unsigned)Leaf->size();

    // Too short to be beneficial; skip it.
    // FIXME: This isn't necessarily true for, say, X86. If we factor in
    // instruction lengths we need more information than this.
    if (StringLen < 2)
      continue;

    // If this is a beneficial class of candidate, then every one is stored in
    // this vector.
    std::vector<Candidate> CandidatesForRepeatedSeq;

    // Describes the start and end point of each candidate. This allows the
    // target to infer some information about each occurrence of each repeated
    // sequence.
    // FIXME: CandidatesForRepeatedSeq and this should be combined.
    std::vector<
        std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>>
        RepeatedSequenceLocs;

    // Figure out the call overhead for each instance of the sequence.
    for (auto &ChildPair : Parent.Children) {
      SuffixTreeNode *M = ChildPair.second;

      if (M && M->IsInTree && M->isLeaf()) {
        // Never visit this leaf again.
        M->IsInTree = false;
        unsigned StartIdx = M->SuffixIdx;
        unsigned EndIdx = StartIdx + StringLen - 1;

        // Trick: Discard some candidates that would be incompatible with the
        // ones we've already found for this sequence. This will save us some
        // work in candidate selection.
        //
        // If two candidates overlap, then we can't outline them both. This
        // happens when we have candidates that look like, say
        //
        // AA (where each "A" is an instruction).
        //
        // We might have some portion of the module that looks like this:
        // AAAAAA (6 A's) 
        //
        // In this case, there are 5 different copies of "AA" in this range, but
        // at most 3 can be outlined. If only outlining 3 of these is going to
        // be unbeneficial, then we ought to not bother.
        //
        // Note that two things DON'T overlap when they look like this:
        // start1...end1 .... start2...end2
        // That is, one must either
        // * End before the other starts
        // * Start after the other ends
        if (std::all_of(CandidatesForRepeatedSeq.begin(),
                        CandidatesForRepeatedSeq.end(),
                        [&StartIdx, &EndIdx](const Candidate &C) {
                          return (EndIdx < C.getStartIdx() ||
                                  StartIdx > C.getEndIdx()); 
                        })) {
          // It doesn't overlap with anything, so we can outline it.
          // Each sequence is over [StartIt, EndIt].
          MachineBasicBlock::iterator StartIt = Mapper.InstrList[StartIdx];
          MachineBasicBlock::iterator EndIt = Mapper.InstrList[EndIdx];

          // Save the MachineFunction containing the Candidate.
          MachineFunction *MF = StartIt->getParent()->getParent();
          assert(MF && "Candidate doesn't have a MF?");

          // Save the candidate and its location.
          CandidatesForRepeatedSeq.emplace_back(StartIdx, StringLen,
                                                FunctionList.size(), MF);
          RepeatedSequenceLocs.emplace_back(std::make_pair(StartIt, EndIt));
        }
      }
    }

    // We've found something we might want to outline.
    // Create an OutlinedFunction to store it and check if it'd be beneficial
    // to outline.
    TargetInstrInfo::MachineOutlinerInfo MInfo =
        TII.getOutlininingCandidateInfo(RepeatedSequenceLocs);
    std::vector<unsigned> Seq;
    for (unsigned i = Leaf->SuffixIdx; i < Leaf->SuffixIdx + StringLen; i++)
      Seq.push_back(ST.Str[i]);
    OutlinedFunction OF(FunctionList.size(), CandidatesForRepeatedSeq.size(),
                        Seq, MInfo);
    unsigned Benefit = OF.getBenefit();

    // Is it better to outline this candidate than not?
    if (Benefit < 1) {
      // Outlining this candidate would take more instructions than not
      // outlining.
      // Emit a remark explaining why we didn't outline this candidate.
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator> C =
          RepeatedSequenceLocs[0];
      MachineOptimizationRemarkEmitter MORE(
          *(C.first->getParent()->getParent()), nullptr);
      MORE.emit([&]() {
        MachineOptimizationRemarkMissed R(DEBUG_TYPE, "NotOutliningCheaper",
                                          C.first->getDebugLoc(),
                                          C.first->getParent());
        R << "Did not outline " << NV("Length", StringLen) << " instructions"
          << " from " << NV("NumOccurrences", RepeatedSequenceLocs.size())
          << " locations."
          << " Bytes from outlining all occurrences ("
          << NV("OutliningCost", OF.getOutliningCost()) << ")"
          << " >= Unoutlined instruction bytes ("
          << NV("NotOutliningCost", OF.getNotOutlinedCost()) << ")"
          << " (Also found at: ";

        // Tell the user the other places the candidate was found.
        for (unsigned i = 1, e = RepeatedSequenceLocs.size(); i < e; i++) {
          R << NV((Twine("OtherStartLoc") + Twine(i)).str(),
                  RepeatedSequenceLocs[i].first->getDebugLoc());
          if (i != e - 1)
            R << ", ";
        }

        R << ")";
        return R;
      });

      // Move to the next candidate.
      continue;
    }

    if (StringLen > MaxLen)
      MaxLen = StringLen;

    // At this point, the candidate class is seen as beneficial. Set their
    // benefit values and save them in the candidate list.
    std::vector<std::shared_ptr<Candidate>> CandidatesForFn;
    for (Candidate &C : CandidatesForRepeatedSeq) {
      C.Benefit = Benefit;
      C.MInfo = MInfo;
      std::shared_ptr<Candidate> Cptr = std::make_shared<Candidate>(C);
      CandidateList.push_back(Cptr);
      CandidatesForFn.push_back(Cptr);
    }

    FunctionList.push_back(OF);
    FunctionList.back().Candidates = CandidatesForFn;

    // Move to the next function.
    Parent.IsInTree = false;
  }

  return MaxLen;
}

// Remove C from the candidate space, and update its OutlinedFunction.
void MachineOutliner::prune(Candidate &C,
                            std::vector<OutlinedFunction> &FunctionList) {
  // Get the OutlinedFunction associated with this Candidate.
  OutlinedFunction &F = FunctionList[C.FunctionIdx];

  // Update C's associated function's occurrence count.
  F.decrement();

  // Remove C from the CandidateList.
  C.InCandidateList = false;

  LLVM_DEBUG(dbgs() << "- Removed a Candidate \n";
             dbgs() << "--- Num fns left for candidate: "
                    << F.getOccurrenceCount() << "\n";
             dbgs() << "--- Candidate's functions's benefit: " << F.getBenefit()
                    << "\n";);
}

void MachineOutliner::pruneOverlaps(
    std::vector<std::shared_ptr<Candidate>> &CandidateList,
    std::vector<OutlinedFunction> &FunctionList, InstructionMapper &Mapper,
    unsigned MaxCandidateLen, const TargetInstrInfo &TII) {

  // Return true if this candidate became unbeneficial for outlining in a
  // previous step.
  auto ShouldSkipCandidate = [&FunctionList, this](Candidate &C) {

    // Check if the candidate was removed in a previous step.
    if (!C.InCandidateList)
      return true;

    // C must be alive. Check if we should remove it.
    if (FunctionList[C.FunctionIdx].getBenefit() < 1) {
      prune(C, FunctionList);
      return true;
    }

    // C is in the list, and F is still beneficial.
    return false;
  };

  // TODO: Experiment with interval trees or other interval-checking structures
  // to lower the time complexity of this function.
  // TODO: Can we do better than the simple greedy choice?
  // Check for overlaps in the range.
  // This is O(MaxCandidateLen * CandidateList.size()).
  for (auto It = CandidateList.begin(), Et = CandidateList.end(); It != Et;
       It++) {
    Candidate &C1 = **It;

    // If C1 was already pruned, or its function is no longer beneficial for
    // outlining, move to the next candidate.
    if (ShouldSkipCandidate(C1))
      continue;

    // The minimum start index of any candidate that could overlap with this
    // one.
    unsigned FarthestPossibleIdx = 0;

    // Either the index is 0, or it's at most MaxCandidateLen indices away.
    if (C1.getStartIdx() > MaxCandidateLen)
      FarthestPossibleIdx = C1.getStartIdx() - MaxCandidateLen;

    // Compare against the candidates in the list that start at most
    // FarthestPossibleIdx indices away from C1. There are at most
    // MaxCandidateLen of these.
    for (auto Sit = It + 1; Sit != Et; Sit++) {
      Candidate &C2 = **Sit;

      // Is this candidate too far away to overlap?
      if (C2.getStartIdx() < FarthestPossibleIdx)
        break;

      // If C2 was already pruned, or its function is no longer beneficial for
      // outlining, move to the next candidate.
      if (ShouldSkipCandidate(C2))
        continue;

      // Do C1 and C2 overlap?
      //
      // Not overlapping:
      // High indices... [C1End ... C1Start][C2End ... C2Start] ...Low indices
      //
      // We sorted our candidate list so C2Start <= C1Start. We know that
      // C2End > C2Start since each candidate has length >= 2. Therefore, all we
      // have to check is C2End < C2Start to see if we overlap.
      if (C2.getEndIdx() < C1.getStartIdx())
        continue;

      // C1 and C2 overlap.
      // We need to choose the better of the two.
      //
      // Approximate this by picking the one which would have saved us the
      // most instructions before any pruning.

      // Is C2 a better candidate?
      if (C2.Benefit > C1.Benefit) {
        // Yes, so prune C1. Since C1 is dead, we don't have to compare it
        // against anything anymore, so break.
        prune(C1, FunctionList);
        break;
      }

      // Prune C2 and move on to the next candidate.
      prune(C2, FunctionList);
    }
  }
}

unsigned MachineOutliner::buildCandidateList(
    std::vector<std::shared_ptr<Candidate>> &CandidateList,
    std::vector<OutlinedFunction> &FunctionList, SuffixTree &ST,
    InstructionMapper &Mapper, const TargetInstrInfo &TII) {

  std::vector<unsigned> CandidateSequence; // Current outlining candidate.
  unsigned MaxCandidateLen = 0;            // Length of the longest candidate.

  MaxCandidateLen =
      findCandidates(ST, TII, Mapper, CandidateList, FunctionList);

  // Sort the candidates in decending order. This will simplify the outlining
  // process when we have to remove the candidates from the mapping by
  // allowing us to cut them out without keeping track of an offset.
  std::stable_sort(
      CandidateList.begin(), CandidateList.end(),
      [](const std::shared_ptr<Candidate> &LHS,
         const std::shared_ptr<Candidate> &RHS) { return *LHS < *RHS; });

  return MaxCandidateLen;
}

MachineFunction *
MachineOutliner::createOutlinedFunction(Module &M, const OutlinedFunction &OF,
                                        InstructionMapper &Mapper) {

  // Create the function name. This should be unique. For now, just hash the
  // module name and include it in the function name plus the number of this
  // function.
  std::ostringstream NameStream;
  NameStream << "OUTLINED_FUNCTION_" << OF.Name;

  // Create the function using an IR-level function.
  LLVMContext &C = M.getContext();
  Function *F = dyn_cast<Function>(
      M.getOrInsertFunction(NameStream.str(), Type::getVoidTy(C)));
  assert(F && "Function was null!");

  // NOTE: If this is linkonceodr, then we can take advantage of linker deduping
  // which gives us better results when we outline from linkonceodr functions.
  F->setLinkage(GlobalValue::InternalLinkage);
  F->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  // FIXME: Set nounwind, so we don't generate eh_frame? Haven't verified it's
  // necessary.

  // Set optsize/minsize, so we don't insert padding between outlined
  // functions.
  F->addFnAttr(Attribute::OptimizeForSize);
  F->addFnAttr(Attribute::MinSize);

  // Save F so that we can add debug info later if we need to.
  CreatedIRFunctions.push_back(F);

  BasicBlock *EntryBB = BasicBlock::Create(C, "entry", F);
  IRBuilder<> Builder(EntryBB);
  Builder.CreateRetVoid();

  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfo>();
  MachineFunction &MF = MMI.getOrCreateMachineFunction(*F);
  MachineBasicBlock &MBB = *MF.CreateMachineBasicBlock();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  const TargetInstrInfo &TII = *STI.getInstrInfo();

  // Insert the new function into the module.
  MF.insert(MF.begin(), &MBB);

  TII.insertOutlinerPrologue(MBB, MF, OF.MInfo);

  // Copy over the instructions for the function using the integer mappings in
  // its sequence.
  for (unsigned Str : OF.Sequence) {
    MachineInstr *NewMI =
        MF.CloneMachineInstr(Mapper.IntegerInstructionMap.find(Str)->second);
    NewMI->dropMemRefs();

    // Don't keep debug information for outlined instructions.
    NewMI->setDebugLoc(DebugLoc());
    MBB.insert(MBB.end(), NewMI);
  }

  TII.insertOutlinerEpilogue(MBB, MF, OF.MInfo);

  // If there's a DISubprogram associated with this outlined function, then
  // emit debug info for the outlined function.
  if (DISubprogram *SP = OF.getSubprogramOrNull()) {
    // We have a DISubprogram. Get its DICompileUnit.
    DICompileUnit *CU = SP->getUnit();
    DIBuilder DB(M, true, CU);
    DIFile *Unit = SP->getFile();
    Mangler Mg;

    // Walk over each IR function we created in the outliner and create
    // DISubprograms for each function.
    for (Function *F : CreatedIRFunctions) {
      // Get the mangled name of the function for the linkage name.
      std::string Dummy;
      llvm::raw_string_ostream MangledNameStream(Dummy);
      Mg.getNameWithPrefix(MangledNameStream, F, false);

      DISubprogram *SP = DB.createFunction(
          Unit /* Context */, F->getName(), StringRef(MangledNameStream.str()),
          Unit /* File */,
          0 /* Line 0 is reserved for compiler-generated code. */,
          DB.createSubroutineType(
              DB.getOrCreateTypeArray(None)), /* void type */
          false, true, 0, /* Line 0 is reserved for compiler-generated code. */
          DINode::DIFlags::FlagArtificial /* Compiler-generated code. */,
          true /* Outlined code is optimized code by definition. */);

      // Don't add any new variables to the subprogram.
      DB.finalizeSubprogram(SP);

      // Attach subprogram to the function.
      F->setSubprogram(SP);
    }

    // We're done with the DIBuilder.
    DB.finalize();
  }

  // Outlined functions shouldn't preserve liveness.
  MF.getProperties().reset(MachineFunctionProperties::Property::TracksLiveness);
  MF.getRegInfo().freezeReservedRegs(MF);
  return &MF;
}

bool MachineOutliner::outline(
    Module &M, const ArrayRef<std::shared_ptr<Candidate>> &CandidateList,
    std::vector<OutlinedFunction> &FunctionList, InstructionMapper &Mapper) {

  bool OutlinedSomething = false;
  // Replace the candidates with calls to their respective outlined functions.
  for (const std::shared_ptr<Candidate> &Cptr : CandidateList) {
    Candidate &C = *Cptr;
    // Was the candidate removed during pruneOverlaps?
    if (!C.InCandidateList)
      continue;

    // If not, then look at its OutlinedFunction.
    OutlinedFunction &OF = FunctionList[C.FunctionIdx];

    // Was its OutlinedFunction made unbeneficial during pruneOverlaps?
    if (OF.getBenefit() < 1)
      continue;

    // If not, then outline it.
    assert(C.getStartIdx() < Mapper.InstrList.size() &&
           "Candidate out of bounds!");
    MachineBasicBlock *MBB = (*Mapper.InstrList[C.getStartIdx()]).getParent();
    MachineBasicBlock::iterator StartIt = Mapper.InstrList[C.getStartIdx()];
    unsigned EndIdx = C.getEndIdx();

    assert(EndIdx < Mapper.InstrList.size() && "Candidate out of bounds!");
    MachineBasicBlock::iterator EndIt = Mapper.InstrList[EndIdx];
    assert(EndIt != MBB->end() && "EndIt out of bounds!");

    // Does this candidate have a function yet?
    if (!OF.MF) {
      OF.MF = createOutlinedFunction(M, OF, Mapper);
      MachineBasicBlock *MBB = &*OF.MF->begin();

      // Output a remark telling the user that an outlined function was created,
      // and explaining where it came from.
      MachineOptimizationRemarkEmitter MORE(*OF.MF, nullptr);
      MachineOptimizationRemark R(DEBUG_TYPE, "OutlinedFunction",
                                  MBB->findDebugLoc(MBB->begin()), MBB);
      R << "Saved " << NV("OutliningBenefit", OF.getBenefit())
        << " bytes by "
        << "outlining " << NV("Length", OF.Sequence.size()) << " instructions "
        << "from " << NV("NumOccurrences", OF.getOccurrenceCount())
        << " locations. "
        << "(Found at: ";

      // Tell the user the other places the candidate was found.
      for (size_t i = 0, e = OF.Candidates.size(); i < e; i++) {

        // Skip over things that were pruned.
        if (!OF.Candidates[i]->InCandidateList)
          continue;

        R << NV(
            (Twine("StartLoc") + Twine(i)).str(),
            Mapper.InstrList[OF.Candidates[i]->getStartIdx()]->getDebugLoc());
        if (i != e - 1)
          R << ", ";
      }

      R << ")";

      MORE.emit(R);
      FunctionsCreated++;
    }

    MachineFunction *MF = OF.MF;
    const TargetSubtargetInfo &STI = MF->getSubtarget();
    const TargetInstrInfo &TII = *STI.getInstrInfo();

    // Insert a call to the new function and erase the old sequence.
    auto CallInst = TII.insertOutlinedCall(M, *MBB, StartIt, *MF, C.MInfo);
    StartIt = Mapper.InstrList[C.getStartIdx()];

    // If the caller tracks liveness, then we need to make sure that anything
    // we outline doesn't break liveness assumptions.
    // The outlined functions themselves currently don't track liveness, but
    // we should make sure that the ranges we yank things out of aren't
    // wrong.
    if (MBB->getParent()->getProperties().hasProperty(
            MachineFunctionProperties::Property::TracksLiveness)) {
      // Helper lambda for adding implicit def operands to the call instruction.
      auto CopyDefs = [&CallInst](MachineInstr &MI) {
        for (MachineOperand &MOP : MI.operands()) {
          // Skip over anything that isn't a register.
          if (!MOP.isReg())
            continue;

          // If it's a def, add it to the call instruction.
          if (MOP.isDef())
            CallInst->addOperand(
                MachineOperand::CreateReg(MOP.getReg(), true, /* isDef = true */
                                          true /* isImp = true */));
        }
      };

      // Copy over the defs in the outlined range.
      // First inst in outlined range <-- Anything that's defined in this
      // ...                           .. range has to be added as an implicit
      // Last inst in outlined range  <-- def to the call instruction.
      std::for_each(CallInst, EndIt, CopyDefs);
    }

    EndIt++; // Erase needs one past the end index.
    MBB->erase(StartIt, EndIt);
    OutlinedSomething = true;

    // Statistics.
    NumOutlined++;
  }

  LLVM_DEBUG(dbgs() << "OutlinedSomething = " << OutlinedSomething << "\n";);

  return OutlinedSomething;
}

bool MachineOutliner::runOnModule(Module &M) {
  // Check if there's anything in the module. If it's empty, then there's
  // nothing to outline.
  if (M.empty())
    return false;

  MachineModuleInfo &MMI = getAnalysis<MachineModuleInfo>();
  const TargetSubtargetInfo &STI =
      MMI.getOrCreateMachineFunction(*M.begin()).getSubtarget();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  const TargetInstrInfo *TII = STI.getInstrInfo();

  // Does the target implement the MachineOutliner? If it doesn't, quit here.
  if (!TII->useMachineOutliner()) {
    // No. So we're done.
    LLVM_DEBUG(
        dbgs()
        << "Skipping pass: Target does not support the MachineOutliner.\n");
    return false;
  }

  // If the user specifies that they want to outline from linkonceodrs, set
  // it here.
  OutlineFromLinkOnceODRs = EnableLinkOnceODROutlining;

  InstructionMapper Mapper;

  // Build instruction mappings for each function in the module. Start by
  // iterating over each Function in M.
  for (Function &F : M) {

    // If there's nothing in F, then there's no reason to try and outline from
    // it.
    if (F.empty())
      continue;

    // There's something in F. Check if it has a MachineFunction associated with
    // it.
    MachineFunction *MF = MMI.getMachineFunction(F);

    // If it doesn't, then there's nothing to outline from. Move to the next
    // Function.
    if (!MF)
      continue;

    // We have a MachineFunction. Ask the target if it's suitable for outlining.
    // If it isn't, then move on to the next Function in the module.
    if (!TII->isFunctionSafeToOutlineFrom(*MF, OutlineFromLinkOnceODRs))
      continue;

    // We have a function suitable for outlining. Iterate over every
    // MachineBasicBlock in MF and try to map its instructions to a list of
    // unsigned integers.
    for (MachineBasicBlock &MBB : *MF) {
      // If there isn't anything in MBB, then there's no point in outlining from
      // it.
      if (MBB.empty())
        continue;

      // Check if MBB could be the target of an indirect branch. If it is, then
      // we don't want to outline from it.
      if (MBB.hasAddressTaken())
        continue;

      // MBB is suitable for outlining. Map it to a list of unsigneds.
      Mapper.convertToUnsignedVec(MBB, *TRI, *TII);
    }
  }

  // Construct a suffix tree, use it to find candidates, and then outline them.
  SuffixTree ST(Mapper.UnsignedVec);
  std::vector<std::shared_ptr<Candidate>> CandidateList;
  std::vector<OutlinedFunction> FunctionList;

  // Find all of the outlining candidates.
  unsigned MaxCandidateLen =
      buildCandidateList(CandidateList, FunctionList, ST, Mapper, *TII);

  // Remove candidates that overlap with other candidates.
  pruneOverlaps(CandidateList, FunctionList, Mapper, MaxCandidateLen, *TII);

  // Outline each of the candidates and return true if something was outlined.
  bool OutlinedSomething = outline(M, CandidateList, FunctionList, Mapper);

  return OutlinedSomething;
}
