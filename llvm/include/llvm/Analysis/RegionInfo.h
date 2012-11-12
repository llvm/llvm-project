//===- RegionInfo.h - SESE region analysis ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Calculate a program structure tree built out of single entry single exit
// regions.
// The basic ideas are taken from "The Program Structure Tree - Richard Johnson,
// David Pearson, Keshav Pingali - 1994", however enriched with ideas from "The
// Refined Process Structure Tree - Jussi Vanhatalo, Hagen Voelyer, Jana
// Koehler - 2009".
// The algorithm to calculate these data structures however is completely
// different, as it takes advantage of existing information already available
// in (Post)dominace tree and dominance frontier passes. This leads to a simpler
// and in practice hopefully better performing algorithm. The runtime of the
// algorithms described in the papers above are both linear in graph size,
// O(V+E), whereas this algorithm is not, as the dominance frontier information
// itself is not, but in practice runtime seems to be in the order of magnitude
// of dominance tree calculation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_REGION_INFO_H
#define LLVM_ANALYSIS_REGION_INFO_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/Allocator.h"
#include <map>

namespace llvm {

class Region;
class RegionInfo;
class raw_ostream;
class Loop;
class LoopInfo;

/// @brief Marker class to iterate over the elements of a Region in flat mode.
///
/// The class is used to either iterate in Flat mode or by not using it to not
/// iterate in Flat mode.  During a Flat mode iteration all Regions are entered
/// and the iteration returns every BasicBlock.  If the Flat mode is not
/// selected for SubRegions just one RegionNode containing the subregion is
/// returned.
template <class GraphType>
class FlatIt {};

/// @brief A RegionNode represents a subregion or a BasicBlock that is part of a
/// Region.
class RegionNode {
  RegionNode(const RegionNode &) LLVM_DELETED_FUNCTION;
  const RegionNode &operator=(const RegionNode &) LLVM_DELETED_FUNCTION;

protected:
  /// This is the entry basic block that starts this region node.  If this is a
  /// BasicBlock RegionNode, then entry is just the basic block, that this
  /// RegionNode represents.  Otherwise it is the entry of this (Sub)RegionNode.
  ///
  /// In the BBtoRegionNode map of the parent of this node, BB will always map
  /// to this node no matter which kind of node this one is.
  ///
  /// The node can hold either a Region or a BasicBlock.
  /// Use one bit to save, if this RegionNode is a subregion or BasicBlock
  /// RegionNode.
  PointerIntPair<BasicBlock*, 1, bool> entry;

  /// @brief The parent Region of this RegionNode.
  /// @see getParent()
  Region* parent;

public:
  /// @brief Create a RegionNode.
  ///
  /// @param Parent      The parent of this RegionNode.
  /// @param Entry       The entry BasicBlock of the RegionNode.  If this
  ///                    RegionNode represents a BasicBlock, this is the
  ///                    BasicBlock itself.  If it represents a subregion, this
  ///                    is the entry BasicBlock of the subregion.
  /// @param isSubRegion If this RegionNode represents a SubRegion.
  inline RegionNode(Region* Parent, BasicBlock* Entry, bool isSubRegion = 0)
    : entry(Entry, isSubRegion), parent(Parent) {}

  /// @brief Get the parent Region of this RegionNode.
  ///
  /// The parent Region is the Region this RegionNode belongs to. If for
  /// example a BasicBlock is element of two Regions, there exist two
  /// RegionNodes for this BasicBlock. Each with the getParent() function
  /// pointing to the Region this RegionNode belongs to.
  ///
  /// @return Get the parent Region of this RegionNode.
  inline Region* getParent() const { return parent; }

  /// @brief Get the entry BasicBlock of this RegionNode.
  ///
  /// If this RegionNode represents a BasicBlock this is just the BasicBlock
  /// itself, otherwise we return the entry BasicBlock of the Subregion
  ///
  /// @return The entry BasicBlock of this RegionNode.
  inline BasicBlock* getEntry() const { return entry.getPointer(); }

  /// @brief Get the content of this RegionNode.
  ///
  /// This can be either a BasicBlock or a subregion. Before calling getNodeAs()
  /// check the type of the content with the isSubRegion() function call.
  ///
  /// @return The content of this RegionNode.
  template<class T>
  inline T* getNodeAs() const;

  /// @brief Is this RegionNode a subregion?
  ///
  /// @return True if it contains a subregion. False if it contains a
  ///         BasicBlock.
  inline bool isSubRegion() const {
    return entry.getInt();
  }
};

/// Print a RegionNode.
inline raw_ostream &operator<<(raw_ostream &OS, const RegionNode &Node);

template<>
inline BasicBlock* RegionNode::getNodeAs<BasicBlock>() const {
  assert(!isSubRegion() && "This is not a BasicBlock RegionNode!");
  return getEntry();
}

template<>
inline Region* RegionNode::getNodeAs<Region>() const {
  assert(isSubRegion() && "This is not a subregion RegionNode!");
  return reinterpret_cast<Region*>(const_cast<RegionNode*>(this));
}

//===----------------------------------------------------------------------===//
/// @brief A single entry single exit Region.
///
/// A Region is a connected subgraph of a control flow graph that has exactly
/// two connections to the remaining graph. It can be used to analyze or
/// optimize parts of the control flow graph.
///
/// A <em> simple Region </em> is connected to the remaining graph by just two
/// edges. One edge entering the Region and another one leaving the Region.
///
/// An <em> extended Region </em> (or just Region) is a subgraph that can be
/// transform into a simple Region. The transformation is done by adding
/// BasicBlocks that merge several entry or exit edges so that after the merge
/// just one entry and one exit edge exists.
///
/// The \e Entry of a Region is the first BasicBlock that is passed after
/// entering the Region. It is an element of the Region. The entry BasicBlock
/// dominates all BasicBlocks in the Region.
///
/// The \e Exit of a Region is the first BasicBlock that is passed after
/// leaving the Region. It is not an element of the Region. The exit BasicBlock,
/// postdominates all BasicBlocks in the Region.
///
/// A <em> canonical Region </em> cannot be constructed by combining smaller
/// Regions.
///
/// Region A is the \e parent of Region B, if B is completely contained in A.
///
/// Two canonical Regions either do not intersect at all or one is
/// the parent of the other.
///
/// The <em> Program Structure Tree</em> is a graph (V, E) where V is the set of
/// Regions in the control flow graph and E is the \e parent relation of these
/// Regions.
///
/// Example:
///
/// \verbatim
/// A simple control flow graph, that contains two regions.
///
///        1
///       / |
///      2   |
///     / \   3
///    4   5  |
///    |   |  |
///    6   7  8
///     \  | /
///      \ |/       Region A: 1 -> 9 {1,2,3,4,5,6,7,8}
///        9        Region B: 2 -> 9 {2,4,5,6,7}
/// \endverbatim
///
/// You can obtain more examples by either calling
///
/// <tt> "opt -regions -analyze anyprogram.ll" </tt>
/// or
/// <tt> "opt -view-regions-only anyprogram.ll" </tt>
///
/// on any LLVM file you are interested in.
///
/// The first call returns a textual representation of the program structure
/// tree, the second one creates a graphical representation using graphviz.
class Region : public RegionNode {
  friend class RegionInfo;
  Region(const Region &) LLVM_DELETED_FUNCTION;
  const Region &operator=(const Region &) LLVM_DELETED_FUNCTION;

  // Information necessary to manage this Region.
  RegionInfo* RI;
  DominatorTree *DT;

  // The exit BasicBlock of this region.
  // (The entry BasicBlock is part of RegionNode)
  BasicBlock *exit;

  typedef std::vector<Region*> RegionSet;

  // The subregions of this region.
  RegionSet children;

  typedef std::map<BasicBlock*, RegionNode*> BBNodeMapT;

  // Save the BasicBlock RegionNodes that are element of this Region.
  mutable BBNodeMapT BBNodeMap;

  /// verifyBBInRegion - Check if a BB is in this Region. This check also works
  /// if the region is incorrectly built. (EXPENSIVE!)
  void verifyBBInRegion(BasicBlock* BB) const;

  /// verifyWalk - Walk over all the BBs of the region starting from BB and
  /// verify that all reachable basic blocks are elements of the region.
  /// (EXPENSIVE!)
  void verifyWalk(BasicBlock* BB, std::set<BasicBlock*>* visitedBB) const;

  /// verifyRegionNest - Verify if the region and its children are valid
  /// regions (EXPENSIVE!)
  void verifyRegionNest() const;

public:
  /// @brief Create a new region.
  ///
  /// @param Entry  The entry basic block of the region.
  /// @param Exit   The exit basic block of the region.
  /// @param RI     The region info object that is managing this region.
  /// @param DT     The dominator tree of the current function.
  /// @param Parent The surrounding region or NULL if this is a top level
  ///               region.
  Region(BasicBlock *Entry, BasicBlock *Exit, RegionInfo* RI,
         DominatorTree *DT, Region *Parent = 0);

  /// Delete the Region and all its subregions.
  ~Region();

  /// @brief Get the entry BasicBlock of the Region.
  /// @return The entry BasicBlock of the region.
  BasicBlock *getEntry() const { return RegionNode::getEntry(); }

  /// @brief Replace the entry basic block of the region with the new basic
  ///        block.
  ///
  /// @param BB  The new entry basic block of the region.
  void replaceEntry(BasicBlock *BB);

  /// @brief Replace the exit basic block of the region with the new basic
  ///        block.
  ///
  /// @param BB  The new exit basic block of the region.
  void replaceExit(BasicBlock *BB);

  /// @brief Get the exit BasicBlock of the Region.
  /// @return The exit BasicBlock of the Region, NULL if this is the TopLevel
  ///         Region.
  BasicBlock *getExit() const { return exit; }

  /// @brief Get the parent of the Region.
  /// @return The parent of the Region or NULL if this is a top level
  ///         Region.
  Region *getParent() const { return RegionNode::getParent(); }

  /// @brief Get the RegionNode representing the current Region.
  /// @return The RegionNode representing the current Region.
  RegionNode* getNode() const {
    return const_cast<RegionNode*>(reinterpret_cast<const RegionNode*>(this));
  }

  /// @brief Get the nesting level of this Region.
  ///
  /// An toplevel Region has depth 0.
  ///
  /// @return The depth of the region.
  unsigned getDepth() const;

  /// @brief Check if a Region is the TopLevel region.
  ///
  /// The toplevel region represents the whole function.
  bool isTopLevelRegion() const { return exit == NULL; }

  /// @brief Return a new (non canonical) region, that is obtained by joining
  ///        this region with its predecessors.
  ///
  /// @return A region also starting at getEntry(), but reaching to the next
  ///         basic block that forms with getEntry() a (non canonical) region.
  ///         NULL if such a basic block does not exist.
  Region *getExpandedRegion() const;

  /// @brief Return the first block of this region's single entry edge,
  ///        if existing.
  ///
  /// @return The BasicBlock starting this region's single entry edge,
  ///         else NULL.
  BasicBlock *getEnteringBlock() const;

  /// @brief Return the first block of this region's single exit edge,
  ///        if existing.
  ///
  /// @return The BasicBlock starting this region's single exit edge,
  ///         else NULL.
  BasicBlock *getExitingBlock() const;

  /// @brief Is this a simple region?
  ///
  /// A region is simple if it has exactly one exit and one entry edge.
  ///
  /// @return True if the Region is simple.
  bool isSimple() const;

  /// @brief Returns the name of the Region.
  /// @return The Name of the Region.
  std::string getNameStr() const;

  /// @brief Return the RegionInfo object, that belongs to this Region.
  RegionInfo *getRegionInfo() const {
    return RI;
  }

  /// PrintStyle - Print region in difference ways.
  enum PrintStyle { PrintNone, PrintBB, PrintRN  };

  /// @brief Print the region.
  ///
  /// @param OS The output stream the Region is printed to.
  /// @param printTree Print also the tree of subregions.
  /// @param level The indentation level used for printing.
  void print(raw_ostream& OS, bool printTree = true, unsigned level = 0,
             enum PrintStyle Style = PrintNone) const;

  /// @brief Print the region to stderr.
  void dump() const;

  /// @brief Check if the region contains a BasicBlock.
  ///
  /// @param BB The BasicBlock that might be contained in this Region.
  /// @return True if the block is contained in the region otherwise false.
  bool contains(const BasicBlock *BB) const;

  /// @brief Check if the region contains another region.
  ///
  /// @param SubRegion The region that might be contained in this Region.
  /// @return True if SubRegion is contained in the region otherwise false.
  bool contains(const Region *SubRegion) const {
    // Toplevel Region.
    if (!getExit())
      return true;

    return contains(SubRegion->getEntry())
      && (contains(SubRegion->getExit()) || SubRegion->getExit() == getExit());
  }

  /// @brief Check if the region contains an Instruction.
  ///
  /// @param Inst The Instruction that might be contained in this region.
  /// @return True if the Instruction is contained in the region otherwise false.
  bool contains(const Instruction *Inst) const {
    return contains(Inst->getParent());
  }

  /// @brief Check if the region contains a loop.
  ///
  /// @param L The loop that might be contained in this region.
  /// @return True if the loop is contained in the region otherwise false.
  ///         In case a NULL pointer is passed to this function the result
  ///         is false, except for the region that describes the whole function.
  ///         In that case true is returned.
  bool contains(const Loop *L) const;

  /// @brief Get the outermost loop in the region that contains a loop.
  ///
  /// Find for a Loop L the outermost loop OuterL that is a parent loop of L
  /// and is itself contained in the region.
  ///
  /// @param L The loop the lookup is started.
  /// @return The outermost loop in the region, NULL if such a loop does not
  ///         exist or if the region describes the whole function.
  Loop *outermostLoopInRegion(Loop *L) const;

  /// @brief Get the outermost loop in the region that contains a basic block.
  ///
  /// Find for a basic block BB the outermost loop L that contains BB and is
  /// itself contained in the region.
  ///
  /// @param LI A pointer to a LoopInfo analysis.
  /// @param BB The basic block surrounded by the loop.
  /// @return The outermost loop in the region, NULL if such a loop does not
  ///         exist or if the region describes the whole function.
  Loop *outermostLoopInRegion(LoopInfo *LI, BasicBlock* BB) const;

  /// @brief Get the subregion that starts at a BasicBlock
  ///
  /// @param BB The BasicBlock the subregion should start.
  /// @return The Subregion if available, otherwise NULL.
  Region* getSubRegionNode(BasicBlock *BB) const;

  /// @brief Get the RegionNode for a BasicBlock
  ///
  /// @param BB The BasicBlock at which the RegionNode should start.
  /// @return If available, the RegionNode that represents the subregion
  ///         starting at BB. If no subregion starts at BB, the RegionNode
  ///         representing BB.
  RegionNode* getNode(BasicBlock *BB) const;

  /// @brief Get the BasicBlock RegionNode for a BasicBlock
  ///
  /// @param BB The BasicBlock for which the RegionNode is requested.
  /// @return The RegionNode representing the BB.
  RegionNode* getBBNode(BasicBlock *BB) const;

  /// @brief Add a new subregion to this Region.
  ///
  /// @param SubRegion The new subregion that will be added.
  /// @param moveChildren Move the children of this region, that are also
  ///                     contained in SubRegion into SubRegion.
  void addSubRegion(Region *SubRegion, bool moveChildren = false);

  /// @brief Remove a subregion from this Region.
  ///
  /// The subregion is not deleted, as it will probably be inserted into another
  /// region.
  /// @param SubRegion The SubRegion that will be removed.
  Region *removeSubRegion(Region *SubRegion);

  /// @brief Move all direct child nodes of this Region to another Region.
  ///
  /// @param To The Region the child nodes will be transferred to.
  void transferChildrenTo(Region *To);

  /// @brief Verify if the region is a correct region.
  ///
  /// Check if this is a correctly build Region. This is an expensive check, as
  /// the complete CFG of the Region will be walked.
  void verifyRegion() const;

  /// @brief Clear the cache for BB RegionNodes.
  ///
  /// After calling this function the BasicBlock RegionNodes will be stored at
  /// different memory locations. RegionNodes obtained before this function is
  /// called are therefore not comparable to RegionNodes abtained afterwords.
  void clearNodeCache();

  /// @name Subregion Iterators
  ///
  /// These iterators iterator over all subregions of this Region.
  //@{
  typedef RegionSet::iterator iterator;
  typedef RegionSet::const_iterator const_iterator;

  iterator begin() { return children.begin(); }
  iterator end() { return children.end(); }

  const_iterator begin() const { return children.begin(); }
  const_iterator end() const { return children.end(); }
  //@}

  /// @name BasicBlock Iterators
  ///
  /// These iterators iterate over all BasicBlocks that are contained in this
  /// Region. The iterator also iterates over BasicBlocks that are elements of
  /// a subregion of this Region. It is therefore called a flat iterator.
  //@{
  template <bool IsConst>
  class block_iterator_wrapper
    : public df_iterator<typename conditional<IsConst,
                                              const BasicBlock,
                                              BasicBlock>::type*> {
    typedef df_iterator<typename conditional<IsConst,
                                             const BasicBlock,
                                             BasicBlock>::type*>
      super;
  public:
    typedef block_iterator_wrapper<IsConst> Self;
    typedef typename super::pointer pointer;

    // Construct the begin iterator.
    block_iterator_wrapper(pointer Entry, pointer Exit) : super(df_begin(Entry))
    {
      // Mark the exit of the region as visited, so that the children of the
      // exit and the exit itself, i.e. the block outside the region will never
      // be visited.
      super::Visited.insert(Exit);
    }

    // Construct the end iterator.
    block_iterator_wrapper() : super(df_end<pointer>((BasicBlock *)0)) {}

    /*implicit*/ block_iterator_wrapper(super I) : super(I) {}

    // FIXME: Even a const_iterator returns a non-const BasicBlock pointer.
    //        This was introduced for backwards compatibility, but should
    //        be removed as soon as all users are fixed.
    BasicBlock *operator*() const {
      return const_cast<BasicBlock*>(super::operator*());
    }
  };

  typedef block_iterator_wrapper<false> block_iterator;
  typedef block_iterator_wrapper<true>  const_block_iterator;

  block_iterator block_begin() {
   return block_iterator(getEntry(), getExit());
  }

  block_iterator block_end() {
   return block_iterator();
  }

  const_block_iterator block_begin() const {
    return const_block_iterator(getEntry(), getExit());
  }
  const_block_iterator block_end() const {
    return const_block_iterator();
  }
  //@}

  /// @name Element Iterators
  ///
  /// These iterators iterate over all BasicBlock and subregion RegionNodes that
  /// are direct children of this Region. It does not iterate over any
  /// RegionNodes that are also element of a subregion of this Region.
  //@{
  typedef df_iterator<RegionNode*, SmallPtrSet<RegionNode*, 8>, false,
                      GraphTraits<RegionNode*> > element_iterator;

  typedef df_iterator<const RegionNode*, SmallPtrSet<const RegionNode*, 8>,
                      false, GraphTraits<const RegionNode*> >
            const_element_iterator;

  element_iterator element_begin();
  element_iterator element_end();

  const_element_iterator element_begin() const;
  const_element_iterator element_end() const;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Analysis that detects all canonical Regions.
///
/// The RegionInfo pass detects all canonical regions in a function. The Regions
/// are connected using the parent relation. This builds a Program Structure
/// Tree.
class RegionInfo : public FunctionPass {
  typedef DenseMap<BasicBlock*,BasicBlock*> BBtoBBMap;
  typedef DenseMap<BasicBlock*, Region*> BBtoRegionMap;
  typedef SmallPtrSet<Region*, 4> RegionSet;

  RegionInfo(const RegionInfo &) LLVM_DELETED_FUNCTION;
  const RegionInfo &operator=(const RegionInfo &) LLVM_DELETED_FUNCTION;

  DominatorTree *DT;
  PostDominatorTree *PDT;
  DominanceFrontier *DF;

  /// The top level region.
  Region *TopLevelRegion;

  /// Map every BB to the smallest region, that contains BB.
  BBtoRegionMap BBtoRegion;

  // isCommonDomFrontier - Returns true if BB is in the dominance frontier of
  // entry, because it was inherited from exit. In the other case there is an
  // edge going from entry to BB without passing exit.
  bool isCommonDomFrontier(BasicBlock* BB, BasicBlock* entry,
                           BasicBlock* exit) const;

  // isRegion - Check if entry and exit surround a valid region, based on
  // dominance tree and dominance frontier.
  bool isRegion(BasicBlock* entry, BasicBlock* exit) const;

  // insertShortCut - Saves a shortcut pointing from entry to exit.
  // This function may extend this shortcut if possible.
  void insertShortCut(BasicBlock* entry, BasicBlock* exit,
                      BBtoBBMap* ShortCut) const;

  // getNextPostDom - Returns the next BB that postdominates N, while skipping
  // all post dominators that cannot finish a canonical region.
  DomTreeNode *getNextPostDom(DomTreeNode* N, BBtoBBMap *ShortCut) const;

  // isTrivialRegion - A region is trivial, if it contains only one BB.
  bool isTrivialRegion(BasicBlock *entry, BasicBlock *exit) const;

  // createRegion - Creates a single entry single exit region.
  Region *createRegion(BasicBlock *entry, BasicBlock *exit);

  // findRegionsWithEntry - Detect all regions starting with bb 'entry'.
  void findRegionsWithEntry(BasicBlock *entry, BBtoBBMap *ShortCut);

  // scanForRegions - Detects regions in F.
  void scanForRegions(Function &F, BBtoBBMap *ShortCut);

  // getTopMostParent - Get the top most parent with the same entry block.
  Region *getTopMostParent(Region *region);

  // buildRegionsTree - build the region hierarchy after all region detected.
  void buildRegionsTree(DomTreeNode *N, Region *region);

  // Calculate - detecte all regions in function and build the region tree.
  void Calculate(Function& F);

  void releaseMemory();

  // updateStatistics - Update statistic about created regions.
  void updateStatistics(Region *R);

  // isSimple - Check if a region is a simple region with exactly one entry
  // edge and exactly one exit edge.
  bool isSimple(Region* R) const;

public:
  static char ID;
  explicit RegionInfo();

  ~RegionInfo();

  /// @name FunctionPass interface
  //@{
  virtual bool runOnFunction(Function &F);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void print(raw_ostream &OS, const Module *) const;
  virtual void verifyAnalysis() const;
  //@}

  /// @brief Get the smallest region that contains a BasicBlock.
  ///
  /// @param BB The basic block.
  /// @return The smallest region, that contains BB or NULL, if there is no
  /// region containing BB.
  Region *getRegionFor(BasicBlock *BB) const;

  /// @brief  Set the smallest region that surrounds a basic block.
  ///
  /// @param BB The basic block surrounded by a region.
  /// @param R The smallest region that surrounds BB.
  void setRegionFor(BasicBlock *BB, Region *R);

  /// @brief A shortcut for getRegionFor().
  ///
  /// @param BB The basic block.
  /// @return The smallest region, that contains BB or NULL, if there is no
  /// region containing BB.
  Region *operator[](BasicBlock *BB) const;

  /// @brief Return the exit of the maximal refined region, that starts at a
  /// BasicBlock.
  ///
  /// @param BB The BasicBlock the refined region starts.
  BasicBlock *getMaxRegionExit(BasicBlock *BB) const;

  /// @brief Find the smallest region that contains two regions.
  ///
  /// @param A The first region.
  /// @param B The second region.
  /// @return The smallest region containing A and B.
  Region *getCommonRegion(Region* A, Region *B) const;

  /// @brief Find the smallest region that contains two basic blocks.
  ///
  /// @param A The first basic block.
  /// @param B The second basic block.
  /// @return The smallest region that contains A and B.
  Region* getCommonRegion(BasicBlock* A, BasicBlock *B) const {
    return getCommonRegion(getRegionFor(A), getRegionFor(B));
  }

  /// @brief Find the smallest region that contains a set of regions.
  ///
  /// @param Regions A vector of regions.
  /// @return The smallest region that contains all regions in Regions.
  Region* getCommonRegion(SmallVectorImpl<Region*> &Regions) const;

  /// @brief Find the smallest region that contains a set of basic blocks.
  ///
  /// @param BBs A vector of basic blocks.
  /// @return The smallest region that contains all basic blocks in BBS.
  Region* getCommonRegion(SmallVectorImpl<BasicBlock*> &BBs) const;

  Region *getTopLevelRegion() const {
    return TopLevelRegion;
  }

  /// @brief Update RegionInfo after a basic block was split.
  ///
  /// @param NewBB The basic block that was created before OldBB.
  /// @param OldBB The old basic block.
  void splitBlock(BasicBlock* NewBB, BasicBlock *OldBB);

  /// @brief Clear the Node Cache for all Regions.
  ///
  /// @see Region::clearNodeCache()
  void clearNodeCache() {
    if (TopLevelRegion)
      TopLevelRegion->clearNodeCache();
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const RegionNode &Node) {
  if (Node.isSubRegion())
    return OS << Node.getNodeAs<Region>()->getNameStr();
  else
    return OS << Node.getNodeAs<BasicBlock>()->getName();
}
} // End llvm namespace
#endif

