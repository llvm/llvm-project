//===- RegionIterator.h - Iterators to iteratate over Regions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file defines the iterators to iterate over the elements of a Region.
//===----------------------------------------------------------------------===//
#ifndef LLVM_ANALYSIS_REGION_ITERATOR_H
#define LLVM_ANALYSIS_REGION_ITERATOR_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
//===----------------------------------------------------------------------===//
/// @brief Hierarchical RegionNode successor iterator.
///
/// This iterator iterates over all successors of a RegionNode.
///
/// For a BasicBlock RegionNode it skips all BasicBlocks that are not part of
/// the parent Region.  Furthermore for BasicBlocks that start a subregion, a
/// RegionNode representing the subregion is returned.
///
/// For a subregion RegionNode there is just one successor. The RegionNode
/// representing the exit of the subregion.
template<class NodeType>
class RNSuccIterator : public std::iterator<std::forward_iterator_tag,
                                           NodeType, ptrdiff_t>
{
  typedef std::iterator<std::forward_iterator_tag, NodeType, ptrdiff_t> super;
  // The iterator works in two modes, bb mode or region mode.
  enum ItMode{
    // In BB mode it returns all successors of this BasicBlock as its
    // successors.
    ItBB,
    // In region mode there is only one successor, thats the regionnode mapping
    // to the exit block of the regionnode
    ItRgBegin, // At the beginning of the regionnode successor.
    ItRgEnd    // At the end of the regionnode successor.
  };

  // Use two bit to represent the mode iterator.
  PointerIntPair<NodeType*, 2, enum ItMode> Node;

  // The block successor iterator.
  succ_iterator BItor;

  // advanceRegionSucc - A region node has only one successor. It reaches end
  // once we advance it.
  void advanceRegionSucc() {
    assert(Node.getInt() == ItRgBegin && "Cannot advance region successor!");
    Node.setInt(ItRgEnd);
  }

  NodeType* getNode() const{ return Node.getPointer(); }

  // isRegionMode - Is the current iterator in region mode?
  bool isRegionMode() const { return Node.getInt() != ItBB; }

  // Get the immediate successor. This function may return a Basic Block
  // RegionNode or a subregion RegionNode.
  RegionNode* getISucc(BasicBlock* BB) const {
    RegionNode *succ;
    succ = getNode()->getParent()->getNode(BB);
    assert(succ && "BB not in Region or entered subregion!");
    return succ;
  }

  // getRegionSucc - Return the successor basic block of a SubRegion RegionNode.
  inline BasicBlock* getRegionSucc() const {
    assert(Node.getInt() == ItRgBegin && "Cannot get the region successor!");
    return getNode()->template getNodeAs<Region>()->getExit();
  }

  // isExit - Is this the exit BB of the Region?
  inline bool isExit(BasicBlock* BB) const {
    return getNode()->getParent()->getExit() == BB;
  }
public:
  typedef RNSuccIterator<NodeType> Self;

  typedef typename super::pointer pointer;

  /// @brief Create begin iterator of a RegionNode.
  inline RNSuccIterator(NodeType* node)
    : Node(node, node->isSubRegion() ? ItRgBegin : ItBB),
    BItor(succ_begin(node->getEntry())) {


    // Skip the exit block
    if (!isRegionMode())
      while (succ_end(node->getEntry()) != BItor && isExit(*BItor))
        ++BItor;

    if (isRegionMode() && isExit(getRegionSucc()))
      advanceRegionSucc();
  }

  /// @brief Create an end iterator.
  inline RNSuccIterator(NodeType* node, bool)
    : Node(node, node->isSubRegion() ? ItRgEnd : ItBB),
    BItor(succ_end(node->getEntry())) {}

  inline bool operator==(const Self& x) const {
    assert(isRegionMode() == x.isRegionMode() && "Broken iterator!");
    if (isRegionMode())
      return Node.getInt() == x.Node.getInt();
    else
      return BItor == x.BItor;
  }

  inline bool operator!=(const Self& x) const { return !operator==(x); }

  inline pointer operator*() const {
    BasicBlock* BB = isRegionMode() ? getRegionSucc() : *BItor;
    assert(!isExit(BB) && "Iterator out of range!");
    return getISucc(BB);
  }

  inline Self& operator++() {
    if(isRegionMode()) {
      // The Region only has 1 successor.
      advanceRegionSucc();
    } else {
      // Skip the exit.
      do
        ++BItor;
      while (BItor != succ_end(getNode()->getEntry())
          && isExit(*BItor));
    }
    return *this;
  }

  inline Self operator++(int) {
    Self tmp = *this;
    ++*this;
    return tmp;
  }

  inline const Self &operator=(const Self &I) {
    if (this != &I) {
      assert(getNode()->getParent() == I.getNode()->getParent()
             && "Cannot assign iterators of two different regions!");
      Node = I.Node;
      BItor = I.BItor;
    }
    return *this;
  }
};


//===----------------------------------------------------------------------===//
/// @brief Flat RegionNode iterator.
///
/// The Flat Region iterator will iterate over all BasicBlock RegionNodes that
/// are contained in the Region and its subregions. This is close to a virtual
/// control flow graph of the Region.
template<class NodeType>
class RNSuccIterator<FlatIt<NodeType> >
  : public std::iterator<std::forward_iterator_tag, NodeType, ptrdiff_t>
{
  typedef std::iterator<std::forward_iterator_tag, NodeType, ptrdiff_t> super;
  NodeType* Node;
  succ_iterator Itor;

public:
  typedef RNSuccIterator<FlatIt<NodeType> > Self;
  typedef typename super::pointer pointer;

  /// @brief Create the iterator from a RegionNode.
  ///
  /// Note that the incoming node must be a bb node, otherwise it will trigger
  /// an assertion when we try to get a BasicBlock.
  inline RNSuccIterator(NodeType* node) : Node(node),
    Itor(succ_begin(node->getEntry())) {
      assert(!Node->isSubRegion()
             && "Subregion node not allowed in flat iterating mode!");
      assert(Node->getParent() && "A BB node must have a parent!");

      // Skip the exit block of the iterating region.
      while (succ_end(Node->getEntry()) != Itor
          && Node->getParent()->getExit() == *Itor)
        ++Itor;
  }
  /// @brief Create an end iterator
  inline RNSuccIterator(NodeType* node, bool) : Node(node),
    Itor(succ_end(node->getEntry())) {
      assert(!Node->isSubRegion()
             && "Subregion node not allowed in flat iterating mode!");
  }

  inline bool operator==(const Self& x) const {
    assert(Node->getParent() == x.Node->getParent()
           && "Cannot compare iterators of different regions!");

    return Itor == x.Itor && Node == x.Node;
  }

  inline bool operator!=(const Self& x) const { return !operator==(x); }

  inline pointer operator*() const {
    BasicBlock* BB = *Itor;

    // Get the iterating region.
    Region* Parent = Node->getParent();

    // The only case that the successor reaches out of the region is it reaches
    // the exit of the region.
    assert(Parent->getExit() != BB && "iterator out of range!");

    return Parent->getBBNode(BB);
  }

  inline Self& operator++() {
    // Skip the exit block of the iterating region.
    do
      ++Itor;
    while (Itor != succ_end(Node->getEntry())
        && Node->getParent()->getExit() == *Itor);

    return *this;
  }

  inline Self operator++(int) {
    Self tmp = *this;
    ++*this;
    return tmp;
  }

  inline const Self &operator=(const Self &I) {
    if (this != &I) {
      assert(Node->getParent() == I.Node->getParent()
             && "Cannot assign iterators to two different regions!");
      Node = I.Node;
      Itor = I.Itor;
    }
    return *this;
  }
};

template<class NodeType>
inline RNSuccIterator<NodeType> succ_begin(NodeType* Node) {
  return RNSuccIterator<NodeType>(Node);
}

template<class NodeType>
inline RNSuccIterator<NodeType> succ_end(NodeType* Node) {
  return RNSuccIterator<NodeType>(Node, true);
}

//===--------------------------------------------------------------------===//
// RegionNode GraphTraits specialization so the bbs in the region can be
// iterate by generic graph iterators.
//
// NodeT can either be region node or const region node, otherwise child_begin
// and child_end fail.

#define RegionNodeGraphTraits(NodeT) \
  template<> struct GraphTraits<NodeT*> { \
  typedef NodeT NodeType; \
  typedef RNSuccIterator<NodeType> ChildIteratorType; \
  static NodeType *getEntryNode(NodeType* N) { return N; } \
  static inline ChildIteratorType child_begin(NodeType *N) { \
    return RNSuccIterator<NodeType>(N); \
  } \
  static inline ChildIteratorType child_end(NodeType *N) { \
    return RNSuccIterator<NodeType>(N, true); \
  } \
}; \
template<> struct GraphTraits<FlatIt<NodeT*> > { \
  typedef NodeT NodeType; \
  typedef RNSuccIterator<FlatIt<NodeT> > ChildIteratorType; \
  static NodeType *getEntryNode(NodeType* N) { return N; } \
  static inline ChildIteratorType child_begin(NodeType *N) { \
    return RNSuccIterator<FlatIt<NodeType> >(N); \
  } \
  static inline ChildIteratorType child_end(NodeType *N) { \
    return RNSuccIterator<FlatIt<NodeType> >(N, true); \
  } \
}

#define RegionGraphTraits(RegionT, NodeT) \
template<> struct GraphTraits<RegionT*> \
  : public GraphTraits<NodeT*> { \
  typedef df_iterator<NodeType*> nodes_iterator; \
  static NodeType *getEntryNode(RegionT* R) { \
    return R->getNode(R->getEntry()); \
  } \
  static nodes_iterator nodes_begin(RegionT* R) { \
    return nodes_iterator::begin(getEntryNode(R)); \
  } \
  static nodes_iterator nodes_end(RegionT* R) { \
    return nodes_iterator::end(getEntryNode(R)); \
  } \
}; \
template<> struct GraphTraits<FlatIt<RegionT*> > \
  : public GraphTraits<FlatIt<NodeT*> > { \
  typedef df_iterator<NodeType*, SmallPtrSet<NodeType*, 8>, false, \
  GraphTraits<FlatIt<NodeType*> > > nodes_iterator; \
  static NodeType *getEntryNode(RegionT* R) { \
    return R->getBBNode(R->getEntry()); \
  } \
  static nodes_iterator nodes_begin(RegionT* R) { \
    return nodes_iterator::begin(getEntryNode(R)); \
  } \
  static nodes_iterator nodes_end(RegionT* R) { \
    return nodes_iterator::end(getEntryNode(R)); \
  } \
}

RegionNodeGraphTraits(RegionNode);
RegionNodeGraphTraits(const RegionNode);

RegionGraphTraits(Region, RegionNode);
RegionGraphTraits(const Region, const RegionNode);

template <> struct GraphTraits<RegionInfo*>
  : public GraphTraits<FlatIt<RegionNode*> > {
  typedef df_iterator<NodeType*, SmallPtrSet<NodeType*, 8>, false,
                      GraphTraits<FlatIt<NodeType*> > > nodes_iterator;

  static NodeType *getEntryNode(RegionInfo *RI) {
    return GraphTraits<FlatIt<Region*> >::getEntryNode(RI->getTopLevelRegion());
  }
  static nodes_iterator nodes_begin(RegionInfo* RI) {
    return nodes_iterator::begin(getEntryNode(RI));
  }
  static nodes_iterator nodes_end(RegionInfo *RI) {
    return nodes_iterator::end(getEntryNode(RI));
  }
};

} // End namespace llvm

#endif
