//===- llvm/Analysis/LoopInfo.h - Natural Loop Calculator -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopInfo class that is used to identify natural loops
// and determine the loop depth of various nodes of the CFG.  Note that natural
// loops may actually be several loops that share the same header node.
//
// This analysis calculates the nesting structure of loops in a function.  For
// each natural loop identified, this analysis identifies natural loops
// contained entirely within the loop and the basic blocks the make up the loop.
//
// It can calculate on the fly various bits of information, for example:
//
//  * whether there is a preheader for the loop
//  * the number of back edges to the header
//  * whether or not a particular block branches out of the loop
//  * the successor blocks of the loop
//  * the loop depth
//  * the trip count
//  * etc...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOP_INFO_H
#define LLVM_ANALYSIS_LOOP_INFO_H

#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Streams.h"
#include <algorithm>
#include <ostream>

namespace llvm {

template<typename T>
static void RemoveFromVector(std::vector<T*> &V, T *N) {
  typename std::vector<T*>::iterator I = std::find(V.begin(), V.end(), N);
  assert(I != V.end() && "N is not in this list!");
  V.erase(I);
}

class DominatorTree;
class LoopInfo;
template<class N> class LoopInfoBase;
template<class N> class LoopBase;

typedef LoopBase<BasicBlock> Loop;

//===----------------------------------------------------------------------===//
/// LoopBase class - Instances of this class are used to represent loops that
/// are detected in the flow graph
///
template<class BlockT>
class LoopBase {
  LoopBase<BlockT> *ParentLoop;
  // SubLoops - Loops contained entirely within this one.
  std::vector<LoopBase<BlockT>*> SubLoops;

  // Blocks - The list of blocks in this loop.  First entry is the header node.
  std::vector<BlockT*> Blocks;

  LoopBase(const LoopBase<BlockT> &);                  // DO NOT IMPLEMENT
  const LoopBase<BlockT>&operator=(const LoopBase<BlockT> &);// DO NOT IMPLEMENT
public:
  /// Loop ctor - This creates an empty loop.
  LoopBase() : ParentLoop(0) {}
  ~LoopBase() {
    for (size_t i = 0, e = SubLoops.size(); i != e; ++i)
      delete SubLoops[i];
  }

  /// getLoopDepth - Return the nesting level of this loop.  An outer-most
  /// loop has depth 1, for consistency with loop depth values used for basic
  /// blocks, where depth 0 is used for blocks not inside any loops.
  unsigned getLoopDepth() const {
    unsigned D = 1;
    for (const LoopBase<BlockT> *CurLoop = ParentLoop; CurLoop;
         CurLoop = CurLoop->ParentLoop)
      ++D;
    return D;
  }
  BlockT *getHeader() const { return Blocks.front(); }
  LoopBase<BlockT> *getParentLoop() const { return ParentLoop; }

  /// contains - Return true if the specified basic block is in this loop
  ///
  bool contains(const BlockT *BB) const {
    return std::find(Blocks.begin(), Blocks.end(), BB) != Blocks.end();
  }

  /// iterator/begin/end - Return the loops contained entirely within this loop.
  ///
  const std::vector<LoopBase<BlockT>*> &getSubLoops() const { return SubLoops; }
  typedef typename std::vector<LoopBase<BlockT>*>::const_iterator iterator;
  iterator begin() const { return SubLoops.begin(); }
  iterator end() const { return SubLoops.end(); }
  bool empty() const { return SubLoops.empty(); }

  /// getBlocks - Get a list of the basic blocks which make up this loop.
  ///
  const std::vector<BlockT*> &getBlocks() const { return Blocks; }
  typedef typename std::vector<BlockT*>::const_iterator block_iterator;
  block_iterator block_begin() const { return Blocks.begin(); }
  block_iterator block_end() const { return Blocks.end(); }

  /// isLoopExit - True if terminator in the block can branch to another block
  /// that is outside of the current loop.
  ///
  bool isLoopExit(const BlockT *BB) const {
    typedef GraphTraits<BlockT*> BlockTraits;
    for (typename BlockTraits::ChildIteratorType SI =
         BlockTraits::child_begin(const_cast<BlockT*>(BB)),
         SE = BlockTraits::child_end(const_cast<BlockT*>(BB)); SI != SE; ++SI) {
      if (!contains(*SI))
        return true;
    }
    return false;
  }

  /// getNumBackEdges - Calculate the number of back edges to the loop header
  ///
  unsigned getNumBackEdges() const {
    unsigned NumBackEdges = 0;
    BlockT *H = getHeader();

    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    for (typename InvBlockTraits::ChildIteratorType I =
         InvBlockTraits::child_begin(const_cast<BlockT*>(H)),
         E = InvBlockTraits::child_end(const_cast<BlockT*>(H)); I != E; ++I)
      if (contains(*I))
        ++NumBackEdges;

    return NumBackEdges;
  }

  /// isLoopInvariant - Return true if the specified value is loop invariant
  ///
  inline bool isLoopInvariant(Value *V) const {
    if (Instruction *I = dyn_cast<Instruction>(V))
      return !contains(I->getParent());
    return true;  // All non-instructions are loop invariant
  }

  //===--------------------------------------------------------------------===//
  // APIs for simple analysis of the loop.
  //
  // Note that all of these methods can fail on general loops (ie, there may not
  // be a preheader, etc).  For best success, the loop simplification and
  // induction variable canonicalization pass should be used to normalize loops
  // for easy analysis.  These methods assume canonical loops.

  /// getExitingBlocks - Return all blocks inside the loop that have successors
  /// outside of the loop.  These are the blocks _inside of the current loop_
  /// which branch out.  The returned list is always unique.
  ///
  void getExitingBlocks(SmallVectorImpl<BlockT *> &ExitingBlocks) const {
    // Sort the blocks vector so that we can use binary search to do quick
    // lookups.
    SmallVector<BlockT*, 128> LoopBBs(block_begin(), block_end());
    std::sort(LoopBBs.begin(), LoopBBs.end());

    typedef GraphTraits<BlockT*> BlockTraits;
    for (typename std::vector<BlockT*>::const_iterator BI = Blocks.begin(),
         BE = Blocks.end(); BI != BE; ++BI)
      for (typename BlockTraits::ChildIteratorType I =
          BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
          I != E; ++I)
        if (!std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I)) {
          // Not in current loop? It must be an exit block.
          ExitingBlocks.push_back(*BI);
          break;
        }
  }

  /// getExitBlocks - Return all of the successor blocks of this loop.  These
  /// are the blocks _outside of the current loop_ which are branched to.
  ///
  void getExitBlocks(SmallVectorImpl<BlockT*> &ExitBlocks) const {
    // Sort the blocks vector so that we can use binary search to do quick
    // lookups.
    SmallVector<BlockT*, 128> LoopBBs(block_begin(), block_end());
    std::sort(LoopBBs.begin(), LoopBBs.end());

    typedef GraphTraits<BlockT*> BlockTraits;
    for (typename std::vector<BlockT*>::const_iterator BI = Blocks.begin(),
         BE = Blocks.end(); BI != BE; ++BI)
      for (typename BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
           I != E; ++I)
        if (!std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I))
          // Not in current loop? It must be an exit block.
          ExitBlocks.push_back(*I);
  }

  /// getUniqueExitBlocks - Return all unique successor blocks of this loop. 
  /// These are the blocks _outside of the current loop_ which are branched to.
  /// This assumes that loop is in canonical form.
  ///
  void getUniqueExitBlocks(SmallVectorImpl<BlockT*> &ExitBlocks) const {
    // Sort the blocks vector so that we can use binary search to do quick
    // lookups.
    SmallVector<BlockT*, 128> LoopBBs(block_begin(), block_end());
    std::sort(LoopBBs.begin(), LoopBBs.end());

    std::vector<BlockT*> switchExitBlocks;  

    for (typename std::vector<BlockT*>::const_iterator BI = Blocks.begin(),
         BE = Blocks.end(); BI != BE; ++BI) {

      BlockT *current = *BI;
      switchExitBlocks.clear();

      typedef GraphTraits<BlockT*> BlockTraits;
      typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
      for (typename BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
           I != E; ++I) {
        if (std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I))
      // If block is inside the loop then it is not a exit block.
          continue;
      
        typename InvBlockTraits::ChildIteratorType PI =
                                                InvBlockTraits::child_begin(*I);
        BlockT *firstPred = *PI;

        // If current basic block is this exit block's first predecessor
        // then only insert exit block in to the output ExitBlocks vector.
        // This ensures that same exit block is not inserted twice into
        // ExitBlocks vector.
        if (current != firstPred) 
          continue;

        // If a terminator has more then two successors, for example SwitchInst,
        // then it is possible that there are multiple edges from current block 
        // to one exit block. 
        if (std::distance(BlockTraits::child_begin(current),
                          BlockTraits::child_end(current)) <= 2) {
          ExitBlocks.push_back(*I);
          continue;
        }

        // In case of multiple edges from current block to exit block, collect
        // only one edge in ExitBlocks. Use switchExitBlocks to keep track of
        // duplicate edges.
        if (std::find(switchExitBlocks.begin(), switchExitBlocks.end(), *I) 
            == switchExitBlocks.end()) {
          switchExitBlocks.push_back(*I);
          ExitBlocks.push_back(*I);
        }
      }
    }
  }

  /// getLoopPreheader - If there is a preheader for this loop, return it.  A
  /// loop has a preheader if there is only one edge to the header of the loop
  /// from outside of the loop.  If this is the case, the block branching to the
  /// header of the loop is the preheader node.
  ///
  /// This method returns null if there is no preheader for the loop.
  ///
  BlockT *getLoopPreheader() const {
    // Keep track of nodes outside the loop branching to the header...
    BlockT *Out = 0;

    // Loop over the predecessors of the header node...
    BlockT *Header = getHeader();
    typedef GraphTraits<BlockT*> BlockTraits;
    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    for (typename InvBlockTraits::ChildIteratorType PI =
         InvBlockTraits::child_begin(Header),
         PE = InvBlockTraits::child_end(Header); PI != PE; ++PI)
      if (!contains(*PI)) {     // If the block is not in the loop...
        if (Out && Out != *PI)
          return 0;             // Multiple predecessors outside the loop
        Out = *PI;
      }

    // Make sure there is only one exit out of the preheader.
    assert(Out && "Header of loop has no predecessors from outside loop?");
    typename BlockTraits::ChildIteratorType SI = BlockTraits::child_begin(Out);
    ++SI;
    if (SI != BlockTraits::child_end(Out))
      return 0;  // Multiple exits from the block, must not be a preheader.

    // If there is exactly one preheader, return it.  If there was zero, then
    // Out is still null.
    return Out;
  }

  /// getLoopLatch - If there is a latch block for this loop, return it.  A
  /// latch block is the canonical backedge for a loop.  A loop header in normal
  /// form has two edges into it: one from a preheader and one from a latch
  /// block.
  BlockT *getLoopLatch() const {
    BlockT *Header = getHeader();
    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    typename InvBlockTraits::ChildIteratorType PI =
                                            InvBlockTraits::child_begin(Header);
    typename InvBlockTraits::ChildIteratorType PE =
                                              InvBlockTraits::child_end(Header);
    if (PI == PE) return 0;  // no preds?

    BlockT *Latch = 0;
    if (contains(*PI))
      Latch = *PI;
    ++PI;
    if (PI == PE) return 0;  // only one pred?

    if (contains(*PI)) {
      if (Latch) return 0;  // multiple backedges
      Latch = *PI;
    }
    ++PI;
    if (PI != PE) return 0;  // more than two preds

    return Latch;
  }
  
  /// getCanonicalInductionVariable - Check to see if the loop has a canonical
  /// induction variable: an integer recurrence that starts at 0 and increments
  /// by one each time through the loop.  If so, return the phi node that
  /// corresponds to it.
  ///
  inline PHINode *getCanonicalInductionVariable() const {
    BlockT *H = getHeader();

    BlockT *Incoming = 0, *Backedge = 0;
    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    typename InvBlockTraits::ChildIteratorType PI =
                                                 InvBlockTraits::child_begin(H);
    assert(PI != InvBlockTraits::child_end(H) &&
           "Loop must have at least one backedge!");
    Backedge = *PI++;
    if (PI == InvBlockTraits::child_end(H)) return 0;  // dead loop
    Incoming = *PI++;
    if (PI != InvBlockTraits::child_end(H)) return 0;  // multiple backedges?

    if (contains(Incoming)) {
      if (contains(Backedge))
        return 0;
      std::swap(Incoming, Backedge);
    } else if (!contains(Backedge))
      return 0;

    // Loop over all of the PHI nodes, looking for a canonical indvar.
    for (typename BlockT::iterator I = H->begin(); isa<PHINode>(I); ++I) {
      PHINode *PN = cast<PHINode>(I);
      if (ConstantInt *CI =
          dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Incoming)))
        if (CI->isNullValue())
          if (Instruction *Inc =
              dyn_cast<Instruction>(PN->getIncomingValueForBlock(Backedge)))
            if (Inc->getOpcode() == Instruction::Add &&
                Inc->getOperand(0) == PN)
              if (ConstantInt *CI = dyn_cast<ConstantInt>(Inc->getOperand(1)))
                if (CI->equalsInt(1))
                  return PN;
    }
    return 0;
  }

  /// getCanonicalInductionVariableIncrement - Return the LLVM value that holds
  /// the canonical induction variable value for the "next" iteration of the
  /// loop.  This always succeeds if getCanonicalInductionVariable succeeds.
  ///
  inline Instruction *getCanonicalInductionVariableIncrement() const {
    if (PHINode *PN = getCanonicalInductionVariable()) {
      bool P1InLoop = contains(PN->getIncomingBlock(1));
      return cast<Instruction>(PN->getIncomingValue(P1InLoop));
    }
    return 0;
  }

  /// getTripCount - Return a loop-invariant LLVM value indicating the number of
  /// times the loop will be executed.  Note that this means that the backedge
  /// of the loop executes N-1 times.  If the trip-count cannot be determined,
  /// this returns null.
  ///
  inline Value *getTripCount() const {
    // Canonical loops will end with a 'cmp ne I, V', where I is the incremented
    // canonical induction variable and V is the trip count of the loop.
    Instruction *Inc = getCanonicalInductionVariableIncrement();
    if (Inc == 0) return 0;
    PHINode *IV = cast<PHINode>(Inc->getOperand(0));

    BlockT *BackedgeBlock =
            IV->getIncomingBlock(contains(IV->getIncomingBlock(1)));

    if (BranchInst *BI = dyn_cast<BranchInst>(BackedgeBlock->getTerminator()))
      if (BI->isConditional()) {
        if (ICmpInst *ICI = dyn_cast<ICmpInst>(BI->getCondition())) {
          if (ICI->getOperand(0) == Inc) {
            if (BI->getSuccessor(0) == getHeader()) {
              if (ICI->getPredicate() == ICmpInst::ICMP_NE)
                return ICI->getOperand(1);
            } else if (ICI->getPredicate() == ICmpInst::ICMP_EQ) {
              return ICI->getOperand(1);
            }
          }
        }
      }

    return 0;
  }
  
  /// getSmallConstantTripCount - Returns the trip count of this loop as a
  /// normal unsigned value, if possible. Returns 0 if the trip count is unknown
  /// of not constant. Will also return 0 if the trip count is very large 
  /// (>= 2^32)
  inline unsigned getSmallConstantTripCount() const {
    Value* TripCount = this->getTripCount();
    if (TripCount) {
      if (ConstantInt *TripCountC = dyn_cast<ConstantInt>(TripCount)) {
        // Guard against huge trip counts.
        if (TripCountC->getValue().getActiveBits() <= 32) {
          return (unsigned)TripCountC->getZExtValue();
        }
      }
    }
    return 0;
  }

  /// getSmallConstantTripMultiple - Returns the largest constant divisor of the
  /// trip count of this loop as a normal unsigned value, if possible. This
  /// means that the actual trip count is always a multiple of the returned
  /// value (don't forget the trip count could very well be zero as well!).
  ///
  /// Returns 1 if the trip count is unknown or not guaranteed to be the
  /// multiple of a constant (which is also the case if the trip count is simply
  /// constant, use getSmallConstantTripCount for that case), Will also return 1
  /// if the trip count is very large (>= 2^32).
  inline unsigned getSmallConstantTripMultiple() const {
    Value* TripCount = this->getTripCount();
    // This will hold the ConstantInt result, if any
    ConstantInt *Result = NULL;
    if (TripCount) {
      // See if the trip count is constant itself
      Result = dyn_cast<ConstantInt>(TripCount);
      // if not, see if it is a multiplication
      if (!Result)
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TripCount)) {
          switch (BO->getOpcode()) {
          case BinaryOperator::Mul:
            Result = dyn_cast<ConstantInt>(BO->getOperand(1));
            break;
          default: 
            break;
          }
        }
    }
    // Guard against huge trip counts.
    if (Result && Result->getValue().getActiveBits() <= 32) {
      return (unsigned)Result->getZExtValue();
    } else {
      return 1;
    }
  }
  
  /// isLCSSAForm - Return true if the Loop is in LCSSA form
  inline bool isLCSSAForm() const {
    // Sort the blocks vector so that we can use binary search to do quick
    // lookups.
    SmallPtrSet<BlockT*, 16> LoopBBs(block_begin(), block_end());

    for (block_iterator BI = block_begin(), E = block_end(); BI != E; ++BI) {
      BlockT *BB = *BI;
      for (typename BlockT::iterator I = BB->begin(), E = BB->end(); I != E;++I)
        for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
             ++UI) {
          BlockT *UserBB = cast<Instruction>(*UI)->getParent();
          if (PHINode *P = dyn_cast<PHINode>(*UI)) {
            UserBB = P->getIncomingBlock(UI);
          }

          // Check the current block, as a fast-path.  Most values are used in
          // the same block they are defined in.
          if (UserBB != BB && !LoopBBs.count(UserBB))
            return false;
        }
    }

    return true;
  }

  //===--------------------------------------------------------------------===//
  // APIs for updating loop information after changing the CFG
  //

  /// addBasicBlockToLoop - This method is used by other analyses to update loop
  /// information.  NewBB is set to be a new member of the current loop.
  /// Because of this, it is added as a member of all parent loops, and is added
  /// to the specified LoopInfo object as being in the current basic block.  It
  /// is not valid to replace the loop header with this method.
  ///
  void addBasicBlockToLoop(BlockT *NewBB, LoopInfoBase<BlockT> &LI);

  /// replaceChildLoopWith - This is used when splitting loops up.  It replaces
  /// the OldChild entry in our children list with NewChild, and updates the
  /// parent pointer of OldChild to be null and the NewChild to be this loop.
  /// This updates the loop depth of the new child.
  void replaceChildLoopWith(LoopBase<BlockT> *OldChild,
                            LoopBase<BlockT> *NewChild) {
    assert(OldChild->ParentLoop == this && "This loop is already broken!");
    assert(NewChild->ParentLoop == 0 && "NewChild already has a parent!");
    typename std::vector<LoopBase<BlockT>*>::iterator I =
                          std::find(SubLoops.begin(), SubLoops.end(), OldChild);
    assert(I != SubLoops.end() && "OldChild not in loop!");
    *I = NewChild;
    OldChild->ParentLoop = 0;
    NewChild->ParentLoop = this;
  }

  /// addChildLoop - Add the specified loop to be a child of this loop.  This
  /// updates the loop depth of the new child.
  ///
  void addChildLoop(LoopBase<BlockT> *NewChild) {
    assert(NewChild->ParentLoop == 0 && "NewChild already has a parent!");
    NewChild->ParentLoop = this;
    SubLoops.push_back(NewChild);
  }

  /// removeChildLoop - This removes the specified child from being a subloop of
  /// this loop.  The loop is not deleted, as it will presumably be inserted
  /// into another loop.
  LoopBase<BlockT> *removeChildLoop(iterator I) {
    assert(I != SubLoops.end() && "Cannot remove end iterator!");
    LoopBase<BlockT> *Child = *I;
    assert(Child->ParentLoop == this && "Child is not a child of this loop!");
    SubLoops.erase(SubLoops.begin()+(I-begin()));
    Child->ParentLoop = 0;
    return Child;
  }

  /// addBlockEntry - This adds a basic block directly to the basic block list.
  /// This should only be used by transformations that create new loops.  Other
  /// transformations should use addBasicBlockToLoop.
  void addBlockEntry(BlockT *BB) {
    Blocks.push_back(BB);
  }

  /// moveToHeader - This method is used to move BB (which must be part of this
  /// loop) to be the loop header of the loop (the block that dominates all
  /// others).
  void moveToHeader(BlockT *BB) {
    if (Blocks[0] == BB) return;
    for (unsigned i = 0; ; ++i) {
      assert(i != Blocks.size() && "Loop does not contain BB!");
      if (Blocks[i] == BB) {
        Blocks[i] = Blocks[0];
        Blocks[0] = BB;
        return;
      }
    }
  }

  /// removeBlockFromLoop - This removes the specified basic block from the
  /// current loop, updating the Blocks as appropriate.  This does not update
  /// the mapping in the LoopInfo class.
  void removeBlockFromLoop(BlockT *BB) {
    RemoveFromVector(Blocks, BB);
  }

  /// verifyLoop - Verify loop structure
  void verifyLoop() const {
#ifndef NDEBUG
    assert (getHeader() && "Loop header is missing");
    assert (getLoopPreheader() && "Loop preheader is missing");
    assert (getLoopLatch() && "Loop latch is missing");
    for (typename std::vector<LoopBase<BlockT>*>::const_iterator I =
         SubLoops.begin(), E = SubLoops.end(); I != E; ++I)
      (*I)->verifyLoop();
#endif
  }

  void print(std::ostream &OS, unsigned Depth = 0) const {
    OS << std::string(Depth*2, ' ') << "Loop Containing: ";

    for (unsigned i = 0; i < getBlocks().size(); ++i) {
      if (i) OS << ",";
      WriteAsOperand(OS, getBlocks()[i], false);
    }
    OS << "\n";

    for (iterator I = begin(), E = end(); I != E; ++I)
      (*I)->print(OS, Depth+2);
  }
  
  void print(std::ostream *O, unsigned Depth = 0) const {
    if (O) print(*O, Depth);
  }
  
  void dump() const {
    print(cerr);
  }
  
private:
  friend class LoopInfoBase<BlockT>;
  explicit LoopBase(BlockT *BB) : ParentLoop(0) {
    Blocks.push_back(BB);
  }
};


//===----------------------------------------------------------------------===//
/// LoopInfo - This class builds and contains all of the top level loop
/// structures in the specified function.
///

template<class BlockT>
class LoopInfoBase {
  // BBMap - Mapping of basic blocks to the inner most loop they occur in
  std::map<BlockT*, LoopBase<BlockT>*> BBMap;
  std::vector<LoopBase<BlockT>*> TopLevelLoops;
  friend class LoopBase<BlockT>;
  
public:
  LoopInfoBase() { }
  ~LoopInfoBase() { releaseMemory(); }
  
  void releaseMemory() {
    for (typename std::vector<LoopBase<BlockT>* >::iterator I =
         TopLevelLoops.begin(), E = TopLevelLoops.end(); I != E; ++I)
      delete *I;   // Delete all of the loops...

    BBMap.clear();                           // Reset internal state of analysis
    TopLevelLoops.clear();
  }
  
  /// iterator/begin/end - The interface to the top-level loops in the current
  /// function.
  ///
  typedef typename std::vector<LoopBase<BlockT>*>::const_iterator iterator;
  iterator begin() const { return TopLevelLoops.begin(); }
  iterator end() const { return TopLevelLoops.end(); }
  bool empty() const { return TopLevelLoops.empty(); }
  
  /// getLoopFor - Return the inner most loop that BB lives in.  If a basic
  /// block is in no loop (for example the entry node), null is returned.
  ///
  LoopBase<BlockT> *getLoopFor(const BlockT *BB) const {
    typename std::map<BlockT *, LoopBase<BlockT>*>::const_iterator I=
      BBMap.find(const_cast<BlockT*>(BB));
    return I != BBMap.end() ? I->second : 0;
  }
  
  /// operator[] - same as getLoopFor...
  ///
  const LoopBase<BlockT> *operator[](const BlockT *BB) const {
    return getLoopFor(BB);
  }
  
  /// getLoopDepth - Return the loop nesting level of the specified block.  A
  /// depth of 0 means the block is not inside any loop.
  ///
  unsigned getLoopDepth(const BlockT *BB) const {
    const LoopBase<BlockT> *L = getLoopFor(BB);
    return L ? L->getLoopDepth() : 0;
  }

  // isLoopHeader - True if the block is a loop header node
  bool isLoopHeader(BlockT *BB) const {
    const LoopBase<BlockT> *L = getLoopFor(BB);
    return L && L->getHeader() == BB;
  }
  
  /// removeLoop - This removes the specified top-level loop from this loop info
  /// object.  The loop is not deleted, as it will presumably be inserted into
  /// another loop.
  LoopBase<BlockT> *removeLoop(iterator I) {
    assert(I != end() && "Cannot remove end iterator!");
    LoopBase<BlockT> *L = *I;
    assert(L->getParentLoop() == 0 && "Not a top-level loop!");
    TopLevelLoops.erase(TopLevelLoops.begin() + (I-begin()));
    return L;
  }
  
  /// changeLoopFor - Change the top-level loop that contains BB to the
  /// specified loop.  This should be used by transformations that restructure
  /// the loop hierarchy tree.
  void changeLoopFor(BlockT *BB, LoopBase<BlockT> *L) {
    LoopBase<BlockT> *&OldLoop = BBMap[BB];
    assert(OldLoop && "Block not in a loop yet!");
    OldLoop = L;
  }
  
  /// changeTopLevelLoop - Replace the specified loop in the top-level loops
  /// list with the indicated loop.
  void changeTopLevelLoop(LoopBase<BlockT> *OldLoop,
                          LoopBase<BlockT> *NewLoop) {
    typename std::vector<LoopBase<BlockT>*>::iterator I =
                 std::find(TopLevelLoops.begin(), TopLevelLoops.end(), OldLoop);
    assert(I != TopLevelLoops.end() && "Old loop not at top level!");
    *I = NewLoop;
    assert(NewLoop->ParentLoop == 0 && OldLoop->ParentLoop == 0 &&
           "Loops already embedded into a subloop!");
  }
  
  /// addTopLevelLoop - This adds the specified loop to the collection of
  /// top-level loops.
  void addTopLevelLoop(LoopBase<BlockT> *New) {
    assert(New->getParentLoop() == 0 && "Loop already in subloop!");
    TopLevelLoops.push_back(New);
  }
  
  /// removeBlock - This method completely removes BB from all data structures,
  /// including all of the Loop objects it is nested in and our mapping from
  /// BasicBlocks to loops.
  void removeBlock(BlockT *BB) {
    typename std::map<BlockT *, LoopBase<BlockT>*>::iterator I = BBMap.find(BB);
    if (I != BBMap.end()) {
      for (LoopBase<BlockT> *L = I->second; L; L = L->getParentLoop())
        L->removeBlockFromLoop(BB);

      BBMap.erase(I);
    }
  }
  
  // Internals
  
  static bool isNotAlreadyContainedIn(const LoopBase<BlockT> *SubLoop,
                                      const LoopBase<BlockT> *ParentLoop) {
    if (SubLoop == 0) return true;
    if (SubLoop == ParentLoop) return false;
    return isNotAlreadyContainedIn(SubLoop->getParentLoop(), ParentLoop);
  }
  
  void Calculate(DominatorTreeBase<BlockT> &DT) {
    BlockT *RootNode = DT.getRootNode()->getBlock();

    for (df_iterator<BlockT*> NI = df_begin(RootNode),
           NE = df_end(RootNode); NI != NE; ++NI)
      if (LoopBase<BlockT> *L = ConsiderForLoop(*NI, DT))
        TopLevelLoops.push_back(L);
  }
  
  LoopBase<BlockT> *ConsiderForLoop(BlockT *BB, DominatorTreeBase<BlockT> &DT) {
    if (BBMap.find(BB) != BBMap.end()) return 0;// Haven't processed this node?

    std::vector<BlockT *> TodoStack;

    // Scan the predecessors of BB, checking to see if BB dominates any of
    // them.  This identifies backedges which target this node...
    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    for (typename InvBlockTraits::ChildIteratorType I =
         InvBlockTraits::child_begin(BB), E = InvBlockTraits::child_end(BB);
         I != E; ++I)
      if (DT.dominates(BB, *I))   // If BB dominates it's predecessor...
        TodoStack.push_back(*I);

    if (TodoStack.empty()) return 0;  // No backedges to this block...

    // Create a new loop to represent this basic block...
    LoopBase<BlockT> *L = new LoopBase<BlockT>(BB);
    BBMap[BB] = L;

    BlockT *EntryBlock = BB->getParent()->begin();

    while (!TodoStack.empty()) {  // Process all the nodes in the loop
      BlockT *X = TodoStack.back();
      TodoStack.pop_back();

      if (!L->contains(X) &&         // As of yet unprocessed??
          DT.dominates(EntryBlock, X)) {   // X is reachable from entry block?
        // Check to see if this block already belongs to a loop.  If this occurs
        // then we have a case where a loop that is supposed to be a child of
        // the current loop was processed before the current loop.  When this
        // occurs, this child loop gets added to a part of the current loop,
        // making it a sibling to the current loop.  We have to reparent this
        // loop.
        if (LoopBase<BlockT> *SubLoop =
            const_cast<LoopBase<BlockT>*>(getLoopFor(X)))
          if (SubLoop->getHeader() == X && isNotAlreadyContainedIn(SubLoop, L)){
            // Remove the subloop from it's current parent...
            assert(SubLoop->ParentLoop && SubLoop->ParentLoop != L);
            LoopBase<BlockT> *SLP = SubLoop->ParentLoop;  // SubLoopParent
            typename std::vector<LoopBase<BlockT>*>::iterator I =
              std::find(SLP->SubLoops.begin(), SLP->SubLoops.end(), SubLoop);
            assert(I != SLP->SubLoops.end() &&"SubLoop not a child of parent?");
            SLP->SubLoops.erase(I);   // Remove from parent...

            // Add the subloop to THIS loop...
            SubLoop->ParentLoop = L;
            L->SubLoops.push_back(SubLoop);
          }

        // Normal case, add the block to our loop...
        L->Blocks.push_back(X);
        
        typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
        
        // Add all of the predecessors of X to the end of the work stack...
        TodoStack.insert(TodoStack.end(), InvBlockTraits::child_begin(X),
                         InvBlockTraits::child_end(X));
      }
    }

    // If there are any loops nested within this loop, create them now!
    for (typename std::vector<BlockT*>::iterator I = L->Blocks.begin(),
         E = L->Blocks.end(); I != E; ++I)
      if (LoopBase<BlockT> *NewLoop = ConsiderForLoop(*I, DT)) {
        L->SubLoops.push_back(NewLoop);
        NewLoop->ParentLoop = L;
      }

    // Add the basic blocks that comprise this loop to the BBMap so that this
    // loop can be found for them.
    //
    for (typename std::vector<BlockT*>::iterator I = L->Blocks.begin(),
           E = L->Blocks.end(); I != E; ++I) {
      typename std::map<BlockT*, LoopBase<BlockT>*>::iterator BBMI =
                                                          BBMap.find(*I);
      if (BBMI == BBMap.end())                       // Not in map yet...
        BBMap.insert(BBMI, std::make_pair(*I, L));   // Must be at this level
    }

    // Now that we have a list of all of the child loops of this loop, check to
    // see if any of them should actually be nested inside of each other.  We
    // can accidentally pull loops our of their parents, so we must make sure to
    // organize the loop nests correctly now.
    {
      std::map<BlockT*, LoopBase<BlockT>*> ContainingLoops;
      for (unsigned i = 0; i != L->SubLoops.size(); ++i) {
        LoopBase<BlockT> *Child = L->SubLoops[i];
        assert(Child->getParentLoop() == L && "Not proper child loop?");

        if (LoopBase<BlockT> *ContainingLoop =
                                          ContainingLoops[Child->getHeader()]) {
          // If there is already a loop which contains this loop, move this loop
          // into the containing loop.
          MoveSiblingLoopInto(Child, ContainingLoop);
          --i;  // The loop got removed from the SubLoops list.
        } else {
          // This is currently considered to be a top-level loop.  Check to see
          // if any of the contained blocks are loop headers for subloops we
          // have already processed.
          for (unsigned b = 0, e = Child->Blocks.size(); b != e; ++b) {
            LoopBase<BlockT> *&BlockLoop = ContainingLoops[Child->Blocks[b]];
            if (BlockLoop == 0) {   // Child block not processed yet...
              BlockLoop = Child;
            } else if (BlockLoop != Child) {
              LoopBase<BlockT> *SubLoop = BlockLoop;
              // Reparent all of the blocks which used to belong to BlockLoops
              for (unsigned j = 0, e = SubLoop->Blocks.size(); j != e; ++j)
                ContainingLoops[SubLoop->Blocks[j]] = Child;

              // There is already a loop which contains this block, that means
              // that we should reparent the loop which the block is currently
              // considered to belong to to be a child of this loop.
              MoveSiblingLoopInto(SubLoop, Child);
              --i;  // We just shrunk the SubLoops list.
            }
          }
        }
      }
    }

    return L;
  }
  
  /// MoveSiblingLoopInto - This method moves the NewChild loop to live inside
  /// of the NewParent Loop, instead of being a sibling of it.
  void MoveSiblingLoopInto(LoopBase<BlockT> *NewChild,
                           LoopBase<BlockT> *NewParent) {
    LoopBase<BlockT> *OldParent = NewChild->getParentLoop();
    assert(OldParent && OldParent == NewParent->getParentLoop() &&
           NewChild != NewParent && "Not sibling loops!");

    // Remove NewChild from being a child of OldParent
    typename std::vector<LoopBase<BlockT>*>::iterator I =
      std::find(OldParent->SubLoops.begin(), OldParent->SubLoops.end(),
                NewChild);
    assert(I != OldParent->SubLoops.end() && "Parent fields incorrect??");
    OldParent->SubLoops.erase(I);   // Remove from parent's subloops list
    NewChild->ParentLoop = 0;

    InsertLoopInto(NewChild, NewParent);
  }
  
  /// InsertLoopInto - This inserts loop L into the specified parent loop.  If
  /// the parent loop contains a loop which should contain L, the loop gets
  /// inserted into L instead.
  void InsertLoopInto(LoopBase<BlockT> *L, LoopBase<BlockT> *Parent) {
    BlockT *LHeader = L->getHeader();
    assert(Parent->contains(LHeader) &&
           "This loop should not be inserted here!");

    // Check to see if it belongs in a child loop...
    for (unsigned i = 0, e = static_cast<unsigned>(Parent->SubLoops.size());
         i != e; ++i)
      if (Parent->SubLoops[i]->contains(LHeader)) {
        InsertLoopInto(L, Parent->SubLoops[i]);
        return;
      }

    // If not, insert it here!
    Parent->SubLoops.push_back(L);
    L->ParentLoop = Parent;
  }
  
  // Debugging
  
  void print(std::ostream &OS, const Module* ) const {
    for (unsigned i = 0; i < TopLevelLoops.size(); ++i)
      TopLevelLoops[i]->print(OS);
  #if 0
    for (std::map<BasicBlock*, Loop*>::const_iterator I = BBMap.begin(),
           E = BBMap.end(); I != E; ++I)
      OS << "BB '" << I->first->getName() << "' level = "
         << I->second->getLoopDepth() << "\n";
  #endif
  }
};

class LoopInfo : public FunctionPass {
  LoopInfoBase<BasicBlock>* LI;
  friend class LoopBase<BasicBlock>;
  
public:
  static char ID; // Pass identification, replacement for typeid

  LoopInfo() : FunctionPass(intptr_t(&ID)) {
    LI = new LoopInfoBase<BasicBlock>();
  }
  
  ~LoopInfo() { delete LI; }

  LoopInfoBase<BasicBlock>& getBase() { return *LI; }

  /// iterator/begin/end - The interface to the top-level loops in the current
  /// function.
  ///
  typedef std::vector<Loop*>::const_iterator iterator;
  inline iterator begin() const { return LI->begin(); }
  inline iterator end() const { return LI->end(); }
  bool empty() const { return LI->empty(); }

  /// getLoopFor - Return the inner most loop that BB lives in.  If a basic
  /// block is in no loop (for example the entry node), null is returned.
  ///
  inline Loop *getLoopFor(const BasicBlock *BB) const {
    return LI->getLoopFor(BB);
  }

  /// operator[] - same as getLoopFor...
  ///
  inline const Loop *operator[](const BasicBlock *BB) const {
    return LI->getLoopFor(BB);
  }

  /// getLoopDepth - Return the loop nesting level of the specified block.  A
  /// depth of 0 means the block is not inside any loop.
  ///
  inline unsigned getLoopDepth(const BasicBlock *BB) const {
    return LI->getLoopDepth(BB);
  }

  // isLoopHeader - True if the block is a loop header node
  inline bool isLoopHeader(BasicBlock *BB) const {
    return LI->isLoopHeader(BB);
  }

  /// runOnFunction - Calculate the natural loop information.
  ///
  virtual bool runOnFunction(Function &F);

  virtual void releaseMemory() { LI->releaseMemory(); }

  virtual void print(std::ostream &O, const Module* M = 0) const {
    if (O) LI->print(O, M);
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  /// removeLoop - This removes the specified top-level loop from this loop info
  /// object.  The loop is not deleted, as it will presumably be inserted into
  /// another loop.
  inline Loop *removeLoop(iterator I) { return LI->removeLoop(I); }

  /// changeLoopFor - Change the top-level loop that contains BB to the
  /// specified loop.  This should be used by transformations that restructure
  /// the loop hierarchy tree.
  inline void changeLoopFor(BasicBlock *BB, Loop *L) {
    LI->changeLoopFor(BB, L);
  }

  /// changeTopLevelLoop - Replace the specified loop in the top-level loops
  /// list with the indicated loop.
  inline void changeTopLevelLoop(Loop *OldLoop, Loop *NewLoop) {
    LI->changeTopLevelLoop(OldLoop, NewLoop);
  }

  /// addTopLevelLoop - This adds the specified loop to the collection of
  /// top-level loops.
  inline void addTopLevelLoop(Loop *New) {
    LI->addTopLevelLoop(New);
  }

  /// removeBlock - This method completely removes BB from all data structures,
  /// including all of the Loop objects it is nested in and our mapping from
  /// BasicBlocks to loops.
  void removeBlock(BasicBlock *BB) {
    LI->removeBlock(BB);
  }
};


// Allow clients to walk the list of nested loops...
template <> struct GraphTraits<const Loop*> {
  typedef const Loop NodeType;
  typedef std::vector<Loop*>::const_iterator ChildIteratorType;

  static NodeType *getEntryNode(const Loop *L) { return L; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->end();
  }
};

template <> struct GraphTraits<Loop*> {
  typedef Loop NodeType;
  typedef std::vector<Loop*>::const_iterator ChildIteratorType;

  static NodeType *getEntryNode(Loop *L) { return L; }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->end();
  }
};

template<class BlockT>
void LoopBase<BlockT>::addBasicBlockToLoop(BlockT *NewBB,
                                           LoopInfoBase<BlockT> &LIB) {
  assert((Blocks.empty() || LIB[getHeader()] == this) &&
         "Incorrect LI specified for this loop!");
  assert(NewBB && "Cannot add a null basic block to the loop!");
  assert(LIB[NewBB] == 0 && "BasicBlock already in the loop!");

  // Add the loop mapping to the LoopInfo object...
  LIB.BBMap[NewBB] = this;

  // Add the basic block to this loop and all parent loops...
  LoopBase<BlockT> *L = this;
  while (L) {
    L->Blocks.push_back(NewBB);
    L = L->getParentLoop();
  }
}

} // End llvm namespace

#endif
