//===- EarlyCSE.cpp - Simple and fast CSE pass ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a simple dominator tree walk that eliminates trivially
// redundant instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "early-cse"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Statistic.h"
#include <deque>
using namespace llvm;

STATISTIC(NumSimplify, "Number of instructions simplified or DCE'd");
STATISTIC(NumCSE,      "Number of instructions CSE'd");
STATISTIC(NumCSELoad,  "Number of load instructions CSE'd");
STATISTIC(NumCSECall,  "Number of call instructions CSE'd");
STATISTIC(NumDSE,      "Number of trivial dead stores removed");

static unsigned getHash(const void *V) {
  return DenseMapInfo<const void*>::getHashValue(V);
}

//===----------------------------------------------------------------------===//
// SimpleValue
//===----------------------------------------------------------------------===//

namespace {
  /// SimpleValue - Instances of this struct represent available values in the
  /// scoped hash table.
  struct SimpleValue {
    Instruction *Inst;

    SimpleValue(Instruction *I) : Inst(I) {
      assert((isSentinel() || canHandle(I)) && "Inst can't be handled!");
    }

    bool isSentinel() const {
      return Inst == DenseMapInfo<Instruction*>::getEmptyKey() ||
             Inst == DenseMapInfo<Instruction*>::getTombstoneKey();
    }

    static bool canHandle(Instruction *Inst) {
      // This can only handle non-void readnone functions.
      if (CallInst *CI = dyn_cast<CallInst>(Inst))
        return CI->doesNotAccessMemory() && !CI->getType()->isVoidTy();
      return isa<CastInst>(Inst) || isa<BinaryOperator>(Inst) ||
             isa<GetElementPtrInst>(Inst) || isa<CmpInst>(Inst) ||
             isa<SelectInst>(Inst) || isa<ExtractElementInst>(Inst) ||
             isa<InsertElementInst>(Inst) || isa<ShuffleVectorInst>(Inst) ||
             isa<ExtractValueInst>(Inst) || isa<InsertValueInst>(Inst);
    }
  };
}

namespace llvm {
// SimpleValue is POD.
template<> struct isPodLike<SimpleValue> {
  static const bool value = true;
};

template<> struct DenseMapInfo<SimpleValue> {
  static inline SimpleValue getEmptyKey() {
    return DenseMapInfo<Instruction*>::getEmptyKey();
  }
  static inline SimpleValue getTombstoneKey() {
    return DenseMapInfo<Instruction*>::getTombstoneKey();
  }
  static unsigned getHashValue(SimpleValue Val);
  static bool isEqual(SimpleValue LHS, SimpleValue RHS);
};
}

unsigned DenseMapInfo<SimpleValue>::getHashValue(SimpleValue Val) {
  Instruction *Inst = Val.Inst;
  // Hash in all of the operands as pointers.
  if (BinaryOperator* BinOp = dyn_cast<BinaryOperator>(Inst)) {
    Value *LHS = BinOp->getOperand(0);
    Value *RHS = BinOp->getOperand(1);
    if (BinOp->isCommutative() && BinOp->getOperand(0) > BinOp->getOperand(1))
      std::swap(LHS, RHS);

    if (isa<OverflowingBinaryOperator>(BinOp)) {
      // Hash the overflow behavior
      unsigned Overflow =
        BinOp->hasNoSignedWrap()   * OverflowingBinaryOperator::NoSignedWrap |
        BinOp->hasNoUnsignedWrap() * OverflowingBinaryOperator::NoUnsignedWrap;
      return hash_combine(BinOp->getOpcode(), Overflow, LHS, RHS);
    }

    return hash_combine(BinOp->getOpcode(), LHS, RHS);
  }

  if (CmpInst *CI = dyn_cast<CmpInst>(Inst)) {
    Value *LHS = CI->getOperand(0);
    Value *RHS = CI->getOperand(1);
    CmpInst::Predicate Pred = CI->getPredicate();
    if (Inst->getOperand(0) > Inst->getOperand(1)) {
      std::swap(LHS, RHS);
      Pred = CI->getSwappedPredicate();
    }
    return hash_combine(Inst->getOpcode(), Pred, LHS, RHS);
  }

  if (CastInst *CI = dyn_cast<CastInst>(Inst))
    return hash_combine(CI->getOpcode(), CI->getType(), CI->getOperand(0));

  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(Inst))
    return hash_combine(EVI->getOpcode(), EVI->getOperand(0),
                        hash_combine_range(EVI->idx_begin(), EVI->idx_end()));

  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(Inst))
    return hash_combine(IVI->getOpcode(), IVI->getOperand(0),
                        IVI->getOperand(1),
                        hash_combine_range(IVI->idx_begin(), IVI->idx_end()));

  assert((isa<CallInst>(Inst) || isa<BinaryOperator>(Inst) ||
          isa<GetElementPtrInst>(Inst) || isa<SelectInst>(Inst) ||
          isa<ExtractElementInst>(Inst) || isa<InsertElementInst>(Inst) ||
          isa<ShuffleVectorInst>(Inst)) && "Invalid/unknown instruction");

  // Mix in the opcode.
  return hash_combine(Inst->getOpcode(),
                      hash_combine_range(Inst->value_op_begin(),
                                         Inst->value_op_end()));
}

bool DenseMapInfo<SimpleValue>::isEqual(SimpleValue LHS, SimpleValue RHS) {
  Instruction *LHSI = LHS.Inst, *RHSI = RHS.Inst;

  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;

  if (LHSI->getOpcode() != RHSI->getOpcode()) return false;
  if (LHSI->isIdenticalTo(RHSI)) return true;

  // If we're not strictly identical, we still might be a commutable instruction
  if (BinaryOperator *LHSBinOp = dyn_cast<BinaryOperator>(LHSI)) {
    if (!LHSBinOp->isCommutative())
      return false;

    assert(isa<BinaryOperator>(RHSI)
           && "same opcode, but different instruction type?");
    BinaryOperator *RHSBinOp = cast<BinaryOperator>(RHSI);

    // Check overflow attributes
    if (isa<OverflowingBinaryOperator>(LHSBinOp)) {
      assert(isa<OverflowingBinaryOperator>(RHSBinOp)
             && "same opcode, but different operator type?");
      if (LHSBinOp->hasNoUnsignedWrap() != RHSBinOp->hasNoUnsignedWrap() ||
          LHSBinOp->hasNoSignedWrap() != RHSBinOp->hasNoSignedWrap())
        return false;
    }

    // Commuted equality
    return LHSBinOp->getOperand(0) == RHSBinOp->getOperand(1) &&
      LHSBinOp->getOperand(1) == RHSBinOp->getOperand(0);
  }
  if (CmpInst *LHSCmp = dyn_cast<CmpInst>(LHSI)) {
    assert(isa<CmpInst>(RHSI)
           && "same opcode, but different instruction type?");
    CmpInst *RHSCmp = cast<CmpInst>(RHSI);
    // Commuted equality
    return LHSCmp->getOperand(0) == RHSCmp->getOperand(1) &&
      LHSCmp->getOperand(1) == RHSCmp->getOperand(0) &&
      LHSCmp->getSwappedPredicate() == RHSCmp->getPredicate();
  }

  return false;
}

//===----------------------------------------------------------------------===//
// CallValue
//===----------------------------------------------------------------------===//

namespace {
  /// CallValue - Instances of this struct represent available call values in
  /// the scoped hash table.
  struct CallValue {
    Instruction *Inst;

    CallValue(Instruction *I) : Inst(I) {
      assert((isSentinel() || canHandle(I)) && "Inst can't be handled!");
    }

    bool isSentinel() const {
      return Inst == DenseMapInfo<Instruction*>::getEmptyKey() ||
             Inst == DenseMapInfo<Instruction*>::getTombstoneKey();
    }

    static bool canHandle(Instruction *Inst) {
      // Don't value number anything that returns void.
      if (Inst->getType()->isVoidTy())
        return false;

      CallInst *CI = dyn_cast<CallInst>(Inst);
      if (CI == 0 || !CI->onlyReadsMemory())
        return false;
      return true;
    }
  };
}

namespace llvm {
  // CallValue is POD.
  template<> struct isPodLike<CallValue> {
    static const bool value = true;
  };

  template<> struct DenseMapInfo<CallValue> {
    static inline CallValue getEmptyKey() {
      return DenseMapInfo<Instruction*>::getEmptyKey();
    }
    static inline CallValue getTombstoneKey() {
      return DenseMapInfo<Instruction*>::getTombstoneKey();
    }
    static unsigned getHashValue(CallValue Val);
    static bool isEqual(CallValue LHS, CallValue RHS);
  };
}
unsigned DenseMapInfo<CallValue>::getHashValue(CallValue Val) {
  Instruction *Inst = Val.Inst;
  // Hash in all of the operands as pointers.
  unsigned Res = 0;
  for (unsigned i = 0, e = Inst->getNumOperands(); i != e; ++i) {
    assert(!Inst->getOperand(i)->getType()->isMetadataTy() &&
           "Cannot value number calls with metadata operands");
    Res ^= getHash(Inst->getOperand(i)) << (i & 0xF);
  }

  // Mix in the opcode.
  return (Res << 1) ^ Inst->getOpcode();
}

bool DenseMapInfo<CallValue>::isEqual(CallValue LHS, CallValue RHS) {
  Instruction *LHSI = LHS.Inst, *RHSI = RHS.Inst;
  if (LHS.isSentinel() || RHS.isSentinel())
    return LHSI == RHSI;
  return LHSI->isIdenticalTo(RHSI);
}


//===----------------------------------------------------------------------===//
// EarlyCSE pass.
//===----------------------------------------------------------------------===//

namespace {

/// EarlyCSE - This pass does a simple depth-first walk over the dominator
/// tree, eliminating trivially redundant instructions and using instsimplify
/// to canonicalize things as it goes.  It is intended to be fast and catch
/// obvious cases so that instcombine and other passes are more effective.  It
/// is expected that a later pass of GVN will catch the interesting/hard
/// cases.
class EarlyCSE : public FunctionPass {
public:
  const DataLayout *TD;
  const TargetLibraryInfo *TLI;
  DominatorTree *DT;
  typedef RecyclingAllocator<BumpPtrAllocator,
                      ScopedHashTableVal<SimpleValue, Value*> > AllocatorTy;
  typedef ScopedHashTable<SimpleValue, Value*, DenseMapInfo<SimpleValue>,
                          AllocatorTy> ScopedHTType;

  /// AvailableValues - This scoped hash table contains the current values of
  /// all of our simple scalar expressions.  As we walk down the domtree, we
  /// look to see if instructions are in this: if so, we replace them with what
  /// we find, otherwise we insert them so that dominated values can succeed in
  /// their lookup.
  ScopedHTType *AvailableValues;

  /// AvailableLoads - This scoped hash table contains the current values
  /// of loads.  This allows us to get efficient access to dominating loads when
  /// we have a fully redundant load.  In addition to the most recent load, we
  /// keep track of a generation count of the read, which is compared against
  /// the current generation count.  The current generation count is
  /// incremented after every possibly writing memory operation, which ensures
  /// that we only CSE loads with other loads that have no intervening store.
  typedef RecyclingAllocator<BumpPtrAllocator,
    ScopedHashTableVal<Value*, std::pair<Value*, unsigned> > > LoadMapAllocator;
  typedef ScopedHashTable<Value*, std::pair<Value*, unsigned>,
                          DenseMapInfo<Value*>, LoadMapAllocator> LoadHTType;
  LoadHTType *AvailableLoads;

  /// AvailableCalls - This scoped hash table contains the current values
  /// of read-only call values.  It uses the same generation count as loads.
  typedef ScopedHashTable<CallValue, std::pair<Value*, unsigned> > CallHTType;
  CallHTType *AvailableCalls;

  /// CurrentGeneration - This is the current generation of the memory value.
  unsigned CurrentGeneration;

  static char ID;
  explicit EarlyCSE() : FunctionPass(ID) {
    initializeEarlyCSEPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F);

private:

  // NodeScope - almost a POD, but needs to call the constructors for the
  // scoped hash tables so that a new scope gets pushed on. These are RAII so
  // that the scope gets popped when the NodeScope is destroyed.
  class NodeScope {
   public:
    NodeScope(ScopedHTType *availableValues,
              LoadHTType *availableLoads,
              CallHTType *availableCalls) :
        Scope(*availableValues),
        LoadScope(*availableLoads),
        CallScope(*availableCalls) {}

   private:
    NodeScope(const NodeScope&) LLVM_DELETED_FUNCTION;
    void operator=(const NodeScope&) LLVM_DELETED_FUNCTION;

    ScopedHTType::ScopeTy Scope;
    LoadHTType::ScopeTy LoadScope;
    CallHTType::ScopeTy CallScope;
  };

  // StackNode - contains all the needed information to create a stack for
  // doing a depth first tranversal of the tree. This includes scopes for
  // values, loads, and calls as well as the generation. There is a child
  // iterator so that the children do not need to be store spearately.
  class StackNode {
   public:
    StackNode(ScopedHTType *availableValues,
              LoadHTType *availableLoads,
              CallHTType *availableCalls,
              unsigned cg, DomTreeNode *n,
              DomTreeNode::iterator child, DomTreeNode::iterator end) :
        CurrentGeneration(cg), ChildGeneration(cg), Node(n),
        ChildIter(child), EndIter(end),
        Scopes(availableValues, availableLoads, availableCalls),
        Processed(false) {}

    // Accessors.
    unsigned currentGeneration() { return CurrentGeneration; }
    unsigned childGeneration() { return ChildGeneration; }
    void childGeneration(unsigned generation) { ChildGeneration = generation; }
    DomTreeNode *node() { return Node; }
    DomTreeNode::iterator childIter() { return ChildIter; }
    DomTreeNode *nextChild() {
      DomTreeNode *child = *ChildIter;
      ++ChildIter;
      return child;
    }
    DomTreeNode::iterator end() { return EndIter; }
    bool isProcessed() { return Processed; }
    void process() { Processed = true; }

   private:
    StackNode(const StackNode&) LLVM_DELETED_FUNCTION;
    void operator=(const StackNode&) LLVM_DELETED_FUNCTION;

    // Members.
    unsigned CurrentGeneration;
    unsigned ChildGeneration;
    DomTreeNode *Node;
    DomTreeNode::iterator ChildIter;
    DomTreeNode::iterator EndIter;
    NodeScope Scopes;
    bool Processed;
  };

  bool processNode(DomTreeNode *Node);

  // This transformation requires dominator postdominator info
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<DominatorTree>();
    AU.addRequired<TargetLibraryInfo>();
    AU.setPreservesCFG();
  }
};
}

char EarlyCSE::ID = 0;

// createEarlyCSEPass - The public interface to this file.
FunctionPass *llvm::createEarlyCSEPass() {
  return new EarlyCSE();
}

INITIALIZE_PASS_BEGIN(EarlyCSE, "early-cse", "Early CSE", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(EarlyCSE, "early-cse", "Early CSE", false, false)

bool EarlyCSE::processNode(DomTreeNode *Node) {
  BasicBlock *BB = Node->getBlock();

  // If this block has a single predecessor, then the predecessor is the parent
  // of the domtree node and all of the live out memory values are still current
  // in this block.  If this block has multiple predecessors, then they could
  // have invalidated the live-out memory values of our parent value.  For now,
  // just be conservative and invalidate memory if this block has multiple
  // predecessors.
  if (BB->getSinglePredecessor() == 0)
    ++CurrentGeneration;

  /// LastStore - Keep track of the last non-volatile store that we saw... for
  /// as long as there in no instruction that reads memory.  If we see a store
  /// to the same location, we delete the dead store.  This zaps trivial dead
  /// stores which can occur in bitfield code among other things.
  StoreInst *LastStore = 0;

  bool Changed = false;

  // See if any instructions in the block can be eliminated.  If so, do it.  If
  // not, add them to AvailableValues.
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    Instruction *Inst = I++;

    // Dead instructions should just be removed.
    if (isInstructionTriviallyDead(Inst, TLI)) {
      DEBUG(dbgs() << "EarlyCSE DCE: " << *Inst << '\n');
      Inst->eraseFromParent();
      Changed = true;
      ++NumSimplify;
      continue;
    }

    // If the instruction can be simplified (e.g. X+0 = X) then replace it with
    // its simpler value.
    if (Value *V = SimplifyInstruction(Inst, TD, TLI, DT)) {
      DEBUG(dbgs() << "EarlyCSE Simplify: " << *Inst << "  to: " << *V << '\n');
      Inst->replaceAllUsesWith(V);
      Inst->eraseFromParent();
      Changed = true;
      ++NumSimplify;
      continue;
    }

    // If this is a simple instruction that we can value number, process it.
    if (SimpleValue::canHandle(Inst)) {
      // See if the instruction has an available value.  If so, use it.
      if (Value *V = AvailableValues->lookup(Inst)) {
        DEBUG(dbgs() << "EarlyCSE CSE: " << *Inst << "  to: " << *V << '\n');
        Inst->replaceAllUsesWith(V);
        Inst->eraseFromParent();
        Changed = true;
        ++NumCSE;
        continue;
      }

      // Otherwise, just remember that this value is available.
      AvailableValues->insert(Inst, Inst);
      continue;
    }

    // If this is a non-volatile load, process it.
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
      // Ignore volatile loads.
      if (!LI->isSimple()) {
        LastStore = 0;
        continue;
      }

      // If we have an available version of this load, and if it is the right
      // generation, replace this instruction.
      std::pair<Value*, unsigned> InVal =
        AvailableLoads->lookup(Inst->getOperand(0));
      if (InVal.first != 0 && InVal.second == CurrentGeneration) {
        DEBUG(dbgs() << "EarlyCSE CSE LOAD: " << *Inst << "  to: "
              << *InVal.first << '\n');
        if (!Inst->use_empty()) Inst->replaceAllUsesWith(InVal.first);
        Inst->eraseFromParent();
        Changed = true;
        ++NumCSELoad;
        continue;
      }

      // Otherwise, remember that we have this instruction.
      AvailableLoads->insert(Inst->getOperand(0),
                          std::pair<Value*, unsigned>(Inst, CurrentGeneration));
      LastStore = 0;
      continue;
    }

    // If this instruction may read from memory, forget LastStore.
    if (Inst->mayReadFromMemory())
      LastStore = 0;

    // If this is a read-only call, process it.
    if (CallValue::canHandle(Inst)) {
      // If we have an available version of this call, and if it is the right
      // generation, replace this instruction.
      std::pair<Value*, unsigned> InVal = AvailableCalls->lookup(Inst);
      if (InVal.first != 0 && InVal.second == CurrentGeneration) {
        DEBUG(dbgs() << "EarlyCSE CSE CALL: " << *Inst << "  to: "
                     << *InVal.first << '\n');
        if (!Inst->use_empty()) Inst->replaceAllUsesWith(InVal.first);
        Inst->eraseFromParent();
        Changed = true;
        ++NumCSECall;
        continue;
      }

      // Otherwise, remember that we have this instruction.
      AvailableCalls->insert(Inst,
                         std::pair<Value*, unsigned>(Inst, CurrentGeneration));
      continue;
    }

    // Okay, this isn't something we can CSE at all.  Check to see if it is
    // something that could modify memory.  If so, our available memory values
    // cannot be used so bump the generation count.
    if (Inst->mayWriteToMemory()) {
      ++CurrentGeneration;

      if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        // We do a trivial form of DSE if there are two stores to the same
        // location with no intervening loads.  Delete the earlier store.
        if (LastStore &&
            LastStore->getPointerOperand() == SI->getPointerOperand()) {
          DEBUG(dbgs() << "EarlyCSE DEAD STORE: " << *LastStore << "  due to: "
                       << *Inst << '\n');
          LastStore->eraseFromParent();
          Changed = true;
          ++NumDSE;
          LastStore = 0;
          continue;
        }

        // Okay, we just invalidated anything we knew about loaded values.  Try
        // to salvage *something* by remembering that the stored value is a live
        // version of the pointer.  It is safe to forward from volatile stores
        // to non-volatile loads, so we don't have to check for volatility of
        // the store.
        AvailableLoads->insert(SI->getPointerOperand(),
         std::pair<Value*, unsigned>(SI->getValueOperand(), CurrentGeneration));

        // Remember that this was the last store we saw for DSE.
        if (SI->isSimple())
          LastStore = SI;
      }
    }
  }

  return Changed;
}


bool EarlyCSE::runOnFunction(Function &F) {
  std::deque<StackNode *> nodesToProcess;

  TD = getAnalysisIfAvailable<DataLayout>();
  TLI = &getAnalysis<TargetLibraryInfo>();
  DT = &getAnalysis<DominatorTree>();

  // Tables that the pass uses when walking the domtree.
  ScopedHTType AVTable;
  AvailableValues = &AVTable;
  LoadHTType LoadTable;
  AvailableLoads = &LoadTable;
  CallHTType CallTable;
  AvailableCalls = &CallTable;

  CurrentGeneration = 0;
  bool Changed = false;

  // Process the root node.
  nodesToProcess.push_front(
      new StackNode(AvailableValues, AvailableLoads, AvailableCalls,
                    CurrentGeneration, DT->getRootNode(),
                    DT->getRootNode()->begin(),
                    DT->getRootNode()->end()));

  // Save the current generation.
  unsigned LiveOutGeneration = CurrentGeneration;

  // Process the stack.
  while (!nodesToProcess.empty()) {
    // Grab the first item off the stack. Set the current generation, remove
    // the node from the stack, and process it.
    StackNode *NodeToProcess = nodesToProcess.front();

    // Initialize class members.
    CurrentGeneration = NodeToProcess->currentGeneration();

    // Check if the node needs to be processed.
    if (!NodeToProcess->isProcessed()) {
      // Process the node.
      Changed |= processNode(NodeToProcess->node());
      NodeToProcess->childGeneration(CurrentGeneration);
      NodeToProcess->process();
    } else if (NodeToProcess->childIter() != NodeToProcess->end()) {
      // Push the next child onto the stack.
      DomTreeNode *child = NodeToProcess->nextChild();
      nodesToProcess.push_front(
          new StackNode(AvailableValues,
                        AvailableLoads,
                        AvailableCalls,
                        NodeToProcess->childGeneration(), child,
                        child->begin(), child->end()));
    } else {
      // It has been processed, and there are no more children to process,
      // so delete it and pop it off the stack.
      delete NodeToProcess;
      nodesToProcess.pop_front();
    }
  } // while (!nodes...)

  // Reset the current generation.
  CurrentGeneration = LiveOutGeneration;

  return Changed;
}
