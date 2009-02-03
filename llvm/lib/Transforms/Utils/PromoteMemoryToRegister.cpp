//===- PromoteMemoryToRegister.cpp - Convert allocas to registers ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file promotes memory references to be register references.  It promotes
// alloca instructions which only have loads and stores as uses.  An alloca is
// transformed by using dominator frontiers to place PHI nodes, then traversing
// the function in depth-first order to rewrite loads and stores as appropriate.
// This is just the standard SSA construction algorithm to construct "pruned"
// SSA form.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mem2reg"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumLocalPromoted, "Number of alloca's promoted within one block");
STATISTIC(NumSingleStore,   "Number of alloca's promoted with a single store");
STATISTIC(NumDeadAlloca,    "Number of dead alloca's removed");
STATISTIC(NumPHIInsert,     "Number of PHI nodes inserted");

// Provide DenseMapInfo for all pointers.
namespace llvm {
template<>
struct DenseMapInfo<std::pair<BasicBlock*, unsigned> > {
  typedef std::pair<BasicBlock*, unsigned> EltTy;
  static inline EltTy getEmptyKey() {
    return EltTy(reinterpret_cast<BasicBlock*>(-1), ~0U);
  }
  static inline EltTy getTombstoneKey() {
    return EltTy(reinterpret_cast<BasicBlock*>(-2), 0U);
  }
  static unsigned getHashValue(const std::pair<BasicBlock*, unsigned> &Val) {
    return DenseMapInfo<void*>::getHashValue(Val.first) + Val.second*2;
  }
  static bool isEqual(const EltTy &LHS, const EltTy &RHS) {
    return LHS == RHS;
  }
  static bool isPod() { return true; }
};
}

/// isAllocaPromotable - Return true if this alloca is legal for promotion.
/// This is true if there are only loads and stores to the alloca.
///
bool llvm::isAllocaPromotable(const AllocaInst *AI) {
  // FIXME: If the memory unit is of pointer or integer type, we can permit
  // assignments to subsections of the memory unit.

  // Only allow direct and non-volatile loads and stores...
  for (Value::use_const_iterator UI = AI->use_begin(), UE = AI->use_end();
       UI != UE; ++UI)     // Loop over all of the uses of the alloca
    if (const LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      if (LI->isVolatile())
        return false;
    } else if (const StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (SI->getOperand(0) == AI)
        return false;   // Don't allow a store OF the AI, only INTO the AI.
      if (SI->isVolatile())
        return false;
    } else if (const BitCastInst *BC = dyn_cast<BitCastInst>(*UI)) {
      // Uses by dbg info shouldn't inhibit promotion.
      if (!BC->hasOneUse() || !isa<DbgInfoIntrinsic>(*BC->use_begin()))
        return false;
    } else {
      return false;
    }

  return true;
}

namespace {
  struct AllocaInfo;

  // Data package used by RenamePass()
  class VISIBILITY_HIDDEN RenamePassData {
  public:
    typedef std::vector<Value *> ValVector;
    
    RenamePassData() {}
    RenamePassData(BasicBlock *B, BasicBlock *P,
                   const ValVector &V) : BB(B), Pred(P), Values(V) {}
    BasicBlock *BB;
    BasicBlock *Pred;
    ValVector Values;
    
    void swap(RenamePassData &RHS) {
      std::swap(BB, RHS.BB);
      std::swap(Pred, RHS.Pred);
      Values.swap(RHS.Values);
    }
  };
  
  /// LargeBlockInfo - This assigns and keeps a per-bb relative ordering of
  /// load/store instructions in the block that directly load or store an alloca.
  ///
  /// This functionality is important because it avoids scanning large basic
  /// blocks multiple times when promoting many allocas in the same block.
  class VISIBILITY_HIDDEN LargeBlockInfo {
    /// InstNumbers - For each instruction that we track, keep the index of the
    /// instruction.  The index starts out as the number of the instruction from
    /// the start of the block.
    DenseMap<const Instruction *, unsigned> InstNumbers;
  public:
    
    /// isInterestingInstruction - This code only looks at accesses to allocas.
    static bool isInterestingInstruction(const Instruction *I) {
      return (isa<LoadInst>(I) && isa<AllocaInst>(I->getOperand(0))) ||
             (isa<StoreInst>(I) && isa<AllocaInst>(I->getOperand(1)));
    }
    
    /// getInstructionIndex - Get or calculate the index of the specified
    /// instruction.
    unsigned getInstructionIndex(const Instruction *I) {
      assert(isInterestingInstruction(I) &&
             "Not a load/store to/from an alloca?");
      
      // If we already have this instruction number, return it.
      DenseMap<const Instruction *, unsigned>::iterator It = InstNumbers.find(I);
      if (It != InstNumbers.end()) return It->second;
      
      // Scan the whole block to get the instruction.  This accumulates
      // information for every interesting instruction in the block, in order to
      // avoid gratuitus rescans.
      const BasicBlock *BB = I->getParent();
      unsigned InstNo = 0;
      for (BasicBlock::const_iterator BBI = BB->begin(), E = BB->end();
           BBI != E; ++BBI)
        if (isInterestingInstruction(BBI))
          InstNumbers[BBI] = InstNo++;
      It = InstNumbers.find(I);
      
      assert(It != InstNumbers.end() && "Didn't insert instruction?");
      return It->second;
    }
    
    void deleteValue(const Instruction *I) {
      InstNumbers.erase(I);
    }
    
    void clear() {
      InstNumbers.clear();
    }
  };

  struct VISIBILITY_HIDDEN PromoteMem2Reg {
    /// Allocas - The alloca instructions being promoted.
    ///
    std::vector<AllocaInst*> Allocas;
    DominatorTree &DT;
    DominanceFrontier &DF;

    /// AST - An AliasSetTracker object to update.  If null, don't update it.
    ///
    AliasSetTracker *AST;

    /// AllocaLookup - Reverse mapping of Allocas.
    ///
    std::map<AllocaInst*, unsigned>  AllocaLookup;

    /// NewPhiNodes - The PhiNodes we're adding.
    ///
    DenseMap<std::pair<BasicBlock*, unsigned>, PHINode*> NewPhiNodes;
    
    /// PhiToAllocaMap - For each PHI node, keep track of which entry in Allocas
    /// it corresponds to.
    DenseMap<PHINode*, unsigned> PhiToAllocaMap;
    
    /// PointerAllocaValues - If we are updating an AliasSetTracker, then for
    /// each alloca that is of pointer type, we keep track of what to copyValue
    /// to the inserted PHI nodes here.
    ///
    std::vector<Value*> PointerAllocaValues;

    /// Visited - The set of basic blocks the renamer has already visited.
    ///
    SmallPtrSet<BasicBlock*, 16> Visited;

    /// BBNumbers - Contains a stable numbering of basic blocks to avoid
    /// non-determinstic behavior.
    DenseMap<BasicBlock*, unsigned> BBNumbers;

    /// BBNumPreds - Lazily compute the number of predecessors a block has.
    DenseMap<const BasicBlock*, unsigned> BBNumPreds;
  public:
    PromoteMem2Reg(const std::vector<AllocaInst*> &A, DominatorTree &dt,
                   DominanceFrontier &df, AliasSetTracker *ast)
      : Allocas(A), DT(dt), DF(df), AST(ast) {}

    void run();

    /// properlyDominates - Return true if I1 properly dominates I2.
    ///
    bool properlyDominates(Instruction *I1, Instruction *I2) const {
      if (InvokeInst *II = dyn_cast<InvokeInst>(I1))
        I1 = II->getNormalDest()->begin();
      return DT.properlyDominates(I1->getParent(), I2->getParent());
    }
    
    /// dominates - Return true if BB1 dominates BB2 using the DominatorTree.
    ///
    bool dominates(BasicBlock *BB1, BasicBlock *BB2) const {
      return DT.dominates(BB1, BB2);
    }

  private:
    void RemoveFromAllocasList(unsigned &AllocaIdx) {
      Allocas[AllocaIdx] = Allocas.back();
      Allocas.pop_back();
      --AllocaIdx;
    }

    unsigned getNumPreds(const BasicBlock *BB) {
      unsigned &NP = BBNumPreds[BB];
      if (NP == 0)
        NP = std::distance(pred_begin(BB), pred_end(BB))+1;
      return NP-1;
    }

    void DetermineInsertionPoint(AllocaInst *AI, unsigned AllocaNum,
                                 AllocaInfo &Info);
    void ComputeLiveInBlocks(AllocaInst *AI, AllocaInfo &Info, 
                             const SmallPtrSet<BasicBlock*, 32> &DefBlocks,
                             SmallPtrSet<BasicBlock*, 32> &LiveInBlocks);
    
    void RewriteSingleStoreAlloca(AllocaInst *AI, AllocaInfo &Info,
                                  LargeBlockInfo &LBI);
    void PromoteSingleBlockAlloca(AllocaInst *AI, AllocaInfo &Info,
                                  LargeBlockInfo &LBI);

    
    void RenamePass(BasicBlock *BB, BasicBlock *Pred,
                    RenamePassData::ValVector &IncVals,
                    std::vector<RenamePassData> &Worklist);
    bool QueuePhiNode(BasicBlock *BB, unsigned AllocaIdx, unsigned &Version,
                      SmallPtrSet<PHINode*, 16> &InsertedPHINodes);
  };
  
  struct AllocaInfo {
    std::vector<BasicBlock*> DefiningBlocks;
    std::vector<BasicBlock*> UsingBlocks;
    
    StoreInst  *OnlyStore;
    BasicBlock *OnlyBlock;
    bool OnlyUsedInOneBlock;
    
    Value *AllocaPointerVal;
    
    void clear() {
      DefiningBlocks.clear();
      UsingBlocks.clear();
      OnlyStore = 0;
      OnlyBlock = 0;
      OnlyUsedInOneBlock = true;
      AllocaPointerVal = 0;
    }
    
    /// AnalyzeAlloca - Scan the uses of the specified alloca, filling in our
    /// ivars.
    void AnalyzeAlloca(AllocaInst *AI) {
      clear();

      // As we scan the uses of the alloca instruction, keep track of stores,
      // and decide whether all of the loads and stores to the alloca are within
      // the same basic block.
      for (Value::use_iterator U = AI->use_begin(), E = AI->use_end();
           U != E;)  {
        Instruction *User = cast<Instruction>(*U);
        ++U;
        if (BitCastInst *BC = dyn_cast<BitCastInst>(User)) {
          // Remove any uses of this alloca in DbgInfoInstrinsics.
          assert(BC->hasOneUse() && "Unexpected alloca uses!");
          DbgInfoIntrinsic *DI = cast<DbgInfoIntrinsic>(*BC->use_begin());
          DI->eraseFromParent();
          BC->eraseFromParent();
          continue;
        } 
        else if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
          // Remember the basic blocks which define new values for the alloca
          DefiningBlocks.push_back(SI->getParent());
          AllocaPointerVal = SI->getOperand(0);
          OnlyStore = SI;
        } else {
          LoadInst *LI = cast<LoadInst>(User);
          // Otherwise it must be a load instruction, keep track of variable
          // reads.
          UsingBlocks.push_back(LI->getParent());
          AllocaPointerVal = LI;
        }
        
        if (OnlyUsedInOneBlock) {
          if (OnlyBlock == 0)
            OnlyBlock = User->getParent();
          else if (OnlyBlock != User->getParent())
            OnlyUsedInOneBlock = false;
        }
      }
    }
  };
}  // end of anonymous namespace


void PromoteMem2Reg::run() {
  Function &F = *DF.getRoot()->getParent();

  if (AST) PointerAllocaValues.resize(Allocas.size());

  AllocaInfo Info;
  LargeBlockInfo LBI;

  for (unsigned AllocaNum = 0; AllocaNum != Allocas.size(); ++AllocaNum) {
    AllocaInst *AI = Allocas[AllocaNum];

    assert(isAllocaPromotable(AI) &&
           "Cannot promote non-promotable alloca!");
    assert(AI->getParent()->getParent() == &F &&
           "All allocas should be in the same function, which is same as DF!");

    if (AI->use_empty()) {
      // If there are no uses of the alloca, just delete it now.
      if (AST) AST->deleteValue(AI);
      AI->eraseFromParent();

      // Remove the alloca from the Allocas list, since it has been processed
      RemoveFromAllocasList(AllocaNum);
      ++NumDeadAlloca;
      continue;
    }
    
    // Calculate the set of read and write-locations for each alloca.  This is
    // analogous to finding the 'uses' and 'definitions' of each variable.
    Info.AnalyzeAlloca(AI);

    // If there is only a single store to this value, replace any loads of
    // it that are directly dominated by the definition with the value stored.
    if (Info.DefiningBlocks.size() == 1) {
      RewriteSingleStoreAlloca(AI, Info, LBI);

      // Finally, after the scan, check to see if the store is all that is left.
      if (Info.UsingBlocks.empty()) {
        // Remove the (now dead) store and alloca.
        Info.OnlyStore->eraseFromParent();
        LBI.deleteValue(Info.OnlyStore);

        if (AST) AST->deleteValue(AI);
        AI->eraseFromParent();
        LBI.deleteValue(AI);
        
        // The alloca has been processed, move on.
        RemoveFromAllocasList(AllocaNum);
        
        ++NumSingleStore;
        continue;
      }
    }
    
    // If the alloca is only read and written in one basic block, just perform a
    // linear sweep over the block to eliminate it.
    if (Info.OnlyUsedInOneBlock) {
      PromoteSingleBlockAlloca(AI, Info, LBI);
      
      // Finally, after the scan, check to see if the stores are all that is
      // left.
      if (Info.UsingBlocks.empty()) {
        
        // Remove the (now dead) stores and alloca.
        while (!AI->use_empty()) {
          StoreInst *SI = cast<StoreInst>(AI->use_back());
          SI->eraseFromParent();
          LBI.deleteValue(SI);
        }
        
        if (AST) AST->deleteValue(AI);
        AI->eraseFromParent();
        LBI.deleteValue(AI);
        
        // The alloca has been processed, move on.
        RemoveFromAllocasList(AllocaNum);
        
        ++NumLocalPromoted;
        continue;
      }
    }
    
    // If we haven't computed a numbering for the BB's in the function, do so
    // now.
    if (BBNumbers.empty()) {
      unsigned ID = 0;
      for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
        BBNumbers[I] = ID++;
    }

    // If we have an AST to keep updated, remember some pointer value that is
    // stored into the alloca.
    if (AST)
      PointerAllocaValues[AllocaNum] = Info.AllocaPointerVal;
    
    // Keep the reverse mapping of the 'Allocas' array for the rename pass.
    AllocaLookup[Allocas[AllocaNum]] = AllocaNum;

    // At this point, we're committed to promoting the alloca using IDF's, and
    // the standard SSA construction algorithm.  Determine which blocks need PHI
    // nodes and see if we can optimize out some work by avoiding insertion of
    // dead phi nodes.
    DetermineInsertionPoint(AI, AllocaNum, Info);
  }

  if (Allocas.empty())
    return; // All of the allocas must have been trivial!

  LBI.clear();
  
  
  // Set the incoming values for the basic block to be null values for all of
  // the alloca's.  We do this in case there is a load of a value that has not
  // been stored yet.  In this case, it will get this null value.
  //
  RenamePassData::ValVector Values(Allocas.size());
  for (unsigned i = 0, e = Allocas.size(); i != e; ++i)
    Values[i] = UndefValue::get(Allocas[i]->getAllocatedType());

  // Walks all basic blocks in the function performing the SSA rename algorithm
  // and inserting the phi nodes we marked as necessary
  //
  std::vector<RenamePassData> RenamePassWorkList;
  RenamePassWorkList.push_back(RenamePassData(F.begin(), 0, Values));
  while (!RenamePassWorkList.empty()) {
    RenamePassData RPD;
    RPD.swap(RenamePassWorkList.back());
    RenamePassWorkList.pop_back();
    // RenamePass may add new worklist entries.
    RenamePass(RPD.BB, RPD.Pred, RPD.Values, RenamePassWorkList);
  }
  
  // The renamer uses the Visited set to avoid infinite loops.  Clear it now.
  Visited.clear();

  // Remove the allocas themselves from the function.
  for (unsigned i = 0, e = Allocas.size(); i != e; ++i) {
    Instruction *A = Allocas[i];

    // If there are any uses of the alloca instructions left, they must be in
    // sections of dead code that were not processed on the dominance frontier.
    // Just delete the users now.
    //
    if (!A->use_empty())
      A->replaceAllUsesWith(UndefValue::get(A->getType()));
    if (AST) AST->deleteValue(A);
    A->eraseFromParent();
  }

  
  // Loop over all of the PHI nodes and see if there are any that we can get
  // rid of because they merge all of the same incoming values.  This can
  // happen due to undef values coming into the PHI nodes.  This process is
  // iterative, because eliminating one PHI node can cause others to be removed.
  bool EliminatedAPHI = true;
  while (EliminatedAPHI) {
    EliminatedAPHI = false;
    
    for (DenseMap<std::pair<BasicBlock*, unsigned>, PHINode*>::iterator I =
           NewPhiNodes.begin(), E = NewPhiNodes.end(); I != E;) {
      PHINode *PN = I->second;
      
      // If this PHI node merges one value and/or undefs, get the value.
      if (Value *V = PN->hasConstantValue(true)) {
        if (!isa<Instruction>(V) ||
            properlyDominates(cast<Instruction>(V), PN)) {
          if (AST && isa<PointerType>(PN->getType()))
            AST->deleteValue(PN);
          PN->replaceAllUsesWith(V);
          PN->eraseFromParent();
          NewPhiNodes.erase(I++);
          EliminatedAPHI = true;
          continue;
        }
      }
      ++I;
    }
  }
  
  // At this point, the renamer has added entries to PHI nodes for all reachable
  // code.  Unfortunately, there may be unreachable blocks which the renamer
  // hasn't traversed.  If this is the case, the PHI nodes may not
  // have incoming values for all predecessors.  Loop over all PHI nodes we have
  // created, inserting undef values if they are missing any incoming values.
  //
  for (DenseMap<std::pair<BasicBlock*, unsigned>, PHINode*>::iterator I =
         NewPhiNodes.begin(), E = NewPhiNodes.end(); I != E; ++I) {
    // We want to do this once per basic block.  As such, only process a block
    // when we find the PHI that is the first entry in the block.
    PHINode *SomePHI = I->second;
    BasicBlock *BB = SomePHI->getParent();
    if (&BB->front() != SomePHI)
      continue;

    // Only do work here if there the PHI nodes are missing incoming values.  We
    // know that all PHI nodes that were inserted in a block will have the same
    // number of incoming values, so we can just check any of them.
    if (SomePHI->getNumIncomingValues() == getNumPreds(BB))
      continue;

    // Get the preds for BB.
    SmallVector<BasicBlock*, 16> Preds(pred_begin(BB), pred_end(BB));
    
    // Ok, now we know that all of the PHI nodes are missing entries for some
    // basic blocks.  Start by sorting the incoming predecessors for efficient
    // access.
    std::sort(Preds.begin(), Preds.end());
    
    // Now we loop through all BB's which have entries in SomePHI and remove
    // them from the Preds list.
    for (unsigned i = 0, e = SomePHI->getNumIncomingValues(); i != e; ++i) {
      // Do a log(n) search of the Preds list for the entry we want.
      SmallVector<BasicBlock*, 16>::iterator EntIt =
        std::lower_bound(Preds.begin(), Preds.end(),
                         SomePHI->getIncomingBlock(i));
      assert(EntIt != Preds.end() && *EntIt == SomePHI->getIncomingBlock(i)&&
             "PHI node has entry for a block which is not a predecessor!");

      // Remove the entry
      Preds.erase(EntIt);
    }

    // At this point, the blocks left in the preds list must have dummy
    // entries inserted into every PHI nodes for the block.  Update all the phi
    // nodes in this block that we are inserting (there could be phis before
    // mem2reg runs).
    unsigned NumBadPreds = SomePHI->getNumIncomingValues();
    BasicBlock::iterator BBI = BB->begin();
    while ((SomePHI = dyn_cast<PHINode>(BBI++)) &&
           SomePHI->getNumIncomingValues() == NumBadPreds) {
      Value *UndefVal = UndefValue::get(SomePHI->getType());
      for (unsigned pred = 0, e = Preds.size(); pred != e; ++pred)
        SomePHI->addIncoming(UndefVal, Preds[pred]);
    }
  }
        
  NewPhiNodes.clear();
}


/// ComputeLiveInBlocks - Determine which blocks the value is live in.  These
/// are blocks which lead to uses.  Knowing this allows us to avoid inserting
/// PHI nodes into blocks which don't lead to uses (thus, the inserted phi nodes
/// would be dead).
void PromoteMem2Reg::
ComputeLiveInBlocks(AllocaInst *AI, AllocaInfo &Info, 
                    const SmallPtrSet<BasicBlock*, 32> &DefBlocks,
                    SmallPtrSet<BasicBlock*, 32> &LiveInBlocks) {
  
  // To determine liveness, we must iterate through the predecessors of blocks
  // where the def is live.  Blocks are added to the worklist if we need to
  // check their predecessors.  Start with all the using blocks.
  SmallVector<BasicBlock*, 64> LiveInBlockWorklist;
  LiveInBlockWorklist.insert(LiveInBlockWorklist.end(), 
                             Info.UsingBlocks.begin(), Info.UsingBlocks.end());
  
  // If any of the using blocks is also a definition block, check to see if the
  // definition occurs before or after the use.  If it happens before the use,
  // the value isn't really live-in.
  for (unsigned i = 0, e = LiveInBlockWorklist.size(); i != e; ++i) {
    BasicBlock *BB = LiveInBlockWorklist[i];
    if (!DefBlocks.count(BB)) continue;
    
    // Okay, this is a block that both uses and defines the value.  If the first
    // reference to the alloca is a def (store), then we know it isn't live-in.
    for (BasicBlock::iterator I = BB->begin(); ; ++I) {
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        if (SI->getOperand(1) != AI) continue;
        
        // We found a store to the alloca before a load.  The alloca is not
        // actually live-in here.
        LiveInBlockWorklist[i] = LiveInBlockWorklist.back();
        LiveInBlockWorklist.pop_back();
        --i, --e;
        break;
      } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        if (LI->getOperand(0) != AI) continue;
        
        // Okay, we found a load before a store to the alloca.  It is actually
        // live into this block.
        break;
      }
    }
  }
  
  // Now that we have a set of blocks where the phi is live-in, recursively add
  // their predecessors until we find the full region the value is live.
  while (!LiveInBlockWorklist.empty()) {
    BasicBlock *BB = LiveInBlockWorklist.back();
    LiveInBlockWorklist.pop_back();
    
    // The block really is live in here, insert it into the set.  If already in
    // the set, then it has already been processed.
    if (!LiveInBlocks.insert(BB))
      continue;
    
    // Since the value is live into BB, it is either defined in a predecessor or
    // live into it to.  Add the preds to the worklist unless they are a
    // defining block.
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *P = *PI;
      
      // The value is not live into a predecessor if it defines the value.
      if (DefBlocks.count(P))
        continue;
      
      // Otherwise it is, add to the worklist.
      LiveInBlockWorklist.push_back(P);
    }
  }
}

/// DetermineInsertionPoint - At this point, we're committed to promoting the
/// alloca using IDF's, and the standard SSA construction algorithm.  Determine
/// which blocks need phi nodes and see if we can optimize out some work by
/// avoiding insertion of dead phi nodes.
void PromoteMem2Reg::DetermineInsertionPoint(AllocaInst *AI, unsigned AllocaNum,
                                             AllocaInfo &Info) {

  // Unique the set of defining blocks for efficient lookup.
  SmallPtrSet<BasicBlock*, 32> DefBlocks;
  DefBlocks.insert(Info.DefiningBlocks.begin(), Info.DefiningBlocks.end());

  // Determine which blocks the value is live in.  These are blocks which lead
  // to uses.
  SmallPtrSet<BasicBlock*, 32> LiveInBlocks;
  ComputeLiveInBlocks(AI, Info, DefBlocks, LiveInBlocks);

  // Compute the locations where PhiNodes need to be inserted.  Look at the
  // dominance frontier of EACH basic-block we have a write in.
  unsigned CurrentVersion = 0;
  SmallPtrSet<PHINode*, 16> InsertedPHINodes;
  std::vector<std::pair<unsigned, BasicBlock*> > DFBlocks;
  while (!Info.DefiningBlocks.empty()) {
    BasicBlock *BB = Info.DefiningBlocks.back();
    Info.DefiningBlocks.pop_back();
    
    // Look up the DF for this write, add it to defining blocks.
    DominanceFrontier::const_iterator it = DF.find(BB);
    if (it == DF.end()) continue;
    
    const DominanceFrontier::DomSetType &S = it->second;
    
    // In theory we don't need the indirection through the DFBlocks vector.
    // In practice, the order of calling QueuePhiNode would depend on the
    // (unspecified) ordering of basic blocks in the dominance frontier,
    // which would give PHI nodes non-determinstic subscripts.  Fix this by
    // processing blocks in order of the occurance in the function.
    for (DominanceFrontier::DomSetType::const_iterator P = S.begin(),
         PE = S.end(); P != PE; ++P) {
      // If the frontier block is not in the live-in set for the alloca, don't
      // bother processing it.
      if (!LiveInBlocks.count(*P))
        continue;
      
      DFBlocks.push_back(std::make_pair(BBNumbers[*P], *P));
    }
    
    // Sort by which the block ordering in the function.
    if (DFBlocks.size() > 1)
      std::sort(DFBlocks.begin(), DFBlocks.end());
    
    for (unsigned i = 0, e = DFBlocks.size(); i != e; ++i) {
      BasicBlock *BB = DFBlocks[i].second;
      if (QueuePhiNode(BB, AllocaNum, CurrentVersion, InsertedPHINodes))
        Info.DefiningBlocks.push_back(BB);
    }
    DFBlocks.clear();
  }
}

/// RewriteSingleStoreAlloca - If there is only a single store to this value,
/// replace any loads of it that are directly dominated by the definition with
/// the value stored.
void PromoteMem2Reg::RewriteSingleStoreAlloca(AllocaInst *AI,
                                              AllocaInfo &Info,
                                              LargeBlockInfo &LBI) {
  StoreInst *OnlyStore = Info.OnlyStore;
  bool StoringGlobalVal = !isa<Instruction>(OnlyStore->getOperand(0));
  BasicBlock *StoreBB = OnlyStore->getParent();
  int StoreIndex = -1;

  // Clear out UsingBlocks.  We will reconstruct it here if needed.
  Info.UsingBlocks.clear();
  
  for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end(); UI != E; ) {
    Instruction *UserInst = cast<Instruction>(*UI++);
    if (!isa<LoadInst>(UserInst)) {
      assert(UserInst == OnlyStore && "Should only have load/stores");
      continue;
    }
    LoadInst *LI = cast<LoadInst>(UserInst);
    
    // Okay, if we have a load from the alloca, we want to replace it with the
    // only value stored to the alloca.  We can do this if the value is
    // dominated by the store.  If not, we use the rest of the mem2reg machinery
    // to insert the phi nodes as needed.
    if (!StoringGlobalVal) {  // Non-instructions are always dominated.
      if (LI->getParent() == StoreBB) {
        // If we have a use that is in the same block as the store, compare the
        // indices of the two instructions to see which one came first.  If the
        // load came before the store, we can't handle it.
        if (StoreIndex == -1)
          StoreIndex = LBI.getInstructionIndex(OnlyStore);

        if (unsigned(StoreIndex) > LBI.getInstructionIndex(LI)) {
          // Can't handle this load, bail out.
          Info.UsingBlocks.push_back(StoreBB);
          continue;
        }
        
      } else if (LI->getParent() != StoreBB &&
                 !dominates(StoreBB, LI->getParent())) {
        // If the load and store are in different blocks, use BB dominance to
        // check their relationships.  If the store doesn't dom the use, bail
        // out.
        Info.UsingBlocks.push_back(LI->getParent());
        continue;
      }
    }
    
    // Otherwise, we *can* safely rewrite this load.
    LI->replaceAllUsesWith(OnlyStore->getOperand(0));
    if (AST && isa<PointerType>(LI->getType()))
      AST->deleteValue(LI);
    LI->eraseFromParent();
    LBI.deleteValue(LI);
  }
}


/// StoreIndexSearchPredicate - This is a helper predicate used to search by the
/// first element of a pair.
struct StoreIndexSearchPredicate {
  bool operator()(const std::pair<unsigned, StoreInst*> &LHS,
                  const std::pair<unsigned, StoreInst*> &RHS) {
    return LHS.first < RHS.first;
  }
};

/// PromoteSingleBlockAlloca - Many allocas are only used within a single basic
/// block.  If this is the case, avoid traversing the CFG and inserting a lot of
/// potentially useless PHI nodes by just performing a single linear pass over
/// the basic block using the Alloca.
///
/// If we cannot promote this alloca (because it is read before it is written),
/// return true.  This is necessary in cases where, due to control flow, the
/// alloca is potentially undefined on some control flow paths.  e.g. code like
/// this is potentially correct:
///
///   for (...) { if (c) { A = undef; undef = B; } }
///
/// ... so long as A is not used before undef is set.
///
void PromoteMem2Reg::PromoteSingleBlockAlloca(AllocaInst *AI, AllocaInfo &Info,
                                              LargeBlockInfo &LBI) {
  // The trickiest case to handle is when we have large blocks. Because of this,
  // this code is optimized assuming that large blocks happen.  This does not
  // significantly pessimize the small block case.  This uses LargeBlockInfo to
  // make it efficient to get the index of various operations in the block.
  
  // Clear out UsingBlocks.  We will reconstruct it here if needed.
  Info.UsingBlocks.clear();
  
  // Walk the use-def list of the alloca, getting the locations of all stores.
  typedef SmallVector<std::pair<unsigned, StoreInst*>, 64> StoresByIndexTy;
  StoresByIndexTy StoresByIndex;
  
  for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end();
       UI != E; ++UI) 
    if (StoreInst *SI = dyn_cast<StoreInst>(*UI))
      StoresByIndex.push_back(std::make_pair(LBI.getInstructionIndex(SI), SI));

  // If there are no stores to the alloca, just replace any loads with undef.
  if (StoresByIndex.empty()) {
    for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end(); UI != E;) 
      if (LoadInst *LI = dyn_cast<LoadInst>(*UI++)) {
        LI->replaceAllUsesWith(UndefValue::get(LI->getType()));
        if (AST && isa<PointerType>(LI->getType()))
          AST->deleteValue(LI);
        LBI.deleteValue(LI);
        LI->eraseFromParent();
      }
    return;
  }
  
  // Sort the stores by their index, making it efficient to do a lookup with a
  // binary search.
  std::sort(StoresByIndex.begin(), StoresByIndex.end());
  
  // Walk all of the loads from this alloca, replacing them with the nearest
  // store above them, if any.
  for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end(); UI != E;) {
    LoadInst *LI = dyn_cast<LoadInst>(*UI++);
    if (!LI) continue;
    
    unsigned LoadIdx = LBI.getInstructionIndex(LI);
    
    // Find the nearest store that has a lower than this load. 
    StoresByIndexTy::iterator I = 
      std::lower_bound(StoresByIndex.begin(), StoresByIndex.end(),
                       std::pair<unsigned, StoreInst*>(LoadIdx, 0),
                       StoreIndexSearchPredicate());
    
    // If there is no store before this load, then we can't promote this load.
    if (I == StoresByIndex.begin()) {
      // Can't handle this load, bail out.
      Info.UsingBlocks.push_back(LI->getParent());
      continue;
    }
      
    // Otherwise, there was a store before this load, the load takes its value.
    --I;
    LI->replaceAllUsesWith(I->second->getOperand(0));
    if (AST && isa<PointerType>(LI->getType()))
      AST->deleteValue(LI);
    LI->eraseFromParent();
    LBI.deleteValue(LI);
  }
}


// QueuePhiNode - queues a phi-node to be added to a basic-block for a specific
// Alloca returns true if there wasn't already a phi-node for that variable
//
bool PromoteMem2Reg::QueuePhiNode(BasicBlock *BB, unsigned AllocaNo,
                                  unsigned &Version,
                                  SmallPtrSet<PHINode*, 16> &InsertedPHINodes) {
  // Look up the basic-block in question.
  PHINode *&PN = NewPhiNodes[std::make_pair(BB, AllocaNo)];

  // If the BB already has a phi node added for the i'th alloca then we're done!
  if (PN) return false;

  // Create a PhiNode using the dereferenced type... and add the phi-node to the
  // BasicBlock.
  PN = PHINode::Create(Allocas[AllocaNo]->getAllocatedType(),
                       Allocas[AllocaNo]->getName() + "." +
                       utostr(Version++), BB->begin());
  ++NumPHIInsert;
  PhiToAllocaMap[PN] = AllocaNo;
  PN->reserveOperandSpace(getNumPreds(BB));
  
  InsertedPHINodes.insert(PN);

  if (AST && isa<PointerType>(PN->getType()))
    AST->copyValue(PointerAllocaValues[AllocaNo], PN);

  return true;
}

// RenamePass - Recursively traverse the CFG of the function, renaming loads and
// stores to the allocas which we are promoting.  IncomingVals indicates what
// value each Alloca contains on exit from the predecessor block Pred.
//
void PromoteMem2Reg::RenamePass(BasicBlock *BB, BasicBlock *Pred,
                                RenamePassData::ValVector &IncomingVals,
                                std::vector<RenamePassData> &Worklist) {
NextIteration:
  // If we are inserting any phi nodes into this BB, they will already be in the
  // block.
  if (PHINode *APN = dyn_cast<PHINode>(BB->begin())) {
    // Pred may have multiple edges to BB.  If so, we want to add N incoming
    // values to each PHI we are inserting on the first time we see the edge.
    // Check to see if APN already has incoming values from Pred.  This also
    // prevents us from modifying PHI nodes that are not currently being
    // inserted.
    bool HasPredEntries = false;
    for (unsigned i = 0, e = APN->getNumIncomingValues(); i != e; ++i) {
      if (APN->getIncomingBlock(i) == Pred) {
        HasPredEntries = true;
        break;
      }
    }
    
    // If we have PHI nodes to update, compute the number of edges from Pred to
    // BB.
    if (!HasPredEntries) {
      // We want to be able to distinguish between PHI nodes being inserted by
      // this invocation of mem2reg from those phi nodes that already existed in
      // the IR before mem2reg was run.  We determine that APN is being inserted
      // because it is missing incoming edges.  All other PHI nodes being
      // inserted by this pass of mem2reg will have the same number of incoming
      // operands so far.  Remember this count.
      unsigned NewPHINumOperands = APN->getNumOperands();
      
      unsigned NumEdges = 0;
      for (succ_iterator I = succ_begin(Pred), E = succ_end(Pred); I != E; ++I)
        if (*I == BB)
          ++NumEdges;
      assert(NumEdges && "Must be at least one edge from Pred to BB!");
      
      // Add entries for all the phis.
      BasicBlock::iterator PNI = BB->begin();
      do {
        unsigned AllocaNo = PhiToAllocaMap[APN];
        
        // Add N incoming values to the PHI node.
        for (unsigned i = 0; i != NumEdges; ++i)
          APN->addIncoming(IncomingVals[AllocaNo], Pred);
        
        // The currently active variable for this block is now the PHI.
        IncomingVals[AllocaNo] = APN;
        
        // Get the next phi node.
        ++PNI;
        APN = dyn_cast<PHINode>(PNI);
        if (APN == 0) break;
        
        // Verify that it is missing entries.  If not, it is not being inserted
        // by this mem2reg invocation so we want to ignore it.
      } while (APN->getNumOperands() == NewPHINumOperands);
    }
  }
  
  // Don't revisit blocks.
  if (!Visited.insert(BB)) return;

  for (BasicBlock::iterator II = BB->begin(); !isa<TerminatorInst>(II); ) {
    Instruction *I = II++; // get the instruction, increment iterator

    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      AllocaInst *Src = dyn_cast<AllocaInst>(LI->getPointerOperand());
      if (!Src) continue;
  
      std::map<AllocaInst*, unsigned>::iterator AI = AllocaLookup.find(Src);
      if (AI == AllocaLookup.end()) continue;

      Value *V = IncomingVals[AI->second];

      // Anything using the load now uses the current value.
      LI->replaceAllUsesWith(V);
      if (AST && isa<PointerType>(LI->getType()))
        AST->deleteValue(LI);
      BB->getInstList().erase(LI);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      // Delete this instruction and mark the name as the current holder of the
      // value
      AllocaInst *Dest = dyn_cast<AllocaInst>(SI->getPointerOperand());
      if (!Dest) continue;
      
      std::map<AllocaInst *, unsigned>::iterator ai = AllocaLookup.find(Dest);
      if (ai == AllocaLookup.end())
        continue;
      
      // what value were we writing?
      IncomingVals[ai->second] = SI->getOperand(0);
      BB->getInstList().erase(SI);
    }
  }

  // 'Recurse' to our successors.
  succ_iterator I = succ_begin(BB), E = succ_end(BB);
  if (I == E) return;

  // Handle the last successor without using the worklist.  This allows us to
  // handle unconditional branches directly, for example.
  --E;
  for (; I != E; ++I)
    Worklist.push_back(RenamePassData(*I, BB, IncomingVals));

  Pred = BB;
  BB = *I;
  goto NextIteration;
}

/// PromoteMemToReg - Promote the specified list of alloca instructions into
/// scalar registers, inserting PHI nodes as appropriate.  This function makes
/// use of DominanceFrontier information.  This function does not modify the CFG
/// of the function at all.  All allocas must be from the same function.
///
/// If AST is specified, the specified tracker is updated to reflect changes
/// made to the IR.
///
void llvm::PromoteMemToReg(const std::vector<AllocaInst*> &Allocas,
                           DominatorTree &DT, DominanceFrontier &DF,
                           AliasSetTracker *AST) {
  // If there is nothing to do, bail out...
  if (Allocas.empty()) return;

  PromoteMem2Reg(Allocas, DT, DF, AST).run();
}
