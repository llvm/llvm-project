//===- LazyValueInfo.cpp - Value constraint analysis ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for lazy computation of value constraint
// information.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lazy-value-info"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <stack>
using namespace llvm;
using namespace PatternMatch;

char LazyValueInfo::ID = 0;
INITIALIZE_PASS_BEGIN(LazyValueInfo, "lazy-value-info",
                "Lazy Value Information Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(LazyValueInfo, "lazy-value-info",
                "Lazy Value Information Analysis", false, true)

namespace llvm {
  FunctionPass *createLazyValueInfoPass() { return new LazyValueInfo(); }
}


//===----------------------------------------------------------------------===//
//                               LVILatticeVal
//===----------------------------------------------------------------------===//

/// LVILatticeVal - This is the information tracked by LazyValueInfo for each
/// value.
///
/// FIXME: This is basically just for bringup, this can be made a lot more rich
/// in the future.
///
namespace {
class LVILatticeVal {
  enum LatticeValueTy {
    /// undefined - This Value has no known value yet.
    undefined,
    
    /// constant - This Value has a specific constant value.
    constant,
    /// notconstant - This Value is known to not have the specified value.
    notconstant,

    /// constantrange - The Value falls within this range.
    constantrange,

    /// overdefined - This value is not known to be constant, and we know that
    /// it has a value.
    overdefined
  };
  
  /// Val: This stores the current lattice value along with the Constant* for
  /// the constant if this is a 'constant' or 'notconstant' value.
  LatticeValueTy Tag;
  Constant *Val;
  ConstantRange Range;
  
public:
  LVILatticeVal() : Tag(undefined), Val(0), Range(1, true) {}

  static LVILatticeVal get(Constant *C) {
    LVILatticeVal Res;
    if (!isa<UndefValue>(C))
      Res.markConstant(C);
    return Res;
  }
  static LVILatticeVal getNot(Constant *C) {
    LVILatticeVal Res;
    if (!isa<UndefValue>(C))
      Res.markNotConstant(C);
    return Res;
  }
  static LVILatticeVal getRange(ConstantRange CR) {
    LVILatticeVal Res;
    Res.markConstantRange(CR);
    return Res;
  }
  
  bool isUndefined() const     { return Tag == undefined; }
  bool isConstant() const      { return Tag == constant; }
  bool isNotConstant() const   { return Tag == notconstant; }
  bool isConstantRange() const { return Tag == constantrange; }
  bool isOverdefined() const   { return Tag == overdefined; }
  
  Constant *getConstant() const {
    assert(isConstant() && "Cannot get the constant of a non-constant!");
    return Val;
  }
  
  Constant *getNotConstant() const {
    assert(isNotConstant() && "Cannot get the constant of a non-notconstant!");
    return Val;
  }
  
  ConstantRange getConstantRange() const {
    assert(isConstantRange() &&
           "Cannot get the constant-range of a non-constant-range!");
    return Range;
  }
  
  /// markOverdefined - Return true if this is a change in status.
  bool markOverdefined() {
    if (isOverdefined())
      return false;
    Tag = overdefined;
    return true;
  }

  /// markConstant - Return true if this is a change in status.
  bool markConstant(Constant *V) {
    assert(V && "Marking constant with NULL");
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return markConstantRange(ConstantRange(CI->getValue()));
    if (isa<UndefValue>(V))
      return false;

    assert((!isConstant() || getConstant() == V) &&
           "Marking constant with different value");
    assert(isUndefined());
    Tag = constant;
    Val = V;
    return true;
  }
  
  /// markNotConstant - Return true if this is a change in status.
  bool markNotConstant(Constant *V) {
    assert(V && "Marking constant with NULL");
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return markConstantRange(ConstantRange(CI->getValue()+1, CI->getValue()));
    if (isa<UndefValue>(V))
      return false;

    assert((!isConstant() || getConstant() != V) &&
           "Marking constant !constant with same value");
    assert((!isNotConstant() || getNotConstant() == V) &&
           "Marking !constant with different value");
    assert(isUndefined() || isConstant());
    Tag = notconstant;
    Val = V;
    return true;
  }
  
  /// markConstantRange - Return true if this is a change in status.
  bool markConstantRange(const ConstantRange NewR) {
    if (isConstantRange()) {
      if (NewR.isEmptySet())
        return markOverdefined();
      
      bool changed = Range != NewR;
      Range = NewR;
      return changed;
    }
    
    assert(isUndefined());
    if (NewR.isEmptySet())
      return markOverdefined();
    
    Tag = constantrange;
    Range = NewR;
    return true;
  }
  
  /// mergeIn - Merge the specified lattice value into this one, updating this
  /// one and returning true if anything changed.
  bool mergeIn(const LVILatticeVal &RHS) {
    if (RHS.isUndefined() || isOverdefined()) return false;
    if (RHS.isOverdefined()) return markOverdefined();

    if (isUndefined()) {
      Tag = RHS.Tag;
      Val = RHS.Val;
      Range = RHS.Range;
      return true;
    }

    if (isConstant()) {
      if (RHS.isConstant()) {
        if (Val == RHS.Val)
          return false;
        return markOverdefined();
      }

      if (RHS.isNotConstant()) {
        if (Val == RHS.Val)
          return markOverdefined();

        // Unless we can prove that the two Constants are different, we must
        // move to overdefined.
        // FIXME: use DataLayout/TargetLibraryInfo for smarter constant folding.
        if (ConstantInt *Res = dyn_cast<ConstantInt>(
                ConstantFoldCompareInstOperands(CmpInst::ICMP_NE,
                                                getConstant(),
                                                RHS.getNotConstant())))
          if (Res->isOne())
            return markNotConstant(RHS.getNotConstant());

        return markOverdefined();
      }

      // RHS is a ConstantRange, LHS is a non-integer Constant.

      // FIXME: consider the case where RHS is a range [1, 0) and LHS is
      // a function. The correct result is to pick up RHS.

      return markOverdefined();
    }

    if (isNotConstant()) {
      if (RHS.isConstant()) {
        if (Val == RHS.Val)
          return markOverdefined();

        // Unless we can prove that the two Constants are different, we must
        // move to overdefined.
        // FIXME: use DataLayout/TargetLibraryInfo for smarter constant folding.
        if (ConstantInt *Res = dyn_cast<ConstantInt>(
                ConstantFoldCompareInstOperands(CmpInst::ICMP_NE,
                                                getNotConstant(),
                                                RHS.getConstant())))
          if (Res->isOne())
            return false;

        return markOverdefined();
      }

      if (RHS.isNotConstant()) {
        if (Val == RHS.Val)
          return false;
        return markOverdefined();
      }

      return markOverdefined();
    }

    assert(isConstantRange() && "New LVILattice type?");
    if (!RHS.isConstantRange())
      return markOverdefined();

    ConstantRange NewR = Range.unionWith(RHS.getConstantRange());
    if (NewR.isFullSet())
      return markOverdefined();
    return markConstantRange(NewR);
  }
};
  
} // end anonymous namespace.

namespace llvm {
raw_ostream &operator<<(raw_ostream &OS, const LVILatticeVal &Val)
    LLVM_ATTRIBUTE_USED;
raw_ostream &operator<<(raw_ostream &OS, const LVILatticeVal &Val) {
  if (Val.isUndefined())
    return OS << "undefined";
  if (Val.isOverdefined())
    return OS << "overdefined";

  if (Val.isNotConstant())
    return OS << "notconstant<" << *Val.getNotConstant() << '>';
  else if (Val.isConstantRange())
    return OS << "constantrange<" << Val.getConstantRange().getLower() << ", "
              << Val.getConstantRange().getUpper() << '>';
  return OS << "constant<" << *Val.getConstant() << '>';
}
}

//===----------------------------------------------------------------------===//
//                          LazyValueInfoCache Decl
//===----------------------------------------------------------------------===//

namespace {
  /// LVIValueHandle - A callback value handle updates the cache when
  /// values are erased.
  class LazyValueInfoCache;
  struct LVIValueHandle : public CallbackVH {
    LazyValueInfoCache *Parent;
      
    LVIValueHandle(Value *V, LazyValueInfoCache *P)
      : CallbackVH(V), Parent(P) { }
      
    void deleted();
    void allUsesReplacedWith(Value *V) {
      deleted();
    }
  };
}

namespace { 
  /// LazyValueInfoCache - This is the cache kept by LazyValueInfo which
  /// maintains information about queries across the clients' queries.
  class LazyValueInfoCache {
    /// ValueCacheEntryTy - This is all of the cached block information for
    /// exactly one Value*.  The entries are sorted by the BasicBlock* of the
    /// entries, allowing us to do a lookup with a binary search.
    typedef std::map<AssertingVH<BasicBlock>, LVILatticeVal> ValueCacheEntryTy;

    /// ValueCache - This is all of the cached information for all values,
    /// mapped from Value* to key information.
    std::map<LVIValueHandle, ValueCacheEntryTy> ValueCache;
    
    /// OverDefinedCache - This tracks, on a per-block basis, the set of 
    /// values that are over-defined at the end of that block.  This is required
    /// for cache updating.
    typedef std::pair<AssertingVH<BasicBlock>, Value*> OverDefinedPairTy;
    DenseSet<OverDefinedPairTy> OverDefinedCache;

    /// SeenBlocks - Keep track of all blocks that we have ever seen, so we
    /// don't spend time removing unused blocks from our caches.
    DenseSet<AssertingVH<BasicBlock> > SeenBlocks;

    /// BlockValueStack - This stack holds the state of the value solver
    /// during a query.  It basically emulates the callstack of the naive
    /// recursive value lookup process.
    std::stack<std::pair<BasicBlock*, Value*> > BlockValueStack;
    
    friend struct LVIValueHandle;
    
    /// OverDefinedCacheUpdater - A helper object that ensures that the
    /// OverDefinedCache is updated whenever solveBlockValue returns.
    struct OverDefinedCacheUpdater {
      LazyValueInfoCache *Parent;
      Value *Val;
      BasicBlock *BB;
      LVILatticeVal &BBLV;
      
      OverDefinedCacheUpdater(Value *V, BasicBlock *B, LVILatticeVal &LV,
                       LazyValueInfoCache *P)
        : Parent(P), Val(V), BB(B), BBLV(LV) { }
      
      bool markResult(bool changed) { 
        if (changed && BBLV.isOverdefined())
          Parent->OverDefinedCache.insert(std::make_pair(BB, Val));
        return changed;
      }
    };
    


    LVILatticeVal getBlockValue(Value *Val, BasicBlock *BB);
    bool getEdgeValue(Value *V, BasicBlock *F, BasicBlock *T,
                      LVILatticeVal &Result);
    bool hasBlockValue(Value *Val, BasicBlock *BB);

    // These methods process one work item and may add more. A false value
    // returned means that the work item was not completely processed and must
    // be revisited after going through the new items.
    bool solveBlockValue(Value *Val, BasicBlock *BB);
    bool solveBlockValueNonLocal(LVILatticeVal &BBLV,
                                 Value *Val, BasicBlock *BB);
    bool solveBlockValuePHINode(LVILatticeVal &BBLV,
                                PHINode *PN, BasicBlock *BB);
    bool solveBlockValueConstantRange(LVILatticeVal &BBLV,
                                      Instruction *BBI, BasicBlock *BB);

    void solve();
    
    ValueCacheEntryTy &lookup(Value *V) {
      return ValueCache[LVIValueHandle(V, this)];
    }

  public:
    /// getValueInBlock - This is the query interface to determine the lattice
    /// value for the specified Value* at the end of the specified block.
    LVILatticeVal getValueInBlock(Value *V, BasicBlock *BB);

    /// getValueOnEdge - This is the query interface to determine the lattice
    /// value for the specified Value* that is true on the specified edge.
    LVILatticeVal getValueOnEdge(Value *V, BasicBlock *FromBB,BasicBlock *ToBB);
    
    /// threadEdge - This is the update interface to inform the cache that an
    /// edge from PredBB to OldSucc has been threaded to be from PredBB to
    /// NewSucc.
    void threadEdge(BasicBlock *PredBB,BasicBlock *OldSucc,BasicBlock *NewSucc);
    
    /// eraseBlock - This is part of the update interface to inform the cache
    /// that a block has been deleted.
    void eraseBlock(BasicBlock *BB);
    
    /// clear - Empty the cache.
    void clear() {
      SeenBlocks.clear();
      ValueCache.clear();
      OverDefinedCache.clear();
    }
  };
} // end anonymous namespace

void LVIValueHandle::deleted() {
  typedef std::pair<AssertingVH<BasicBlock>, Value*> OverDefinedPairTy;
  
  SmallVector<OverDefinedPairTy, 4> ToErase;
  for (DenseSet<OverDefinedPairTy>::iterator 
       I = Parent->OverDefinedCache.begin(),
       E = Parent->OverDefinedCache.end();
       I != E; ++I) {
    if (I->second == getValPtr())
      ToErase.push_back(*I);
  }
  
  for (SmallVector<OverDefinedPairTy, 4>::iterator I = ToErase.begin(),
       E = ToErase.end(); I != E; ++I)
    Parent->OverDefinedCache.erase(*I);
  
  // This erasure deallocates *this, so it MUST happen after we're done
  // using any and all members of *this.
  Parent->ValueCache.erase(*this);
}

void LazyValueInfoCache::eraseBlock(BasicBlock *BB) {
  // Shortcut if we have never seen this block.
  DenseSet<AssertingVH<BasicBlock> >::iterator I = SeenBlocks.find(BB);
  if (I == SeenBlocks.end())
    return;
  SeenBlocks.erase(I);

  SmallVector<OverDefinedPairTy, 4> ToErase;
  for (DenseSet<OverDefinedPairTy>::iterator  I = OverDefinedCache.begin(),
       E = OverDefinedCache.end(); I != E; ++I) {
    if (I->first == BB)
      ToErase.push_back(*I);
  }
  
  for (SmallVector<OverDefinedPairTy, 4>::iterator I = ToErase.begin(),
       E = ToErase.end(); I != E; ++I)
    OverDefinedCache.erase(*I);

  for (std::map<LVIValueHandle, ValueCacheEntryTy>::iterator
       I = ValueCache.begin(), E = ValueCache.end(); I != E; ++I)
    I->second.erase(BB);
}

void LazyValueInfoCache::solve() {
  while (!BlockValueStack.empty()) {
    std::pair<BasicBlock*, Value*> &e = BlockValueStack.top();
    if (solveBlockValue(e.second, e.first)) {
      assert(BlockValueStack.top() == e);
      BlockValueStack.pop();
    }
  }
}

bool LazyValueInfoCache::hasBlockValue(Value *Val, BasicBlock *BB) {
  // If already a constant, there is nothing to compute.
  if (isa<Constant>(Val))
    return true;

  LVIValueHandle ValHandle(Val, this);
  std::map<LVIValueHandle, ValueCacheEntryTy>::iterator I =
    ValueCache.find(ValHandle);
  if (I == ValueCache.end()) return false;
  return I->second.count(BB);
}

LVILatticeVal LazyValueInfoCache::getBlockValue(Value *Val, BasicBlock *BB) {
  // If already a constant, there is nothing to compute.
  if (Constant *VC = dyn_cast<Constant>(Val))
    return LVILatticeVal::get(VC);

  SeenBlocks.insert(BB);
  return lookup(Val)[BB];
}

bool LazyValueInfoCache::solveBlockValue(Value *Val, BasicBlock *BB) {
  if (isa<Constant>(Val))
    return true;

  ValueCacheEntryTy &Cache = lookup(Val);
  SeenBlocks.insert(BB);
  LVILatticeVal &BBLV = Cache[BB];
  
  // OverDefinedCacheUpdater is a helper object that will update
  // the OverDefinedCache for us when this method exits.  Make sure to
  // call markResult on it as we exist, passing a bool to indicate if the
  // cache needs updating, i.e. if we have solve a new value or not.
  OverDefinedCacheUpdater ODCacheUpdater(Val, BB, BBLV, this);

  // If we've already computed this block's value, return it.
  if (!BBLV.isUndefined()) {
    DEBUG(dbgs() << "  reuse BB '" << BB->getName() << "' val=" << BBLV <<'\n');
    
    // Since we're reusing a cached value here, we don't need to update the 
    // OverDefinedCahce.  The cache will have been properly updated 
    // whenever the cached value was inserted.
    ODCacheUpdater.markResult(false);
    return true;
  }

  // Otherwise, this is the first time we're seeing this block.  Reset the
  // lattice value to overdefined, so that cycles will terminate and be
  // conservatively correct.
  BBLV.markOverdefined();
  
  Instruction *BBI = dyn_cast<Instruction>(Val);
  if (BBI == 0 || BBI->getParent() != BB) {
    return ODCacheUpdater.markResult(solveBlockValueNonLocal(BBLV, Val, BB));
  }

  if (PHINode *PN = dyn_cast<PHINode>(BBI)) {
    return ODCacheUpdater.markResult(solveBlockValuePHINode(BBLV, PN, BB));
  }

  if (AllocaInst *AI = dyn_cast<AllocaInst>(BBI)) {
    BBLV = LVILatticeVal::getNot(ConstantPointerNull::get(AI->getType()));
    return ODCacheUpdater.markResult(true);
  }

  // We can only analyze the definitions of certain classes of instructions
  // (integral binops and casts at the moment), so bail if this isn't one.
  LVILatticeVal Result;
  if ((!isa<BinaryOperator>(BBI) && !isa<CastInst>(BBI)) ||
     !BBI->getType()->isIntegerTy()) {
    DEBUG(dbgs() << " compute BB '" << BB->getName()
                 << "' - overdefined because inst def found.\n");
    BBLV.markOverdefined();
    return ODCacheUpdater.markResult(true);
  }

  // FIXME: We're currently limited to binops with a constant RHS.  This should
  // be improved.
  BinaryOperator *BO = dyn_cast<BinaryOperator>(BBI);
  if (BO && !isa<ConstantInt>(BO->getOperand(1))) { 
    DEBUG(dbgs() << " compute BB '" << BB->getName()
                 << "' - overdefined because inst def found.\n");

    BBLV.markOverdefined();
    return ODCacheUpdater.markResult(true);
  }

  return ODCacheUpdater.markResult(solveBlockValueConstantRange(BBLV, BBI, BB));
}

static bool InstructionDereferencesPointer(Instruction *I, Value *Ptr) {
  if (LoadInst *L = dyn_cast<LoadInst>(I)) {
    return L->getPointerAddressSpace() == 0 &&
        GetUnderlyingObject(L->getPointerOperand()) == Ptr;
  }
  if (StoreInst *S = dyn_cast<StoreInst>(I)) {
    return S->getPointerAddressSpace() == 0 &&
        GetUnderlyingObject(S->getPointerOperand()) == Ptr;
  }
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I)) {
    if (MI->isVolatile()) return false;

    // FIXME: check whether it has a valuerange that excludes zero?
    ConstantInt *Len = dyn_cast<ConstantInt>(MI->getLength());
    if (!Len || Len->isZero()) return false;

    if (MI->getDestAddressSpace() == 0)
      if (GetUnderlyingObject(MI->getRawDest()) == Ptr)
        return true;
    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI))
      if (MTI->getSourceAddressSpace() == 0)
        if (GetUnderlyingObject(MTI->getRawSource()) == Ptr)
          return true;
  }
  return false;
}

bool LazyValueInfoCache::solveBlockValueNonLocal(LVILatticeVal &BBLV,
                                                 Value *Val, BasicBlock *BB) {
  LVILatticeVal Result;  // Start Undefined.

  // If this is a pointer, and there's a load from that pointer in this BB,
  // then we know that the pointer can't be NULL.
  bool NotNull = false;
  if (Val->getType()->isPointerTy()) {
    if (isKnownNonNull(Val)) {
      NotNull = true;
    } else {
      Value *UnderlyingVal = GetUnderlyingObject(Val);
      // If 'GetUnderlyingObject' didn't converge, skip it. It won't converge
      // inside InstructionDereferencesPointer either.
      if (UnderlyingVal == GetUnderlyingObject(UnderlyingVal, NULL, 1)) {
        for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
             BI != BE; ++BI) {
          if (InstructionDereferencesPointer(BI, UnderlyingVal)) {
            NotNull = true;
            break;
          }
        }
      }
    }
  }

  // If this is the entry block, we must be asking about an argument.  The
  // value is overdefined.
  if (BB == &BB->getParent()->getEntryBlock()) {
    assert(isa<Argument>(Val) && "Unknown live-in to the entry block");
    if (NotNull) {
      PointerType *PTy = cast<PointerType>(Val->getType());
      Result = LVILatticeVal::getNot(ConstantPointerNull::get(PTy));
    } else {
      Result.markOverdefined();
    }
    BBLV = Result;
    return true;
  }

  // Loop over all of our predecessors, merging what we know from them into
  // result.
  bool EdgesMissing = false;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    LVILatticeVal EdgeResult;
    EdgesMissing |= !getEdgeValue(Val, *PI, BB, EdgeResult);
    if (EdgesMissing)
      continue;

    Result.mergeIn(EdgeResult);

    // If we hit overdefined, exit early.  The BlockVals entry is already set
    // to overdefined.
    if (Result.isOverdefined()) {
      DEBUG(dbgs() << " compute BB '" << BB->getName()
            << "' - overdefined because of pred.\n");
      // If we previously determined that this is a pointer that can't be null
      // then return that rather than giving up entirely.
      if (NotNull) {
        PointerType *PTy = cast<PointerType>(Val->getType());
        Result = LVILatticeVal::getNot(ConstantPointerNull::get(PTy));
      }
      
      BBLV = Result;
      return true;
    }
  }
  if (EdgesMissing)
    return false;

  // Return the merged value, which is more precise than 'overdefined'.
  assert(!Result.isOverdefined());
  BBLV = Result;
  return true;
}
  
bool LazyValueInfoCache::solveBlockValuePHINode(LVILatticeVal &BBLV,
                                                PHINode *PN, BasicBlock *BB) {
  LVILatticeVal Result;  // Start Undefined.

  // Loop over all of our predecessors, merging what we know from them into
  // result.
  bool EdgesMissing = false;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *PhiBB = PN->getIncomingBlock(i);
    Value *PhiVal = PN->getIncomingValue(i);
    LVILatticeVal EdgeResult;
    EdgesMissing |= !getEdgeValue(PhiVal, PhiBB, BB, EdgeResult);
    if (EdgesMissing)
      continue;

    Result.mergeIn(EdgeResult);

    // If we hit overdefined, exit early.  The BlockVals entry is already set
    // to overdefined.
    if (Result.isOverdefined()) {
      DEBUG(dbgs() << " compute BB '" << BB->getName()
            << "' - overdefined because of pred.\n");
      
      BBLV = Result;
      return true;
    }
  }
  if (EdgesMissing)
    return false;

  // Return the merged value, which is more precise than 'overdefined'.
  assert(!Result.isOverdefined() && "Possible PHI in entry block?");
  BBLV = Result;
  return true;
}

bool LazyValueInfoCache::solveBlockValueConstantRange(LVILatticeVal &BBLV,
                                                      Instruction *BBI,
                                                      BasicBlock *BB) {
  // Figure out the range of the LHS.  If that fails, bail.
  if (!hasBlockValue(BBI->getOperand(0), BB)) {
    BlockValueStack.push(std::make_pair(BB, BBI->getOperand(0)));
    return false;
  }

  LVILatticeVal LHSVal = getBlockValue(BBI->getOperand(0), BB);
  if (!LHSVal.isConstantRange()) {
    BBLV.markOverdefined();
    return true;
  }
  
  ConstantRange LHSRange = LHSVal.getConstantRange();
  ConstantRange RHSRange(1);
  IntegerType *ResultTy = cast<IntegerType>(BBI->getType());
  if (isa<BinaryOperator>(BBI)) {
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(BBI->getOperand(1))) {
      RHSRange = ConstantRange(RHS->getValue());
    } else {
      BBLV.markOverdefined();
      return true;
    }
  }

  // NOTE: We're currently limited by the set of operations that ConstantRange
  // can evaluate symbolically.  Enhancing that set will allows us to analyze
  // more definitions.
  LVILatticeVal Result;
  switch (BBI->getOpcode()) {
  case Instruction::Add:
    Result.markConstantRange(LHSRange.add(RHSRange));
    break;
  case Instruction::Sub:
    Result.markConstantRange(LHSRange.sub(RHSRange));
    break;
  case Instruction::Mul:
    Result.markConstantRange(LHSRange.multiply(RHSRange));
    break;
  case Instruction::UDiv:
    Result.markConstantRange(LHSRange.udiv(RHSRange));
    break;
  case Instruction::Shl:
    Result.markConstantRange(LHSRange.shl(RHSRange));
    break;
  case Instruction::LShr:
    Result.markConstantRange(LHSRange.lshr(RHSRange));
    break;
  case Instruction::Trunc:
    Result.markConstantRange(LHSRange.truncate(ResultTy->getBitWidth()));
    break;
  case Instruction::SExt:
    Result.markConstantRange(LHSRange.signExtend(ResultTy->getBitWidth()));
    break;
  case Instruction::ZExt:
    Result.markConstantRange(LHSRange.zeroExtend(ResultTy->getBitWidth()));
    break;
  case Instruction::BitCast:
    Result.markConstantRange(LHSRange);
    break;
  case Instruction::And:
    Result.markConstantRange(LHSRange.binaryAnd(RHSRange));
    break;
  case Instruction::Or:
    Result.markConstantRange(LHSRange.binaryOr(RHSRange));
    break;
  
  // Unhandled instructions are overdefined.
  default:
    DEBUG(dbgs() << " compute BB '" << BB->getName()
                 << "' - overdefined because inst def found.\n");
    Result.markOverdefined();
    break;
  }
  
  BBLV = Result;
  return true;
}

/// \brief Compute the value of Val on the edge BBFrom -> BBTo. Returns false if
/// Val is not constrained on the edge.
static bool getEdgeValueLocal(Value *Val, BasicBlock *BBFrom,
                              BasicBlock *BBTo, LVILatticeVal &Result) {
  // TODO: Handle more complex conditionals.  If (v == 0 || v2 < 1) is false, we
  // know that v != 0.
  if (BranchInst *BI = dyn_cast<BranchInst>(BBFrom->getTerminator())) {
    // If this is a conditional branch and only one successor goes to BBTo, then
    // we maybe able to infer something from the condition. 
    if (BI->isConditional() &&
        BI->getSuccessor(0) != BI->getSuccessor(1)) {
      bool isTrueDest = BI->getSuccessor(0) == BBTo;
      assert(BI->getSuccessor(!isTrueDest) == BBTo &&
             "BBTo isn't a successor of BBFrom");
      
      // If V is the condition of the branch itself, then we know exactly what
      // it is.
      if (BI->getCondition() == Val) {
        Result = LVILatticeVal::get(ConstantInt::get(
                              Type::getInt1Ty(Val->getContext()), isTrueDest));
        return true;
      }
      
      // If the condition of the branch is an equality comparison, we may be
      // able to infer the value.
      ICmpInst *ICI = dyn_cast<ICmpInst>(BI->getCondition());
      if (ICI && isa<Constant>(ICI->getOperand(1))) {
        if (ICI->isEquality() && ICI->getOperand(0) == Val) {
          // We know that V has the RHS constant if this is a true SETEQ or
          // false SETNE. 
          if (isTrueDest == (ICI->getPredicate() == ICmpInst::ICMP_EQ))
            Result = LVILatticeVal::get(cast<Constant>(ICI->getOperand(1)));
          else
            Result = LVILatticeVal::getNot(cast<Constant>(ICI->getOperand(1)));
          return true;
        }

        // Recognize the range checking idiom that InstCombine produces.
        // (X-C1) u< C2 --> [C1, C1+C2)
        ConstantInt *NegOffset = 0;
        if (ICI->getPredicate() == ICmpInst::ICMP_ULT)
          match(ICI->getOperand(0), m_Add(m_Specific(Val),
                                          m_ConstantInt(NegOffset)));

        ConstantInt *CI = dyn_cast<ConstantInt>(ICI->getOperand(1));
        if (CI && (ICI->getOperand(0) == Val || NegOffset)) {
          // Calculate the range of values that would satisfy the comparison.
          ConstantRange CmpRange(CI->getValue());
          ConstantRange TrueValues =
            ConstantRange::makeICmpRegion(ICI->getPredicate(), CmpRange);

          if (NegOffset) // Apply the offset from above.
            TrueValues = TrueValues.subtract(NegOffset->getValue());

          // If we're interested in the false dest, invert the condition.
          if (!isTrueDest) TrueValues = TrueValues.inverse();

          Result = LVILatticeVal::getRange(TrueValues);
          return true;
        }
      }
    }
  }

  // If the edge was formed by a switch on the value, then we may know exactly
  // what it is.
  if (SwitchInst *SI = dyn_cast<SwitchInst>(BBFrom->getTerminator())) {
    if (SI->getCondition() != Val)
      return false;

    bool DefaultCase = SI->getDefaultDest() == BBTo;
    unsigned BitWidth = Val->getType()->getIntegerBitWidth();
    ConstantRange EdgesVals(BitWidth, DefaultCase/*isFullSet*/);

    for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
         i != e; ++i) {
      ConstantRange EdgeVal(i.getCaseValue()->getValue());
      if (DefaultCase) {
        // It is possible that the default destination is the destination of
        // some cases. There is no need to perform difference for those cases.
        if (i.getCaseSuccessor() != BBTo)
          EdgesVals = EdgesVals.difference(EdgeVal);
      } else if (i.getCaseSuccessor() == BBTo)
        EdgesVals = EdgesVals.unionWith(EdgeVal);
    }
    Result = LVILatticeVal::getRange(EdgesVals);
    return true;
  }
  return false;
}

/// \brief Compute the value of Val on the edge BBFrom -> BBTo, or the value at
/// the basic block if the edge does not constraint Val.
bool LazyValueInfoCache::getEdgeValue(Value *Val, BasicBlock *BBFrom,
                                      BasicBlock *BBTo, LVILatticeVal &Result) {
  // If already a constant, there is nothing to compute.
  if (Constant *VC = dyn_cast<Constant>(Val)) {
    Result = LVILatticeVal::get(VC);
    return true;
  }

  if (getEdgeValueLocal(Val, BBFrom, BBTo, Result)) {
    if (!Result.isConstantRange() ||
      Result.getConstantRange().getSingleElement())
      return true;

    // FIXME: this check should be moved to the beginning of the function when
    // LVI better supports recursive values. Even for the single value case, we
    // can intersect to detect dead code (an empty range).
    if (!hasBlockValue(Val, BBFrom)) {
      BlockValueStack.push(std::make_pair(BBFrom, Val));
      return false;
    }

    // Try to intersect ranges of the BB and the constraint on the edge.
    LVILatticeVal InBlock = getBlockValue(Val, BBFrom);
    if (!InBlock.isConstantRange())
      return true;

    ConstantRange Range =
      Result.getConstantRange().intersectWith(InBlock.getConstantRange());
    Result = LVILatticeVal::getRange(Range);
    return true;
  }

  if (!hasBlockValue(Val, BBFrom)) {
    BlockValueStack.push(std::make_pair(BBFrom, Val));
    return false;
  }

  // if we couldn't compute the value on the edge, use the value from the BB
  Result = getBlockValue(Val, BBFrom);
  return true;
}

LVILatticeVal LazyValueInfoCache::getValueInBlock(Value *V, BasicBlock *BB) {
  DEBUG(dbgs() << "LVI Getting block end value " << *V << " at '"
        << BB->getName() << "'\n");
  
  BlockValueStack.push(std::make_pair(BB, V));
  solve();
  LVILatticeVal Result = getBlockValue(V, BB);

  DEBUG(dbgs() << "  Result = " << Result << "\n");
  return Result;
}

LVILatticeVal LazyValueInfoCache::
getValueOnEdge(Value *V, BasicBlock *FromBB, BasicBlock *ToBB) {
  DEBUG(dbgs() << "LVI Getting edge value " << *V << " from '"
        << FromBB->getName() << "' to '" << ToBB->getName() << "'\n");
  
  LVILatticeVal Result;
  if (!getEdgeValue(V, FromBB, ToBB, Result)) {
    solve();
    bool WasFastQuery = getEdgeValue(V, FromBB, ToBB, Result);
    (void)WasFastQuery;
    assert(WasFastQuery && "More work to do after problem solved?");
  }

  DEBUG(dbgs() << "  Result = " << Result << "\n");
  return Result;
}

void LazyValueInfoCache::threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc,
                                    BasicBlock *NewSucc) {
  // When an edge in the graph has been threaded, values that we could not 
  // determine a value for before (i.e. were marked overdefined) may be possible
  // to solve now.  We do NOT try to proactively update these values.  Instead,
  // we clear their entries from the cache, and allow lazy updating to recompute
  // them when needed.
  
  // The updating process is fairly simple: we need to dropped cached info
  // for all values that were marked overdefined in OldSucc, and for those same
  // values in any successor of OldSucc (except NewSucc) in which they were
  // also marked overdefined.
  std::vector<BasicBlock*> worklist;
  worklist.push_back(OldSucc);
  
  DenseSet<Value*> ClearSet;
  for (DenseSet<OverDefinedPairTy>::iterator I = OverDefinedCache.begin(),
       E = OverDefinedCache.end(); I != E; ++I) {
    if (I->first == OldSucc)
      ClearSet.insert(I->second);
  }
  
  // Use a worklist to perform a depth-first search of OldSucc's successors.
  // NOTE: We do not need a visited list since any blocks we have already
  // visited will have had their overdefined markers cleared already, and we
  // thus won't loop to their successors.
  while (!worklist.empty()) {
    BasicBlock *ToUpdate = worklist.back();
    worklist.pop_back();
    
    // Skip blocks only accessible through NewSucc.
    if (ToUpdate == NewSucc) continue;
    
    bool changed = false;
    for (DenseSet<Value*>::iterator I = ClearSet.begin(), E = ClearSet.end();
         I != E; ++I) {
      // If a value was marked overdefined in OldSucc, and is here too...
      DenseSet<OverDefinedPairTy>::iterator OI =
        OverDefinedCache.find(std::make_pair(ToUpdate, *I));
      if (OI == OverDefinedCache.end()) continue;

      // Remove it from the caches.
      ValueCacheEntryTy &Entry = ValueCache[LVIValueHandle(*I, this)];
      ValueCacheEntryTy::iterator CI = Entry.find(ToUpdate);

      assert(CI != Entry.end() && "Couldn't find entry to update?");
      Entry.erase(CI);
      OverDefinedCache.erase(OI);

      // If we removed anything, then we potentially need to update 
      // blocks successors too.
      changed = true;
    }

    if (!changed) continue;
    
    worklist.insert(worklist.end(), succ_begin(ToUpdate), succ_end(ToUpdate));
  }
}

//===----------------------------------------------------------------------===//
//                            LazyValueInfo Impl
//===----------------------------------------------------------------------===//

/// getCache - This lazily constructs the LazyValueInfoCache.
static LazyValueInfoCache &getCache(void *&PImpl) {
  if (!PImpl)
    PImpl = new LazyValueInfoCache();
  return *static_cast<LazyValueInfoCache*>(PImpl);
}

bool LazyValueInfo::runOnFunction(Function &F) {
  if (PImpl)
    getCache(PImpl).clear();

  TD = getAnalysisIfAvailable<DataLayout>();
  TLI = &getAnalysis<TargetLibraryInfo>();

  // Fully lazy.
  return false;
}

void LazyValueInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TargetLibraryInfo>();
}

void LazyValueInfo::releaseMemory() {
  // If the cache was allocated, free it.
  if (PImpl) {
    delete &getCache(PImpl);
    PImpl = 0;
  }
}

Constant *LazyValueInfo::getConstant(Value *V, BasicBlock *BB) {
  LVILatticeVal Result = getCache(PImpl).getValueInBlock(V, BB);
  
  if (Result.isConstant())
    return Result.getConstant();
  if (Result.isConstantRange()) {
    ConstantRange CR = Result.getConstantRange();
    if (const APInt *SingleVal = CR.getSingleElement())
      return ConstantInt::get(V->getContext(), *SingleVal);
  }
  return 0;
}

/// getConstantOnEdge - Determine whether the specified value is known to be a
/// constant on the specified edge.  Return null if not.
Constant *LazyValueInfo::getConstantOnEdge(Value *V, BasicBlock *FromBB,
                                           BasicBlock *ToBB) {
  LVILatticeVal Result = getCache(PImpl).getValueOnEdge(V, FromBB, ToBB);
  
  if (Result.isConstant())
    return Result.getConstant();
  if (Result.isConstantRange()) {
    ConstantRange CR = Result.getConstantRange();
    if (const APInt *SingleVal = CR.getSingleElement())
      return ConstantInt::get(V->getContext(), *SingleVal);
  }
  return 0;
}

/// getPredicateOnEdge - Determine whether the specified value comparison
/// with a constant is known to be true or false on the specified CFG edge.
/// Pred is a CmpInst predicate.
LazyValueInfo::Tristate
LazyValueInfo::getPredicateOnEdge(unsigned Pred, Value *V, Constant *C,
                                  BasicBlock *FromBB, BasicBlock *ToBB) {
  LVILatticeVal Result = getCache(PImpl).getValueOnEdge(V, FromBB, ToBB);
  
  // If we know the value is a constant, evaluate the conditional.
  Constant *Res = 0;
  if (Result.isConstant()) {
    Res = ConstantFoldCompareInstOperands(Pred, Result.getConstant(), C, TD,
                                          TLI);
    if (ConstantInt *ResCI = dyn_cast<ConstantInt>(Res))
      return ResCI->isZero() ? False : True;
    return Unknown;
  }
  
  if (Result.isConstantRange()) {
    ConstantInt *CI = dyn_cast<ConstantInt>(C);
    if (!CI) return Unknown;
    
    ConstantRange CR = Result.getConstantRange();
    if (Pred == ICmpInst::ICMP_EQ) {
      if (!CR.contains(CI->getValue()))
        return False;
      
      if (CR.isSingleElement() && CR.contains(CI->getValue()))
        return True;
    } else if (Pred == ICmpInst::ICMP_NE) {
      if (!CR.contains(CI->getValue()))
        return True;
      
      if (CR.isSingleElement() && CR.contains(CI->getValue()))
        return False;
    }
    
    // Handle more complex predicates.
    ConstantRange TrueValues =
        ICmpInst::makeConstantRange((ICmpInst::Predicate)Pred, CI->getValue());
    if (TrueValues.contains(CR))
      return True;
    if (TrueValues.inverse().contains(CR))
      return False;
    return Unknown;
  }
  
  if (Result.isNotConstant()) {
    // If this is an equality comparison, we can try to fold it knowing that
    // "V != C1".
    if (Pred == ICmpInst::ICMP_EQ) {
      // !C1 == C -> false iff C1 == C.
      Res = ConstantFoldCompareInstOperands(ICmpInst::ICMP_NE,
                                            Result.getNotConstant(), C, TD,
                                            TLI);
      if (Res->isNullValue())
        return False;
    } else if (Pred == ICmpInst::ICMP_NE) {
      // !C1 != C -> true iff C1 == C.
      Res = ConstantFoldCompareInstOperands(ICmpInst::ICMP_NE,
                                            Result.getNotConstant(), C, TD,
                                            TLI);
      if (Res->isNullValue())
        return True;
    }
    return Unknown;
  }
  
  return Unknown;
}

void LazyValueInfo::threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc,
                               BasicBlock *NewSucc) {
  if (PImpl) getCache(PImpl).threadEdge(PredBB, OldSucc, NewSucc);
}

void LazyValueInfo::eraseBlock(BasicBlock *BB) {
  if (PImpl) getCache(PImpl).eraseBlock(BB);
}
