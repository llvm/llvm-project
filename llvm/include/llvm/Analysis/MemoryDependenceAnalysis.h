//===- llvm/Analysis/MemoryDependenceAnalysis.h - Memory Deps  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MemoryDependenceAnalysis analysis pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORY_DEPENDENCE_H
#define LLVM_ANALYSIS_MEMORY_DEPENDENCE_H

#include "llvm/BasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/PointerIntPair.h"

namespace llvm {
  class Function;
  class FunctionPass;
  class Instruction;
  class CallSite;
  class AliasAnalysis;
  class TargetData;
  class MemoryDependenceAnalysis;
  class PredIteratorCache;
  
  /// MemDepResult - A memory dependence query can return one of three different
  /// answers, described below.
  class MemDepResult {
    enum DepType {
      /// Invalid - Clients of MemDep never see this.
      Invalid = 0,
      
      /// Clobber - This is a dependence on the specified instruction which
      /// clobbers the desired value.  The pointer member of the MemDepResult
      /// pair holds the instruction that clobbers the memory.  For example,
      /// this occurs when we see a may-aliased store to the memory location we
      /// care about.
      Clobber,

      /// Def - This is a dependence on the specified instruction which
      /// defines/produces the desired memory location.  The pointer member of
      /// the MemDepResult pair holds the instruction that defines the memory.
      /// Cases of interest:
      ///   1. This could be a load or store for dependence queries on
      ///      load/store.  The value loaded or stored is the produced value.
      ///      Note that the pointer operand may be different than that of the
      ///      queried pointer due to must aliases and phi translation.  Note
      ///      that the def may not be the same type as the query, the pointers
      ///      may just be must aliases.
      ///   2. For loads and stores, this could be an allocation instruction. In
      ///      this case, the load is loading an undef value or a store is the
      ///      first store to (that part of) the allocation.
      ///   3. Dependence queries on calls return Def only when they are
      ///      readonly calls with identical callees and no intervening
      ///      clobbers.  No validation is done that the operands to the calls
      ///      are the same.
      Def,
      
      /// NonLocal - This marker indicates that the query has no dependency in
      /// the specified block.  To find out more, the client should query other
      /// predecessor blocks.
      NonLocal
    };
    typedef PointerIntPair<Instruction*, 2, DepType> PairTy;
    PairTy Value;
    explicit MemDepResult(PairTy V) : Value(V) {}
  public:
    MemDepResult() : Value(0, Invalid) {}
    
    /// get methods: These are static ctor methods for creating various
    /// MemDepResult kinds.
    static MemDepResult getDef(Instruction *Inst) {
      return MemDepResult(PairTy(Inst, Def));
    }
    static MemDepResult getClobber(Instruction *Inst) {
      return MemDepResult(PairTy(Inst, Clobber));
    }
    static MemDepResult getNonLocal() {
      return MemDepResult(PairTy(0, NonLocal));
    }

    /// isClobber - Return true if this MemDepResult represents a query that is
    /// a instruction clobber dependency.
    bool isClobber() const { return Value.getInt() == Clobber; }

    /// isDef - Return true if this MemDepResult represents a query that is
    /// a instruction definition dependency.
    bool isDef() const { return Value.getInt() == Def; }
    
    /// isNonLocal - Return true if this MemDepResult represents an query that
    /// is transparent to the start of the block, but where a non-local hasn't
    /// been done.
    bool isNonLocal() const { return Value.getInt() == NonLocal; }
    
    /// getInst() - If this is a normal dependency, return the instruction that
    /// is depended on.  Otherwise, return null.
    Instruction *getInst() const { return Value.getPointer(); }
    
    bool operator==(const MemDepResult &M) const { return Value == M.Value; }
    bool operator!=(const MemDepResult &M) const { return Value != M.Value; }
    bool operator<(const MemDepResult &M) const { return Value < M.Value; }
    bool operator>(const MemDepResult &M) const { return Value > M.Value; }
  private:
    friend class MemoryDependenceAnalysis;
    /// Dirty - Entries with this marker occur in a LocalDeps map or
    /// NonLocalDeps map when the instruction they previously referenced was
    /// removed from MemDep.  In either case, the entry may include an
    /// instruction pointer.  If so, the pointer is an instruction in the
    /// block where scanning can start from, saving some work.
    ///
    /// In a default-constructed MemDepResult object, the type will be Dirty
    /// and the instruction pointer will be null.
    ///
         
    /// isDirty - Return true if this is a MemDepResult in its dirty/invalid.
    /// state.
    bool isDirty() const { return Value.getInt() == Invalid; }
    
    static MemDepResult getDirty(Instruction *Inst) {
      return MemDepResult(PairTy(Inst, Invalid));
    }
  };

  /// MemoryDependenceAnalysis - This is an analysis that determines, for a
  /// given memory operation, what preceding memory operations it depends on.
  /// It builds on alias analysis information, and tries to provide a lazy,
  /// caching interface to a common kind of alias information query.
  ///
  /// The dependency information returned is somewhat unusual, but is pragmatic.
  /// If queried about a store or call that might modify memory, the analysis
  /// will return the instruction[s] that may either load from that memory or
  /// store to it.  If queried with a load or call that can never modify memory,
  /// the analysis will return calls and stores that might modify the pointer,
  /// but generally does not return loads unless a) they are volatile, or
  /// b) they load from *must-aliased* pointers.  Returning a dependence on
  /// must-alias'd pointers instead of all pointers interacts well with the
  /// internal caching mechanism.
  ///
  class MemoryDependenceAnalysis : public FunctionPass {
    // A map from instructions to their dependency.
    typedef DenseMap<Instruction*, MemDepResult> LocalDepMapType;
    LocalDepMapType LocalDeps;

  public:
    typedef std::pair<BasicBlock*, MemDepResult> NonLocalDepEntry;
    typedef std::vector<NonLocalDepEntry> NonLocalDepInfo;
  private:
    /// ValueIsLoadPair - This is a pair<Value*, bool> where the bool is true if
    /// the dependence is a read only dependence, false if read/write.
    typedef PointerIntPair<Value*, 1, bool> ValueIsLoadPair;

    /// BBSkipFirstBlockPair - This pair is used when caching information for a
    /// block.  If the pointer is null, the cache value is not a full query that
    /// starts at the specified block.  If non-null, the bool indicates whether
    /// or not the contents of the block was skipped.
    typedef PointerIntPair<BasicBlock*, 1, bool> BBSkipFirstBlockPair;

    /// CachedNonLocalPointerInfo - This map stores the cached results of doing
    /// a pointer lookup at the bottom of a block.  The key of this map is the
    /// pointer+isload bit, the value is a list of <bb->result> mappings.
    typedef DenseMap<ValueIsLoadPair, std::pair<BBSkipFirstBlockPair, 
                  NonLocalDepInfo> > CachedNonLocalPointerInfo;
    CachedNonLocalPointerInfo NonLocalPointerDeps;

    // A map from instructions to their non-local pointer dependencies.
    // The elements of the SmallPtrSet are ValueIsLoadPair's.
    typedef DenseMap<Instruction*, 
                     SmallPtrSet<void*, 4> > ReverseNonLocalPtrDepTy;
    ReverseNonLocalPtrDepTy ReverseNonLocalPtrDeps;

    
    /// PerInstNLInfo - This is the instruction we keep for each cached access
    /// that we have for an instruction.  The pointer is an owning pointer and
    /// the bool indicates whether we have any dirty bits in the set.
    typedef std::pair<NonLocalDepInfo, bool> PerInstNLInfo;
    
    // A map from instructions to their non-local dependencies.
    typedef DenseMap<Instruction*, PerInstNLInfo> NonLocalDepMapType;
      
    NonLocalDepMapType NonLocalDeps;
    
    // A reverse mapping from dependencies to the dependees.  This is
    // used when removing instructions to keep the cache coherent.
    typedef DenseMap<Instruction*,
                     SmallPtrSet<Instruction*, 4> > ReverseDepMapType;
    ReverseDepMapType ReverseLocalDeps;
    
    // A reverse mapping form dependencies to the non-local dependees.
    ReverseDepMapType ReverseNonLocalDeps;
    
    /// Current AA implementation, just a cache.
    AliasAnalysis *AA;
    TargetData *TD;
    OwningPtr<PredIteratorCache> PredCache;
  public:
    MemoryDependenceAnalysis();
    ~MemoryDependenceAnalysis();
    static char ID;

    /// Pass Implementation stuff.  This doesn't do any analysis eagerly.
    bool runOnFunction(Function &);
    
    /// Clean up memory in between runs
    void releaseMemory();
    
    /// getAnalysisUsage - Does not modify anything.  It uses Value Numbering
    /// and Alias Analysis.
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    
    /// getDependency - Return the instruction on which a memory operation
    /// depends.  See the class comment for more details.  It is illegal to call
    /// this on non-memory instructions.
    MemDepResult getDependency(Instruction *QueryInst);

    /// getNonLocalCallDependency - Perform a full dependency query for the
    /// specified call, returning the set of blocks that the value is
    /// potentially live across.  The returned set of results will include a
    /// "NonLocal" result for all blocks where the value is live across.
    ///
    /// This method assumes the instruction returns a "NonLocal" dependency
    /// within its own block.
    ///
    /// This returns a reference to an internal data structure that may be
    /// invalidated on the next non-local query or when an instruction is
    /// removed.  Clients must copy this data if they want it around longer than
    /// that.
    const NonLocalDepInfo &getNonLocalCallDependency(CallSite QueryCS);
    
    
    /// getNonLocalPointerDependency - Perform a full dependency query for an
    /// access to the specified (non-volatile) memory location, returning the
    /// set of instructions that either define or clobber the value.
    ///
    /// This method assumes the pointer has a "NonLocal" dependency within BB.
    void getNonLocalPointerDependency(Value *Pointer, bool isLoad,
                                      BasicBlock *BB,
                                     SmallVectorImpl<NonLocalDepEntry> &Result);
    
    /// removeInstruction - Remove an instruction from the dependence analysis,
    /// updating the dependence of instructions that previously depended on it.
    void removeInstruction(Instruction *InstToRemove);
    
    /// invalidateCachedPointerInfo - This method is used to invalidate cached
    /// information about the specified pointer, because it may be too
    /// conservative in memdep.  This is an optional call that can be used when
    /// the client detects an equivalence between the pointer and some other
    /// value and replaces the other value with ptr. This can make Ptr available
    /// in more places that cached info does not necessarily keep.
    void invalidateCachedPointerInfo(Value *Ptr);
    
  private:
    MemDepResult getPointerDependencyFrom(Value *Pointer, uint64_t MemSize,
                                          bool isLoad, 
                                          BasicBlock::iterator ScanIt,
                                          BasicBlock *BB);
    MemDepResult getCallSiteDependencyFrom(CallSite C, bool isReadOnlyCall,
                                           BasicBlock::iterator ScanIt,
                                           BasicBlock *BB);
    bool getNonLocalPointerDepFromBB(Value *Pointer, uint64_t Size,
                                     bool isLoad, BasicBlock *BB,
                                     SmallVectorImpl<NonLocalDepEntry> &Result,
                                     DenseMap<BasicBlock*, Value*> &Visited,
                                     bool SkipFirstBlock = false);
    MemDepResult GetNonLocalInfoForBlock(Value *Pointer, uint64_t PointeeSize,
                                         bool isLoad, BasicBlock *BB,
                                         NonLocalDepInfo *Cache,
                                         unsigned NumSortedEntries);

    void RemoveCachedNonLocalPointerDependencies(ValueIsLoadPair P);
    
    /// verifyRemoved - Verify that the specified instruction does not occur
    /// in our internal data structures.
    void verifyRemoved(Instruction *Inst) const;
    
  };

} // End llvm namespace

#endif
