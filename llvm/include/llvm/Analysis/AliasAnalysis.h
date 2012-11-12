//===- llvm/Analysis/AliasAnalysis.h - Alias Analysis Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the generic AliasAnalysis interface, which is used as the
// common interface used by all clients of alias analysis information, and
// implemented by all alias analysis implementations.  Mod/Ref information is
// also captured by this interface.
//
// Implementations of this interface must implement the various virtual methods,
// which automatically provides functionality for the entire suite of client
// APIs.
//
// This API identifies memory regions with the Location class. The pointer
// component specifies the base memory address of the region. The Size specifies
// the maximum size (in address units) of the memory region, or UnknownSize if
// the size is not known. The TBAA tag identifies the "type" of the memory
// reference; see the TypeBasedAliasAnalysis class for details.
//
// Some non-obvious details include:
//  - Pointers that point to two completely different objects in memory never
//    alias, regardless of the value of the Size component.
//  - NoAlias doesn't imply inequal pointers. The most obvious example of this
//    is two pointers to constant memory. Even if they are equal, constant
//    memory is never stored to, so there will never be any dependencies.
//    In this and other situations, the pointers may be both NoAlias and
//    MustAlias at the same time. The current API can only return one result,
//    though this is rarely a problem in practice.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIAS_ANALYSIS_H
#define LLVM_ANALYSIS_ALIAS_ANALYSIS_H

#include "llvm/Support/CallSite.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class LoadInst;
class StoreInst;
class VAArgInst;
class DataLayout;
class TargetLibraryInfo;
class Pass;
class AnalysisUsage;
class MemTransferInst;
class MemIntrinsic;
class DominatorTree;

class AliasAnalysis {
protected:
  const DataLayout *TD;
  const TargetLibraryInfo *TLI;

private:
  AliasAnalysis *AA;       // Previous Alias Analysis to chain to.

protected:
  /// InitializeAliasAnalysis - Subclasses must call this method to initialize
  /// the AliasAnalysis interface before any other methods are called.  This is
  /// typically called by the run* methods of these subclasses.  This may be
  /// called multiple times.
  ///
  void InitializeAliasAnalysis(Pass *P);

  /// getAnalysisUsage - All alias analysis implementations should invoke this
  /// directly (using AliasAnalysis::getAnalysisUsage(AU)).
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

public:
  static char ID; // Class identification, replacement for typeinfo
  AliasAnalysis() : TD(0), TLI(0), AA(0) {}
  virtual ~AliasAnalysis();  // We want to be subclassed

  /// UnknownSize - This is a special value which can be used with the
  /// size arguments in alias queries to indicate that the caller does not
  /// know the sizes of the potential memory references.
  static uint64_t const UnknownSize = ~UINT64_C(0);

  /// getDataLayout - Return a pointer to the current DataLayout object, or
  /// null if no DataLayout object is available.
  ///
  const DataLayout *getDataLayout() const { return TD; }

  /// getTargetLibraryInfo - Return a pointer to the current TargetLibraryInfo
  /// object, or null if no TargetLibraryInfo object is available.
  ///
  const TargetLibraryInfo *getTargetLibraryInfo() const { return TLI; }

  /// getTypeStoreSize - Return the DataLayout store size for the given type,
  /// if known, or a conservative value otherwise.
  ///
  uint64_t getTypeStoreSize(Type *Ty);

  //===--------------------------------------------------------------------===//
  /// Alias Queries...
  ///

  /// Location - A description of a memory location.
  struct Location {
    /// Ptr - The address of the start of the location.
    const Value *Ptr;
    /// Size - The maximum size of the location, in address-units, or
    /// UnknownSize if the size is not known.  Note that an unknown size does
    /// not mean the pointer aliases the entire virtual address space, because
    /// there are restrictions on stepping out of one object and into another.
    /// See http://llvm.org/docs/LangRef.html#pointeraliasing
    uint64_t Size;
    /// TBAATag - The metadata node which describes the TBAA type of
    /// the location, or null if there is no known unique tag.
    const MDNode *TBAATag;

    explicit Location(const Value *P = 0, uint64_t S = UnknownSize,
                      const MDNode *N = 0)
      : Ptr(P), Size(S), TBAATag(N) {}

    Location getWithNewPtr(const Value *NewPtr) const {
      Location Copy(*this);
      Copy.Ptr = NewPtr;
      return Copy;
    }

    Location getWithNewSize(uint64_t NewSize) const {
      Location Copy(*this);
      Copy.Size = NewSize;
      return Copy;
    }

    Location getWithoutTBAATag() const {
      Location Copy(*this);
      Copy.TBAATag = 0;
      return Copy;
    }
  };

  /// getLocation - Fill in Loc with information about the memory reference by
  /// the given instruction.
  Location getLocation(const LoadInst *LI);
  Location getLocation(const StoreInst *SI);
  Location getLocation(const VAArgInst *VI);
  Location getLocation(const AtomicCmpXchgInst *CXI);
  Location getLocation(const AtomicRMWInst *RMWI);
  static Location getLocationForSource(const MemTransferInst *MTI);
  static Location getLocationForDest(const MemIntrinsic *MI);

  /// Alias analysis result - Either we know for sure that it does not alias, we
  /// know for sure it must alias, or we don't know anything: The two pointers
  /// _might_ alias.  This enum is designed so you can do things like:
  ///     if (AA.alias(P1, P2)) { ... }
  /// to check to see if two pointers might alias.
  ///
  /// See docs/AliasAnalysis.html for more information on the specific meanings
  /// of these values.
  ///
  enum AliasResult {
    NoAlias = 0,        ///< No dependencies.
    MayAlias,           ///< Anything goes.
    PartialAlias,       ///< Pointers differ, but pointees overlap.
    MustAlias           ///< Pointers are equal.
  };

  /// alias - The main low level interface to the alias analysis implementation.
  /// Returns an AliasResult indicating whether the two pointers are aliased to
  /// each other.  This is the interface that must be implemented by specific
  /// alias analysis implementations.
  virtual AliasResult alias(const Location &LocA, const Location &LocB);

  /// alias - A convenience wrapper.
  AliasResult alias(const Value *V1, uint64_t V1Size,
                    const Value *V2, uint64_t V2Size) {
    return alias(Location(V1, V1Size), Location(V2, V2Size));
  }

  /// alias - A convenience wrapper.
  AliasResult alias(const Value *V1, const Value *V2) {
    return alias(V1, UnknownSize, V2, UnknownSize);
  }

  /// isNoAlias - A trivial helper function to check to see if the specified
  /// pointers are no-alias.
  bool isNoAlias(const Location &LocA, const Location &LocB) {
    return alias(LocA, LocB) == NoAlias;
  }

  /// isNoAlias - A convenience wrapper.
  bool isNoAlias(const Value *V1, uint64_t V1Size,
                 const Value *V2, uint64_t V2Size) {
    return isNoAlias(Location(V1, V1Size), Location(V2, V2Size));
  }
  
  /// isNoAlias - A convenience wrapper.
  bool isNoAlias(const Value *V1, const Value *V2) {
    return isNoAlias(Location(V1), Location(V2));
  }
  
  /// isMustAlias - A convenience wrapper.
  bool isMustAlias(const Location &LocA, const Location &LocB) {
    return alias(LocA, LocB) == MustAlias;
  }

  /// isMustAlias - A convenience wrapper.
  bool isMustAlias(const Value *V1, const Value *V2) {
    return alias(V1, 1, V2, 1) == MustAlias;
  }
  
  /// pointsToConstantMemory - If the specified memory location is
  /// known to be constant, return true. If OrLocal is true and the
  /// specified memory location is known to be "local" (derived from
  /// an alloca), return true. Otherwise return false.
  virtual bool pointsToConstantMemory(const Location &Loc,
                                      bool OrLocal = false);

  /// pointsToConstantMemory - A convenient wrapper.
  bool pointsToConstantMemory(const Value *P, bool OrLocal = false) {
    return pointsToConstantMemory(Location(P), OrLocal);
  }

  //===--------------------------------------------------------------------===//
  /// Simple mod/ref information...
  ///

  /// ModRefResult - Represent the result of a mod/ref query.  Mod and Ref are
  /// bits which may be or'd together.
  ///
  enum ModRefResult { NoModRef = 0, Ref = 1, Mod = 2, ModRef = 3 };

  /// These values define additional bits used to define the
  /// ModRefBehavior values.
  enum { Nowhere = 0, ArgumentPointees = 4, Anywhere = 8 | ArgumentPointees };

  /// ModRefBehavior - Summary of how a function affects memory in the program.
  /// Loads from constant globals are not considered memory accesses for this
  /// interface.  Also, functions may freely modify stack space local to their
  /// invocation without having to report it through these interfaces.
  enum ModRefBehavior {
    /// DoesNotAccessMemory - This function does not perform any non-local loads
    /// or stores to memory.
    ///
    /// This property corresponds to the GCC 'const' attribute.
    /// This property corresponds to the LLVM IR 'readnone' attribute.
    /// This property corresponds to the IntrNoMem LLVM intrinsic flag.
    DoesNotAccessMemory = Nowhere | NoModRef,

    /// OnlyReadsArgumentPointees - The only memory references in this function
    /// (if it has any) are non-volatile loads from objects pointed to by its
    /// pointer-typed arguments, with arbitrary offsets.
    ///
    /// This property corresponds to the IntrReadArgMem LLVM intrinsic flag.
    OnlyReadsArgumentPointees = ArgumentPointees | Ref,

    /// OnlyAccessesArgumentPointees - The only memory references in this
    /// function (if it has any) are non-volatile loads and stores from objects
    /// pointed to by its pointer-typed arguments, with arbitrary offsets.
    ///
    /// This property corresponds to the IntrReadWriteArgMem LLVM intrinsic flag.
    OnlyAccessesArgumentPointees = ArgumentPointees | ModRef,

    /// OnlyReadsMemory - This function does not perform any non-local stores or
    /// volatile loads, but may read from any memory location.
    ///
    /// This property corresponds to the GCC 'pure' attribute.
    /// This property corresponds to the LLVM IR 'readonly' attribute.
    /// This property corresponds to the IntrReadMem LLVM intrinsic flag.
    OnlyReadsMemory = Anywhere | Ref,

    /// UnknownModRefBehavior - This indicates that the function could not be
    /// classified into one of the behaviors above.
    UnknownModRefBehavior = Anywhere | ModRef
  };

  /// getModRefBehavior - Return the behavior when calling the given call site.
  virtual ModRefBehavior getModRefBehavior(ImmutableCallSite CS);

  /// getModRefBehavior - Return the behavior when calling the given function.
  /// For use when the call site is not known.
  virtual ModRefBehavior getModRefBehavior(const Function *F);

  /// doesNotAccessMemory - If the specified call is known to never read or
  /// write memory, return true.  If the call only reads from known-constant
  /// memory, it is also legal to return true.  Calls that unwind the stack
  /// are legal for this predicate.
  ///
  /// Many optimizations (such as CSE and LICM) can be performed on such calls
  /// without worrying about aliasing properties, and many calls have this
  /// property (e.g. calls to 'sin' and 'cos').
  ///
  /// This property corresponds to the GCC 'const' attribute.
  ///
  bool doesNotAccessMemory(ImmutableCallSite CS) {
    return getModRefBehavior(CS) == DoesNotAccessMemory;
  }

  /// doesNotAccessMemory - If the specified function is known to never read or
  /// write memory, return true.  For use when the call site is not known.
  ///
  bool doesNotAccessMemory(const Function *F) {
    return getModRefBehavior(F) == DoesNotAccessMemory;
  }

  /// onlyReadsMemory - If the specified call is known to only read from
  /// non-volatile memory (or not access memory at all), return true.  Calls
  /// that unwind the stack are legal for this predicate.
  ///
  /// This property allows many common optimizations to be performed in the
  /// absence of interfering store instructions, such as CSE of strlen calls.
  ///
  /// This property corresponds to the GCC 'pure' attribute.
  ///
  bool onlyReadsMemory(ImmutableCallSite CS) {
    return onlyReadsMemory(getModRefBehavior(CS));
  }

  /// onlyReadsMemory - If the specified function is known to only read from
  /// non-volatile memory (or not access memory at all), return true.  For use
  /// when the call site is not known.
  ///
  bool onlyReadsMemory(const Function *F) {
    return onlyReadsMemory(getModRefBehavior(F));
  }

  /// onlyReadsMemory - Return true if functions with the specified behavior are
  /// known to only read from non-volatile memory (or not access memory at all).
  ///
  static bool onlyReadsMemory(ModRefBehavior MRB) {
    return !(MRB & Mod);
  }

  /// onlyAccessesArgPointees - Return true if functions with the specified
  /// behavior are known to read and write at most from objects pointed to by
  /// their pointer-typed arguments (with arbitrary offsets).
  ///
  static bool onlyAccessesArgPointees(ModRefBehavior MRB) {
    return !(MRB & Anywhere & ~ArgumentPointees);
  }

  /// doesAccessArgPointees - Return true if functions with the specified
  /// behavior are known to potentially read or write from objects pointed
  /// to be their pointer-typed arguments (with arbitrary offsets).
  ///
  static bool doesAccessArgPointees(ModRefBehavior MRB) {
    return (MRB & ModRef) && (MRB & ArgumentPointees);
  }

  /// getModRefInfo - Return information about whether or not an instruction may
  /// read or write the specified memory location.  An instruction
  /// that doesn't read or write memory may be trivially LICM'd for example.
  ModRefResult getModRefInfo(const Instruction *I,
                             const Location &Loc) {
    switch (I->getOpcode()) {
    case Instruction::VAArg:  return getModRefInfo((const VAArgInst*)I, Loc);
    case Instruction::Load:   return getModRefInfo((const LoadInst*)I,  Loc);
    case Instruction::Store:  return getModRefInfo((const StoreInst*)I, Loc);
    case Instruction::Fence:  return getModRefInfo((const FenceInst*)I, Loc);
    case Instruction::AtomicCmpXchg:
      return getModRefInfo((const AtomicCmpXchgInst*)I, Loc);
    case Instruction::AtomicRMW:
      return getModRefInfo((const AtomicRMWInst*)I, Loc);
    case Instruction::Call:   return getModRefInfo((const CallInst*)I,  Loc);
    case Instruction::Invoke: return getModRefInfo((const InvokeInst*)I,Loc);
    default:                  return NoModRef;
    }
  }

  /// getModRefInfo - A convenience wrapper.
  ModRefResult getModRefInfo(const Instruction *I,
                             const Value *P, uint64_t Size) {
    return getModRefInfo(I, Location(P, Size));
  }

  /// getModRefInfo (for call sites) - Return whether information about whether
  /// a particular call site modifies or reads the specified memory location.
  virtual ModRefResult getModRefInfo(ImmutableCallSite CS,
                                     const Location &Loc);

  /// getModRefInfo (for call sites) - A convenience wrapper.
  ModRefResult getModRefInfo(ImmutableCallSite CS,
                             const Value *P, uint64_t Size) {
    return getModRefInfo(CS, Location(P, Size));
  }

  /// getModRefInfo (for calls) - Return whether information about whether
  /// a particular call modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const CallInst *C, const Location &Loc) {
    return getModRefInfo(ImmutableCallSite(C), Loc);
  }

  /// getModRefInfo (for calls) - A convenience wrapper.
  ModRefResult getModRefInfo(const CallInst *C, const Value *P, uint64_t Size) {
    return getModRefInfo(C, Location(P, Size));
  }

  /// getModRefInfo (for invokes) - Return whether information about whether
  /// a particular invoke modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const InvokeInst *I,
                             const Location &Loc) {
    return getModRefInfo(ImmutableCallSite(I), Loc);
  }

  /// getModRefInfo (for invokes) - A convenience wrapper.
  ModRefResult getModRefInfo(const InvokeInst *I,
                             const Value *P, uint64_t Size) {
    return getModRefInfo(I, Location(P, Size));
  }

  /// getModRefInfo (for loads) - Return whether information about whether
  /// a particular load modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const LoadInst *L, const Location &Loc);

  /// getModRefInfo (for loads) - A convenience wrapper.
  ModRefResult getModRefInfo(const LoadInst *L, const Value *P, uint64_t Size) {
    return getModRefInfo(L, Location(P, Size));
  }

  /// getModRefInfo (for stores) - Return whether information about whether
  /// a particular store modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const StoreInst *S, const Location &Loc);

  /// getModRefInfo (for stores) - A convenience wrapper.
  ModRefResult getModRefInfo(const StoreInst *S, const Value *P, uint64_t Size){
    return getModRefInfo(S, Location(P, Size));
  }

  /// getModRefInfo (for fences) - Return whether information about whether
  /// a particular store modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const FenceInst *S, const Location &Loc) {
    // Conservatively correct.  (We could possibly be a bit smarter if
    // Loc is a alloca that doesn't escape.)
    return ModRef;
  }

  /// getModRefInfo (for fences) - A convenience wrapper.
  ModRefResult getModRefInfo(const FenceInst *S, const Value *P, uint64_t Size){
    return getModRefInfo(S, Location(P, Size));
  }

  /// getModRefInfo (for cmpxchges) - Return whether information about whether
  /// a particular cmpxchg modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const AtomicCmpXchgInst *CX, const Location &Loc);

  /// getModRefInfo (for cmpxchges) - A convenience wrapper.
  ModRefResult getModRefInfo(const AtomicCmpXchgInst *CX,
                             const Value *P, unsigned Size) {
    return getModRefInfo(CX, Location(P, Size));
  }

  /// getModRefInfo (for atomicrmws) - Return whether information about whether
  /// a particular atomicrmw modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const AtomicRMWInst *RMW, const Location &Loc);

  /// getModRefInfo (for atomicrmws) - A convenience wrapper.
  ModRefResult getModRefInfo(const AtomicRMWInst *RMW,
                             const Value *P, unsigned Size) {
    return getModRefInfo(RMW, Location(P, Size));
  }

  /// getModRefInfo (for va_args) - Return whether information about whether
  /// a particular va_arg modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const VAArgInst* I, const Location &Loc);

  /// getModRefInfo (for va_args) - A convenience wrapper.
  ModRefResult getModRefInfo(const VAArgInst* I, const Value* P, uint64_t Size){
    return getModRefInfo(I, Location(P, Size));
  }

  /// getModRefInfo - Return information about whether two call sites may refer
  /// to the same set of memory locations.  See 
  ///   http://llvm.org/docs/AliasAnalysis.html#ModRefInfo
  /// for details.
  virtual ModRefResult getModRefInfo(ImmutableCallSite CS1,
                                     ImmutableCallSite CS2);

  /// callCapturesBefore - Return information about whether a particular call 
  /// site modifies or reads the specified memory location.
  ModRefResult callCapturesBefore(const Instruction *I,
                                  const AliasAnalysis::Location &MemLoc,
                                  DominatorTree *DT);

  /// callCapturesBefore - A convenience wrapper.
  ModRefResult callCapturesBefore(const Instruction *I, const Value *P,
                                  uint64_t Size, DominatorTree *DT) {
    return callCapturesBefore(I, Location(P, Size), DT);
  }

  //===--------------------------------------------------------------------===//
  /// Higher level methods for querying mod/ref information.
  ///

  /// canBasicBlockModify - Return true if it is possible for execution of the
  /// specified basic block to modify the value pointed to by Ptr.
  bool canBasicBlockModify(const BasicBlock &BB, const Location &Loc);

  /// canBasicBlockModify - A convenience wrapper.
  bool canBasicBlockModify(const BasicBlock &BB, const Value *P, uint64_t Size){
    return canBasicBlockModify(BB, Location(P, Size));
  }

  /// canInstructionRangeModify - Return true if it is possible for the
  /// execution of the specified instructions to modify the value pointed to by
  /// Ptr.  The instructions to consider are all of the instructions in the
  /// range of [I1,I2] INCLUSIVE.  I1 and I2 must be in the same basic block.
  bool canInstructionRangeModify(const Instruction &I1, const Instruction &I2,
                                 const Location &Loc);

  /// canInstructionRangeModify - A convenience wrapper.
  bool canInstructionRangeModify(const Instruction &I1, const Instruction &I2,
                                 const Value *Ptr, uint64_t Size) {
    return canInstructionRangeModify(I1, I2, Location(Ptr, Size));
  }

  //===--------------------------------------------------------------------===//
  /// Methods that clients should call when they transform the program to allow
  /// alias analyses to update their internal data structures.  Note that these
  /// methods may be called on any instruction, regardless of whether or not
  /// they have pointer-analysis implications.
  ///

  /// deleteValue - This method should be called whenever an LLVM Value is
  /// deleted from the program, for example when an instruction is found to be
  /// redundant and is eliminated.
  ///
  virtual void deleteValue(Value *V);

  /// copyValue - This method should be used whenever a preexisting value in the
  /// program is copied or cloned, introducing a new value.  Note that analysis
  /// implementations should tolerate clients that use this method to introduce
  /// the same value multiple times: if the analysis already knows about a
  /// value, it should ignore the request.
  ///
  virtual void copyValue(Value *From, Value *To);

  /// addEscapingUse - This method should be used whenever an escaping use is
  /// added to a pointer value.  Analysis implementations may either return
  /// conservative responses for that value in the future, or may recompute
  /// some or all internal state to continue providing precise responses.
  ///
  /// Escaping uses are considered by anything _except_ the following:
  ///  - GEPs or bitcasts of the pointer
  ///  - Loads through the pointer
  ///  - Stores through (but not of) the pointer
  virtual void addEscapingUse(Use &U);

  /// replaceWithNewValue - This method is the obvious combination of the two
  /// above, and it provided as a helper to simplify client code.
  ///
  void replaceWithNewValue(Value *Old, Value *New) {
    copyValue(Old, New);
    deleteValue(Old);
  }
};

// Specialize DenseMapInfo for Location.
template<>
struct DenseMapInfo<AliasAnalysis::Location> {
  static inline AliasAnalysis::Location getEmptyKey() {
    return
      AliasAnalysis::Location(DenseMapInfo<const Value *>::getEmptyKey(),
                              0, 0);
  }
  static inline AliasAnalysis::Location getTombstoneKey() {
    return
      AliasAnalysis::Location(DenseMapInfo<const Value *>::getTombstoneKey(),
                              0, 0);
  }
  static unsigned getHashValue(const AliasAnalysis::Location &Val) {
    return DenseMapInfo<const Value *>::getHashValue(Val.Ptr) ^
           DenseMapInfo<uint64_t>::getHashValue(Val.Size) ^
           DenseMapInfo<const MDNode *>::getHashValue(Val.TBAATag);
  }
  static bool isEqual(const AliasAnalysis::Location &LHS,
                      const AliasAnalysis::Location &RHS) {
    return LHS.Ptr == RHS.Ptr &&
           LHS.Size == RHS.Size &&
           LHS.TBAATag == RHS.TBAATag;
  }
};

/// isNoAliasCall - Return true if this pointer is returned by a noalias
/// function.
bool isNoAliasCall(const Value *V);

/// isIdentifiedObject - Return true if this pointer refers to a distinct and
/// identifiable object.  This returns true for:
///    Global Variables and Functions (but not Global Aliases)
///    Allocas and Mallocs
///    ByVal and NoAlias Arguments
///    NoAlias returns
///
bool isIdentifiedObject(const Value *V);

/// isKnownNonNull - Return true if this pointer couldn't possibly be null by
/// its definition.  This returns true for allocas, non-extern-weak globals and
/// byval arguments.
bool isKnownNonNull(const Value *V);

} // End llvm namespace

#endif
