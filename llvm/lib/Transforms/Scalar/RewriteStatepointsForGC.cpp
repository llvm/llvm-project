//===- RewriteStatepointsForGC.cpp - Make GC relocations explicit ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Rewrite an existing set of gc.statepoints such that they make potential
// relocations performed by the garbage collector explicit in the IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#define DEBUG_TYPE "rewrite-statepoints-for-gc"

using namespace llvm;

// Print the liveset found at the insert location
static cl::opt<bool> PrintLiveSet("spp-print-liveset", cl::Hidden,
                                  cl::init(false));
static cl::opt<bool> PrintLiveSetSize("spp-print-liveset-size", cl::Hidden,
                                      cl::init(false));
// Print out the base pointers for debugging
static cl::opt<bool> PrintBasePointers("spp-print-base-pointers", cl::Hidden,
                                       cl::init(false));

// Cost threshold measuring when it is profitable to rematerialize value instead
// of relocating it
static cl::opt<unsigned>
RematerializationThreshold("spp-rematerialization-threshold", cl::Hidden,
                           cl::init(6));

#ifdef XDEBUG
static bool ClobberNonLive = true;
#else
static bool ClobberNonLive = false;
#endif
static cl::opt<bool, true> ClobberNonLiveOverride("rs4gc-clobber-non-live",
                                                  cl::location(ClobberNonLive),
                                                  cl::Hidden);

static cl::opt<bool> UseDeoptBundles("rs4gc-use-deopt-bundles", cl::Hidden,
                                     cl::init(false));
static cl::opt<bool>
    AllowStatepointWithNoDeoptInfo("rs4gc-allow-statepoint-with-no-deopt-info",
                                   cl::Hidden, cl::init(true));

/// Should we split vectors of pointers into their individual elements?  This
/// is known to be buggy, but the alternate implementation isn't yet ready.
/// This is purely to provide a debugging and dianostic hook until the vector
/// split is replaced with vector relocations.
static cl::opt<bool> UseVectorSplit("rs4gc-split-vector-values", cl::Hidden,
                                    cl::init(true));

namespace {
struct RewriteStatepointsForGC : public ModulePass {
  static char ID; // Pass identification, replacement for typeid

  RewriteStatepointsForGC() : ModulePass(ID) {
    initializeRewriteStatepointsForGCPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F);
  bool runOnModule(Module &M) override {
    bool Changed = false;
    for (Function &F : M)
      Changed |= runOnFunction(F);

    if (Changed) {
      // stripNonValidAttributes asserts that shouldRewriteStatepointsIn
      // returns true for at least one function in the module.  Since at least
      // one function changed, we know that the precondition is satisfied.
      stripNonValidAttributes(M);
    }

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We add and rewrite a bunch of instructions, but don't really do much
    // else.  We could in theory preserve a lot more analyses here.
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  /// The IR fed into RewriteStatepointsForGC may have had attributes implying
  /// dereferenceability that are no longer valid/correct after
  /// RewriteStatepointsForGC has run.  This is because semantically, after
  /// RewriteStatepointsForGC runs, all calls to gc.statepoint "free" the entire
  /// heap.  stripNonValidAttributes (conservatively) restores correctness
  /// by erasing all attributes in the module that externally imply
  /// dereferenceability.
  /// Similar reasoning also applies to the noalias attributes. gc.statepoint
  /// can touch the entire heap including noalias objects.
  void stripNonValidAttributes(Module &M);

  // Helpers for stripNonValidAttributes
  void stripNonValidAttributesFromBody(Function &F);
  void stripNonValidAttributesFromPrototype(Function &F);
};
} // namespace

char RewriteStatepointsForGC::ID = 0;

ModulePass *llvm::createRewriteStatepointsForGCPass() {
  return new RewriteStatepointsForGC();
}

INITIALIZE_PASS_BEGIN(RewriteStatepointsForGC, "rewrite-statepoints-for-gc",
                      "Make relocations explicit at statepoints", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(RewriteStatepointsForGC, "rewrite-statepoints-for-gc",
                    "Make relocations explicit at statepoints", false, false)

namespace {
struct GCPtrLivenessData {
  /// Values defined in this block.
  DenseMap<BasicBlock *, DenseSet<Value *>> KillSet;
  /// Values used in this block (and thus live); does not included values
  /// killed within this block.
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveSet;

  /// Values live into this basic block (i.e. used by any
  /// instruction in this basic block or ones reachable from here)
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveIn;

  /// Values live out of this basic block (i.e. live into
  /// any successor block)
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveOut;
};

// The type of the internal cache used inside the findBasePointers family
// of functions.  From the callers perspective, this is an opaque type and
// should not be inspected.
//
// In the actual implementation this caches two relations:
// - The base relation itself (i.e. this pointer is based on that one)
// - The base defining value relation (i.e. before base_phi insertion)
// Generally, after the execution of a full findBasePointer call, only the
// base relation will remain.  Internally, we add a mixture of the two
// types, then update all the second type to the first type
typedef DenseMap<Value *, Value *> DefiningValueMapTy;
typedef DenseSet<Value *> StatepointLiveSetTy;
typedef DenseMap<AssertingVH<Instruction>, AssertingVH<Value>>
  RematerializedValueMapTy;

struct PartiallyConstructedSafepointRecord {
  /// The set of values known to be live across this safepoint
  StatepointLiveSetTy LiveSet;

  /// Mapping from live pointers to a base-defining-value
  DenseMap<Value *, Value *> PointerToBase;

  /// The *new* gc.statepoint instruction itself.  This produces the token
  /// that normal path gc.relocates and the gc.result are tied to.
  Instruction *StatepointToken;

  /// Instruction to which exceptional gc relocates are attached
  /// Makes it easier to iterate through them during relocationViaAlloca.
  Instruction *UnwindToken;

  /// Record live values we are rematerialized instead of relocating.
  /// They are not included into 'LiveSet' field.
  /// Maps rematerialized copy to it's original value.
  RematerializedValueMapTy RematerializedValues;
};
}

static ArrayRef<Use> GetDeoptBundleOperands(ImmutableCallSite CS) {
  assert(UseDeoptBundles && "Should not be called otherwise!");

  Optional<OperandBundleUse> DeoptBundle = CS.getOperandBundle("deopt");

  if (!DeoptBundle.hasValue()) {
    assert(AllowStatepointWithNoDeoptInfo &&
           "Found non-leaf call without deopt info!");
    return None;
  }

  return DeoptBundle.getValue().Inputs;
}

/// Compute the live-in set for every basic block in the function
static void computeLiveInValues(DominatorTree &DT, Function &F,
                                GCPtrLivenessData &Data);

/// Given results from the dataflow liveness computation, find the set of live
/// Values at a particular instruction.
static void findLiveSetAtInst(Instruction *inst, GCPtrLivenessData &Data,
                              StatepointLiveSetTy &out);

// TODO: Once we can get to the GCStrategy, this becomes
// Optional<bool> isGCManagedPointer(const Type *Ty) const override {

static bool isGCPointerType(Type *T) {
  if (auto *PT = dyn_cast<PointerType>(T))
    // For the sake of this example GC, we arbitrarily pick addrspace(1) as our
    // GC managed heap.  We know that a pointer into this heap needs to be
    // updated and that no other pointer does.
    return (1 == PT->getAddressSpace());
  return false;
}

// Return true if this type is one which a) is a gc pointer or contains a GC
// pointer and b) is of a type this code expects to encounter as a live value.
// (The insertion code will assert that a type which matches (a) and not (b)
// is not encountered.)
static bool isHandledGCPointerType(Type *T) {
  // We fully support gc pointers
  if (isGCPointerType(T))
    return true;
  // We partially support vectors of gc pointers. The code will assert if it
  // can't handle something.
  if (auto VT = dyn_cast<VectorType>(T))
    if (isGCPointerType(VT->getElementType()))
      return true;
  return false;
}

#ifndef NDEBUG
/// Returns true if this type contains a gc pointer whether we know how to
/// handle that type or not.
static bool containsGCPtrType(Type *Ty) {
  if (isGCPointerType(Ty))
    return true;
  if (VectorType *VT = dyn_cast<VectorType>(Ty))
    return isGCPointerType(VT->getScalarType());
  if (ArrayType *AT = dyn_cast<ArrayType>(Ty))
    return containsGCPtrType(AT->getElementType());
  if (StructType *ST = dyn_cast<StructType>(Ty))
    return std::any_of(ST->subtypes().begin(), ST->subtypes().end(),
                       containsGCPtrType);
  return false;
}

// Returns true if this is a type which a) is a gc pointer or contains a GC
// pointer and b) is of a type which the code doesn't expect (i.e. first class
// aggregates).  Used to trip assertions.
static bool isUnhandledGCPointerType(Type *Ty) {
  return containsGCPtrType(Ty) && !isHandledGCPointerType(Ty);
}
#endif

static bool order_by_name(Value *a, Value *b) {
  if (a->hasName() && b->hasName()) {
    return -1 == a->getName().compare(b->getName());
  } else if (a->hasName() && !b->hasName()) {
    return true;
  } else if (!a->hasName() && b->hasName()) {
    return false;
  } else {
    // Better than nothing, but not stable
    return a < b;
  }
}

// Return the name of the value suffixed with the provided value, or if the
// value didn't have a name, the default value specified.
static std::string suffixed_name_or(Value *V, StringRef Suffix,
                                    StringRef DefaultName) {
  return V->hasName() ? (V->getName() + Suffix).str() : DefaultName.str();
}

// Conservatively identifies any definitions which might be live at the
// given instruction. The  analysis is performed immediately before the
// given instruction. Values defined by that instruction are not considered
// live.  Values used by that instruction are considered live.
static void analyzeParsePointLiveness(
    DominatorTree &DT, GCPtrLivenessData &OriginalLivenessData,
    const CallSite &CS, PartiallyConstructedSafepointRecord &result) {
  Instruction *inst = CS.getInstruction();

  StatepointLiveSetTy LiveSet;
  findLiveSetAtInst(inst, OriginalLivenessData, LiveSet);

  if (PrintLiveSet) {
    // Note: This output is used by several of the test cases
    // The order of elements in a set is not stable, put them in a vec and sort
    // by name
    SmallVector<Value *, 64> Temp;
    Temp.insert(Temp.end(), LiveSet.begin(), LiveSet.end());
    std::sort(Temp.begin(), Temp.end(), order_by_name);
    errs() << "Live Variables:\n";
    for (Value *V : Temp)
      dbgs() << " " << V->getName() << " " << *V << "\n";
  }
  if (PrintLiveSetSize) {
    errs() << "Safepoint For: " << CS.getCalledValue()->getName() << "\n";
    errs() << "Number live values: " << LiveSet.size() << "\n";
  }
  result.LiveSet = LiveSet;
}

static bool isKnownBaseResult(Value *V);
namespace {
/// A single base defining value - An immediate base defining value for an
/// instruction 'Def' is an input to 'Def' whose base is also a base of 'Def'.
/// For instructions which have multiple pointer [vector] inputs or that
/// transition between vector and scalar types, there is no immediate base
/// defining value.  The 'base defining value' for 'Def' is the transitive
/// closure of this relation stopping at the first instruction which has no
/// immediate base defining value.  The b.d.v. might itself be a base pointer,
/// but it can also be an arbitrary derived pointer. 
struct BaseDefiningValueResult {
  /// Contains the value which is the base defining value.
  Value * const BDV;
  /// True if the base defining value is also known to be an actual base
  /// pointer.
  const bool IsKnownBase;
  BaseDefiningValueResult(Value *BDV, bool IsKnownBase)
    : BDV(BDV), IsKnownBase(IsKnownBase) {
#ifndef NDEBUG
    // Check consistency between new and old means of checking whether a BDV is
    // a base.
    bool MustBeBase = isKnownBaseResult(BDV);
    assert(!MustBeBase || MustBeBase == IsKnownBase);
#endif
  }
};
}

static BaseDefiningValueResult findBaseDefiningValue(Value *I);

/// Return a base defining value for the 'Index' element of the given vector
/// instruction 'I'.  If Index is null, returns a BDV for the entire vector
/// 'I'.  As an optimization, this method will try to determine when the 
/// element is known to already be a base pointer.  If this can be established,
/// the second value in the returned pair will be true.  Note that either a
/// vector or a pointer typed value can be returned.  For the former, the
/// vector returned is a BDV (and possibly a base) of the entire vector 'I'.
/// If the later, the return pointer is a BDV (or possibly a base) for the
/// particular element in 'I'.  
static BaseDefiningValueResult
findBaseDefiningValueOfVector(Value *I) {
  assert(I->getType()->isVectorTy() &&
         cast<VectorType>(I->getType())->getElementType()->isPointerTy() &&
         "Illegal to ask for the base pointer of a non-pointer type");

  // Each case parallels findBaseDefiningValue below, see that code for
  // detailed motivation.

  if (isa<Argument>(I))
    // An incoming argument to the function is a base pointer
    return BaseDefiningValueResult(I, true);

  // We shouldn't see the address of a global as a vector value?
  assert(!isa<GlobalVariable>(I) &&
         "unexpected global variable found in base of vector");

  // inlining could possibly introduce phi node that contains
  // undef if callee has multiple returns
  if (isa<UndefValue>(I))
    // utterly meaningless, but useful for dealing with partially optimized
    // code.
    return BaseDefiningValueResult(I, true);

  // Due to inheritance, this must be _after_ the global variable and undef
  // checks
  if (Constant *Con = dyn_cast<Constant>(I)) {
    assert(!isa<GlobalVariable>(I) && !isa<UndefValue>(I) &&
           "order of checks wrong!");
    assert(Con->isNullValue() && "null is the only case which makes sense");
    return BaseDefiningValueResult(Con, true);
  }
  
  if (isa<LoadInst>(I))
    return BaseDefiningValueResult(I, true);

  if (isa<InsertElementInst>(I))
    // We don't know whether this vector contains entirely base pointers or
    // not.  To be conservatively correct, we treat it as a BDV and will
    // duplicate code as needed to construct a parallel vector of bases.
    return BaseDefiningValueResult(I, false);

  if (isa<ShuffleVectorInst>(I))
    // We don't know whether this vector contains entirely base pointers or
    // not.  To be conservatively correct, we treat it as a BDV and will
    // duplicate code as needed to construct a parallel vector of bases.
    // TODO: There a number of local optimizations which could be applied here
    // for particular sufflevector patterns.
    return BaseDefiningValueResult(I, false);

  // A PHI or Select is a base defining value.  The outer findBasePointer
  // algorithm is responsible for constructing a base value for this BDV.
  assert((isa<SelectInst>(I) || isa<PHINode>(I)) &&
         "unknown vector instruction - no base found for vector element");
  return BaseDefiningValueResult(I, false);
}

/// Helper function for findBasePointer - Will return a value which either a)
/// defines the base pointer for the input, b) blocks the simple search
/// (i.e. a PHI or Select of two derived pointers), or c) involves a change
/// from pointer to vector type or back.
static BaseDefiningValueResult findBaseDefiningValue(Value *I) {
  if (I->getType()->isVectorTy())
    return findBaseDefiningValueOfVector(I);
  
  assert(I->getType()->isPointerTy() &&
         "Illegal to ask for the base pointer of a non-pointer type");

  if (isa<Argument>(I))
    // An incoming argument to the function is a base pointer
    // We should have never reached here if this argument isn't an gc value
    return BaseDefiningValueResult(I, true);

  if (isa<Constant>(I))
    // We assume that objects with a constant base (e.g. a global) can't move
    // and don't need to be reported to the collector because they are always
    // live.  All constants have constant bases.  Besides global references, all
    // kinds of constants (e.g. undef, constant expressions, null pointers) can
    // be introduced by the inliner or the optimizer, especially on dynamically
    // dead paths.  See e.g. test4 in constants.ll.
    return BaseDefiningValueResult(I, true);

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    Value *Def = CI->stripPointerCasts();
    // If stripping pointer casts changes the address space there is an
    // addrspacecast in between.
    assert(cast<PointerType>(Def->getType())->getAddressSpace() ==
               cast<PointerType>(CI->getType())->getAddressSpace() &&
           "unsupported addrspacecast");
    // If we find a cast instruction here, it means we've found a cast which is
    // not simply a pointer cast (i.e. an inttoptr).  We don't know how to
    // handle int->ptr conversion.
    assert(!isa<CastInst>(Def) && "shouldn't find another cast here");
    return findBaseDefiningValue(Def);
  }

  if (isa<LoadInst>(I))
    // The value loaded is an gc base itself
    return BaseDefiningValueResult(I, true);
  

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I))
    // The base of this GEP is the base
    return findBaseDefiningValue(GEP->getPointerOperand());

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default:
      // fall through to general call handling
      break;
    case Intrinsic::experimental_gc_statepoint:
      llvm_unreachable("statepoints don't produce pointers");
    case Intrinsic::experimental_gc_relocate: {
      // Rerunning safepoint insertion after safepoints are already
      // inserted is not supported.  It could probably be made to work,
      // but why are you doing this?  There's no good reason.
      llvm_unreachable("repeat safepoint insertion is not supported");
    }
    case Intrinsic::gcroot:
      // Currently, this mechanism hasn't been extended to work with gcroot.
      // There's no reason it couldn't be, but I haven't thought about the
      // implications much.
      llvm_unreachable(
          "interaction with the gcroot mechanism is not supported");
    }
  }
  // We assume that functions in the source language only return base
  // pointers.  This should probably be generalized via attributes to support
  // both source language and internal functions.
  if (isa<CallInst>(I) || isa<InvokeInst>(I))
    return BaseDefiningValueResult(I, true);

  // I have absolutely no idea how to implement this part yet.  It's not
  // necessarily hard, I just haven't really looked at it yet.
  assert(!isa<LandingPadInst>(I) && "Landing Pad is unimplemented");

  if (isa<AtomicCmpXchgInst>(I))
    // A CAS is effectively a atomic store and load combined under a
    // predicate.  From the perspective of base pointers, we just treat it
    // like a load.
    return BaseDefiningValueResult(I, true);

  assert(!isa<AtomicRMWInst>(I) && "Xchg handled above, all others are "
                                   "binary ops which don't apply to pointers");

  // The aggregate ops.  Aggregates can either be in the heap or on the
  // stack, but in either case, this is simply a field load.  As a result,
  // this is a defining definition of the base just like a load is.
  if (isa<ExtractValueInst>(I))
    return BaseDefiningValueResult(I, true);

  // We should never see an insert vector since that would require we be
  // tracing back a struct value not a pointer value.
  assert(!isa<InsertValueInst>(I) &&
         "Base pointer for a struct is meaningless");

  // An extractelement produces a base result exactly when it's input does.
  // We may need to insert a parallel instruction to extract the appropriate
  // element out of the base vector corresponding to the input. Given this,
  // it's analogous to the phi and select case even though it's not a merge.
  if (isa<ExtractElementInst>(I))
    // Note: There a lot of obvious peephole cases here.  This are deliberately
    // handled after the main base pointer inference algorithm to make writing
    // test cases to exercise that code easier.
    return BaseDefiningValueResult(I, false);

  // The last two cases here don't return a base pointer.  Instead, they
  // return a value which dynamically selects from among several base
  // derived pointers (each with it's own base potentially).  It's the job of
  // the caller to resolve these.
  assert((isa<SelectInst>(I) || isa<PHINode>(I)) &&
         "missing instruction case in findBaseDefiningValing");
  return BaseDefiningValueResult(I, false);
}

/// Returns the base defining value for this value.
static Value *findBaseDefiningValueCached(Value *I, DefiningValueMapTy &Cache) {
  Value *&Cached = Cache[I];
  if (!Cached) {
    Cached = findBaseDefiningValue(I).BDV;
    DEBUG(dbgs() << "fBDV-cached: " << I->getName() << " -> "
                 << Cached->getName() << "\n");
  }
  assert(Cache[I] != nullptr);
  return Cached;
}

/// Return a base pointer for this value if known.  Otherwise, return it's
/// base defining value.
static Value *findBaseOrBDV(Value *I, DefiningValueMapTy &Cache) {
  Value *Def = findBaseDefiningValueCached(I, Cache);
  auto Found = Cache.find(Def);
  if (Found != Cache.end()) {
    // Either a base-of relation, or a self reference.  Caller must check.
    return Found->second;
  }
  // Only a BDV available
  return Def;
}

/// Given the result of a call to findBaseDefiningValue, or findBaseOrBDV,
/// is it known to be a base pointer?  Or do we need to continue searching.
static bool isKnownBaseResult(Value *V) {
  if (!isa<PHINode>(V) && !isa<SelectInst>(V) &&
      !isa<ExtractElementInst>(V) && !isa<InsertElementInst>(V) &&
      !isa<ShuffleVectorInst>(V)) {
    // no recursion possible
    return true;
  }
  if (isa<Instruction>(V) &&
      cast<Instruction>(V)->getMetadata("is_base_value")) {
    // This is a previously inserted base phi or select.  We know
    // that this is a base value.
    return true;
  }

  // We need to keep searching
  return false;
}

namespace {
/// Models the state of a single base defining value in the findBasePointer
/// algorithm for determining where a new instruction is needed to propagate
/// the base of this BDV.
class BDVState {
public:
  enum Status { Unknown, Base, Conflict };

  BDVState(Status s, Value *b = nullptr) : status(s), base(b) {
    assert(status != Base || b);
  }
  explicit BDVState(Value *b) : status(Base), base(b) {}
  BDVState() : status(Unknown), base(nullptr) {}

  Status getStatus() const { return status; }
  Value *getBase() const { return base; }

  bool isBase() const { return getStatus() == Base; }
  bool isUnknown() const { return getStatus() == Unknown; }
  bool isConflict() const { return getStatus() == Conflict; }

  bool operator==(const BDVState &other) const {
    return base == other.base && status == other.status;
  }

  bool operator!=(const BDVState &other) const { return !(*this == other); }

  LLVM_DUMP_METHOD
  void dump() const { print(dbgs()); dbgs() << '\n'; }
  
  void print(raw_ostream &OS) const {
    switch (status) {
    case Unknown:
      OS << "U";
      break;
    case Base:
      OS << "B";
      break;
    case Conflict:
      OS << "C";
      break;
    };
    OS << " (" << base << " - "
       << (base ? base->getName() : "nullptr") << "): ";
  }

private:
  Status status;
  AssertingVH<Value> base; // non null only if status == base
};
}

#ifndef NDEBUG
static raw_ostream &operator<<(raw_ostream &OS, const BDVState &State) {
  State.print(OS);
  return OS;
}
#endif

namespace {
// Values of type BDVState form a lattice, and this is a helper
// class that implementes the meet operation.  The meat of the meet
// operation is implemented in MeetBDVStates::pureMeet
class MeetBDVStates {
public:
  /// Initializes the currentResult to the TOP state so that if can be met with
  /// any other state to produce that state.
  MeetBDVStates() {}

  // Destructively meet the current result with the given BDVState
  void meetWith(BDVState otherState) {
    currentResult = meet(otherState, currentResult);
  }

  BDVState getResult() const { return currentResult; }

private:
  BDVState currentResult;

  /// Perform a meet operation on two elements of the BDVState lattice.
  static BDVState meet(BDVState LHS, BDVState RHS) {
    assert((pureMeet(LHS, RHS) == pureMeet(RHS, LHS)) &&
           "math is wrong: meet does not commute!");
    BDVState Result = pureMeet(LHS, RHS);
    DEBUG(dbgs() << "meet of " << LHS << " with " << RHS
                 << " produced " << Result << "\n");
    return Result;
  }

  static BDVState pureMeet(const BDVState &stateA, const BDVState &stateB) {
    switch (stateA.getStatus()) {
    case BDVState::Unknown:
      return stateB;

    case BDVState::Base:
      assert(stateA.getBase() && "can't be null");
      if (stateB.isUnknown())
        return stateA;

      if (stateB.isBase()) {
        if (stateA.getBase() == stateB.getBase()) {
          assert(stateA == stateB && "equality broken!");
          return stateA;
        }
        return BDVState(BDVState::Conflict);
      }
      assert(stateB.isConflict() && "only three states!");
      return BDVState(BDVState::Conflict);

    case BDVState::Conflict:
      return stateA;
    }
    llvm_unreachable("only three states!");
  }
};
}


/// For a given value or instruction, figure out what base ptr it's derived
/// from.  For gc objects, this is simply itself.  On success, returns a value
/// which is the base pointer.  (This is reliable and can be used for
/// relocation.)  On failure, returns nullptr.
static Value *findBasePointer(Value *I, DefiningValueMapTy &cache) {
  Value *def = findBaseOrBDV(I, cache);

  if (isKnownBaseResult(def)) {
    return def;
  }

  // Here's the rough algorithm:
  // - For every SSA value, construct a mapping to either an actual base
  //   pointer or a PHI which obscures the base pointer.
  // - Construct a mapping from PHI to unknown TOP state.  Use an
  //   optimistic algorithm to propagate base pointer information.  Lattice
  //   looks like:
  //   UNKNOWN
  //   b1 b2 b3 b4
  //   CONFLICT
  //   When algorithm terminates, all PHIs will either have a single concrete
  //   base or be in a conflict state.
  // - For every conflict, insert a dummy PHI node without arguments.  Add
  //   these to the base[Instruction] = BasePtr mapping.  For every
  //   non-conflict, add the actual base.
  //  - For every conflict, add arguments for the base[a] of each input
  //   arguments.
  //
  // Note: A simpler form of this would be to add the conflict form of all
  // PHIs without running the optimistic algorithm.  This would be
  // analogous to pessimistic data flow and would likely lead to an
  // overall worse solution.

#ifndef NDEBUG
  auto isExpectedBDVType = [](Value *BDV) {
    return isa<PHINode>(BDV) || isa<SelectInst>(BDV) ||
           isa<ExtractElementInst>(BDV) || isa<InsertElementInst>(BDV);
  };
#endif

  // Once populated, will contain a mapping from each potentially non-base BDV
  // to a lattice value (described above) which corresponds to that BDV.
  // We use the order of insertion (DFS over the def/use graph) to provide a
  // stable deterministic ordering for visiting DenseMaps (which are unordered)
  // below.  This is important for deterministic compilation.
  MapVector<Value *, BDVState> States;

  // Recursively fill in all base defining values reachable from the initial
  // one for which we don't already know a definite base value for
  /* scope */ {
    SmallVector<Value*, 16> Worklist;
    Worklist.push_back(def);
    States.insert(std::make_pair(def, BDVState()));
    while (!Worklist.empty()) {
      Value *Current = Worklist.pop_back_val();
      assert(!isKnownBaseResult(Current) && "why did it get added?");

      auto visitIncomingValue = [&](Value *InVal) {
        Value *Base = findBaseOrBDV(InVal, cache);
        if (isKnownBaseResult(Base))
          // Known bases won't need new instructions introduced and can be
          // ignored safely
          return;
        assert(isExpectedBDVType(Base) && "the only non-base values "
               "we see should be base defining values");
        if (States.insert(std::make_pair(Base, BDVState())).second)
          Worklist.push_back(Base);
      };
      if (PHINode *Phi = dyn_cast<PHINode>(Current)) {
        for (Value *InVal : Phi->incoming_values())
          visitIncomingValue(InVal);
      } else if (SelectInst *Sel = dyn_cast<SelectInst>(Current)) {
        visitIncomingValue(Sel->getTrueValue());
        visitIncomingValue(Sel->getFalseValue());
      } else if (auto *EE = dyn_cast<ExtractElementInst>(Current)) {
        visitIncomingValue(EE->getVectorOperand());
      } else if (auto *IE = dyn_cast<InsertElementInst>(Current)) {
        visitIncomingValue(IE->getOperand(0)); // vector operand
        visitIncomingValue(IE->getOperand(1)); // scalar operand
      } else {
        // There is one known class of instructions we know we don't handle.
        assert(isa<ShuffleVectorInst>(Current));
        llvm_unreachable("unimplemented instruction case");
      }
    }
  }

#ifndef NDEBUG
  DEBUG(dbgs() << "States after initialization:\n");
  for (auto Pair : States) {
    DEBUG(dbgs() << " " << Pair.second << " for " << *Pair.first << "\n");
  }
#endif

  // Return a phi state for a base defining value.  We'll generate a new
  // base state for known bases and expect to find a cached state otherwise.
  auto getStateForBDV = [&](Value *baseValue) {
    if (isKnownBaseResult(baseValue))
      return BDVState(baseValue);
    auto I = States.find(baseValue);
    assert(I != States.end() && "lookup failed!");
    return I->second;
  };

  bool progress = true;
  while (progress) {
#ifndef NDEBUG
    const size_t oldSize = States.size();
#endif
    progress = false;
    // We're only changing values in this loop, thus safe to keep iterators.
    // Since this is computing a fixed point, the order of visit does not
    // effect the result.  TODO: We could use a worklist here and make this run
    // much faster.
    for (auto Pair : States) {
      Value *BDV = Pair.first;
      assert(!isKnownBaseResult(BDV) && "why did it get added?");

      // Given an input value for the current instruction, return a BDVState
      // instance which represents the BDV of that value.
      auto getStateForInput = [&](Value *V) mutable {
        Value *BDV = findBaseOrBDV(V, cache);
        return getStateForBDV(BDV);
      };

      MeetBDVStates calculateMeet;
      if (SelectInst *select = dyn_cast<SelectInst>(BDV)) {
        calculateMeet.meetWith(getStateForInput(select->getTrueValue()));
        calculateMeet.meetWith(getStateForInput(select->getFalseValue()));
      } else if (PHINode *Phi = dyn_cast<PHINode>(BDV)) {
        for (Value *Val : Phi->incoming_values())
          calculateMeet.meetWith(getStateForInput(Val));
      } else if (auto *EE = dyn_cast<ExtractElementInst>(BDV)) {
        // The 'meet' for an extractelement is slightly trivial, but it's still
        // useful in that it drives us to conflict if our input is.
        calculateMeet.meetWith(getStateForInput(EE->getVectorOperand()));
      } else {
        // Given there's a inherent type mismatch between the operands, will
        // *always* produce Conflict.
        auto *IE = cast<InsertElementInst>(BDV);
        calculateMeet.meetWith(getStateForInput(IE->getOperand(0)));
        calculateMeet.meetWith(getStateForInput(IE->getOperand(1)));
      }

      BDVState oldState = States[BDV];
      BDVState newState = calculateMeet.getResult();
      if (oldState != newState) {
        progress = true;
        States[BDV] = newState;
      }
    }

    assert(oldSize == States.size() &&
           "fixed point shouldn't be adding any new nodes to state");
  }

#ifndef NDEBUG
  DEBUG(dbgs() << "States after meet iteration:\n");
  for (auto Pair : States) {
    DEBUG(dbgs() << " " << Pair.second << " for " << *Pair.first << "\n");
  }
#endif
  
  // Insert Phis for all conflicts
  // TODO: adjust naming patterns to avoid this order of iteration dependency
  for (auto Pair : States) {
    Instruction *I = cast<Instruction>(Pair.first);
    BDVState State = Pair.second;
    assert(!isKnownBaseResult(I) && "why did it get added?");
    assert(!State.isUnknown() && "Optimistic algorithm didn't complete!");

    // extractelement instructions are a bit special in that we may need to
    // insert an extract even when we know an exact base for the instruction.
    // The problem is that we need to convert from a vector base to a scalar
    // base for the particular indice we're interested in.
    if (State.isBase() && isa<ExtractElementInst>(I) &&
        isa<VectorType>(State.getBase()->getType())) {
      auto *EE = cast<ExtractElementInst>(I);
      // TODO: In many cases, the new instruction is just EE itself.  We should
      // exploit this, but can't do it here since it would break the invariant
      // about the BDV not being known to be a base.
      auto *BaseInst = ExtractElementInst::Create(State.getBase(),
                                                  EE->getIndexOperand(),
                                                  "base_ee", EE);
      BaseInst->setMetadata("is_base_value", MDNode::get(I->getContext(), {}));
      States[I] = BDVState(BDVState::Base, BaseInst);
    }

    // Since we're joining a vector and scalar base, they can never be the
    // same.  As a result, we should always see insert element having reached
    // the conflict state.
    if (isa<InsertElementInst>(I)) {
      assert(State.isConflict());
    }
    
    if (!State.isConflict())
      continue;

    /// Create and insert a new instruction which will represent the base of
    /// the given instruction 'I'.
    auto MakeBaseInstPlaceholder = [](Instruction *I) -> Instruction* {
      if (isa<PHINode>(I)) {
        BasicBlock *BB = I->getParent();
        int NumPreds = std::distance(pred_begin(BB), pred_end(BB));
        assert(NumPreds > 0 && "how did we reach here");
        std::string Name = suffixed_name_or(I, ".base", "base_phi");
        return PHINode::Create(I->getType(), NumPreds, Name, I);
      } else if (SelectInst *Sel = dyn_cast<SelectInst>(I)) {
        // The undef will be replaced later
        UndefValue *Undef = UndefValue::get(Sel->getType());
        std::string Name = suffixed_name_or(I, ".base", "base_select");
        return SelectInst::Create(Sel->getCondition(), Undef,
                                  Undef, Name, Sel);
      } else if (auto *EE = dyn_cast<ExtractElementInst>(I)) {
        UndefValue *Undef = UndefValue::get(EE->getVectorOperand()->getType());
        std::string Name = suffixed_name_or(I, ".base", "base_ee");
        return ExtractElementInst::Create(Undef, EE->getIndexOperand(), Name,
                                          EE);
      } else {
        auto *IE = cast<InsertElementInst>(I);
        UndefValue *VecUndef = UndefValue::get(IE->getOperand(0)->getType());
        UndefValue *ScalarUndef = UndefValue::get(IE->getOperand(1)->getType());
        std::string Name = suffixed_name_or(I, ".base", "base_ie");
        return InsertElementInst::Create(VecUndef, ScalarUndef,
                                         IE->getOperand(2), Name, IE);
      }

    };
    Instruction *BaseInst = MakeBaseInstPlaceholder(I);
    // Add metadata marking this as a base value
    BaseInst->setMetadata("is_base_value", MDNode::get(I->getContext(), {}));
    States[I] = BDVState(BDVState::Conflict, BaseInst);
  }

  // Returns a instruction which produces the base pointer for a given
  // instruction.  The instruction is assumed to be an input to one of the BDVs
  // seen in the inference algorithm above.  As such, we must either already
  // know it's base defining value is a base, or have inserted a new
  // instruction to propagate the base of it's BDV and have entered that newly
  // introduced instruction into the state table.  In either case, we are
  // assured to be able to determine an instruction which produces it's base
  // pointer. 
  auto getBaseForInput = [&](Value *Input, Instruction *InsertPt) {
    Value *BDV = findBaseOrBDV(Input, cache);
    Value *Base = nullptr;
    if (isKnownBaseResult(BDV)) {
      Base = BDV;
    } else {
      // Either conflict or base.
      assert(States.count(BDV));
      Base = States[BDV].getBase();
    }
    assert(Base && "can't be null");
    // The cast is needed since base traversal may strip away bitcasts
    if (Base->getType() != Input->getType() &&
        InsertPt) {
      Base = new BitCastInst(Base, Input->getType(), "cast",
                             InsertPt);
    }
    return Base;
  };

  // Fixup all the inputs of the new PHIs.  Visit order needs to be
  // deterministic and predictable because we're naming newly created
  // instructions.
  for (auto Pair : States) {
    Instruction *BDV = cast<Instruction>(Pair.first);
    BDVState State = Pair.second;

    assert(!isKnownBaseResult(BDV) && "why did it get added?");
    assert(!State.isUnknown() && "Optimistic algorithm didn't complete!");
    if (!State.isConflict())
      continue;

    if (PHINode *basephi = dyn_cast<PHINode>(State.getBase())) {
      PHINode *phi = cast<PHINode>(BDV);
      unsigned NumPHIValues = phi->getNumIncomingValues();
      for (unsigned i = 0; i < NumPHIValues; i++) {
        Value *InVal = phi->getIncomingValue(i);
        BasicBlock *InBB = phi->getIncomingBlock(i);

        // If we've already seen InBB, add the same incoming value
        // we added for it earlier.  The IR verifier requires phi
        // nodes with multiple entries from the same basic block
        // to have the same incoming value for each of those
        // entries.  If we don't do this check here and basephi
        // has a different type than base, we'll end up adding two
        // bitcasts (and hence two distinct values) as incoming
        // values for the same basic block.

        int blockIndex = basephi->getBasicBlockIndex(InBB);
        if (blockIndex != -1) {
          Value *oldBase = basephi->getIncomingValue(blockIndex);
          basephi->addIncoming(oldBase, InBB);
          
#ifndef NDEBUG
          Value *Base = getBaseForInput(InVal, nullptr);
          // In essence this assert states: the only way two
          // values incoming from the same basic block may be
          // different is by being different bitcasts of the same
          // value.  A cleanup that remains TODO is changing
          // findBaseOrBDV to return an llvm::Value of the correct
          // type (and still remain pure).  This will remove the
          // need to add bitcasts.
          assert(Base->stripPointerCasts() == oldBase->stripPointerCasts() &&
                 "sanity -- findBaseOrBDV should be pure!");
#endif
          continue;
        }

        // Find the instruction which produces the base for each input.  We may
        // need to insert a bitcast in the incoming block.
        // TODO: Need to split critical edges if insertion is needed
        Value *Base = getBaseForInput(InVal, InBB->getTerminator());
        basephi->addIncoming(Base, InBB);
      }
      assert(basephi->getNumIncomingValues() == NumPHIValues);
    } else if (SelectInst *BaseSel = dyn_cast<SelectInst>(State.getBase())) {
      SelectInst *Sel = cast<SelectInst>(BDV);
      // Operand 1 & 2 are true, false path respectively. TODO: refactor to
      // something more safe and less hacky.
      for (int i = 1; i <= 2; i++) {
        Value *InVal = Sel->getOperand(i);
        // Find the instruction which produces the base for each input.  We may
        // need to insert a bitcast.
        Value *Base = getBaseForInput(InVal, BaseSel);
        BaseSel->setOperand(i, Base);
      }
    } else if (auto *BaseEE = dyn_cast<ExtractElementInst>(State.getBase())) {
      Value *InVal = cast<ExtractElementInst>(BDV)->getVectorOperand();
      // Find the instruction which produces the base for each input.  We may
      // need to insert a bitcast.
      Value *Base = getBaseForInput(InVal, BaseEE);
      BaseEE->setOperand(0, Base);
    } else {
      auto *BaseIE = cast<InsertElementInst>(State.getBase());
      auto *BdvIE = cast<InsertElementInst>(BDV);
      auto UpdateOperand = [&](int OperandIdx) {
        Value *InVal = BdvIE->getOperand(OperandIdx);
        Value *Base = getBaseForInput(InVal, BaseIE);
        BaseIE->setOperand(OperandIdx, Base);
      };
      UpdateOperand(0); // vector operand
      UpdateOperand(1); // scalar operand
    }

  }

  // Now that we're done with the algorithm, see if we can optimize the 
  // results slightly by reducing the number of new instructions needed. 
  // Arguably, this should be integrated into the algorithm above, but 
  // doing as a post process step is easier to reason about for the moment.
  DenseMap<Value *, Value *> ReverseMap;
  SmallPtrSet<Instruction *, 16> NewInsts;
  SmallSetVector<AssertingVH<Instruction>, 16> Worklist;
  // Note: We need to visit the states in a deterministic order.  We uses the
  // Keys we sorted above for this purpose.  Note that we are papering over a
  // bigger problem with the algorithm above - it's visit order is not
  // deterministic.  A larger change is needed to fix this.
  for (auto Pair : States) {
    auto *BDV = Pair.first;
    auto State = Pair.second;
    Value *Base = State.getBase();
    assert(BDV && Base);
    assert(!isKnownBaseResult(BDV) && "why did it get added?");
    assert(isKnownBaseResult(Base) &&
           "must be something we 'know' is a base pointer");
    if (!State.isConflict())
      continue;

    ReverseMap[Base] = BDV;
    if (auto *BaseI = dyn_cast<Instruction>(Base)) {
      NewInsts.insert(BaseI);
      Worklist.insert(BaseI);
    }
  }
  auto ReplaceBaseInstWith = [&](Value *BDV, Instruction *BaseI,
                                 Value *Replacement) {
    // Add users which are new instructions (excluding self references)
    for (User *U : BaseI->users())
      if (auto *UI = dyn_cast<Instruction>(U))
        if (NewInsts.count(UI) && UI != BaseI)
          Worklist.insert(UI);
    // Then do the actual replacement
    NewInsts.erase(BaseI);
    ReverseMap.erase(BaseI);
    BaseI->replaceAllUsesWith(Replacement);
    assert(States.count(BDV));
    assert(States[BDV].isConflict() && States[BDV].getBase() == BaseI);
    States[BDV] = BDVState(BDVState::Conflict, Replacement);
    BaseI->eraseFromParent();
  };
  const DataLayout &DL = cast<Instruction>(def)->getModule()->getDataLayout();
  while (!Worklist.empty()) {
    Instruction *BaseI = Worklist.pop_back_val();
    assert(NewInsts.count(BaseI));
    Value *Bdv = ReverseMap[BaseI];
    if (auto *BdvI = dyn_cast<Instruction>(Bdv))
      if (BaseI->isIdenticalTo(BdvI)) {
        DEBUG(dbgs() << "Identical Base: " << *BaseI << "\n");
        ReplaceBaseInstWith(Bdv, BaseI, Bdv);
        continue;
      }
    if (Value *V = SimplifyInstruction(BaseI, DL)) {
      DEBUG(dbgs() << "Base " << *BaseI << " simplified to " << *V << "\n");
      ReplaceBaseInstWith(Bdv, BaseI, V);
      continue;
    }
  }

  // Cache all of our results so we can cheaply reuse them
  // NOTE: This is actually two caches: one of the base defining value
  // relation and one of the base pointer relation!  FIXME
  for (auto Pair : States) {
    auto *BDV = Pair.first;
    Value *base = Pair.second.getBase();
    assert(BDV && base);

    std::string fromstr = cache.count(BDV) ? cache[BDV]->getName() : "none";
    DEBUG(dbgs() << "Updating base value cache"
          << " for: " << BDV->getName()
          << " from: " << fromstr
          << " to: " << base->getName() << "\n");

    if (cache.count(BDV)) {
      // Once we transition from the BDV relation being store in the cache to
      // the base relation being stored, it must be stable
      assert((!isKnownBaseResult(cache[BDV]) || cache[BDV] == base) &&
             "base relation should be stable");
    }
    cache[BDV] = base;
  }
  assert(cache.count(def));
  return cache[def];
}

// For a set of live pointers (base and/or derived), identify the base
// pointer of the object which they are derived from.  This routine will
// mutate the IR graph as needed to make the 'base' pointer live at the
// definition site of 'derived'.  This ensures that any use of 'derived' can
// also use 'base'.  This may involve the insertion of a number of
// additional PHI nodes.
//
// preconditions: live is a set of pointer type Values
//
// side effects: may insert PHI nodes into the existing CFG, will preserve
// CFG, will not remove or mutate any existing nodes
//
// post condition: PointerToBase contains one (derived, base) pair for every
// pointer in live.  Note that derived can be equal to base if the original
// pointer was a base pointer.
static void
findBasePointers(const StatepointLiveSetTy &live,
                 DenseMap<Value *, Value *> &PointerToBase,
                 DominatorTree *DT, DefiningValueMapTy &DVCache) {
  // For the naming of values inserted to be deterministic - which makes for
  // much cleaner and more stable tests - we need to assign an order to the
  // live values.  DenseSets do not provide a deterministic order across runs.
  SmallVector<Value *, 64> Temp;
  Temp.insert(Temp.end(), live.begin(), live.end());
  std::sort(Temp.begin(), Temp.end(), order_by_name);
  for (Value *ptr : Temp) {
    Value *base = findBasePointer(ptr, DVCache);
    assert(base && "failed to find base pointer");
    PointerToBase[ptr] = base;
    assert((!isa<Instruction>(base) || !isa<Instruction>(ptr) ||
            DT->dominates(cast<Instruction>(base)->getParent(),
                          cast<Instruction>(ptr)->getParent())) &&
           "The base we found better dominate the derived pointer");

    // If you see this trip and like to live really dangerously, the code should
    // be correct, just with idioms the verifier can't handle.  You can try
    // disabling the verifier at your own substantial risk.
    assert(!isa<ConstantPointerNull>(base) &&
           "the relocation code needs adjustment to handle the relocation of "
           "a null pointer constant without causing false positives in the "
           "safepoint ir verifier.");
  }
}

/// Find the required based pointers (and adjust the live set) for the given
/// parse point.
static void findBasePointers(DominatorTree &DT, DefiningValueMapTy &DVCache,
                             const CallSite &CS,
                             PartiallyConstructedSafepointRecord &result) {
  DenseMap<Value *, Value *> PointerToBase;
  findBasePointers(result.LiveSet, PointerToBase, &DT, DVCache);

  if (PrintBasePointers) {
    // Note: Need to print these in a stable order since this is checked in
    // some tests.
    errs() << "Base Pairs (w/o Relocation):\n";
    SmallVector<Value *, 64> Temp;
    Temp.reserve(PointerToBase.size());
    for (auto Pair : PointerToBase) {
      Temp.push_back(Pair.first);
    }
    std::sort(Temp.begin(), Temp.end(), order_by_name);
    for (Value *Ptr : Temp) {
      Value *Base = PointerToBase[Ptr];
      errs() << " derived ";
      Ptr->printAsOperand(errs(), false);
      errs() << " base ";
      Base->printAsOperand(errs(), false);
      errs() << "\n";;
    }
  }

  result.PointerToBase = PointerToBase;
}

/// Given an updated version of the dataflow liveness results, update the
/// liveset and base pointer maps for the call site CS.
static void recomputeLiveInValues(GCPtrLivenessData &RevisedLivenessData,
                                  const CallSite &CS,
                                  PartiallyConstructedSafepointRecord &result);

static void recomputeLiveInValues(
    Function &F, DominatorTree &DT, ArrayRef<CallSite> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records) {
  // TODO-PERF: reuse the original liveness, then simply run the dataflow
  // again.  The old values are still live and will help it stabilize quickly.
  GCPtrLivenessData RevisedLivenessData;
  computeLiveInValues(DT, F, RevisedLivenessData);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const CallSite &CS = toUpdate[i];
    recomputeLiveInValues(RevisedLivenessData, CS, info);
  }
}

// When inserting gc.relocate and gc.result calls, we need to ensure there are
// no uses of the original value / return value between the gc.statepoint and
// the gc.relocate / gc.result call.  One case which can arise is a phi node
// starting one of the successor blocks.  We also need to be able to insert the
// gc.relocates only on the path which goes through the statepoint.  We might
// need to split an edge to make this possible.
static BasicBlock *
normalizeForInvokeSafepoint(BasicBlock *BB, BasicBlock *InvokeParent,
                            DominatorTree &DT) {
  BasicBlock *Ret = BB;
  if (!BB->getUniquePredecessor())
    Ret = SplitBlockPredecessors(BB, InvokeParent, "", &DT);

  // Now that 'Ret' has unique predecessor we can safely remove all phi nodes
  // from it
  FoldSingleEntryPHINodes(Ret);
  assert(!isa<PHINode>(Ret->begin()) &&
         "All PHI nodes should have been removed!");

  // At this point, we can safely insert a gc.relocate or gc.result as the first
  // instruction in Ret if needed.
  return Ret;
}

// Create new attribute set containing only attributes which can be transferred
// from original call to the safepoint.
static AttributeSet legalizeCallAttributes(AttributeSet AS) {
  AttributeSet Ret;

  for (unsigned Slot = 0; Slot < AS.getNumSlots(); Slot++) {
    unsigned Index = AS.getSlotIndex(Slot);

    if (Index == AttributeSet::ReturnIndex ||
        Index == AttributeSet::FunctionIndex) {

      for (Attribute Attr : make_range(AS.begin(Slot), AS.end(Slot))) {

        // Do not allow certain attributes - just skip them
        // Safepoint can not be read only or read none.
        if (Attr.hasAttribute(Attribute::ReadNone) ||
            Attr.hasAttribute(Attribute::ReadOnly))
          continue;

        // These attributes control the generation of the gc.statepoint call /
        // invoke itself; and once the gc.statepoint is in place, they're of no
        // use.
        if (Attr.hasAttribute("statepoint-num-patch-bytes") ||
            Attr.hasAttribute("statepoint-id"))
          continue;

        Ret = Ret.addAttributes(
            AS.getContext(), Index,
            AttributeSet::get(AS.getContext(), Index, AttrBuilder(Attr)));
      }
    }

    // Just skip parameter attributes for now
  }

  return Ret;
}

/// Helper function to place all gc relocates necessary for the given
/// statepoint.
/// Inputs:
///   liveVariables - list of variables to be relocated.
///   liveStart - index of the first live variable.
///   basePtrs - base pointers.
///   statepointToken - statepoint instruction to which relocates should be
///   bound.
///   Builder - Llvm IR builder to be used to construct new calls.
static void CreateGCRelocates(ArrayRef<Value *> LiveVariables,
                              const int LiveStart,
                              ArrayRef<Value *> BasePtrs,
                              Instruction *StatepointToken,
                              IRBuilder<> Builder) {
  if (LiveVariables.empty())
    return;

  auto FindIndex = [](ArrayRef<Value *> LiveVec, Value *Val) {
    auto ValIt = std::find(LiveVec.begin(), LiveVec.end(), Val);
    assert(ValIt != LiveVec.end() && "Val not found in LiveVec!");
    size_t Index = std::distance(LiveVec.begin(), ValIt);
    assert(Index < LiveVec.size() && "Bug in std::find?");
    return Index;
  };

  // All gc_relocate are set to i8 addrspace(1)* type. We originally generated
  // unique declarations for each pointer type, but this proved problematic
  // because the intrinsic mangling code is incomplete and fragile.  Since
  // we're moving towards a single unified pointer type anyways, we can just
  // cast everything to an i8* of the right address space.  A bitcast is added
  // later to convert gc_relocate to the actual value's type. 
  Module *M = StatepointToken->getModule();
  auto AS = cast<PointerType>(LiveVariables[0]->getType())->getAddressSpace();
  Type *Types[] = {Type::getInt8PtrTy(M->getContext(), AS)};
  Value *GCRelocateDecl =
    Intrinsic::getDeclaration(M, Intrinsic::experimental_gc_relocate, Types);

  for (unsigned i = 0; i < LiveVariables.size(); i++) {
    // Generate the gc.relocate call and save the result
    Value *BaseIdx =
      Builder.getInt32(LiveStart + FindIndex(LiveVariables, BasePtrs[i]));
    Value *LiveIdx = Builder.getInt32(LiveStart + i);

    // only specify a debug name if we can give a useful one
    CallInst *Reloc = Builder.CreateCall(
        GCRelocateDecl, {StatepointToken, BaseIdx, LiveIdx},
        suffixed_name_or(LiveVariables[i], ".relocated", ""));
    // Trick CodeGen into thinking there are lots of free registers at this
    // fake call.
    Reloc->setCallingConv(CallingConv::Cold);
  }
}

namespace {

/// This struct is used to defer RAUWs and `eraseFromParent` s.  Using this
/// avoids having to worry about keeping around dangling pointers to Values.
class DeferredReplacement {
  AssertingVH<Instruction> Old;
  AssertingVH<Instruction> New;

public:
  explicit DeferredReplacement(Instruction *Old, Instruction *New) :
    Old(Old), New(New) {
    assert(Old != New && "Not allowed!");
  }

  /// Does the task represented by this instance.
  void doReplacement() {
    Instruction *OldI = Old;
    Instruction *NewI = New;

    assert(OldI != NewI && "Disallowed at construction?!");

    Old = nullptr;
    New = nullptr;

    if (NewI)
      OldI->replaceAllUsesWith(NewI);
    OldI->eraseFromParent();
  }
};
}

static void
makeStatepointExplicitImpl(const CallSite CS, /* to replace */
                           const SmallVectorImpl<Value *> &BasePtrs,
                           const SmallVectorImpl<Value *> &LiveVariables,
                           PartiallyConstructedSafepointRecord &Result,
                           std::vector<DeferredReplacement> &Replacements) {
  assert(BasePtrs.size() == LiveVariables.size());
  assert((UseDeoptBundles || isStatepoint(CS)) &&
         "This method expects to be rewriting a statepoint");

  // Then go ahead and use the builder do actually do the inserts.  We insert
  // immediately before the previous instruction under the assumption that all
  // arguments will be available here.  We can't insert afterwards since we may
  // be replacing a terminator.
  Instruction *InsertBefore = CS.getInstruction();
  IRBuilder<> Builder(InsertBefore);

  ArrayRef<Value *> GCArgs(LiveVariables);
  uint64_t StatepointID = 0xABCDEF00;
  uint32_t NumPatchBytes = 0;
  uint32_t Flags = uint32_t(StatepointFlags::None);

  ArrayRef<Use> CallArgs;
  ArrayRef<Use> DeoptArgs;
  ArrayRef<Use> TransitionArgs;

  Value *CallTarget = nullptr;

  if (UseDeoptBundles) {
    CallArgs = {CS.arg_begin(), CS.arg_end()};
    DeoptArgs = GetDeoptBundleOperands(CS);
    // TODO: we don't fill in TransitionArgs or Flags in this branch, but we
    // could have an operand bundle for that too.
    AttributeSet OriginalAttrs = CS.getAttributes();

    Attribute AttrID = OriginalAttrs.getAttribute(AttributeSet::FunctionIndex,
                                                  "statepoint-id");
    if (AttrID.isStringAttribute())
      AttrID.getValueAsString().getAsInteger(10, StatepointID);

    Attribute AttrNumPatchBytes = OriginalAttrs.getAttribute(
        AttributeSet::FunctionIndex, "statepoint-num-patch-bytes");
    if (AttrNumPatchBytes.isStringAttribute())
      AttrNumPatchBytes.getValueAsString().getAsInteger(10, NumPatchBytes);

    CallTarget = CS.getCalledValue();
  } else {
    // This branch will be gone soon, and we will soon only support the
    // UseDeoptBundles == true configuration.
    Statepoint OldSP(CS);
    StatepointID = OldSP.getID();
    NumPatchBytes = OldSP.getNumPatchBytes();
    Flags = OldSP.getFlags();

    CallArgs = {OldSP.arg_begin(), OldSP.arg_end()};
    DeoptArgs = {OldSP.vm_state_begin(), OldSP.vm_state_end()};
    TransitionArgs = {OldSP.gc_transition_args_begin(),
                      OldSP.gc_transition_args_end()};
    CallTarget = OldSP.getCalledValue();
  }

  // Create the statepoint given all the arguments
  Instruction *Token = nullptr;
  AttributeSet ReturnAttrs;
  if (CS.isCall()) {
    CallInst *ToReplace = cast<CallInst>(CS.getInstruction());
    CallInst *Call = Builder.CreateGCStatepointCall(
        StatepointID, NumPatchBytes, CallTarget, Flags, CallArgs,
        TransitionArgs, DeoptArgs, GCArgs, "safepoint_token");

    Call->setTailCall(ToReplace->isTailCall());
    Call->setCallingConv(ToReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    AttributeSet NewAttrs = legalizeCallAttributes(ToReplace->getAttributes());
    // In case if we can handle this set of attributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    Call->setAttributes(NewAttrs.getFnAttributes());
    ReturnAttrs = NewAttrs.getRetAttributes();

    Token = Call;

    // Put the following gc_result and gc_relocate calls immediately after the
    // the old call (which we're about to delete)
    assert(ToReplace->getNextNode() && "Not a terminator, must have next!");
    Builder.SetInsertPoint(ToReplace->getNextNode());
    Builder.SetCurrentDebugLocation(ToReplace->getNextNode()->getDebugLoc());
  } else {
    InvokeInst *ToReplace = cast<InvokeInst>(CS.getInstruction());

    // Insert the new invoke into the old block.  We'll remove the old one in a
    // moment at which point this will become the new terminator for the
    // original block.
    InvokeInst *Invoke = Builder.CreateGCStatepointInvoke(
        StatepointID, NumPatchBytes, CallTarget, ToReplace->getNormalDest(),
        ToReplace->getUnwindDest(), Flags, CallArgs, TransitionArgs, DeoptArgs,
        GCArgs, "statepoint_token");

    Invoke->setCallingConv(ToReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    AttributeSet NewAttrs = legalizeCallAttributes(ToReplace->getAttributes());
    // In case if we can handle this set of attributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    Invoke->setAttributes(NewAttrs.getFnAttributes());
    ReturnAttrs = NewAttrs.getRetAttributes();

    Token = Invoke;

    // Generate gc relocates in exceptional path
    BasicBlock *UnwindBlock = ToReplace->getUnwindDest();
    assert(!isa<PHINode>(UnwindBlock->begin()) &&
           UnwindBlock->getUniquePredecessor() &&
           "can't safely insert in this block!");

    Builder.SetInsertPoint(&*UnwindBlock->getFirstInsertionPt());
    Builder.SetCurrentDebugLocation(ToReplace->getDebugLoc());

    // Attach exceptional gc relocates to the landingpad.
    Instruction *ExceptionalToken = UnwindBlock->getLandingPadInst();
    Result.UnwindToken = ExceptionalToken;

    const unsigned LiveStartIdx = Statepoint(Token).gcArgsStartIdx();
    CreateGCRelocates(LiveVariables, LiveStartIdx, BasePtrs, ExceptionalToken,
                      Builder);

    // Generate gc relocates and returns for normal block
    BasicBlock *NormalDest = ToReplace->getNormalDest();
    assert(!isa<PHINode>(NormalDest->begin()) &&
           NormalDest->getUniquePredecessor() &&
           "can't safely insert in this block!");

    Builder.SetInsertPoint(&*NormalDest->getFirstInsertionPt());

    // gc relocates will be generated later as if it were regular call
    // statepoint
  }
  assert(Token && "Should be set in one of the above branches!");

  if (UseDeoptBundles) {
    Token->setName("statepoint_token");
    if (!CS.getType()->isVoidTy() && !CS.getInstruction()->use_empty()) {
      StringRef Name =
          CS.getInstruction()->hasName() ? CS.getInstruction()->getName() : "";
      CallInst *GCResult = Builder.CreateGCResult(Token, CS.getType(), Name);
      GCResult->setAttributes(CS.getAttributes().getRetAttributes());

      // We cannot RAUW or delete CS.getInstruction() because it could be in the
      // live set of some other safepoint, in which case that safepoint's
      // PartiallyConstructedSafepointRecord will hold a raw pointer to this
      // llvm::Instruction.  Instead, we defer the replacement and deletion to
      // after the live sets have been made explicit in the IR, and we no longer
      // have raw pointers to worry about.
      Replacements.emplace_back(CS.getInstruction(), GCResult);
    } else {
      Replacements.emplace_back(CS.getInstruction(), nullptr);
    }
  } else {
    assert(!CS.getInstruction()->hasNUsesOrMore(2) &&
           "only valid use before rewrite is gc.result");
    assert(!CS.getInstruction()->hasOneUse() ||
           isGCResult(cast<Instruction>(*CS.getInstruction()->user_begin())));

    // Take the name of the original statepoint token if there was one.
    Token->takeName(CS.getInstruction());

    // Update the gc.result of the original statepoint (if any) to use the newly
    // inserted statepoint.  This is safe to do here since the token can't be
    // considered a live reference.
    CS.getInstruction()->replaceAllUsesWith(Token);
    CS.getInstruction()->eraseFromParent();
  }

  Result.StatepointToken = Token;

  // Second, create a gc.relocate for every live variable
  const unsigned LiveStartIdx = Statepoint(Token).gcArgsStartIdx();
  CreateGCRelocates(LiveVariables, LiveStartIdx, BasePtrs, Token, Builder);
}

namespace {
struct NameOrdering {
  Value *Base;
  Value *Derived;

  bool operator()(NameOrdering const &a, NameOrdering const &b) {
    return -1 == a.Derived->getName().compare(b.Derived->getName());
  }
};
}

static void StabilizeOrder(SmallVectorImpl<Value *> &BaseVec,
                           SmallVectorImpl<Value *> &LiveVec) {
  assert(BaseVec.size() == LiveVec.size());

  SmallVector<NameOrdering, 64> Temp;
  for (size_t i = 0; i < BaseVec.size(); i++) {
    NameOrdering v;
    v.Base = BaseVec[i];
    v.Derived = LiveVec[i];
    Temp.push_back(v);
  }

  std::sort(Temp.begin(), Temp.end(), NameOrdering());
  for (size_t i = 0; i < BaseVec.size(); i++) {
    BaseVec[i] = Temp[i].Base;
    LiveVec[i] = Temp[i].Derived;
  }
}

// Replace an existing gc.statepoint with a new one and a set of gc.relocates
// which make the relocations happening at this safepoint explicit.
//
// WARNING: Does not do any fixup to adjust users of the original live
// values.  That's the callers responsibility.
static void
makeStatepointExplicit(DominatorTree &DT, const CallSite &CS,
                       PartiallyConstructedSafepointRecord &Result,
                       std::vector<DeferredReplacement> &Replacements) {
  const auto &LiveSet = Result.LiveSet;
  const auto &PointerToBase = Result.PointerToBase;

  // Convert to vector for efficient cross referencing.
  SmallVector<Value *, 64> BaseVec, LiveVec;
  LiveVec.reserve(LiveSet.size());
  BaseVec.reserve(LiveSet.size());
  for (Value *L : LiveSet) {
    LiveVec.push_back(L);
    assert(PointerToBase.count(L));
    Value *Base = PointerToBase.find(L)->second;
    BaseVec.push_back(Base);
  }
  assert(LiveVec.size() == BaseVec.size());

  // To make the output IR slightly more stable (for use in diffs), ensure a
  // fixed order of the values in the safepoint (by sorting the value name).
  // The order is otherwise meaningless.
  StabilizeOrder(BaseVec, LiveVec);

  // Do the actual rewriting and delete the old statepoint
  makeStatepointExplicitImpl(CS, BaseVec, LiveVec, Result, Replacements);
}

// Helper function for the relocationViaAlloca.
//
// It receives iterator to the statepoint gc relocates and emits a store to the
// assigned location (via allocaMap) for the each one of them.  It adds the
// visited values into the visitedLiveValues set, which we will later use them
// for sanity checking.
static void
insertRelocationStores(iterator_range<Value::user_iterator> GCRelocs,
                       DenseMap<Value *, Value *> &AllocaMap,
                       DenseSet<Value *> &VisitedLiveValues) {

  for (User *U : GCRelocs) {
    GCRelocateInst *Relocate = dyn_cast<GCRelocateInst>(U);
    if (!Relocate)
      continue;

    Value *OriginalValue = const_cast<Value *>(Relocate->getDerivedPtr());
    assert(AllocaMap.count(OriginalValue));
    Value *Alloca = AllocaMap[OriginalValue];

    // Emit store into the related alloca
    // All gc_relocates are i8 addrspace(1)* typed, and it must be bitcasted to
    // the correct type according to alloca.
    assert(Relocate->getNextNode() &&
           "Should always have one since it's not a terminator");
    IRBuilder<> Builder(Relocate->getNextNode());
    Value *CastedRelocatedValue =
      Builder.CreateBitCast(Relocate,
                            cast<AllocaInst>(Alloca)->getAllocatedType(),
                            suffixed_name_or(Relocate, ".casted", ""));

    StoreInst *Store = new StoreInst(CastedRelocatedValue, Alloca);
    Store->insertAfter(cast<Instruction>(CastedRelocatedValue));

#ifndef NDEBUG
    VisitedLiveValues.insert(OriginalValue);
#endif
  }
}

// Helper function for the "relocationViaAlloca". Similar to the
// "insertRelocationStores" but works for rematerialized values.
static void
insertRematerializationStores(
  RematerializedValueMapTy RematerializedValues,
  DenseMap<Value *, Value *> &AllocaMap,
  DenseSet<Value *> &VisitedLiveValues) {

  for (auto RematerializedValuePair: RematerializedValues) {
    Instruction *RematerializedValue = RematerializedValuePair.first;
    Value *OriginalValue = RematerializedValuePair.second;

    assert(AllocaMap.count(OriginalValue) &&
           "Can not find alloca for rematerialized value");
    Value *Alloca = AllocaMap[OriginalValue];

    StoreInst *Store = new StoreInst(RematerializedValue, Alloca);
    Store->insertAfter(RematerializedValue);

#ifndef NDEBUG
    VisitedLiveValues.insert(OriginalValue);
#endif
  }
}

/// Do all the relocation update via allocas and mem2reg
static void relocationViaAlloca(
    Function &F, DominatorTree &DT, ArrayRef<Value *> Live,
    ArrayRef<PartiallyConstructedSafepointRecord> Records) {
#ifndef NDEBUG
  // record initial number of (static) allocas; we'll check we have the same
  // number when we get done.
  int InitialAllocaNum = 0;
  for (auto I = F.getEntryBlock().begin(), E = F.getEntryBlock().end(); I != E;
       I++)
    if (isa<AllocaInst>(*I))
      InitialAllocaNum++;
#endif

  // TODO-PERF: change data structures, reserve
  DenseMap<Value *, Value *> AllocaMap;
  SmallVector<AllocaInst *, 200> PromotableAllocas;
  // Used later to chack that we have enough allocas to store all values
  std::size_t NumRematerializedValues = 0;
  PromotableAllocas.reserve(Live.size());

  // Emit alloca for "LiveValue" and record it in "allocaMap" and
  // "PromotableAllocas"
  auto emitAllocaFor = [&](Value *LiveValue) {
    AllocaInst *Alloca = new AllocaInst(LiveValue->getType(), "",
                                        F.getEntryBlock().getFirstNonPHI());
    AllocaMap[LiveValue] = Alloca;
    PromotableAllocas.push_back(Alloca);
  };

  // Emit alloca for each live gc pointer
  for (Value *V : Live)
    emitAllocaFor(V);

  // Emit allocas for rematerialized values
  for (const auto &Info : Records)
    for (auto RematerializedValuePair : Info.RematerializedValues) {
      Value *OriginalValue = RematerializedValuePair.second;
      if (AllocaMap.count(OriginalValue) != 0)
        continue;

      emitAllocaFor(OriginalValue);
      ++NumRematerializedValues;
    }

  // The next two loops are part of the same conceptual operation.  We need to
  // insert a store to the alloca after the original def and at each
  // redefinition.  We need to insert a load before each use.  These are split
  // into distinct loops for performance reasons.

  // Update gc pointer after each statepoint: either store a relocated value or
  // null (if no relocated value was found for this gc pointer and it is not a
  // gc_result).  This must happen before we update the statepoint with load of
  // alloca otherwise we lose the link between statepoint and old def.
  for (const auto &Info : Records) {
    Value *Statepoint = Info.StatepointToken;

    // This will be used for consistency check
    DenseSet<Value *> VisitedLiveValues;

    // Insert stores for normal statepoint gc relocates
    insertRelocationStores(Statepoint->users(), AllocaMap, VisitedLiveValues);

    // In case if it was invoke statepoint
    // we will insert stores for exceptional path gc relocates.
    if (isa<InvokeInst>(Statepoint)) {
      insertRelocationStores(Info.UnwindToken->users(), AllocaMap,
                             VisitedLiveValues);
    }

    // Do similar thing with rematerialized values
    insertRematerializationStores(Info.RematerializedValues, AllocaMap,
                                  VisitedLiveValues);

    if (ClobberNonLive) {
      // As a debugging aid, pretend that an unrelocated pointer becomes null at
      // the gc.statepoint.  This will turn some subtle GC problems into
      // slightly easier to debug SEGVs.  Note that on large IR files with
      // lots of gc.statepoints this is extremely costly both memory and time
      // wise.
      SmallVector<AllocaInst *, 64> ToClobber;
      for (auto Pair : AllocaMap) {
        Value *Def = Pair.first;
        AllocaInst *Alloca = cast<AllocaInst>(Pair.second);

        // This value was relocated
        if (VisitedLiveValues.count(Def)) {
          continue;
        }
        ToClobber.push_back(Alloca);
      }

      auto InsertClobbersAt = [&](Instruction *IP) {
        for (auto *AI : ToClobber) {
          auto AIType = cast<PointerType>(AI->getType());
          auto PT = cast<PointerType>(AIType->getElementType());
          Constant *CPN = ConstantPointerNull::get(PT);
          StoreInst *Store = new StoreInst(CPN, AI);
          Store->insertBefore(IP);
        }
      };

      // Insert the clobbering stores.  These may get intermixed with the
      // gc.results and gc.relocates, but that's fine.
      if (auto II = dyn_cast<InvokeInst>(Statepoint)) {
        InsertClobbersAt(&*II->getNormalDest()->getFirstInsertionPt());
        InsertClobbersAt(&*II->getUnwindDest()->getFirstInsertionPt());
      } else {
        InsertClobbersAt(cast<Instruction>(Statepoint)->getNextNode());
      }
    }
  }

  // Update use with load allocas and add store for gc_relocated.
  for (auto Pair : AllocaMap) {
    Value *Def = Pair.first;
    Value *Alloca = Pair.second;

    // We pre-record the uses of allocas so that we dont have to worry about
    // later update that changes the user information..

    SmallVector<Instruction *, 20> Uses;
    // PERF: trade a linear scan for repeated reallocation
    Uses.reserve(std::distance(Def->user_begin(), Def->user_end()));
    for (User *U : Def->users()) {
      if (!isa<ConstantExpr>(U)) {
        // If the def has a ConstantExpr use, then the def is either a
        // ConstantExpr use itself or null.  In either case
        // (recursively in the first, directly in the second), the oop
        // it is ultimately dependent on is null and this particular
        // use does not need to be fixed up.
        Uses.push_back(cast<Instruction>(U));
      }
    }

    std::sort(Uses.begin(), Uses.end());
    auto Last = std::unique(Uses.begin(), Uses.end());
    Uses.erase(Last, Uses.end());

    for (Instruction *Use : Uses) {
      if (isa<PHINode>(Use)) {
        PHINode *Phi = cast<PHINode>(Use);
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          if (Def == Phi->getIncomingValue(i)) {
            LoadInst *Load = new LoadInst(
                Alloca, "", Phi->getIncomingBlock(i)->getTerminator());
            Phi->setIncomingValue(i, Load);
          }
        }
      } else {
        LoadInst *Load = new LoadInst(Alloca, "", Use);
        Use->replaceUsesOfWith(Def, Load);
      }
    }

    // Emit store for the initial gc value.  Store must be inserted after load,
    // otherwise store will be in alloca's use list and an extra load will be
    // inserted before it.
    StoreInst *Store = new StoreInst(Def, Alloca);
    if (Instruction *Inst = dyn_cast<Instruction>(Def)) {
      if (InvokeInst *Invoke = dyn_cast<InvokeInst>(Inst)) {
        // InvokeInst is a TerminatorInst so the store need to be inserted
        // into its normal destination block.
        BasicBlock *NormalDest = Invoke->getNormalDest();
        Store->insertBefore(NormalDest->getFirstNonPHI());
      } else {
        assert(!Inst->isTerminator() &&
               "The only TerminatorInst that can produce a value is "
               "InvokeInst which is handled above.");
        Store->insertAfter(Inst);
      }
    } else {
      assert(isa<Argument>(Def));
      Store->insertAfter(cast<Instruction>(Alloca));
    }
  }

  assert(PromotableAllocas.size() == Live.size() + NumRematerializedValues &&
         "we must have the same allocas with lives");
  if (!PromotableAllocas.empty()) {
    // Apply mem2reg to promote alloca to SSA
    PromoteMemToReg(PromotableAllocas, DT);
  }

#ifndef NDEBUG
  for (auto &I : F.getEntryBlock())
    if (isa<AllocaInst>(I))
      InitialAllocaNum--;
  assert(InitialAllocaNum == 0 && "We must not introduce any extra allocas");
#endif
}

/// Implement a unique function which doesn't require we sort the input
/// vector.  Doing so has the effect of changing the output of a couple of
/// tests in ways which make them less useful in testing fused safepoints.
template <typename T> static void unique_unsorted(SmallVectorImpl<T> &Vec) {
  SmallSet<T, 8> Seen;
  Vec.erase(std::remove_if(Vec.begin(), Vec.end(), [&](const T &V) {
              return !Seen.insert(V).second;
            }), Vec.end());
}

/// Insert holders so that each Value is obviously live through the entire
/// lifetime of the call.
static void insertUseHolderAfter(CallSite &CS, const ArrayRef<Value *> Values,
                                 SmallVectorImpl<CallInst *> &Holders) {
  if (Values.empty())
    // No values to hold live, might as well not insert the empty holder
    return;

  Module *M = CS.getInstruction()->getModule();
  // Use a dummy vararg function to actually hold the values live
  Function *Func = cast<Function>(M->getOrInsertFunction(
      "__tmp_use", FunctionType::get(Type::getVoidTy(M->getContext()), true)));
  if (CS.isCall()) {
    // For call safepoints insert dummy calls right after safepoint
    Holders.push_back(CallInst::Create(Func, Values, "",
                                       &*++CS.getInstruction()->getIterator()));
    return;
  }
  // For invoke safepooints insert dummy calls both in normal and
  // exceptional destination blocks
  auto *II = cast<InvokeInst>(CS.getInstruction());
  Holders.push_back(CallInst::Create(
      Func, Values, "", &*II->getNormalDest()->getFirstInsertionPt()));
  Holders.push_back(CallInst::Create(
      Func, Values, "", &*II->getUnwindDest()->getFirstInsertionPt()));
}

static void findLiveReferences(
    Function &F, DominatorTree &DT, ArrayRef<CallSite> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records) {
  GCPtrLivenessData OriginalLivenessData;
  computeLiveInValues(DT, F, OriginalLivenessData);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const CallSite &CS = toUpdate[i];
    analyzeParsePointLiveness(DT, OriginalLivenessData, CS, info);
  }
}

/// Remove any vector of pointers from the live set by scalarizing them over the
/// statepoint instruction.  Adds the scalarized pieces to the live set.  It
/// would be preferable to include the vector in the statepoint itself, but
/// the lowering code currently does not handle that.  Extending it would be
/// slightly non-trivial since it requires a format change.  Given how rare
/// such cases are (for the moment?) scalarizing is an acceptable compromise.
static void splitVectorValues(Instruction *StatepointInst,
                              StatepointLiveSetTy &LiveSet,
                              DenseMap<Value *, Value *>& PointerToBase,
                              DominatorTree &DT) {
  SmallVector<Value *, 16> ToSplit;
  for (Value *V : LiveSet)
    if (isa<VectorType>(V->getType()))
      ToSplit.push_back(V);

  if (ToSplit.empty())
    return;

  DenseMap<Value *, SmallVector<Value *, 16>> ElementMapping;

  Function &F = *(StatepointInst->getParent()->getParent());

  DenseMap<Value *, AllocaInst *> AllocaMap;
  // First is normal return, second is exceptional return (invoke only)
  DenseMap<Value *, std::pair<Value *, Value *>> Replacements;
  for (Value *V : ToSplit) {
    AllocaInst *Alloca =
        new AllocaInst(V->getType(), "", F.getEntryBlock().getFirstNonPHI());
    AllocaMap[V] = Alloca;

    VectorType *VT = cast<VectorType>(V->getType());
    IRBuilder<> Builder(StatepointInst);
    SmallVector<Value *, 16> Elements;
    for (unsigned i = 0; i < VT->getNumElements(); i++)
      Elements.push_back(Builder.CreateExtractElement(V, Builder.getInt32(i)));
    ElementMapping[V] = Elements;

    auto InsertVectorReform = [&](Instruction *IP) {
      Builder.SetInsertPoint(IP);
      Builder.SetCurrentDebugLocation(IP->getDebugLoc());
      Value *ResultVec = UndefValue::get(VT);
      for (unsigned i = 0; i < VT->getNumElements(); i++)
        ResultVec = Builder.CreateInsertElement(ResultVec, Elements[i],
                                                Builder.getInt32(i));
      return ResultVec;
    };

    if (isa<CallInst>(StatepointInst)) {
      BasicBlock::iterator Next(StatepointInst);
      Next++;
      Instruction *IP = &*(Next);
      Replacements[V].first = InsertVectorReform(IP);
      Replacements[V].second = nullptr;
    } else {
      InvokeInst *Invoke = cast<InvokeInst>(StatepointInst);
      // We've already normalized - check that we don't have shared destination
      // blocks
      BasicBlock *NormalDest = Invoke->getNormalDest();
      assert(!isa<PHINode>(NormalDest->begin()));
      BasicBlock *UnwindDest = Invoke->getUnwindDest();
      assert(!isa<PHINode>(UnwindDest->begin()));
      // Insert insert element sequences in both successors
      Instruction *IP = &*(NormalDest->getFirstInsertionPt());
      Replacements[V].first = InsertVectorReform(IP);
      IP = &*(UnwindDest->getFirstInsertionPt());
      Replacements[V].second = InsertVectorReform(IP);
    }
  }

  for (Value *V : ToSplit) {
    AllocaInst *Alloca = AllocaMap[V];

    // Capture all users before we start mutating use lists
    SmallVector<Instruction *, 16> Users;
    for (User *U : V->users())
      Users.push_back(cast<Instruction>(U));

    for (Instruction *I : Users) {
      if (auto Phi = dyn_cast<PHINode>(I)) {
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++)
          if (V == Phi->getIncomingValue(i)) {
            LoadInst *Load = new LoadInst(
                Alloca, "", Phi->getIncomingBlock(i)->getTerminator());
            Phi->setIncomingValue(i, Load);
          }
      } else {
        LoadInst *Load = new LoadInst(Alloca, "", I);
        I->replaceUsesOfWith(V, Load);
      }
    }

    // Store the original value and the replacement value into the alloca
    StoreInst *Store = new StoreInst(V, Alloca);
    if (auto I = dyn_cast<Instruction>(V))
      Store->insertAfter(I);
    else
      Store->insertAfter(Alloca);

    // Normal return for invoke, or call return
    Instruction *Replacement = cast<Instruction>(Replacements[V].first);
    (new StoreInst(Replacement, Alloca))->insertAfter(Replacement);
    // Unwind return for invoke only
    Replacement = cast_or_null<Instruction>(Replacements[V].second);
    if (Replacement)
      (new StoreInst(Replacement, Alloca))->insertAfter(Replacement);
  }

  // apply mem2reg to promote alloca to SSA
  SmallVector<AllocaInst *, 16> Allocas;
  for (Value *V : ToSplit)
    Allocas.push_back(AllocaMap[V]);
  PromoteMemToReg(Allocas, DT);

  // Update our tracking of live pointers and base mappings to account for the
  // changes we just made.
  for (Value *V : ToSplit) {
    auto &Elements = ElementMapping[V];

    LiveSet.erase(V);
    LiveSet.insert(Elements.begin(), Elements.end());
    // We need to update the base mapping as well.
    assert(PointerToBase.count(V));
    Value *OldBase = PointerToBase[V];
    auto &BaseElements = ElementMapping[OldBase];
    PointerToBase.erase(V);
    assert(Elements.size() == BaseElements.size());
    for (unsigned i = 0; i < Elements.size(); i++) {
      Value *Elem = Elements[i];
      PointerToBase[Elem] = BaseElements[i];
    }
  }
}

// Helper function for the "rematerializeLiveValues". It walks use chain
// starting from the "CurrentValue" until it meets "BaseValue". Only "simple"
// values are visited (currently it is GEP's and casts). Returns true if it
// successfully reached "BaseValue" and false otherwise.
// Fills "ChainToBase" array with all visited values. "BaseValue" is not
// recorded.
static bool findRematerializableChainToBasePointer(
  SmallVectorImpl<Instruction*> &ChainToBase,
  Value *CurrentValue, Value *BaseValue) {

  // We have found a base value
  if (CurrentValue == BaseValue) {
    return true;
  }

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurrentValue)) {
    ChainToBase.push_back(GEP);
    return findRematerializableChainToBasePointer(ChainToBase,
                                                  GEP->getPointerOperand(),
                                                  BaseValue);
  }

  if (CastInst *CI = dyn_cast<CastInst>(CurrentValue)) {
    if (!CI->isNoopCast(CI->getModule()->getDataLayout()))
      return false;

    ChainToBase.push_back(CI);
    return findRematerializableChainToBasePointer(ChainToBase,
                                                  CI->getOperand(0), BaseValue);
  }

  // Not supported instruction in the chain
  return false;
}

// Helper function for the "rematerializeLiveValues". Compute cost of the use
// chain we are going to rematerialize.
static unsigned
chainToBasePointerCost(SmallVectorImpl<Instruction*> &Chain,
                       TargetTransformInfo &TTI) {
  unsigned Cost = 0;

  for (Instruction *Instr : Chain) {
    if (CastInst *CI = dyn_cast<CastInst>(Instr)) {
      assert(CI->isNoopCast(CI->getModule()->getDataLayout()) &&
             "non noop cast is found during rematerialization");

      Type *SrcTy = CI->getOperand(0)->getType();
      Cost += TTI.getCastInstrCost(CI->getOpcode(), CI->getType(), SrcTy);

    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Instr)) {
      // Cost of the address calculation
      Type *ValTy = GEP->getPointerOperandType()->getPointerElementType();
      Cost += TTI.getAddressComputationCost(ValTy);

      // And cost of the GEP itself
      // TODO: Use TTI->getGEPCost here (it exists, but appears to be not
      //       allowed for the external usage)
      if (!GEP->hasAllConstantIndices())
        Cost += 2;

    } else {
      llvm_unreachable("unsupported instruciton type during rematerialization");
    }
  }

  return Cost;
}

// From the statepoint live set pick values that are cheaper to recompute then
// to relocate. Remove this values from the live set, rematerialize them after
// statepoint and record them in "Info" structure. Note that similar to
// relocated values we don't do any user adjustments here.
static void rematerializeLiveValues(CallSite CS,
                                    PartiallyConstructedSafepointRecord &Info,
                                    TargetTransformInfo &TTI) {
  const unsigned int ChainLengthThreshold = 10;

  // Record values we are going to delete from this statepoint live set.
  // We can not di this in following loop due to iterator invalidation.
  SmallVector<Value *, 32> LiveValuesToBeDeleted;

  for (Value *LiveValue: Info.LiveSet) {
    // For each live pointer find it's defining chain
    SmallVector<Instruction *, 3> ChainToBase;
    assert(Info.PointerToBase.count(LiveValue));
    bool FoundChain =
      findRematerializableChainToBasePointer(ChainToBase,
                                             LiveValue,
                                             Info.PointerToBase[LiveValue]);
    // Nothing to do, or chain is too long
    if (!FoundChain ||
        ChainToBase.size() == 0 ||
        ChainToBase.size() > ChainLengthThreshold)
      continue;

    // Compute cost of this chain
    unsigned Cost = chainToBasePointerCost(ChainToBase, TTI);
    // TODO: We can also account for cases when we will be able to remove some
    //       of the rematerialized values by later optimization passes. I.e if
    //       we rematerialized several intersecting chains. Or if original values
    //       don't have any uses besides this statepoint.

    // For invokes we need to rematerialize each chain twice - for normal and
    // for unwind basic blocks. Model this by multiplying cost by two.
    if (CS.isInvoke()) {
      Cost *= 2;
    }
    // If it's too expensive - skip it
    if (Cost >= RematerializationThreshold)
      continue;

    // Remove value from the live set
    LiveValuesToBeDeleted.push_back(LiveValue);

    // Clone instructions and record them inside "Info" structure

    // Walk backwards to visit top-most instructions first
    std::reverse(ChainToBase.begin(), ChainToBase.end());

    // Utility function which clones all instructions from "ChainToBase"
    // and inserts them before "InsertBefore". Returns rematerialized value
    // which should be used after statepoint.
    auto rematerializeChain = [&ChainToBase](Instruction *InsertBefore) {
      Instruction *LastClonedValue = nullptr;
      Instruction *LastValue = nullptr;
      for (Instruction *Instr: ChainToBase) {
        // Only GEP's and casts are suported as we need to be careful to not
        // introduce any new uses of pointers not in the liveset.
        // Note that it's fine to introduce new uses of pointers which were
        // otherwise not used after this statepoint.
        assert(isa<GetElementPtrInst>(Instr) || isa<CastInst>(Instr));

        Instruction *ClonedValue = Instr->clone();
        ClonedValue->insertBefore(InsertBefore);
        ClonedValue->setName(Instr->getName() + ".remat");

        // If it is not first instruction in the chain then it uses previously
        // cloned value. We should update it to use cloned value.
        if (LastClonedValue) {
          assert(LastValue);
          ClonedValue->replaceUsesOfWith(LastValue, LastClonedValue);
#ifndef NDEBUG
          // Assert that cloned instruction does not use any instructions from
          // this chain other than LastClonedValue
          for (auto OpValue : ClonedValue->operand_values()) {
            assert(std::find(ChainToBase.begin(), ChainToBase.end(), OpValue) ==
                       ChainToBase.end() &&
                   "incorrect use in rematerialization chain");
          }
#endif
        }

        LastClonedValue = ClonedValue;
        LastValue = Instr;
      }
      assert(LastClonedValue);
      return LastClonedValue;
    };

    // Different cases for calls and invokes. For invokes we need to clone
    // instructions both on normal and unwind path.
    if (CS.isCall()) {
      Instruction *InsertBefore = CS.getInstruction()->getNextNode();
      assert(InsertBefore);
      Instruction *RematerializedValue = rematerializeChain(InsertBefore);
      Info.RematerializedValues[RematerializedValue] = LiveValue;
    } else {
      InvokeInst *Invoke = cast<InvokeInst>(CS.getInstruction());

      Instruction *NormalInsertBefore =
          &*Invoke->getNormalDest()->getFirstInsertionPt();
      Instruction *UnwindInsertBefore =
          &*Invoke->getUnwindDest()->getFirstInsertionPt();

      Instruction *NormalRematerializedValue =
          rematerializeChain(NormalInsertBefore);
      Instruction *UnwindRematerializedValue =
          rematerializeChain(UnwindInsertBefore);

      Info.RematerializedValues[NormalRematerializedValue] = LiveValue;
      Info.RematerializedValues[UnwindRematerializedValue] = LiveValue;
    }
  }

  // Remove rematerializaed values from the live set
  for (auto LiveValue: LiveValuesToBeDeleted) {
    Info.LiveSet.erase(LiveValue);
  }
}

static bool insertParsePoints(Function &F, DominatorTree &DT,
                              TargetTransformInfo &TTI,
                              SmallVectorImpl<CallSite> &ToUpdate) {
#ifndef NDEBUG
  // sanity check the input
  std::set<CallSite> Uniqued;
  Uniqued.insert(ToUpdate.begin(), ToUpdate.end());
  assert(Uniqued.size() == ToUpdate.size() && "no duplicates please!");

  for (CallSite CS : ToUpdate) {
    assert(CS.getInstruction()->getParent()->getParent() == &F);
    assert((UseDeoptBundles || isStatepoint(CS)) &&
           "expected to already be a deopt statepoint");
  }
#endif

  // When inserting gc.relocates for invokes, we need to be able to insert at
  // the top of the successor blocks.  See the comment on
  // normalForInvokeSafepoint on exactly what is needed.  Note that this step
  // may restructure the CFG.
  for (CallSite CS : ToUpdate) {
    if (!CS.isInvoke())
      continue;
    auto *II = cast<InvokeInst>(CS.getInstruction());
    normalizeForInvokeSafepoint(II->getNormalDest(), II->getParent(), DT);
    normalizeForInvokeSafepoint(II->getUnwindDest(), II->getParent(), DT);
  }

  // A list of dummy calls added to the IR to keep various values obviously
  // live in the IR.  We'll remove all of these when done.
  SmallVector<CallInst *, 64> Holders;

  // Insert a dummy call with all of the arguments to the vm_state we'll need
  // for the actual safepoint insertion.  This ensures reference arguments in
  // the deopt argument list are considered live through the safepoint (and
  // thus makes sure they get relocated.)
  for (CallSite CS : ToUpdate) {
    SmallVector<Value *, 64> DeoptValues;

    iterator_range<const Use *> DeoptStateRange =
        UseDeoptBundles
            ? iterator_range<const Use *>(GetDeoptBundleOperands(CS))
            : iterator_range<const Use *>(Statepoint(CS).vm_state_args());

    for (Value *Arg : DeoptStateRange) {
      assert(!isUnhandledGCPointerType(Arg->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(Arg->getType()))
        DeoptValues.push_back(Arg);
    }

    insertUseHolderAfter(CS, DeoptValues, Holders);
  }

  SmallVector<PartiallyConstructedSafepointRecord, 64> Records(ToUpdate.size());

  // A) Identify all gc pointers which are statically live at the given call
  // site.
  findLiveReferences(F, DT, ToUpdate, Records);

  // B) Find the base pointers for each live pointer
  /* scope for caching */ {
    // Cache the 'defining value' relation used in the computation and
    // insertion of base phis and selects.  This ensures that we don't insert
    // large numbers of duplicate base_phis.
    DefiningValueMapTy DVCache;

    for (size_t i = 0; i < Records.size(); i++) {
      PartiallyConstructedSafepointRecord &info = Records[i];
      findBasePointers(DT, DVCache, ToUpdate[i], info);
    }
  } // end of cache scope

  // The base phi insertion logic (for any safepoint) may have inserted new
  // instructions which are now live at some safepoint.  The simplest such
  // example is:
  // loop:
  //   phi a  <-- will be a new base_phi here
  //   safepoint 1 <-- that needs to be live here
  //   gep a + 1
  //   safepoint 2
  //   br loop
  // We insert some dummy calls after each safepoint to definitely hold live
  // the base pointers which were identified for that safepoint.  We'll then
  // ask liveness for _every_ base inserted to see what is now live.  Then we
  // remove the dummy calls.
  Holders.reserve(Holders.size() + Records.size());
  for (size_t i = 0; i < Records.size(); i++) {
    PartiallyConstructedSafepointRecord &Info = Records[i];

    SmallVector<Value *, 128> Bases;
    for (auto Pair : Info.PointerToBase)
      Bases.push_back(Pair.second);

    insertUseHolderAfter(ToUpdate[i], Bases, Holders);
  }

  // By selecting base pointers, we've effectively inserted new uses. Thus, we
  // need to rerun liveness.  We may *also* have inserted new defs, but that's
  // not the key issue.
  recomputeLiveInValues(F, DT, ToUpdate, Records);

  if (PrintBasePointers) {
    for (auto &Info : Records) {
      errs() << "Base Pairs: (w/Relocation)\n";
      for (auto Pair : Info.PointerToBase) {
        errs() << " derived ";
        Pair.first->printAsOperand(errs(), false);
        errs() << " base ";
        Pair.second->printAsOperand(errs(), false);
        errs() << "\n";
      }
    }
  }

  // It is possible that non-constant live variables have a constant base.  For
  // example, a GEP with a variable offset from a global.  In this case we can
  // remove it from the liveset.  We already don't add constants to the liveset
  // because we assume they won't move at runtime and the GC doesn't need to be
  // informed about them.  The same reasoning applies if the base is constant.
  // Note that the relocation placement code relies on this filtering for
  // correctness as it expects the base to be in the liveset, which isn't true
  // if the base is constant.
  for (auto &Info : Records)
    for (auto &BasePair : Info.PointerToBase)
      if (isa<Constant>(BasePair.second))
        Info.LiveSet.erase(BasePair.first);

  for (CallInst *CI : Holders)
    CI->eraseFromParent();

  Holders.clear();

  // Do a limited scalarization of any live at safepoint vector values which
  // contain pointers.  This enables this pass to run after vectorization at
  // the cost of some possible performance loss.  Note: This is known to not
  // handle updating of the side tables correctly which can lead to relocation
  // bugs when the same vector is live at multiple statepoints.  We're in the
  // process of implementing the alternate lowering - relocating the
  // vector-of-pointers as first class item and updating the backend to
  // understand that - but that's not yet complete.  
  if (UseVectorSplit)
    for (size_t i = 0; i < Records.size(); i++) {
      PartiallyConstructedSafepointRecord &Info = Records[i];
      Instruction *Statepoint = ToUpdate[i].getInstruction();
      splitVectorValues(cast<Instruction>(Statepoint), Info.LiveSet,
                        Info.PointerToBase, DT);
    }

  // In order to reduce live set of statepoint we might choose to rematerialize
  // some values instead of relocating them. This is purely an optimization and
  // does not influence correctness.
  for (size_t i = 0; i < Records.size(); i++)
    rematerializeLiveValues(ToUpdate[i], Records[i], TTI);

  // We need this to safely RAUW and delete call or invoke return values that
  // may themselves be live over a statepoint.  For details, please see usage in
  // makeStatepointExplicitImpl.
  std::vector<DeferredReplacement> Replacements;

  // Now run through and replace the existing statepoints with new ones with
  // the live variables listed.  We do not yet update uses of the values being
  // relocated. We have references to live variables that need to
  // survive to the last iteration of this loop.  (By construction, the
  // previous statepoint can not be a live variable, thus we can and remove
  // the old statepoint calls as we go.)
  for (size_t i = 0; i < Records.size(); i++)
    makeStatepointExplicit(DT, ToUpdate[i], Records[i], Replacements);

  ToUpdate.clear(); // prevent accident use of invalid CallSites

  for (auto &PR : Replacements)
    PR.doReplacement();

  Replacements.clear();

  for (auto &Info : Records) {
    // These live sets may contain state Value pointers, since we replaced calls
    // with operand bundles with calls wrapped in gc.statepoint, and some of
    // those calls may have been def'ing live gc pointers.  Clear these out to
    // avoid accidentally using them.
    //
    // TODO: We should create a separate data structure that does not contain
    // these live sets, and migrate to using that data structure from this point
    // onward.
    Info.LiveSet.clear();
    Info.PointerToBase.clear();
  }

  // Do all the fixups of the original live variables to their relocated selves
  SmallVector<Value *, 128> Live;
  for (size_t i = 0; i < Records.size(); i++) {
    PartiallyConstructedSafepointRecord &Info = Records[i];

    // We can't simply save the live set from the original insertion.  One of
    // the live values might be the result of a call which needs a safepoint.
    // That Value* no longer exists and we need to use the new gc_result.
    // Thankfully, the live set is embedded in the statepoint (and updated), so
    // we just grab that.
    Statepoint Statepoint(Info.StatepointToken);
    Live.insert(Live.end(), Statepoint.gc_args_begin(),
                Statepoint.gc_args_end());
#ifndef NDEBUG
    // Do some basic sanity checks on our liveness results before performing
    // relocation.  Relocation can and will turn mistakes in liveness results
    // into non-sensical code which is must harder to debug.
    // TODO: It would be nice to test consistency as well
    assert(DT.isReachableFromEntry(Info.StatepointToken->getParent()) &&
           "statepoint must be reachable or liveness is meaningless");
    for (Value *V : Statepoint.gc_args()) {
      if (!isa<Instruction>(V))
        // Non-instruction values trivial dominate all possible uses
        continue;
      auto *LiveInst = cast<Instruction>(V);
      assert(DT.isReachableFromEntry(LiveInst->getParent()) &&
             "unreachable values should never be live");
      assert(DT.dominates(LiveInst, Info.StatepointToken) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
#endif
  }
  unique_unsorted(Live);

#ifndef NDEBUG
  // sanity check
  for (auto *Ptr : Live)
    assert(isGCPointerType(Ptr->getType()) && "must be a gc pointer type");
#endif

  relocationViaAlloca(F, DT, Live, Records);
  return !Records.empty();
}

// Handles both return values and arguments for Functions and CallSites.
template <typename AttrHolder>
static void RemoveNonValidAttrAtIndex(LLVMContext &Ctx, AttrHolder &AH,
                                      unsigned Index) {
  AttrBuilder R;
  if (AH.getDereferenceableBytes(Index))
    R.addAttribute(Attribute::get(Ctx, Attribute::Dereferenceable,
                                  AH.getDereferenceableBytes(Index)));
  if (AH.getDereferenceableOrNullBytes(Index))
    R.addAttribute(Attribute::get(Ctx, Attribute::DereferenceableOrNull,
                                  AH.getDereferenceableOrNullBytes(Index)));
  if (AH.doesNotAlias(Index))
    R.addAttribute(Attribute::NoAlias);

  if (!R.empty())
    AH.setAttributes(AH.getAttributes().removeAttributes(
        Ctx, Index, AttributeSet::get(Ctx, Index, R)));
}

void
RewriteStatepointsForGC::stripNonValidAttributesFromPrototype(Function &F) {
  LLVMContext &Ctx = F.getContext();

  for (Argument &A : F.args())
    if (isa<PointerType>(A.getType()))
      RemoveNonValidAttrAtIndex(Ctx, F, A.getArgNo() + 1);

  if (isa<PointerType>(F.getReturnType()))
    RemoveNonValidAttrAtIndex(Ctx, F, AttributeSet::ReturnIndex);
}

void RewriteStatepointsForGC::stripNonValidAttributesFromBody(Function &F) {
  if (F.empty())
    return;

  LLVMContext &Ctx = F.getContext();
  MDBuilder Builder(Ctx);

  for (Instruction &I : instructions(F)) {
    if (const MDNode *MD = I.getMetadata(LLVMContext::MD_tbaa)) {
      assert(MD->getNumOperands() < 5 && "unrecognized metadata shape!");
      bool IsImmutableTBAA =
          MD->getNumOperands() == 4 &&
          mdconst::extract<ConstantInt>(MD->getOperand(3))->getValue() == 1;

      if (!IsImmutableTBAA)
        continue; // no work to do, MD_tbaa is already marked mutable

      MDNode *Base = cast<MDNode>(MD->getOperand(0));
      MDNode *Access = cast<MDNode>(MD->getOperand(1));
      uint64_t Offset =
          mdconst::extract<ConstantInt>(MD->getOperand(2))->getZExtValue();

      MDNode *MutableTBAA =
          Builder.createTBAAStructTagNode(Base, Access, Offset);
      I.setMetadata(LLVMContext::MD_tbaa, MutableTBAA);
    }

    if (CallSite CS = CallSite(&I)) {
      for (int i = 0, e = CS.arg_size(); i != e; i++)
        if (isa<PointerType>(CS.getArgument(i)->getType()))
          RemoveNonValidAttrAtIndex(Ctx, CS, i + 1);
      if (isa<PointerType>(CS.getType()))
        RemoveNonValidAttrAtIndex(Ctx, CS, AttributeSet::ReturnIndex);
    }
  }
}

/// Returns true if this function should be rewritten by this pass.  The main
/// point of this function is as an extension point for custom logic.
static bool shouldRewriteStatepointsIn(Function &F) {
  // TODO: This should check the GCStrategy
  if (F.hasGC()) {
    const char *FunctionGCName = F.getGC();
    const StringRef StatepointExampleName("statepoint-example");
    const StringRef CoreCLRName("coreclr");
    return (StatepointExampleName == FunctionGCName) ||
           (CoreCLRName == FunctionGCName);
  } else
    return false;
}

void RewriteStatepointsForGC::stripNonValidAttributes(Module &M) {
#ifndef NDEBUG
  assert(std::any_of(M.begin(), M.end(), shouldRewriteStatepointsIn) &&
         "precondition!");
#endif

  for (Function &F : M)
    stripNonValidAttributesFromPrototype(F);

  for (Function &F : M)
    stripNonValidAttributesFromBody(F);
}

bool RewriteStatepointsForGC::runOnFunction(Function &F) {
  // Nothing to do for declarations.
  if (F.isDeclaration() || F.empty())
    return false;

  // Policy choice says not to rewrite - the most common reason is that we're
  // compiling code without a GCStrategy.
  if (!shouldRewriteStatepointsIn(F))
    return false;

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  auto NeedsRewrite = [](Instruction &I) {
    if (UseDeoptBundles) {
      if (ImmutableCallSite CS = ImmutableCallSite(&I))
        return !callsGCLeafFunction(CS);
      return false;
    }

    return isStatepoint(I);
  };

  // Gather all the statepoints which need rewritten.  Be careful to only
  // consider those in reachable code since we need to ask dominance queries
  // when rewriting.  We'll delete the unreachable ones in a moment.
  SmallVector<CallSite, 64> ParsePointNeeded;
  bool HasUnreachableStatepoint = false;
  for (Instruction &I : instructions(F)) {
    // TODO: only the ones with the flag set!
    if (NeedsRewrite(I)) {
      if (DT.isReachableFromEntry(I.getParent()))
        ParsePointNeeded.push_back(CallSite(&I));
      else
        HasUnreachableStatepoint = true;
    }
  }

  bool MadeChange = false;

  // Delete any unreachable statepoints so that we don't have unrewritten
  // statepoints surviving this pass.  This makes testing easier and the
  // resulting IR less confusing to human readers.  Rather than be fancy, we
  // just reuse a utility function which removes the unreachable blocks.
  if (HasUnreachableStatepoint)
    MadeChange |= removeUnreachableBlocks(F);

  // Return early if no work to do.
  if (ParsePointNeeded.empty())
    return MadeChange;

  // As a prepass, go ahead and aggressively destroy single entry phi nodes.
  // These are created by LCSSA.  They have the effect of increasing the size
  // of liveness sets for no good reason.  It may be harder to do this post
  // insertion since relocations and base phis can confuse things.
  for (BasicBlock &BB : F)
    if (BB.getUniquePredecessor()) {
      MadeChange = true;
      FoldSingleEntryPHINodes(&BB);
    }

  // Before we start introducing relocations, we want to tweak the IR a bit to
  // avoid unfortunate code generation effects.  The main example is that we 
  // want to try to make sure the comparison feeding a branch is after any
  // safepoints.  Otherwise, we end up with a comparison of pre-relocation
  // values feeding a branch after relocation.  This is semantically correct,
  // but results in extra register pressure since both the pre-relocation and
  // post-relocation copies must be available in registers.  For code without
  // relocations this is handled elsewhere, but teaching the scheduler to
  // reverse the transform we're about to do would be slightly complex.
  // Note: This may extend the live range of the inputs to the icmp and thus
  // increase the liveset of any statepoint we move over.  This is profitable
  // as long as all statepoints are in rare blocks.  If we had in-register
  // lowering for live values this would be a much safer transform.
  auto getConditionInst = [](TerminatorInst *TI) -> Instruction* {
    if (auto *BI = dyn_cast<BranchInst>(TI))
      if (BI->isConditional())
        return dyn_cast<Instruction>(BI->getCondition());
    // TODO: Extend this to handle switches
    return nullptr;
  };
  for (BasicBlock &BB : F) {
    TerminatorInst *TI = BB.getTerminator();
    if (auto *Cond = getConditionInst(TI))
      // TODO: Handle more than just ICmps here.  We should be able to move
      // most instructions without side effects or memory access.  
      if (isa<ICmpInst>(Cond) && Cond->hasOneUse()) {
        MadeChange = true;
        Cond->moveBefore(TI);
      }
  }

  MadeChange |= insertParsePoints(F, DT, TTI, ParsePointNeeded);
  return MadeChange;
}

// liveness computation via standard dataflow
// -------------------------------------------------------------------

// TODO: Consider using bitvectors for liveness, the set of potentially
// interesting values should be small and easy to pre-compute.

/// Compute the live-in set for the location rbegin starting from
/// the live-out set of the basic block
static void computeLiveInValues(BasicBlock::reverse_iterator rbegin,
                                BasicBlock::reverse_iterator rend,
                                DenseSet<Value *> &LiveTmp) {

  for (BasicBlock::reverse_iterator ritr = rbegin; ritr != rend; ritr++) {
    Instruction *I = &*ritr;

    // KILL/Def - Remove this definition from LiveIn
    LiveTmp.erase(I);

    // Don't consider *uses* in PHI nodes, we handle their contribution to
    // predecessor blocks when we seed the LiveOut sets
    if (isa<PHINode>(I))
      continue;

    // USE - Add to the LiveIn set for this instruction
    for (Value *V : I->operands()) {
      assert(!isUnhandledGCPointerType(V->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(V->getType()) && !isa<Constant>(V)) {
        // The choice to exclude all things constant here is slightly subtle.
        // There are two independent reasons:
        // - We assume that things which are constant (from LLVM's definition)
        // do not move at runtime.  For example, the address of a global
        // variable is fixed, even though it's contents may not be.
        // - Second, we can't disallow arbitrary inttoptr constants even
        // if the language frontend does.  Optimization passes are free to
        // locally exploit facts without respect to global reachability.  This
        // can create sections of code which are dynamically unreachable and
        // contain just about anything.  (see constants.ll in tests)
        LiveTmp.insert(V);
      }
    }
  }
}

static void computeLiveOutSeed(BasicBlock *BB, DenseSet<Value *> &LiveTmp) {

  for (BasicBlock *Succ : successors(BB)) {
    const BasicBlock::iterator E(Succ->getFirstNonPHI());
    for (BasicBlock::iterator I = Succ->begin(); I != E; I++) {
      PHINode *Phi = cast<PHINode>(&*I);
      Value *V = Phi->getIncomingValueForBlock(BB);
      assert(!isUnhandledGCPointerType(V->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(V->getType()) && !isa<Constant>(V)) {
        LiveTmp.insert(V);
      }
    }
  }
}

static DenseSet<Value *> computeKillSet(BasicBlock *BB) {
  DenseSet<Value *> KillSet;
  for (Instruction &I : *BB)
    if (isHandledGCPointerType(I.getType()))
      KillSet.insert(&I);
  return KillSet;
}

#ifndef NDEBUG
/// Check that the items in 'Live' dominate 'TI'.  This is used as a basic
/// sanity check for the liveness computation.
static void checkBasicSSA(DominatorTree &DT, DenseSet<Value *> &Live,
                          TerminatorInst *TI, bool TermOkay = false) {
  for (Value *V : Live) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      // The terminator can be a member of the LiveOut set.  LLVM's definition
      // of instruction dominance states that V does not dominate itself.  As
      // such, we need to special case this to allow it.
      if (TermOkay && TI == I)
        continue;
      assert(DT.dominates(I, TI) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
  }
}

/// Check that all the liveness sets used during the computation of liveness
/// obey basic SSA properties.  This is useful for finding cases where we miss
/// a def.
static void checkBasicSSA(DominatorTree &DT, GCPtrLivenessData &Data,
                          BasicBlock &BB) {
  checkBasicSSA(DT, Data.LiveSet[&BB], BB.getTerminator());
  checkBasicSSA(DT, Data.LiveOut[&BB], BB.getTerminator(), true);
  checkBasicSSA(DT, Data.LiveIn[&BB], BB.getTerminator());
}
#endif

static void computeLiveInValues(DominatorTree &DT, Function &F,
                                GCPtrLivenessData &Data) {

  SmallSetVector<BasicBlock *, 200> Worklist;
  auto AddPredsToWorklist = [&](BasicBlock *BB) {
    // We use a SetVector so that we don't have duplicates in the worklist.
    Worklist.insert(pred_begin(BB), pred_end(BB));
  };
  auto NextItem = [&]() {
    BasicBlock *BB = Worklist.back();
    Worklist.pop_back();
    return BB;
  };

  // Seed the liveness for each individual block
  for (BasicBlock &BB : F) {
    Data.KillSet[&BB] = computeKillSet(&BB);
    Data.LiveSet[&BB].clear();
    computeLiveInValues(BB.rbegin(), BB.rend(), Data.LiveSet[&BB]);

#ifndef NDEBUG
    for (Value *Kill : Data.KillSet[&BB])
      assert(!Data.LiveSet[&BB].count(Kill) && "live set contains kill");
#endif

    Data.LiveOut[&BB] = DenseSet<Value *>();
    computeLiveOutSeed(&BB, Data.LiveOut[&BB]);
    Data.LiveIn[&BB] = Data.LiveSet[&BB];
    set_union(Data.LiveIn[&BB], Data.LiveOut[&BB]);
    set_subtract(Data.LiveIn[&BB], Data.KillSet[&BB]);
    if (!Data.LiveIn[&BB].empty())
      AddPredsToWorklist(&BB);
  }

  // Propagate that liveness until stable
  while (!Worklist.empty()) {
    BasicBlock *BB = NextItem();

    // Compute our new liveout set, then exit early if it hasn't changed
    // despite the contribution of our successor.
    DenseSet<Value *> LiveOut = Data.LiveOut[BB];
    const auto OldLiveOutSize = LiveOut.size();
    for (BasicBlock *Succ : successors(BB)) {
      assert(Data.LiveIn.count(Succ));
      set_union(LiveOut, Data.LiveIn[Succ]);
    }
    // assert OutLiveOut is a subset of LiveOut
    if (OldLiveOutSize == LiveOut.size()) {
      // If the sets are the same size, then we didn't actually add anything
      // when unioning our successors LiveIn  Thus, the LiveIn of this block
      // hasn't changed.
      continue;
    }
    Data.LiveOut[BB] = LiveOut;

    // Apply the effects of this basic block
    DenseSet<Value *> LiveTmp = LiveOut;
    set_union(LiveTmp, Data.LiveSet[BB]);
    set_subtract(LiveTmp, Data.KillSet[BB]);

    assert(Data.LiveIn.count(BB));
    const DenseSet<Value *> &OldLiveIn = Data.LiveIn[BB];
    // assert: OldLiveIn is a subset of LiveTmp
    if (OldLiveIn.size() != LiveTmp.size()) {
      Data.LiveIn[BB] = LiveTmp;
      AddPredsToWorklist(BB);
    }
  } // while( !worklist.empty() )

#ifndef NDEBUG
  // Sanity check our output against SSA properties.  This helps catch any
  // missing kills during the above iteration.
  for (BasicBlock &BB : F) {
    checkBasicSSA(DT, Data, BB);
  }
#endif
}

static void findLiveSetAtInst(Instruction *Inst, GCPtrLivenessData &Data,
                              StatepointLiveSetTy &Out) {

  BasicBlock *BB = Inst->getParent();

  // Note: The copy is intentional and required
  assert(Data.LiveOut.count(BB));
  DenseSet<Value *> LiveOut = Data.LiveOut[BB];

  // We want to handle the statepoint itself oddly.  It's
  // call result is not live (normal), nor are it's arguments
  // (unless they're used again later).  This adjustment is
  // specifically what we need to relocate
  BasicBlock::reverse_iterator rend(Inst->getIterator());
  computeLiveInValues(BB->rbegin(), rend, LiveOut);
  LiveOut.erase(Inst);
  Out.insert(LiveOut.begin(), LiveOut.end());
}

static void recomputeLiveInValues(GCPtrLivenessData &RevisedLivenessData,
                                  const CallSite &CS,
                                  PartiallyConstructedSafepointRecord &Info) {
  Instruction *Inst = CS.getInstruction();
  StatepointLiveSetTy Updated;
  findLiveSetAtInst(Inst, RevisedLivenessData, Updated);

#ifndef NDEBUG
  DenseSet<Value *> Bases;
  for (auto KVPair : Info.PointerToBase) {
    Bases.insert(KVPair.second);
  }
#endif
  // We may have base pointers which are now live that weren't before.  We need
  // to update the PointerToBase structure to reflect this.
  for (auto V : Updated)
    if (!Info.PointerToBase.count(V)) {
      assert(Bases.count(V) && "can't find base for unexpected live value");
      Info.PointerToBase[V] = V;
      continue;
    }

#ifndef NDEBUG
  for (auto V : Updated) {
    assert(Info.PointerToBase.count(V) &&
           "must be able to find base for live value");
  }
#endif

  // Remove any stale base mappings - this can happen since our liveness is
  // more precise then the one inherent in the base pointer analysis
  DenseSet<Value *> ToErase;
  for (auto KVPair : Info.PointerToBase)
    if (!Updated.count(KVPair.first))
      ToErase.insert(KVPair.first);
  for (auto V : ToErase)
    Info.PointerToBase.erase(V);

#ifndef NDEBUG
  for (auto KVPair : Info.PointerToBase)
    assert(Updated.count(KVPair.first) && "record for non-live value");
#endif

  Info.LiveSet = Updated;
}
