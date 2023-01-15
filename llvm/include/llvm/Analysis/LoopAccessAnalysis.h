//===- llvm/Analysis/LoopAccessAnalysis.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the loop memory dependence framework that
// was originally developed for the Loop Vectorizer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPACCESSANALYSIS_H
#define LLVM_ANALYSIS_LOOPACCESSANALYSIS_H

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Pass.h"
#include <optional>

namespace llvm {

class AAResults;
class DataLayout;
class Loop;
class LoopAccessInfo;
class raw_ostream;
class SCEV;
class SCEVUnionPredicate;
class Value;

/// Collection of parameters shared beetween the Loop Vectorizer and the
/// Loop Access Analysis.
struct VectorizerParams {
  /// Maximum SIMD width.
  static const unsigned MaxVectorWidth;

  /// VF as overridden by the user.
  static unsigned VectorizationFactor;
  /// Interleave factor as overridden by the user.
  static unsigned VectorizationInterleave;
  /// True if force-vector-interleave was specified by the user.
  static bool isInterleaveForced();

  /// \When performing memory disambiguation checks at runtime do not
  /// make more than this number of comparisons.
  static unsigned RuntimeMemoryCheckThreshold;
};

/// Checks memory dependences among accesses to the same underlying
/// object to determine whether there vectorization is legal or not (and at
/// which vectorization factor).
///
/// Note: This class will compute a conservative dependence for access to
/// different underlying pointers. Clients, such as the loop vectorizer, will
/// sometimes deal these potential dependencies by emitting runtime checks.
///
/// We use the ScalarEvolution framework to symbolically evalutate access
/// functions pairs. Since we currently don't restructure the loop we can rely
/// on the program order of memory accesses to determine their safety.
/// At the moment we will only deem accesses as safe for:
///  * A negative constant distance assuming program order.
///
///      Safe: tmp = a[i + 1];     OR     a[i + 1] = x;
///            a[i] = tmp;                y = a[i];
///
///   The latter case is safe because later checks guarantuee that there can't
///   be a cycle through a phi node (that is, we check that "x" and "y" is not
///   the same variable: a header phi can only be an induction or a reduction, a
///   reduction can't have a memory sink, an induction can't have a memory
///   source). This is important and must not be violated (or we have to
///   resort to checking for cycles through memory).
///
///  * A positive constant distance assuming program order that is bigger
///    than the biggest memory access.
///
///     tmp = a[i]        OR              b[i] = x
///     a[i+2] = tmp                      y = b[i+2];
///
///     Safe distance: 2 x sizeof(a[0]), and 2 x sizeof(b[0]), respectively.
///
///  * Zero distances and all accesses have the same size.
///
class MemoryDepChecker {
public:
  typedef PointerIntPair<Value *, 1, bool> MemAccessInfo;
  typedef SmallVector<MemAccessInfo, 8> MemAccessInfoList;
  /// Set of potential dependent memory accesses.
  typedef EquivalenceClasses<MemAccessInfo> DepCandidates;

  /// Type to keep track of the status of the dependence check. The order of
  /// the elements is important and has to be from most permissive to least
  /// permissive.
  enum class VectorizationSafetyStatus {
    // Can vectorize safely without RT checks. All dependences are known to be
    // safe.
    Safe,
    // Can possibly vectorize with RT checks to overcome unknown dependencies.
    PossiblySafeWithRtChecks,
    // Cannot vectorize due to known unsafe dependencies.
    Unsafe,
  };

  /// Dependece between memory access instructions.
  struct Dependence {
    /// The type of the dependence.
    enum DepType {
      // No dependence.
      NoDep,
      // We couldn't determine the direction or the distance.
      Unknown,
      // Lexically forward.
      //
      // FIXME: If we only have loop-independent forward dependences (e.g. a
      // read and write of A[i]), LAA will locally deem the dependence "safe"
      // without querying the MemoryDepChecker.  Therefore we can miss
      // enumerating loop-independent forward dependences in
      // getDependences.  Note that as soon as there are different
      // indices used to access the same array, the MemoryDepChecker *is*
      // queried and the dependence list is complete.
      Forward,
      // Forward, but if vectorized, is likely to prevent store-to-load
      // forwarding.
      ForwardButPreventsForwarding,
      // Lexically backward.
      Backward,
      // Backward, but the distance allows a vectorization factor of
      // MaxSafeDepDistBytes.
      BackwardVectorizable,
      // Same, but may prevent store-to-load forwarding.
      BackwardVectorizableButPreventsForwarding
    };

    /// String version of the types.
    static const char *DepName[];

    /// Index of the source of the dependence in the InstMap vector.
    unsigned Source;
    /// Index of the destination of the dependence in the InstMap vector.
    unsigned Destination;
    /// The type of the dependence.
    DepType Type;

    Dependence(unsigned Source, unsigned Destination, DepType Type)
        : Source(Source), Destination(Destination), Type(Type) {}

    /// Return the source instruction of the dependence.
    Instruction *getSource(const LoopAccessInfo &LAI) const;
    /// Return the destination instruction of the dependence.
    Instruction *getDestination(const LoopAccessInfo &LAI) const;

    /// Dependence types that don't prevent vectorization.
    static VectorizationSafetyStatus isSafeForVectorization(DepType Type);

    /// Lexically forward dependence.
    bool isForward() const;
    /// Lexically backward dependence.
    bool isBackward() const;

    /// May be a lexically backward dependence type (includes Unknown).
    bool isPossiblyBackward() const;

    /// Print the dependence.  \p Instr is used to map the instruction
    /// indices to instructions.
    void print(raw_ostream &OS, unsigned Depth,
               const SmallVectorImpl<Instruction *> &Instrs) const;
  };

  MemoryDepChecker(PredicatedScalarEvolution &PSE, const Loop *L)
      : PSE(PSE), InnermostLoop(L) {}

  /// Register the location (instructions are given increasing numbers)
  /// of a write access.
  void addAccess(StoreInst *SI);

  /// Register the location (instructions are given increasing numbers)
  /// of a write access.
  void addAccess(LoadInst *LI);

  /// Check whether the dependencies between the accesses are safe.
  ///
  /// Only checks sets with elements in \p CheckDeps.
  bool areDepsSafe(DepCandidates &AccessSets, MemAccessInfoList &CheckDeps,
                   const ValueToValueMap &Strides);

  /// No memory dependence was encountered that would inhibit
  /// vectorization.
  bool isSafeForVectorization() const {
    return Status == VectorizationSafetyStatus::Safe;
  }

  /// Return true if the number of elements that are safe to operate on
  /// simultaneously is not bounded.
  bool isSafeForAnyVectorWidth() const {
    return MaxSafeVectorWidthInBits == UINT_MAX;
  }

  /// The maximum number of bytes of a vector register we can vectorize
  /// the accesses safely with.
  uint64_t getMaxSafeDepDistBytes() { return MaxSafeDepDistBytes; }

  /// Return the number of elements that are safe to operate on
  /// simultaneously, multiplied by the size of the element in bits.
  uint64_t getMaxSafeVectorWidthInBits() const {
    return MaxSafeVectorWidthInBits;
  }

  /// In same cases when the dependency check fails we can still
  /// vectorize the loop with a dynamic array access check.
  bool shouldRetryWithRuntimeCheck() const {
    return FoundNonConstantDistanceDependence &&
           Status == VectorizationSafetyStatus::PossiblySafeWithRtChecks;
  }

  /// Returns the memory dependences.  If null is returned we exceeded
  /// the MaxDependences threshold and this information is not
  /// available.
  const SmallVectorImpl<Dependence> *getDependences() const {
    return RecordDependences ? &Dependences : nullptr;
  }

  void clearDependences() { Dependences.clear(); }

  /// The vector of memory access instructions.  The indices are used as
  /// instruction identifiers in the Dependence class.
  const SmallVectorImpl<Instruction *> &getMemoryInstructions() const {
    return InstMap;
  }

  /// Generate a mapping between the memory instructions and their
  /// indices according to program order.
  DenseMap<Instruction *, unsigned> generateInstructionOrderMap() const {
    DenseMap<Instruction *, unsigned> OrderMap;

    for (unsigned I = 0; I < InstMap.size(); ++I)
      OrderMap[InstMap[I]] = I;

    return OrderMap;
  }

  /// Find the set of instructions that read or write via \p Ptr.
  SmallVector<Instruction *, 4> getInstructionsForAccess(Value *Ptr,
                                                         bool isWrite) const;

  /// Return the program order indices for the access location (Ptr, IsWrite).
  /// Returns an empty ArrayRef if there are no accesses for the location.
  ArrayRef<unsigned> getOrderForAccess(Value *Ptr, bool IsWrite) const {
    auto I = Accesses.find({Ptr, IsWrite});
    if (I != Accesses.end())
      return I->second;
    return {};
  }

  const Loop *getInnermostLoop() const { return InnermostLoop; }

private:
  /// A wrapper around ScalarEvolution, used to add runtime SCEV checks, and
  /// applies dynamic knowledge to simplify SCEV expressions and convert them
  /// to a more usable form. We need this in case assumptions about SCEV
  /// expressions need to be made in order to avoid unknown dependences. For
  /// example we might assume a unit stride for a pointer in order to prove
  /// that a memory access is strided and doesn't wrap.
  PredicatedScalarEvolution &PSE;
  const Loop *InnermostLoop;

  /// Maps access locations (ptr, read/write) to program order.
  DenseMap<MemAccessInfo, std::vector<unsigned> > Accesses;

  /// Memory access instructions in program order.
  SmallVector<Instruction *, 16> InstMap;

  /// The program order index to be used for the next instruction.
  unsigned AccessIdx = 0;

  // We can access this many bytes in parallel safely.
  uint64_t MaxSafeDepDistBytes = 0;

  /// Number of elements (from consecutive iterations) that are safe to
  /// operate on simultaneously, multiplied by the size of the element in bits.
  /// The size of the element is taken from the memory access that is most
  /// restrictive.
  uint64_t MaxSafeVectorWidthInBits = -1U;

  /// If we see a non-constant dependence distance we can still try to
  /// vectorize this loop with runtime checks.
  bool FoundNonConstantDistanceDependence = false;

  /// Result of the dependence checks, indicating whether the checked
  /// dependences are safe for vectorization, require RT checks or are known to
  /// be unsafe.
  VectorizationSafetyStatus Status = VectorizationSafetyStatus::Safe;

  //// True if Dependences reflects the dependences in the
  //// loop.  If false we exceeded MaxDependences and
  //// Dependences is invalid.
  bool RecordDependences = true;

  /// Memory dependences collected during the analysis.  Only valid if
  /// RecordDependences is true.
  SmallVector<Dependence, 8> Dependences;

  /// Check whether there is a plausible dependence between the two
  /// accesses.
  ///
  /// Access \p A must happen before \p B in program order. The two indices
  /// identify the index into the program order map.
  ///
  /// This function checks  whether there is a plausible dependence (or the
  /// absence of such can't be proved) between the two accesses. If there is a
  /// plausible dependence but the dependence distance is bigger than one
  /// element access it records this distance in \p MaxSafeDepDistBytes (if this
  /// distance is smaller than any other distance encountered so far).
  /// Otherwise, this function returns true signaling a possible dependence.
  Dependence::DepType isDependent(const MemAccessInfo &A, unsigned AIdx,
                                  const MemAccessInfo &B, unsigned BIdx,
                                  const ValueToValueMap &Strides);

  /// Check whether the data dependence could prevent store-load
  /// forwarding.
  ///
  /// \return false if we shouldn't vectorize at all or avoid larger
  /// vectorization factors by limiting MaxSafeDepDistBytes.
  bool couldPreventStoreLoadForward(uint64_t Distance, uint64_t TypeByteSize);

  /// Updates the current safety status with \p S. We can go from Safe to
  /// either PossiblySafeWithRtChecks or Unsafe and from
  /// PossiblySafeWithRtChecks to Unsafe.
  void mergeInStatus(VectorizationSafetyStatus S);
};

class RuntimePointerChecking;
/// A grouping of pointers. A single memcheck is required between
/// two groups.
struct RuntimeCheckingPtrGroup {
  /// Create a new pointer checking group containing a single
  /// pointer, with index \p Index in RtCheck.
  RuntimeCheckingPtrGroup(unsigned Index, RuntimePointerChecking &RtCheck);

  /// Tries to add the pointer recorded in RtCheck at index
  /// \p Index to this pointer checking group. We can only add a pointer
  /// to a checking group if we will still be able to get
  /// the upper and lower bounds of the check. Returns true in case
  /// of success, false otherwise.
  bool addPointer(unsigned Index, RuntimePointerChecking &RtCheck);
  bool addPointer(unsigned Index, const SCEV *Start, const SCEV *End,
                  unsigned AS, bool NeedsFreeze, ScalarEvolution &SE);

  /// The SCEV expression which represents the upper bound of all the
  /// pointers in this group.
  const SCEV *High;
  /// The SCEV expression which represents the lower bound of all the
  /// pointers in this group.
  const SCEV *Low;
  /// Indices of all the pointers that constitute this grouping.
  SmallVector<unsigned, 2> Members;
  /// Address space of the involved pointers.
  unsigned AddressSpace;
  /// Whether the pointer needs to be frozen after expansion, e.g. because it
  /// may be poison outside the loop.
  bool NeedsFreeze = false;
};

/// A memcheck which made up of a pair of grouped pointers.
typedef std::pair<const RuntimeCheckingPtrGroup *,
                  const RuntimeCheckingPtrGroup *>
    RuntimePointerCheck;

struct PointerDiffInfo {
  const SCEV *SrcStart;
  const SCEV *SinkStart;
  unsigned AccessSize;
  bool NeedsFreeze;

  PointerDiffInfo(const SCEV *SrcStart, const SCEV *SinkStart,
                  unsigned AccessSize, bool NeedsFreeze)
      : SrcStart(SrcStart), SinkStart(SinkStart), AccessSize(AccessSize),
        NeedsFreeze(NeedsFreeze) {}
};

/// Holds information about the memory runtime legality checks to verify
/// that a group of pointers do not overlap.
class RuntimePointerChecking {
  friend struct RuntimeCheckingPtrGroup;

public:
  struct PointerInfo {
    /// Holds the pointer value that we need to check.
    TrackingVH<Value> PointerValue;
    /// Holds the smallest byte address accessed by the pointer throughout all
    /// iterations of the loop.
    const SCEV *Start;
    /// Holds the largest byte address accessed by the pointer throughout all
    /// iterations of the loop, plus 1.
    const SCEV *End;
    /// Holds the information if this pointer is used for writing to memory.
    bool IsWritePtr;
    /// Holds the id of the set of pointers that could be dependent because of a
    /// shared underlying object.
    unsigned DependencySetId;
    /// Holds the id of the disjoint alias set to which this pointer belongs.
    unsigned AliasSetId;
    /// SCEV for the access.
    const SCEV *Expr;
    /// True if the pointer expressions needs to be frozen after expansion.
    bool NeedsFreeze;

    PointerInfo(Value *PointerValue, const SCEV *Start, const SCEV *End,
                bool IsWritePtr, unsigned DependencySetId, unsigned AliasSetId,
                const SCEV *Expr, bool NeedsFreeze)
        : PointerValue(PointerValue), Start(Start), End(End),
          IsWritePtr(IsWritePtr), DependencySetId(DependencySetId),
          AliasSetId(AliasSetId), Expr(Expr), NeedsFreeze(NeedsFreeze) {}
  };

  RuntimePointerChecking(MemoryDepChecker &DC, ScalarEvolution *SE)
      : DC(DC), SE(SE) {}

  /// Reset the state of the pointer runtime information.
  void reset() {
    Need = false;
    Pointers.clear();
    Checks.clear();
  }

  /// Insert a pointer and calculate the start and end SCEVs.
  /// We need \p PSE in order to compute the SCEV expression of the pointer
  /// according to the assumptions that we've made during the analysis.
  /// The method might also version the pointer stride according to \p Strides,
  /// and add new predicates to \p PSE.
  void insert(Loop *Lp, Value *Ptr, const SCEV *PtrExpr, Type *AccessTy,
              bool WritePtr, unsigned DepSetId, unsigned ASId,
              PredicatedScalarEvolution &PSE, bool NeedsFreeze);

  /// No run-time memory checking is necessary.
  bool empty() const { return Pointers.empty(); }

  /// Generate the checks and store it.  This also performs the grouping
  /// of pointers to reduce the number of memchecks necessary.
  void generateChecks(MemoryDepChecker::DepCandidates &DepCands,
                      bool UseDependencies);

  /// Returns the checks that generateChecks created. They can be used to ensure
  /// no read/write accesses overlap across all loop iterations.
  const SmallVectorImpl<RuntimePointerCheck> &getChecks() const {
    return Checks;
  }

  // Returns an optional list of (pointer-difference expressions, access size)
  // pairs that can be used to prove that there are no vectorization-preventing
  // dependencies at runtime. There are is a vectorization-preventing dependency
  // if any pointer-difference is <u VF * InterleaveCount * access size. Returns
  // std::nullopt if pointer-difference checks cannot be used.
  std::optional<ArrayRef<PointerDiffInfo>> getDiffChecks() const {
    if (!CanUseDiffCheck)
      return std::nullopt;
    return {DiffChecks};
  }

  /// Decide if we need to add a check between two groups of pointers,
  /// according to needsChecking.
  bool needsChecking(const RuntimeCheckingPtrGroup &M,
                     const RuntimeCheckingPtrGroup &N) const;

  /// Returns the number of run-time checks required according to
  /// needsChecking.
  unsigned getNumberOfChecks() const { return Checks.size(); }

  /// Print the list run-time memory checks necessary.
  void print(raw_ostream &OS, unsigned Depth = 0) const;

  /// Print \p Checks.
  void printChecks(raw_ostream &OS,
                   const SmallVectorImpl<RuntimePointerCheck> &Checks,
                   unsigned Depth = 0) const;

  /// This flag indicates if we need to add the runtime check.
  bool Need = false;

  /// Information about the pointers that may require checking.
  SmallVector<PointerInfo, 2> Pointers;

  /// Holds a partitioning of pointers into "check groups".
  SmallVector<RuntimeCheckingPtrGroup, 2> CheckingGroups;

  /// Check if pointers are in the same partition
  ///
  /// \p PtrToPartition contains the partition number for pointers (-1 if the
  /// pointer belongs to multiple partitions).
  static bool
  arePointersInSamePartition(const SmallVectorImpl<int> &PtrToPartition,
                             unsigned PtrIdx1, unsigned PtrIdx2);

  /// Decide whether we need to issue a run-time check for pointer at
  /// index \p I and \p J to prove their independence.
  bool needsChecking(unsigned I, unsigned J) const;

  /// Return PointerInfo for pointer at index \p PtrIdx.
  const PointerInfo &getPointerInfo(unsigned PtrIdx) const {
    return Pointers[PtrIdx];
  }

  ScalarEvolution *getSE() const { return SE; }

private:
  /// Groups pointers such that a single memcheck is required
  /// between two different groups. This will clear the CheckingGroups vector
  /// and re-compute it. We will only group dependecies if \p UseDependencies
  /// is true, otherwise we will create a separate group for each pointer.
  void groupChecks(MemoryDepChecker::DepCandidates &DepCands,
                   bool UseDependencies);

  /// Generate the checks and return them.
  SmallVector<RuntimePointerCheck, 4> generateChecks();

  /// Try to create add a new (pointer-difference, access size) pair to
  /// DiffCheck for checking groups \p CGI and \p CGJ. If pointer-difference
  /// checks cannot be used for the groups, set CanUseDiffCheck to false.
  void tryToCreateDiffCheck(const RuntimeCheckingPtrGroup &CGI,
                            const RuntimeCheckingPtrGroup &CGJ);

  MemoryDepChecker &DC;

  /// Holds a pointer to the ScalarEvolution analysis.
  ScalarEvolution *SE;

  /// Set of run-time checks required to establish independence of
  /// otherwise may-aliasing pointers in the loop.
  SmallVector<RuntimePointerCheck, 4> Checks;

  /// Flag indicating if pointer-difference checks can be used
  bool CanUseDiffCheck = true;

  /// A list of (pointer-difference, access size) pairs that can be used to
  /// prove that there are no vectorization-preventing dependencies.
  SmallVector<PointerDiffInfo> DiffChecks;
};

/// Drive the analysis of memory accesses in the loop
///
/// This class is responsible for analyzing the memory accesses of a loop.  It
/// collects the accesses and then its main helper the AccessAnalysis class
/// finds and categorizes the dependences in buildDependenceSets.
///
/// For memory dependences that can be analyzed at compile time, it determines
/// whether the dependence is part of cycle inhibiting vectorization.  This work
/// is delegated to the MemoryDepChecker class.
///
/// For memory dependences that cannot be determined at compile time, it
/// generates run-time checks to prove independence.  This is done by
/// AccessAnalysis::canCheckPtrAtRT and the checks are maintained by the
/// RuntimePointerCheck class.
///
/// If pointers can wrap or can't be expressed as affine AddRec expressions by
/// ScalarEvolution, we will generate run-time checks by emitting a
/// SCEVUnionPredicate.
///
/// Checks for both memory dependences and the SCEV predicates contained in the
/// PSE must be emitted in order for the results of this analysis to be valid.
class LoopAccessInfo {
public:
  LoopAccessInfo(Loop *L, ScalarEvolution *SE, const TargetLibraryInfo *TLI,
                 AAResults *AA, DominatorTree *DT, LoopInfo *LI);

  /// Return true we can analyze the memory accesses in the loop and there are
  /// no memory dependence cycles.
  bool canVectorizeMemory() const { return CanVecMem; }

  /// Return true if there is a convergent operation in the loop. There may
  /// still be reported runtime pointer checks that would be required, but it is
  /// not legal to insert them.
  bool hasConvergentOp() const { return HasConvergentOp; }

  const RuntimePointerChecking *getRuntimePointerChecking() const {
    return PtrRtChecking.get();
  }

  /// Number of memchecks required to prove independence of otherwise
  /// may-alias pointers.
  unsigned getNumRuntimePointerChecks() const {
    return PtrRtChecking->getNumberOfChecks();
  }

  /// Return true if the block BB needs to be predicated in order for the loop
  /// to be vectorized.
  static bool blockNeedsPredication(BasicBlock *BB, Loop *TheLoop,
                                    DominatorTree *DT);

  /// Returns true if the value V is uniform within the loop.
  bool isUniform(Value *V) const;

  uint64_t getMaxSafeDepDistBytes() const { return MaxSafeDepDistBytes; }
  unsigned getNumStores() const { return NumStores; }
  unsigned getNumLoads() const { return NumLoads;}

  /// The diagnostics report generated for the analysis.  E.g. why we
  /// couldn't analyze the loop.
  const OptimizationRemarkAnalysis *getReport() const { return Report.get(); }

  /// the Memory Dependence Checker which can determine the
  /// loop-independent and loop-carried dependences between memory accesses.
  const MemoryDepChecker &getDepChecker() const { return *DepChecker; }

  /// Return the list of instructions that use \p Ptr to read or write
  /// memory.
  SmallVector<Instruction *, 4> getInstructionsForAccess(Value *Ptr,
                                                         bool isWrite) const {
    return DepChecker->getInstructionsForAccess(Ptr, isWrite);
  }

  /// If an access has a symbolic strides, this maps the pointer value to
  /// the stride symbol.
  const ValueToValueMap &getSymbolicStrides() const { return SymbolicStrides; }

  /// Pointer has a symbolic stride.
  bool hasStride(Value *V) const { return StrideSet.count(V); }

  /// Print the information about the memory accesses in the loop.
  void print(raw_ostream &OS, unsigned Depth = 0) const;

  /// If the loop has memory dependence involving an invariant address, i.e. two
  /// stores or a store and a load, then return true, else return false.
  bool hasDependenceInvolvingLoopInvariantAddress() const {
    return HasDependenceInvolvingLoopInvariantAddress;
  }

  /// Return the list of stores to invariant addresses.
  ArrayRef<StoreInst *> getStoresToInvariantAddresses() const {
    return StoresToInvariantAddresses;
  }

  /// Used to add runtime SCEV checks. Simplifies SCEV expressions and converts
  /// them to a more usable form.  All SCEV expressions during the analysis
  /// should be re-written (and therefore simplified) according to PSE.
  /// A user of LoopAccessAnalysis will need to emit the runtime checks
  /// associated with this predicate.
  const PredicatedScalarEvolution &getPSE() const { return *PSE; }

private:
  /// Analyze the loop.
  void analyzeLoop(AAResults *AA, LoopInfo *LI,
                   const TargetLibraryInfo *TLI, DominatorTree *DT);

  /// Check if the structure of the loop allows it to be analyzed by this
  /// pass.
  bool canAnalyzeLoop();

  /// Save the analysis remark.
  ///
  /// LAA does not directly emits the remarks.  Instead it stores it which the
  /// client can retrieve and presents as its own analysis
  /// (e.g. -Rpass-analysis=loop-vectorize).
  OptimizationRemarkAnalysis &recordAnalysis(StringRef RemarkName,
                                             Instruction *Instr = nullptr);

  /// Collect memory access with loop invariant strides.
  ///
  /// Looks for accesses like "a[i * StrideA]" where "StrideA" is loop
  /// invariant.
  void collectStridedAccess(Value *LoadOrStoreInst);

  // Emits the first unsafe memory dependence in a loop.
  // Emits nothing if there are no unsafe dependences
  // or if the dependences were not recorded.
  void emitUnsafeDependenceRemark();

  std::unique_ptr<PredicatedScalarEvolution> PSE;

  /// We need to check that all of the pointers in this list are disjoint
  /// at runtime. Using std::unique_ptr to make using move ctor simpler.
  std::unique_ptr<RuntimePointerChecking> PtrRtChecking;

  /// the Memory Dependence Checker which can determine the
  /// loop-independent and loop-carried dependences between memory accesses.
  std::unique_ptr<MemoryDepChecker> DepChecker;

  Loop *TheLoop;

  unsigned NumLoads = 0;
  unsigned NumStores = 0;

  uint64_t MaxSafeDepDistBytes = -1;

  /// Cache the result of analyzeLoop.
  bool CanVecMem = false;
  bool HasConvergentOp = false;

  /// Indicator that there are non vectorizable stores to a uniform address.
  bool HasDependenceInvolvingLoopInvariantAddress = false;

  /// List of stores to invariant addresses.
  SmallVector<StoreInst *> StoresToInvariantAddresses;

  /// The diagnostics report generated for the analysis.  E.g. why we
  /// couldn't analyze the loop.
  std::unique_ptr<OptimizationRemarkAnalysis> Report;

  /// If an access has a symbolic strides, this maps the pointer value to
  /// the stride symbol.
  ValueToValueMap SymbolicStrides;

  /// Set of symbolic strides values.
  SmallPtrSet<Value *, 8> StrideSet;
};

Value *stripIntegerCast(Value *V);

/// Return the SCEV corresponding to a pointer with the symbolic stride
/// replaced with constant one, assuming the SCEV predicate associated with
/// \p PSE is true.
///
/// If necessary this method will version the stride of the pointer according
/// to \p PtrToStride and therefore add further predicates to \p PSE.
///
/// \p PtrToStride provides the mapping between the pointer value and its
/// stride as collected by LoopVectorizationLegality::collectStridedAccess.
const SCEV *replaceSymbolicStrideSCEV(PredicatedScalarEvolution &PSE,
                                      const ValueToValueMap &PtrToStride,
                                      Value *Ptr);

/// If the pointer has a constant stride return it in units of the access type
/// size.  Otherwise return std::nullopt.
///
/// Ensure that it does not wrap in the address space, assuming the predicate
/// associated with \p PSE is true.
///
/// If necessary this method will version the stride of the pointer according
/// to \p PtrToStride and therefore add further predicates to \p PSE.
/// The \p Assume parameter indicates if we are allowed to make additional
/// run-time assumptions.
std::optional<int64_t>
getPtrStride(PredicatedScalarEvolution &PSE, Type *AccessTy, Value *Ptr,
             const Loop *Lp,
             const ValueToValueMap &StridesMap = ValueToValueMap(),
             bool Assume = false, bool ShouldCheckWrap = true);

/// Returns the distance between the pointers \p PtrA and \p PtrB iff they are
/// compatible and it is possible to calculate the distance between them. This
/// is a simple API that does not depend on the analysis pass.
/// \param StrictCheck Ensure that the calculated distance matches the
/// type-based one after all the bitcasts removal in the provided pointers.
std::optional<int> getPointersDiff(Type *ElemTyA, Value *PtrA, Type *ElemTyB,
                                   Value *PtrB, const DataLayout &DL,
                                   ScalarEvolution &SE,
                                   bool StrictCheck = false,
                                   bool CheckType = true);

/// Attempt to sort the pointers in \p VL and return the sorted indices
/// in \p SortedIndices, if reordering is required.
///
/// Returns 'true' if sorting is legal, otherwise returns 'false'.
///
/// For example, for a given \p VL of memory accesses in program order, a[i+4],
/// a[i+0], a[i+1] and a[i+7], this function will sort the \p VL and save the
/// sorted indices in \p SortedIndices as a[i+0], a[i+1], a[i+4], a[i+7] and
/// saves the mask for actual memory accesses in program order in
/// \p SortedIndices as <1,2,0,3>
bool sortPtrAccesses(ArrayRef<Value *> VL, Type *ElemTy, const DataLayout &DL,
                     ScalarEvolution &SE,
                     SmallVectorImpl<unsigned> &SortedIndices);

/// Returns true if the memory operations \p A and \p B are consecutive.
/// This is a simple API that does not depend on the analysis pass.
bool isConsecutiveAccess(Value *A, Value *B, const DataLayout &DL,
                         ScalarEvolution &SE, bool CheckType = true);

class LoopAccessInfoManager {
  /// The cache.
  DenseMap<Loop *, std::unique_ptr<LoopAccessInfo>> LoopAccessInfoMap;

  // The used analysis passes.
  ScalarEvolution &SE;
  AAResults &AA;
  DominatorTree &DT;
  LoopInfo &LI;
  const TargetLibraryInfo *TLI = nullptr;

public:
  LoopAccessInfoManager(ScalarEvolution &SE, AAResults &AA, DominatorTree &DT,
                        LoopInfo &LI, const TargetLibraryInfo *TLI)
      : SE(SE), AA(AA), DT(DT), LI(LI), TLI(TLI) {}

  const LoopAccessInfo &getInfo(Loop &L);

  void clear() { LoopAccessInfoMap.clear(); }
};

/// This analysis provides dependence information for the memory accesses
/// of a loop.
///
/// It runs the analysis for a loop on demand.  This can be initiated by
/// querying the loop access info via LAA::getInfo.  getInfo return a
/// LoopAccessInfo object.  See this class for the specifics of what information
/// is provided.
class LoopAccessLegacyAnalysis : public FunctionPass {
public:
  static char ID;

  LoopAccessLegacyAnalysis();

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Return the proxy object for retrieving LoopAccessInfo for individual
  /// loops.
  ///
  /// If there is no cached result available run the analysis.
  LoopAccessInfoManager &getLAIs() { return *LAIs; }

  void releaseMemory() override {
    // Invalidate the cache when the pass is freed.
    LAIs->clear();
  }

private:
  std::unique_ptr<LoopAccessInfoManager> LAIs;
};

/// This analysis provides dependence information for the memory
/// accesses of a loop.
///
/// It runs the analysis for a loop on demand.  This can be initiated by
/// querying the loop access info via AM.getResult<LoopAccessAnalysis>.
/// getResult return a LoopAccessInfo object.  See this class for the
/// specifics of what information is provided.
class LoopAccessAnalysis
    : public AnalysisInfoMixin<LoopAccessAnalysis> {
  friend AnalysisInfoMixin<LoopAccessAnalysis>;
  static AnalysisKey Key;

public:
  typedef LoopAccessInfoManager Result;

  Result run(Function &F, FunctionAnalysisManager &AM);
};

inline Instruction *MemoryDepChecker::Dependence::getSource(
    const LoopAccessInfo &LAI) const {
  return LAI.getDepChecker().getMemoryInstructions()[Source];
}

inline Instruction *MemoryDepChecker::Dependence::getDestination(
    const LoopAccessInfo &LAI) const {
  return LAI.getDepChecker().getMemoryInstructions()[Destination];
}

} // End llvm namespace

#endif
