//===- llvm/Analysis/LoopAccessAnalysis.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Value;
class DataLayout;
class ScalarEvolution;
class Loop;
class SCEV;
class SCEVUnionPredicate;
class LoopAccessInfo;
class OptimizationRemarkEmitter;

/// \brief Collection of parameters shared beetween the Loop Vectorizer and the
/// Loop Access Analysis.
struct VectorizerParams {
  /// \brief Maximum SIMD width.
  static const unsigned MaxVectorWidth;

  /// \brief VF as overridden by the user.
  static unsigned VectorizationFactor;
  /// \brief Interleave factor as overridden by the user.
  static unsigned VectorizationInterleave;
  /// \brief True if force-vector-interleave was specified by the user.
  static bool isInterleaveForced();

  /// \\brief When performing memory disambiguation checks at runtime do not
  /// make more than this number of comparisons.
  static unsigned RuntimeMemoryCheckThreshold;
};

/// \brief Checks memory dependences among accesses to the same underlying
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
  /// \brief Set of potential dependent memory accesses.
  typedef EquivalenceClasses<MemAccessInfo> DepCandidates;

  /// \brief Dependece between memory access instructions.
  struct Dependence {
    /// \brief The type of the dependence.
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

    /// \brief String version of the types.
    static const char *DepName[];

    /// \brief Index of the source of the dependence in the InstMap vector.
    unsigned Source;
    /// \brief Index of the destination of the dependence in the InstMap vector.
    unsigned Destination;
    /// \brief The type of the dependence.
    DepType Type;

    Dependence(unsigned Source, unsigned Destination, DepType Type)
        : Source(Source), Destination(Destination), Type(Type) {}

    /// \brief Return the source instruction of the dependence.
    Instruction *getSource(const LoopAccessInfo &LAI) const;
    /// \brief Return the destination instruction of the dependence.
    Instruction *getDestination(const LoopAccessInfo &LAI) const;

    /// \brief Dependence types that don't prevent vectorization.
    static bool isSafeForVectorization(DepType Type);

    /// \brief Lexically forward dependence.
    bool isForward() const;
    /// \brief Lexically backward dependence.
    bool isBackward() const;

    /// \brief May be a lexically backward dependence type (includes Unknown).
    bool isPossiblyBackward() const;

    /// \brief Print the dependence.  \p Instr is used to map the instruction
    /// indices to instructions.
    void print(raw_ostream &OS, unsigned Depth,
               const SmallVectorImpl<Instruction *> &Instrs) const;
  };

  MemoryDepChecker(PredicatedScalarEvolution &PSE, const Loop *L)
      : PSE(PSE), InnermostLoop(L), AccessIdx(0),
        ShouldRetryWithRuntimeCheck(false), SafeForVectorization(true),
        RecordDependences(true) {}

  /// \brief Register the location (instructions are given increasing numbers)
  /// of a write access.
  void addAccess(StoreInst *SI) {
    Value *Ptr = SI->getPointerOperand();
    Accesses[MemAccessInfo(Ptr, true)].push_back(AccessIdx);
    InstMap.push_back(SI);
    ++AccessIdx;
  }

  /// \brief Register the location (instructions are given increasing numbers)
  /// of a write access.
  void addAccess(LoadInst *LI) {
    Value *Ptr = LI->getPointerOperand();
    Accesses[MemAccessInfo(Ptr, false)].push_back(AccessIdx);
    InstMap.push_back(LI);
    ++AccessIdx;
  }

  /// \brief Check whether the dependencies between the accesses are safe.
  ///
  /// Only checks sets with elements in \p CheckDeps.
  bool areDepsSafe(DepCandidates &AccessSets, MemAccessInfoList &CheckDeps,
                   const ValueToValueMap &Strides);

  /// \brief No memory dependence was encountered that would inhibit
  /// vectorization.
  bool isSafeForVectorization() const { return SafeForVectorization; }

  /// \brief The maximum number of bytes of a vector register we can vectorize
  /// the accesses safely with.
  uint64_t getMaxSafeDepDistBytes() { return MaxSafeDepDistBytes; }

  /// \brief In same cases when the dependency check fails we can still
  /// vectorize the loop with a dynamic array access check.
  bool shouldRetryWithRuntimeCheck() { return ShouldRetryWithRuntimeCheck; }

  /// \brief Returns the memory dependences.  If null is returned we exceeded
  /// the MaxDependences threshold and this information is not
  /// available.
  const SmallVectorImpl<Dependence> *getDependences() const {
    return RecordDependences ? &Dependences : nullptr;
  }

  void clearDependences() { Dependences.clear(); }

  /// \brief The vector of memory access instructions.  The indices are used as
  /// instruction identifiers in the Dependence class.
  const SmallVectorImpl<Instruction *> &getMemoryInstructions() const {
    return InstMap;
  }

  /// \brief Generate a mapping between the memory instructions and their
  /// indices according to program order.
  DenseMap<Instruction *, unsigned> generateInstructionOrderMap() const {
    DenseMap<Instruction *, unsigned> OrderMap;

    for (unsigned I = 0; I < InstMap.size(); ++I)
      OrderMap[InstMap[I]] = I;

    return OrderMap;
  }

  /// \brief Find the set of instructions that read or write via \p Ptr.
  SmallVector<Instruction *, 4> getInstructionsForAccess(Value *Ptr,
                                                         bool isWrite) const;

private:
  /// A wrapper around ScalarEvolution, used to add runtime SCEV checks, and
  /// applies dynamic knowledge to simplify SCEV expressions and convert them
  /// to a more usable form. We need this in case assumptions about SCEV
  /// expressions need to be made in order to avoid unknown dependences. For
  /// example we might assume a unit stride for a pointer in order to prove
  /// that a memory access is strided and doesn't wrap.
  PredicatedScalarEvolution &PSE;
  const Loop *InnermostLoop;

  /// \brief Maps access locations (ptr, read/write) to program order.
  DenseMap<MemAccessInfo, std::vector<unsigned> > Accesses;

  /// \brief Memory access instructions in program order.
  SmallVector<Instruction *, 16> InstMap;

  /// \brief The program order index to be used for the next instruction.
  unsigned AccessIdx;

  // We can access this many bytes in parallel safely.
  uint64_t MaxSafeDepDistBytes;

  /// \brief If we see a non-constant dependence distance we can still try to
  /// vectorize this loop with runtime checks.
  bool ShouldRetryWithRuntimeCheck;

  /// \brief No memory dependence was encountered that would inhibit
  /// vectorization.
  bool SafeForVectorization;

  //// \brief True if Dependences reflects the dependences in the
  //// loop.  If false we exceeded MaxDependences and
  //// Dependences is invalid.
  bool RecordDependences;

  /// \brief Memory dependences collected during the analysis.  Only valid if
  /// RecordDependences is true.
  SmallVector<Dependence, 8> Dependences;

  /// \brief Check whether there is a plausible dependence between the two
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

  /// \brief Check whether the data dependence could prevent store-load
  /// forwarding.
  ///
  /// \return false if we shouldn't vectorize at all or avoid larger
  /// vectorization factors by limiting MaxSafeDepDistBytes.
  bool couldPreventStoreLoadForward(uint64_t Distance, uint64_t TypeByteSize);
};

/// \brief Holds information about the memory runtime legality checks to verify
/// that a group of pointers do not overlap.
class RuntimePointerChecking {
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

    PointerInfo(Value *PointerValue, const SCEV *Start, const SCEV *End,
                bool IsWritePtr, unsigned DependencySetId, unsigned AliasSetId,
                const SCEV *Expr)
        : PointerValue(PointerValue), Start(Start), End(End),
          IsWritePtr(IsWritePtr), DependencySetId(DependencySetId),
          AliasSetId(AliasSetId), Expr(Expr) {}
  };

  RuntimePointerChecking(ScalarEvolution *SE) : Need(false), SE(SE) {}

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
  void insert(Loop *Lp, Value *Ptr, bool WritePtr, unsigned DepSetId,
              unsigned ASId, const ValueToValueMap &Strides,
              PredicatedScalarEvolution &PSE);

  /// \brief No run-time memory checking is necessary.
  bool empty() const { return Pointers.empty(); }

  /// A grouping of pointers. A single memcheck is required between
  /// two groups.
  struct CheckingPtrGroup {
    /// \brief Create a new pointer checking group containing a single
    /// pointer, with index \p Index in RtCheck.
    CheckingPtrGroup(unsigned Index, RuntimePointerChecking &RtCheck)
        : RtCheck(RtCheck), High(RtCheck.Pointers[Index].End),
          Low(RtCheck.Pointers[Index].Start) {
      Members.push_back(Index);
    }

    /// \brief Tries to add the pointer recorded in RtCheck at index
    /// \p Index to this pointer checking group. We can only add a pointer
    /// to a checking group if we will still be able to get
    /// the upper and lower bounds of the check. Returns true in case
    /// of success, false otherwise.
    bool addPointer(unsigned Index);

    /// Constitutes the context of this pointer checking group. For each
    /// pointer that is a member of this group we will retain the index
    /// at which it appears in RtCheck.
    RuntimePointerChecking &RtCheck;
    /// The SCEV expression which represents the upper bound of all the
    /// pointers in this group.
    const SCEV *High;
    /// The SCEV expression which represents the lower bound of all the
    /// pointers in this group.
    const SCEV *Low;
    /// Indices of all the pointers that constitute this grouping.
    SmallVector<unsigned, 2> Members;
  };

  /// \brief A memcheck which made up of a pair of grouped pointers.
  ///
  /// These *have* to be const for now, since checks are generated from
  /// CheckingPtrGroups in LAI::addRuntimeChecks which is a const member
  /// function.  FIXME: once check-generation is moved inside this class (after
  /// the PtrPartition hack is removed), we could drop const.
  typedef std::pair<const CheckingPtrGroup *, const CheckingPtrGroup *>
      PointerCheck;

  /// \brief Generate the checks and store it.  This also performs the grouping
  /// of pointers to reduce the number of memchecks necessary.
  void generateChecks(MemoryDepChecker::DepCandidates &DepCands,
                      bool UseDependencies);

  /// \brief Returns the checks that generateChecks created.
  const SmallVector<PointerCheck, 4> &getChecks() const { return Checks; }

  /// \brief Decide if we need to add a check between two groups of pointers,
  /// according to needsChecking.
  bool needsChecking(const CheckingPtrGroup &M,
                     const CheckingPtrGroup &N) const;

  /// \brief Returns the number of run-time checks required according to
  /// needsChecking.
  unsigned getNumberOfChecks() const { return Checks.size(); }

  /// \brief Print the list run-time memory checks necessary.
  void print(raw_ostream &OS, unsigned Depth = 0) const;

  /// Print \p Checks.
  void printChecks(raw_ostream &OS, const SmallVectorImpl<PointerCheck> &Checks,
                   unsigned Depth = 0) const;

  /// This flag indicates if we need to add the runtime check.
  bool Need;

  /// Information about the pointers that may require checking.
  SmallVector<PointerInfo, 2> Pointers;

  /// Holds a partitioning of pointers into "check groups".
  SmallVector<CheckingPtrGroup, 2> CheckingGroups;

  /// \brief Check if pointers are in the same partition
  ///
  /// \p PtrToPartition contains the partition number for pointers (-1 if the
  /// pointer belongs to multiple partitions).
  static bool
  arePointersInSamePartition(const SmallVectorImpl<int> &PtrToPartition,
                             unsigned PtrIdx1, unsigned PtrIdx2);

  /// \brief Decide whether we need to issue a run-time check for pointer at
  /// index \p I and \p J to prove their independence.
  bool needsChecking(unsigned I, unsigned J) const;

  /// \brief Return PointerInfo for pointer at index \p PtrIdx.
  const PointerInfo &getPointerInfo(unsigned PtrIdx) const {
    return Pointers[PtrIdx];
  }

private:
  /// \brief Groups pointers such that a single memcheck is required
  /// between two different groups. This will clear the CheckingGroups vector
  /// and re-compute it. We will only group dependecies if \p UseDependencies
  /// is true, otherwise we will create a separate group for each pointer.
  void groupChecks(MemoryDepChecker::DepCandidates &DepCands,
                   bool UseDependencies);

  /// Generate the checks and return them.
  SmallVector<PointerCheck, 4>
  generateChecks() const;

  /// Holds a pointer to the ScalarEvolution analysis.
  ScalarEvolution *SE;

  /// \brief Set of run-time checks required to establish independence of
  /// otherwise may-aliasing pointers in the loop.
  SmallVector<PointerCheck, 4> Checks;
};

/// \brief Drive the analysis of memory accesses in the loop
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
                 AliasAnalysis *AA, DominatorTree *DT, LoopInfo *LI);

  /// Return true we can analyze the memory accesses in the loop and there are
  /// no memory dependence cycles.
  bool canVectorizeMemory() const { return CanVecMem; }

  const RuntimePointerChecking *getRuntimePointerChecking() const {
    return PtrRtChecking.get();
  }

  /// \brief Number of memchecks required to prove independence of otherwise
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

  /// \brief Add code that checks at runtime if the accessed arrays overlap.
  ///
  /// Returns a pair of instructions where the first element is the first
  /// instruction generated in possibly a sequence of instructions and the
  /// second value is the final comparator value or NULL if no check is needed.
  std::pair<Instruction *, Instruction *>
  addRuntimeChecks(Instruction *Loc) const;

  /// \brief Generete the instructions for the checks in \p PointerChecks.
  ///
  /// Returns a pair of instructions where the first element is the first
  /// instruction generated in possibly a sequence of instructions and the
  /// second value is the final comparator value or NULL if no check is needed.
  std::pair<Instruction *, Instruction *>
  addRuntimeChecks(Instruction *Loc,
                   const SmallVectorImpl<RuntimePointerChecking::PointerCheck>
                       &PointerChecks) const;

  /// \brief The diagnostics report generated for the analysis.  E.g. why we
  /// couldn't analyze the loop.
  const OptimizationRemarkAnalysis *getReport() const { return Report.get(); }

  /// \brief the Memory Dependence Checker which can determine the
  /// loop-independent and loop-carried dependences between memory accesses.
  const MemoryDepChecker &getDepChecker() const { return *DepChecker; }

  /// \brief Return the list of instructions that use \p Ptr to read or write
  /// memory.
  SmallVector<Instruction *, 4> getInstructionsForAccess(Value *Ptr,
                                                         bool isWrite) const {
    return DepChecker->getInstructionsForAccess(Ptr, isWrite);
  }

  /// \brief If an access has a symbolic strides, this maps the pointer value to
  /// the stride symbol.
  const ValueToValueMap &getSymbolicStrides() const { return SymbolicStrides; }

  /// \brief Pointer has a symbolic stride.
  bool hasStride(Value *V) const { return StrideSet.count(V); }

  /// \brief Print the information about the memory accesses in the loop.
  void print(raw_ostream &OS, unsigned Depth = 0) const;

  /// \brief Checks existence of store to invariant address inside loop.
  /// If the loop has any store to invariant address, then it returns true,
  /// else returns false.
  bool hasStoreToLoopInvariantAddress() const {
    return StoreToLoopInvariantAddress;
  }

  /// Used to add runtime SCEV checks. Simplifies SCEV expressions and converts
  /// them to a more usable form.  All SCEV expressions during the analysis
  /// should be re-written (and therefore simplified) according to PSE.
  /// A user of LoopAccessAnalysis will need to emit the runtime checks
  /// associated with this predicate.
  const PredicatedScalarEvolution &getPSE() const { return *PSE; }

private:
  /// \brief Analyze the loop.
  void analyzeLoop(AliasAnalysis *AA, LoopInfo *LI,
                   const TargetLibraryInfo *TLI, DominatorTree *DT);

  /// \brief Check if the structure of the loop allows it to be analyzed by this
  /// pass.
  bool canAnalyzeLoop();

  /// \brief Save the analysis remark.
  ///
  /// LAA does not directly emits the remarks.  Instead it stores it which the
  /// client can retrieve and presents as its own analysis
  /// (e.g. -Rpass-analysis=loop-vectorize).
  OptimizationRemarkAnalysis &recordAnalysis(StringRef RemarkName,
                                             Instruction *Instr = nullptr);

  /// \brief Collect memory access with loop invariant strides.
  ///
  /// Looks for accesses like "a[i * StrideA]" where "StrideA" is loop
  /// invariant.
  void collectStridedAccess(Value *LoadOrStoreInst);

  std::unique_ptr<PredicatedScalarEvolution> PSE;

  /// We need to check that all of the pointers in this list are disjoint
  /// at runtime. Using std::unique_ptr to make using move ctor simpler.
  std::unique_ptr<RuntimePointerChecking> PtrRtChecking;

  /// \brief the Memory Dependence Checker which can determine the
  /// loop-independent and loop-carried dependences between memory accesses.
  std::unique_ptr<MemoryDepChecker> DepChecker;

  Loop *TheLoop;

  unsigned NumLoads;
  unsigned NumStores;

  uint64_t MaxSafeDepDistBytes;

  /// \brief Cache the result of analyzeLoop.
  bool CanVecMem;

  /// \brief Indicator for storing to uniform addresses.
  /// If a loop has write to a loop invariant address then it should be true.
  bool StoreToLoopInvariantAddress;

  /// \brief The diagnostics report generated for the analysis.  E.g. why we
  /// couldn't analyze the loop.
  std::unique_ptr<OptimizationRemarkAnalysis> Report;

  /// \brief If an access has a symbolic strides, this maps the pointer value to
  /// the stride symbol.
  ValueToValueMap SymbolicStrides;

  /// \brief Set of symbolic strides values.
  SmallPtrSet<Value *, 8> StrideSet;
};

Value *stripIntegerCast(Value *V);

/// \brief Return the SCEV corresponding to a pointer with the symbolic stride
/// replaced with constant one, assuming the SCEV predicate associated with
/// \p PSE is true.
///
/// If necessary this method will version the stride of the pointer according
/// to \p PtrToStride and therefore add further predicates to \p PSE.
///
/// If \p OrigPtr is not null, use it to look up the stride value instead of \p
/// Ptr.  \p PtrToStride provides the mapping between the pointer value and its
/// stride as collected by LoopVectorizationLegality::collectStridedAccess.
const SCEV *replaceSymbolicStrideSCEV(PredicatedScalarEvolution &PSE,
                                      const ValueToValueMap &PtrToStride,
                                      Value *Ptr, Value *OrigPtr = nullptr);

/// \brief If the pointer has a constant stride return it in units of its
/// element size.  Otherwise return zero.
///
/// Ensure that it does not wrap in the address space, assuming the predicate
/// associated with \p PSE is true.
///
/// If necessary this method will version the stride of the pointer according
/// to \p PtrToStride and therefore add further predicates to \p PSE.
/// The \p Assume parameter indicates if we are allowed to make additional
/// run-time assumptions.
int64_t getPtrStride(PredicatedScalarEvolution &PSE, Value *Ptr, const Loop *Lp,
                     const ValueToValueMap &StridesMap = ValueToValueMap(),
                     bool Assume = false, bool ShouldCheckWrap = true);

/// \brief Try to sort an array of loads / stores.
///
/// An array of loads / stores can only be sorted if all pointer operands
/// refer to the same object, and the differences between these pointers 
/// are known to be constant. If that is the case, this returns true, and the
/// sorted array is returned in \p Sorted. Otherwise, this returns false, and
/// \p Sorted is invalid.
bool sortMemAccesses(ArrayRef<Value *> VL, const DataLayout &DL,
                     ScalarEvolution &SE, SmallVectorImpl<Value *> &Sorted);

/// \brief Returns true if the memory operations \p A and \p B are consecutive.
/// This is a simple API that does not depend on the analysis pass. 
bool isConsecutiveAccess(Value *A, Value *B, const DataLayout &DL,
                         ScalarEvolution &SE, bool CheckType = true);

/// \brief This analysis provides dependence information for the memory accesses
/// of a loop.
///
/// It runs the analysis for a loop on demand.  This can be initiated by
/// querying the loop access info via LAA::getInfo.  getInfo return a
/// LoopAccessInfo object.  See this class for the specifics of what information
/// is provided.
class LoopAccessLegacyAnalysis : public FunctionPass {
public:
  static char ID;

  LoopAccessLegacyAnalysis() : FunctionPass(ID) {
    initializeLoopAccessLegacyAnalysisPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// \brief Query the result of the loop access information for the loop \p L.
  ///
  /// If there is no cached result available run the analysis.
  const LoopAccessInfo &getInfo(Loop *L);

  void releaseMemory() override {
    // Invalidate the cache when the pass is freed.
    LoopAccessInfoMap.clear();
  }

  /// \brief Print the result of the analysis when invoked with -analyze.
  void print(raw_ostream &OS, const Module *M = nullptr) const override;

private:
  /// \brief The cache.
  DenseMap<Loop *, std::unique_ptr<LoopAccessInfo>> LoopAccessInfoMap;

  // The used analysis passes.
  ScalarEvolution *SE;
  const TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
  DominatorTree *DT;
  LoopInfo *LI;
};

/// \brief This analysis provides dependence information for the memory
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
  typedef LoopAccessInfo Result;

  Result run(Loop &L, LoopAnalysisManager &AM, LoopStandardAnalysisResults &AR);
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
