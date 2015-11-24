//===--- ScopDetection.h - Detect Scops -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect the maximal Scops of a function.
//
// A static control part (Scop) is a subgraph of the control flow graph (CFG)
// that only has statically known control flow and can therefore be described
// within the polyhedral model.
//
// Every Scop fullfills these restrictions:
//
// * It is a single entry single exit region
//
// * Only affine linear bounds in the loops
//
// Every natural loop in a Scop must have a number of loop iterations that can
// be described as an affine linear function in surrounding loop iterators or
// parameters. (A parameter is a scalar that does not change its value during
// execution of the Scop).
//
// * Only comparisons of affine linear expressions in conditions
//
// * All loops and conditions perfectly nested
//
// The control flow needs to be structured such that it could be written using
// just 'for' and 'if' statements, without the need for any 'goto', 'break' or
// 'continue'.
//
// * Side effect free functions call
//
// Only function calls and intrinsics that do not have side effects are allowed
// (readnone).
//
// The Scop detection finds the largest Scops by checking if the largest
// region is a Scop. If this is not the case, its canonical subregions are
// checked until a region is a Scop. It is now tried to extend this Scop by
// creating a larger non canonical region.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOP_DETECTION_H
#define POLLY_SCOP_DETECTION_H

#include "polly/ScopDetectionDiagnostic.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Pass.h"
#include <map>
#include <memory>
#include <set>

using namespace llvm;

namespace llvm {
class RegionInfo;
class Region;
class LoopInfo;
class Loop;
class ScalarEvolution;
class SCEV;
class SCEVAddRecExpr;
class SCEVUnknown;
class CallInst;
class Instruction;
class Value;
}

namespace polly {
typedef std::set<const SCEV *> ParamSetType;

// Description of the shape of an array.
struct ArrayShape {
  // Base pointer identifying all accesses to this array.
  const SCEVUnknown *BasePointer;

  // Sizes of each delinearized dimension.
  SmallVector<const SCEV *, 4> DelinearizedSizes;

  ArrayShape(const SCEVUnknown *B) : BasePointer(B), DelinearizedSizes() {}
};

struct MemAcc {
  const Instruction *Insn;

  // A pointer to the shape description of the array.
  std::shared_ptr<ArrayShape> Shape;

  // Subscripts computed by delinearization.
  SmallVector<const SCEV *, 4> DelinearizedSubscripts;

  MemAcc(const Instruction *I, std::shared_ptr<ArrayShape> S)
      : Insn(I), Shape(S), DelinearizedSubscripts() {}
};

typedef std::map<const Instruction *, MemAcc> MapInsnToMemAcc;
typedef std::pair<const Instruction *, const SCEV *> PairInstSCEV;
typedef std::vector<PairInstSCEV> AFs;
typedef std::map<const SCEVUnknown *, AFs> BaseToAFs;
typedef std::map<const SCEVUnknown *, const SCEV *> BaseToElSize;

extern bool PollyTrackFailures;
extern bool PollyDelinearize;
extern bool PollyUseRuntimeAliasChecks;
extern bool PollyProcessUnprofitable;

/// @brief A function attribute which will cause Polly to skip the function
extern llvm::StringRef PollySkipFnAttr;

//===----------------------------------------------------------------------===//
/// @brief Pass to detect the maximal static control parts (Scops) of a
/// function.
class ScopDetection : public FunctionPass {
public:
  typedef SetVector<const Region *> RegionSet;

  // Remember the valid regions
  RegionSet ValidRegions;

  /// @brief Set of loops (used to remember loops in non-affine subregions).
  using BoxedLoopsSetTy = SetVector<const Loop *>;

  /// @brief Set to remember non-affine branches in regions.
  using NonAffineSubRegionSetTy = RegionSet;

  /// @brief Context variables for SCoP detection.
  struct DetectionContext {
    Region &CurRegion;   // The region to check.
    AliasSetTracker AST; // The AliasSetTracker to hold the alias information.
    bool Verifying;      // If we are in the verification phase?
    RejectLog Log;

    /// @brief Map a base pointer to all access functions accessing it.
    ///
    /// This map is indexed by the base pointer. Each element of the map
    /// is a list of memory accesses that reference this base pointer.
    BaseToAFs Accesses;

    /// @brief The set of base pointers with non-affine accesses.
    ///
    /// This set contains all base pointers which are used in memory accesses
    /// that can not be detected as affine accesses.
    SetVector<const SCEVUnknown *> NonAffineAccesses;
    BaseToElSize ElementSize;

    /// @brief The region has at least one load instruction.
    bool hasLoads;

    /// @brief The region has at least one store instruction.
    bool hasStores;

    /// @brief The set of non-affine subregions in the region we analyze.
    NonAffineSubRegionSetTy NonAffineSubRegionSet;

    /// @brief The set of loops contained in non-affine regions.
    BoxedLoopsSetTy BoxedLoopsSet;

    /// @brief Loads that need to be invariant during execution.
    InvariantLoadsSetTy RequiredILS;

    /// @brief Initialize a DetectionContext from scratch.
    DetectionContext(Region &R, AliasAnalysis &AA, bool Verify)
        : CurRegion(R), AST(AA), Verifying(Verify), Log(&R), hasLoads(false),
          hasStores(false) {}

    /// @brief Initialize a DetectionContext with the data from @p DC.
    DetectionContext(const DetectionContext &&DC)
        : CurRegion(DC.CurRegion), AST(DC.AST.getAliasAnalysis()),
          Verifying(DC.Verifying), Log(std::move(DC.Log)),
          Accesses(std::move(DC.Accesses)),
          NonAffineAccesses(std::move(DC.NonAffineAccesses)),
          ElementSize(std::move(DC.ElementSize)), hasLoads(DC.hasLoads),
          hasStores(DC.hasStores),
          NonAffineSubRegionSet(std::move(DC.NonAffineSubRegionSet)),
          BoxedLoopsSet(std::move(DC.BoxedLoopsSet)),
          RequiredILS(std::move(DC.RequiredILS)) {
      AST.add(DC.AST);
    }
  };

private:
  //===--------------------------------------------------------------------===//
  ScopDetection(const ScopDetection &) = delete;
  const ScopDetection &operator=(const ScopDetection &) = delete;

  /// @brief Analysis passes used.
  //@{
  const DominatorTree *DT;
  ScalarEvolution *SE;
  LoopInfo *LI;
  RegionInfo *RI;
  AliasAnalysis *AA;
  //@}

  /// @brief Map to remember detection contexts for valid regions.
  using DetectionContextMapTy = DenseMap<const Region *, DetectionContext>;
  mutable DetectionContextMapTy DetectionContextMap;

  // Remember a list of errors for every region.
  mutable RejectLogsContainer RejectLogs;

  /// @brief Remove cached results for @p R.
  void removeCachedResults(const Region &R);

  /// @brief Remove cached results for the children of @p R recursively.
  ///
  /// @returns The number of regions erased regions.
  unsigned removeCachedResultsRecursively(const Region &R);

  /// @brief Add the region @p AR as over approximated sub-region in @p Context.
  ///
  /// @param AR      The non-affine subregion.
  /// @param Context The current detection context.
  ///
  /// @returns True if the subregion can be over approximated, false otherwise.
  bool addOverApproximatedRegion(Region *AR, DetectionContext &Context) const;

  /// @brief Find for a given base pointer terms that hint towards dimension
  ///        sizes of a multi-dimensional array.
  ///
  /// @param Context      The current detection context.
  /// @param BasePointer  A base pointer indicating the virtual array we are
  ///                     interested in.
  SmallVector<const SCEV *, 4>
  getDelinearizationTerms(DetectionContext &Context,
                          const SCEVUnknown *BasePointer) const;

  /// @brief Check if the dimension size of a delinearized array is valid.
  ///
  /// @param Context     The current detection context.
  /// @param Sizes       The sizes of the different array dimensions.
  /// @param BasePointer The base pointer we are interested in.
  /// @returns True if one or more array sizes could be derived - meaning: we
  ///          see this array as multi-dimensional.
  bool hasValidArraySizes(DetectionContext &Context,
                          SmallVectorImpl<const SCEV *> &Sizes,
                          const SCEVUnknown *BasePointer) const;

  /// @brief Derive access functions for a given base pointer.
  ///
  /// @param Context     The current detection context.
  /// @param Sizes       The sizes of the different array dimensions.
  /// @param BasePointer The base pointer of all the array for which to compute
  ///                    access functions.
  /// @param Shape       The shape that describes the derived array sizes and
  ///                    which should be filled with newly computed access
  ///                    functions.
  /// @returns True if a set of affine access functions could be derived.
  bool computeAccessFunctions(DetectionContext &Context,
                              const SCEVUnknown *BasePointer,
                              std::shared_ptr<ArrayShape> Shape) const;

  /// @brief Check if all accesses to a given BasePointer are affine.
  ///
  /// @param Context     The current detection context.
  /// @param basepointer the base pointer we are interested in.
  /// @param True if consistent (multi-dimensional) array accesses could be
  ///        derived for this array.
  bool hasBaseAffineAccesses(DetectionContext &Context,
                             const SCEVUnknown *BasePointer) const;

  // Delinearize all non affine memory accesses and return false when there
  // exists a non affine memory access that cannot be delinearized. Return true
  // when all array accesses are affine after delinearization.
  bool hasAffineMemoryAccesses(DetectionContext &Context) const;

  // Try to expand the region R. If R can be expanded return the expanded
  // region, NULL otherwise.
  Region *expandRegion(Region &R);

  /// Find the Scops in this region tree.
  ///
  /// @param The region tree to scan for scops.
  void findScops(Region &R);

  /// @brief Check if all basic block in the region are valid.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if all blocks in R are valid, false otherwise.
  bool allBlocksValid(DetectionContext &Context) const;

  /// @brief Check if a region is a Scop.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if R is a Scop, false otherwise.
  bool isValidRegion(DetectionContext &Context) const;

  /// @brief Check if a call instruction can be part of a Scop.
  ///
  /// @param CI The call instruction to check.
  /// @return True if the call instruction is valid, false otherwise.
  static bool isValidCallInst(CallInst &CI);

  /// @brief Check if the given loads could be invariant and can be hoisted.
  ///
  /// If true is returned the loads are added to the required invariant loads
  /// contained in the @p Context.
  ///
  /// @param RequiredILS The loads to check.
  /// @param Context     The current detection context.
  ///
  /// @return True if all loads can be assumed invariant.
  bool onlyValidRequiredInvariantLoads(InvariantLoadsSetTy &RequiredILS,
                                       DetectionContext &Context) const;

  /// @brief Check if a value is invariant in the region Reg.
  ///
  /// @param Val Value to check for invariance.
  /// @param Reg The region to consider for the invariance of Val.
  ///
  /// @return True if the value represented by Val is invariant in the region
  ///         identified by Reg.
  bool isInvariant(const Value &Val, const Region &Reg) const;

  /// @brief Check if a memory access can be part of a Scop.
  ///
  /// @param Inst The instruction accessing the memory.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the memory access is valid, false otherwise.
  bool isValidMemoryAccess(Instruction &Inst, DetectionContext &Context) const;

  /// @brief Check if an instruction has any non trivial scalar dependencies
  ///        as part of a Scop.
  ///
  /// @param Inst The instruction to check.
  /// @param RefRegion The region in respect to which we check the access
  ///                  function.
  ///
  /// @return True if the instruction has scalar dependences, false otherwise.
  bool hasScalarDependency(Instruction &Inst, Region &RefRegion) const;

  /// @brief Check if an instruction can be part of a Scop.
  ///
  /// @param Inst The instruction to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the instruction is valid, false otherwise.
  bool isValidInstruction(Instruction &Inst, DetectionContext &Context) const;

  /// @brief Check if the switch @p SI with condition @p Condition is valid.
  ///
  /// @param BB           The block to check.
  /// @param SI           The switch to check.
  /// @param Condition    The switch condition.
  /// @param IsLoopBranch Flag to indicate the branch is a loop exit/latch.
  /// @param Context      The context of scop detection.
  ///
  /// @return True if the branch @p BI is valid.
  bool isValidSwitch(BasicBlock &BB, SwitchInst *SI, Value *Condition,
                     bool IsLoopBranch, DetectionContext &Context) const;

  /// @brief Check if the branch @p BI with condition @p Condition is valid.
  ///
  /// @param BB           The block to check.
  /// @param BI           The branch to check.
  /// @param Condition    The branch condition.
  /// @param IsLoopBranch Flag to indicate the branch is a loop exit/latch.
  /// @param Context      The context of scop detection.
  ///
  /// @return True if the branch @p BI is valid.
  bool isValidBranch(BasicBlock &BB, BranchInst *BI, Value *Condition,
                     bool IsLoopBranch, DetectionContext &Context) const;

  /// @brief Check if the SCEV @p S is affine in the current @p Context.
  ///
  /// This will also use a heuristic to decide if we want to require loads to be
  /// invariant to make the expression affine or if we want to treat is as
  /// non-affine.
  ///
  /// @param S           The expression to be checked.
  /// @param Context     The context of scop detection.
  /// @param BaseAddress The base address of the expression @p S (if any).
  bool isAffine(const SCEV *S, DetectionContext &Context,
                Value *BaseAddress = nullptr) const;

  /// @brief Check if the control flow in a basic block is valid.
  ///
  /// This function checks if a certain basic block is terminated by a
  /// Terminator instruction we can handle or, if this is not the case,
  /// registers this basic block as the start of a non-affine region.
  ///
  /// This function optionally allows unreachable statements.
  ///
  /// @param BB               The BB to check the control flow.
  /// @param IsLoopBranch     Flag to indicate the branch is a loop exit/latch.
  //  @param AllowUnreachable Allow unreachable statements.
  /// @param Context          The context of scop detection.
  ///
  /// @return True if the BB contains only valid control flow.
  bool isValidCFG(BasicBlock &BB, bool IsLoopBranch, bool AllowUnreachable,
                  DetectionContext &Context) const;

  /// @brief Is a loop valid with respect to a given region.
  ///
  /// @param L The loop to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the loop is valid in the region.
  bool isValidLoop(Loop *L, DetectionContext &Context) const;

  /// @brief Count the number of beneficial loops in @p R.
  ///
  /// @param R The region to check
  int countBeneficialLoops(Region *R) const;

  /// @brief Check if the function @p F is marked as invalid.
  ///
  /// @note An OpenMP subfunction will be marked as invalid.
  bool isValidFunction(llvm::Function &F);

  /// @brief Can ISL compute the trip count of a loop.
  ///
  /// @param L The loop to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if ISL can compute the trip count of the loop.
  bool canUseISLTripCount(Loop *L, DetectionContext &Context) const;

  /// @brief Print the locations of all detected scops.
  void printLocations(llvm::Function &F);

  /// @brief Track diagnostics for invalid scops.
  ///
  /// @param Context The context of scop detection.
  /// @param Assert Throw an assert in verify mode or not.
  /// @param Args Argument list that gets passed to the constructor of RR.
  template <class RR, typename... Args>
  inline bool invalid(DetectionContext &Context, bool Assert,
                      Args &&... Arguments) const;

public:
  static char ID;
  explicit ScopDetection();

  /// @brief Get the RegionInfo stored in this pass.
  ///
  /// This was added to give the DOT printer easy access to this information.
  RegionInfo *getRI() const { return RI; }

  /// @brief Is the region is the maximum region of a Scop?
  ///
  /// @param R The Region to test if it is maximum.
  /// @param Verify Rerun the scop detection to verify SCoP was not invalidated
  ///               meanwhile.
  ///
  /// @return Return true if R is the maximum Region in a Scop, false otherwise.
  bool isMaxRegionInScop(const Region &R, bool Verify = true) const;

  /// @brief Return the detection context for @p R, nullptr if @p R was invalid.
  const DetectionContext *getDetectionContext(const Region *R) const;

  /// @brief Return the set of loops in non-affine subregions for @p R.
  const BoxedLoopsSetTy *getBoxedLoops(const Region *R) const;

  /// @brief Return the set of required invariant loads for @p R.
  const InvariantLoadsSetTy *getRequiredInvariantLoads(const Region *R) const;

  /// @brief Return true if @p SubR is a non-affine subregion in @p ScopR.
  bool isNonAffineSubRegion(const Region *SubR, const Region *ScopR) const;

  /// @brief Get a message why a region is invalid
  ///
  /// @param R The region for which we get the error message
  ///
  /// @return The error or "" if no error appeared.
  std::string regionIsInvalidBecause(const Region *R) const;

  /// @name Maximum Region In Scops Iterators
  ///
  /// These iterators iterator over all maximum region in Scops of this
  /// function.
  //@{
  typedef RegionSet::iterator iterator;
  typedef RegionSet::const_iterator const_iterator;

  iterator begin() { return ValidRegions.begin(); }
  iterator end() { return ValidRegions.end(); }

  const_iterator begin() const { return ValidRegions.begin(); }
  const_iterator end() const { return ValidRegions.end(); }
  //@}

  /// @name Reject log iterators
  ///
  /// These iterators iterate over the logs of all rejected regions of this
  //  function.
  //@{
  typedef std::map<const Region *, RejectLog>::iterator reject_iterator;
  typedef std::map<const Region *, RejectLog>::const_iterator
      const_reject_iterator;

  reject_iterator reject_begin() { return RejectLogs.begin(); }
  reject_iterator reject_end() { return RejectLogs.end(); }

  const_reject_iterator reject_begin() const { return RejectLogs.begin(); }
  const_reject_iterator reject_end() const { return RejectLogs.end(); }
  //@}

  /// @brief Emit rejection remarks for all smallest invalid regions.
  ///
  /// @param F The function to emit remarks for.
  /// @param R The region to start the region tree traversal for.
  void emitMissedRemarksForLeaves(const Function &F, const Region *R);

  /// @brief Emit rejection remarks for the parent regions of all valid regions.
  ///
  /// Emitting rejection remarks for the parent regions of all valid regions
  /// may give the end-user clues about how to increase the size of the
  /// detected Scops.
  ///
  /// @param F The function to emit remarks for.
  void emitMissedRemarksForValidRegions(const Function &F);

  /// @brief Mark the function as invalid so we will not extract any scop from
  ///        the function.
  ///
  /// @param F The function to mark as invalid.
  void markFunctionAsInvalid(Function *F) const;

  /// @brief Verify if all valid Regions in this Function are still valid
  /// after some transformations.
  void verifyAnalysis() const;

  /// @brief Verify if R is still a valid part of Scop after some
  /// transformations.
  ///
  /// @param R The Region to verify.
  void verifyRegion(const Region &R) const;

  /// @name FunctionPass interface
  //@{
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory();
  virtual bool runOnFunction(Function &F);
  virtual void print(raw_ostream &OS, const Module *) const;
  //@}
};

} // end namespace polly

namespace llvm {
class PassRegistry;
void initializeScopDetectionPass(llvm::PassRegistry &);
}

#endif
