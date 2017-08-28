//===- ScopDetection.h - Detect Scops ---------------------------*- C++ -*-===//
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
// Every Scop fulfills these restrictions:
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

#ifndef POLLY_SCOPDETECTION_H
#define POLLY_SCOPDETECTION_H

#include "polly/ScopDetectionDiagnostic.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Pass.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace llvm {

class BasicBlock;
class BranchInst;
class CallInst;
class DebugLoc;
class DominatorTree;
class Function;
class Instruction;
class IntrinsicInst;
class Loop;
class LoopInfo;
class OptimizationRemarkEmitter;
class PassRegistry;
class raw_ostream;
class ScalarEvolution;
class SCEV;
class SCEVUnknown;
class SwitchInst;
class Value;

void initializeScopDetectionWrapperPassPass(PassRegistry &);

} // namespace llvm

namespace polly {

using ParamSetType = std::set<const SCEV *>;

// Description of the shape of an array.
struct ArrayShape {
  // Base pointer identifying all accesses to this array.
  const SCEVUnknown *BasePointer;

  // Sizes of each delinearized dimension.
  SmallVector<const SCEV *, 4> DelinearizedSizes;

  ArrayShape(const SCEVUnknown *B) : BasePointer(B) {}
};

struct MemAcc {
  const Instruction *Insn;

  // A pointer to the shape description of the array.
  std::shared_ptr<ArrayShape> Shape;

  // Subscripts computed by delinearization.
  SmallVector<const SCEV *, 4> DelinearizedSubscripts;

  MemAcc(const Instruction *I, std::shared_ptr<ArrayShape> S)
      : Insn(I), Shape(S) {}
};

using MapInsnToMemAcc = std::map<const Instruction *, MemAcc>;
using PairInstSCEV = std::pair<const Instruction *, const SCEV *>;
using AFs = std::vector<PairInstSCEV>;
using BaseToAFs = std::map<const SCEVUnknown *, AFs>;
using BaseToElSize = std::map<const SCEVUnknown *, const SCEV *>;

extern bool PollyTrackFailures;
extern bool PollyDelinearize;
extern bool PollyUseRuntimeAliasChecks;
extern bool PollyProcessUnprofitable;
extern bool PollyInvariantLoadHoisting;
extern bool PollyAllowUnsignedOperations;
extern bool PollyAllowFullFunction;

/// A function attribute which will cause Polly to skip the function
extern StringRef PollySkipFnAttr;

//===----------------------------------------------------------------------===//
/// Pass to detect the maximal static control parts (Scops) of a
/// function.
class ScopDetection {
public:
  using RegionSet = SetVector<const Region *>;

  // Remember the valid regions
  RegionSet ValidRegions;

  /// Context variables for SCoP detection.
  struct DetectionContext {
    Region &CurRegion;   // The region to check.
    AliasSetTracker AST; // The AliasSetTracker to hold the alias information.
    bool Verifying;      // If we are in the verification phase?

    /// Container to remember rejection reasons for this region.
    RejectLog Log;

    /// Map a base pointer to all access functions accessing it.
    ///
    /// This map is indexed by the base pointer. Each element of the map
    /// is a list of memory accesses that reference this base pointer.
    BaseToAFs Accesses;

    /// The set of base pointers with non-affine accesses.
    ///
    /// This set contains all base pointers and the locations where they are
    /// used for memory accesses that can not be detected as affine accesses.
    SetVector<std::pair<const SCEVUnknown *, Loop *>> NonAffineAccesses;
    BaseToElSize ElementSize;

    /// The region has at least one load instruction.
    bool hasLoads = false;

    /// The region has at least one store instruction.
    bool hasStores = false;

    /// Flag to indicate the region has at least one unknown access.
    bool HasUnknownAccess = false;

    /// The set of non-affine subregions in the region we analyze.
    RegionSet NonAffineSubRegionSet;

    /// The set of loops contained in non-affine regions.
    BoxedLoopsSetTy BoxedLoopsSet;

    /// Loads that need to be invariant during execution.
    InvariantLoadsSetTy RequiredILS;

    /// Map to memory access description for the corresponding LLVM
    ///        instructions.
    MapInsnToMemAcc InsnToMemAcc;

    /// Initialize a DetectionContext from scratch.
    DetectionContext(Region &R, AliasAnalysis &AA, bool Verify)
        : CurRegion(R), AST(AA), Verifying(Verify), Log(&R) {}

    /// Initialize a DetectionContext with the data from @p DC.
    DetectionContext(const DetectionContext &&DC)
        : CurRegion(DC.CurRegion), AST(DC.AST.getAliasAnalysis()),
          Verifying(DC.Verifying), Log(std::move(DC.Log)),
          Accesses(std::move(DC.Accesses)),
          NonAffineAccesses(std::move(DC.NonAffineAccesses)),
          ElementSize(std::move(DC.ElementSize)), hasLoads(DC.hasLoads),
          hasStores(DC.hasStores), HasUnknownAccess(DC.HasUnknownAccess),
          NonAffineSubRegionSet(std::move(DC.NonAffineSubRegionSet)),
          BoxedLoopsSet(std::move(DC.BoxedLoopsSet)),
          RequiredILS(std::move(DC.RequiredILS)) {
      AST.add(DC.AST);
    }
  };

  /// Helper data structure to collect statistics about loop counts.
  struct LoopStats {
    int NumLoops;
    int MaxDepth;
  };

private:
  //===--------------------------------------------------------------------===//

  /// Analyses used
  //@{
  const DominatorTree &DT;
  ScalarEvolution &SE;
  LoopInfo &LI;
  RegionInfo &RI;
  AliasAnalysis &AA;
  //@}

  /// Map to remember detection contexts for all regions.
  using DetectionContextMapTy = DenseMap<BBPair, DetectionContext>;
  mutable DetectionContextMapTy DetectionContextMap;

  /// Remove cached results for @p R.
  void removeCachedResults(const Region &R);

  /// Remove cached results for the children of @p R recursively.
  void removeCachedResultsRecursively(const Region &R);

  /// Check if @p S0 and @p S1 do contain multiple possibly aliasing pointers.
  ///
  /// @param S0    A expression to check.
  /// @param S1    Another expression to check or nullptr.
  /// @param Scope The loop/scope the expressions are checked in.
  ///
  /// @returns True, if multiple possibly aliasing pointers are used in @p S0
  ///          (and @p S1 if given).
  bool involvesMultiplePtrs(const SCEV *S0, const SCEV *S1, Loop *Scope) const;

  /// Add the region @p AR as over approximated sub-region in @p Context.
  ///
  /// @param AR      The non-affine subregion.
  /// @param Context The current detection context.
  ///
  /// @returns True if the subregion can be over approximated, false otherwise.
  bool addOverApproximatedRegion(Region *AR, DetectionContext &Context) const;

  /// Find for a given base pointer terms that hint towards dimension
  ///        sizes of a multi-dimensional array.
  ///
  /// @param Context      The current detection context.
  /// @param BasePointer  A base pointer indicating the virtual array we are
  ///                     interested in.
  SmallVector<const SCEV *, 4>
  getDelinearizationTerms(DetectionContext &Context,
                          const SCEVUnknown *BasePointer) const;

  /// Check if the dimension size of a delinearized array is valid.
  ///
  /// @param Context     The current detection context.
  /// @param Sizes       The sizes of the different array dimensions.
  /// @param BasePointer The base pointer we are interested in.
  /// @param Scope       The location where @p BasePointer is being used.
  /// @returns True if one or more array sizes could be derived - meaning: we
  ///          see this array as multi-dimensional.
  bool hasValidArraySizes(DetectionContext &Context,
                          SmallVectorImpl<const SCEV *> &Sizes,
                          const SCEVUnknown *BasePointer, Loop *Scope) const;

  /// Derive access functions for a given base pointer.
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

  /// Check if all accesses to a given BasePointer are affine.
  ///
  /// @param Context     The current detection context.
  /// @param BasePointer the base pointer we are interested in.
  /// @param Scope       The location where @p BasePointer is being used.
  /// @param True if consistent (multi-dimensional) array accesses could be
  ///        derived for this array.
  bool hasBaseAffineAccesses(DetectionContext &Context,
                             const SCEVUnknown *BasePointer, Loop *Scope) const;

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

  /// Check if all basic block in the region are valid.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if all blocks in R are valid, false otherwise.
  bool allBlocksValid(DetectionContext &Context) const;

  /// Check if a region has sufficient compute instructions.
  ///
  /// This function checks if a region has a non-trivial number of instructions
  /// in each loop. This can be used as an indicator whether a loop is worth
  /// optimizing.
  ///
  /// @param Context  The context of scop detection.
  /// @param NumLoops The number of loops in the region.
  ///
  /// @return True if region is has sufficient compute instructions,
  ///         false otherwise.
  bool hasSufficientCompute(DetectionContext &Context,
                            int NumAffineLoops) const;

  /// Check if the unique affine loop might be amendable to distribution.
  ///
  /// This function checks if the number of non-trivial blocks in the unique
  /// affine loop in Context.CurRegion is at least two, thus if the loop might
  /// be amendable to distribution.
  ///
  /// @param Context  The context of scop detection.
  ///
  /// @return True only if the affine loop might be amendable to distributable.
  bool hasPossiblyDistributableLoop(DetectionContext &Context) const;

  /// Check if a region is profitable to optimize.
  ///
  /// Regions that are unlikely to expose interesting optimization opportunities
  /// are called 'unprofitable' and may be skipped during scop detection.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if region is profitable to optimize, false otherwise.
  bool isProfitableRegion(DetectionContext &Context) const;

  /// Check if a region is a Scop.
  ///
  /// @param Context The context of scop detection.
  ///
  /// @return True if R is a Scop, false otherwise.
  bool isValidRegion(DetectionContext &Context) const;

  /// Check if an intrinsic call can be part of a Scop.
  ///
  /// @param II      The intrinsic call instruction to check.
  /// @param Context The current detection context.
  ///
  /// @return True if the call instruction is valid, false otherwise.
  bool isValidIntrinsicInst(IntrinsicInst &II, DetectionContext &Context) const;

  /// Check if a call instruction can be part of a Scop.
  ///
  /// @param CI      The call instruction to check.
  /// @param Context The current detection context.
  ///
  /// @return True if the call instruction is valid, false otherwise.
  bool isValidCallInst(CallInst &CI, DetectionContext &Context) const;

  /// Check if the given loads could be invariant and can be hoisted.
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

  /// Check if a value is invariant in the region Reg.
  ///
  /// @param Val Value to check for invariance.
  /// @param Reg The region to consider for the invariance of Val.
  /// @param Ctx The current detection context.
  ///
  /// @return True if the value represented by Val is invariant in the region
  ///         identified by Reg.
  bool isInvariant(Value &Val, const Region &Reg, DetectionContext &Ctx) const;

  /// Check if the memory access caused by @p Inst is valid.
  ///
  /// @param Inst    The access instruction.
  /// @param AF      The access function.
  /// @param BP      The access base pointer.
  /// @param Context The current detection context.
  bool isValidAccess(Instruction *Inst, const SCEV *AF, const SCEVUnknown *BP,
                     DetectionContext &Context) const;

  /// Check if a memory access can be part of a Scop.
  ///
  /// @param Inst The instruction accessing the memory.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the memory access is valid, false otherwise.
  bool isValidMemoryAccess(MemAccInst Inst, DetectionContext &Context) const;

  /// Check if an instruction has any non trivial scalar dependencies as part of
  /// a Scop.
  ///
  /// @param Inst The instruction to check.
  /// @param RefRegion The region in respect to which we check the access
  ///                  function.
  ///
  /// @return True if the instruction has scalar dependences, false otherwise.
  bool hasScalarDependency(Instruction &Inst, Region &RefRegion) const;

  /// Check if an instruction can be part of a Scop.
  ///
  /// @param Inst The instruction to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the instruction is valid, false otherwise.
  bool isValidInstruction(Instruction &Inst, DetectionContext &Context) const;

  /// Check if the switch @p SI with condition @p Condition is valid.
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

  /// Check if the branch @p BI with condition @p Condition is valid.
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

  /// Check if the SCEV @p S is affine in the current @p Context.
  ///
  /// This will also use a heuristic to decide if we want to require loads to be
  /// invariant to make the expression affine or if we want to treat is as
  /// non-affine.
  ///
  /// @param S           The expression to be checked.
  /// @param Scope       The loop nest in which @p S is used.
  /// @param Context     The context of scop detection.
  bool isAffine(const SCEV *S, Loop *Scope, DetectionContext &Context) const;

  /// Check if the control flow in a basic block is valid.
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

  /// Is a loop valid with respect to a given region.
  ///
  /// @param L The loop to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if the loop is valid in the region.
  bool isValidLoop(Loop *L, DetectionContext &Context) const;

  /// Count the number of loops and the maximal loop depth in @p L.
  ///
  /// @param L The loop to check.
  /// @param SE The scalar evolution analysis.
  /// @param MinProfitableTrips The minimum number of trip counts from which
  ///                           a loop is assumed to be profitable and
  ///                           consequently is counted.
  /// returns A tuple of number of loops and their maximal depth.
  static ScopDetection::LoopStats
  countBeneficialSubLoops(Loop *L, ScalarEvolution &SE,
                          unsigned MinProfitableTrips);

  /// Check if the function @p F is marked as invalid.
  ///
  /// @note An OpenMP subfunction will be marked as invalid.
  bool isValidFunction(Function &F);

  /// Can ISL compute the trip count of a loop.
  ///
  /// @param L The loop to check.
  /// @param Context The context of scop detection.
  ///
  /// @return True if ISL can compute the trip count of the loop.
  bool canUseISLTripCount(Loop *L, DetectionContext &Context) const;

  /// Print the locations of all detected scops.
  void printLocations(Function &F);

  /// Check if a region is reducible or not.
  ///
  /// @param Region The region to check.
  /// @param DbgLoc Parameter to save the location of instruction that
  ///               causes irregular control flow if the region is irreducible.
  ///
  /// @return True if R is reducible, false otherwise.
  bool isReducibleRegion(Region &R, DebugLoc &DbgLoc) const;

  /// Track diagnostics for invalid scops.
  ///
  /// @param Context The context of scop detection.
  /// @param Assert Throw an assert in verify mode or not.
  /// @param Args Argument list that gets passed to the constructor of RR.
  template <class RR, typename... Args>
  inline bool invalid(DetectionContext &Context, bool Assert,
                      Args &&... Arguments) const;

public:
  ScopDetection(Function &F, const DominatorTree &DT, ScalarEvolution &SE,
                LoopInfo &LI, RegionInfo &RI, AliasAnalysis &AA,
                OptimizationRemarkEmitter &ORE);

  /// Get the RegionInfo stored in this pass.
  ///
  /// This was added to give the DOT printer easy access to this information.
  RegionInfo *getRI() const { return &RI; }

  /// Get the LoopInfo stored in this pass.
  LoopInfo *getLI() const { return &LI; }

  /// Is the region is the maximum region of a Scop?
  ///
  /// @param R The Region to test if it is maximum.
  /// @param Verify Rerun the scop detection to verify SCoP was not invalidated
  ///               meanwhile.
  ///
  /// @return Return true if R is the maximum Region in a Scop, false otherwise.
  bool isMaxRegionInScop(const Region &R, bool Verify = true) const;

  /// Return the detection context for @p R, nullptr if @p R was invalid.
  DetectionContext *getDetectionContext(const Region *R) const;

  /// Return the set of rejection causes for @p R.
  const RejectLog *lookupRejectionLog(const Region *R) const;

  /// Return true if @p SubR is a non-affine subregion in @p ScopR.
  bool isNonAffineSubRegion(const Region *SubR, const Region *ScopR) const;

  /// Get a message why a region is invalid
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
  using iterator = RegionSet::iterator;
  using const_iterator = RegionSet::const_iterator;

  iterator begin() { return ValidRegions.begin(); }
  iterator end() { return ValidRegions.end(); }

  const_iterator begin() const { return ValidRegions.begin(); }
  const_iterator end() const { return ValidRegions.end(); }
  //@}

  /// Emit rejection remarks for all rejected regions.
  ///
  /// @param F The function to emit remarks for.
  void emitMissedRemarks(const Function &F);

  /// Mark the function as invalid so we will not extract any scop from
  ///        the function.
  ///
  /// @param F The function to mark as invalid.
  static void markFunctionAsInvalid(Function *F);

  /// Verify if all valid Regions in this Function are still valid
  /// after some transformations.
  void verifyAnalysis() const;

  /// Verify if R is still a valid part of Scop after some transformations.
  ///
  /// @param R The Region to verify.
  void verifyRegion(const Region &R) const;

  /// Count the number of loops and the maximal loop depth in @p R.
  ///
  /// @param R The region to check
  /// @param SE The scalar evolution analysis.
  /// @param MinProfitableTrips The minimum number of trip counts from which
  ///                           a loop is assumed to be profitable and
  ///                           consequently is counted.
  /// returns A tuple of number of loops and their maximal depth.
  static ScopDetection::LoopStats
  countBeneficialLoops(Region *R, ScalarEvolution &SE, LoopInfo &LI,
                       unsigned MinProfitableTrips);

private:
  /// OptimizationRemarkEmitter object used to emit diagnostic remarks
  OptimizationRemarkEmitter &ORE;
};

struct ScopAnalysis : public AnalysisInfoMixin<ScopAnalysis> {
  static AnalysisKey Key;

  using Result = ScopDetection;

  ScopAnalysis();

  Result run(Function &F, FunctionAnalysisManager &FAM);
};

struct ScopAnalysisPrinterPass : public PassInfoMixin<ScopAnalysisPrinterPass> {
  ScopAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

  raw_ostream &OS;
};

struct ScopDetectionWrapperPass : public FunctionPass {
  static char ID;
  std::unique_ptr<ScopDetection> Result;

  ScopDetectionWrapperPass();

  /// @name FunctionPass interface
  //@{
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  bool runOnFunction(Function &F) override;
  void print(raw_ostream &OS, const Module *) const override;
  //@}

  ScopDetection &getSD() { return *Result; }
  const ScopDetection &getSD() const { return *Result; }
};

} // namespace polly

#endif // POLLY_SCOPDETECTION_H
