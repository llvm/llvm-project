//===- polly/ScopBuilder.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a static control flow region.
//
// The pass creates a polyhedral description of the Scops detected by the SCoP
// detection derived from their LLVM-IR code.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOPBUILDER_H
#define POLLY_SCOPBUILDER_H

#include "polly/ScopInfo.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"

namespace polly {
using llvm::SmallSetVector;

class ScopDetection;

/// Command line switch whether to model read-only accesses.
extern bool ModelReadOnlyScalars;

/// Build the Polly IR (Scop and ScopStmt) on a Region.
class ScopBuilder final {

  /// The AAResults to build AliasSetTracker.
  AAResults &AA;

  /// Target data for element size computing.
  const DataLayout &DL;

  /// DominatorTree to reason about guaranteed execution.
  DominatorTree &DT;

  /// LoopInfo for information about loops.
  LoopInfo &LI;

  /// Valid Regions for Scop
  ScopDetection &SD;

  /// The ScalarEvolution to help building Scop.
  ScalarEvolution &SE;

  /// An optimization diagnostic interface to add optimization remarks.
  OptimizationRemarkEmitter &ORE;

  /// Set of instructions that might read any memory location.
  SmallVector<std::pair<ScopStmt *, Instruction *>, 16> GlobalReads;

  /// Set of all accessed array base pointers.
  SmallSetVector<Value *, 16> ArrayBasePointers;

  // The Scop
  std::unique_ptr<Scop> scop;

  /// Collection to hold taken assumptions.
  ///
  /// There are two reasons why we want to record assumptions first before we
  /// add them to the assumed/invalid context:
  ///   1) If the SCoP is not profitable or otherwise invalid without the
  ///      assumed/invalid context we do not have to compute it.
  ///   2) Information about the context are gathered rather late in the SCoP
  ///      construction (basically after we know all parameters), thus the user
  ///      might see overly complicated assumptions to be taken while they will
  ///      only be simplified later on.
  RecordedAssumptionsTy RecordedAssumptions;

  // Build the SCoP for Region @p R.
  void buildScop(Region &R, AssumptionCache &AC);

  /// Adjust the dimensions of @p Dom that was constructed for @p OldL
  ///        to be compatible to domains constructed for loop @p NewL.
  ///
  /// This function assumes @p NewL and @p OldL are equal or there is a CFG
  /// edge from @p OldL to @p NewL.
  isl::set adjustDomainDimensions(isl::set Dom, Loop *OldL, Loop *NewL);

  /// Compute the domain for each basic block in @p R.
  ///
  /// @param R                The region we currently traverse.
  /// @param InvalidDomainMap BB to InvalidDomain map for the BB of current
  ///                         region.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool buildDomains(Region *R,
                    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap);

  /// Compute the branching constraints for each basic block in @p R.
  ///
  /// @param R                The region we currently build branching conditions
  ///                         for.
  /// @param InvalidDomainMap BB to InvalidDomain map for the BB of current
  ///                         region.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool buildDomainsWithBranchConstraints(
      Region *R, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap);

  /// Build the conditions sets for the terminator @p TI in the @p Domain.
  ///
  /// This will fill @p ConditionSets with the conditions under which control
  /// will be moved from @p TI to its successors. Hence, @p ConditionSets will
  /// have as many elements as @p TI has successors.
  bool buildConditionSets(BasicBlock *BB, Instruction *TI, Loop *L,
                          __isl_keep isl_set *Domain,
                          DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
                          SmallVectorImpl<__isl_give isl_set *> &ConditionSets);

  /// Build the conditions sets for the branch condition @p Condition in
  /// the @p Domain.
  ///
  /// This will fill @p ConditionSets with the conditions under which control
  /// will be moved from @p TI to its successors. Hence, @p ConditionSets will
  /// have as many elements as @p TI has successors. If @p TI is nullptr the
  /// context under which @p Condition is true/false will be returned as the
  /// new elements of @p ConditionSets.
  bool buildConditionSets(BasicBlock *BB, Value *Condition, Instruction *TI,
                          Loop *L, __isl_keep isl_set *Domain,
                          DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
                          SmallVectorImpl<__isl_give isl_set *> &ConditionSets);

  /// Build the conditions sets for the switch @p SI in the @p Domain.
  ///
  /// This will fill @p ConditionSets with the conditions under which control
  /// will be moved from @p SI to its successors. Hence, @p ConditionSets will
  /// have as many elements as @p SI has successors.
  bool buildConditionSets(BasicBlock *BB, SwitchInst *SI, Loop *L,
                          __isl_keep isl_set *Domain,
                          DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
                          SmallVectorImpl<__isl_give isl_set *> &ConditionSets);

  /// Build condition sets for unsigned ICmpInst(s).
  /// Special handling is required for unsigned operands to ensure that if
  /// MSB (aka the Sign bit) is set for an operands in an unsigned ICmpInst
  /// it should wrap around.
  ///
  /// @param IsStrictUpperBound holds information on the predicate relation
  /// between TestVal and UpperBound, i.e,
  /// TestVal < UpperBound  OR  TestVal <= UpperBound
  __isl_give isl_set *buildUnsignedConditionSets(
      BasicBlock *BB, Value *Condition, __isl_keep isl_set *Domain,
      const SCEV *SCEV_TestVal, const SCEV *SCEV_UpperBound,
      DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
      bool IsStrictUpperBound);

  /// Propagate the domain constraints through the region @p R.
  ///
  /// @param R                The region we currently build branching
  /// conditions for.
  /// @param InvalidDomainMap BB to InvalidDomain map for the BB of current
  ///                         region.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool propagateDomainConstraints(
      Region *R, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap);

  /// Propagate domains that are known due to graph properties.
  ///
  /// As a CFG is mostly structured we use the graph properties to propagate
  /// domains without the need to compute all path conditions. In particular,
  /// if a block A dominates a block B and B post-dominates A we know that the
  /// domain of B is a superset of the domain of A. As we do not have
  /// post-dominator information available here we use the less precise region
  /// information. Given a region R, we know that the exit is always executed
  /// if the entry was executed, thus the domain of the exit is a superset of
  /// the domain of the entry. In case the exit can only be reached from
  /// within the region the domains are in fact equal. This function will use
  /// this property to avoid the generation of condition constraints that
  /// determine when a branch is taken. If @p BB is a region entry block we
  /// will propagate its domain to the region exit block. Additionally, we put
  /// the region exit block in the @p FinishedExitBlocks set so we can later
  /// skip edges from within the region to that block.
  ///
  /// @param BB                 The block for which the domain is currently
  ///                           propagated.
  /// @param BBLoop             The innermost affine loop surrounding @p BB.
  /// @param FinishedExitBlocks Set of region exits the domain was set for.
  /// @param InvalidDomainMap   BB to InvalidDomain map for the BB of current
  ///                           region.
  void propagateDomainConstraintsToRegionExit(
      BasicBlock *BB, Loop *BBLoop,
      SmallPtrSetImpl<BasicBlock *> &FinishedExitBlocks,
      DenseMap<BasicBlock *, isl::set> &InvalidDomainMap);

  /// Propagate invalid domains of statements through @p R.
  ///
  /// This method will propagate invalid statement domains through @p R and at
  /// the same time add error block domains to them. Additionally, the domains
  /// of error statements and those only reachable via error statements will
  /// be replaced by an empty set. Later those will be removed completely.
  ///
  /// @param R                The currently traversed region.
  /// @param InvalidDomainMap BB to InvalidDomain map for the BB of current
  ///                         region.
  //
  /// @returns True if there was no problem and false otherwise.
  bool propagateInvalidStmtDomains(
      Region *R, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap);

  /// Compute the union of predecessor domains for @p BB.
  ///
  /// To compute the union of all domains of predecessors of @p BB this
  /// function applies similar reasoning on the CFG structure as described for
  ///   @see propagateDomainConstraintsToRegionExit
  ///
  /// @param BB     The block for which the predecessor domains are collected.
  /// @param Domain The domain under which BB is executed.
  ///
  /// @returns The domain under which @p BB is executed.
  isl::set getPredecessorDomainConstraints(BasicBlock *BB, isl::set Domain);

  /// Add loop carried constraints to the header block of the loop @p L.
  ///
  /// @param L                The loop to process.
  /// @param InvalidDomainMap BB to InvalidDomain map for the BB of current
  ///                         region.
  ///
  /// @returns True if there was no problem and false otherwise.
  bool addLoopBoundsToHeaderDomain(
      Loop *L, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap);

  /// Compute the isl representation for the SCEV @p E in this BB.
  ///
  /// @param BB               The BB for which isl representation is to be
  /// computed.
  /// @param InvalidDomainMap A map of BB to their invalid domains.
  /// @param E                The SCEV that should be translated.
  /// @param NonNegative      Flag to indicate the @p E has to be
  /// non-negative.
  ///
  /// Note that this function will also adjust the invalid context
  /// accordingly.
  __isl_give isl_pw_aff *
  getPwAff(BasicBlock *BB, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
           const SCEV *E, bool NonNegative = false);

  /// Create equivalence classes for required invariant accesses.
  ///
  /// These classes will consolidate multiple required invariant loads from the
  /// same address in order to keep the number of dimensions in the SCoP
  /// description small. For each such class equivalence class only one
  /// representing element, hence one required invariant load, will be chosen
  /// and modeled as parameter. The method
  /// Scop::getRepresentingInvariantLoadSCEV() will replace each element from an
  /// equivalence class with the representing element that is modeled. As a
  /// consequence Scop::getIdForParam() will only return an id for the
  /// representing element of each equivalence class, thus for each required
  /// invariant location.
  void buildInvariantEquivalenceClasses();

  /// Try to build a multi-dimensional fixed sized MemoryAccess from the
  /// Load/Store instruction.
  ///
  /// @param Inst       The Load/Store instruction that access the memory
  /// @param Stmt       The parent statement of the instruction
  ///
  /// @returns True if the access could be built, False otherwise.
  bool buildAccessMultiDimFixed(MemAccInst Inst, ScopStmt *Stmt);

  /// Try to build a multi-dimensional parametric sized MemoryAccess.
  ///        from the Load/Store instruction.
  ///
  /// @param Inst       The Load/Store instruction that access the memory
  /// @param Stmt       The parent statement of the instruction
  ///
  /// @returns True if the access could be built, False otherwise.
  bool buildAccessMultiDimParam(MemAccInst Inst, ScopStmt *Stmt);

  /// Try to build a MemoryAccess for a memory intrinsic.
  ///
  /// @param Inst       The instruction that access the memory
  /// @param Stmt       The parent statement of the instruction
  ///
  /// @returns True if the access could be built, False otherwise.
  bool buildAccessMemIntrinsic(MemAccInst Inst, ScopStmt *Stmt);

  /// Try to build a MemoryAccess for a call instruction.
  ///
  /// @param Inst       The call instruction that access the memory
  /// @param Stmt       The parent statement of the instruction
  ///
  /// @returns True if the access could be built, False otherwise.
  bool buildAccessCallInst(MemAccInst Inst, ScopStmt *Stmt);

  /// Build a single-dimensional parametric sized MemoryAccess
  ///        from the Load/Store instruction.
  ///
  /// @param Inst       The Load/Store instruction that access the memory
  /// @param Stmt       The parent statement of the instruction
  ///
  /// @returns True if the access could be built, False otherwise.
  bool buildAccessSingleDim(MemAccInst Inst, ScopStmt *Stmt);

  /// Finalize all access relations.
  ///
  /// When building up access relations, temporary access relations that
  /// correctly represent each individual access are constructed. However, these
  /// access relations can be inconsistent or non-optimal when looking at the
  /// set of accesses as a whole. This function finalizes the memory accesses
  /// and constructs a globally consistent state.
  void finalizeAccesses();

  /// Update access dimensionalities.
  ///
  /// When detecting memory accesses different accesses to the same array may
  /// have built with different dimensionality, as outer zero-values dimensions
  /// may not have been recognized as separate dimensions. This function goes
  /// again over all memory accesses and updates their dimensionality to match
  /// the dimensionality of the underlying ScopArrayInfo object.
  void updateAccessDimensionality();

  /// Fold size constants to the right.
  ///
  /// In case all memory accesses in a given dimension are multiplied with a
  /// common constant, we can remove this constant from the individual access
  /// functions and move it to the size of the memory access. We do this as this
  /// increases the size of the innermost dimension, consequently widens the
  /// valid range the array subscript in this dimension can evaluate to, and
  /// as a result increases the likelihood that our delinearization is
  /// correct.
  ///
  /// Example:
  ///
  ///    A[][n]
  ///    S[i,j] -> A[2i][2j+1]
  ///    S[i,j] -> A[2i][2j]
  ///
  ///    =>
  ///
  ///    A[][2n]
  ///    S[i,j] -> A[i][2j+1]
  ///    S[i,j] -> A[i][2j]
  ///
  /// Constants in outer dimensions can arise when the elements of a parametric
  /// multi-dimensional array are not elementary data types, but e.g.,
  /// structures.
  void foldSizeConstantsToRight();

  /// Fold memory accesses to handle parametric offset.
  ///
  /// As a post-processing step, we 'fold' memory accesses to parametric
  /// offsets in the access functions. @see MemoryAccess::foldAccess for
  /// details.
  void foldAccessRelations();

  /// Assume that all memory accesses are within bounds.
  ///
  /// After we have built a model of all memory accesses, we need to assume
  /// that the model we built matches reality -- aka. all modeled memory
  /// accesses always remain within bounds. We do this as last step, after
  /// all memory accesses have been modeled and canonicalized.
  void assumeNoOutOfBounds();

  /// Build the alias checks for this SCoP.
  bool buildAliasChecks();

  /// A vector of memory accesses that belong to an alias group.
  using AliasGroupTy = SmallVector<MemoryAccess *, 4>;

  /// A vector of alias groups.
  using AliasGroupVectorTy = SmallVector<AliasGroupTy, 4>;

  /// Build a given alias group and its access data.
  ///
  /// @param AliasGroup     The alias group to build.
  /// @param HasWriteAccess A set of arrays through which memory is not only
  ///                       read, but also written.
  //
  /// @returns True if __no__ error occurred, false otherwise.
  bool buildAliasGroup(AliasGroupTy &AliasGroup,
                       DenseSet<const ScopArrayInfo *> HasWriteAccess);

  /// Build all alias groups for this SCoP.
  ///
  /// @returns True if __no__ error occurred, false otherwise.
  bool buildAliasGroups();

  /// Build alias groups for all memory accesses in the Scop.
  ///
  /// Using the alias analysis and an alias set tracker we build alias sets
  /// for all memory accesses inside the Scop. For each alias set we then map
  /// the aliasing pointers back to the memory accesses we know, thus obtain
  /// groups of memory accesses which might alias. We also collect the set of
  /// arrays through which memory is written.
  ///
  /// @returns A pair consistent of a vector of alias groups and a set of arrays
  ///          through which memory is written.
  std::tuple<AliasGroupVectorTy, DenseSet<const ScopArrayInfo *>>
  buildAliasGroupsForAccesses();

  ///  Split alias groups by iteration domains.
  ///
  ///  We split each group based on the domains of the minimal/maximal accesses.
  ///  That means two minimal/maximal accesses are only in a group if their
  ///  access domains intersect. Otherwise, they are in different groups.
  ///
  ///  @param AliasGroups The alias groups to split
  void splitAliasGroupsByDomain(AliasGroupVectorTy &AliasGroups);

  /// Build an instance of MemoryAccess from the Load/Store instruction.
  ///
  /// @param Inst       The Load/Store instruction that access the memory
  /// @param Stmt       The parent statement of the instruction
  void buildMemoryAccess(MemAccInst Inst, ScopStmt *Stmt);

  /// Analyze and extract the cross-BB scalar dependences (or, dataflow
  /// dependencies) of an instruction.
  ///
  /// @param UserStmt The statement @p Inst resides in.
  /// @param Inst     The instruction to be analyzed.
  void buildScalarDependences(ScopStmt *UserStmt, Instruction *Inst);

  /// Build the escaping dependences for @p Inst.
  ///
  /// Search for uses of the llvm::Value defined by @p Inst that are not
  /// within the SCoP. If there is such use, add a SCALAR WRITE such that
  /// it is available after the SCoP as escaping value.
  ///
  /// @param Inst The instruction to be analyzed.
  void buildEscapingDependences(Instruction *Inst);

  /// Create MemoryAccesses for the given PHI node in the given region.
  ///
  /// @param PHIStmt            The statement @p PHI resides in.
  /// @param PHI                The PHI node to be handled
  /// @param NonAffineSubRegion The non affine sub-region @p PHI is in.
  /// @param IsExitBlock        Flag to indicate that @p PHI is in the exit BB.
  void buildPHIAccesses(ScopStmt *PHIStmt, PHINode *PHI,
                        Region *NonAffineSubRegion, bool IsExitBlock = false);

  /// Build the access functions for the subregion @p SR.
  void buildAccessFunctions();

  /// Should an instruction be modeled in a ScopStmt.
  ///
  /// @param Inst The instruction to check.
  /// @param L    The loop in which context the instruction is looked at.
  ///
  /// @returns True if the instruction should be modeled.
  bool shouldModelInst(Instruction *Inst, Loop *L);

  /// Create one or more ScopStmts for @p BB.
  ///
  /// Consecutive instructions are associated to the same statement until a
  /// separator is found.
  void buildSequentialBlockStmts(BasicBlock *BB, bool SplitOnStore = false);

  /// Create one or more ScopStmts for @p BB using equivalence classes.
  ///
  /// Instructions of a basic block that belong to the same equivalence class
  /// are added to the same statement.
  void buildEqivClassBlockStmts(BasicBlock *BB);

  /// Create ScopStmt for all BBs and non-affine subregions of @p SR.
  ///
  /// @param SR A subregion of @p R.
  ///
  /// Some of the statements might be optimized away later when they do not
  /// access any memory and thus have no effect.
  void buildStmts(Region &SR);

  /// Build the access functions for the statement @p Stmt in or represented by
  /// @p BB.
  ///
  /// @param Stmt               Statement to add MemoryAccesses to.
  /// @param BB                 A basic block in @p R.
  /// @param NonAffineSubRegion The non affine sub-region @p BB is in.
  void buildAccessFunctions(ScopStmt *Stmt, BasicBlock &BB,
                            Region *NonAffineSubRegion = nullptr);

  /// Create a new MemoryAccess object and add it to #AccFuncMap.
  ///
  /// @param Stmt        The statement where the access takes place.
  /// @param Inst        The instruction doing the access. It is not necessarily
  ///                    inside @p BB.
  /// @param AccType     The kind of access.
  /// @param BaseAddress The accessed array's base address.
  /// @param ElemType    The type of the accessed array elements.
  /// @param Affine      Whether all subscripts are affine expressions.
  /// @param AccessValue Value read or written.
  /// @param Subscripts  Access subscripts per dimension.
  /// @param Sizes       The array dimension's sizes.
  /// @param Kind        The kind of memory accessed.
  ///
  /// @return The created MemoryAccess, or nullptr if the access is not within
  ///         the SCoP.
  MemoryAccess *addMemoryAccess(ScopStmt *Stmt, Instruction *Inst,
                                MemoryAccess::AccessType AccType,
                                Value *BaseAddress, Type *ElemType, bool Affine,
                                Value *AccessValue,
                                ArrayRef<const SCEV *> Subscripts,
                                ArrayRef<const SCEV *> Sizes, MemoryKind Kind);

  /// Create a MemoryAccess that represents either a LoadInst or
  /// StoreInst.
  ///
  /// @param Stmt        The statement to add the MemoryAccess to.
  /// @param MemAccInst  The LoadInst or StoreInst.
  /// @param AccType     The kind of access.
  /// @param BaseAddress The accessed array's base address.
  /// @param ElemType    The type of the accessed array elements.
  /// @param IsAffine    Whether all subscripts are affine expressions.
  /// @param Subscripts  Access subscripts per dimension.
  /// @param Sizes       The array dimension's sizes.
  /// @param AccessValue Value read or written.
  ///
  /// @see MemoryKind
  void addArrayAccess(ScopStmt *Stmt, MemAccInst MemAccInst,
                      MemoryAccess::AccessType AccType, Value *BaseAddress,
                      Type *ElemType, bool IsAffine,
                      ArrayRef<const SCEV *> Subscripts,
                      ArrayRef<const SCEV *> Sizes, Value *AccessValue);

  /// Create a MemoryAccess for writing an llvm::Instruction.
  ///
  /// The access will be created at the position of @p Inst.
  ///
  /// @param Inst The instruction to be written.
  ///
  /// @see ensureValueRead()
  /// @see MemoryKind
  void ensureValueWrite(Instruction *Inst);

  /// Ensure an llvm::Value is available in the BB's statement, creating a
  /// MemoryAccess for reloading it if necessary.
  ///
  /// @param V        The value expected to be loaded.
  /// @param UserStmt Where to reload the value.
  ///
  /// @see ensureValueStore()
  /// @see MemoryKind
  void ensureValueRead(Value *V, ScopStmt *UserStmt);

  /// Create a write MemoryAccess for the incoming block of a phi node.
  ///
  /// Each of the incoming blocks write their incoming value to be picked in the
  /// phi's block.
  ///
  /// @param PHI           PHINode under consideration.
  /// @param IncomingStmt  The statement to add the MemoryAccess to.
  /// @param IncomingBlock Some predecessor block.
  /// @param IncomingValue @p PHI's value when coming from @p IncomingBlock.
  /// @param IsExitBlock   When true, uses the .s2a alloca instead of the
  ///                      .phiops one. Required for values escaping through a
  ///                      PHINode in the SCoP region's exit block.
  /// @see addPHIReadAccess()
  /// @see MemoryKind
  void ensurePHIWrite(PHINode *PHI, ScopStmt *IncomintStmt,
                      BasicBlock *IncomingBlock, Value *IncomingValue,
                      bool IsExitBlock);

  /// Add user provided parameter constraints to context (command line).
  void addUserContext();

  /// Add user provided parameter constraints to context (source code).
  void addUserAssumptions(AssumptionCache &AC,
                          DenseMap<BasicBlock *, isl::set> &InvalidDomainMap);

  /// Add all recorded assumptions to the assumed context.
  void addRecordedAssumptions();

  /// Create a MemoryAccess for reading the value of a phi.
  ///
  /// The modeling assumes that all incoming blocks write their incoming value
  /// to the same location. Thus, this access will read the incoming block's
  /// value as instructed by this @p PHI.
  ///
  /// @param PHIStmt Statement @p PHI resides in.
  /// @param PHI     PHINode under consideration; the READ access will be added
  ///                here.
  ///
  /// @see ensurePHIWrite()
  /// @see MemoryKind
  void addPHIReadAccess(ScopStmt *PHIStmt, PHINode *PHI);

  /// Wrapper function to calculate minimal/maximal accesses to each array.
  bool calculateMinMaxAccess(AliasGroupTy AliasGroup,
                             Scop::MinMaxVectorTy &MinMaxAccesses);
  /// Build the domain of @p Stmt.
  void buildDomain(ScopStmt &Stmt);

  /// Fill NestLoops with loops surrounding @p Stmt.
  void collectSurroundingLoops(ScopStmt &Stmt);

  /// Check for reductions in @p Stmt.
  ///
  /// Iterate over all store memory accesses and check for valid binary
  /// reduction like chains. For all candidates we check if they have the same
  /// base address and there are no other accesses which overlap with them. The
  /// base address check rules out impossible reductions candidates early. The
  /// overlap check, together with the "only one user" check in
  /// collectCandidateReductionLoads, guarantees that none of the intermediate
  /// results will escape during execution of the loop nest. We basically check
  /// here that no other memory access can access the same memory as the
  /// potential reduction.
  void checkForReductions(ScopStmt &Stmt);

  /// Verify that all required invariant loads have been hoisted.
  ///
  /// Invariant load hoisting is not guaranteed to hoist all loads that were
  /// assumed to be scop invariant during scop detection. This function checks
  /// for cases where the hoisting failed, but where it would have been
  /// necessary for our scop modeling to be correct. In case of insufficient
  /// hoisting the scop is marked as invalid.
  ///
  /// In the example below Bound[1] is required to be invariant:
  ///
  /// for (int i = 1; i < Bound[0]; i++)
  ///   for (int j = 1; j < Bound[1]; j++)
  ///     ...
  void verifyInvariantLoads();

  /// Hoist invariant memory loads and check for required ones.
  ///
  /// We first identify "common" invariant loads, thus loads that are invariant
  /// and can be hoisted. Then we check if all required invariant loads have
  /// been identified as (common) invariant. A load is a required invariant load
  /// if it was assumed to be invariant during SCoP detection, e.g., to assume
  /// loop bounds to be affine or runtime alias checks to be placeable. In case
  /// a required invariant load was not identified as (common) invariant we will
  /// drop this SCoP. An example for both "common" as well as required invariant
  /// loads is given below:
  ///
  /// for (int i = 1; i < *LB[0]; i++)
  ///   for (int j = 1; j < *LB[1]; j++)
  ///     A[i][j] += A[0][0] + (*V);
  ///
  /// Common inv. loads: V, A[0][0], LB[0], LB[1]
  /// Required inv. loads: LB[0], LB[1], (V, if it may alias with A or LB)
  void hoistInvariantLoads();

  /// Add invariant loads listed in @p InvMAs with the domain of @p Stmt.
  void addInvariantLoads(ScopStmt &Stmt, InvariantAccessesTy &InvMAs);

  /// Check if @p MA can always be hoisted without execution context.
  bool canAlwaysBeHoisted(MemoryAccess *MA, bool StmtInvalidCtxIsEmpty,
                          bool MAInvalidCtxIsEmpty,
                          bool NonHoistableCtxIsEmpty);

  /// Return true if and only if @p LI is a required invariant load.
  bool isRequiredInvariantLoad(LoadInst *LI) const {
    return scop->getRequiredInvariantLoads().count(LI);
  }

  /// Check if the base ptr of @p MA is in the SCoP but not hoistable.
  bool hasNonHoistableBasePtrInScop(MemoryAccess *MA, isl::union_map Writes);

  /// Return the context under which the access cannot be hoisted.
  ///
  /// @param Access The access to check.
  /// @param Writes The set of all memory writes in the scop.
  ///
  /// @return Return the context under which the access cannot be hoisted or a
  ///         nullptr if it cannot be hoisted at all.
  isl::set getNonHoistableCtx(MemoryAccess *Access, isl::union_map Writes);

  /// Build the access relation of all memory accesses of @p Stmt.
  void buildAccessRelations(ScopStmt &Stmt);

  /// Canonicalize arrays with base pointers from the same equivalence class.
  ///
  /// Some context: in our normal model we assume that each base pointer is
  /// related to a single specific memory region, where memory regions
  /// associated with different base pointers are disjoint. Consequently we do
  /// not need to compute additional data dependences that model possible
  /// overlaps of these memory regions. To verify our assumption we compute
  /// alias checks that verify that modeled arrays indeed do not overlap. In
  /// case an overlap is detected the runtime check fails and we fall back to
  /// the original code.
  ///
  /// In case of arrays where the base pointers are know to be identical,
  /// because they are dynamically loaded by accesses that are in the same
  /// invariant load equivalence class, such run-time alias check would always
  /// be false.
  ///
  /// This function makes sure that we do not generate consistently failing
  /// run-time checks for code that contains distinct arrays with known
  /// equivalent base pointers. It identifies for each invariant load
  /// equivalence class a single canonical array and canonicalizes all memory
  /// accesses that reference arrays that have base pointers that are known to
  /// be equal to the base pointer of such a canonical array to this canonical
  /// array.
  ///
  /// We currently do not canonicalize arrays for which certain memory accesses
  /// have been hoisted as loop invariant.
  void canonicalizeDynamicBasePtrs();

  /// Construct the schedule of this SCoP.
  void buildSchedule();

  /// A loop stack element to keep track of per-loop information during
  ///        schedule construction.
  using LoopStackElementTy = struct LoopStackElement {
    // The loop for which we keep information.
    Loop *L;

    // The (possibly incomplete) schedule for this loop.
    isl::schedule Schedule;

    // The number of basic blocks in the current loop, for which a schedule has
    // already been constructed.
    unsigned NumBlocksProcessed;

    LoopStackElement(Loop *L, isl::schedule S, unsigned NumBlocksProcessed)
        : L(L), Schedule(S), NumBlocksProcessed(NumBlocksProcessed) {}
  };

  /// The loop stack used for schedule construction.
  ///
  /// The loop stack keeps track of schedule information for a set of nested
  /// loops as well as an (optional) 'nullptr' loop that models the outermost
  /// schedule dimension. The loops in a loop stack always have a parent-child
  /// relation where the loop at position n is the parent of the loop at
  /// position n + 1.
  using LoopStackTy = SmallVector<LoopStackElementTy, 4>;

  /// Construct schedule information for a given Region and add the
  ///        derived information to @p LoopStack.
  ///
  /// Given a Region we derive schedule information for all RegionNodes
  /// contained in this region ensuring that the assigned execution times
  /// correctly model the existing control flow relations.
  ///
  /// @param R              The region which to process.
  /// @param LoopStack      A stack of loops that are currently under
  ///                       construction.
  void buildSchedule(Region *R, LoopStackTy &LoopStack);

  /// Build Schedule for the region node @p RN and add the derived
  ///        information to @p LoopStack.
  ///
  /// In case @p RN is a BasicBlock or a non-affine Region, we construct the
  /// schedule for this @p RN and also finalize loop schedules in case the
  /// current @p RN completes the loop.
  ///
  /// In case @p RN is a not-non-affine Region, we delegate the construction to
  /// buildSchedule(Region *R, ...).
  ///
  /// @param RN             The RegionNode region traversed.
  /// @param LoopStack      A stack of loops that are currently under
  ///                       construction.
  void buildSchedule(RegionNode *RN, LoopStackTy &LoopStack);

public:
  explicit ScopBuilder(Region *R, AssumptionCache &AC, AAResults &AA,
                       const DataLayout &DL, DominatorTree &DT, LoopInfo &LI,
                       ScopDetection &SD, ScalarEvolution &SE,
                       OptimizationRemarkEmitter &ORE);
  ScopBuilder(const ScopBuilder &) = delete;
  ScopBuilder &operator=(const ScopBuilder &) = delete;
  ~ScopBuilder() = default;

  /// Try to build the Polly IR of static control part on the current
  /// SESE-Region.
  ///
  /// @return Give up the ownership of the scop object or static control part
  ///         for the region
  std::unique_ptr<Scop> getScop() { return std::move(scop); }
};
} // end namespace polly

#endif // POLLY_SCOPBUILDER_H
