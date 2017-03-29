//===- polly/ScopBuilder.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a static control flow region.
//
// The pass creates a polyhedral description of the Scops detected by the SCoP
// detection derived from their LLVM-IR code.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOP_BUILDER_H
#define POLLY_SCOP_BUILDER_H

#include "polly/ScopInfo.h"

namespace polly {

/// Build the Polly IR (Scop and ScopStmt) on a Region.
class ScopBuilder {
  //===-------------------------------------------------------------------===//
  ScopBuilder(const ScopBuilder &) = delete;
  const ScopBuilder &operator=(const ScopBuilder &) = delete;

  /// The AliasAnalysis to build AliasSetTracker.
  AliasAnalysis &AA;

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

  /// Set of instructions that might read any memory location.
  SmallVector<Instruction *, 16> GlobalReads;

  /// Set of all accessed array base pointers.
  SmallSetVector<Value *, 16> ArrayBasePointers;

  // The Scop
  std::unique_ptr<Scop> scop;

  // Build the SCoP for Region @p R.
  void buildScop(Region &R, AssumptionCache &AC);

  /// Try to build a multi-dimensional fixed sized MemoryAccess from the
  /// Load/Store instruction.
  ///
  /// @param Inst       The Load/Store instruction that access the memory
  /// @param Stmt       The parent statement of the instruction
  ///
  /// @returns True if the access could be built, False otherwise.
  bool buildAccessMultiDimFixed(MemAccInst Inst, ScopStmt *Stmt);

  /// Try to build a multi-dimensional parameteric sized MemoryAccess.
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
  void buildAccessSingleDim(MemAccInst Inst, ScopStmt *Stmt);

  /// Build an instance of MemoryAccess from the Load/Store instruction.
  ///
  /// @param Inst       The Load/Store instruction that access the memory
  /// @param Stmt       The parent statement of the instruction
  void buildMemoryAccess(MemAccInst Inst, ScopStmt *Stmt);

  /// Analyze and extract the cross-BB scalar dependences (or, dataflow
  /// dependencies) of an instruction.
  ///
  /// @param Inst    The instruction to be analyzed.
  void buildScalarDependences(Instruction *Inst);

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
  /// @param PHI                The PHI node to be handled
  /// @param NonAffineSubRegion The non affine sub-region @p PHI is in.
  /// @param IsExitBlock        Flag to indicate that @p PHI is in the exit BB.
  void buildPHIAccesses(PHINode *PHI, Region *NonAffineSubRegion,
                        bool IsExitBlock = false);

  /// Build the access functions for the subregion @p SR.
  ///
  /// @param SR           A subregion of @p R.
  /// @param InsnToMemAcc The Instruction to MemoryAccess mapping.
  void buildAccessFunctions(Region &SR);

  /// Create ScopStmt for all BBs and non-affine subregions of @p SR.
  ///
  /// @param SR A subregion of @p R.
  ///
  /// Some of the statments might be optimized away later when they do not
  /// access any memory and thus have no effect.
  void buildStmts(Region &SR);

  /// Build the access functions for the basic block @p BB.
  ///
  /// @param BB                 A basic block in @p R.
  /// @param NonAffineSubRegion The non affine sub-region @p BB is in.
  /// @param IsExitBlock        Flag to indicate that @p BB is in the exit BB.
  void buildAccessFunctions(BasicBlock &BB,
                            Region *NonAffineSubRegion = nullptr,
                            bool IsExitBlock = false);

  /// Create a new MemoryAccess object and add it to #AccFuncMap.
  ///
  /// @param BB          The block where the access takes place.
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
  MemoryAccess *addMemoryAccess(BasicBlock *BB, Instruction *Inst,
                                MemoryAccess::AccessType AccType,
                                Value *BaseAddress, Type *ElemType, bool Affine,
                                Value *AccessValue,
                                ArrayRef<const SCEV *> Subscripts,
                                ArrayRef<const SCEV *> Sizes, MemoryKind Kind);

  /// Create a MemoryAccess that represents either a LoadInst or
  /// StoreInst.
  ///
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
  void addArrayAccess(MemAccInst MemAccInst, MemoryAccess::AccessType AccType,
                      Value *BaseAddress, Type *ElemType, bool IsAffine,
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
  /// @param V      The value expected to be loaded.
  /// @param UserBB Where to reload the value.
  ///
  /// @see ensureValueStore()
  /// @see MemoryKind
  void ensureValueRead(Value *V, BasicBlock *UserBB);

  /// Create a write MemoryAccess for the incoming block of a phi node.
  ///
  /// Each of the incoming blocks write their incoming value to be picked in the
  /// phi's block.
  ///
  /// @param PHI           PHINode under consideration.
  /// @param IncomingBlock Some predecessor block.
  /// @param IncomingValue @p PHI's value when coming from @p IncomingBlock.
  /// @param IsExitBlock   When true, uses the .s2a alloca instead of the
  ///                      .phiops one. Required for values escaping through a
  ///                      PHINode in the SCoP region's exit block.
  /// @see addPHIReadAccess()
  /// @see MemoryKind
  void ensurePHIWrite(PHINode *PHI, BasicBlock *IncomingBlock,
                      Value *IncomingValue, bool IsExitBlock);

  /// Create a MemoryAccess for reading the value of a phi.
  ///
  /// The modeling assumes that all incoming blocks write their incoming value
  /// to the same location. Thus, this access will read the incoming block's
  /// value as instructed by this @p PHI.
  ///
  /// @param PHI PHINode under consideration; the READ access will be added
  /// here.
  ///
  /// @see ensurePHIWrite()
  /// @see MemoryKind
  void addPHIReadAccess(PHINode *PHI);

public:
  explicit ScopBuilder(Region *R, AssumptionCache &AC, AliasAnalysis &AA,
                       const DataLayout &DL, DominatorTree &DT, LoopInfo &LI,
                       ScopDetection &SD, ScalarEvolution &SE);
  ~ScopBuilder() {}

  /// Try to build the Polly IR of static control part on the current
  /// SESE-Region.
  ///
  /// @return Give up the ownership of the scop object or static control part
  ///         for the region
  std::unique_ptr<Scop> getScop() { return std::move(scop); }
};

} // end namespace polly

namespace llvm {
class PassRegistry;
void initializeScopInfoRegionPassPass(llvm::PassRegistry &);
void initializeScopInfoWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif
