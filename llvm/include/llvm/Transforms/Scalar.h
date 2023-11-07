//===-- Scalar.h - Scalar Transformations -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Scalar transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_H
#define LLVM_TRANSFORMS_SCALAR_H

#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include <functional>

namespace llvm {

class Function;
class FunctionPass;
class Pass;

//===----------------------------------------------------------------------===//
//
// RedundantDbgInstElimination - This pass removes redundant dbg intrinsics
// without modifying the CFG of the function.  It is a FunctionPass.
//
LLVM_FUNC_ABI Pass *createRedundantDbgInstEliminationPass();

//===----------------------------------------------------------------------===//
//
// DeadCodeElimination - This pass is more powerful than DeadInstElimination,
// because it is worklist driven that can potentially revisit instructions when
// their other instructions become dead, to eliminate chains of dead
// computations.
//
LLVM_FUNC_ABI FunctionPass *createDeadCodeEliminationPass();


//===----------------------------------------------------------------------===//
//
// GuardWidening - An optimization over the @llvm.experimental.guard intrinsic
// that (optimistically) combines multiple guards into one to have fewer checks
// at runtime.
//
LLVM_FUNC_ABI FunctionPass *createGuardWideningPass();


//===----------------------------------------------------------------------===//
//
// LoopGuardWidening - Analogous to the GuardWidening pass, but restricted to a
// single loop at a time for use within a LoopPassManager.  Desired effect is
// to widen guards into preheader or a single guard within loop if that's not
// possible.
//
LLVM_FUNC_ABI Pass *createLoopGuardWideningPass();


//===----------------------------------------------------------------------===//
//
// SROA - Replace aggregates or pieces of aggregates with scalar SSA values.
//
LLVM_FUNC_ABI FunctionPass *createSROAPass(bool PreserveCFG = true);

//===----------------------------------------------------------------------===//
//
// LICM - This pass is a loop invariant code motion and memory promotion pass.
//
LLVM_FUNC_ABI Pass *createLICMPass();

//===----------------------------------------------------------------------===//
//
// LoopSink - This pass sinks invariants from preheader to loop body where
// frequency is lower than loop preheader.
//
LLVM_FUNC_ABI Pass *createLoopSinkPass();

//===----------------------------------------------------------------------===//
//
// LoopPredication - This pass does loop predication on guards.
//
LLVM_FUNC_ABI Pass *createLoopPredicationPass();

//===----------------------------------------------------------------------===//
//
// LoopStrengthReduce - This pass is strength reduces GEP instructions that use
// a loop's canonical induction variable as one of their indices.
//
LLVM_FUNC_ABI Pass *createLoopStrengthReducePass();

//===----------------------------------------------------------------------===//
//
// LoopInstSimplify - This pass simplifies instructions in a loop's body.
//
LLVM_FUNC_ABI Pass *createLoopInstSimplifyPass();

//===----------------------------------------------------------------------===//
//
// LoopUnroll - This pass is a simple loop unrolling pass.
//
LLVM_FUNC_ABI Pass *createLoopUnrollPass(int OptLevel = 2, bool OnlyWhenForced = false,
                           bool ForgetAllSCEV = false, int Threshold = -1,
                           int Count = -1, int AllowPartial = -1,
                           int Runtime = -1, int UpperBound = -1,
                           int AllowPeeling = -1);

//===----------------------------------------------------------------------===//
//
// LoopRotate - This pass is a simple loop rotating pass.
//
LLVM_FUNC_ABI Pass *createLoopRotatePass(int MaxHeaderSize = -1, bool PrepareForLTO = false);

//===----------------------------------------------------------------------===//
//
// DemoteRegisterToMemoryPass - This pass is used to demote registers to memory
// references. In basically undoes the PromoteMemoryToRegister pass to make cfg
// hacking easier.
//
LLVM_FUNC_ABI FunctionPass *createDemoteRegisterToMemoryPass();
LLVM_FUNC_ABI extern char &DemoteRegisterToMemoryID;

//===----------------------------------------------------------------------===//
//
// Reassociate - This pass reassociates commutative expressions in an order that
// is designed to promote better constant propagation, GCSE, LICM, PRE...
//
// For example:  4 + (x + 5)  ->  x + (4 + 5)
//
LLVM_FUNC_ABI FunctionPass *createReassociatePass();

//===----------------------------------------------------------------------===//
//
// CFGSimplification - Merge basic blocks, eliminate unreachable blocks,
// simplify terminator instructions, convert switches to lookup tables, etc.
//
LLVM_FUNC_ABI FunctionPass *createCFGSimplificationPass(
    SimplifyCFGOptions Options = SimplifyCFGOptions(),
    std::function<bool(const Function &)> Ftor = nullptr);

//===----------------------------------------------------------------------===//
//
// FlattenCFG - flatten CFG, reduce number of conditional branches by using
// parallel-and and parallel-or mode, etc...
//
LLVM_FUNC_ABI FunctionPass *createFlattenCFGPass();

//===----------------------------------------------------------------------===//
//
// CFG Structurization - Remove irreducible control flow
//
///
/// When \p SkipUniformRegions is true the structizer will not structurize
/// regions that only contain uniform branches.
LLVM_FUNC_ABI Pass *createStructurizeCFGPass(bool SkipUniformRegions = false);

//===----------------------------------------------------------------------===//
//
// TailCallElimination - This pass eliminates call instructions to the current
// function which occur immediately before return instructions.
//
LLVM_FUNC_ABI FunctionPass *createTailCallEliminationPass();

//===----------------------------------------------------------------------===//
//
// EarlyCSE - This pass performs a simple and fast CSE pass over the dominator
// tree.
//
LLVM_FUNC_ABI FunctionPass *createEarlyCSEPass(bool UseMemorySSA = false);

//===----------------------------------------------------------------------===//
//
// MergedLoadStoreMotion - This pass merges loads and stores in diamonds. Loads
// are hoisted into the header, while stores sink into the footer.
//
LLVM_FUNC_ABI FunctionPass *createMergedLoadStoreMotionPass(bool SplitFooterBB = false);

//===----------------------------------------------------------------------===//
//
// ConstantHoisting - This pass prepares a function for expensive constants.
//
LLVM_FUNC_ABI FunctionPass *createConstantHoistingPass();

//===----------------------------------------------------------------------===//
//
// Sink - Code Sinking
//
LLVM_FUNC_ABI FunctionPass *createSinkingPass();

//===----------------------------------------------------------------------===//
//
// LowerAtomic - Lower atomic intrinsics to non-atomic form
//
LLVM_FUNC_ABI Pass *createLowerAtomicPass();

//===----------------------------------------------------------------------===//
//
// LowerWidenableCondition - Lower widenable condition to i1 true.
//
LLVM_FUNC_ABI Pass *createLowerWidenableConditionPass();

//===----------------------------------------------------------------------===//
//
// MergeICmps - Merge integer comparison chains into a memcmp
//
LLVM_FUNC_ABI Pass *createMergeICmpsLegacyPass();

//===----------------------------------------------------------------------===//
//
// InferAddressSpaces - Modify users of addrspacecast instructions with values
// in the source address space if using the destination address space is slower
// on the target. If AddressSpace is left to its default value, it will be
// obtained from the TargetTransformInfo.
//
LLVM_FUNC_ABI FunctionPass *createInferAddressSpacesPass(unsigned AddressSpace = ~0u);
LLVM_FUNC_ABI extern char &InferAddressSpacesID;

//===----------------------------------------------------------------------===//
//
// TLSVariableHoist - This pass reduce duplicated TLS address call.
//
LLVM_FUNC_ABI FunctionPass *createTLSVariableHoistPass();

//===----------------------------------------------------------------------===//
//
// LowerConstantIntrinsicss - Expand any remaining llvm.objectsize and
// llvm.is.constant intrinsic calls, even for the unknown cases.
//
LLVM_FUNC_ABI FunctionPass *createLowerConstantIntrinsicsPass();

//===----------------------------------------------------------------------===//
//
// PartiallyInlineLibCalls - Tries to inline the fast path of library
// calls such as sqrt.
//
LLVM_FUNC_ABI FunctionPass *createPartiallyInlineLibCallsPass();

//===----------------------------------------------------------------------===//
//
// SeparateConstOffsetFromGEP - Split GEPs for better CSE
//
LLVM_FUNC_ABI FunctionPass *createSeparateConstOffsetFromGEPPass(bool LowerGEP = false);

//===----------------------------------------------------------------------===//
//
// SpeculativeExecution - Aggressively hoist instructions to enable
// speculative execution on targets where branches are expensive.
//
LLVM_FUNC_ABI FunctionPass *createSpeculativeExecutionPass();

// Same as createSpeculativeExecutionPass, but does nothing unless
// TargetTransformInfo::hasBranchDivergence() is true.
LLVM_FUNC_ABI FunctionPass *createSpeculativeExecutionIfHasBranchDivergencePass();

//===----------------------------------------------------------------------===//
//
// StraightLineStrengthReduce - This pass strength-reduces some certain
// instruction patterns in straight-line code.
//
LLVM_FUNC_ABI FunctionPass *createStraightLineStrengthReducePass();

//===----------------------------------------------------------------------===//
//
// NaryReassociate - Simplify n-ary operations by reassociation.
//
LLVM_FUNC_ABI FunctionPass *createNaryReassociatePass();

//===----------------------------------------------------------------------===//
//
// LoopDataPrefetch - Perform data prefetching in loops.
//
LLVM_FUNC_ABI FunctionPass *createLoopDataPrefetchPass();

//===----------------------------------------------------------------------===//
//
// LoopSimplifyCFG - This pass performs basic CFG simplification on loops,
// primarily to help other loop passes.
//
LLVM_FUNC_ABI Pass *createLoopSimplifyCFGPass();

//===----------------------------------------------------------------------===//
//
// This pass does instruction simplification on each
// instruction in a function.
//
LLVM_FUNC_ABI FunctionPass *createInstSimplifyLegacyPass();


//===----------------------------------------------------------------------===//
//
// createScalarizeMaskedMemIntrinPass - Replace masked load, store, gather
// and scatter intrinsics with scalar code when target doesn't support them.
//
LLVM_FUNC_ABI FunctionPass *createScalarizeMaskedMemIntrinLegacyPass();
} // End llvm namespace

#endif
