//===-- Scalar.h - Scalar Transformations -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Scalar transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_H
#define LLVM_TRANSFORMS_SCALAR_H

namespace llvm {

class FunctionPass;
class Pass;
class GetElementPtrInst;
class PassInfo;
class TerminatorInst;
class TargetLowering;

//===----------------------------------------------------------------------===//
//
// ConstantPropagation - A worklist driven constant propagation pass
//
FunctionPass *createConstantPropagationPass();

//===----------------------------------------------------------------------===//
//
// SCCP - Sparse conditional constant propagation.
//
FunctionPass *createSCCPPass();

//===----------------------------------------------------------------------===//
//
// DeadInstElimination - This pass quickly removes trivially dead instructions
// without modifying the CFG of the function.  It is a BasicBlockPass, so it
// runs efficiently when queued next to other BasicBlockPass's.
//
Pass *createDeadInstEliminationPass();

//===----------------------------------------------------------------------===//
//
// DeadCodeElimination - This pass is more powerful than DeadInstElimination,
// because it is worklist driven that can potentially revisit instructions when
// their other instructions become dead, to eliminate chains of dead
// computations.
//
FunctionPass *createDeadCodeEliminationPass();

//===----------------------------------------------------------------------===//
//
// DeadStoreElimination - This pass deletes stores that are post-dominated by
// must-aliased stores and are not loaded used between the stores.
//
FunctionPass *createDeadStoreEliminationPass();

//===----------------------------------------------------------------------===//
//
// AggressiveDCE - This pass uses the SSA based Aggressive DCE algorithm.  This
// algorithm assumes instructions are dead until proven otherwise, which makes
// it more successful are removing non-obviously dead instructions.
//
FunctionPass *createAggressiveDCEPass();

//===----------------------------------------------------------------------===//
//
// SROA - Replace aggregates or pieces of aggregates with scalar SSA values.
//
FunctionPass *createSROAPass(bool RequiresDomTree = true);

//===----------------------------------------------------------------------===//
//
// ScalarReplAggregates - Break up alloca's of aggregates into multiple allocas
// if possible.
//
FunctionPass *createScalarReplAggregatesPass(signed Threshold = -1,
                                             bool UseDomTree = true,
                                             signed StructMemberThreshold = -1,
                                             signed ArrayElementThreshold = -1,
                                             signed ScalarLoadThreshold = -1);

//===----------------------------------------------------------------------===//
//
// InductionVariableSimplify - Transform induction variables in a program to all
// use a single canonical induction variable per loop.
//
Pass *createIndVarSimplifyPass();

//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions. This pass does not modify the CFG, and has a tendency to make
// instructions dead, so a subsequent DCE pass is useful.
//
// This pass combines things like:
//    %Y = add int 1, %X
//    %Z = add int 1, %Y
// into:
//    %Z = add int 2, %X
//
FunctionPass *createInstructionCombiningPass();

//===----------------------------------------------------------------------===//
//
// LICM - This pass is a loop invariant code motion and memory promotion pass.
//
Pass *createLICMPass();

//===----------------------------------------------------------------------===//
//
// LoopStrengthReduce - This pass is strength reduces GEP instructions that use
// a loop's canonical induction variable as one of their indices.  It takes an
// optional parameter used to consult the target machine whether certain
// transformations are profitable.
//
Pass *createLoopStrengthReducePass(const TargetLowering *TLI = 0);

Pass *createGlobalMergePass(const TargetLowering *TLI = 0);

//===----------------------------------------------------------------------===//
//
// LoopUnswitch - This pass is a simple loop unswitching pass.
//
Pass *createLoopUnswitchPass(bool OptimizeForSize = false);

//===----------------------------------------------------------------------===//
//
// LoopInstSimplify - This pass simplifies instructions in a loop's body.
//
Pass *createLoopInstSimplifyPass();

//===----------------------------------------------------------------------===//
//
// LoopUnroll - This pass is a simple loop unrolling pass.
//
Pass *createLoopUnrollPass(int Threshold = -1, int Count = -1, int AllowPartial = -1);

//===----------------------------------------------------------------------===//
//
// LoopRotate - This pass is a simple loop rotating pass.
//
Pass *createLoopRotatePass();

//===----------------------------------------------------------------------===//
//
// LoopIdiom - This pass recognizes and replaces idioms in loops.
//
Pass *createLoopIdiomPass();
  
//===----------------------------------------------------------------------===//
//
// PromoteMemoryToRegister - This pass is used to promote memory references to
// be register references. A simple example of the transformation performed by
// this pass is:
//
//        FROM CODE                           TO CODE
//   %X = alloca i32, i32 1                 ret i32 42
//   store i32 42, i32 *%X
//   %Y = load i32* %X
//   ret i32 %Y
//
FunctionPass *createPromoteMemoryToRegisterPass();

//===----------------------------------------------------------------------===//
//
// DemoteRegisterToMemoryPass - This pass is used to demote registers to memory
// references. In basically undoes the PromoteMemoryToRegister pass to make cfg
// hacking easier.
//
FunctionPass *createDemoteRegisterToMemoryPass();
extern char &DemoteRegisterToMemoryID;

//===----------------------------------------------------------------------===//
//
// Reassociate - This pass reassociates commutative expressions in an order that
// is designed to promote better constant propagation, GCSE, LICM, PRE...
//
// For example:  4 + (x + 5)  ->  x + (4 + 5)
//
FunctionPass *createReassociatePass();

//===----------------------------------------------------------------------===//
//
// JumpThreading - Thread control through mult-pred/multi-succ blocks where some
// preds always go to some succ.
//
FunctionPass *createJumpThreadingPass();
  
//===----------------------------------------------------------------------===//
//
// CFGSimplification - Merge basic blocks, eliminate unreachable blocks,
// simplify terminator instructions, etc...
//
FunctionPass *createCFGSimplificationPass();

//===----------------------------------------------------------------------===//
//
// BreakCriticalEdges - Break all of the critical edges in the CFG by inserting
// a dummy basic block. This pass may be "required" by passes that cannot deal
// with critical edges. For this usage, a pass must call:
//
//   AU.addRequiredID(BreakCriticalEdgesID);
//
// This pass obviously invalidates the CFG, but can update forward dominator
// (set, immediate dominators, tree, and frontier) information.
//
FunctionPass *createBreakCriticalEdgesPass();
extern char &BreakCriticalEdgesID;

//===----------------------------------------------------------------------===//
//
// LoopSimplify - Insert Pre-header blocks into the CFG for every function in
// the module.  This pass updates dominator information, loop information, and
// does not add critical edges to the CFG.
//
//   AU.addRequiredID(LoopSimplifyID);
//
Pass *createLoopSimplifyPass();
extern char &LoopSimplifyID;

//===----------------------------------------------------------------------===//
//
// TailCallElimination - This pass eliminates call instructions to the current
// function which occur immediately before return instructions.
//
FunctionPass *createTailCallEliminationPass();

//===----------------------------------------------------------------------===//
//
// LowerSwitch - This pass converts SwitchInst instructions into a sequence of
// chained binary branch instructions.
//
FunctionPass *createLowerSwitchPass();
extern char &LowerSwitchID;

//===----------------------------------------------------------------------===//
//
// LowerInvoke - This pass converts invoke and unwind instructions to use sjlj
// exception handling mechanisms.  Note that after this pass runs the CFG is not
// entirely accurate (exceptional control flow edges are not correct anymore) so
// only very simple things should be done after the lowerinvoke pass has run
// (like generation of native code).  This should *NOT* be used as a general
// purpose "my LLVM-to-LLVM pass doesn't support the invoke instruction yet"
// lowering pass.
//
FunctionPass *createLowerInvokePass(const TargetLowering *TLI = 0);
FunctionPass *createLowerInvokePass(const TargetLowering *TLI,
                                    bool useExpensiveEHSupport);
extern char &LowerInvokePassID;

//===----------------------------------------------------------------------===//
//
// BlockPlacement - This pass reorders basic blocks in order to increase the
// number of fall-through conditional branches.
//
FunctionPass *createBlockPlacementPass();

//===----------------------------------------------------------------------===//
//
// LCSSA - This pass inserts phi nodes at loop boundaries to simplify other loop
// optimizations.
//
Pass *createLCSSAPass();
extern char &LCSSAID;

//===----------------------------------------------------------------------===//
//
// EarlyCSE - This pass performs a simple and fast CSE pass over the dominator
// tree.
//
FunctionPass *createEarlyCSEPass();
  
//===----------------------------------------------------------------------===//
//
// GVN - This pass performs global value numbering and redundant load 
// elimination cotemporaneously.
//
FunctionPass *createGVNPass(bool NoLoads = false);

//===----------------------------------------------------------------------===//
//
// MemCpyOpt - This pass performs optimizations related to eliminating memcpy
// calls and/or combining multiple stores into memset's.
//
FunctionPass *createMemCpyOptPass();

//===----------------------------------------------------------------------===//
//
// LoopDeletion - This pass performs DCE of non-infinite loops that it
// can prove are dead.
//
Pass *createLoopDeletionPass();
  
//===----------------------------------------------------------------------===//
//
/// createSimplifyLibCallsPass - This pass optimizes specific calls to
/// specific well-known (library) functions.
FunctionPass *createSimplifyLibCallsPass();

//===----------------------------------------------------------------------===//
//
// CodeGenPrepare - This pass prepares a function for instruction selection.
//
FunctionPass *createCodeGenPreparePass(const TargetLowering *TLI = 0);

//===----------------------------------------------------------------------===//
//
// InstructionNamer - Give any unnamed non-void instructions "tmp" names.
//
FunctionPass *createInstructionNamerPass();
extern char &InstructionNamerID;
  
//===----------------------------------------------------------------------===//
//
// Sink - Code Sinking
//
FunctionPass *createSinkingPass();

//===----------------------------------------------------------------------===//
//
// LowerAtomic - Lower atomic intrinsics to non-atomic form
//
Pass *createLowerAtomicPass();

//===----------------------------------------------------------------------===//
//
// ValuePropagation - Propagate CFG-derived value information
//
Pass *createCorrelatedValuePropagationPass();

//===----------------------------------------------------------------------===//
//
// ObjCARCAPElim - ObjC ARC autorelease pool elimination.
//
Pass *createObjCARCAPElimPass();

//===----------------------------------------------------------------------===//
//
// ObjCARCExpand - ObjC ARC preliminary simplifications.
//
Pass *createObjCARCExpandPass();

//===----------------------------------------------------------------------===//
//
// ObjCARCContract - Late ObjC ARC cleanups.
//
Pass *createObjCARCContractPass();

//===----------------------------------------------------------------------===//
//
// ObjCARCOpt - ObjC ARC optimization.
//
Pass *createObjCARCOptPass();

//===----------------------------------------------------------------------===//
//
// InstructionSimplifier - Remove redundant instructions.
//
FunctionPass *createInstructionSimplifierPass();
extern char &InstructionSimplifierID;


//===----------------------------------------------------------------------===//
//
// LowerExpectIntriniscs - Removes llvm.expect intrinsics and creates
// "block_weights" metadata.
FunctionPass *createLowerExpectIntrinsicPass();


} // End llvm namespace

#endif
