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
// ScalarReplAggregates - Break up alloca's of aggregates into multiple allocas
// if possible.
//
FunctionPass *createScalarReplAggregatesPass(signed Threshold = -1);

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

//===----------------------------------------------------------------------===//
//
// LoopUnswitch - This pass is a simple loop unswitching pass.
//
Pass *createLoopUnswitchPass(bool OptimizeForSize = false);

//===----------------------------------------------------------------------===//
//
// LoopUnroll - This pass is a simple loop unrolling pass.
//
Pass *createLoopUnrollPass();

//===----------------------------------------------------------------------===//
//
// LoopRotate - This pass is a simple loop rotating pass.
//
Pass *createLoopRotatePass();

//===----------------------------------------------------------------------===//
//
// LoopIndexSplit - This pass divides loop's iteration range by spliting loop
// such that each individual loop is executed efficiently.
//
Pass *createLoopIndexSplitPass();

//===----------------------------------------------------------------------===//
//
// PromoteMemoryToRegister - This pass is used to promote memory references to
// be register references. A simple example of the transformation performed by
// this pass is:
//
//        FROM CODE                           TO CODE
//   %X = alloca int, uint 1                 ret int 42
//   store int 42, int *%X
//   %Y = load int* %X
//   ret int %Y
//
FunctionPass *createPromoteMemoryToRegisterPass();
extern const PassInfo *const PromoteMemoryToRegisterID;

//===----------------------------------------------------------------------===//
//
// DemoteRegisterToMemoryPass - This pass is used to demote registers to memory
// references. In basically undoes the PromoteMemoryToRegister pass to make cfg
// hacking easier.
//
FunctionPass *createDemoteRegisterToMemoryPass();
extern const PassInfo *const DemoteRegisterToMemoryID;

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
// CondPropagationPass - This pass propagates information about conditional
// expressions through the program, allowing it to eliminate conditional
// branches in some cases.
//
FunctionPass *createCondPropagationPass();

//===----------------------------------------------------------------------===//
//
// TailDuplication - Eliminate unconditional branches through controlled code
// duplication, creating simpler CFG structures.
//
FunctionPass *createTailDuplicationPass();

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
extern const PassInfo *const BreakCriticalEdgesID;

//===----------------------------------------------------------------------===//
//
// LoopSimplify - Insert Pre-header blocks into the CFG for every function in
// the module.  This pass updates dominator information, loop information, and
// does not add critical edges to the CFG.
//
//   AU.addRequiredID(LoopSimplifyID);
//
FunctionPass *createLoopSimplifyPass();
extern const PassInfo *const LoopSimplifyID;

//===----------------------------------------------------------------------===//
//
// LowerAllocations - Turn malloc and free instructions into %malloc and %free
// calls.
//
//   AU.addRequiredID(LowerAllocationsID);
//
Pass *createLowerAllocationsPass(bool LowerMallocArgToInteger = false);
extern const PassInfo *const LowerAllocationsID;

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
extern const PassInfo *const LowerSwitchID;

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
extern const PassInfo *const LowerInvokePassID;

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
extern const PassInfo *const LCSSAID;

//===----------------------------------------------------------------------===//
//
// PredicateSimplifier - This pass collapses duplicate variables into one
// canonical form, and tries to simplify expressions along the way.
//
FunctionPass *createPredicateSimplifierPass();

//===----------------------------------------------------------------------===//
//
// GVN-PRE - This pass performs global value numbering and partial redundancy
// elimination.
//
FunctionPass *createGVNPREPass();

//===----------------------------------------------------------------------===//
//
// GVN - This pass performs global value numbering and redundant load 
// elimination cotemporaneously.
//
FunctionPass *createGVNPass();

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
/// createSimplifyHalfPowrLibCallsPass - This is an experimental pass that
/// optimizes specific half_pow functions.
FunctionPass *createSimplifyHalfPowrLibCallsPass();

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
extern const PassInfo *const InstructionNamerID;
  
} // End llvm namespace

#endif
