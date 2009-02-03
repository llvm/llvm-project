//===-- Local.h - Functions to perform local transformations ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOCAL_H
#define LLVM_TRANSFORMS_UTILS_LOCAL_H

namespace llvm {

class BasicBlock;
class Instruction;
class Value;
class Pass;
class PHINode;
class AllocaInst;
class ConstantExpr;
class TargetData;

template<typename T> class SmallVectorImpl;
  
//===----------------------------------------------------------------------===//
//  Local constant propagation.
//

/// ConstantFoldTerminator - If a terminator instruction is predicated on a
/// constant value, convert it into an unconditional branch to the constant
/// destination.  This is a nontrivial operation because the successors of this
/// basic block must have their PHI nodes updated.
///
bool ConstantFoldTerminator(BasicBlock *BB);

//===----------------------------------------------------------------------===//
//  Local dead code elimination.
//

/// isInstructionTriviallyDead - Return true if the result produced by the
/// instruction is not used, and the instruction has no side effects.
///
bool isInstructionTriviallyDead(Instruction *I);

  
/// RecursivelyDeleteTriviallyDeadInstructions - If the specified value is a
/// trivially dead instruction, delete it.  If that makes any of its operands
/// trivially dead, delete them too, recursively.
///
/// If DeadInst is specified, the vector is filled with the instructions that
/// are actually deleted.
void RecursivelyDeleteTriviallyDeadInstructions(Value *V,
                                  SmallVectorImpl<Instruction*> *DeadInst = 0);
    
//===----------------------------------------------------------------------===//
//  Control Flow Graph Restructuring.
//

/// MergeBasicBlockIntoOnlyPred - BB is a block with one predecessor and its
/// predecessor is known to have one successor (BB!).  Eliminate the edge
/// between them, moving the instructions in the predecessor into BB.  This
/// deletes the predecessor block.
///
void MergeBasicBlockIntoOnlyPred(BasicBlock *BB);
    
  
/// SimplifyCFG - This function is used to do simplification of a CFG.  For
/// example, it adjusts branches to branches to eliminate the extra hop, it
/// eliminates unreachable basic blocks, and does other "peephole" optimization
/// of the CFG.  It returns true if a modification was made, possibly deleting
/// the basic block that was pointed to.
///
/// WARNING:  The entry node of a method may not be simplified.
///
bool SimplifyCFG(BasicBlock *BB);

/// DemoteRegToStack - This function takes a virtual register computed by an
/// Instruction and replaces it with a slot in the stack frame, allocated via
/// alloca.  This allows the CFG to be changed around without fear of
/// invalidating the SSA information for the value.  It returns the pointer to
/// the alloca inserted to create a stack slot for X.
///
AllocaInst *DemoteRegToStack(Instruction &X, bool VolatileLoads = false,
                             Instruction *AllocaPoint = 0);

/// DemotePHIToStack - This function takes a virtual register computed by a phi
/// node and replaces it with a slot in the stack frame, allocated via alloca.
/// The phi node is deleted and it returns the pointer to the alloca inserted. 
AllocaInst *DemotePHIToStack(PHINode *P, Instruction *AllocaPoint = 0);

} // End llvm namespace

#endif
