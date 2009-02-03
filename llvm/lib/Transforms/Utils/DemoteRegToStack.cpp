//===- DemoteRegToStack.cpp - Move a virtual register to the stack --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provide the function DemoteRegToStack().  This function takes a
// virtual register computed by an Instruction and replaces it with a slot in
// the stack frame, allocated via alloca. It returns the pointer to the
// AllocaInst inserted.  After this function is called on an instruction, we are
// guaranteed that the only user of the instruction is a store that is
// immediately after it.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include <map>
using namespace llvm;

/// DemoteRegToStack - This function takes a virtual register computed by an
/// Instruction and replaces it with a slot in the stack frame, allocated via
/// alloca.  This allows the CFG to be changed around without fear of
/// invalidating the SSA information for the value.  It returns the pointer to
/// the alloca inserted to create a stack slot for I.
///
AllocaInst* llvm::DemoteRegToStack(Instruction &I, bool VolatileLoads,
                                   Instruction *AllocaPoint) {
  if (I.use_empty()) {
    I.eraseFromParent();
    return 0;
  }
  
  // Create a stack slot to hold the value.
  AllocaInst *Slot;
  if (AllocaPoint) {
    Slot = new AllocaInst(I.getType(), 0, I.getName()+".reg2mem", AllocaPoint);
  } else {
    Function *F = I.getParent()->getParent();
    Slot = new AllocaInst(I.getType(), 0, I.getName()+".reg2mem",
                          F->getEntryBlock().begin());
  }
  
  // Change all of the users of the instruction to read from the stack slot
  // instead.
  while (!I.use_empty()) {
    Instruction *U = cast<Instruction>(I.use_back());
    if (PHINode *PN = dyn_cast<PHINode>(U)) {
      // If this is a PHI node, we can't insert a load of the value before the
      // use.  Instead, insert the load in the predecessor block corresponding
      // to the incoming value.
      //
      // Note that if there are multiple edges from a basic block to this PHI
      // node that we cannot multiple loads.  The problem is that the resultant
      // PHI node will have multiple values (from each load) coming in from the
      // same block, which is illegal SSA form.  For this reason, we keep track
      // and reuse loads we insert.
      std::map<BasicBlock*, Value*> Loads;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingValue(i) == &I) {
          Value *&V = Loads[PN->getIncomingBlock(i)];
          if (V == 0) {
            // Insert the load into the predecessor block
            V = new LoadInst(Slot, I.getName()+".reload", VolatileLoads, 
                             PN->getIncomingBlock(i)->getTerminator());
          }
          PN->setIncomingValue(i, V);
        }

    } else {
      // If this is a normal instruction, just insert a load.
      Value *V = new LoadInst(Slot, I.getName()+".reload", VolatileLoads, U);
      U->replaceUsesOfWith(&I, V);
    }
  }


  // Insert stores of the computed value into the stack slot.  We have to be
  // careful is I is an invoke instruction though, because we can't insert the
  // store AFTER the terminator instruction.
  BasicBlock::iterator InsertPt;
  if (!isa<TerminatorInst>(I)) {
    InsertPt = &I;
    ++InsertPt;
  } else {
    // We cannot demote invoke instructions to the stack if their normal edge
    // is critical.
    InvokeInst &II = cast<InvokeInst>(I);
    assert(II.getNormalDest()->getSinglePredecessor() &&
           "Cannot demote invoke with a critical successor!");
    InsertPt = II.getNormalDest()->begin();
  }

  for (; isa<PHINode>(InsertPt); ++InsertPt)
  /* empty */;   // Don't insert before any PHI nodes.
  new StoreInst(&I, Slot, InsertPt);

  return Slot;
}


/// DemotePHIToStack - This function takes a virtual register computed by a phi
/// node and replaces it with a slot in the stack frame, allocated via alloca.
/// The phi node is deleted and it returns the pointer to the alloca inserted.
AllocaInst* llvm::DemotePHIToStack(PHINode *P, Instruction *AllocaPoint) {
  if (P->use_empty()) {
    P->eraseFromParent();    
    return 0;                
  }

  // Create a stack slot to hold the value.
  AllocaInst *Slot;
  if (AllocaPoint) {
    Slot = new AllocaInst(P->getType(), 0, P->getName()+".reg2mem", AllocaPoint);
  } else {
    Function *F = P->getParent()->getParent();
    Slot = new AllocaInst(P->getType(), 0, P->getName()+".reg2mem",
                          F->getEntryBlock().begin());
  }
  
  // Iterate over each operand, insert store in each predecessor.
  for (unsigned i = 0, e = P->getNumIncomingValues(); i < e; ++i) {
    if (InvokeInst *II = dyn_cast<InvokeInst>(P->getIncomingValue(i))) {
      assert(II->getParent() != P->getIncomingBlock(i) && 
             "Invoke edge not supported yet"); II=II;
    }
    new StoreInst(P->getIncomingValue(i), Slot, 
                  P->getIncomingBlock(i)->getTerminator());
  }
  
  // Insert load in place of the phi and replace all uses.
  Value *V = new LoadInst(Slot, P->getName()+".reload", P);
  P->replaceAllUsesWith(V);
  
  // Delete phi.
  P->eraseFromParent();
  
  return Slot;
}
