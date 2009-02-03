//===- CloneTrace.cpp - Clone a trace -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CloneTrace interface, which is used when writing
// runtime optimizations. It takes a vector of basic blocks clones the basic
// blocks, removes internal phi nodes, adds it to the same function as the
// original (although there is no jump to it) and returns the new vector of
// basic blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Trace.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

//Clones the trace (a vector of basic blocks)
std::vector<BasicBlock *>
llvm::CloneTrace(const std::vector<BasicBlock*> &origTrace) {
  std::vector<BasicBlock *> clonedTrace;
  DenseMap<const Value*, Value*> ValueMap;

  //First, loop over all the Basic Blocks in the trace and copy
  //them using CloneBasicBlock. Also fix the phi nodes during
  //this loop. To fix the phi nodes, we delete incoming branches
  //that are not in the trace.
  for (std::vector<BasicBlock *>::const_iterator T = origTrace.begin(),
    End = origTrace.end(); T != End; ++T) {

    //Clone Basic Block
    BasicBlock *clonedBlock =
      CloneBasicBlock(*T, ValueMap, ".tr", (*T)->getParent());

    //Add it to our new trace
    clonedTrace.push_back(clonedBlock);

    //Add this new mapping to our Value Map
    ValueMap[*T] = clonedBlock;

    //Loop over the phi instructions and delete operands
    //that are from blocks not in the trace
    //only do this if we are NOT the first block
    if (T != origTrace.begin()) {
      for (BasicBlock::iterator I = clonedBlock->begin();
           isa<PHINode>(I); ++I) {
        PHINode *PN = cast<PHINode>(I);
        //get incoming value for the previous BB
        Value *V = PN->getIncomingValueForBlock(*(T-1));
        assert(V && "No incoming value from a BasicBlock in our trace!");

        //remap our phi node to point to incoming value
        ValueMap[*&I] = V;

        //remove phi node
        clonedBlock->getInstList().erase(PN);
      }
    }
  }

  //Second loop to do the remapping
  for (std::vector<BasicBlock *>::const_iterator BB = clonedTrace.begin(),
    BE = clonedTrace.end(); BB != BE; ++BB) {
    for (BasicBlock::iterator I = (*BB)->begin(); I != (*BB)->end(); ++I) {
      //Loop over all the operands of the instruction
      for (unsigned op=0, E = I->getNumOperands(); op != E; ++op) {
        const Value *Op = I->getOperand(op);

        //Get it out of the value map
        Value *V = ValueMap[Op];

        //If not in the value map, then its outside our trace so ignore
        if (V != 0)
          I->setOperand(op,V);
      }
    }
  }

  //return new vector of basic blocks
  return clonedTrace;
}

/// CloneTraceInto - Clone T into NewFunc. Original<->clone mapping is
/// saved in ValueMap.
///
void llvm::CloneTraceInto(Function *NewFunc, Trace &T,
                          DenseMap<const Value*, Value*> &ValueMap,
                          const char *NameSuffix) {
  assert(NameSuffix && "NameSuffix cannot be null!");

  // Loop over all of the basic blocks in the trace, cloning them as
  // appropriate.
  //
  for (Trace::const_iterator BI = T.begin(), BE = T.end(); BI != BE; ++BI) {
    const BasicBlock *BB = *BI;

    // Create a new basic block and copy instructions into it!
    BasicBlock *CBB = CloneBasicBlock(BB, ValueMap, NameSuffix, NewFunc);
    ValueMap[BB] = CBB;                       // Add basic block mapping.
  }

  // Loop over all of the instructions in the new function, fixing up operand
  // references as we go.  This uses ValueMap to do all the hard work.
  //
  for (Function::iterator BB =
         cast<BasicBlock>(ValueMap[T.getEntryBasicBlock()]),
         BE = NewFunc->end(); BB != BE; ++BB)
    // Loop over all instructions, fixing each one as we find it...
    for (BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II)
      RemapInstruction(II, ValueMap);
}

