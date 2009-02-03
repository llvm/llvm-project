//===- Reg2Mem.cpp - Convert registers to allocas -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file demotes all registers to memory references.  It is intented to be
// the inverse of PromoteMemoryToRegister.  By converting to loads, the only
// values live accross basic blocks are allocas and loads before phi nodes.
// It is intended that this should make CFG hacking much easier.
// To make later hacking easier, the entry block is split into two, such that
// all introduced allocas and nothing else are in the entry block.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg2mem"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CFG.h"
#include <list>
using namespace llvm;

STATISTIC(NumRegsDemoted, "Number of registers demoted");
STATISTIC(NumPhisDemoted, "Number of phi-nodes demoted");

namespace {
  struct VISIBILITY_HIDDEN RegToMem : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    RegToMem() : FunctionPass(&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addPreservedID(BreakCriticalEdgesID);
    }

   bool valueEscapes(Instruction* i) {
      BasicBlock* bb = i->getParent();
      for (Value::use_iterator ii = i->use_begin(), ie = i->use_end();
           ii != ie; ++ii)
        if (cast<Instruction>(*ii)->getParent() != bb ||
            isa<PHINode>(*ii))
          return true;
      return false;
    }

    virtual bool runOnFunction(Function &F) {
      if (!F.isDeclaration()) {
        // Insert all new allocas into entry block.
        BasicBlock* BBEntry = &F.getEntryBlock();
        assert(pred_begin(BBEntry) == pred_end(BBEntry) &&
               "Entry block to function must not have predecessors!");

        // Find first non-alloca instruction and create insertion point. This is
        // safe if block is well-formed: it always have terminator, otherwise
        // we'll get and assertion.
        BasicBlock::iterator I = BBEntry->begin();
        while (isa<AllocaInst>(I)) ++I;

        CastInst *AllocaInsertionPoint =
          CastInst::Create(Instruction::BitCast,
                           Constant::getNullValue(Type::Int32Ty), Type::Int32Ty,
                           "reg2mem alloca point", I);

        // Find the escaped instructions. But don't create stack slots for
        // allocas in entry block.
        std::list<Instruction*> worklist;
        for (Function::iterator ibb = F.begin(), ibe = F.end();
             ibb != ibe; ++ibb)
          for (BasicBlock::iterator iib = ibb->begin(), iie = ibb->end();
               iib != iie; ++iib) {
            if (!(isa<AllocaInst>(iib) && iib->getParent() == BBEntry) &&
                valueEscapes(iib)) {
              worklist.push_front(&*iib);
            }
          }

        // Demote escaped instructions
        NumRegsDemoted += worklist.size();
        for (std::list<Instruction*>::iterator ilb = worklist.begin(), 
               ile = worklist.end(); ilb != ile; ++ilb)
          DemoteRegToStack(**ilb, false, AllocaInsertionPoint);

        worklist.clear();

        // Find all phi's
        for (Function::iterator ibb = F.begin(), ibe = F.end();
             ibb != ibe; ++ibb)
          for (BasicBlock::iterator iib = ibb->begin(), iie = ibb->end();
               iib != iie; ++iib)
            if (isa<PHINode>(iib))
              worklist.push_front(&*iib);

        // Demote phi nodes
        NumPhisDemoted += worklist.size();
        for (std::list<Instruction*>::iterator ilb = worklist.begin(), 
               ile = worklist.end(); ilb != ile; ++ilb)
          DemotePHIToStack(cast<PHINode>(*ilb), AllocaInsertionPoint);

        return true;
      }
      return false;
    }
  };
}
  
char RegToMem::ID = 0;
static RegisterPass<RegToMem>
X("reg2mem", "Demote all values to stack slots");

// createDemoteRegisterToMemory - Provide an entry point to create this pass.
//
const PassInfo *const llvm::DemoteRegisterToMemoryID = &X;
FunctionPass *llvm::createDemoteRegisterToMemoryPass() {
  return new RegToMem();
}
