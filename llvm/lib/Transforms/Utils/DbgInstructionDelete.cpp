#include "llvm/Transforms/Utils/DbgInstructionDelete.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"


using namespace llvm;
PreservedAnalyses DbgInstructionDeletePass::run(Function& F, FunctionAnalysisManager& AM)
{
    for(BasicBlock& BB : F)
    {
        for(BasicBlock::iterator BB_iterator = BB.begin(), BB_end = BB.end(); BB_iterator != BB_end;)
        {
            Instruction& tmpI = *BB_iterator; 
            BB_iterator++;   
            if(isa<DbgInfoIntrinsic>(tmpI))
            {
                tmpI.eraseFromParent();
            }
        }
    }
    return PreservedAnalyses::none();
}