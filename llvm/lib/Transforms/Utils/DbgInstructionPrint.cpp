#include "llvm/Transforms/Utils/DbgInstructionPrint.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

PreservedAnalyses DbgInstructionPrintPass::run(Function& F, FunctionAnalysisManager& AM)
{
    errs() << F.getName() <<":\n";
    int dbgVals = 0;
    int dbgDeclares = 0;
    int dbgAssigns = 0;
    for(const BasicBlock& BB: F)
    {
        for(const Instruction& I: BB)
        {
            if(isa<CallInst> (I))
            {
                
                // errs() << "\t" << cast<CallInst>(I).getCalledFunction()->getName() << "\n";
                StringRef ref = cast<CallInst>(I).getCalledFunction()->getName();
                if(ref == "llvm.dbg.value")
                {
                    dbgVals++;
                }
                else if(ref == "llvm.dbg.declare")
                {
                    dbgDeclares++;
                }
                else if(ref == "llvm.dbg.assign")
                {
                    dbgAssigns++;
                }
            }
        }
    }
    errs() << "\t" << "llvm.dbg.value: " << dbgVals << "\n";
    errs() << "\t" << "llvm.dbg.declare: " << dbgDeclares << "\n";
    errs() << "\t" << "llvm.dbg.assign: " << dbgAssigns << "\n";

    return PreservedAnalyses::all();
}
