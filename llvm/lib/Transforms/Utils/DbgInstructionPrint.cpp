#include "llvm/Transforms/Utils/DbgInstructionPrint.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include <map>
#include "llvm/Support/Regex.h"

using namespace llvm;

PreservedAnalyses DbgInstructionPrintPass::run(Function& F, FunctionAnalysisManager& AM)
{
    errs() << F.getName() <<":\n";
    std::map<StringRef, int> dbgCntMap;

    for(const BasicBlock& BB: F)
    {
        for(const Instruction& I: BB)
        {
            if(isa<DbgInfoIntrinsic> (I))
            {
                StringRef dbgInstructionName = cast<DbgInfoIntrinsic>(I).getCalledFunction()->getName();
                if(dbgCntMap.count(dbgInstructionName) > 0)
                {
                    dbgCntMap[dbgInstructionName]++;
                }
                else
                {
                        dbgCntMap[dbgInstructionName] = 1;
                }
            }
        }
    }
    //Onih kojih nema, njih je nula
    for(auto it = dbgCntMap.cbegin(); it != dbgCntMap.cend(); it++)
    {
        errs() << "\t" << it->first << ": " << it->second << "\n";
        
    }

    return PreservedAnalyses::all();
}
