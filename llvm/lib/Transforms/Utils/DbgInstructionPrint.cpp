#include "llvm/Transforms/Utils/DbgInstructionPrint.h"
#include "llvm/IR/Instructions.h"
#include <map>
#include "llvm/Support/Regex.h"

using namespace llvm;

PreservedAnalyses DbgInstructionPrintPass::run(Function& F, FunctionAnalysisManager& AM)
{
    errs() << F.getName() <<":\n";
    std::map<StringRef, int> dbgCntMap;
    Regex dbgRegex("^llvm.dbg.*");
    for(const BasicBlock& BB: F)
    {
        for(const Instruction& I: BB)
        {
            if(isa<CallInst> (I))
            {
                StringRef functionName = cast<CallInst>(I).getCalledFunction()->getName();
                if(dbgRegex.match(functionName))
                {
                    if(dbgCntMap.count(functionName) > 0)
                    {
                        dbgCntMap[functionName]++;
                    }
                    else
                    {
                        dbgCntMap[functionName] = 1;
                    }
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
