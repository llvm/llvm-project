#include "llvm/Transforms/Utils/DebugRecordCounter.h"
#include "llvm/IR/Function.h" 
#include "llvm/IR/Module.h" 

using namespace llvm;

// build/bin/opt -disable-output /home/ana-marija/Documents/foo_00.ll -passes=dbg-counter
PreservedAnalyses DebugRecordCounterPass::run(Module &M, ModuleAnalysisManager &AM) {
    
    int dbg_values, dbg_declares, dbg_assigns;

    for (Function &F : M) { 
        errs() << "Function: " << F.getName() << "\n"; 

        dbg_assigns = dbg_declares = dbg_values = 0;

        for(BasicBlock &BB : F) {
            for(Instruction &I : BB) {
                for(DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
                    if(DVR.isDbgDeclare()) dbg_declares++;
                    if(DVR.isDbgValue())   dbg_values++;
                    if(DVR.isDbgAssign())  dbg_assigns++;
                }
            }
        }
        errs() << "\t#dbg_value:   " << dbg_values   << "\n";  
        errs() << "\t#dbg_declare: " << dbg_declares << "\n";  
        errs() << "\t#dbg_assign:  " << dbg_assigns  << "\n"; 
    }
    return PreservedAnalyses::all(); 
}
