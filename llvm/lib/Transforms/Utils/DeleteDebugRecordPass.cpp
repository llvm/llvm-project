#include "llvm/Transforms/Utils/DeleteDebugRecordPass.h"

#include "llvm/IR/Analysis.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

PreservedAnalyses DeleteDebugRecordPass::run(Function &F, FunctionAnalysisManager &) {
    
    bool Changed = false;

    for (auto &BB : F) {
        for (auto &I : BB) {
            SmallVector<DbgRecord *, 4> ToDelete;

            for( auto &D : I.getDbgRecordRange())
                if(isa<DbgVariableRecord>(&D))
                    ToDelete.push_back(&D);
            
            for (auto *I : ToDelete) {
                I->eraseFromParent();
                Changed = true;
            }
        }
    }

    

    return !Changed? PreservedAnalyses::all(): PreservedAnalyses::none();
}