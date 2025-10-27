#include "llvm/Transforms/Utils/DebugRecordDeleter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h" 
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"

using namespace llvm;

// build/bin/opt -S /home/ana-marija/Documents/foo_00.ll -passes=dbg-deleter
PreservedAnalyses DebugRecordDeleterPass::run(Module &M, ModuleAnalysisManager &AM) {
    
    bool modified = StripDebugInfo(M); 

    return PreservedAnalyses::all();
         
}