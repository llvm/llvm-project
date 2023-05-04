#ifndef LLVM_TRANSFORMS_HELLONEW_MYHELLOWORLD_H
#define LLVM_TRANSFORMS_HELLONEW_MYHELLOWORLD_H

#include "llvm/IR/PassManager.h"

namespace llvm {

    class MyHelloWorldPass : public PassInfoMixin<MyHelloWorldPass> {
        public:
        PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
    };
}

#endif