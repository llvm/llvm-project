/* --- PEVectorToIntrinsicPass.h --- */

/* ------------------------------------------
Author: 高宇翔
Date: 6/23/2025
------------------------------------------ */

#ifndef PEVECTORTOINTRINSICPASS_H
#define PEVECTORTOINTRINSICPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

struct PEVectorToIntrinsicPass : public PassInfoMixin<PEVectorToIntrinsicPass>{
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
};
}//end namespace llvm
#endif // PEVECTORTOINTRINSICPASS_H
