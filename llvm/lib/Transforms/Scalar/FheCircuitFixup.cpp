//===----------------------------------------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/FheCircuitFixup.h"

using namespace llvm;

PreservedAnalyses FheCircuitFixupPass::run(Function &F, FunctionAnalysisManager &M) {
    if (!F.hasFnAttribute(Attribute::FheCircuit)) {
        return PreservedAnalyses::all();
    }

    for (auto &arg : F.args()) {
        auto ty = arg.getType();
        if (!ty->isPointerTy()) {
            
        }
    }

    return PreservedAnalyses::none();
}
