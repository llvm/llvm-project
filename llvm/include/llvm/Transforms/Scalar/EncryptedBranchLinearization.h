//===----------------------------------------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#ifndef ENCRYPTED_BRANCH_LINEARIZATION_H
#define ENCRYPTED_BRANCH_LINEARIZATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class EncryptedBranchLinearizationPass : public PassInfoMixin<EncryptedBranchLinearizationPass> {
public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &M);

    static bool is_required() { return true; }
};
}

#endif
