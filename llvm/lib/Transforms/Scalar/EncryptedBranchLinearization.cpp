//===----------------------------------------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/EncryptedBranchLinearization.h"
#include "llvm/Analysis/EncryptionColoring.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

PreservedAnalyses EncryptedBranchLinearizationPass::run(Function &F, FunctionAnalysisManager &M) {
    auto encryptionColoring = M.getResult<EncryptionColoringAnalysis>(F);

    for (const auto &inst : instructions(F)) {
        auto color = encryptionColoring.getColor(&inst);

        errs() << color << "\n";
    }

    return PreservedAnalyses::none();
}
