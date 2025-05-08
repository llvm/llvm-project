//===----------------------------------------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#ifndef FHE_CIRCUIT_FIXUP
#define FHE_CIRCUIT_FIXUP

#include "llvm/IR/PassManager.h"

namespace llvm {

class FheCircuitFixupPass : public PassInfoMixin<FheCircuitFixupPass> {
public:
    PreservedAnalyses run(Function&F, FunctionAnalysisManager &M);

    static bool is_required() { return true; }
};
}

#endif
