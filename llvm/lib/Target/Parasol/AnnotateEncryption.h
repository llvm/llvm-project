//===----------------------------------------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANNOTATE_ENCRYPTION_H
#define LLVM_ANNOTATE_ENCRYPTION_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class AnnotateEncryptionPass : public PassInfoMixin<AnnotateEncryptionPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
};
} // namespace llvm

#endif
