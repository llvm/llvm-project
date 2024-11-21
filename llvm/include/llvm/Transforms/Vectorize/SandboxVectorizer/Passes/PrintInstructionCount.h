#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PRINTINSTRUCTIONCOUNT_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PRINTINSTRUCTIONCOUNT_H

#include "llvm/SandboxIR/Pass.h"
#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

/// A Region pass that prints the instruction count for the region to stdout.
/// Used to test -sbvec-passes while we don't have any actual optimization
/// passes.
class PrintInstructionCount final : public RegionPass {
public:
  PrintInstructionCount() : RegionPass("null") {}
  bool runOnRegion(Region &R, const Analyses &A) final {
    outs() << "InstructionCount: " << std::distance(R.begin(), R.end()) << "\n";
    return false;
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PRINTINSTRUCTIONCOUNTPASS_H
