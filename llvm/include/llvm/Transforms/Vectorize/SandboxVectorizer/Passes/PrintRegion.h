#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PRINTREGION_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PRINTREGION_H

#include "llvm/SandboxIR/Pass.h"
#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

/// A Region pass that does nothing, for use as a placeholder in tests.
class PrintRegion final : public RegionPass {
public:
  PrintRegion() : RegionPass("print-region") {}
  bool runOnRegion(Region &R, const Analyses &A) final {
    raw_ostream &OS = outs();
#ifndef NDEBUG
    OS << "-- Region --\n";
    OS << R << "\n";
#else
    // TODO: Make this available in all builds, depends on enabling SandboxIR
    // dumps in non-debug builds.
    OS << "Region dump only available in DEBUG build!";
#endif
    return false;
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PRINTREGION_H
