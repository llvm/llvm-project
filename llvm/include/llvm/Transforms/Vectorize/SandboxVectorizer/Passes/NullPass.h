#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_NULLPASS_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_NULLPASS_H

#include "llvm/SandboxIR/Pass.h"

namespace llvm::sandboxir {

class Region;

/// A Region pass that does nothing, for use as a placeholder in tests.
class NullPass final : public RegionPass {
public:
  NullPass() : RegionPass("null") {}
  bool runOnRegion(Region &R, const Analyses &A) final { return false; }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_NULLPASS_H
