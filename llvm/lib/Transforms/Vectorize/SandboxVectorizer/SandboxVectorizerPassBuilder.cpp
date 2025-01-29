#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/BottomUpVec.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/NullPass.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/PrintInstructionCount.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/RegionsFromMetadata.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/SeedSelection.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/TransactionAcceptOrRevert.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/TransactionAlwaysAccept.h"

namespace llvm::sandboxir {

std::unique_ptr<sandboxir::RegionPass>
SandboxVectorizerPassBuilder::createRegionPass(StringRef Name, StringRef Args) {
#define REGION_PASS(NAME, CLASS_NAME)                                          \
  if (Name == NAME) {                                                          \
    assert(Args.empty() && "Unexpected arguments for pass '" NAME "'.");       \
    return std::make_unique<CLASS_NAME>();                                     \
  }
// TODO: Support region passes with params.
#include "Passes/PassRegistry.def"
  return nullptr;
}

std::unique_ptr<sandboxir::FunctionPass>
SandboxVectorizerPassBuilder::createFunctionPass(StringRef Name,
                                                 StringRef Args) {
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS_NAME)                            \
  if (Name == NAME)                                                            \
    return std::make_unique<CLASS_NAME>(Args);
#include "Passes/PassRegistry.def"
  return nullptr;
}

} // namespace llvm::sandboxir
