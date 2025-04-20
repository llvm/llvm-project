#include "llvm/Target/TargetVerifier.h"
#include "llvm/Target/TargetVerify/AMDGPUTargetVerifier.h"

#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

namespace llvm {

void TargetVerify::run(Function &F, FunctionAnalysisManager &AM) {
  if (TT.isAMDGPU()) {
    auto *UA = &AM.getResult<UniformityInfoAnalysis>(F);
    auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
    auto *PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);

    AMDGPUTargetVerify TV(Mod, DT, PDT, UA);
    TV.run(F);

    dbgs() << TV.MessagesStr.str();
    if (!TV.MessagesStr.str().empty()) {
      TV.IsValid = false;
    }
  }
}

} // namespace llvm
