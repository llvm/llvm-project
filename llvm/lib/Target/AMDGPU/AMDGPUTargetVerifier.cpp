#include "llvm/Target/TargetVerify/AMDGPUTargetVerifier.h"

#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Check - We know that cond should be true, if not print an error message.
#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      TargetVerify::CheckFailed(__VA_ARGS__);                                  \
      return false;                                                                  \
    }                                                                          \
  } while (false)

namespace llvm {
/*class AMDGPUTargetVerify : public TargetVerify {
public:
  Module *Mod;

  DominatorTree *DT;
  PostDominatorTree *PDT;
  UniformityInfo *UA;

  AMDGPUTargetVerify(Module *Mod, DominatorTree *DT, PostDominatorTree *PDT, UniformityInfo *UA)
    : TargetVerify(Mod), Mod(Mod), DT(DT), PDT(PDT), UA(UA) {}

  void run(Function &F);
};*/

static bool IsValidInt(const Type *Ty) {
  return Ty->isIntegerTy(1) ||
         Ty->isIntegerTy(8) ||
         Ty->isIntegerTy(16) ||
         Ty->isIntegerTy(32) ||
         Ty->isIntegerTy(64) ||
         Ty->isIntegerTy(128);
}

static bool isShader(CallingConv::ID CC) {
  switch(CC) {
    case CallingConv::AMDGPU_VS:
    case CallingConv::AMDGPU_LS:
    case CallingConv::AMDGPU_HS:
    case CallingConv::AMDGPU_ES:
    case CallingConv::AMDGPU_GS:
    case CallingConv::AMDGPU_PS:
    case CallingConv::AMDGPU_CS_Chain:
    case CallingConv::AMDGPU_CS_ChainPreserve:
    case CallingConv::AMDGPU_CS:
      return true;
    default:
      return false;
  }
}

bool AMDGPUTargetVerify::run(Function &F) {
  // Ensure shader calling convention returns void
  if (isShader(F.getCallingConv()))
    Check(F.getReturnType() == Type::getVoidTy(F.getContext()), "Shaders must return void");

  for (auto &BB : F) {

    for (auto &I : BB) {

      // Ensure integral types are valid: i8, i16, i32, i64, i128
      if (I.getType()->isIntegerTy())
        Check(IsValidInt(I.getType()), "Int type is invalid.", &I);
      for (unsigned i = 0; i < I.getNumOperands(); ++i)
        if (I.getOperand(i)->getType()->isIntegerTy())
          Check(IsValidInt(I.getOperand(i)->getType()),
                "Int type is invalid.", I.getOperand(i));

      if (auto *CI = dyn_cast<CallInst>(&I))
      {
        // Ensure no kernel to kernel calls.
        CallingConv::ID CalleeCC = CI->getCallingConv();
        if (CalleeCC == CallingConv::AMDGPU_KERNEL)
        {
          CallingConv::ID CallerCC = CI->getParent()->getParent()->getCallingConv();
          Check(CallerCC != CallingConv::AMDGPU_KERNEL,
            "A kernel may not call a kernel", CI->getParent()->getParent());
        }

        // Ensure chain intrinsics are followed by unreachables.
        if (CI->getIntrinsicID() == Intrinsic::amdgcn_cs_chain)
          Check(isa_and_present<UnreachableInst>(CI->getNextNode()),
            "llvm.amdgcn.cs.chain must be followed by unreachable", CI);
      }
    }
  }

  if (!MessagesStr.str().empty())
    return false;
  return true;
}

PreservedAnalyses AMDGPUTargetVerifierPass::run(Function &F, FunctionAnalysisManager &AM) {

  auto *Mod = F.getParent();

  auto UA = &AM.getResult<UniformityInfoAnalysis>(F);
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *PDT = &AM.getResult<PostDominatorTreeAnalysis>(F);

  AMDGPUTargetVerify TV(Mod, DT, PDT, UA);
  TV.run(F);

  dbgs() << TV.MessagesStr.str();
  if (!TV.MessagesStr.str().empty()) {
    TV.IsValid = false;
    return PreservedAnalyses::none();
  }

  return PreservedAnalyses::all();
}

struct AMDGPUTargetVerifierLegacyPass : public FunctionPass {
  static char ID;

  std::unique_ptr<AMDGPUTargetVerify> TV;
  bool FatalErrors = true;

  AMDGPUTargetVerifierLegacyPass() : FunctionPass(ID) {
    initializeAMDGPUTargetVerifierLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  AMDGPUTargetVerifierLegacyPass(bool FatalErrors)
      : FunctionPass(ID),
        FatalErrors(FatalErrors) {
    initializeAMDGPUTargetVerifierLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool doInitialization(Module &M) override {
    TV = std::make_unique<AMDGPUTargetVerify>(&M);
    return false;
  }

  bool runOnFunction(Function &F) override {
    if (TV->run(F) && FatalErrors) {
      errs() << "in function " << F.getName() << '\n';
      report_fatal_error("Broken function found, compilation aborted!");
    }
    return false;
  }

  bool doFinalization(Module &M) override {
    bool IsValid = true;
    for (Function &F : M)
      if (F.isDeclaration())
        IsValid &= TV->run(F);

    //IsValid &= TV->run();
    if (FatalErrors && !IsValid)
      report_fatal_error("Broken module found, compilation aborted!");
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
char AMDGPUTargetVerifierLegacyPass::ID = 0;
} // namespace llvm
INITIALIZE_PASS(AMDGPUTargetVerifierLegacyPass, "amdgpu-tgtverify", "AMDGPU Target Verifier", false, false)
