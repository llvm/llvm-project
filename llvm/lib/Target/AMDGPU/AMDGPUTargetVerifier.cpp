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

#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
MarkUniform("mark-uniform", cl::desc("Mark instructions as uniform"), cl::init(false));

// Check - We know that cond should be true, if not print an error message.
#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      TargetVerify::CheckFailed(__VA_ARGS__);                                  \
      return;                                                                  \
    }                                                                          \
  } while (false)

static bool isMFMA(unsigned IID) {
  switch (IID) {
    case Intrinsic::amdgcn_mfma_f32_4x4x1f32:
    case Intrinsic::amdgcn_mfma_f32_4x4x4f16:
    case Intrinsic::amdgcn_mfma_i32_4x4x4i8:
    case Intrinsic::amdgcn_mfma_f32_4x4x2bf16:

    case Intrinsic::amdgcn_mfma_f32_16x16x1f32:
    case Intrinsic::amdgcn_mfma_f32_16x16x4f32:
    case Intrinsic::amdgcn_mfma_f32_16x16x4f16:
    case Intrinsic::amdgcn_mfma_f32_16x16x16f16:
    case Intrinsic::amdgcn_mfma_i32_16x16x4i8:
    case Intrinsic::amdgcn_mfma_i32_16x16x16i8:
    case Intrinsic::amdgcn_mfma_f32_16x16x2bf16:
    case Intrinsic::amdgcn_mfma_f32_16x16x8bf16:

    case Intrinsic::amdgcn_mfma_f32_32x32x1f32:
    case Intrinsic::amdgcn_mfma_f32_32x32x2f32:
    case Intrinsic::amdgcn_mfma_f32_32x32x4f16:
    case Intrinsic::amdgcn_mfma_f32_32x32x8f16:
    case Intrinsic::amdgcn_mfma_i32_32x32x4i8:
    case Intrinsic::amdgcn_mfma_i32_32x32x8i8:
    case Intrinsic::amdgcn_mfma_f32_32x32x2bf16:
    case Intrinsic::amdgcn_mfma_f32_32x32x4bf16:

    case Intrinsic::amdgcn_mfma_f32_4x4x4bf16_1k:
    case Intrinsic::amdgcn_mfma_f32_16x16x4bf16_1k:
    case Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k:
    case Intrinsic::amdgcn_mfma_f32_32x32x4bf16_1k:
    case Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k:

    case Intrinsic::amdgcn_mfma_f64_16x16x4f64:
    case Intrinsic::amdgcn_mfma_f64_4x4x4f64:

    case Intrinsic::amdgcn_mfma_i32_16x16x32_i8:
    case Intrinsic::amdgcn_mfma_i32_32x32x16_i8:
    case Intrinsic::amdgcn_mfma_f32_16x16x8_xf32:
    case Intrinsic::amdgcn_mfma_f32_32x32x4_xf32:

    case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8:
    case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8:
    case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8:
    case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8:

    case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8:
    case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8:
    case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8:
    case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8:
      return true;
    default:
      return false;
  }
}

namespace llvm {
class AMDGPUTargetVerify : public TargetVerify {
public:
  Module *Mod;

  DominatorTree *DT;
  PostDominatorTree *PDT;
  UniformityInfo *UA;

  AMDGPUTargetVerify(Module *Mod, DominatorTree *DT, PostDominatorTree *PDT, UniformityInfo *UA)
    : TargetVerify(Mod), Mod(Mod), DT(DT), PDT(PDT), UA(UA) {}

  void run(Function &F);
};

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

void AMDGPUTargetVerify::run(Function &F) {
  // Ensure shader calling convention returns void
  if (isShader(F.getCallingConv()))
    Check(F.getReturnType() == Type::getVoidTy(F.getContext()), "Shaders must return void");

  for (auto &BB : F) {

    for (auto &I : BB) {
      if (MarkUniform)
        outs() << UA->isUniform(&I) << ' ' << I << '\n';

      // Ensure integral types are valid: i8, i16, i32, i64, i128
      if (I.getType()->isIntegerTy())
        Check(IsValidInt(I.getType()), "Int type is invalid.", &I);
      for (unsigned i = 0; i < I.getNumOperands(); ++i)
        if (I.getOperand(i)->getType()->isIntegerTy())
          Check(IsValidInt(I.getOperand(i)->getType()),
                "Int type is invalid.", I.getOperand(i));

      // Ensure alloca array size is constant
      if (auto *AI = dyn_cast<AllocaInst>(&I))
      {
        auto *AS = AI->getArraySize();
        Check(!isa<Constant>(AS), "Dynamically-sized alloca disallowed");
      }

      // Ensure no store to const memory
      if (auto *SI = dyn_cast<StoreInst>(&I))
      {
        unsigned AS = SI->getPointerAddressSpace();
        Check(AS != 4, "Write to const memory", SI);
      }

      // Ensure no kernel to kernel calls.
      if (auto *CI = dyn_cast<CallInst>(&I))
      {
        CallingConv::ID CalleeCC = CI->getCallingConv();
        if (CalleeCC == CallingConv::AMDGPU_KERNEL)
        {
          CallingConv::ID CallerCC = CI->getParent()->getParent()->getCallingConv();
          Check(CallerCC != CallingConv::AMDGPU_KERNEL,
            "A kernel may not call a kernel", CI->getParent()->getParent());
        }
      }

      // Ensure MFMA is not in control flow with diverging operands
      if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
        if (isMFMA(II->getIntrinsicID())) {
          bool InControlFlow = false;
          for (const auto &P : predecessors(&BB))
            if (!PDT->dominates(&BB, P)) {
              InControlFlow = true;
              break;
            }
          for (const auto &S : successors(&BB))
            if (!DT->dominates(&BB, S)) {
              InControlFlow = true;
              break;
            }
          if (InControlFlow) {
            // If operands to MFMA are not uniform, MFMA cannot be in control flow
            bool hasUniformOperands = true;
            for (unsigned i = 0; i < II->getNumOperands(); i++) {
              if (!UA->isUniform(II->getOperand(i))) {
                dbgs() << "Not uniform: " << *II->getOperand(i) << '\n';
                hasUniformOperands = false;
              }
            }
            if (!hasUniformOperands) Check(false, "MFMA in control flow", II);
            //else Check(false, "MFMA in control flow (uniform operands)", II);
          }
          //else Check(false, "MFMA not in control flow", II);
        }
      }
    }
  }
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
    F.getParent()->IsValid = false;
  }

  return PreservedAnalyses::all();
}
} // namespace llvm
