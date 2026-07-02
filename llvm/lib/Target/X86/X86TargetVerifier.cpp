//===-- X86TargetVerifier.cpp - X86 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the X86 implementation of the target-dependent verifier. It is the
// "TargetVerify" half of the verification the (target-independent)
// TargetVerifierPass runs for X86 modules; the other half is the generic IR
// verifier.
//
// The checks here are the ones that are *subtarget/feature-dependent*:
// they need to know the specific CPU/features selected via the
// target-cpu/target-features function attributes. The MCSubtargetInfo is
// derived from those attributes, so no TargetMachine is required and the
// verifier can run from generic pipelines.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetVerifier.h"

using namespace llvm;

// Check - We know that cond should be true, if not print an error message.
#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C))                                                                  \
      checkFailed(__VA_ARGS__);                                                \
  } while (false)

namespace {

class X86TargetVerify : public TargetVerify {
public:
  X86TargetVerify(Module *Mod) : TargetVerify(Mod) {}
  bool run(Function &F) override;

private:
  /// Per-function subtarget/feature-dependent checks.
  void verifyFunctionChecks(Function &F, const MCSubtargetInfo &STI);

  MCSubtargetInfo *getSubtargetInfo(const Function &F) const;
};

static bool usesAMXType(const Instruction &I) {
  if (I.getType()->isX86_AMXTy())
    return true;
  for (const Value *Op : I.operands())
    if (Op->getType()->isX86_AMXTy())
      return true;
  return false;
}

MCSubtargetInfo *X86TargetVerify::getSubtargetInfo(const Function &F) const {
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T)
    return nullptr;
  StringRef CPU = F.getFnAttribute("target-cpu").getValueAsString();
  StringRef FS = F.getFnAttribute("target-features").getValueAsString();
  return T->createMCSubtargetInfo(TT, CPU, FS);
}

/// Instruction-set intrinsic families, each gated on a subtarget feature. The
/// name prefixes are disjoint (the trailing '.' keeps e.g. "avx" from matching
/// "avx2"/"avx512"), so an intrinsic matches at most one entry.
static const struct {
  StringRef Prefix;
  unsigned Feature;
  StringRef Name;
} IntrinsicFeatures[] = {
    {"llvm.x86.avx512.", X86::FeatureAVX512, "AVX-512"},
    {"llvm.x86.avx2.", X86::FeatureAVX2, "AVX2"},
    {"llvm.x86.avx.", X86::FeatureAVX, "AVX"},
};

/// \returns true if the inline-asm constraint string \p C names an AVX-512
/// mask register, i.e. a "{k0}"..."{k7}" token.
static bool referencesMaskReg(StringRef C) {
  for (size_t Pos = C.find("{k"); Pos != StringRef::npos;
       Pos = C.find("{k", Pos + 2)) {
    if (Pos + 3 < C.size() && C[Pos + 2] >= '0' && C[Pos + 2] <= '7' &&
        C[Pos + 3] == '}')
      return true;
  }
  return false;
}

void X86TargetVerify::verifyFunctionChecks(Function &F,
                                           const MCSubtargetInfo &STI) {
  bool HasAMXTILE = STI.hasFeature(X86::FeatureAMXTILE);

  for (const Instruction &I : instructions(F)) {
    // An instruction-set intrinsic requires its feature on the selected
    // subtarget.
    if (const auto *CB = dyn_cast<CallBase>(&I)) {
      if (const Function *CF = CB->getCalledFunction()) {
        StringRef Name = CF->getName();
        for (const auto &IF : IntrinsicFeatures)
          if (Name.starts_with(IF.Prefix))
            Check(STI.hasFeature(IF.Feature),
                  Twine(IF.Name) +
                      " intrinsic used, but the subtarget does not support " +
                      IF.Name + ".",
                  &I);

        // The 128/256-bit (EVEX.128/256) forms of an AVX-512 intrinsic
        // additionally require AVX512VL, which AVX512F alone does not provide.
        // The width is encoded as a ".128"/".256" suffix on the intrinsic name.
        if (Name.starts_with("llvm.x86.avx512.") &&
            (Name.ends_with(".128") || Name.ends_with(".256")))
          Check(STI.hasFeature(X86::FeatureVLX),
                "128/256-bit AVX-512 intrinsic used, but the subtarget does "
                "not support AVX512VL.",
                &I);
      }
    }

    // The x86_amx type requires the AMX-TILE feature on the selected
    // subtarget.
    if (usesAMXType(I))
      Check(HasAMXTILE,
            "x86_amx type used, but the subtarget does not support AMX-TILE.",
            &I);

    // Inline asm may name physical registers (in clobbers or register
    // constraints) that only exist on subtargets with a given feature. zmm and
    // mask (k) registers require AVX-512; ymm registers require AVX.
    if (const auto *CB = dyn_cast<CallBase>(&I)) {
      if (CB->isInlineAsm()) {
        StringRef Constraints =
            cast<InlineAsm>(CB->getCalledOperand())->getConstraintString();
        if (Constraints.contains("zmm") || referencesMaskReg(Constraints))
          Check(STI.hasFeature(X86::FeatureAVX512),
                "inline asm references an AVX-512 register (zmm/k), but the "
                "subtarget does not support AVX-512.",
                &I);
        if (Constraints.contains("ymm"))
          Check(STI.hasFeature(X86::FeatureAVX),
                "inline asm references an AVX register (ymm), but the "
                "subtarget does not support AVX.",
                &I);
      }
    }
  }
}

bool X86TargetVerify::run(Function &F) {
  IsValid = true;

  MCSubtargetInfo *STI = getSubtargetInfo(F);
  if (!STI)
    return IsValid;

  verifyFunctionChecks(F, *STI);

  delete STI;
  return IsValid;
}

} // anonymous namespace

namespace llvm {
TargetVerify *createX86TargetVerify(Module &M) {
  return new X86TargetVerify(&M);
}
} // namespace llvm
