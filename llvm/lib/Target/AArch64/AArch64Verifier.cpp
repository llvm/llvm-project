//===- AArch64Verifier.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// IR verifier plugin for AArch64 intrinsics.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/Verifier.h"

using namespace llvm;

namespace {

#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      VS.CheckFailed(__VA_ARGS__);                                             \
      return;                                                                  \
    }                                                                          \
  } while (false)

class AArch64Verifier : public VerifierPlugin {
public:
  void verifyIntrinsicCall(CallBase &Call, VerifierSupport &VS) const override {
    switch (Call.getIntrinsicID()) {
    default:
      break;
    case Intrinsic::aarch64_ldaxr:
    case Intrinsic::aarch64_ldxr: {
      Type *ElemTy = Call.getParamElementType(0);
      Check(ElemTy,
            "Intrinsic requires elementtype attribute on first argument.",
            &Call);
      break;
    }
    case Intrinsic::aarch64_stlxr:
    case Intrinsic::aarch64_stxr: {
      Type *ElemTy = Call.getParamElementType(1);
      Check(ElemTy,
            "Intrinsic requires elementtype attribute on second argument.",
            &Call);
      break;
    }
    case Intrinsic::aarch64_prefetch: {
      Check(cast<ConstantInt>(Call.getArgOperand(1))->getZExtValue() < 2,
            "write argument to llvm.aarch64.prefetch must be 0 or 1", Call);
      Check(cast<ConstantInt>(Call.getArgOperand(2))->getZExtValue() < 4,
            "target argument to llvm.aarch64.prefetch must be 0-3", Call);
      Check(cast<ConstantInt>(Call.getArgOperand(3))->getZExtValue() < 2,
            "stream argument to llvm.aarch64.prefetch must be 0 or 1", Call);
      Check(cast<ConstantInt>(Call.getArgOperand(4))->getZExtValue() < 2,
            "isdata argument to llvm.aarch64.prefetch must be 0 or 1", Call);
      break;
    }
    }
  }
};

} // anonymous namespace

void llvm::initializeAArch64Verifier() { static AArch64Verifier Verifier; }
