//===- ARMVerifier.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// IR verifier plugin for ARM intrinsics.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsARM.h"
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

class ARMVerifier : public VerifierPlugin {
public:
  void verifyIntrinsicCall(CallBase &Call, VerifierSupport &VS) const override {
    switch (Call.getIntrinsicID()) {
    default:
      break;
    case Intrinsic::arm_ldaex:
    case Intrinsic::arm_ldrex: {
      Type *ElemTy = Call.getParamElementType(0);
      Check(ElemTy,
            "Intrinsic requires elementtype attribute on first argument.",
            &Call);
      break;
    }
    case Intrinsic::arm_stlex:
    case Intrinsic::arm_strex: {
      Type *ElemTy = Call.getParamElementType(1);
      Check(ElemTy,
            "Intrinsic requires elementtype attribute on second argument.",
            &Call);
      break;
    }
    }
  }
};

} // anonymous namespace

void llvm::initializeARMVerifier() { static ARMVerifier Verifier; }
