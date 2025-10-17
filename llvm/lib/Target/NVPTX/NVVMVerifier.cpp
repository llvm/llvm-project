//===- NVVMVerifier.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// IR verifier plugin for NVVM intrinsics.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
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

class NVVMVerifier : public VerifierPlugin {
public:
  void verifyIntrinsicCall(CallBase &Call, VerifierSupport &VS) const override {
    switch (Call.getIntrinsicID()) {
    default:
      break;
    case Intrinsic::nvvm_setmaxnreg_inc_sync_aligned_u32:
    case Intrinsic::nvvm_setmaxnreg_dec_sync_aligned_u32: {
      Value *V = Call.getArgOperand(0);
      unsigned RegCount = cast<ConstantInt>(V)->getZExtValue();
      Check(RegCount % 8 == 0,
            "reg_count argument to nvvm.setmaxnreg must be in multiples of 8");
      break;
    }
    }
  }
};

} // anonymous namespace

void llvm::initializeNVVMVerifier() { static NVVMVerifier Verifier; }
