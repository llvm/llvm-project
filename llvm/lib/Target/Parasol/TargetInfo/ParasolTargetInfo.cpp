//===-- ParasolTargetInfo.cpp - Parasol Target Implementation -------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/ParasolTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheParasolTarget() {
  static Target TheParasolTarget;
  return TheParasolTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeParasolTargetInfo() {
  RegisterTarget<Triple::parasol> X(getTheParasolTarget(), "parasol", "Parasol",
                                    "Parasol");
}
