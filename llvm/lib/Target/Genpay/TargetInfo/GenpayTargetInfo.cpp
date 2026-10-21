#include "GenpayTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

Target &llvm::getTheGenpayTarget() {
  static Target TheGenpayTarget;
  return TheGenpayTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeGenpayTargetInfo() {
  RegisterTarget<Triple::genpay, /*HasJIT=*/true> X(getTheGenpayTarget(),
                                                    "genpay",
                                                    "Genpay 64-bit custom ISA",
                                                    "Genpay");
}
