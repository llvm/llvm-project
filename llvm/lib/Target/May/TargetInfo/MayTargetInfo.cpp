#include "TargetInfo/MayTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
using namespace llvm;

Target &llvm::getTheMayTarget() {
  static Target TheMayTarget;
  return TheMayTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMayTargetInfo() {
  RegisterTarget<Triple::may> X(getTheMayTarget(), "may", "May Instruction Set",
                                "May");
}