#include "TargetInfo/InArchTargetInfo.h"
#include "InArch.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheInArchTarget() {
  static Target TheInArchTarget;
  return TheInArchTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeInArchTargetInfo() {
  RegisterTarget<Triple::inarch> X(getTheInArchTarget(), "inarch",
                                "InArch target for LLVM course", "INARCH");
}