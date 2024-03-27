#include "TargetInfo/InArchTargetInfo.h"
#include "InArch.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheInArchTarget() {
  INARCH_DUMP_YELLOW
  static Target TheInArchTarget;
  return TheInArchTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeInArchTargetInfo() {
  INARCH_DUMP_YELLOW
  RegisterTarget<Triple::inarch> X(getTheInArchTarget(), "inarch",
                                "InArch target for LLVM course", "INARCH");
}