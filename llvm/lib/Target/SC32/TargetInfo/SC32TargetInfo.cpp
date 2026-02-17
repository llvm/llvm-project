#include "TargetInfo/SC32TargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

Target &llvm::getTheSC32Target() {
  static Target TheSC32Target;
  return TheSC32Target;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeSC32TargetInfo() {
  RegisterTarget<Triple::sc32>{getTheSC32Target(), "sc32", "SanicCPU32",
                               "SanicCPU32"};
}
