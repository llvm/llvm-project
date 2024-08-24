#include "TargetInfo/MyLoongArchTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

namespace llvm  {

Target &getMyLoongArchTarget() {
  static Target MyLoongArchTarget;
  return MyLoongArchTarget;
}

}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMyLoongArchTargetInfo() {
  RegisterTarget<Triple::myloongarch> A(getMyLoongArchTarget(), "myloongarch", "64-bit LoongArch", "LoongArch");
}
