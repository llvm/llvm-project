#include "TargetInfo/MyLoongArchTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

namespace llvm {

Target &getMyLoongArchTarget() {
  static Target MyLoongArchTarget;
  return MyLoongArchTarget;
}

} // namespace llvm

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMyLoongArchTargetInfo() {
  RegisterTarget<Triple::myloongarch> X(getMyLoongArchTarget(), "myloongarch",
                                        "32-bit LoongArch", "LoongArch");
}
