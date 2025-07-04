#include "llvm/Target/RegisterTargetPassConfigCallback.h"

namespace llvm {
// TargetPassConfig callbacks
static SmallVector<RegisterTargetPassConfigCallback *, 1>
    TargetPassConfigCallbacks{};

void invokeGlobalTargetPassConfigCallbacks(TargetMachine &TM,
                                           PassManagerBase &PM,
                                           TargetPassConfig *PassConfig) {
  for (const RegisterTargetPassConfigCallback *Reg : TargetPassConfigCallbacks)
    Reg->Callback(TM, PM, PassConfig);
}

RegisterTargetPassConfigCallback::RegisterTargetPassConfigCallback(
    PassConfigCallback &&C)
    : Callback(std::move(C)) {
  TargetPassConfigCallbacks.push_back(this);
}

RegisterTargetPassConfigCallback::~RegisterTargetPassConfigCallback() {
  const auto &It = find(TargetPassConfigCallbacks, this);
  if (It != TargetPassConfigCallbacks.end())
    TargetPassConfigCallbacks.erase(It);
}
} // namespace llvm
