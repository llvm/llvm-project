#include "TargetInfo/Cpu0TargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
using namespace llvm;

Target &llvm::getTheCpu0Target() {
    static Target TheCpu0Target;
    return TheCpu0Target;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCpu0TargetInfo() {
    RegisterTarget<Triple::cpu0,
                    /*HasJIT=*/true>
        X(getTheCpu0Target(), "cpu0", "CPU0 (32-bit big endian)", "Cpu0");
}