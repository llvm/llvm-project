#include "InArchSubtarget.h"
#include "InArch.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "InArch-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "InArchGenSubtargetInfo.inc"

void InArchSubtarget::anchor() {}

InArchSubtarget::InArchSubtarget(const Triple &TT, const std::string &CPU,
                             const std::string &FS, const TargetMachine &TM)
    : InArchGenSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS), InstrInfo(),
      FrameLowering(*this), TLInfo(TM, *this) {}