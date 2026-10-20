//===-- Toy64Subtarget.cpp - Toy64 Subtarget Information ------------------===//

#include "Toy64Subtarget.h"

#include "llvm/Support/Debug.h"

#include <string>

#define DEBUG_TYPE "toy64-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "Toy64GenSubtargetInfo.inc"

using namespace llvm;

Toy64Subtarget &
Toy64Subtarget::initializeSubtargetDependencies(
    StringRef CPU,
    StringRef FS) {

  std::string CPUName = CPU.str();

  if (CPUName.empty())
    CPUName = "generic";

  ParseSubtargetFeatures(
      CPUName,
      /*TuneCPU=*/CPUName,
      FS);

  return *this;
}

Toy64Subtarget::Toy64Subtarget(
    const Triple &TT,
    StringRef CPU,
    StringRef FS,
    const TargetMachine &TM)
    : Toy64GenSubtargetInfo(
          TT,
          CPU,
          /*TuneCPU=*/CPU,
          FS),
      RegInfo(),
      InstrInfo(*this, RegInfo),
      FrameLowering(),
      TLInfo(TM, *this) {

  initializeSubtargetDependencies(CPU, FS);
}