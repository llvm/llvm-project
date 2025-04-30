//===-- Next32Subtarget.cpp - Next32 Subtarget Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Next32 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "Next32Subtarget.h"
#include "Next32.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

#define DEBUG_TYPE "next32-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "Next32GenSubtargetInfo.inc"

void Next32Subtarget::anchor() {}

Next32Subtarget &
Next32Subtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  initializeEnvironment();
  initSubtargetFeatures(CPU, FS);
  return *this;
}

void Next32Subtarget::initializeEnvironment() {
  HasLongShift = false;
  HasLEA = false;
  HasPrefetch = false;
  HasAtomicFAddFSub = false;
  IsGen1 = false;
  IsGen2 = false;
  HasVectorInst = false;
}

void Next32Subtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  StringRef CPUName = CPU;
  if (CPUName.empty())
    CPUName = "next32gen1";

  ParseSubtargetFeatures(CPUName, /*TuneCPU*/ CPUName, FS);

  if (IsGen1) {
    if (HasVectorInst)
      report_fatal_error("Vector instructions are not supported in GEN1.",
                         false);
    if (HasAtomicFAddFSub)
      report_fatal_error(
          "Atomic FAdd/FSub instructions are not supported in GEN1.", false);
  }
}

Next32Subtarget::Next32Subtarget(const Triple &TT, StringRef CPU, StringRef FS,
                                 const TargetMachine &TM)
    : Next32GenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), FrameLowering(*this),
      TLInfo(TM, *this) {}
