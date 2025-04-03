/* --- PESubtarget.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/2/2025
------------------------------------------ */

#include "PESubtarget.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "PEGenSubtargetInfo.inc"

#define DEBUG_TYPE "PE-Subtarget"
using namespace llvm;


PESubtarget::PESubtarget(const Triple &TT, StringRef CPU,
    StringRef FS, const TargetMachine &TM) : PEGenSubtargetInfo(TT,CPU, CPU,FS),FrameLowering(*this){
}

PESubtarget::~PESubtarget() {
    // Destructor
}

PESubtarget &PESubtarget::initializeSubtargetDependencies(
    const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS,
    StringRef ABIName) {
    if (CPU.empty() || CPU == "generic")
    CPU = "PE";

    ParseSubtargetFeatures(CPU, CPU, FS);

    return *this;
}
