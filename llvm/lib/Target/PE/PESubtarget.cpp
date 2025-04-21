/* --- PESubtarget.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/2/2025
------------------------------------------ */

#include "PESubtarget.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#define DEBUG_TYPE "PE-Subtarget"

#include "PEGenSubtargetInfo.inc"
using namespace llvm;


PESubtarget::PESubtarget(const Triple &TT, StringRef CPU,
    StringRef FS, const TargetMachine &TM) 
    : PEGenSubtargetInfo(TT,CPU,CPU,FS),
    FrameLowering(*this),
    TLI(TM,*this){
}

PESubtarget &PESubtarget::initializeSubtargetDependencies(
    const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS,
    StringRef ABIName) {
    if (CPU.empty())
    CPU = "PE";

    ParseSubtargetFeatures(CPU, CPU, FS);

    return *this;
}
