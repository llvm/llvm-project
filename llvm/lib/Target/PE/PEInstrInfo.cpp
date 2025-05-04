/* --- PEInstrInfo.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/1/2025
------------------------------------------ */

#include "PEInstrInfo.h"
#include "MCTargetDesc/PEMCTargetDesc.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "PEGenInstrInfo.inc"

PEInstrInfo::PEInstrInfo() : PEGenInstrInfo(){
    // Constructor
}

PEInstrInfo::~PEInstrInfo() {
    // Destructor
}
