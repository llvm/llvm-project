//===-- Toy64InstrInfo.cpp - Toy64 Instruction Information ----------------===//

#include "Toy64InstrInfo.h"
#include "Toy64Subtarget.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "Toy64GenInstrInfo.inc"

using namespace llvm;

Toy64InstrInfo::Toy64InstrInfo(
    const Toy64Subtarget &STI,
    const Toy64RegisterInfo &RI)
    : Toy64GenInstrInfo(
          STI,
          RI,
          Toy64::ADJCALLSTACKDOWN,
          Toy64::ADJCALLSTACKUP) {}