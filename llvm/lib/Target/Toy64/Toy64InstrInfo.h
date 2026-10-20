//===-- Toy64InstrInfo.h - Toy64 Instruction Information ------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_TOY64INSTRINFO_H
#define LLVM_LIB_TARGET_TOY64_TOY64INSTRINFO_H

#include "Toy64RegisterInfo.h"
#include "MCTargetDesc/Toy64MCTargetDesc.h"

#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "Toy64GenInstrInfo.inc"

namespace llvm {

class Toy64Subtarget;

class Toy64InstrInfo : public Toy64GenInstrInfo {
public:
  Toy64InstrInfo(
      const Toy64Subtarget &STI,
      const Toy64RegisterInfo &RI);
};

} // namespace llvm

#endif