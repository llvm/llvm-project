//===-- Toy64Subtarget.h - Define Subtarget for Toy64 ----------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_TOY64SUBTARGET_H
#define LLVM_LIB_TARGET_TOY64_TOY64SUBTARGET_H

#include "Toy64FrameLowering.h"
#include "Toy64InstrInfo.h"
#include "Toy64ISelLowering.h"
#include "Toy64RegisterInfo.h"

#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "Toy64GenSubtargetInfo.inc"

namespace llvm {

class Toy64Subtarget : public Toy64GenSubtargetInfo {
  Toy64RegisterInfo RegInfo;
  Toy64InstrInfo InstrInfo;
  Toy64FrameLowering FrameLowering;
  Toy64TargetLowering TLInfo;

  SelectionDAGTargetInfo TSInfo;

public:
  Toy64Subtarget(
      const Triple &TT,
      StringRef CPU,
      StringRef FS,
      const TargetMachine &TM);

  Toy64Subtarget &
  initializeSubtargetDependencies(
      StringRef CPU,
      StringRef FS);

  void ParseSubtargetFeatures(
      StringRef CPU,
      StringRef TuneCPU,
      StringRef FS);

  const Toy64RegisterInfo *
  getRegisterInfo() const override {
    return &RegInfo;
  }

  const Toy64InstrInfo *
  getInstrInfo() const override {
    return &InstrInfo;
  }

  const Toy64FrameLowering *
  getFrameLowering() const override {
    return &FrameLowering;
  }

  const Toy64TargetLowering *
  getTargetLowering() const override {
    return &TLInfo;
  }

    const SelectionDAGTargetInfo *
    getSelectionDAGInfo() const override {
    return &TSInfo;
    }
};

} // namespace llvm

#endif