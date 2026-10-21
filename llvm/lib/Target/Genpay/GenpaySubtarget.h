#ifndef LLVM_LIB_TARGET_GENPAY_GENPAYSUBTARGET_H
#define LLVM_LIB_TARGET_GENPAY_GENPAYSUBTARGET_H

#include "GenpayFrameLowering.h"
#include "GenpayISelLowering.h"
#include "GenpayInstrInfo.h"
#include "GenpayRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

// Minimal SelectionDAG target info (no custom mem-intrinsics yet).
class GenpaySelectionDAGInfo : public SelectionDAGTargetInfo {};

class GenpaySubtarget : public TargetSubtargetInfo {
  GenpayInstrInfo InstrInfo;
  GenpayRegisterInfo RegInfo;
  GenpayTargetLowering TLInfo;
  GenpayFrameLowering FrameLowering;
  GenpaySelectionDAGInfo SELDAGInfo;

public:
  GenpaySubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                  const TargetMachine &TM, const TargetOptions &Options,
                  CodeModel::Model CM, CodeGenOptLevel OL);

  const GenpayInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const GenpayRegisterInfo *getRegisterInfo() const override { return &RegInfo; }
  const GenpayTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const TargetFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override {
    return &SELDAGInfo;
  }
};

} // namespace llvm

#endif
