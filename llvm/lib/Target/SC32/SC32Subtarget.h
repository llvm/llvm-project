#ifndef LLVM_LIB_TARGET_SC32_SC32SUBTARGET_H
#define LLVM_LIB_TARGET_SC32_SC32SUBTARGET_H

#include "SC32FrameLowering.h"
#include "SC32InstrInfo.h"
#include "SC32RegisterInfo.h"
#include "SC32SelectionDAGInfo.h"
#include "SC32TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "SC32GenSubtargetInfo.inc"

namespace llvm {

class SC32Subtarget : public SC32GenSubtargetInfo {
private:
  SC32RegisterInfo RI;
  SC32TargetLowering TLInfo;
  SC32FrameLowering FrameLowering;
  SC32InstrInfo InstrInfo;
  SC32SelectionDAGInfo TSInfo;

public:
  SC32Subtarget(const TargetMachine &TM);

  const TargetRegisterInfo *getRegisterInfo() const override;

  const TargetLowering *getTargetLowering() const override;

  const TargetFrameLowering *getFrameLowering() const override;

  const TargetInstrInfo *getInstrInfo() const override;

  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override;
};

} // namespace llvm

#endif
