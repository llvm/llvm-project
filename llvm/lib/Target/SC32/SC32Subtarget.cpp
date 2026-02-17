#include "SC32Subtarget.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define GET_SUBTARGETINFO_CTOR
#include "SC32GenSubtargetInfo.inc"

SC32Subtarget::SC32Subtarget(const TargetMachine &TM)
    : SC32GenSubtargetInfo(TM.getTargetTriple(), TM.getTargetCPU(),
                           TM.getTargetCPU(), TM.getTargetFeatureString()),
      TLInfo(TM, *this), InstrInfo(*this, RI) {}

const TargetRegisterInfo *SC32Subtarget::getRegisterInfo() const { return &RI; }

const TargetLowering *SC32Subtarget::getTargetLowering() const {
  return &TLInfo;
}

const TargetFrameLowering *SC32Subtarget::getFrameLowering() const {
  return &FrameLowering;
}

const TargetInstrInfo *SC32Subtarget::getInstrInfo() const {
  return &InstrInfo;
}

const SelectionDAGTargetInfo *SC32Subtarget::getSelectionDAGInfo() const {
  return &TSInfo;
}
