#ifndef LLVM_LIB_TARGET_SC32_SC32INSTRINFO_H
#define LLVM_LIB_TARGET_SC32_SC32INSTRINFO_H

#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "SC32GenInstrInfo.inc"

namespace llvm {

class SC32InstrInfo : public SC32GenInstrInfo {
public:
  using SC32GenInstrInfo::SC32GenInstrInfo;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, Register DestReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest,
                   bool RenamableSrc) const override;
};

} // namespace llvm

#endif
