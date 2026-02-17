#include "SC32InstrInfo.h"
#include "MCTargetDesc/SC32MCTargetDesc.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "SC32GenInstrInfo.inc"

void SC32InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI,
                                const DebugLoc &DL, Register DestReg,
                                Register SrcReg, bool KillSrc,
                                bool RenamableDest, bool RenamableSrc) const {
  BuildMI(MBB, MI, DL, get(SC32::MOV), DestReg)
      .addReg(SrcReg, getKillRegState(KillSrc));
}
