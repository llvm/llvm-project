// X86MatchJumptablePass.h
#ifndef X86_MATCH_JUMPTABLE_PASS_H
#define X86_MATCH_JUMPTABLE_PASS_H

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

namespace llvm {

class X86MatchJumptablePass : public MachineFunctionPass {
private:
  void insertIdentifyingMarker(MachineInstr* MI, MachineFunction &MF, unsigned JTIndex);
  void recordJumpLocation(MachineInstr* MI, MachineFunction &MF, unsigned JTIndex);
  MachineInstr* traceIndirectJumps(MachineFunction &MF, unsigned JTIndex, 
                                  MachineJumpTableInfo *JumpTableInfo);
  bool isJumpTableRelated(MachineInstr &MI, const MachineJumpTableEntry &JTEntry, 
                         MachineFunction &MF);
  bool isJumpTableLoad(MachineInstr &MI, const MachineJumpTableEntry &JTEntry);
  bool isRegUsedInJumpTableLoad(Register Reg,MachineFunction &MF,
                                                    const MachineJumpTableEntry &JTEntry);

public:
  static char ID;

  X86MatchJumptablePass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return "Match Jump Table Pass"; }
};

FunctionPass *createX86MatchJumptablePass();

// Pass initialization declaration
void initializeX86MatchJumptablePassPass(PassRegistry &Registry);

} // namespace llvm

#endif // X86_MATCH_JUMPTABLE_PASS_H

