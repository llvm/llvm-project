//=== HexagonSplitConst32AndConst64.cpp - split CONST32/Const64 into HI/LO ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// When the compiler is invoked with no small data, for instance, with the -G0
// command line option, then all CONST32_* opcodes should be broken down into
// appropriate LO and HI instructions. This splitting is done by this pass.
// The only reason this is not done in the DAG lowering itself is that there
// is no simple way of getting the register allocator to allot the same hard
// register to the result of LO and HI instructions. This pass is always
// scheduled after register allocation.
//
//===----------------------------------------------------------------------===//

#include "HexagonMachineFunctionInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "HexagonTargetObjectFile.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "xfer"

namespace llvm {
  FunctionPass *createHexagonSplitConst32AndConst64();
  void initializeHexagonSplitConst32AndConst64Pass(PassRegistry&);
}

namespace {

class HexagonSplitConst32AndConst64 : public MachineFunctionPass {
 public:
    static char ID;
    HexagonSplitConst32AndConst64() : MachineFunctionPass(ID) {}

    const char *getPassName() const override {
      return "Hexagon Split Const32s and Const64s";
    }
    bool runOnMachineFunction(MachineFunction &Fn) override;
    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::AllVRegsAllocated);
    }
};


char HexagonSplitConst32AndConst64::ID = 0;


bool HexagonSplitConst32AndConst64::runOnMachineFunction(MachineFunction &Fn) {

  const HexagonTargetObjectFile &TLOF =
      *static_cast<const HexagonTargetObjectFile *>(
          Fn.getTarget().getObjFileLowering());
  if (TLOF.isSmallDataEnabled())
    return true;

  const TargetInstrInfo *TII = Fn.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = Fn.getSubtarget().getRegisterInfo();

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBBb = Fn.begin(), MBBe = Fn.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock *MBB = &*MBBb;
    // Traverse the basic block
    MachineBasicBlock::iterator MII = MBB->begin();
    MachineBasicBlock::iterator MIE = MBB->end ();
    while (MII != MIE) {
      MachineInstr &MI = *MII;
      int Opc = MI.getOpcode();
      if (Opc == Hexagon::CONST32_Int_Real &&
          MI.getOperand(1).isBlockAddress()) {
        int DestReg = MI.getOperand(0).getReg();
        MachineOperand &Symbol = MI.getOperand(1);

        BuildMI(*MBB, MII, MI.getDebugLoc(), TII->get(Hexagon::LO), DestReg)
            .addOperand(Symbol);
        BuildMI(*MBB, MII, MI.getDebugLoc(), TII->get(Hexagon::HI), DestReg)
            .addOperand(Symbol);
        // MBB->erase returns the iterator to the next instruction, which is the
        // one we want to process next
        MII = MBB->erase(&MI);
        continue;
      }

      else if (Opc == Hexagon::CONST32_Int_Real ||
               Opc == Hexagon::CONST32_Float_Real) {
        int DestReg = MI.getOperand(0).getReg();

        // We have to convert an FP immediate into its corresponding integer
        // representation
        int64_t ImmValue;
        if (Opc == Hexagon::CONST32_Float_Real) {
          APFloat Val = MI.getOperand(1).getFPImm()->getValueAPF();
          ImmValue = *Val.bitcastToAPInt().getRawData();
        }
        else
          ImmValue = MI.getOperand(1).getImm();

        BuildMI(*MBB, MII, MI.getDebugLoc(), TII->get(Hexagon::A2_tfrsi),
                DestReg)
            .addImm(ImmValue);
        MII = MBB->erase(&MI);
        continue;
      }
      else if (Opc == Hexagon::CONST64_Int_Real ||
               Opc == Hexagon::CONST64_Float_Real) {
        int DestReg = MI.getOperand(0).getReg();

        // We have to convert an FP immediate into its corresponding integer
        // representation
        int64_t ImmValue;
        if (Opc == Hexagon::CONST64_Float_Real) {
          APFloat Val = MI.getOperand(1).getFPImm()->getValueAPF();
          ImmValue = *Val.bitcastToAPInt().getRawData();
        }
        else
          ImmValue = MI.getOperand(1).getImm();

        unsigned DestLo = TRI->getSubReg(DestReg, Hexagon::subreg_loreg);
        unsigned DestHi = TRI->getSubReg(DestReg, Hexagon::subreg_hireg);

        int32_t LowWord = (ImmValue & 0xFFFFFFFF);
        int32_t HighWord = (ImmValue >> 32) & 0xFFFFFFFF;

        BuildMI(*MBB, MII, MI.getDebugLoc(), TII->get(Hexagon::A2_tfrsi),
                DestLo)
            .addImm(LowWord);
        BuildMI(*MBB, MII, MI.getDebugLoc(), TII->get(Hexagon::A2_tfrsi),
                DestHi)
            .addImm(HighWord);
        MII = MBB->erase(&MI);
        continue;
      }
      ++MII;
    }
  }

  return true;
}

}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *
llvm::createHexagonSplitConst32AndConst64() {
  return new HexagonSplitConst32AndConst64();
}
