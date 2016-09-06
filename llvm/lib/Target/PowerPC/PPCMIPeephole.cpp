//===-------------- PPCMIPeephole.cpp - MI Peephole Cleanups -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This pass performs peephole optimizations to clean up ugly code
// sequences at the MachineInstruction layer.  It runs at the end of
// the SSA phases, following VSX swap removal.  A pass of dead code
// elimination follows this one for quick clean-up of any dead
// instructions introduced here.  Although we could do this as callbacks
// from the generic peephole pass, this would have a couple of bad
// effects:  it might remove optimization opportunities for VSX swap
// removal, and it would miss cleanups made possible following VSX
// swap removal.
//
//===---------------------------------------------------------------------===//

#include "PPCInstrInfo.h"
#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "ppc-mi-peepholes"

namespace llvm {
  void initializePPCMIPeepholePass(PassRegistry&);
}

namespace {

struct PPCMIPeephole : public MachineFunctionPass {

  static char ID;
  const PPCInstrInfo *TII;
  MachineFunction *MF;
  MachineRegisterInfo *MRI;

  PPCMIPeephole() : MachineFunctionPass(ID) {
    initializePPCMIPeepholePass(*PassRegistry::getPassRegistry());
  }

private:
  // Initialize class variables.
  void initialize(MachineFunction &MFParm);

  // Perform peepholes.
  bool simplifyCode(void);

  // Find the "true" register represented by SrcReg (following chains
  // of copies and subreg_to_reg operations).
  unsigned lookThruCopyLike(unsigned SrcReg);

public:
  // Main entry point for this pass.
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(*MF.getFunction()))
      return false;
    initialize(MF);
    return simplifyCode();
  }
};

// Initialize class variables.
void PPCMIPeephole::initialize(MachineFunction &MFParm) {
  MF = &MFParm;
  MRI = &MF->getRegInfo();
  TII = MF->getSubtarget<PPCSubtarget>().getInstrInfo();
  DEBUG(dbgs() << "*** PowerPC MI peephole pass ***\n\n");
  DEBUG(MF->dump());
}

// Perform peephole optimizations.
bool PPCMIPeephole::simplifyCode(void) {
  bool Simplified = false;
  MachineInstr* ToErase = nullptr;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {

      // If the previous instruction was marked for elimination,
      // remove it now.
      if (ToErase) {
        ToErase->eraseFromParent();
        ToErase = nullptr;
      }

      // Ignore debug instructions.
      if (MI.isDebugValue())
        continue;

      // Per-opcode peepholes.
      switch (MI.getOpcode()) {

      default:
        break;

      case PPC::XXPERMDI: {
        // Perform simplifications of 2x64 vector swaps and splats.
        // A swap is identified by an immediate value of 2, and a splat
        // is identified by an immediate value of 0 or 3.
        int Immed = MI.getOperand(3).getImm();

        if (Immed != 1) {

          // For each of these simplifications, we need the two source
          // regs to match.  Unfortunately, MachineCSE ignores COPY and
          // SUBREG_TO_REG, so for example we can see
          //   XXPERMDI t, SUBREG_TO_REG(s), SUBREG_TO_REG(s), immed.
          // We have to look through chains of COPY and SUBREG_TO_REG
          // to find the real source values for comparison.
          unsigned TrueReg1 = lookThruCopyLike(MI.getOperand(1).getReg());
          unsigned TrueReg2 = lookThruCopyLike(MI.getOperand(2).getReg());

          if (TrueReg1 == TrueReg2
              && TargetRegisterInfo::isVirtualRegister(TrueReg1)) {
            MachineInstr *DefMI = MRI->getVRegDef(TrueReg1);

            // If this is a splat or a swap fed by another splat, we
            // can replace it with a copy.
            if (DefMI && DefMI->getOpcode() == PPC::XXPERMDI) {
              unsigned FeedImmed = DefMI->getOperand(3).getImm();
              unsigned FeedReg1
                = lookThruCopyLike(DefMI->getOperand(1).getReg());
              unsigned FeedReg2
                = lookThruCopyLike(DefMI->getOperand(2).getReg());

              if ((FeedImmed == 0 || FeedImmed == 3) && FeedReg1 == FeedReg2) {
                DEBUG(dbgs()
                      << "Optimizing splat/swap or splat/splat "
                      "to splat/copy: ");
                DEBUG(MI.dump());
                BuildMI(MBB, &MI, MI.getDebugLoc(),
                        TII->get(PPC::COPY), MI.getOperand(0).getReg())
                  .addOperand(MI.getOperand(1));
                ToErase = &MI;
                Simplified = true;
              }

              // If this is a splat fed by a swap, we can simplify modify
              // the splat to splat the other value from the swap's input
              // parameter.
              else if ((Immed == 0 || Immed == 3)
                       && FeedImmed == 2 && FeedReg1 == FeedReg2) {
                DEBUG(dbgs() << "Optimizing swap/splat => splat: ");
                DEBUG(MI.dump());
                MI.getOperand(1).setReg(DefMI->getOperand(1).getReg());
                MI.getOperand(2).setReg(DefMI->getOperand(2).getReg());
                MI.getOperand(3).setImm(3 - Immed);
                Simplified = true;
              }

              // If this is a swap fed by a swap, we can replace it
              // with a copy from the first swap's input.
              else if (Immed == 2 && FeedImmed == 2 && FeedReg1 == FeedReg2) {
                DEBUG(dbgs() << "Optimizing swap/swap => copy: ");
                DEBUG(MI.dump());
                BuildMI(MBB, &MI, MI.getDebugLoc(),
                        TII->get(PPC::COPY), MI.getOperand(0).getReg())
                  .addOperand(DefMI->getOperand(1));
                ToErase = &MI;
                Simplified = true;
              }
            }
          }
        }
        break;
      }
      }
    }

    // If the last instruction was marked for elimination,
    // remove it now.
    if (ToErase) {
      ToErase->eraseFromParent();
      ToErase = nullptr;
    }
  }

  return Simplified;
}

// This is used to find the "true" source register for an
// XXPERMDI instruction, since MachineCSE does not handle the
// "copy-like" operations (Copy and SubregToReg).  Returns
// the original SrcReg unless it is the target of a copy-like
// operation, in which case we chain backwards through all
// such operations to the ultimate source register.  If a
// physical register is encountered, we stop the search.
unsigned PPCMIPeephole::lookThruCopyLike(unsigned SrcReg) {

  while (true) {

    MachineInstr *MI = MRI->getVRegDef(SrcReg);
    if (!MI->isCopyLike())
      return SrcReg;

    unsigned CopySrcReg;
    if (MI->isCopy())
      CopySrcReg = MI->getOperand(1).getReg();
    else {
      assert(MI->isSubregToReg() && "bad opcode for lookThruCopyLike");
      CopySrcReg = MI->getOperand(2).getReg();
    }

    if (!TargetRegisterInfo::isVirtualRegister(CopySrcReg))
      return CopySrcReg;

    SrcReg = CopySrcReg;
  }
}

} // end default namespace

INITIALIZE_PASS_BEGIN(PPCMIPeephole, DEBUG_TYPE,
                      "PowerPC MI Peephole Optimization", false, false)
INITIALIZE_PASS_END(PPCMIPeephole, DEBUG_TYPE,
                    "PowerPC MI Peephole Optimization", false, false)

char PPCMIPeephole::ID = 0;
FunctionPass*
llvm::createPPCMIPeepholePass() { return new PPCMIPeephole(); }

