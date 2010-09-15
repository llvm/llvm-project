//===-- PeepholeOptimizer.cpp - Peephole Optimizations --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Perform peephole optimizations on the machine code:
//
// - Optimize Extensions
//
//     Optimization of sign / zero extension instructions. It may be extended to
//     handle other instructions with similar properties.
//
//     On some targets, some instructions, e.g. X86 sign / zero extension, may
//     leave the source value in the lower part of the result. This optimization
//     will replace some uses of the pre-extension value with uses of the
//     sub-register of the results.
//
// - Optimize Comparisons
//
//     Optimization of comparison instructions. For instance, in this code:
//
//       sub r1, 1
//       cmp r1, 0
//       bz  L1
//
//     If the "sub" instruction all ready sets (or could be modified to set) the
//     same flag that the "cmp" instruction sets and that "bz" uses, then we can
//     eliminate the "cmp" instruction.
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "peephole-opt"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

// Optimize Extensions
static cl::opt<bool>
Aggressive("aggressive-ext-opt", cl::Hidden,
           cl::desc("Aggressive extension optimization"));

STATISTIC(NumReuse,      "Number of extension results reused");
STATISTIC(NumEliminated, "Number of compares eliminated");

namespace {
  class PeepholeOptimizer : public MachineFunctionPass {
    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    MachineRegisterInfo   *MRI;
    MachineDominatorTree  *DT;  // Machine dominator tree

  public:
    static char ID; // Pass identification
    PeepholeOptimizer() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      if (Aggressive) {
        AU.addRequired<MachineDominatorTree>();
        AU.addPreserved<MachineDominatorTree>();
      }
    }

  private:
    bool OptimizeCmpInstr(MachineInstr *MI, MachineBasicBlock *MBB);
    bool OptimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                          SmallPtrSet<MachineInstr*, 8> &LocalMIs);
  };
}

char PeepholeOptimizer::ID = 0;
INITIALIZE_PASS(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false);

FunctionPass *llvm::createPeepholeOptimizerPass() {
  return new PeepholeOptimizer();
}

/// OptimizeExtInstr - If instruction is a copy-like instruction, i.e. it reads
/// a single register and writes a single register and it does not modify the
/// source, and if the source value is preserved as a sub-register of the
/// result, then replace all reachable uses of the source with the subreg of the
/// result.
/// 
/// Do not generate an EXTRACT that is used only in a debug use, as this changes
/// the code. Since this code does not currently share EXTRACTs, just ignore all
/// debug uses.
bool PeepholeOptimizer::
OptimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                 SmallPtrSet<MachineInstr*, 8> &LocalMIs) {
  LocalMIs.insert(MI);

  unsigned SrcReg, DstReg, SubIdx;
  if (!TII->isCoalescableExtInstr(*MI, SrcReg, DstReg, SubIdx))
    return false;

  if (TargetRegisterInfo::isPhysicalRegister(DstReg) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg))
    return false;

  MachineRegisterInfo::use_nodbg_iterator UI = MRI->use_nodbg_begin(SrcReg);
  if (++UI == MRI->use_nodbg_end())
    // No other uses.
    return false;

  // The source has other uses. See if we can replace the other uses with use of
  // the result of the extension.
  SmallPtrSet<MachineBasicBlock*, 4> ReachedBBs;
  UI = MRI->use_nodbg_begin(DstReg);
  for (MachineRegisterInfo::use_nodbg_iterator UE = MRI->use_nodbg_end();
       UI != UE; ++UI)
    ReachedBBs.insert(UI->getParent());

  // Uses that are in the same BB of uses of the result of the instruction.
  SmallVector<MachineOperand*, 8> Uses;

  // Uses that the result of the instruction can reach.
  SmallVector<MachineOperand*, 8> ExtendedUses;

  bool ExtendLife = true;
  UI = MRI->use_nodbg_begin(SrcReg);
  for (MachineRegisterInfo::use_nodbg_iterator UE = MRI->use_nodbg_end();
       UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = &*UI;
    if (UseMI == MI)
      continue;

    if (UseMI->isPHI()) {
      ExtendLife = false;
      continue;
    }

    // It's an error to translate this:
    //
    //    %reg1025 = <sext> %reg1024
    //     ...
    //    %reg1026 = SUBREG_TO_REG 0, %reg1024, 4
    //
    // into this:
    //
    //    %reg1025 = <sext> %reg1024
    //     ...
    //    %reg1027 = COPY %reg1025:4
    //    %reg1026 = SUBREG_TO_REG 0, %reg1027, 4
    //
    // The problem here is that SUBREG_TO_REG is there to assert that an
    // implicit zext occurs. It doesn't insert a zext instruction. If we allow
    // the COPY here, it will give us the value after the <sext>, not the
    // original value of %reg1024 before <sext>.
    if (UseMI->getOpcode() == TargetOpcode::SUBREG_TO_REG)
      continue;

    MachineBasicBlock *UseMBB = UseMI->getParent();
    if (UseMBB == MBB) {
      // Local uses that come after the extension.
      if (!LocalMIs.count(UseMI))
        Uses.push_back(&UseMO);
    } else if (ReachedBBs.count(UseMBB)) {
      // Non-local uses where the result of the extension is used. Always
      // replace these unless it's a PHI.
      Uses.push_back(&UseMO);
    } else if (Aggressive && DT->dominates(MBB, UseMBB)) {
      // We may want to extend the live range of the extension result in order
      // to replace these uses.
      ExtendedUses.push_back(&UseMO);
    } else {
      // Both will be live out of the def MBB anyway. Don't extend live range of
      // the extension result.
      ExtendLife = false;
      break;
    }
  }

  if (ExtendLife && !ExtendedUses.empty())
    // Extend the liveness of the extension result.
    std::copy(ExtendedUses.begin(), ExtendedUses.end(),
              std::back_inserter(Uses));

  // Now replace all uses.
  bool Changed = false;
  if (!Uses.empty()) {
    SmallPtrSet<MachineBasicBlock*, 4> PHIBBs;

    // Look for PHI uses of the extended result, we don't want to extend the
    // liveness of a PHI input. It breaks all kinds of assumptions down
    // stream. A PHI use is expected to be the kill of its source values.
    UI = MRI->use_nodbg_begin(DstReg);
    for (MachineRegisterInfo::use_nodbg_iterator
           UE = MRI->use_nodbg_end(); UI != UE; ++UI)
      if (UI->isPHI())
        PHIBBs.insert(UI->getParent());

    const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
    for (unsigned i = 0, e = Uses.size(); i != e; ++i) {
      MachineOperand *UseMO = Uses[i];
      MachineInstr *UseMI = UseMO->getParent();
      MachineBasicBlock *UseMBB = UseMI->getParent();
      if (PHIBBs.count(UseMBB))
        continue;

      unsigned NewVR = MRI->createVirtualRegister(RC);
      BuildMI(*UseMBB, UseMI, UseMI->getDebugLoc(),
              TII->get(TargetOpcode::COPY), NewVR)
        .addReg(DstReg, 0, SubIdx);

      UseMO->setReg(NewVR);
      ++NumReuse;
      Changed = true;
    }
  }

  return Changed;
}

/// OptimizeCmpInstr - If the instruction is a compare and the previous
/// instruction it's comparing against all ready sets (or could be modified to
/// set) the same flag as the compare, then we can remove the comparison and use
/// the flag from the previous instruction.
bool PeepholeOptimizer::OptimizeCmpInstr(MachineInstr *MI,
                                         MachineBasicBlock *MBB) {
  // If this instruction is a comparison against zero and isn't comparing a
  // physical register, we can try to optimize it.
  unsigned SrcReg;
  int CmpValue;
  if (!TII->AnalyzeCompare(MI, SrcReg, CmpValue) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg) || CmpValue != 0)
    return false;

  MachineRegisterInfo::def_iterator DI = MRI->def_begin(SrcReg);
  if (llvm::next(DI) != MRI->def_end())
    // Only support one definition.
    return false;

  // Attempt to convert the defining instruction to set the "zero" flag.
  if (TII->ConvertToSetZeroFlag(&*DI, MI)) {
    ++NumEliminated;
    return true;
  }

  return false;
}

bool PeepholeOptimizer::runOnMachineFunction(MachineFunction &MF) {
  TM  = &MF.getTarget();
  TII = TM->getInstrInfo();
  MRI = &MF.getRegInfo();
  DT  = Aggressive ? &getAnalysis<MachineDominatorTree>() : 0;

  bool Changed = false;

  SmallPtrSet<MachineInstr*, 8> LocalMIs;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    LocalMIs.clear();

    for (MachineBasicBlock::iterator
           MII = I->begin(), ME = I->end(); MII != ME; ) {
      MachineInstr *MI = &*MII;

      if (MI->getDesc().isCompare() &&
          !MI->getDesc().hasUnmodeledSideEffects()) {
        ++MII; // The iterator may become invalid if the compare is deleted.
        Changed |= OptimizeCmpInstr(MI, MBB);
      } else {
        Changed |= OptimizeExtInstr(MI, MBB, LocalMIs);
        ++MII;
      }
    }
  }

  return Changed;
}
