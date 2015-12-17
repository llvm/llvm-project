//===-- X86OptimizeLEAs.cpp - optimize usage of LEA instructions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that performs some optimizations with LEA
// instructions in order to improve code size.
// Currently, it does one thing:
// 1) Address calculations in load and store instructions are replaced by
//    existing LEA def registers where possible.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "x86-optimize-LEAs"

static cl::opt<bool> EnableX86LEAOpt("enable-x86-lea-opt", cl::Hidden,
                                     cl::desc("X86: Enable LEA optimizations."),
                                     cl::init(false));

STATISTIC(NumSubstLEAs, "Number of LEA instruction substitutions");

namespace {
class OptimizeLEAPass : public MachineFunctionPass {
public:
  OptimizeLEAPass() : MachineFunctionPass(ID) {}

  const char *getPassName() const override { return "X86 LEA Optimize"; }

  /// \brief Loop over all of the basic blocks, replacing address
  /// calculations in load and store instructions, if it's already
  /// been calculated by LEA. Also, remove redundant LEAs.
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// \brief Returns a distance between two instructions inside one basic block.
  /// Negative result means, that instructions occur in reverse order.
  int calcInstrDist(const MachineInstr &First, const MachineInstr &Last);

  /// \brief Choose the best \p LEA instruction from the \p List to replace
  /// address calculation in \p MI instruction. Return the address displacement
  /// and the distance between \p MI and the choosen \p LEA in \p AddrDispShift
  /// and \p Dist.
  bool chooseBestLEA(const SmallVectorImpl<MachineInstr *> &List,
                     const MachineInstr &MI, MachineInstr *&LEA,
                     int64_t &AddrDispShift, int &Dist);

  /// \brief Returns true if two machine operand are identical and they are not
  /// physical registers.
  bool isIdenticalOp(const MachineOperand &MO1, const MachineOperand &MO2);

  /// \brief Returns true if the instruction is LEA.
  bool isLEA(const MachineInstr &MI);

  /// \brief Returns true if two instructions have memory operands that only
  /// differ by displacement. The numbers of the first memory operands for both
  /// instructions are specified through \p N1 and \p N2. The address
  /// displacement is returned through AddrDispShift.
  bool isSimilarMemOp(const MachineInstr &MI1, unsigned N1,
                      const MachineInstr &MI2, unsigned N2,
                      int64_t &AddrDispShift);

  /// \brief Find all LEA instructions in the basic block.
  void findLEAs(const MachineBasicBlock &MBB,
                SmallVectorImpl<MachineInstr *> &List);

  /// \brief Removes redundant address calculations.
  bool removeRedundantAddrCalc(const SmallVectorImpl<MachineInstr *> &List);

  MachineRegisterInfo *MRI;
  const X86InstrInfo *TII;
  const X86RegisterInfo *TRI;

  static char ID;
};
char OptimizeLEAPass::ID = 0;
}

FunctionPass *llvm::createX86OptimizeLEAs() { return new OptimizeLEAPass(); }

int OptimizeLEAPass::calcInstrDist(const MachineInstr &First,
                                   const MachineInstr &Last) {
  const MachineBasicBlock *MBB = First.getParent();

  // Both instructions must be in the same basic block.
  assert(Last.getParent() == MBB &&
         "Instructions are in different basic blocks");

  return std::distance(MBB->begin(), MachineBasicBlock::const_iterator(&Last)) -
         std::distance(MBB->begin(), MachineBasicBlock::const_iterator(&First));
}

// Find the best LEA instruction in the List to replace address recalculation in
// MI. Such LEA must meet these requirements:
// 1) The address calculated by the LEA differs only by the displacement from
//    the address used in MI.
// 2) The register class of the definition of the LEA is compatible with the
//    register class of the address base register of MI.
// 3) Displacement of the new memory operand should fit in 1 byte if possible.
// 4) The LEA should be as close to MI as possible, and prior to it if
//    possible.
bool OptimizeLEAPass::chooseBestLEA(const SmallVectorImpl<MachineInstr *> &List,
                                    const MachineInstr &MI, MachineInstr *&LEA,
                                    int64_t &AddrDispShift, int &Dist) {
  const MachineFunction *MF = MI.getParent()->getParent();
  const MCInstrDesc &Desc = MI.getDesc();
  int MemOpNo = X86II::getMemoryOperandNo(Desc.TSFlags, MI.getOpcode()) +
                X86II::getOperandBias(Desc);

  LEA = nullptr;

  // Loop over all LEA instructions.
  for (auto DefMI : List) {
    int64_t AddrDispShiftTemp = 0;

    // Compare instructions memory operands.
    if (!isSimilarMemOp(MI, MemOpNo, *DefMI, 1, AddrDispShiftTemp))
      continue;

    // Make sure address displacement fits 4 bytes.
    if (!isInt<32>(AddrDispShiftTemp))
      continue;

    // Check that LEA def register can be used as MI address base. Some
    // instructions can use a limited set of registers as address base, for
    // example MOV8mr_NOREX. We could constrain the register class of the LEA
    // def to suit MI, however since this case is very rare and hard to
    // reproduce in a test it's just more reliable to skip the LEA.
    if (TII->getRegClass(Desc, MemOpNo + X86::AddrBaseReg, TRI, *MF) !=
        MRI->getRegClass(DefMI->getOperand(0).getReg()))
      continue;

    // Choose the closest LEA instruction from the list, prior to MI if
    // possible. Note that we took into account resulting address displacement
    // as well. Also note that the list is sorted by the order in which the LEAs
    // occur, so the break condition is pretty simple.
    int DistTemp = calcInstrDist(*DefMI, MI);
    assert(DistTemp != 0 &&
           "The distance between two different instructions cannot be zero");
    if (DistTemp > 0 || LEA == nullptr) {
      // Do not update return LEA, if the current one provides a displacement
      // which fits in 1 byte, while the new candidate does not.
      if (LEA != nullptr && !isInt<8>(AddrDispShiftTemp) &&
          isInt<8>(AddrDispShift))
        continue;

      LEA = DefMI;
      AddrDispShift = AddrDispShiftTemp;
      Dist = DistTemp;
    }

    // FIXME: Maybe we should not always stop at the first LEA after MI.
    if (DistTemp < 0)
      break;
  }

  return LEA != nullptr;
}

bool OptimizeLEAPass::isIdenticalOp(const MachineOperand &MO1,
                                    const MachineOperand &MO2) {
  return MO1.isIdenticalTo(MO2) &&
         (!MO1.isReg() ||
          !TargetRegisterInfo::isPhysicalRegister(MO1.getReg()));
}

bool OptimizeLEAPass::isLEA(const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  return Opcode == X86::LEA16r || Opcode == X86::LEA32r ||
         Opcode == X86::LEA64r || Opcode == X86::LEA64_32r;
}

// Check if MI1 and MI2 have memory operands which represent addresses that
// differ only by displacement.
bool OptimizeLEAPass::isSimilarMemOp(const MachineInstr &MI1, unsigned N1,
                                     const MachineInstr &MI2, unsigned N2,
                                     int64_t &AddrDispShift) {
  // Address base, scale, index and segment operands must be identical.
  static const int IdenticalOpNums[] = {X86::AddrBaseReg, X86::AddrScaleAmt,
                                        X86::AddrIndexReg, X86::AddrSegmentReg};
  for (auto &N : IdenticalOpNums)
    if (!isIdenticalOp(MI1.getOperand(N1 + N), MI2.getOperand(N2 + N)))
      return false;

  // Address displacement operands may differ by a constant.
  const MachineOperand *Op1 = &MI1.getOperand(N1 + X86::AddrDisp);
  const MachineOperand *Op2 = &MI2.getOperand(N2 + X86::AddrDisp);
  if (!isIdenticalOp(*Op1, *Op2)) {
    if (Op1->isImm() && Op2->isImm())
      AddrDispShift = Op1->getImm() - Op2->getImm();
    else if (Op1->isGlobal() && Op2->isGlobal() &&
             Op1->getGlobal() == Op2->getGlobal())
      AddrDispShift = Op1->getOffset() - Op2->getOffset();
    else
      return false;
  }

  return true;
}

void OptimizeLEAPass::findLEAs(const MachineBasicBlock &MBB,
                               SmallVectorImpl<MachineInstr *> &List) {
  for (auto &MI : MBB) {
    if (isLEA(MI))
      List.push_back(const_cast<MachineInstr *>(&MI));
  }
}

// Try to find load and store instructions which recalculate addresses already
// calculated by some LEA and replace their memory operands with its def
// register.
bool OptimizeLEAPass::removeRedundantAddrCalc(
    const SmallVectorImpl<MachineInstr *> &List) {
  bool Changed = false;

  assert(List.size() > 0);
  MachineBasicBlock *MBB = List[0]->getParent();

  // Process all instructions in basic block.
  for (auto I = MBB->begin(), E = MBB->end(); I != E;) {
    MachineInstr &MI = *I++;
    unsigned Opcode = MI.getOpcode();

    // Instruction must be load or store.
    if (!MI.mayLoadOrStore())
      continue;

    // Get the number of the first memory operand.
    const MCInstrDesc &Desc = MI.getDesc();
    int MemOpNo = X86II::getMemoryOperandNo(Desc.TSFlags, Opcode);

    // If instruction has no memory operand - skip it.
    if (MemOpNo < 0)
      continue;

    MemOpNo += X86II::getOperandBias(Desc);

    // Get the best LEA instruction to replace address calculation.
    MachineInstr *DefMI;
    int64_t AddrDispShift;
    int Dist;
    if (!chooseBestLEA(List, MI, DefMI, AddrDispShift, Dist))
      continue;

    // If LEA occurs before current instruction, we can freely replace
    // the instruction. If LEA occurs after, we can lift LEA above the
    // instruction and this way to be able to replace it. Since LEA and the
    // instruction have similar memory operands (thus, the same def
    // instructions for these operands), we can always do that, without
    // worries of using registers before their defs.
    if (Dist < 0) {
      DefMI->removeFromParent();
      MBB->insert(MachineBasicBlock::iterator(&MI), DefMI);
    }

    // Since we can possibly extend register lifetime, clear kill flags.
    MRI->clearKillFlags(DefMI->getOperand(0).getReg());

    ++NumSubstLEAs;
    DEBUG(dbgs() << "OptimizeLEAs: Candidate to replace: "; MI.dump(););

    // Change instruction operands.
    MI.getOperand(MemOpNo + X86::AddrBaseReg)
        .ChangeToRegister(DefMI->getOperand(0).getReg(), false);
    MI.getOperand(MemOpNo + X86::AddrScaleAmt).ChangeToImmediate(1);
    MI.getOperand(MemOpNo + X86::AddrIndexReg)
        .ChangeToRegister(X86::NoRegister, false);
    MI.getOperand(MemOpNo + X86::AddrDisp).ChangeToImmediate(AddrDispShift);
    MI.getOperand(MemOpNo + X86::AddrSegmentReg)
        .ChangeToRegister(X86::NoRegister, false);

    DEBUG(dbgs() << "OptimizeLEAs: Replaced by: "; MI.dump(););

    Changed = true;
  }

  return Changed;
}

bool OptimizeLEAPass::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  // Perform this optimization only if we care about code size.
  if (!EnableX86LEAOpt || !MF.getFunction()->optForSize())
    return false;

  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  TRI = MF.getSubtarget<X86Subtarget>().getRegisterInfo();

  // Process all basic blocks.
  for (auto &MBB : MF) {
    SmallVector<MachineInstr *, 16> LEAs;

    // Find all LEA instructions in basic block.
    findLEAs(MBB, LEAs);

    // If current basic block has no LEAs, move on to the next one.
    if (LEAs.empty())
      continue;

    // Remove redundant address calculations.
    Changed |= removeRedundantAddrCalc(LEAs);
  }

  return Changed;
}
