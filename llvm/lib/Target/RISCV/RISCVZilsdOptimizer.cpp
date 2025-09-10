//===-- RISCVZilsdOptimizer.cpp - RISC-V Zilsd Load/Store Optimizer ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that performs load/store optimizations for the
// RISC-V Zilsd extension. It combines pairs of 32-bit load/store instructions
// into single 64-bit LD/SD instructions when possible.
//
// The pass runs in two phases:
// 1. Pre-allocation: Reschedules loads/stores to bring consecutive memory
//    accesses closer together and forms LD/SD pairs with register hints.
// 2. Post-allocation: Fixes invalid LD/SD instructions if register allocation
//    didn't provide suitable consecutive registers.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "riscv-zilsd-opt"

STATISTIC(NumLDFormed, "Number of LD instructions formed");
STATISTIC(NumSDFormed, "Number of SD instructions formed");
STATISTIC(NumLD2LW, "Number of LD instructions split back to LW");
STATISTIC(NumSD2SW, "Number of SD instructions split back to SW");

static cl::opt<bool>
    DisableZilsdOpt("disable-riscv-zilsd-opt", cl::Hidden, cl::init(false),
                    cl::desc("Disable Zilsd load/store optimization"));

namespace {

//===----------------------------------------------------------------------===//
// Pre-allocation Zilsd optimization pass
//===----------------------------------------------------------------------===//
class RISCVPreAllocZilsdOpt : public MachineFunctionPass {
public:
  static char ID;

  RISCVPreAllocZilsdOpt() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V pre-allocation Zilsd load/store optimization";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool isMemoryOp(const MachineInstr &MI);
  bool rescheduleLoadStoreInstrs(MachineBasicBlock *MBB);
  bool canFormLdSdPair(MachineInstr *Op0, MachineInstr *Op1, unsigned &NewOpc,
                       Register &FirstReg, Register &SecondReg,
                       Register &BaseReg, int &Offset);
  bool rescheduleOps(MachineBasicBlock *MBB,
                     SmallVectorImpl<MachineInstr *> &Ops, unsigned Base,
                     bool isLoad,
                     DenseMap<MachineInstr *, unsigned> &MI2LocMap);
  bool isSafeToMove(MachineInstr *MI, MachineInstr *Target, bool MoveForward);
  int getMemoryOpOffset(const MachineInstr &MI);

  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;
  const RISCVRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  AliasAnalysis *AA;
  MachineDominatorTree *DT;
};

//===----------------------------------------------------------------------===//
// Post-allocation Zilsd optimization pass
//===----------------------------------------------------------------------===//
class RISCVPostAllocZilsdOpt : public MachineFunctionPass {
public:
  static char ID;

  RISCVPostAllocZilsdOpt() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V post-allocation Zilsd load/store optimization";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool fixInvalidRegPairOp(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MBBI);
  bool isConsecutiveRegPair(Register First, Register Second);
  void splitLdSdIntoTwo(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &MBBI, bool isLoad);

  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;
  const RISCVRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
};

} // end anonymous namespace

char RISCVPreAllocZilsdOpt::ID = 0;
char RISCVPostAllocZilsdOpt::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVPreAllocZilsdOpt, "riscv-prera-zilsd-opt",
                      "RISC-V pre-allocation Zilsd optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(RISCVPreAllocZilsdOpt, "riscv-prera-zilsd-opt",
                    "RISC-V pre-allocation Zilsd optimization", false, false)

INITIALIZE_PASS(RISCVPostAllocZilsdOpt, "riscv-postra-zilsd-opt",
                "RISC-V post-allocation Zilsd optimization", false, false)

//===----------------------------------------------------------------------===//
// Pre-allocation pass implementation
//===----------------------------------------------------------------------===//

bool RISCVPreAllocZilsdOpt::runOnMachineFunction(MachineFunction &MF) {

  if (DisableZilsdOpt || skipFunction(MF.getFunction()))
    return false;

  STI = &MF.getSubtarget<RISCVSubtarget>();

  // Only run on RV32 with Zilsd extension
  if (STI->is64Bit() || !STI->hasStdExtZilsd())
    return false;

  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &MF.getRegInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  DT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  bool Modified = false;
  for (auto &MBB : MF) {
    Modified |= rescheduleLoadStoreInstrs(&MBB);
  }

  return Modified;
}

int RISCVPreAllocZilsdOpt::getMemoryOpOffset(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::LW:
  case RISCV::SW:
    // For LW/SW, the offset is in operand 2
    if (MI.getOperand(2).isImm())
      return MI.getOperand(2).getImm();
    break;
  default:
    break;
  }
  return 0;
}

bool RISCVPreAllocZilsdOpt::canFormLdSdPair(MachineInstr *Op0,
                                            MachineInstr *Op1, unsigned &NewOpc,
                                            Register &FirstReg,
                                            Register &SecondReg,
                                            Register &BaseReg, int &Offset) {

  unsigned Opcode = Op0->getOpcode();

  // Check if we have two LW or two SW instructions
  if (Opcode != Op1->getOpcode())
    return false;

  if (Opcode == RISCV::LW) {
    NewOpc = RISCV::PseudoLD_RV32_OPT;
  } else if (Opcode == RISCV::SW) {
    NewOpc = RISCV::PseudoSD_RV32_OPT;
  } else {
    return false;
  }

  if (!Op0->hasOneMemOperand() || !Op1->hasOneMemOperand())
    return false;

  // Get offsets and check they are consecutive
  int Offset0 = getMemoryOpOffset(*Op0);
  int Offset1 = getMemoryOpOffset(*Op1);

  // Offsets must be 4 bytes apart
  if (std::abs(Offset1 - Offset0) != 4)
    return false;

  // Make sure we have the same base register
  Register Base0 = Op0->getOperand(1).getReg();
  Register Base1 = Op1->getOperand(1).getReg();
  if (Base0 != Base1)
    return false;

  // Set output parameters
  if (Offset0 < Offset1) {
    FirstReg = Op0->getOperand(0).getReg();
    SecondReg = Op1->getOperand(0).getReg();
    Offset = Offset0;
  } else {
    FirstReg = Op1->getOperand(0).getReg();
    SecondReg = Op0->getOperand(0).getReg();
    Offset = Offset1;
  }

  BaseReg = Base0;

  // Check that the two destination registers are different
  if (FirstReg == SecondReg)
    return false;

  // For loads, check that neither destination register is the same as the base
  // register This prevents register reuse issues where the first load
  // overwrites the base
  if (Opcode == RISCV::LW) {
    if (FirstReg == BaseReg || SecondReg == BaseReg)
      return false;
  }

  return true;
}

bool RISCVPreAllocZilsdOpt::isSafeToMove(MachineInstr *MI, MachineInstr *Target,
                                         bool MoveForward) {
  // Enhanced safety check with call and terminator handling

  MachineBasicBlock *MBB = MI->getParent();
  MachineBasicBlock::iterator Start = MI->getIterator();
  MachineBasicBlock::iterator End = Target->getIterator();

  if (!MoveForward) {
    std::swap(Start, End);
  }

  // Increment Start to skip the current instruction
  if (Start != MBB->end())
    ++Start;

  Register DefReg = MI->getOperand(0).getReg();
  Register BaseReg = MI->getOperand(1).getReg();

  unsigned ScanCount = 0;
  for (auto It = Start; It != End; ++It, ++ScanCount) {
    // Don't move across calls or terminators
    if (It->isCall() || It->isTerminator()) {
      LLVM_DEBUG(dbgs() << "Cannot move across call/terminator: " << *It);
      return false;
    }

    // Don't move across instructions that modify memory barrier
    if (It->hasUnmodeledSideEffects()) {
      LLVM_DEBUG(dbgs() << "Cannot move across instruction with side effects: "
                        << *It);
      return false;
    }

    // Check if the base register is modified
    if (It->modifiesRegister(BaseReg, TRI)) {
      LLVM_DEBUG(dbgs() << "Base register " << BaseReg
                        << " modified by: " << *It);
      return false;
    }

    // For loads, check if the loaded value is used
    if (MI->mayLoad() &&
        (It->readsRegister(DefReg, TRI) || It->modifiesRegister(DefReg, TRI))) {
      LLVM_DEBUG(dbgs() << "Destination register " << DefReg
                        << " used by: " << *It);
      return false;
    }

    // For stores, check if the stored register is modified
    if (MI->mayStore() && It->modifiesRegister(DefReg, TRI)) {
      LLVM_DEBUG(dbgs() << "Source register " << DefReg
                        << " modified by: " << *It);
      return false;
    }

    // Check for memory operation interference
    if (MI->mayLoadOrStore() && It->mayLoadOrStore() &&
        It->mayAlias(AA, *MI, /*UseTBAA*/ false)) {
      LLVM_DEBUG(dbgs() << "Memory operation interference detected\n");
      return false;
    }
  }

  return true;
}

bool RISCVPreAllocZilsdOpt::rescheduleOps(
    MachineBasicBlock *MBB, SmallVectorImpl<MachineInstr *> &Ops, unsigned Base,
    bool isLoad, DenseMap<MachineInstr *, unsigned> &MI2LocMap) {

  if (Ops.size() < 2)
    return false;

  // Sort by offset
  std::sort(Ops.begin(), Ops.end(), [this](MachineInstr *A, MachineInstr *B) {
    return getMemoryOpOffset(*A) < getMemoryOpOffset(*B);
  });

  bool Modified = false;

  // Try to pair consecutive operations
  for (size_t i = 0; i + 1 < Ops.size(); i++) {
    MachineInstr *Op0 = Ops[i];
    MachineInstr *Op1 = Ops[i + 1];

    // Skip if either instruction was already processed
    if (!Op0->getParent() || !Op1->getParent())
      continue;

    unsigned NewOpc;
    Register FirstReg, SecondReg, BaseReg;
    int Offset;

    if (!canFormLdSdPair(Op0, Op1, NewOpc, FirstReg, SecondReg, BaseReg,
                         Offset))
      continue;

    // Check if we can safely and profitably move the instructions together
    SmallPtrSet<MachineInstr *, 4> MemOps;
    SmallSet<unsigned, 4> MemRegs;
    MemOps.insert(Op0);
    MemRegs.insert(Op0->getOperand(0).getReg().id());

    // Use MI2LocMap to determine which instruction appears later in program
    // order
    bool Op1IsLater = MI2LocMap[Op1] > MI2LocMap[Op0];

    // For loads: move later instruction up (backwards) to earlier instruction
    // For stores: move earlier instruction down (forwards) to later instruction
    MachineInstr *MoveInstr, *TargetInstr;
    if (isLoad) {
      // For loads: move the later instruction to the earlier one
      MoveInstr = Op1IsLater ? Op1 : Op0;
      TargetInstr = Op1IsLater ? Op0 : Op1;
    } else {
      // For stores: move the earlier instruction to the later one
      MoveInstr = Op1IsLater ? Op0 : Op1;
      TargetInstr = Op1IsLater ? Op1 : Op0;
    }

    unsigned Distance = Op1IsLater ? MI2LocMap[Op1] - MI2LocMap[Op0]
                                   : MI2LocMap[Op0] - MI2LocMap[Op1];
    // FIXME: Decide what's maximum distance
    if (!isSafeToMove(MoveInstr, TargetInstr, !isLoad) || Distance > 10)
      continue;

    // Move the instruction to the target position
    MachineBasicBlock::iterator InsertPos = TargetInstr->getIterator();
    ++InsertPos;

    // If we need to move an instruction, do it now
    if (MoveInstr != TargetInstr) {
      MBB->splice(InsertPos, MBB, MoveInstr->getIterator());
    }

    // Create the paired instruction
    MachineInstrBuilder MIB;
    DebugLoc DL = Op0->getDebugLoc();

    if (isLoad) {
      MIB = BuildMI(*MBB, InsertPos, DL, TII->get(NewOpc))
                .addReg(FirstReg, RegState::Define)
                .addReg(SecondReg, RegState::Define)
                .addReg(BaseReg)
                .addImm(Offset);
      ++NumLDFormed;
      LLVM_DEBUG(dbgs() << "Formed LD: " << *MIB << "\n");
    } else {
      MIB = BuildMI(*MBB, InsertPos, DL, TII->get(NewOpc))
                .addReg(FirstReg)
                .addReg(SecondReg)
                .addReg(BaseReg)
                .addImm(Offset);
      ++NumSDFormed;
      LLVM_DEBUG(dbgs() << "Formed SD: " << *MIB << "\n");
    }

    // Copy memory operands
    MIB.cloneMergedMemRefs({Op0, Op1});

    // Add register allocation hints for consecutive registers
    // RISC-V Zilsd requires even/odd register pairs
    // Only set hints for virtual registers (physical registers already have
    // encoding)
    if (FirstReg.isVirtual() && SecondReg.isVirtual()) {
      // For virtual registers, we can't determine even/odd yet, but we can hint
      // that they should be allocated as a consecutive pair
      MRI->setRegAllocationHint(FirstReg, RISCVRI::RegPairEven, SecondReg);
      MRI->setRegAllocationHint(SecondReg, RISCVRI::RegPairOdd, FirstReg);
    }

    // Remove the original instructions
    Op0->eraseFromParent();
    Op1->eraseFromParent();

    Modified = true;

    // Skip the next instruction since we've already processed it
    i++;
  }

  return Modified;
}

bool RISCVPreAllocZilsdOpt::isMemoryOp(const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  if (Opcode != RISCV::LW && Opcode != RISCV::SW)
    return false;

  if (!MI.getOperand(1).isReg())
    return false;

  // When no memory operands are present, conservatively assume unaligned,
  // volatile, unfoldable.
  if (!MI.hasOneMemOperand())
    return false;

  const MachineMemOperand *MMO = *MI.memoperands_begin();

  // Check alignment: default is 8-byte, but allow 4-byte with tune feature
  Align RequiredAlign = STI->allowZilsd4ByteAlign() ? Align(4) : Align(8);
  if (MMO->getAlign() < RequiredAlign)
    return false;

  if (MMO->isVolatile() || MMO->isAtomic())
    return false;

  // sw <undef> could probably be eliminated entirely, but for now we just want
  // to avoid making a mess of it.
  if (MI.getOperand(0).isReg() && MI.getOperand(0).isUndef())
    return false;

  // Likewise don't mess with references to undefined addresses.
  if (MI.getOperand(1).isUndef())
    return false;

  return true;
}

bool RISCVPreAllocZilsdOpt::rescheduleLoadStoreInstrs(MachineBasicBlock *MBB) {
  bool Modified = false;

  // Process the basic block in windows delimited by calls, terminators,
  // or instructions with duplicate base+offset pairs
  MachineBasicBlock::iterator MBBI = MBB->begin();
  MachineBasicBlock::iterator E = MBB->end();

  while (MBBI != E) {
    // Map from instruction to its location in the current window
    DenseMap<MachineInstr *, unsigned> MI2LocMap;

    // Map from base register to list of load/store instructions
    using Base2InstMap = DenseMap<unsigned, SmallVector<MachineInstr *, 4>>;
    using BaseVec = SmallVector<unsigned, 4>;
    Base2InstMap Base2LdsMap;
    Base2InstMap Base2StsMap;
    BaseVec LdBases;
    BaseVec StBases;

    unsigned Loc = 0;

    // Build the current window of instructions
    for (; MBBI != E; ++MBBI) {
      MachineInstr &MI = *MBBI;

      // Stop at barriers (calls and terminators)
      if (MI.isCall() || MI.isTerminator()) {
        // Move past the barrier for next iteration
        ++MBBI;
        break;
      }

      // Track instruction location in window
      if (!MI.isDebugInstr())
        MI2LocMap[&MI] = ++Loc;

      // Skip non-memory operations
      if (!isMemoryOp(MI))
        continue;

      bool isLd = (MI.getOpcode() == RISCV::LW);
      Register Base = MI.getOperand(1).getReg();
      int Offset = getMemoryOpOffset(MI);
      bool StopHere = false;

      // Lambda to find or add base register entries
      auto FindBases = [&](Base2InstMap &Base2Ops, BaseVec &Bases) {
        auto [BI, Inserted] = Base2Ops.try_emplace(Base.id());
        if (Inserted) {
          // First time seeing this base register
          BI->second.push_back(&MI);
          Bases.push_back(Base.id());
          return;
        }
        // Check if we've seen this exact base+offset before
        for (const MachineInstr *PrevMI : BI->second) {
          if (Offset == getMemoryOpOffset(*PrevMI)) {
            // Found duplicate base+offset - stop here to process current window
            StopHere = true;
            break;
          }
        }
        if (!StopHere)
          BI->second.push_back(&MI);
      };

      if (isLd)
        FindBases(Base2LdsMap, LdBases);
      else
        FindBases(Base2StsMap, StBases);

      if (StopHere) {
        // Found a duplicate (a base+offset combination that's seen earlier).
        // Backtrack to process the current window.
        --Loc;
        break;
      }
    }

    // Process the current window - reschedule loads
    for (unsigned Base : LdBases) {
      SmallVectorImpl<MachineInstr *> &Lds = Base2LdsMap[Base];
      if (Lds.size() > 1) {
        Modified |= rescheduleOps(MBB, Lds, Base, true, MI2LocMap);
      }
    }

    // Process the current window - reschedule stores
    for (unsigned Base : StBases) {
      SmallVectorImpl<MachineInstr *> &Sts = Base2StsMap[Base];
      if (Sts.size() > 1) {
        Modified |= rescheduleOps(MBB, Sts, Base, false, MI2LocMap);
      }
    }

    // Clear all buffers before processing next window
    Base2LdsMap.clear();
    Base2StsMap.clear();
    LdBases.clear();
    StBases.clear();
    MI2LocMap.clear();

    // If we stopped at a duplicate, move past it for the next window
    if (MBBI != E && !MBBI->isCall() && !MBBI->isTerminator()) {
      ++MBBI;
    }
  }

  return Modified;
}

//===----------------------------------------------------------------------===//
// Post-allocation pass implementation
//===----------------------------------------------------------------------===//

bool RISCVPostAllocZilsdOpt::runOnMachineFunction(MachineFunction &MF) {
  if (DisableZilsdOpt || skipFunction(MF.getFunction()))
    return false;

  STI = &MF.getSubtarget<RISCVSubtarget>();

  // Only run on RV32 with Zilsd extension
  if (STI->is64Bit() || !STI->hasStdExtZilsd())
    return false;

  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool Modified = false;

  for (auto &MBB : MF) {
    for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E;) {
      if (fixInvalidRegPairOp(MBB, MBBI)) {
        Modified = true;
        // Iterator was updated by fixInvalidRegPairOp
      } else {
        ++MBBI;
      }
    }
  }

  return Modified;
}

bool RISCVPostAllocZilsdOpt::isConsecutiveRegPair(Register First,
                                                  Register Second) {
  // Special case: both registers are zero register - this is valid for storing zeros
  if (First == RISCV::X0 && Second == RISCV::X0)
    return true;

  // Check if registers form a valid even/odd pair for Zilsd
  unsigned FirstNum = TRI->getEncodingValue(First);
  unsigned SecondNum = TRI->getEncodingValue(Second);

  // Must be consecutive and first must be even
  return (FirstNum % 2 == 0) && (SecondNum == FirstNum + 1);
}

void RISCVPostAllocZilsdOpt::splitLdSdIntoTwo(MachineBasicBlock &MBB,
                                              MachineBasicBlock::iterator &MBBI,
                                              bool isLoad) {
  MachineInstr *MI = &*MBBI;
  DebugLoc DL = MI->getDebugLoc();

  Register FirstReg = MI->getOperand(0).getReg();
  Register SecondReg = MI->getOperand(1).getReg();
  Register BaseReg = MI->getOperand(2).getReg();
  int Offset = MI->getOperand(3).getImm();

  unsigned Opc = isLoad ? RISCV::LW : RISCV::SW;

  // Create two separate instructions
  if (isLoad) {
    auto MIB1 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
                    .addReg(FirstReg, RegState::Define)
                    .addReg(BaseReg)
                    .addImm(Offset);

    auto MIB2 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
                    .addReg(SecondReg, RegState::Define)
                    .addReg(BaseReg)
                    .addImm(Offset + 4);

    // Copy memory operands if the original instruction had them
    // FIXME: This is overly conservative; the new instruction accesses 4 bytes,
    // not 8.
    if (MI->memoperands_begin() != MI->memoperands_end()) {
      MIB1.cloneMemRefs(*MI);
      MIB2.cloneMemRefs(*MI);
    }

    ++NumLD2LW;
    LLVM_DEBUG(dbgs() << "Split LD back to two LW instructions\n");
  } else {
    auto MIB1 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
                    .addReg(FirstReg)
                    .addReg(BaseReg)
                    .addImm(Offset);

    auto MIB2 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
                    .addReg(SecondReg)
                    .addReg(BaseReg)
                    .addImm(Offset + 4);

    // Copy memory operands if the original instruction had them
    // FIXME: This is overly conservative; the new instruction accesses 4 bytes,
    // not 8.
    if (MI->memoperands_begin() != MI->memoperands_end()) {
      MIB1.cloneMemRefs(*MI);
      MIB2.cloneMemRefs(*MI);
    }

    ++NumSD2SW;
    LLVM_DEBUG(dbgs() << "Split SD back to two SW instructions\n");
  }

  // Remove the original paired instruction and update iterator
  MBBI = MBB.erase(MBBI);
}

bool RISCVPostAllocZilsdOpt::fixInvalidRegPairOp(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI) {
  MachineInstr *MI = &*MBBI;
  unsigned Opcode = MI->getOpcode();

  // Check if this is a Zilsd pseudo that needs fixing
  if (Opcode != RISCV::PseudoLD_RV32_OPT && Opcode != RISCV::PseudoSD_RV32_OPT)
    return false;

  bool isLoad = (Opcode == RISCV::PseudoLD_RV32_OPT);

  Register FirstReg = MI->getOperand(0).getReg();
  Register SecondReg = MI->getOperand(1).getReg();

  // Check if we have valid consecutive registers
  if (!isConsecutiveRegPair(FirstReg, SecondReg)) {
    // Need to split back into two instructions
    splitLdSdIntoTwo(MBB, MBBI, isLoad);
    return true;
  }

  // Registers are valid, convert to real LD/SD instruction
  Register BaseReg = MI->getOperand(2).getReg();
  int Offset = MI->getOperand(3).getImm();
  DebugLoc DL = MI->getDebugLoc();

  unsigned RealOpc = isLoad ? RISCV::LD_RV32 : RISCV::SD_RV32;

  // Create register pair from the two individual registers
  unsigned RegPair = TRI->getMatchingSuperReg(FirstReg, RISCV::sub_gpr_even,
                                              &RISCV::GPRPairRegClass);
  // Create the real LD/SD instruction with register pair
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(RealOpc));

  if (isLoad) {
    // For LD, the register pair is the destination
    MIB.addReg(RegPair, RegState::Define);
  } else {
    // For SD, the register pair is the source
    MIB.addReg(RegPair);
  }

  MIB.addReg(BaseReg).addImm(Offset);

  // Copy memory operands if the original instruction had them
  if (MI->memoperands_begin() != MI->memoperands_end())
    MIB.cloneMemRefs(*MI);

  LLVM_DEBUG(dbgs() << "Converted pseudo to real instruction: " << *MIB
                    << "\n");

  // Remove the pseudo instruction and update iterator
  MBBI = MBB.erase(MBBI);

  return true;
}

//===----------------------------------------------------------------------===//
// Pass creation functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createRISCVPreAllocZilsdOptPass() {
  return new RISCVPreAllocZilsdOpt();
}

FunctionPass *llvm::createRISCVPostAllocZilsdOptPass() {
  return new RISCVPostAllocZilsdOpt();
}
