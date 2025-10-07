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
// Note: second phase is integrated into RISCVLoadStoreOptimizer
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

static cl::opt<bool>
    DisableZilsdOpt("disable-riscv-zilsd-opt", cl::Hidden, cl::init(false),
                    cl::desc("Disable Zilsd load/store optimization"));

static cl::opt<unsigned> MaxRescheduleDistance(
    "riscv-zilsd-max-reschedule-distance", cl::Hidden, cl::init(10),
    cl::desc("Maximum distance for rescheduling load/store instructions"));

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
                     bool IsLoad,
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

} // end anonymous namespace

char RISCVPreAllocZilsdOpt::ID = 0;

INITIALIZE_PASS_BEGIN(RISCVPreAllocZilsdOpt, "riscv-prera-zilsd-opt",
                      "RISC-V pre-allocation Zilsd optimization", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(RISCVPreAllocZilsdOpt, "riscv-prera-zilsd-opt",
                    "RISC-V pre-allocation Zilsd optimization", false, false)

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
  case RISCV::SW: {
    // For LW/SW, the offset is in operand 2
    const MachineOperand &OffsetOp = MI.getOperand(2);

    // Handle immediate offset
    if (OffsetOp.isImm())
      return OffsetOp.getImm();

    // Handle symbolic operands with MO_LO flag (from MergeBaseOffset)
    if (OffsetOp.getTargetFlags() & RISCVII::MO_LO)
      if (OffsetOp.isGlobal() || OffsetOp.isCPI() ||
          OffsetOp.isBlockAddress() || OffsetOp.isSymbol())
        return OffsetOp.getOffset();

    break;
  }
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

  if (Opcode == RISCV::LW)
    NewOpc = RISCV::PseudoLD_RV32_OPT;
  else if (Opcode == RISCV::SW)
    NewOpc = RISCV::PseudoSD_RV32_OPT;
  else
    return false;

  if (!Op0->hasOneMemOperand() || !Op1->hasOneMemOperand())
    return false;

  // Check if operands are compatible for merging
  const MachineOperand &OffsetOp0 = Op0->getOperand(2);
  const MachineOperand &OffsetOp1 = Op1->getOperand(2);

  // Both must be the same type
  if (OffsetOp0.getType() != OffsetOp1.getType())
    return false;

  // If they're symbolic, they must reference the same symbol
  if (!OffsetOp0.isImm()) {
    // Check if both have MO_LO flag
    if ((OffsetOp0.getTargetFlags() & RISCVII::MO_LO) !=
        (OffsetOp1.getTargetFlags() & RISCVII::MO_LO))
      return false;

    // For global addresses, must be the same global
    if (OffsetOp0.isGlobal()) {
      if (!OffsetOp1.isGlobal() ||
          OffsetOp0.getGlobal() != OffsetOp1.getGlobal())
        return false;
    }
    // For constant pool indices, must be the same index
    else if (OffsetOp0.isCPI()) {
      if (!OffsetOp1.isCPI() || OffsetOp0.getIndex() != OffsetOp1.getIndex())
        return false;
    }
    // For symbols, must be the same symbol name
    else if (OffsetOp0.isSymbol()) {
      if (!OffsetOp1.isSymbol() ||
          strcmp(OffsetOp0.getSymbolName(), OffsetOp1.getSymbolName()) != 0)
        return false;
    }
    // For block addresses, must be the same block
    else if (OffsetOp0.isBlockAddress()) {
      if (!OffsetOp1.isBlockAddress() ||
          OffsetOp0.getBlockAddress() != OffsetOp1.getBlockAddress())
        return false;
    }
  }

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
  // register. This prevents register reuse issues where the first load
  // overwrites the base.
  if (Opcode == RISCV::LW) {
    if (FirstReg == BaseReg || SecondReg == BaseReg)
      return false;
  }

  return true;
}

bool RISCVPreAllocZilsdOpt::isSafeToMove(MachineInstr *MI, MachineInstr *Target,
                                         bool MoveForward) {
  MachineBasicBlock *MBB = MI->getParent();
  MachineBasicBlock::iterator Start = MI->getIterator();
  MachineBasicBlock::iterator End = Target->getIterator();

  if (!MoveForward)
    std::swap(Start, End);

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
    bool IsLoad, DenseMap<MachineInstr *, unsigned> &MI2LocMap) {
  // Sort by offset
  llvm::sort(Ops.begin(), Ops.end(), [this](MachineInstr *A, MachineInstr *B) {
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
    if (IsLoad) {
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
    if (!isSafeToMove(MoveInstr, TargetInstr, !IsLoad) ||
        Distance > MaxRescheduleDistance)
      continue;

    // Move the instruction to the target position
    MachineBasicBlock::iterator InsertPos = TargetInstr->getIterator();
    ++InsertPos;

    // If we need to move an instruction, do it now
    if (MoveInstr != TargetInstr)
      MBB->splice(InsertPos, MBB, MoveInstr->getIterator());

    // Create the paired instruction
    MachineInstrBuilder MIB;
    DebugLoc DL = Op0->getDebugLoc();

    if (IsLoad) {
      MIB = BuildMI(*MBB, InsertPos, DL, TII->get(NewOpc))
                .addReg(FirstReg, RegState::Define)
                .addReg(SecondReg, RegState::Define)
                .addReg(BaseReg);
      ++NumLDFormed;
      LLVM_DEBUG(dbgs() << "Formed LD: " << *MIB << "\n");
    } else {
      MIB = BuildMI(*MBB, InsertPos, DL, TII->get(NewOpc))
                .addReg(FirstReg)
                .addReg(SecondReg)
                .addReg(BaseReg);
      ++NumSDFormed;
      LLVM_DEBUG(dbgs() << "Formed SD: " << *MIB << "\n");
    }

    // Add the offset operand - preserve symbolic references
    const MachineOperand &OffsetOp = (Offset == getMemoryOpOffset(*Op0))
                                         ? Op0->getOperand(2)
                                         : Op1->getOperand(2);

    if (OffsetOp.isImm())
      MIB.addImm(Offset);
    else if (OffsetOp.isGlobal())
      MIB.addGlobalAddress(OffsetOp.getGlobal(), Offset,
                           OffsetOp.getTargetFlags());
    else if (OffsetOp.isCPI())
      MIB.addConstantPoolIndex(OffsetOp.getIndex(), Offset,
                               OffsetOp.getTargetFlags());
    else if (OffsetOp.isSymbol())
      MIB.addExternalSymbol(OffsetOp.getSymbolName(),
                            OffsetOp.getTargetFlags());
    else if (OffsetOp.isBlockAddress())
      MIB.addBlockAddress(OffsetOp.getBlockAddress(), Offset,
                          OffsetOp.getTargetFlags());

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
  // If unaligned scalar memory is enabled, allow any alignment
  Align RequiredAlign = STI->enableUnalignedScalarMem() ? Align(1)
                        : STI->allowZilsd4ByteAlign()   ? Align(4)
                                                        : Align(8);
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

      bool IsLd = (MI.getOpcode() == RISCV::LW);
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

      if (IsLd)
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
  }

  return Modified;
}

//===----------------------------------------------------------------------===//
// Pass creation functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createRISCVPreAllocZilsdOptPass() {
  return new RISCVPreAllocZilsdOpt();
}
