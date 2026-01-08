//===----- RISCVLoadStoreOptimizer.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Load/Store Pairing: It identifies pairs of load or store instructions
// operating on consecutive memory locations and merges them into a single
// paired instruction, leveraging hardware support for paired memory accesses.
// Much of the pairing logic is adapted from the AArch64LoadStoreOpt pass.
//
// Post-allocation Zilsd decomposition: Fixes invalid LD/SD instructions if
// register allocation didn't provide suitable consecutive registers.
//
// NOTE: The AArch64LoadStoreOpt pass performs additional optimizations such as
// merging zero store instructions, promoting loads that read directly from a
// preceding store, and merging base register updates with load/store
// instructions (via pre-/post-indexed addressing). These advanced
// transformations are not yet implemented in the RISC-V pass but represent
// potential future enhancements for further optimizing RISC-V memory
// operations.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-load-store-opt"
#define RISCV_LOAD_STORE_OPT_NAME "RISC-V Load / Store Optimizer"

// The LdStLimit limits number of instructions how far we search for load/store
// pairs.
static cl::opt<unsigned> LdStLimit("riscv-load-store-scan-limit", cl::init(128),
                                   cl::Hidden);
STATISTIC(NumLD2LW, "Number of LD instructions split back to LW");
STATISTIC(NumSD2SW, "Number of SD instructions split back to SW");

namespace {

struct RISCVLoadStoreOpt : public MachineFunctionPass {
  static char ID;
  bool runOnMachineFunction(MachineFunction &Fn) override;

  RISCVLoadStoreOpt() : MachineFunctionPass(ID) {}

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_LOAD_STORE_OPT_NAME; }

  // Find and pair load/store instructions.
  bool tryToPairLdStInst(MachineBasicBlock::iterator &MBBI);

  // Convert load/store pairs to single instructions.
  bool tryConvertToLdStPair(MachineBasicBlock::iterator First,
                            MachineBasicBlock::iterator Second);
  bool tryConvertToXqcilsmLdStPair(MachineFunction *MF,
                                   MachineBasicBlock::iterator First,
                                   MachineBasicBlock::iterator Second);
  bool tryConvertToXqcilsmMultiLdSt(MachineBasicBlock::iterator &First);
  bool tryConvertToMIPSLdStPair(MachineFunction *MF,
                                MachineBasicBlock::iterator First,
                                MachineBasicBlock::iterator Second);

  // Scan the instructions looking for a load/store that can be combined
  // with the current instruction into a load/store pair.
  // Return the matching instruction if one is found, else MBB->end().
  MachineBasicBlock::iterator findMatchingInsn(MachineBasicBlock::iterator I,
                                               bool &MergeForward);

  MachineBasicBlock::iterator
  mergePairedInsns(MachineBasicBlock::iterator I,
                   MachineBasicBlock::iterator Paired, bool MergeForward);

  // Post reg-alloc zilsd part
  bool fixInvalidRegPairOp(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MBBI);
  bool isValidZilsdRegPair(Register First, Register Second);
  void splitLdSdIntoTwo(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &MBBI, bool IsLoad);

private:
  AliasAnalysis *AA;
  MachineRegisterInfo *MRI;
  const RISCVInstrInfo *TII;
  const RISCVRegisterInfo *TRI;
  const RISCVSubtarget *STI = nullptr;
  LiveRegUnits ModifiedRegUnits, UsedRegUnits;
};
} // end anonymous namespace

char RISCVLoadStoreOpt::ID = 0;
INITIALIZE_PASS(RISCVLoadStoreOpt, DEBUG_TYPE, RISCV_LOAD_STORE_OPT_NAME, false,
                false)

bool RISCVLoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;

  bool MadeChange = false;
  STI = &Fn.getSubtarget<RISCVSubtarget>();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &Fn.getRegInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  ModifiedRegUnits.init(*TRI);
  UsedRegUnits.init(*TRI);

  if (STI->useMIPSLoadStorePairs() || STI->hasVendorXqcilsm()) {
    for (MachineBasicBlock &MBB : Fn) {
      LLVM_DEBUG(dbgs() << "MBB: " << MBB.getName() << "\n");

      for (MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
           MBBI != E;) {
        if (TII->isPairableLdStInstOpc(MBBI->getOpcode()) &&
            tryToPairLdStInst(MBBI))
          MadeChange = true;
        else
          ++MBBI;
      }
    }
  }

  if (!STI->is64Bit() && STI->hasStdExtZilsd()) {
    for (auto &MBB : Fn) {
      for (auto MBBI = MBB.begin(), E = MBB.end(); MBBI != E;) {
        if (fixInvalidRegPairOp(MBB, MBBI)) {
          MadeChange = true;
          // Iterator was updated by fixInvalidRegPairOp
        } else {
          ++MBBI;
        }
      }
    }
  }

  return MadeChange;
}

// Find loads and stores that can be merged into a single load or store pair
// instruction.
bool RISCVLoadStoreOpt::tryToPairLdStInst(MachineBasicBlock::iterator &MBBI) {
  MachineInstr &MI = *MBBI;

  // If this is volatile, it is not a candidate.
  if (MI.hasOrderedMemoryRef())
    return false;

  if (!TII->isLdStSafeToPair(MI, TRI))
    return false;

  // If Xqcilsm is available, first try to form a multi-instruction group (>2).
  if (!STI->is64Bit() && STI->hasVendorXqcilsm()) {
    if (tryConvertToXqcilsmMultiLdSt(MBBI))
      return true;
  }

  // Look ahead for a pairable instruction.
  MachineBasicBlock::iterator E = MI.getParent()->end();
  bool MergeForward;
  MachineBasicBlock::iterator Paired = findMatchingInsn(MBBI, MergeForward);
  if (Paired != E) {
    MBBI = mergePairedInsns(MBBI, Paired, MergeForward);
    return true;
  }
  return false;
}

static bool isMemOpAligned(MachineInstr &MI, Align RequiredAlignment) {
  const MachineMemOperand *MMO = *MI.memoperands_begin();
  Align MMOAlign = MMO->getAlign();
  return MMOAlign >= RequiredAlignment;
}

// Convert set of 3 or more LW/SW instructions to QC_LWMI/QC_SWMI/QC_SETWMI.
// For now this only handles consecutive loads and stores traversing the basic
// block top-down.
// TODO: Traverse the basic block bottom-up as well.
bool RISCVLoadStoreOpt::tryConvertToXqcilsmMultiLdSt(
    MachineBasicBlock::iterator &FirstIt) {
  MachineInstr &FirstMI = *FirstIt;
  MachineFunction *MF = FirstMI.getMF();

  if (STI->is64Bit() || !STI->hasVendorXqcilsm())
    return false;

  unsigned Opc = FirstMI.getOpcode();
  if (Opc != RISCV::LW && Opc != RISCV::SW)
    return false;

  if (!FirstMI.hasOneMemOperand())
    return false;

  if (!isMemOpAligned(FirstMI, Align(4)))
    return false;

  // Require simple reg+imm addressing.
  const MachineOperand &BaseOp = FirstMI.getOperand(1);
  const MachineOperand &OffOp = FirstMI.getOperand(2);
  if (!BaseOp.isReg() || !OffOp.isImm())
    return false;

  Register Base = BaseOp.getReg();
  int64_t BaseOff = OffOp.getImm();

  if (!isShiftedUInt<5, 2>(BaseOff))
    return false;

  Register StartReg = FirstMI.getOperand(0).getReg();
  bool IsLoad = (Opc == RISCV::LW);

  // Load rd cannot be x0 and must not clobber the base register.
  if (IsLoad) {
    if (StartReg == RISCV::X0)
      return false;
    if (StartReg == Base)
      return false;
  }

  // Collect a set of consecutive matching instructions.
  SmallVector<MachineInstr *, 8> Group;
  Group.push_back(&FirstMI);

  MachineBasicBlock::iterator E = FirstIt->getParent()->end();
  MachineBasicBlock::iterator It = next_nodbg(FirstIt, E);
  int64_t ExpectedOff = BaseOff + 4;
  unsigned Index = 1;
  enum class StoreMode { Unknown, Setwmi, Swmi };
  StoreMode SMode = StoreMode::Unknown;

  while (It != E) {
    MachineInstr &MI = *It;

    if (!TII->isPairableLdStInstOpc(MI.getOpcode()))
      break;
    if (MI.getOpcode() != Opc)
      break;
    if (!TII->isLdStSafeToPair(MI, TRI))
      break;
    if (!MI.hasOneMemOperand())
      break;
    if (!isMemOpAligned(MI, Align(4)))
      break;

    const MachineOperand &BaseMIOp = MI.getOperand(1);
    const MachineOperand &OffsetMIOp = MI.getOperand(2);
    if (!BaseMIOp.isReg() || !OffsetMIOp.isImm())
      break;
    if (BaseMIOp.getReg() != Base)
      break;
    int64_t Off = OffsetMIOp.getImm();
    if (Off != ExpectedOff)
      break;

    Register Reg = MI.getOperand(0).getReg();
    if (IsLoad) {
      // For loads, require consecutive destination registers.
      if (Reg != StartReg + Index)
        break;
      if (Reg == Base)
        break;
    } else {
      // For stores, decide mode based on the second instruction and then
      // enforce the same for the rest.
      if (SMode == StoreMode::Unknown) {
        if (Reg == StartReg)
          SMode = StoreMode::Setwmi;
        else if (Reg == StartReg + 1)
          SMode = StoreMode::Swmi;
        else
          break;
      } else if (SMode == StoreMode::Setwmi) {
        if (Reg != StartReg)
          break;
      } else {
        if (Reg != StartReg + Index)
          break;
      }
    }

    // Passed checks, extend the group.
    Group.push_back(&MI);
    ++Index;
    ExpectedOff += 4;
    It = next_nodbg(It, E);
  }

  // We only handle more than 2 here. Pairs are handled in
  // tryConvertToXqcilsmLdStPair.
  unsigned Len = Group.size();
  if (Len < 3 || Len > 31)
    return false;

  unsigned NewOpc;
  unsigned StartRegState;
  bool AddImplicitRegs = true;

  if (IsLoad) {
    NewOpc = RISCV::QC_LWMI;
    StartRegState = static_cast<unsigned>(RegState::Define);
  } else {
    assert(SMode != StoreMode::Unknown &&
           "Group should be large enough to know the store mode");
    if (SMode == StoreMode::Setwmi) {
      NewOpc = RISCV::QC_SETWMI;
      // Kill if any of the individual stores killed the reg.
      bool StartKill = false;
      for (MachineInstr *MI : Group)
        StartKill |= MI->getOperand(0).isKill();
      StartRegState = getKillRegState(StartKill);
      AddImplicitRegs = false;
    } else {
      // SWMI requires consecutive source regs and rd != x0.
      if (StartReg == RISCV::X0)
        return false;
      NewOpc = RISCV::QC_SWMI;
      StartRegState = getKillRegState(Group.front()->getOperand(0).isKill());
    }
  }

  // Aggregate kill on base.
  bool BaseKill = false;
  for (MachineInstr *MI : Group)
    BaseKill |= MI->getOperand(1).isKill();

  // Build the new instruction.
  DebugLoc DL = FirstMI.getDebugLoc();
  if (!DL)
    DL = Group.back()->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(*MF, DL, TII->get(NewOpc));
  MIB.addReg(StartReg, StartRegState)
      .addReg(Base, getKillRegState(BaseKill))
      .addImm(Len)
      .addImm(BaseOff);

  // Merge memory references.
  MIB.cloneMergedMemRefs(Group);

  if (AddImplicitRegs) {
    // Add implicit operands for the additional registers.
    for (unsigned i = 1; i < Len; ++i) {
      Register R = StartReg + i;
      unsigned State = 0;
      if (IsLoad)
        State = static_cast<unsigned>(RegState::ImplicitDefine);
      else
        State = RegState::Implicit |
                getKillRegState(Group[i]->getOperand(0).isKill());
      MIB.addReg(R, State);
    }
  }

  // Insert before the first instruction and remove all in the group.
  MachineBasicBlock *MBB = FirstIt->getParent();
  MachineBasicBlock::iterator NewIt = MBB->insert(FirstIt, MIB);
  for (MachineInstr *MI : Group)
    MI->removeFromParent();

  // Advance the cursor to the next non-debug instruction after the group.
  FirstIt = next_nodbg(NewIt, MBB->end());
  return true;
}

bool RISCVLoadStoreOpt::tryConvertToXqcilsmLdStPair(
    MachineFunction *MF, MachineBasicBlock::iterator First,
    MachineBasicBlock::iterator Second) {
  unsigned Opc = First->getOpcode();
  if ((Opc != RISCV::LW && Opc != RISCV::SW) || Second->getOpcode() != Opc)
    return false;

  const auto &FirstOp1 = First->getOperand(1);
  const auto &SecondOp1 = Second->getOperand(1);
  const auto &FirstOp2 = First->getOperand(2);
  const auto &SecondOp2 = Second->getOperand(2);

  // Require simple reg+imm addressing for both.
  if (!FirstOp1.isReg() || !SecondOp1.isReg() || !FirstOp2.isImm() ||
      !SecondOp2.isImm())
    return false;

  Register Base1 = FirstOp1.getReg();
  Register Base2 = SecondOp1.getReg();

  if (Base1 != Base2)
    return false;

  if (!First->hasOneMemOperand() || !Second->hasOneMemOperand())
    return false;

  if (!isMemOpAligned(*First, Align(4)) || !isMemOpAligned(*Second, Align(4)))
    return false;

  auto &FirstOp0 = First->getOperand(0);
  auto &SecondOp0 = Second->getOperand(0);

  int64_t Off1 = FirstOp2.getImm();
  int64_t Off2 = SecondOp2.getImm();

  if (Off2 < Off1) {
    std::swap(FirstOp0, SecondOp0);
    std::swap(Off1, Off2);
  }

  if (!isShiftedUInt<5, 2>(Off1) || (Off2 - Off1 != 4))
    return false;

  Register StartReg = FirstOp0.getReg();
  Register NextReg = SecondOp0.getReg();

  unsigned XqciOpc;
  unsigned StartRegState;
  unsigned NextRegState = 0;
  bool AddNextReg = true;

  if (Opc == RISCV::LW) {

    if (StartReg == RISCV::X0)
      return false;

    // If the base reg gets overwritten by one of the loads bail out.
    if (StartReg == Base1 || NextReg == Base1)
      return false;

    // The registers need to be consecutive.
    if (NextReg != StartReg + 1)
      return false;

    XqciOpc = RISCV::QC_LWMI;
    StartRegState = static_cast<unsigned>(RegState::Define);
    NextRegState = static_cast<unsigned>(RegState::ImplicitDefine);
  } else {
    assert(Opc == RISCV::SW && "Expected a SW instruction");
    if (StartReg == NextReg) {
      XqciOpc = RISCV::QC_SETWMI;
      StartRegState = getKillRegState(FirstOp0.isKill() || SecondOp0.isKill());
      AddNextReg = false;
    } else if (NextReg == StartReg + 1 && StartReg != RISCV::X0) {
      XqciOpc = RISCV::QC_SWMI;
      StartRegState = getKillRegState(FirstOp0.isKill());
      NextRegState = RegState::Implicit | getKillRegState(SecondOp0.isKill());
    } else {
      return false;
    }
  }

  DebugLoc DL =
      First->getDebugLoc() ? First->getDebugLoc() : Second->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(*MF, DL, TII->get(XqciOpc));
  MIB.addReg(StartReg, StartRegState)
      .addReg(Base1, getKillRegState(FirstOp1.isKill() || SecondOp1.isKill()))
      .addImm(2)
      .addImm(Off1)
      .cloneMergedMemRefs({&*First, &*Second});

  if (AddNextReg)
    MIB.addReg(NextReg, NextRegState);

  First->getParent()->insert(First, MIB);
  First->removeFromParent();
  Second->removeFromParent();

  return true;
}

bool RISCVLoadStoreOpt::tryConvertToMIPSLdStPair(
    MachineFunction *MF, MachineBasicBlock::iterator First,
    MachineBasicBlock::iterator Second) {
  // Try converting to SWP/LWP/LDP/SDP.
  // SWP/LWP requires 8-byte alignment whereas LDP/SDP needs 16-byte alignment.
  unsigned PairOpc;
  Align RequiredAlignment;
  switch (First->getOpcode()) {
  default:
    llvm_unreachable("Unsupported load/store instruction for pairing");
  case RISCV::SW:
    PairOpc = RISCV::MIPS_SWP;
    RequiredAlignment = Align(8);
    break;
  case RISCV::LW:
    PairOpc = RISCV::MIPS_LWP;
    RequiredAlignment = Align(8);
    break;
  case RISCV::SD:
    PairOpc = RISCV::MIPS_SDP;
    RequiredAlignment = Align(16);
    break;
  case RISCV::LD:
    PairOpc = RISCV::MIPS_LDP;
    RequiredAlignment = Align(16);
    break;
  }

  if (!First->hasOneMemOperand())
    return false;

  if (!isMemOpAligned(*First, RequiredAlignment))
    return false;

  int64_t Offset = First->getOperand(2).getImm();
  if (!isUInt<7>(Offset))
    return false;

  MachineInstrBuilder MIB = BuildMI(
      *MF, First->getDebugLoc() ? First->getDebugLoc() : Second->getDebugLoc(),
      TII->get(PairOpc));
  MIB.add(First->getOperand(0))
      .add(Second->getOperand(0))
      .add(First->getOperand(1))
      .add(First->getOperand(2))
      .cloneMergedMemRefs({&*First, &*Second});

  First->getParent()->insert(First, MIB);

  First->removeFromParent();
  Second->removeFromParent();

  return true;
}

// Merge two adjacent load/store instructions into a paired instruction.
// This function calls the vendor specific implementation that seelects the
// appropriate paired opcode, verifies that the memory operand is properly
// aligned, and checks that the offset is valid. If all conditions are met, it
// builds and inserts the paired instruction.
bool RISCVLoadStoreOpt::tryConvertToLdStPair(
    MachineBasicBlock::iterator First, MachineBasicBlock::iterator Second) {
  MachineFunction *MF = First->getMF();

  // Try converting to QC_LWMI/QC_SWMI if the XQCILSM extension is enabled.
  if (!STI->is64Bit() && STI->hasVendorXqcilsm())
    return tryConvertToXqcilsmLdStPair(MF, First, Second);

  // Else try to convert them into MIPS Paired Loads/Stores.
  return tryConvertToMIPSLdStPair(MF, First, Second);
}

static bool mayAlias(MachineInstr &MIa,
                     SmallVectorImpl<MachineInstr *> &MemInsns,
                     AliasAnalysis *AA) {
  for (MachineInstr *MIb : MemInsns)
    if (MIa.mayAlias(AA, *MIb, /*UseTBAA*/ false))
      return true;

  return false;
}

// Scan the instructions looking for a load/store that can be combined with the
// current instruction into a wider equivalent or a load/store pair.
// TODO: Extend pairing logic to consider reordering both instructions
// to a safe "middle" position rather than only merging forward/backward.
// This requires more sophisticated checks for aliasing, register
// liveness, and potential scheduling hazards.
MachineBasicBlock::iterator
RISCVLoadStoreOpt::findMatchingInsn(MachineBasicBlock::iterator I,
                                    bool &MergeForward) {
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator MBBI = I;
  MachineInstr &FirstMI = *I;
  MBBI = next_nodbg(MBBI, E);

  bool MayLoad = FirstMI.mayLoad();
  Register Reg = FirstMI.getOperand(0).getReg();
  Register BaseReg = FirstMI.getOperand(1).getReg();
  int64_t Offset = FirstMI.getOperand(2).getImm();
  int64_t OffsetStride = (*FirstMI.memoperands_begin())->getSize().getValue();

  MergeForward = false;

  // Track which register units have been modified and used between the first
  // insn (inclusive) and the second insn.
  ModifiedRegUnits.clear();
  UsedRegUnits.clear();

  // Remember any instructions that read/write memory between FirstMI and MI.
  SmallVector<MachineInstr *, 4> MemInsns;

  for (unsigned Count = 0; MBBI != E && Count < LdStLimit;
       MBBI = next_nodbg(MBBI, E)) {
    MachineInstr &MI = *MBBI;

    // Don't count transient instructions towards the search limit since there
    // may be different numbers of them if e.g. debug information is present.
    if (!MI.isTransient())
      ++Count;

    if (MI.getOpcode() == FirstMI.getOpcode() &&
        TII->isLdStSafeToPair(MI, TRI)) {
      Register MIBaseReg = MI.getOperand(1).getReg();
      int64_t MIOffset = MI.getOperand(2).getImm();

      if (BaseReg == MIBaseReg) {
        if ((Offset != MIOffset + OffsetStride) &&
            (Offset + OffsetStride != MIOffset)) {
          LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits,
                                            TRI);
          MemInsns.push_back(&MI);
          continue;
        }

        // If the destination register of one load is the same register or a
        // sub/super register of the other load, bail and keep looking.
        if (MayLoad &&
            TRI->isSuperOrSubRegisterEq(Reg, MI.getOperand(0).getReg())) {
          LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits,
                                            TRI);
          MemInsns.push_back(&MI);
          continue;
        }

        // If the BaseReg has been modified, then we cannot do the optimization.
        if (!ModifiedRegUnits.available(BaseReg))
          return E;

        // If the Rt of the second instruction was not modified or used between
        // the two instructions and none of the instructions between the second
        // and first alias with the second, we can combine the second into the
        // first.
        if (ModifiedRegUnits.available(MI.getOperand(0).getReg()) &&
            !(MI.mayLoad() &&
              !UsedRegUnits.available(MI.getOperand(0).getReg())) &&
            !mayAlias(MI, MemInsns, AA)) {

          MergeForward = false;
          return MBBI;
        }

        // Likewise, if the Rt of the first instruction is not modified or used
        // between the two instructions and none of the instructions between the
        // first and the second alias with the first, we can combine the first
        // into the second.
        if (!(MayLoad &&
              !UsedRegUnits.available(FirstMI.getOperand(0).getReg())) &&
            !mayAlias(FirstMI, MemInsns, AA)) {

          if (ModifiedRegUnits.available(FirstMI.getOperand(0).getReg())) {
            MergeForward = true;
            return MBBI;
          }
        }
        // Unable to combine these instructions due to interference in between.
        // Keep looking.
      }
    }

    // If the instruction wasn't a matching load or store.  Stop searching if we
    // encounter a call instruction that might modify memory.
    if (MI.isCall())
      return E;

    // Update modified / uses register units.
    LiveRegUnits::accumulateUsedDefed(MI, ModifiedRegUnits, UsedRegUnits, TRI);

    // Otherwise, if the base register is modified, we have no match, so
    // return early.
    if (!ModifiedRegUnits.available(BaseReg))
      return E;

    // Update list of instructions that read/write memory.
    if (MI.mayLoadOrStore())
      MemInsns.push_back(&MI);
  }
  return E;
}

MachineBasicBlock::iterator
RISCVLoadStoreOpt::mergePairedInsns(MachineBasicBlock::iterator I,
                                    MachineBasicBlock::iterator Paired,
                                    bool MergeForward) {
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator NextI = next_nodbg(I, E);
  // If NextI is the second of the two instructions to be merged, skip one
  // further for now. For the MIPS load/store, the merge will invalidate the
  // iterator, and we don't need to scan the new instruction, as it's a pairwise
  // instruction, which we're not considering for further action anyway. For the
  // Xqcilsm load/store, we may not want to do this as the second instruction
  // could possibly be the first in another pair if we do not merge here. This
  // is handled in the else block after the call to tryConvertToLdStPair below.
  if (NextI == Paired)
    NextI = next_nodbg(NextI, E);

  // Insert our new paired instruction after whichever of the paired
  // instructions MergeForward indicates.
  MachineBasicBlock::iterator InsertionPoint = MergeForward ? Paired : I;
  MachineBasicBlock::iterator DeletionPoint = MergeForward ? I : Paired;
  int Offset = I->getOperand(2).getImm();
  int PairedOffset = Paired->getOperand(2).getImm();
  bool InsertAfter = (Offset < PairedOffset) ^ MergeForward;

  if (!MergeForward)
    Paired->getOperand(1).setIsKill(false);

  // Kill flags may become invalid when moving stores for pairing.
  if (I->getOperand(0).isUse()) {
    if (!MergeForward) {
      // Check if the Paired store's source register has a kill flag and clear
      // it only if there are intermediate uses between I and Paired.
      MachineOperand &PairedRegOp = Paired->getOperand(0);
      if (PairedRegOp.isKill()) {
        for (auto It = std::next(I); It != Paired; ++It) {
          if (It->readsRegister(PairedRegOp.getReg(), TRI)) {
            PairedRegOp.setIsKill(false);
            break;
          }
        }
      }
    } else {
      // Clear kill flags of the first store's register in the forward
      // direction.
      Register Reg = I->getOperand(0).getReg();
      for (MachineInstr &MI : make_range(std::next(I), std::next(Paired)))
        MI.clearRegisterKills(Reg, TRI);
    }
  }

  MachineInstr *ToInsert = DeletionPoint->removeFromParent();
  MachineBasicBlock &MBB = *InsertionPoint->getParent();
  MachineBasicBlock::iterator First, Second;

  if (!InsertAfter) {
    First = MBB.insert(InsertionPoint, ToInsert);
    Second = InsertionPoint;
  } else {
    Second = MBB.insertAfter(InsertionPoint, ToInsert);
    First = InsertionPoint;
  }

  if (tryConvertToLdStPair(First, Second)) {
    LLVM_DEBUG(dbgs() << "Pairing load/store:\n    ");
    LLVM_DEBUG(prev_nodbg(NextI, MBB.begin())->print(dbgs()));
  } else if (!STI->is64Bit() && STI->hasVendorXqcilsm()) {
    // We were unable to form the pair, so use the next non-debug instruction
    // after the first instruction we had wanted to merge.
    NextI = next_nodbg(I, E);
  }

  return NextI;
}

//===----------------------------------------------------------------------===//
// Post reg-alloc zilsd pass implementation
//===----------------------------------------------------------------------===//

bool RISCVLoadStoreOpt::isValidZilsdRegPair(Register First, Register Second) {
  // Special case: First register can not be zero unless both registers are
  // zeros.
  // Spec says: LD instructions with destination x0 are processed as any other
  // load, but the result is discarded entirely and x1 is not written. If using
  // x0 as src of SD, the entire 64-bit operand is zero — i.e., register x1 is
  // not accessed.
  if (First == RISCV::X0)
    return Second == RISCV::X0;

  // Check if registers form a valid even/odd pair for Zilsd
  unsigned FirstNum = TRI->getEncodingValue(First);
  unsigned SecondNum = TRI->getEncodingValue(Second);

  // Must be consecutive and first must be even
  return (FirstNum % 2 == 0) && (SecondNum == FirstNum + 1);
}

void RISCVLoadStoreOpt::splitLdSdIntoTwo(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator &MBBI,
                                         bool IsLoad) {
  MachineInstr *MI = &*MBBI;
  DebugLoc DL = MI->getDebugLoc();

  const MachineOperand &FirstOp = MI->getOperand(0);
  const MachineOperand &SecondOp = MI->getOperand(1);
  const MachineOperand &BaseOp = MI->getOperand(2);
  Register FirstReg = FirstOp.getReg();
  Register SecondReg = SecondOp.getReg();
  Register BaseReg = BaseOp.getReg();

  // Handle both immediate and symbolic operands for offset
  const MachineOperand &OffsetOp = MI->getOperand(3);
  int BaseOffset;
  if (OffsetOp.isImm())
    BaseOffset = OffsetOp.getImm();
  else
    // For symbolic operands, extract the embedded offset
    BaseOffset = OffsetOp.getOffset();

  unsigned Opc = IsLoad ? RISCV::LW : RISCV::SW;
  MachineInstrBuilder MIB1, MIB2;

  // Create two separate instructions
  if (IsLoad) {
    // It's possible that first register is same as base register, when we split
    // it becomes incorrect because base register is overwritten, e.g.
    // X10, X13 = PseudoLD_RV32_OPT killed X10, 0
    // =>
    // X10 = LW X10, 0
    // X13 = LW killed X10, 4
    // we can just switch the order to resolve that:
    // X13 = LW X10, 4
    // X10 = LW killed X10, 0
    if (FirstReg == BaseReg) {
      MIB2 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
                 .addReg(SecondReg,
                         RegState::Define | getDeadRegState(SecondOp.isDead()))
                 .addReg(BaseReg);
      MIB1 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
                 .addReg(FirstReg,
                         RegState::Define | getDeadRegState(FirstOp.isDead()))
                 .addReg(BaseReg, getKillRegState(BaseOp.isKill()));

    } else {
      MIB1 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
                 .addReg(FirstReg,
                         RegState::Define | getDeadRegState(FirstOp.isDead()))
                 .addReg(BaseReg);

      MIB2 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
                 .addReg(SecondReg,
                         RegState::Define | getDeadRegState(SecondOp.isDead()))
                 .addReg(BaseReg, getKillRegState(BaseOp.isKill()));
    }

    ++NumLD2LW;
    LLVM_DEBUG(dbgs() << "Split LD back to two LW instructions\n");
  } else {
    assert(
        FirstReg != SecondReg &&
        "First register and second register is impossible to be same register");
    MIB1 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
               .addReg(FirstReg, getKillRegState(FirstOp.isKill()))
               .addReg(BaseReg);

    MIB2 = BuildMI(MBB, MBBI, DL, TII->get(Opc))
               .addReg(SecondReg, getKillRegState(SecondOp.isKill()))
               .addReg(BaseReg, getKillRegState(BaseOp.isKill()));

    ++NumSD2SW;
    LLVM_DEBUG(dbgs() << "Split SD back to two SW instructions\n");
  }

  // Add offset operands - preserve symbolic references
  MIB1.add(OffsetOp);
  if (OffsetOp.isImm())
    MIB2.addImm(BaseOffset + 4);
  else if (OffsetOp.isGlobal())
    MIB2.addGlobalAddress(OffsetOp.getGlobal(), BaseOffset + 4,
                          OffsetOp.getTargetFlags());
  else if (OffsetOp.isCPI())
    MIB2.addConstantPoolIndex(OffsetOp.getIndex(), BaseOffset + 4,
                              OffsetOp.getTargetFlags());
  else if (OffsetOp.isBlockAddress())
    MIB2.addBlockAddress(OffsetOp.getBlockAddress(), BaseOffset + 4,
                         OffsetOp.getTargetFlags());

  // Copy memory operands if the original instruction had them
  // FIXME: This is overly conservative; the new instruction accesses 4 bytes,
  // not 8.
  MIB1.cloneMemRefs(*MI);
  MIB2.cloneMemRefs(*MI);

  // Remove the original paired instruction and update iterator
  MBBI = MBB.erase(MBBI);
}

bool RISCVLoadStoreOpt::fixInvalidRegPairOp(MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator &MBBI) {
  MachineInstr *MI = &*MBBI;
  unsigned Opcode = MI->getOpcode();

  // Check if this is a Zilsd pseudo that needs fixing
  if (Opcode != RISCV::PseudoLD_RV32_OPT && Opcode != RISCV::PseudoSD_RV32_OPT)
    return false;

  bool IsLoad = Opcode == RISCV::PseudoLD_RV32_OPT;

  const MachineOperand &FirstOp = MI->getOperand(0);
  const MachineOperand &SecondOp = MI->getOperand(1);
  Register FirstReg = FirstOp.getReg();
  Register SecondReg = SecondOp.getReg();

  if (!isValidZilsdRegPair(FirstReg, SecondReg)) {
    // Need to split back into two instructions
    splitLdSdIntoTwo(MBB, MBBI, IsLoad);
    return true;
  }

  // Registers are valid, convert to real LD/SD instruction
  const MachineOperand &BaseOp = MI->getOperand(2);
  Register BaseReg = BaseOp.getReg();
  DebugLoc DL = MI->getDebugLoc();
  // Handle both immediate and symbolic operands for offset
  const MachineOperand &OffsetOp = MI->getOperand(3);

  unsigned RealOpc = IsLoad ? RISCV::LD_RV32 : RISCV::SD_RV32;

  // Create register pair from the two individual registers
  unsigned RegPair = TRI->getMatchingSuperReg(FirstReg, RISCV::sub_gpr_even,
                                              &RISCV::GPRPairRegClass);
  // Create the real LD/SD instruction with register pair
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(RealOpc));

  if (IsLoad) {
    // For LD, the register pair is the destination
    MIB.addReg(RegPair, RegState::Define | getDeadRegState(FirstOp.isDead() &&
                                                           SecondOp.isDead()));
  } else {
    // For SD, the register pair is the source
    MIB.addReg(RegPair, getKillRegState(FirstOp.isKill() && SecondOp.isKill()));
  }

  MIB.addReg(BaseReg, getKillRegState(BaseOp.isKill()))
      .add(OffsetOp)
      .cloneMemRefs(*MI);

  LLVM_DEBUG(dbgs() << "Converted pseudo to real instruction: " << *MIB
                    << "\n");

  // Remove the pseudo instruction and update iterator
  MBBI = MBB.erase(MBBI);

  return true;
}

// Returns an instance of the Load / Store Optimization pass.
FunctionPass *llvm::createRISCVLoadStoreOptPass() {
  return new RISCVLoadStoreOpt();
}
