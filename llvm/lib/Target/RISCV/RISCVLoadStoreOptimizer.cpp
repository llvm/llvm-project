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

  // Scan the instructions looking for a load/store that can be combined
  // with the current instruction into a load/store pair.
  // Return the matching instruction if one is found, else MBB->end().
  MachineBasicBlock::iterator findMatchingInsn(MachineBasicBlock::iterator I,
                                               bool &MergeForward);

  MachineBasicBlock::iterator
  mergePairedInsns(MachineBasicBlock::iterator I,
                   MachineBasicBlock::iterator Paired, bool MergeForward);

private:
  AliasAnalysis *AA;
  MachineRegisterInfo *MRI;
  const RISCVInstrInfo *TII;
  const RISCVRegisterInfo *TRI;
  LiveRegUnits ModifiedRegUnits, UsedRegUnits;
};
} // end anonymous namespace

char RISCVLoadStoreOpt::ID = 0;
INITIALIZE_PASS(RISCVLoadStoreOpt, DEBUG_TYPE, RISCV_LOAD_STORE_OPT_NAME, false,
                false)

bool RISCVLoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  if (skipFunction(Fn.getFunction()))
    return false;
  const RISCVSubtarget &Subtarget = Fn.getSubtarget<RISCVSubtarget>();
  if (!Subtarget.useMIPSLoadStorePairs())
    return false;

  bool MadeChange = false;
  TII = Subtarget.getInstrInfo();
  TRI = Subtarget.getRegisterInfo();
  MRI = &Fn.getRegInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  ModifiedRegUnits.init(*TRI);
  UsedRegUnits.init(*TRI);

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

// Merge two adjacent load/store instructions into a paired instruction
// (LDP/SDP/SWP/LWP) if the effective address is 8-byte aligned in case of
// SWP/LWP 16-byte aligned in case of LDP/SDP. This function selects the
// appropriate paired opcode, verifies that the memory operand is properly
// aligned, and checks that the offset is valid. If all conditions are met, it
// builds and inserts the paired instruction.
bool RISCVLoadStoreOpt::tryConvertToLdStPair(
    MachineBasicBlock::iterator First, MachineBasicBlock::iterator Second) {
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

  MachineFunction *MF = First->getMF();
  const MachineMemOperand *MMO = *First->memoperands_begin();
  Align MMOAlign = MMO->getAlign();

  if (MMOAlign < RequiredAlignment)
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
  // If NextI is the second of the two instructions to be merged, we need
  // to skip one further. Either way we merge will invalidate the iterator,
  // and we don't need to scan the new instruction, as it's a pairwise
  // instruction, which we're not considering for further action anyway.
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
  }

  return NextI;
}

// Returns an instance of the Load / Store Optimization pass.
FunctionPass *llvm::createRISCVLoadStoreOptPass() {
  return new RISCVLoadStoreOpt();
}
