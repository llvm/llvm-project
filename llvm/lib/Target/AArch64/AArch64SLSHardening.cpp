//===- AArch64SLSHardening.cpp - Harden Straight Line Missspeculation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass to insert code to mitigate against side channel
// vulnerabilities that may happen under straight line miss-speculation.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/IndirectThunks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "aarch64-sls-hardening"

#define AARCH64_SLS_HARDENING_NAME "AArch64 sls hardening pass"

namespace {

class AArch64SLSHardening : public MachineFunctionPass {
public:
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const AArch64Subtarget *ST;

  static char ID;

  AArch64SLSHardening() : MachineFunctionPass(ID) {
    initializeAArch64SLSHardeningPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return AARCH64_SLS_HARDENING_NAME; }

private:
  bool hardenReturnsAndBRs(MachineBasicBlock &MBB) const;
  bool hardenBLRs(MachineBasicBlock &MBB) const;
  MachineBasicBlock &ConvertBLRToBL(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator) const;
};

} // end anonymous namespace

char AArch64SLSHardening::ID = 0;

INITIALIZE_PASS(AArch64SLSHardening, "aarch64-sls-hardening",
                AARCH64_SLS_HARDENING_NAME, false, false)

static void insertSpeculationBarrier(const AArch64Subtarget *ST,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     DebugLoc DL,
                                     bool AlwaysUseISBDSB = false) {
  assert(MBBI != MBB.begin() &&
         "Must not insert SpeculationBarrierEndBB as only instruction in MBB.");
  assert(std::prev(MBBI)->isBarrier() &&
         "SpeculationBarrierEndBB must only follow unconditional control flow "
         "instructions.");
  assert(std::prev(MBBI)->isTerminator() &&
         "SpeculationBarrierEndBB must only follow terminators.");
  const TargetInstrInfo *TII = ST->getInstrInfo();
  unsigned BarrierOpc = ST->hasSB() && !AlwaysUseISBDSB
                            ? AArch64::SpeculationBarrierSBEndBB
                            : AArch64::SpeculationBarrierISBDSBEndBB;
  if (MBBI == MBB.end() ||
      (MBBI->getOpcode() != AArch64::SpeculationBarrierSBEndBB &&
       MBBI->getOpcode() != AArch64::SpeculationBarrierISBDSBEndBB))
    BuildMI(MBB, MBBI, DL, TII->get(BarrierOpc));
}

bool AArch64SLSHardening::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<AArch64Subtarget>();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();

  bool Modified = false;
  for (auto &MBB : MF) {
    Modified |= hardenReturnsAndBRs(MBB);
    Modified |= hardenBLRs(MBB);
  }

  return Modified;
}

static bool isBLR(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AArch64::BLR:
  case AArch64::BLRNoIP:
    return true;
  case AArch64::BLRAA:
  case AArch64::BLRAB:
  case AArch64::BLRAAZ:
  case AArch64::BLRABZ:
    llvm_unreachable("Currently, LLVM's code generator does not support "
                     "producing BLRA* instructions. Therefore, there's no "
                     "support in this pass for those instructions.");
  }
  return false;
}

bool AArch64SLSHardening::hardenReturnsAndBRs(MachineBasicBlock &MBB) const {
  if (!ST->hardenSlsRetBr())
    return false;
  bool Modified = false;
  MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator(), E = MBB.end();
  MachineBasicBlock::iterator NextMBBI;
  for (; MBBI != E; MBBI = NextMBBI) {
    MachineInstr &MI = *MBBI;
    NextMBBI = std::next(MBBI);
    if (MI.isReturn() || isIndirectBranchOpcode(MI.getOpcode())) {
      assert(MI.isTerminator());
      insertSpeculationBarrier(ST, MBB, std::next(MBBI), MI.getDebugLoc());
      Modified = true;
    }
  }
  return Modified;
}

static const char SLSBLRNamePrefix[] = "__llvm_slsblr_thunk_";

static std::array<const char *, 29> SLSBLRThunkNames{{
    "__llvm_slsblr_thunk_x0",  "__llvm_slsblr_thunk_x1",
    "__llvm_slsblr_thunk_x2",  "__llvm_slsblr_thunk_x3",
    "__llvm_slsblr_thunk_x4",  "__llvm_slsblr_thunk_x5",
    "__llvm_slsblr_thunk_x6",  "__llvm_slsblr_thunk_x7",
    "__llvm_slsblr_thunk_x8",  "__llvm_slsblr_thunk_x9",
    "__llvm_slsblr_thunk_x10", "__llvm_slsblr_thunk_x11",
    "__llvm_slsblr_thunk_x12", "__llvm_slsblr_thunk_x13",
    "__llvm_slsblr_thunk_x14", "__llvm_slsblr_thunk_x15",
    // X16 and X17 are deliberately missing, as the mitigation requires those
    // register to not be used in BLR. See comment in ConvertBLRToBL for more
    // details.
    "__llvm_slsblr_thunk_x18", "__llvm_slsblr_thunk_x19",
    "__llvm_slsblr_thunk_x20", "__llvm_slsblr_thunk_x21",
    "__llvm_slsblr_thunk_x22", "__llvm_slsblr_thunk_x23",
    "__llvm_slsblr_thunk_x24", "__llvm_slsblr_thunk_x25",
    "__llvm_slsblr_thunk_x26", "__llvm_slsblr_thunk_x27",
    "__llvm_slsblr_thunk_x28", "__llvm_slsblr_thunk_x29",
    // X30 is deliberately missing, for similar reasons as X16 and X17 are
    // missing.
    "__llvm_slsblr_thunk_x31",
}};
static std::array<unsigned, 29> SLSBLRThunkRegs{{
    AArch64::X0,  AArch64::X1,  AArch64::X2,  AArch64::X3,  AArch64::X4,
    AArch64::X5,  AArch64::X6,  AArch64::X7,  AArch64::X8,  AArch64::X9,
    AArch64::X10, AArch64::X11, AArch64::X12, AArch64::X13, AArch64::X14,
    AArch64::X15, AArch64::X18, AArch64::X19, AArch64::X20, AArch64::X21,
    AArch64::X22, AArch64::X23, AArch64::X24, AArch64::X25, AArch64::X26,
    AArch64::X27, AArch64::X28, AArch64::FP,  AArch64::XZR}};

namespace {
struct SLSBLRThunkInserter : ThunkInserter<SLSBLRThunkInserter> {
  const char *getThunkPrefix() { return SLSBLRNamePrefix; }
  bool mayUseThunk(const MachineFunction &MF) {
    // FIXME: This could also check if there are any BLRs in the function
    // to more accurately reflect if a thunk will be needed.
    return MF.getSubtarget<AArch64Subtarget>().hardenSlsBlr();
  }
  void insertThunks(MachineModuleInfo &MMI);
  void populateThunk(MachineFunction &MF);
};
} // namespace

void SLSBLRThunkInserter::insertThunks(MachineModuleInfo &MMI) {
  // FIXME: It probably would be possible to filter which thunks to produce
  // based on which registers are actually used in BLR instructions in this
  // function. But would that be a worthwhile optimization?
  for (StringRef Name : SLSBLRThunkNames)
    createThunkFunction(MMI, Name);
}

void SLSBLRThunkInserter::populateThunk(MachineFunction &MF) {
  // FIXME: How to better communicate Register number, rather than through
  // name and lookup table?
  assert(MF.getName().startswith(getThunkPrefix()));
  int Index = -1;
  for (int i = 0; i < (int)SLSBLRThunkNames.size(); ++i)
    if (MF.getName() == SLSBLRThunkNames[i]) {
      Index = i;
      break;
    }
  assert(Index != -1);
  Register ThunkReg = SLSBLRThunkRegs[Index];

  const TargetInstrInfo *TII =
      MF.getSubtarget<AArch64Subtarget>().getInstrInfo();
  // Grab the entry MBB and erase any other blocks. O0 codegen appears to
  // generate two bbs for the entry block.
  MachineBasicBlock *Entry = &MF.front();
  Entry->clear();
  while (MF.size() > 1)
    MF.erase(std::next(MF.begin()));

  //  These thunks need to consist of the following instructions:
  //  __llvm_slsblr_thunk_xN:
  //      BR xN
  //      barrierInsts
  Entry->addLiveIn(ThunkReg);
  BuildMI(Entry, DebugLoc(), TII->get(AArch64::BR)).addReg(ThunkReg);
  // Make sure the thunks do not make use of the SB extension in case there is
  // a function somewhere that will call to it that for some reason disabled
  // the SB extension locally on that function, even though it's enabled for
  // the module otherwise. Therefore set AlwaysUseISBSDB to true.
  insertSpeculationBarrier(&MF.getSubtarget<AArch64Subtarget>(), *Entry,
                           Entry->end(), DebugLoc(), true /*AlwaysUseISBDSB*/);
}

MachineBasicBlock &
AArch64SLSHardening::ConvertBLRToBL(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI) const {
  // Transform a BLR to a BL as follows:
  // Before:
  //   |-----------------------------|
  //   |      ...                    |
  //   |  instI                      |
  //   |  BLR xN                     |
  //   |  instJ                      |
  //   |      ...                    |
  //   |-----------------------------|
  //
  // After:
  //   |-----------------------------|
  //   |      ...                    |
  //   |  instI                      |
  //   |  BL __llvm_slsblr_thunk_xN  |
  //   |  instJ                      |
  //   |      ...                    |
  //   |-----------------------------|
  //
  //   __llvm_slsblr_thunk_xN:
  //   |-----------------------------|
  //   |  BR xN                      |
  //   |  barrierInsts               |
  //   |-----------------------------|
  //
  // The __llvm_slsblr_thunk_xN thunks are created by the SLSBLRThunkInserter.
  // This function merely needs to transform BLR xN into BL
  // __llvm_slsblr_thunk_xN.
  //
  // Since linkers are allowed to clobber X16 and X17 on function calls, the
  // above mitigation only works if the original BLR instruction was not
  // BLR X16 nor BLR X17. Code generation before must make sure that no BLR
  // X16|X17 was produced if the mitigation is enabled.

  MachineInstr &BLR = *MBBI;
  assert(isBLR(BLR));
  unsigned BLOpcode;
  Register Reg;
  bool RegIsKilled;
  switch (BLR.getOpcode()) {
  case AArch64::BLR:
  case AArch64::BLRNoIP:
    BLOpcode = AArch64::BL;
    Reg = BLR.getOperand(0).getReg();
    assert(Reg != AArch64::X16 && Reg != AArch64::X17 && Reg != AArch64::LR);
    RegIsKilled = BLR.getOperand(0).isKill();
    break;
  case AArch64::BLRAA:
  case AArch64::BLRAB:
  case AArch64::BLRAAZ:
  case AArch64::BLRABZ:
    llvm_unreachable("BLRA instructions cannot yet be produced by LLVM, "
                     "therefore there is no need to support them for now.");
  default:
    llvm_unreachable("unhandled BLR");
  }
  DebugLoc DL = BLR.getDebugLoc();

  // If we'd like to support also BLRAA and BLRAB instructions, we'd need
  // a lot more different kind of thunks.
  // For example, a
  //
  // BLRAA xN, xM
  //
  // instruction probably would need to be transformed to something like:
  //
  // BL __llvm_slsblraa_thunk_x<N>_x<M>
  //
  // __llvm_slsblraa_thunk_x<N>_x<M>:
  //   BRAA x<N>, x<M>
  //   barrierInsts
  //
  // Given that about 30 different values of N are possible and about 30
  // different values of M are possible in the above, with the current way
  // of producing indirect thunks, we'd be producing about 30 times 30, i.e.
  // about 900 thunks (where most might not be actually called). This would
  // multiply further by two to support both BLRAA and BLRAB variants of those
  // instructions.
  // If we'd want to support this, we'd probably need to look into a different
  // way to produce thunk functions, based on which variants are actually
  // needed, rather than producing all possible variants.
  // So far, LLVM does never produce BLRA* instructions, so let's leave this
  // for the future when LLVM can start producing BLRA* instructions.
  MachineFunction &MF = *MBBI->getMF();
  MCContext &Context = MBB.getParent()->getContext();
  MCSymbol *Sym = Context.getOrCreateSymbol("__llvm_slsblr_thunk_x" +
                                            utostr(Reg - AArch64::X0));

  MachineInstr *BL = BuildMI(MBB, MBBI, DL, TII->get(BLOpcode)).addSym(Sym);

  // Now copy the implicit operands from BLR to BL and copy other necessary
  // info.
  // However, both BLR and BL instructions implictly use SP and implicitly
  // define LR. Blindly copying implicit operands would result in SP and LR
  // operands to be present multiple times. While this may not be too much of
  // an issue, let's avoid that for cleanliness, by removing those implicit
  // operands from the BL created above before we copy over all implicit
  // operands from the BLR.
  int ImpLROpIdx = -1;
  int ImpSPOpIdx = -1;
  for (unsigned OpIdx = BL->getNumExplicitOperands();
       OpIdx < BL->getNumOperands(); OpIdx++) {
    MachineOperand Op = BL->getOperand(OpIdx);
    if (!Op.isReg())
      continue;
    if (Op.getReg() == AArch64::LR && Op.isDef())
      ImpLROpIdx = OpIdx;
    if (Op.getReg() == AArch64::SP && !Op.isDef())
      ImpSPOpIdx = OpIdx;
  }
  assert(ImpLROpIdx != -1);
  assert(ImpSPOpIdx != -1);
  int FirstOpIdxToRemove = std::max(ImpLROpIdx, ImpSPOpIdx);
  int SecondOpIdxToRemove = std::min(ImpLROpIdx, ImpSPOpIdx);
  BL->RemoveOperand(FirstOpIdxToRemove);
  BL->RemoveOperand(SecondOpIdxToRemove);
  // Now copy over the implicit operands from the original BLR
  BL->copyImplicitOps(MF, BLR);
  MF.moveCallSiteInfo(&BLR, BL);
  // Also add the register called in the BLR as being used in the called thunk.
  BL->addOperand(MachineOperand::CreateReg(Reg, false /*isDef*/, true /*isImp*/,
                                           RegIsKilled /*isKill*/));
  // Remove BLR instruction
  MBB.erase(MBBI);

  return MBB;
}

bool AArch64SLSHardening::hardenBLRs(MachineBasicBlock &MBB) const {
  if (!ST->hardenSlsBlr())
    return false;
  bool Modified = false;
  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  MachineBasicBlock::iterator NextMBBI;
  for (; MBBI != E; MBBI = NextMBBI) {
    MachineInstr &MI = *MBBI;
    NextMBBI = std::next(MBBI);
    if (isBLR(MI)) {
      ConvertBLRToBL(MBB, MBBI);
      Modified = true;
    }
  }
  return Modified;
}

FunctionPass *llvm::createAArch64SLSHardeningPass() {
  return new AArch64SLSHardening();
}

namespace {
class AArch64IndirectThunks : public MachineFunctionPass {
public:
  static char ID;

  AArch64IndirectThunks() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "AArch64 Indirect Thunks"; }

  bool doInitialization(Module &M) override;
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.addPreserved<MachineModuleInfoWrapperPass>();
  }

private:
  std::tuple<SLSBLRThunkInserter> TIs;

  // FIXME: When LLVM moves to C++17, these can become folds
  template <typename... ThunkInserterT>
  static void initTIs(Module &M,
                      std::tuple<ThunkInserterT...> &ThunkInserters) {
    (void)std::initializer_list<int>{
        (std::get<ThunkInserterT>(ThunkInserters).init(M), 0)...};
  }
  template <typename... ThunkInserterT>
  static bool runTIs(MachineModuleInfo &MMI, MachineFunction &MF,
                     std::tuple<ThunkInserterT...> &ThunkInserters) {
    bool Modified = false;
    (void)std::initializer_list<int>{
        Modified |= std::get<ThunkInserterT>(ThunkInserters).run(MMI, MF)...};
    return Modified;
  }
};

} // end anonymous namespace

char AArch64IndirectThunks::ID = 0;

FunctionPass *llvm::createAArch64IndirectThunks() {
  return new AArch64IndirectThunks();
}

bool AArch64IndirectThunks::doInitialization(Module &M) {
  initTIs(M, TIs);
  return false;
}

bool AArch64IndirectThunks::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << getPassName() << '\n');
  auto &MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  return runTIs(MMI, MF, TIs);
}
