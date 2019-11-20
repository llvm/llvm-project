//===-- AArch64StackTaggingPreRA.cpp --- Stack Tagging for AArch64 -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "AArch64.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64InstrInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineTraceMetrics.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-stack-tagging-pre-ra"

enum UncheckedLdStMode { UncheckedNever, UncheckedSafe, UncheckedAlways };

cl::opt<UncheckedLdStMode> ClUncheckedLdSt(
    "stack-tagging-unchecked-ld-st", cl::Hidden,
    cl::init(UncheckedSafe),
    cl::desc(
        "Unconditionally apply unchecked-ld-st optimization (even for large "
        "stack frames, or in the presence of variable sized allocas)."),
    cl::values(
        clEnumValN(UncheckedNever, "never", "never apply unchecked-ld-st"),
        clEnumValN(
            UncheckedSafe, "safe",
            "apply unchecked-ld-st when the target is definitely within range"),
        clEnumValN(UncheckedAlways, "always", "always apply unchecked-ld-st")));

namespace {

class AArch64StackTaggingPreRA : public MachineFunctionPass {
  MachineFunction *MF;
  AArch64FunctionInfo *AFI;
  MachineFrameInfo *MFI;
  MachineRegisterInfo *MRI;
  const AArch64RegisterInfo *TRI;
  const AArch64InstrInfo *TII;

  SmallVector<MachineInstr*, 16> ReTags;

public:
  static char ID;
  AArch64StackTaggingPreRA() : MachineFunctionPass(ID) {
    initializeAArch64StackTaggingPreRAPass(*PassRegistry::getPassRegistry());
  }

  bool mayUseUncheckedLoadStore();
  void uncheckUsesOf(unsigned TaggedReg, int FI);
  void uncheckLoadsAndStores();

  bool runOnMachineFunction(MachineFunction &Func) override;
  StringRef getPassName() const override {
    return "AArch64 Stack Tagging PreRA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char AArch64StackTaggingPreRA::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64StackTaggingPreRA, "aarch64-stack-tagging-pre-ra",
                      "AArch64 Stack Tagging PreRA Pass", false, false)
INITIALIZE_PASS_END(AArch64StackTaggingPreRA, "aarch64-stack-tagging-pre-ra",
                    "AArch64 Stack Tagging PreRA Pass", false, false)

FunctionPass *llvm::createAArch64StackTaggingPreRAPass() {
  return new AArch64StackTaggingPreRA();
}

static bool isUncheckedLoadOrStoreOpcode(unsigned Opcode) {
  switch (Opcode) {
  case AArch64::LDRBBui:
  case AArch64::LDRHHui:
  case AArch64::LDRWui:
  case AArch64::LDRXui:

  case AArch64::LDRBui:
  case AArch64::LDRHui:
  case AArch64::LDRSui:
  case AArch64::LDRDui:
  case AArch64::LDRQui:

  case AArch64::LDRSHWui:
  case AArch64::LDRSHXui:

  case AArch64::LDRSBWui:
  case AArch64::LDRSBXui:

  case AArch64::LDRSWui:

  case AArch64::STRBBui:
  case AArch64::STRHHui:
  case AArch64::STRWui:
  case AArch64::STRXui:

  case AArch64::STRBui:
  case AArch64::STRHui:
  case AArch64::STRSui:
  case AArch64::STRDui:
  case AArch64::STRQui:

  case AArch64::LDPWi:
  case AArch64::LDPXi:
  case AArch64::LDPSi:
  case AArch64::LDPDi:
  case AArch64::LDPQi:

  case AArch64::LDPSWi:

  case AArch64::STPWi:
  case AArch64::STPXi:
  case AArch64::STPSi:
  case AArch64::STPDi:
  case AArch64::STPQi:
    return true;
  default:
    return false;
  }
}

bool AArch64StackTaggingPreRA::mayUseUncheckedLoadStore() {
  if (ClUncheckedLdSt == UncheckedNever)
    return false;
  else if (ClUncheckedLdSt == UncheckedAlways)
    return true;

  // This estimate can be improved if we had harder guarantees about stack frame
  // layout. With LocalStackAllocation we can estimate SP offset to any
  // preallocated slot. AArch64FrameLowering::orderFrameObjects could put tagged
  // objects ahead of non-tagged ones, but that's not always desirable.
  //
  // Underestimating SP offset here may require the use of LDG to materialize
  // the tagged address of the stack slot, along with a scratch register
  // allocation (post-regalloc!).
  //
  // For now we do the safe thing here and require that the entire stack frame
  // is within range of the shortest of the unchecked instructions.
  unsigned FrameSize = 0;
  for (unsigned i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i)
    FrameSize += MFI->getObjectSize(i);
  bool EntireFrameReachableFromSP = FrameSize < 0xf00;
  return !MFI->hasVarSizedObjects() && EntireFrameReachableFromSP;
}

void AArch64StackTaggingPreRA::uncheckUsesOf(unsigned TaggedReg, int FI) {
  for (auto UI = MRI->use_instr_begin(TaggedReg), E = MRI->use_instr_end();
       UI != E;) {
    MachineInstr *UseI = &*(UI++);
    if (isUncheckedLoadOrStoreOpcode(UseI->getOpcode())) {
      // FI operand is always the one before the immediate offset.
      unsigned OpIdx = TII->getLoadStoreImmIdx(UseI->getOpcode()) - 1;
      if (UseI->getOperand(OpIdx).isReg() &&
          UseI->getOperand(OpIdx).getReg() == TaggedReg) {
        UseI->getOperand(OpIdx).ChangeToFrameIndex(FI);
        UseI->getOperand(OpIdx).setTargetFlags(AArch64II::MO_TAGGED);
      }
    } else if (UseI->isCopy() &&
               Register::isVirtualRegister(UseI->getOperand(0).getReg())) {
      uncheckUsesOf(UseI->getOperand(0).getReg(), FI);
    }
  }
}

void AArch64StackTaggingPreRA::uncheckLoadsAndStores() {
  for (auto *I : ReTags) {
    unsigned TaggedReg = I->getOperand(0).getReg();
    int FI = I->getOperand(1).getIndex();
    uncheckUsesOf(TaggedReg, FI);
  }
}

bool AArch64StackTaggingPreRA::runOnMachineFunction(MachineFunction &Func) {
  MF = &Func;
  MRI = &MF->getRegInfo();
  AFI = MF->getInfo<AArch64FunctionInfo>();
  TII = static_cast<const AArch64InstrInfo *>(MF->getSubtarget().getInstrInfo());
  TRI = static_cast<const AArch64RegisterInfo *>(
      MF->getSubtarget().getRegisterInfo());
  MFI = &MF->getFrameInfo();
  ReTags.clear();

  assert(MRI->isSSA());

  LLVM_DEBUG(dbgs() << "********** AArch64 Stack Tagging PreRA **********\n"
                    << "********** Function: " << MF->getName() << '\n');

  SmallSetVector<int, 8> TaggedSlots;
  for (auto &BB : *MF) {
    for (auto &I : BB) {
      if (I.getOpcode() == AArch64::TAGPstack) {
        ReTags.push_back(&I);
        int FI = I.getOperand(1).getIndex();
        TaggedSlots.insert(FI);
        // There should be no offsets in TAGP yet.
        assert(I.getOperand(2).getImm() == 0);
      }
    }
  }

  if (ReTags.empty())
    return false;

  if (mayUseUncheckedLoadStore())
    uncheckLoadsAndStores();

  return true;
}
