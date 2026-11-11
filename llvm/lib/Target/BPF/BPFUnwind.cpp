//===-- BPFUnwind.cpp - Lower unwinding to unwind_goto and unwind_exit ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// There are two ways to leave a frame, and a call that may unwind is followed
// by an insn that tells them apart:
//   ret                           exit
//   invoke f() unwind pad         call f; unwind_goto pad
//   resume                        r0 = 0; unwind_exit
//   call llvm.bpf.unwind.exit     r0 = 0; unwind_exit
//   invoke llvm.bpf.unwind.exit   goto pad
// unwind_goto is taken when the callee left with unwind_exit. No register says
// so: how one insn tells the other is up to the JIT. Otherwise unwind_exit is
// exit, hence r0, and r2 as well in a function that returns more than 8 bytes.
// A pad expects the exception object in r0. There is none, so a pad that looks
// at it starts with r0 = 0.
//
// There is no unwinder library. A runtime starts an unwind by calling
// llvm.bpf.unwind.exit, which leaves the frame with unwind_exit. resume, which
// continues an unwind into the caller, does the same, so BPFUnwindPrepare
// replaces it with the intrinsic.
//
// There are no tables and nothing for a loader to do.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFSubtarget.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsBPF.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

/// Not being nounwind is the default in IR, so that alone is not enough:
/// uwtable is how a frontend asks for unwinding through a function to work.
static bool canUnwind(const Function &F) {
  return F.hasUWTable() && !F.doesNotThrow();
}

namespace {

/// Turn resume into llvm.bpf.unwind.exit, and calls that may unwind into
/// invokes.
struct BPFUnwindPrepare : public ModulePass {
  static char ID;
  BPFUnwindPrepare() : ModulePass(ID) {}
  StringRef getPassName() const override { return "BPF Unwind Prepare"; }

  Function *UnwindExit = nullptr;

  Function *getUnwindExit(Module &M) {
    if (!UnwindExit)
      UnwindExit =
          Intrinsic::getOrInsertDeclaration(&M, Intrinsic::bpf_unwind_exit);
    return UnwindExit;
  }

  static bool mayUnwind(const Instruction &I) {
    const auto *CB = dyn_cast<CallBase>(&I);
    return CB && !CB->doesNotThrow() && !CB->isInlineAsm() &&
           !isa<IntrinsicInst>(CB);
  }

  bool runOnModule(Module &M) override {
    bool Changed = false;
    LLVMContext &C = M.getContext();
    UnwindExit = nullptr;

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      // Replace resume even if F cannot unwind, so BPFUnwindLower reports it.
      for (BasicBlock &BB : F) {
        if (auto *RI = dyn_cast<ResumeInst>(BB.getTerminator())) {
          CallInst::Create(getUnwindExit(M), "", RI->getIterator())
              ->setDebugLoc(RI->getDebugLoc());
          changeToUnreachable(RI);
          Changed = true;
        }
      }

      if (!canUnwind(F))
        continue;
      SmallVector<CallInst *, 8> Calls;
      for (Instruction &I : instructions(F)) {
        if (isa<CallInst>(I) && mayUnwind(I))
          Calls.push_back(cast<CallInst>(&I));
      }

      if (Calls.empty())
        continue;

      if (!F.hasPersonalityFn())
        F.setPersonalityFn(cast<Constant>(
            M.getOrInsertFunction("__gcc_personality_v0",
                                  FunctionType::get(Type::getInt32Ty(C), true))
                .getCallee()));
      BasicBlock *Pad = BasicBlock::Create(C, "unwind.resume", &F);
      IRBuilder<> B(Pad);
      LandingPadInst *LP = B.CreateLandingPad(
          StructType::get(PointerType::getUnqual(C), Type::getInt32Ty(C)), 0);
      LP->setCleanup(true);
      B.CreateCall(getUnwindExit(M));
      B.CreateUnreachable();
      for (CallInst *CI : Calls)
        changeToInvokeAndSplitBasicBlock(CI, Pad);
      Changed = true;
    }
    return Changed;
  }
};

/// After register allocation, turn edges to landing pads into branches.
struct BPFUnwindLower : public MachineFunctionPass {
  static char ID;
  BPFUnwindLower() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "BPF Unwind Lower"; }

  /// Nothing after \p MI in its block runs: \p Succ, if any, is what follows.
  static void endBlockAfter(MachineInstr &MI, MachineBasicBlock *Succ) {
    MachineBasicBlock &MBB = *MI.getParent();
    MBB.erase(std::next(MI.getIterator()), MBB.end());
    while (!MBB.succ_empty())
      MBB.removeSuccessor(MBB.succ_begin());
    if (Succ)
      MBB.addSuccessor(Succ);
  }

  static const char *unsupported(const MachineFunction &MF) {
    if (MF.getLandingPads().empty())
      return nullptr;
    // Not canUnwind(): a function that catches everything is nounwind.
    if (!MF.getFunction().hasUWTable())
      return "BPF needs uwtable on a function with a landing pad";
    // Every pad is a catch-all: the branch is taken whatever was thrown.
    for (const LandingPadInfo &LP : MF.getLandingPads()) {
      for (int TId : LP.TypeIds) {
        if (TId > 0 && MF.getTypeInfos()[TId - 1])
          return "BPF does not support type-specific exception catches yet";
        // An empty filter lets nothing through, so it is a catch-all too. rustc
        // guards every function that must not unwind with one.
        if (TId < 0 && MF.getFilterIds()[-1 - TId])
          return "BPF does not support exception filters yet";
      }
    }
    return nullptr;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
    const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
    DenseMap<MCSymbol *, MachineBasicBlock *> Begin;
    SmallVector<std::pair<MachineInstr *, MachineBasicBlock *>, 8> Sites;
    SmallVector<MachineInstr *, 8> Labels, Resumes;
    SmallSetVector<MachineBasicBlock *, 8> Pads;

    const char *Msg = unsupported(MF);
    if (Msg) {
      const Function &F = MF.getFunction();
      F.getContext().diagnose(DiagnosticInfoUnsupported(F, Msg));
    }

    for (const LandingPadInfo &LP : MF.getLandingPads())
      for (MCSymbol *Sym : LP.BeginLabels)
        Begin[Sym] = LP.LandingPadBlock;

    // The call is not always in the block of the label: an argument that is
    // a select is computed after the label and takes blocks of its own.
    SmallPtrSet<MachineInstr *, 8> Invokes;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &Label : MBB) {
        if (!Label.isEHLabel())
          continue;
        Labels.push_back(&Label);
        MachineBasicBlock *Pad =
            Begin.lookup(Label.getOperand(0).getMCSymbol());
        if (!Pad)
          continue;
        SmallVector<std::pair<MachineBasicBlock *, MachineBasicBlock::iterator>,
                    4>
            Work{{&MBB, std::next(Label.getIterator())}};
        SmallPtrSet<MachineBasicBlock *, 4> Seen;
        while (!Work.empty()) {
          auto [Cur, I] = Work.pop_back_val();
          // The next label ends the range.
          for (; I != Cur->end() && !I->isEHLabel(); ++I)
            if (I->isCall() && Invokes.insert(&*I).second)
              Sites.push_back({&*I, Pad});
          if (I != Cur->end())
            continue;
          for (MachineBasicBlock *Succ : Cur->successors())
            if (!Succ->isEHPad() && Seen.insert(Succ).second)
              Work.push_back({Succ, Succ->begin()});
        }
      }
    }
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB)
        if (MI.getOpcode() == BPF::UNWIND_EXIT_PSEUDO && !Invokes.count(&MI))
          Resumes.push_back(&MI);
    if (Labels.empty() && Resumes.empty())
      return false;
    // There is no unwind_goto after the calls of it.
    if (!Msg && !Resumes.empty() && !canUnwind(MF.getFunction())) {
      const Function &F = MF.getFunction();
      F.getContext().diagnose(DiagnosticInfoUnsupported(
          F, "BPF cannot unwind out of a function that is nounwind or has no "
             "uwtable"));
    }
    for (MachineInstr *MI : Labels)
      MI->eraseFromParent();

    for (auto [Call, Pad] : Sites) {
      MachineBasicBlock *MBB = Call->getParent();
      DebugLoc DL = Call->getDebugLoc();
      const auto *II =
          dyn_cast_or_null<InvokeInst>(MBB->getBasicBlock()->getTerminator());

      Pads.insert(Pad);
      if (Call->getOpcode() == BPF::UNWIND_EXIT_PSEUDO) {
        endBlockAfter(*Call, Pad);
        BuildMI(MBB, DL, TII.get(BPF::JMP)).addMBB(Pad);
        Call->eraseFromParent();
      } else if (II && II->doesNotReturn()) {
        // Nothing sensible follows the call, but something has to.
        endBlockAfter(*Call, Pad);
        BuildMI(MBB, DL, TII.get(BPF::UNWIND_GOTO)).addMBB(Pad);
        BuildMI(MBB, DL, TII.get(BPF::JMP)).addMBB(Pad);
      } else {
        MachineBasicBlock *Rest = MBB->splitAt(*Call);
        if (Rest != MBB) {
          Rest->removeSuccessor(Pad);
          MBB->addSuccessor(Pad);
        }
        BuildMI(MBB, DL, TII.get(BPF::UNWIND_GOTO)).addMBB(Pad);
      }
    }

    for (MachineBasicBlock *Pad : Pads) {
      LivePhysRegs Live(TRI);
      Live.addLiveOuts(*Pad);
      for (MachineInstr &MI : reverse(*Pad))
        Live.stepBackward(MI);
      if (!Live.available(MF.getRegInfo(), BPF::R0))
        BuildMI(*Pad, Pad->begin(), DebugLoc(), TII.get(BPF::MOV_ri), BPF::R0)
            .addImm(0);
      Pad->removeLiveIn(BPF::R0);
    }

    // unwind_exit returns what exit returns: a value of more than 8 bytes is in
    // r0 and r2.
    SmallVector<ISD::OutputArg, 2> Outs;
    const Function &F = MF.getFunction();
    GetReturnInfo(F.getCallingConv(), F.getReturnType(), F.getAttributes(),
                  Outs, *MF.getSubtarget().getTargetLowering(),
                  MF.getDataLayout());

    for (MachineInstr *Call : Resumes) {
      MachineBasicBlock *MBB = Call->getParent();
      DebugLoc DL = Call->getDebugLoc();

      BuildMI(*MBB, Call, DL, TII.get(BPF::MOV_ri), BPF::R0).addImm(0);
      MachineInstrBuilder Exit =
          BuildMI(*MBB, Call, DL, TII.get(BPF::UNWIND_EXIT))
              .addReg(BPF::R0, RegState::Implicit);
      if (Outs.size() > 1) {
        BuildMI(*MBB, *Exit, DL, TII.get(BPF::MOV_ri), BPF::R2).addImm(0);
        Exit.addReg(BPF::R2, RegState::Implicit);
      }
      endBlockAfter(*Exit, nullptr);
    }

    // The pads are plain blocks now and there is no table to emit.
    for (MachineBasicBlock &MBB : MF)
      MBB.setIsEHPad(false);
    MF.clearLandingPads();

    // What was behind a call that does not return. It can be a loop.
    df_iterator_default_set<MachineBasicBlock *> Reachable;
    for (MachineBasicBlock *MBB : depth_first_ext(&MF, Reachable))
      (void)MBB;
    for (MachineBasicBlock &MBB : make_early_inc_range(MF)) {
      if (Reachable.count(&MBB) || MBB.hasAddressTaken())
        continue;
      while (!MBB.succ_empty())
        MBB.removeSuccessor(MBB.succ_begin());
      MBB.eraseFromParent();
    }
    for (MachineBasicBlock &MBB : MF)
      if (!MBB.empty() && MBB.back().getOpcode() == BPF::JMP &&
          MBB.isLayoutSuccessor(MBB.back().getOperand(0).getMBB()))
        MBB.back().eraseFromParent();
    return true;
  }
};

} // namespace

char BPFUnwindPrepare::ID = 0;
char BPFUnwindLower::ID = 0;

INITIALIZE_PASS(BPFUnwindPrepare, "bpf-unwind-prepare", "BPF Unwind Prepare",
                false, false)
INITIALIZE_PASS(BPFUnwindLower, "bpf-unwind-lower", "BPF Unwind Lower", false,
                false)

ModulePass *llvm::createBPFUnwindPreparePass() {
  return new BPFUnwindPrepare();
}
FunctionPass *llvm::createBPFUnwindLowerPass() { return new BPFUnwindLower(); }
