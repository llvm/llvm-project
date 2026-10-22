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
//   call _Unwind_RaiseException   r0 = 0; unwind_exit
//   invoke _Unwind_RaiseException goto pad
// unwind_goto is taken when the callee left with unwind_exit. No register says
// so: how one insn tells the other is up to the JIT. Otherwise unwind_exit is
// exit, hence r0, and r2 as well in a function that returns more than 8 bytes.
// A pad expects the exception object in r0. There is none, so a pad that looks
// at it starts with r0 = 0.
//
// _Unwind_RaiseException() is how a language runtime starts an unwind with the
// Itanium ABI, e.g. at the end of a panic handler. To start one and to carry
// one on is the same thing here: leave the frame with unwind_exit, after its
// own pad if it has one. It cannot fail, so it does not return.
//
// There are no tables and nothing for a loader to do.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFInstrInfo.h"
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
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

/// On other targets it is a function of the unwinder library, which is why
/// nothing else in LLVM knows the name.
static constexpr StringLiteral UnwindRaise = "_Unwind_RaiseException";

/// Not being nounwind is the default in IR, so that alone is not enough:
/// uwtable is how a frontend asks for unwinding through a function to work.
static bool canUnwind(const Function &F) {
  return F.hasUWTable() && !F.doesNotThrow();
}

namespace {

/// Make every call that may unwind an invoke: it ends its block and has a pad.
struct BPFUnwindPrepare : public ModulePass {
  static char ID;
  BPFUnwindPrepare() : ModulePass(ID) {}
  StringRef getPassName() const override { return "BPF Unwind Prepare"; }

  static bool isRaise(const CallBase &CB) {
    const auto *F = dyn_cast<Function>(CB.getCalledOperand());
    return F && F->isDeclaration() && F->getName() == UnwindRaise;
  }

  static bool mayUnwind(const Instruction &I) {
    const auto *CB = dyn_cast<CallBase>(&I);
    return CB && !CB->doesNotThrow() && !CB->isInlineAsm() &&
           !isa<IntrinsicInst>(CB);
  }

  bool runOnModule(Module &M) override {
    bool Changed = false;

    for (Function &F : M) {
      if (F.isDeclaration() || !canUnwind(F))
        continue;

      LLVMContext &C = M.getContext();
      SmallVector<CallInst *, 8> Calls;
      for (Instruction &I : instructions(F)) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (CB && isRaise(*CB)) {
          // Nobody looks at the exception object.
          for (Use &Arg : CB->args())
            Arg.set(PoisonValue::get(Arg->getType()));
          CB->setDoesNotReturn();
          // Without a pad it leaves the frame.
          if (isa<CallInst>(CB))
            CB->setDoesNotThrow();
          Changed = true;
        } else if (isa<CallInst>(I) && mayUnwind(I)) {
          Calls.push_back(cast<CallInst>(&I));
        } else if (auto *RI = dyn_cast<ResumeInst>(&I)) {
          // DwarfEHPrepare passes it to _Unwind_Resume(). Don't keep it alive
          // over the pad for that.
          RI->setOperand(0, PoisonValue::get(RI->getValue()->getType()));
          Changed = true;
        }
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
      B.CreateResume(PoisonValue::get(LP->getType()));
      for (CallInst *CI : Calls)
        changeToInvokeAndSplitBasicBlock(CI, Pad);
      Changed = true;
    }
    return Changed;
  }
};

/// After register allocation, which took call -> pad for an EH edge and so
/// keeps only r6-r9 and the stack alive over it, turn the edge into a branch.
struct BPFUnwindLower : public MachineFunctionPass {
  static char ID;
  BPFUnwindLower() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "BPF Unwind Lower"; }

  /// What DwarfEHPrepare made of 'resume'.
  StringRef UnwindResume;

  bool leavesFrame(const MachineInstr &MI) const {
    const MachineOperand &MO = MI.getOperand(0);
    // A program that defines them gets what it defines.
    if (!MO.isGlobal() || !MO.getGlobal()->isDeclaration())
      return false;
    StringRef Name = MO.getGlobal()->getName();
    return Name == UnwindRaise || Name == UnwindResume;
  }

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
    UnwindResume = MF.getSubtarget().getTargetLowering()->getLibcallName(
        RTLIB::UNWIND_RESUME);
    DenseMap<MCSymbol *, MachineBasicBlock *> Begin;
    SmallVector<std::pair<MachineInstr *, MachineBasicBlock *>, 8> Sites;
    SmallVector<MachineInstr *, 8> Labels, Resumes;
    SmallSetVector<MachineBasicBlock *, 8> Pads;

    if (const char *Msg = unsupported(MF)) {
      const Function &F = MF.getFunction();
      F.getContext().diagnose(DiagnosticInfoUnsupported(F, Msg));
      return false;
    }

    for (const LandingPadInfo &LP : MF.getLandingPads())
      for (MCSymbol *Sym : LP.BeginLabels)
        Begin[Sym] = LP.LandingPadBlock;

    for (MachineBasicBlock &MBB : MF) {
      MachineBasicBlock *Pad = nullptr;
      for (MachineInstr &MI : MBB) {
        if (MI.isEHLabel()) {
          Pad = Begin.lookup(MI.getOperand(0).getMCSymbol());
          Labels.push_back(&MI);
        } else if (MI.isCall() && Pad) {
          Sites.push_back({&MI, Pad});
        } else if (MI.isCall() && leavesFrame(MI)) {
          Resumes.push_back(&MI);
        }
      }
    }
    if (Labels.empty() && Resumes.empty())
      return false;
    // There is no unwind_goto after the calls of it.
    if (!Resumes.empty() && !canUnwind(MF.getFunction())) {
      const Function &F = MF.getFunction();
      F.getContext().diagnose(DiagnosticInfoUnsupported(
          F, "BPF cannot unwind out of a function that is nounwind or has no "
             "uwtable"));
      return false;
    }
    for (MachineInstr *MI : Labels)
      MI->eraseFromParent();

    for (auto [Call, Pad] : Sites) {
      MachineBasicBlock *MBB = Call->getParent();
      DebugLoc DL = Call->getDebugLoc();
      const auto *II =
          dyn_cast_or_null<InvokeInst>(MBB->getBasicBlock()->getTerminator());

      Pads.insert(Pad);
      if (leavesFrame(*Call)) {
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

ModulePass *llvm::createBPFUnwindPreparePass() {
  return new BPFUnwindPrepare();
}
FunctionPass *llvm::createBPFUnwindLowerPass() { return new BPFUnwindLower(); }
