//===-- BPFUnwind.cpp - Lower unwinding to a flag in r3 and a branch ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A function that can unwind says in r3 how it returns:
//   ret                           r3 = 0; exit
//   invoke f() unwind pad         call f; if r3 != 0 goto pad
//   resume                        r0 = 0; r3 = 1; exit
//   call _Unwind_RaiseException   r0 = 0; r3 = 1; exit
//   invoke _Unwind_RaiseException r0 = 0; goto pad
// 'ret' is done by BPFTargetLowering::LowerReturn(), the rest is here.
//
// _Unwind_RaiseException() is how a language runtime starts an unwind with the
// Itanium ABI, e.g. at the end of a panic handler. To start one and to carry
// one on is the same thing here: leave the frame with r3 = 1, after its own pad
// if it has one. It cannot fail, so it does not return.
// There are no tables and nothing for a loader or the kernel to dispatch.
// BTFDebug tags such functions with decl_tag "unwind".
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFInstrInfo.h"
#include "BPFSubtarget.h"
#include "llvm/ADT/DepthFirstIterator.h"
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
    bool Changed = false, IndirectUnwind = false;

    for (Function &F : M)
      if (bpfCanUnwind(F))
        for (Instruction &I : instructions(F))
          IndirectUnwind |= mayUnwind(I) && cast<CallBase>(I).isIndirectCall();

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      // The caller of a function pointer looks at r3 whatever the callee is.
      if (IndirectUnwind && F.hasUWTable() && F.doesNotThrow() &&
          F.hasAddressTaken()) {
        F.removeFnAttr(Attribute::NoUnwind);
        Changed = true;
      }
      if (!bpfCanUnwind(F))
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
    // Not bpfCanUnwind(): a function that catches everything is nounwind.
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
    // Its caller does not look at r3: the unwind would be lost silently.
    if (!Resumes.empty() && !bpfCanUnwind(MF.getFunction())) {
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

      // The pad reads r0, which the callee zeroes when it unwinds.
      Call->addRegisterDefined(BPF::R0);
      Call->clearRegisterDeads(BPF::R0);

      if (leavesFrame(*Call) || (II && II->doesNotReturn())) {
        endBlockAfter(*Call, Pad);
        BuildMI(MBB, DL, TII.get(BPF::JMP)).addMBB(Pad);
        if (leavesFrame(*Call)) {
          if (Pad->computeRegisterLiveness(&TRI, BPF::R0, Pad->begin()) !=
              MachineBasicBlock::LQR_Dead)
            BuildMI(*MBB, Call, DL, TII.get(BPF::MOV_ri), BPF::R0).addImm(0);
          Call->eraseFromParent();
        }
      } else {
        MachineBasicBlock *Rest = MBB->splitAt(*Call);
        if (Rest != MBB) {
          Rest->removeSuccessor(Pad);
          MBB->addSuccessor(Pad);
        }
        Call->addRegisterDefined(BPF::R3);
        Call->clearRegisterDeads(BPF::R3);
        BuildMI(MBB, DL, TII.get(BPF::JNE_ri))
            .addReg(BPF::R3)
            .addImm(0)
            .addMBB(Pad);
      }
    }

    for (MachineInstr *Call : Resumes) {
      MachineBasicBlock *MBB = Call->getParent();
      DebugLoc DL = Call->getDebugLoc();

      BuildMI(*MBB, Call, DL, TII.get(BPF::MOV_ri), BPF::R0).addImm(0);
      BuildMI(*MBB, Call, DL, TII.get(BPF::MOV_ri), BPF::R3).addImm(1);
      MachineInstr *Ret = BuildMI(*MBB, Call, DL, TII.get(BPF::RET))
                              .addReg(BPF::R0, RegState::Implicit)
                              .addReg(BPF::R3, RegState::Implicit);
      endBlockAfter(*Ret, nullptr);
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
