//=== WebAssemblyLateEHPrepare.cpp - WebAssembly Exception Preparation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Does various transformations for exception handling.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/WasmEHFuncInfo.h"
#include "llvm/MC/MCAsmInfo.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-exception-prepare"

namespace {
class WebAssemblyLateEHPrepare final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Late Prepare Exception";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool removeUnnecessaryUnreachables(MachineFunction &MF);
  bool replaceFuncletReturns(MachineFunction &MF);
  bool addCatches(MachineFunction &MF);
  bool addExceptionExtraction(MachineFunction &MF);
  bool restoreStackPointer(MachineFunction &MF);

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyLateEHPrepare() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyLateEHPrepare::ID = 0;
INITIALIZE_PASS(WebAssemblyLateEHPrepare, DEBUG_TYPE,
                "WebAssembly Late Exception Preparation", false, false)

FunctionPass *llvm::createWebAssemblyLateEHPrepare() {
  return new WebAssemblyLateEHPrepare();
}

// Returns the nearest EH pad that dominates this instruction. This does not use
// dominator analysis; it just does BFS on its predecessors until arriving at an
// EH pad. This assumes valid EH scopes so the first EH pad it arrives in all
// possible search paths should be the same.
// Returns nullptr in case it does not find any EH pad in the search, or finds
// multiple different EH pads.
static MachineBasicBlock *getMatchingEHPad(MachineInstr *MI) {
  MachineFunction *MF = MI->getParent()->getParent();
  SmallVector<MachineBasicBlock *, 2> WL;
  SmallPtrSet<MachineBasicBlock *, 2> Visited;
  WL.push_back(MI->getParent());
  MachineBasicBlock *EHPad = nullptr;
  while (!WL.empty()) {
    MachineBasicBlock *MBB = WL.pop_back_val();
    if (Visited.count(MBB))
      continue;
    Visited.insert(MBB);
    if (MBB->isEHPad()) {
      if (EHPad && EHPad != MBB)
        return nullptr;
      EHPad = MBB;
      continue;
    }
    if (MBB == &MF->front())
      return nullptr;
    WL.append(MBB->pred_begin(), MBB->pred_end());
  }
  return EHPad;
}

// Erase the specified BBs if the BB does not have any remaining predecessors,
// and also all its dead children.
template <typename Container>
static void eraseDeadBBsAndChildren(const Container &MBBs) {
  SmallVector<MachineBasicBlock *, 8> WL(MBBs.begin(), MBBs.end());
  while (!WL.empty()) {
    MachineBasicBlock *MBB = WL.pop_back_val();
    if (!MBB->pred_empty())
      continue;
    SmallVector<MachineBasicBlock *, 4> Succs(MBB->succ_begin(),
                                              MBB->succ_end());
    WL.append(MBB->succ_begin(), MBB->succ_end());
    for (auto *Succ : Succs)
      MBB->removeSuccessor(Succ);
    MBB->eraseFromParent();
  }
}

bool WebAssemblyLateEHPrepare::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Late EH Prepare **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');

  if (MF.getTarget().getMCAsmInfo()->getExceptionHandlingType() !=
      ExceptionHandling::Wasm)
    return false;

  bool Changed = false;
  Changed |= removeUnnecessaryUnreachables(MF);
  if (!MF.getFunction().hasPersonalityFn())
    return Changed;
  Changed |= replaceFuncletReturns(MF);
  Changed |= addCatches(MF);
  Changed |= addExceptionExtraction(MF);
  Changed |= restoreStackPointer(MF);
  return Changed;
}

bool WebAssemblyLateEHPrepare::removeUnnecessaryUnreachables(
    MachineFunction &MF) {
  bool Changed = false;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() != WebAssembly::THROW &&
          MI.getOpcode() != WebAssembly::RETHROW)
        continue;
      Changed = true;

      // The instruction after the throw should be an unreachable or a branch to
      // another BB that should eventually lead to an unreachable. Delete it
      // because throw itself is a terminator, and also delete successors if
      // any.
      MBB.erase(std::next(MachineBasicBlock::iterator(MI)), MBB.end());
      SmallVector<MachineBasicBlock *, 8> Succs(MBB.succ_begin(),
                                                MBB.succ_end());
      for (auto *Succ : Succs)
        MBB.removeSuccessor(Succ);
      eraseDeadBBsAndChildren(Succs);
    }
  }

  return Changed;
}

bool WebAssemblyLateEHPrepare::replaceFuncletReturns(MachineFunction &MF) {
  bool Changed = false;
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  for (auto &MBB : MF) {
    auto Pos = MBB.getFirstTerminator();
    if (Pos == MBB.end())
      continue;
    MachineInstr *TI = &*Pos;

    switch (TI->getOpcode()) {
    case WebAssembly::CATCHRET: {
      // Replace a catchret with a branch
      MachineBasicBlock *TBB = TI->getOperand(0).getMBB();
      if (!MBB.isLayoutSuccessor(TBB))
        BuildMI(MBB, TI, TI->getDebugLoc(), TII.get(WebAssembly::BR))
            .addMBB(TBB);
      TI->eraseFromParent();
      Changed = true;
      break;
    }
    case WebAssembly::CLEANUPRET: {
      // Replace a cleanupret with a rethrow
      BuildMI(MBB, TI, TI->getDebugLoc(), TII.get(WebAssembly::RETHROW));
      TI->eraseFromParent();
      Changed = true;
      break;
    }
    }
  }
  return Changed;
}

// Add catch instruction to beginning of catchpads and cleanuppads.
bool WebAssemblyLateEHPrepare::addCatches(MachineFunction &MF) {
  bool Changed = false;
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (auto &MBB : MF) {
    if (MBB.isEHPad()) {
      Changed = true;
      unsigned DstReg =
          MRI.createVirtualRegister(&WebAssembly::EXCEPT_REFRegClass);
      BuildMI(MBB, MBB.begin(), MBB.begin()->getDebugLoc(),
              TII.get(WebAssembly::CATCH), DstReg);
    }
  }
  return Changed;
}

// Wasm uses 'br_on_exn' instruction to check the tag of an exception. It takes
// except_ref type object returned by 'catch', and branches to the destination
// if it matches a given tag. We currently use __cpp_exception symbol to
// represent the tag for all C++ exceptions.
//
// block $l (result i32)
//   ...
//   ;; except_ref $e is on the stack at this point
//   br_on_exn $l $e ;; branch to $l with $e's arguments
//   ...
// end
// ;; Here we expect the extracted values are on top of the wasm value stack
// ... Handle exception using values ...
//
// br_on_exn takes an except_ref object and branches if it matches the given
// tag. There can be multiple br_on_exn instructions if we want to match for
// another tag, but for now we only test for __cpp_exception tag, and if it does
// not match, i.e., it is a foreign exception, we rethrow it.
//
// In the destination BB that's the target of br_on_exn, extracted exception
// values (in C++'s case a single i32, which represents an exception pointer)
// are placed on top of the wasm stack. Because we can't model wasm stack in
// LLVM instruction, we use 'extract_exception' pseudo instruction to retrieve
// it. The pseudo instruction will be deleted later.
bool WebAssemblyLateEHPrepare::addExceptionExtraction(MachineFunction &MF) {
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  auto *EHInfo = MF.getWasmEHFuncInfo();
  SmallVector<MachineInstr *, 16> ExtractInstrs;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() == WebAssembly::EXTRACT_EXCEPTION_I32) {
        if (MI.getOperand(0).isDead())
          MI.eraseFromParent();
        else
          ExtractInstrs.push_back(&MI);
      }
    }
  }
  if (ExtractInstrs.empty())
    return false;

  // Find terminate pads.
  SmallSet<MachineBasicBlock *, 8> TerminatePads;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.isCall()) {
        const MachineOperand &CalleeOp = MI.getOperand(0);
        if (CalleeOp.isGlobal() && CalleeOp.getGlobal()->getName() ==
                                       WebAssembly::ClangCallTerminateFn)
          TerminatePads.insert(getMatchingEHPad(&MI));
      }
    }
  }

  for (auto *Extract : ExtractInstrs) {
    MachineBasicBlock *EHPad = getMatchingEHPad(Extract);
    assert(EHPad && "No matching EH pad for extract_exception");
    MachineInstr *Catch = &*EHPad->begin();
    if (Catch->getNextNode() != Extract)
      EHPad->insert(Catch->getNextNode(), Extract->removeFromParent());

    // - Before:
    // ehpad:
    //   %exnref:except_ref = catch
    //   %exn:i32 = extract_exception
    //   ... use exn ...
    //
    // - After:
    // ehpad:
    //   %exnref:except_ref = catch
    //   br_on_exn %thenbb, $__cpp_exception, %exnref
    //   br %elsebb
    // elsebb:
    //   rethrow
    // thenbb:
    //   %exn:i32 = extract_exception
    //   ... use exn ...
    unsigned ExnRefReg = Catch->getOperand(0).getReg();
    auto *ThenMBB = MF.CreateMachineBasicBlock();
    auto *ElseMBB = MF.CreateMachineBasicBlock();
    MF.insert(std::next(MachineFunction::iterator(EHPad)), ElseMBB);
    MF.insert(std::next(MachineFunction::iterator(ElseMBB)), ThenMBB);
    ThenMBB->splice(ThenMBB->end(), EHPad, Extract, EHPad->end());
    ThenMBB->transferSuccessors(EHPad);
    EHPad->addSuccessor(ThenMBB);
    EHPad->addSuccessor(ElseMBB);

    DebugLoc DL = Extract->getDebugLoc();
    const char *CPPExnSymbol = MF.createExternalSymbolName("__cpp_exception");
    BuildMI(EHPad, DL, TII.get(WebAssembly::BR_ON_EXN))
        .addMBB(ThenMBB)
        .addExternalSymbol(CPPExnSymbol, WebAssemblyII::MO_SYMBOL_EVENT)
        .addReg(ExnRefReg);
    BuildMI(EHPad, DL, TII.get(WebAssembly::BR)).addMBB(ElseMBB);

    // When this is a terminate pad with __clang_call_terminate() call, we don't
    // rethrow it anymore and call __clang_call_terminate() with a nullptr
    // argument, which will call std::terminate().
    //
    // - Before:
    // ehpad:
    //   %exnref:except_ref = catch
    //   %exn:i32 = extract_exception
    //   call @__clang_call_terminate(%exn)
    //   unreachable
    //
    // - After:
    // ehpad:
    //   %exnref:except_ref = catch
    //   br_on_exn %thenbb, $__cpp_exception, %exnref
    //   br %elsebb
    // elsebb:
    //   call @__clang_call_terminate(0)
    //   unreachable
    // thenbb:
    //   %exn:i32 = extract_exception
    //   call @__clang_call_terminate(%exn)
    //   unreachable
    if (TerminatePads.count(EHPad)) {
      Function *ClangCallTerminateFn =
          MF.getFunction().getParent()->getFunction(
              WebAssembly::ClangCallTerminateFn);
      assert(ClangCallTerminateFn &&
             "There is no __clang_call_terminate() function");
      BuildMI(ElseMBB, DL, TII.get(WebAssembly::CALL_VOID))
          .addGlobalAddress(ClangCallTerminateFn)
          .addImm(0);
      BuildMI(ElseMBB, DL, TII.get(WebAssembly::UNREACHABLE));

    } else {
      BuildMI(ElseMBB, DL, TII.get(WebAssembly::RETHROW));
      if (EHInfo->hasEHPadUnwindDest(EHPad))
        EHInfo->setThrowUnwindDest(ElseMBB, EHInfo->getEHPadUnwindDest(EHPad));
    }
  }

  return true;
}

// After the stack is unwound due to a thrown exception, the __stack_pointer
// global can point to an invalid address. This inserts instructions that
// restore __stack_pointer global.
bool WebAssemblyLateEHPrepare::restoreStackPointer(MachineFunction &MF) {
  const auto *FrameLowering = static_cast<const WebAssemblyFrameLowering *>(
      MF.getSubtarget().getFrameLowering());
  if (!FrameLowering->needsPrologForEH(MF))
    return false;
  bool Changed = false;

  for (auto &MBB : MF) {
    if (!MBB.isEHPad())
      continue;
    Changed = true;

    // Insert __stack_pointer restoring instructions at the beginning of each EH
    // pad, after the catch instruction. Here it is safe to assume that SP32
    // holds the latest value of __stack_pointer, because the only exception for
    // this case is when a function uses the red zone, but that only happens
    // with leaf functions, and we don't restore __stack_pointer in leaf
    // functions anyway.
    auto InsertPos = MBB.begin();
    if (MBB.begin()->getOpcode() == WebAssembly::CATCH)
      InsertPos++;
    FrameLowering->writeSPToGlobal(WebAssembly::SP32, MF, MBB, InsertPos,
                                   MBB.begin()->getDebugLoc());
  }
  return Changed;
}
