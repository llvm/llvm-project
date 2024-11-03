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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/WasmEHFuncInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-late-eh-prepare"

namespace {
class WebAssemblyLateEHPrepare final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Late Prepare Exception";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool removeUnreachableEHPads(MachineFunction &MF);
  void recordCatchRetBBs(MachineFunction &MF);
  bool hoistCatches(MachineFunction &MF);
  bool addCatchAlls(MachineFunction &MF);
  bool addCatchRefsAndThrowRefs(MachineFunction &MF);
  bool replaceFuncletReturns(MachineFunction &MF);
  bool removeUnnecessaryUnreachables(MachineFunction &MF);
  bool restoreStackPointer(MachineFunction &MF);

  MachineBasicBlock *getMatchingEHPad(MachineInstr *MI);
  SmallPtrSet<MachineBasicBlock *, 8> CatchRetBBs;

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
MachineBasicBlock *
WebAssemblyLateEHPrepare::getMatchingEHPad(MachineInstr *MI) {
  MachineFunction *MF = MI->getParent()->getParent();
  SmallVector<MachineBasicBlock *, 2> WL;
  SmallPtrSet<MachineBasicBlock *, 2> Visited;
  WL.push_back(MI->getParent());
  MachineBasicBlock *EHPad = nullptr;
  while (!WL.empty()) {
    MachineBasicBlock *MBB = WL.pop_back_val();
    if (!Visited.insert(MBB).second)
      continue;
    if (MBB->isEHPad()) {
      if (EHPad && EHPad != MBB)
        return nullptr;
      EHPad = MBB;
      continue;
    }
    if (MBB == &MF->front())
      return nullptr;
    for (auto *Pred : MBB->predecessors())
      if (!CatchRetBBs.count(Pred)) // We don't go into child scopes
        WL.push_back(Pred);
  }
  return EHPad;
}

// Erase the specified BBs if the BB does not have any remaining predecessors,
// and also all its dead children.
template <typename Container>
static void eraseDeadBBsAndChildren(const Container &MBBs) {
  SmallVector<MachineBasicBlock *, 8> WL(MBBs.begin(), MBBs.end());
  SmallPtrSet<MachineBasicBlock *, 8> Deleted;
  while (!WL.empty()) {
    MachineBasicBlock *MBB = WL.pop_back_val();
    if (Deleted.count(MBB) || !MBB->pred_empty())
      continue;
    SmallVector<MachineBasicBlock *, 4> Succs(MBB->successors());
    WL.append(MBB->succ_begin(), MBB->succ_end());
    for (auto *Succ : Succs)
      MBB->removeSuccessor(Succ);
    // To prevent deleting the same BB multiple times, which can happen when
    // 'MBBs' contain both a parent and a child
    Deleted.insert(MBB);
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
  if (MF.getFunction().hasPersonalityFn()) {
    Changed |= removeUnreachableEHPads(MF);
    recordCatchRetBBs(MF);
    Changed |= hoistCatches(MF);
    Changed |= addCatchAlls(MF);
    Changed |= replaceFuncletReturns(MF);
    if (WebAssembly::WasmEnableExnref)
      Changed |= addCatchRefsAndThrowRefs(MF);
  }
  Changed |= removeUnnecessaryUnreachables(MF);
  if (MF.getFunction().hasPersonalityFn())
    Changed |= restoreStackPointer(MF);
  return Changed;
}

// Remove unreachable EH pads and its children. If they remain, CFG
// stackification can be tricky.
bool WebAssemblyLateEHPrepare::removeUnreachableEHPads(MachineFunction &MF) {
  SmallVector<MachineBasicBlock *, 4> ToDelete;
  for (auto &MBB : MF)
    if (MBB.isEHPad() && MBB.pred_empty())
      ToDelete.push_back(&MBB);
  eraseDeadBBsAndChildren(ToDelete);
  return !ToDelete.empty();
}

// Record which BB ends with catchret instruction, because this will be replaced
// with 'br's later. This set of catchret BBs is necessary in 'getMatchingEHPad'
// function.
void WebAssemblyLateEHPrepare::recordCatchRetBBs(MachineFunction &MF) {
  CatchRetBBs.clear();
  for (auto &MBB : MF) {
    auto Pos = MBB.getFirstTerminator();
    if (Pos == MBB.end())
      continue;
    MachineInstr *TI = &*Pos;
    if (TI->getOpcode() == WebAssembly::CATCHRET)
      CatchRetBBs.insert(&MBB);
  }
}

// Hoist catch instructions to the beginning of their matching EH pad BBs in
// case,
// (1) catch instruction is not the first instruction in EH pad.
// ehpad:
//   some_other_instruction
//   ...
//   %exn = catch 0
// (2) catch instruction is in a non-EH pad BB. For example,
// ehpad:
//   br bb0
// bb0:
//   %exn = catch 0
bool WebAssemblyLateEHPrepare::hoistCatches(MachineFunction &MF) {
  bool Changed = false;
  SmallVector<MachineInstr *, 16> Catches;
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      if (WebAssembly::isCatch(MI.getOpcode()))
        Catches.push_back(&MI);

  for (auto *Catch : Catches) {
    MachineBasicBlock *EHPad = getMatchingEHPad(Catch);
    assert(EHPad && "No matching EH pad for catch");
    auto InsertPos = EHPad->begin();
    // Skip EH_LABELs in the beginning of an EH pad if present. We don't use
    // these labels at the moment, but other targets also seem to have an
    // EH_LABEL instruction in the beginning of an EH pad.
    while (InsertPos != EHPad->end() && InsertPos->isEHLabel())
      InsertPos++;
    if (InsertPos == Catch)
      continue;
    Changed = true;
    EHPad->insert(InsertPos, Catch->removeFromParent());
  }
  return Changed;
}

// Add catch_all to beginning of cleanup pads.
bool WebAssemblyLateEHPrepare::addCatchAlls(MachineFunction &MF) {
  bool Changed = false;
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  for (auto &MBB : MF) {
    if (!MBB.isEHPad())
      continue;
    auto InsertPos = MBB.begin();
    // Skip EH_LABELs in the beginning of an EH pad if present.
    while (InsertPos != MBB.end() && InsertPos->isEHLabel())
      InsertPos++;
    // This runs after hoistCatches(), so we assume that if there is a catch,
    // that should be the first non-EH-label instruction in an EH pad.
    if (InsertPos == MBB.end() ||
        !WebAssembly::isCatch(InsertPos->getOpcode())) {
      Changed = true;
      unsigned CatchAllOpcode = WebAssembly::WasmEnableExnref
                                    ? WebAssembly::CATCH_ALL
                                    : WebAssembly::CATCH_ALL_LEGACY;
      BuildMI(MBB, InsertPos,
              InsertPos == MBB.end() ? DebugLoc() : InsertPos->getDebugLoc(),
              TII.get(CatchAllOpcode));
    }
  }
  return Changed;
}

// Replace pseudo-instructions catchret and cleanupret with br and rethrow
// respectively.
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
      // Replace a cleanupret with a rethrow. For C++ support, currently
      // rethrow's immediate argument is always 0 (= the latest exception).
      //
      // Even when -wasm-enable-exnref is true, we use a RETHROW here for the
      // moment. This will be converted to a THROW_REF in
      // addCatchRefsAndThrowRefs.
      BuildMI(MBB, TI, TI->getDebugLoc(), TII.get(WebAssembly::RETHROW))
          .addImm(0);
      TI->eraseFromParent();
      Changed = true;
      break;
    }
    }
  }
  return Changed;
}

// Add CATCH_REF and CATCH_ALL_REF pseudo instructions to EH pads, and convert
// RETHROWs to THROW_REFs.
bool WebAssemblyLateEHPrepare::addCatchRefsAndThrowRefs(MachineFunction &MF) {
  bool Changed = false;
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  auto &MRI = MF.getRegInfo();
  DenseMap<MachineBasicBlock *, SmallVector<MachineInstr *, 2>> EHPadToRethrows;

  // Create a map of <EH pad, a vector of RETHROWs rethrowing its exception>
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() == WebAssembly::RETHROW) {
        Changed = true;
        auto *EHPad = getMatchingEHPad(&MI);
        EHPadToRethrows[EHPad].push_back(&MI);
      }
    }
  }

  // Convert CATCH into CATCH_REF and CATCH_ALL into CATCH_ALL_REF, when the
  // caught exception is rethrown. And convert RETHROWs to THROW_REFs.
  for (auto &[EHPad, Rethrows] : EHPadToRethrows) {
    auto *Catch = WebAssembly::findCatch(EHPad);
    auto *InsertPos = Catch->getIterator()->getNextNode();
    auto ExnReg = MRI.createVirtualRegister(&WebAssembly::EXNREFRegClass);
    if (Catch->getOpcode() == WebAssembly::CATCH) {
      MachineInstrBuilder MIB = BuildMI(*EHPad, InsertPos, Catch->getDebugLoc(),
                                        TII.get(WebAssembly::CATCH_REF));
      // Copy defs (= extracted values) from the old CATCH to the new CATCH_REF
      for (const auto &Def : Catch->defs())
        MIB.addDef(Def.getReg());
      MIB.addDef(ExnReg); // Attach the exnref def after extracted values
      // Copy the tag symbol (The only use operand a CATCH can have is the tag
      // symbol)
      for (const auto &Use : Catch->uses()) {
        MIB.addExternalSymbol(Use.getSymbolName());
        break;
      }
    } else if (Catch->getOpcode() == WebAssembly::CATCH_ALL) {
      BuildMI(*EHPad, InsertPos, Catch->getDebugLoc(),
              TII.get(WebAssembly::CATCH_ALL_REF))
          .addDef(ExnReg);
    } else {
      assert(false);
    }
    Catch->eraseFromParent();

    for (auto *Rethrow : Rethrows) {
      auto InsertPos = std::next(Rethrow->getIterator());
      BuildMI(*Rethrow->getParent(), InsertPos, Rethrow->getDebugLoc(),
              TII.get(WebAssembly::THROW_REF))
          .addReg(ExnReg);
      Rethrow->eraseFromParent();
    }
  }

  return Changed;
}

// Remove unnecessary unreachables after a throw/rethrow/throw_ref.
bool WebAssemblyLateEHPrepare::removeUnnecessaryUnreachables(
    MachineFunction &MF) {
  bool Changed = false;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() != WebAssembly::THROW &&
          MI.getOpcode() != WebAssembly::RETHROW &&
          MI.getOpcode() != WebAssembly::THROW_REF)
        continue;
      Changed = true;

      // The instruction after the throw should be an unreachable or a branch to
      // another BB that should eventually lead to an unreachable. Delete it
      // because throw itself is a terminator, and also delete successors if
      // any.
      MBB.erase(std::next(MI.getIterator()), MBB.end());
      SmallVector<MachineBasicBlock *, 8> Succs(MBB.successors());
      for (auto *Succ : Succs)
        if (!Succ->isEHPad())
          MBB.removeSuccessor(Succ);
      eraseDeadBBsAndChildren(Succs);
    }
  }

  return Changed;
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
    // Skip EH_LABELs in the beginning of an EH pad if present.
    while (InsertPos != MBB.end() && InsertPos->isEHLabel())
      InsertPos++;
    assert(InsertPos != MBB.end() &&
           WebAssembly::isCatch(InsertPos->getOpcode()) &&
           "catch/catch_all should be present in every EH pad at this point");
    ++InsertPos; // Skip the catch instruction
    FrameLowering->writeSPToGlobal(FrameLowering->getSPReg(MF), MF, MBB,
                                   InsertPos, MBB.begin()->getDebugLoc());
  }
  return Changed;
}
