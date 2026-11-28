//===----- X86AvoidTrailingCall.cpp - Insert int3 after trailing calls ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Windows x64 unwinder decodes the instruction stream during unwinding.
// The unwinder decodes forward from the current PC to detect epilogue code
// patterns.
//
// First, this means that there must be an instruction after every
// call instruction for the unwinder to decode. LLVM must maintain the invariant
// that the last instruction of a function or funclet is not a call, or the
// unwinder may decode into the next function. Similarly, a call may not
// immediately precede an epilogue code pattern, or return across an IP2State
// boundary. In both cases the return address must stay in the call's EH
// region, so insert padding before the boundary.
//
// Second, all non-tail call jump targets must be within the *half-open*
// interval of the bounds of the function. The unwinder distinguishes between
// internal jump instructions and tail calls in an epilogue sequence by checking
// the jump target against the function bounds from the .pdata section. This
// means that the last regular MBB of an LLVM function must not be empty if
// there are regular jumps targeting it.
//
// This pass inserts padding after calls and ensures that blocks at the end of
// a function or funclet are not empty.
//
// Unwinder implementation for reference:
// https://github.com/dotnet/coreclr/blob/a9f3fc16483eecfc47fb79c362811d870be02249/src/unwinder/amd64/unwinder_amd64.cpp#L1015
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/MC/MCAsmInfo.h"
#include <optional>

#define AVOIDCALL_DESC "X86 avoid trailing call pass"
#define AVOIDCALL_NAME "x86-avoid-trailing-call"

#define DEBUG_TYPE AVOIDCALL_NAME

using namespace llvm;

namespace {
class X86AvoidTrailingCallLegacyPass : public MachineFunctionPass {
public:
  X86AvoidTrailingCallLegacyPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;

private:
  StringRef getPassName() const override { return AVOIDCALL_DESC; }
};
} // end anonymous namespace

char X86AvoidTrailingCallLegacyPass::ID = 0;

FunctionPass *llvm::createX86AvoidTrailingCallLegacyPass() {
  return new X86AvoidTrailingCallLegacyPass();
}

INITIALIZE_PASS(X86AvoidTrailingCallLegacyPass, AVOIDCALL_NAME, AVOIDCALL_DESC,
                false, false)

// A real instruction is a non-meta, non-pseudo instruction.  Some pseudos
// expand to nothing, and some expand to code. This logic conservatively assumes
// they might expand to nothing.
static bool isCallOrRealInstruction(MachineInstr &MI) {
  return MI.isCall() || (!MI.isPseudo() && !MI.isMetaInstruction());
}

// Return true if this is a call instruction, but not a tail call.
static bool isCallInstruction(const MachineInstr &MI) {
  return MI.isCall() && !MI.isReturn();
}

static bool isUnwindingInlineAsm(const MachineInstr &MI) {
  // Empty inline asm emits no instructions, even if marked as unwinding.
  return MI.isInlineAsm() &&
         MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName()[0] != '\0' &&
         (MI.getOperand(InlineAsm::MIOp_ExtraInfo).getImm() &
          InlineAsm::Extra_MayUnwind);
}

static std::optional<unsigned> getWinEHCallPadding(const MachineInstr &MI) {
  // During a call, the "current address" for the calling frame is the return
  // address from that call. This can cause issues with Win x64 unwinding if
  // the instruction immediately following the call is in a different IP2State
  // (that is EH) region than the call instruction itself. To workaround this,
  // we may need to insert padding after the call instruction.

  const MachineBasicBlock &CallMBB = *MI.getParent();
  const MachineFunction &MF = *CallMBB.getParent();

  // This padding is only needed for table-based Win64 unwinding.
  if (MF.getFunction().getParent()->getExceptionModel() !=
          ExceptionHandling::WinEH ||
      MF.getTarget().getMCAsmInfo().getWinEHEncodingType() !=
          WinEH::EncodingType::Itanium)
    return std::nullopt;

  // Start from the instruction after the call and check if padding is needed.
  bool HasEHPersonality = MF.getWinEHFuncInfo() != nullptr;
  auto MBBI = CallMBB.getIterator();
  auto I = std::next(MI.getIterator());
  while (true) {
    for (; I != MBBI->instr_end(); ++I) {
      // Keep the return address outside the epilogue, even without an EH
      // personality.
      if (I->getOpcode() == X86::SEH_BeginEpilogue)
        return X86::NOOP;

      // Keep the return address in the call's IP2State range.
      if (I->isEHLabel()) {
        if (HasEHPersonality)
          return X86::NOOP;
        continue;
      }

      // A real instruction already provides a return address in the same
      // range. Keep looking past pseudos, which may emit no bytes.
      if (!I->isPseudo() && !I->isMetaInstruction())
        return std::nullopt;
    }
    if (++MBBI == MF.end())
      break;
    I = MBBI->instr_begin();
  }

  // At function end, there is no IP2State range to extend without an EH
  // personality.
  if (!HasEHPersonality)
    return std::nullopt;

  // Trap if the call's block has no successors; otherwise use a NOP so
  // execution can continue if the call returns.
  return CallMBB.succ_empty() ? X86::INT3 : X86::NOOP;
}

static bool padFuncletEnds(MachineFunction &MF, const X86InstrInfo &TII) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // Look for basic blocks that precede funclet entries or are at the end of
    // the function.
    MachineBasicBlock *NextMBB = MBB.getNextNode();
    if (NextMBB && !NextMBB->isEHFuncletEntry())
      continue;

    // Find the last real instruction in this block.
    auto LastRealInstr = llvm::find_if(reverse(MBB), isCallOrRealInstruction);

    // If the block is empty or the last real instruction is a call instruction,
    // insert an int3. If there is a call instruction, insert the int3 between
    // the call and any labels or other meta instructions. If the block is
    // empty, insert at block end.
    bool IsEmpty = LastRealInstr == MBB.rend();
    bool IsCall = !IsEmpty && isCallInstruction(*LastRealInstr);
    if (IsEmpty || IsCall) {
      LLVM_DEBUG({
        if (IsCall) {
          dbgs() << "inserting int3 after trailing call instruction:\n";
          LastRealInstr->dump();
          dbgs() << '\n';
        } else {
          dbgs() << "inserting int3 in trailing empty MBB:\n";
          MBB.dump();
        }
      });

      MachineBasicBlock::iterator MBBI = MBB.end();
      DebugLoc DL;
      if (IsCall) {
        MBBI = std::next(LastRealInstr.getReverse());
        DL = LastRealInstr->getDebugLoc();
      }
      BuildMI(MBB, MBBI, DL, TII.get(X86::INT3));
      Changed = true;
    }
  }

  return Changed;
}

static bool updateX86AvoidTrailingCall(MachineFunction &MF) {
  const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
  const X86InstrInfo &TII = *STI.getInstrInfo();
  bool Changed =
      STI.isTargetWin64() && MF.hasWinCFI() && padFuncletEnds(MF, TII);

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Pad statepoints even if they reserve patch bytes instead of emitting a
      // call. The runtime can replace those bytes with a call, so the WinEH
      // padding must be outside the patch region.
      if (!(isCallInstruction(MI) && !MI.isPseudo()) &&
          MI.getOpcode() != TargetOpcode::STATEPOINT &&
          !isUnwindingInlineAsm(MI))
        continue;

      if (auto Padding = getWinEHCallPadding(MI)) {
        // Insert in the call's block, before any following EH labels.
        // Post-instruction symbols and statepoint stack map labels will point
        // to the start of the padding.
        BuildMI(MBB, std::next(MI.getIterator()), DebugLoc(),
                TII.get(*Padding));
        Changed = true;
      }
    }
  }
  return Changed;
}

bool X86AvoidTrailingCallLegacyPass::runOnMachineFunction(MachineFunction &MF) {
  return updateX86AvoidTrailingCall(MF);
}

PreservedAnalyses
X86AvoidTrailingCallPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  bool Changed = updateX86AvoidTrailingCall(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
