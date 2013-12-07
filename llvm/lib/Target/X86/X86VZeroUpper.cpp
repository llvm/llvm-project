//===-- X86VZeroUpper.cpp - AVX vzeroupper instruction inserter -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass which inserts x86 AVX vzeroupper instructions
// before calls to SSE encoded functions. This avoids transition latency
// penalty when tranfering control between AVX encoded instructions and old
// SSE encoding mode.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-vzeroupper"
#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
using namespace llvm;

STATISTIC(NumVZU, "Number of vzeroupper instructions inserted");

namespace {
  struct VZeroUpperInserter : public MachineFunctionPass {
    static char ID;
    VZeroUpperInserter() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    bool processBasicBlock(MachineFunction &MF, MachineBasicBlock &MBB);

    virtual const char *getPassName() const { return "X86 vzeroupper inserter";}

  private:
    const TargetInstrInfo *TII; // Machine instruction info.

    // Any YMM register live-in to this function?
    bool FnHasLiveInYmm;

    // BBState - Contains the state of each MBB: unknown, clean, dirty
    SmallVector<uint8_t, 8> BBState;

    // BBSolved - Keep track of all MBB which had been already analyzed
    // and there is no further processing required.
    BitVector BBSolved;

    // Machine Basic Blocks are classified according this pass:
    //
    //  ST_UNKNOWN - The MBB state is unknown, meaning from the entry state
    //    until the MBB exit there isn't a instruction using YMM to change
    //    the state to dirty, or one of the incoming predecessors is unknown
    //    and there's not a dirty predecessor between them.
    //
    //  ST_CLEAN - No YMM usage in the end of the MBB. A MBB could have
    //    instructions using YMM and be marked ST_CLEAN, as long as the state
    //    is cleaned by a vzeroupper before any call.
    //
    //  ST_DIRTY - Any MBB ending with a YMM usage not cleaned up by a
    //    vzeroupper instruction.
    //
    //  ST_INIT - Placeholder for an empty state set
    //
    enum {
      ST_UNKNOWN = 0,
      ST_CLEAN   = 1,
      ST_DIRTY   = 2,
      ST_INIT    = 3
    };

    // computeState - Given two states, compute the resulting state, in
    // the following way
    //
    //  1) One dirty state yields another dirty state
    //  2) All states must be clean for the result to be clean
    //  3) If none above and one unknown, the result state is also unknown
    //
    static unsigned computeState(unsigned PrevState, unsigned CurState) {
      if (PrevState == ST_INIT)
        return CurState;

      if (PrevState == ST_DIRTY || CurState == ST_DIRTY)
        return ST_DIRTY;

      if (PrevState == ST_CLEAN && CurState == ST_CLEAN)
        return ST_CLEAN;

      return ST_UNKNOWN;
    }

  };
  char VZeroUpperInserter::ID = 0;
}

FunctionPass *llvm::createX86IssueVZeroUpperPass() {
  return new VZeroUpperInserter();
}

static bool isYmmReg(unsigned Reg) {
  return (Reg >= X86::YMM0 && Reg <= X86::YMM31);
}

static bool isZmmReg(unsigned Reg) {
  return (Reg >= X86::ZMM0 && Reg <= X86::ZMM31);
}

static bool checkFnHasLiveInYmm(MachineRegisterInfo &MRI) {
  for (MachineRegisterInfo::livein_iterator I = MRI.livein_begin(),
       E = MRI.livein_end(); I != E; ++I)
    if (isYmmReg(I->first) || isZmmReg(I->first))
      return true;

  return false;
}

static bool clobbersAllYmmRegs(const MachineOperand &MO) {
  for (unsigned reg = X86::YMM0; reg <= X86::YMM31; ++reg) {
    if (!MO.clobbersPhysReg(reg))
      return false;
  }
  for (unsigned reg = X86::ZMM0; reg <= X86::ZMM31; ++reg) {
    if (!MO.clobbersPhysReg(reg))
      return false;
  }
  return true;
}

static bool hasYmmReg(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MI->isCall() && MO.isRegMask() && !clobbersAllYmmRegs(MO))
      return true;
    if (!MO.isReg())
      continue;
    if (MO.isDebug())
      continue;
    if (isYmmReg(MO.getReg()))
      return true;
  }
  return false;
}

/// clobbersAnyYmmReg() - Check if any YMM register will be clobbered by this
/// instruction.
static bool clobbersAnyYmmReg(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isRegMask())
      continue;
    for (unsigned reg = X86::YMM0; reg <= X86::YMM31; ++reg) {
      if (MO.clobbersPhysReg(reg))
        return true;
    }
    for (unsigned reg = X86::ZMM0; reg <= X86::ZMM31; ++reg) {
      if (MO.clobbersPhysReg(reg))
        return true;
    }
  }
  return false;
}

/// runOnMachineFunction - Loop over all of the basic blocks, inserting
/// vzero upper instructions before function calls.
bool VZeroUpperInserter::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool EverMadeChange = false;

  // Fast check: if the function doesn't use any ymm registers, we don't need
  // to insert any VZEROUPPER instructions.  This is constant-time, so it is
  // cheap in the common case of no ymm use.
  bool YMMUsed = false;
  const TargetRegisterClass *RC = &X86::VR256RegClass;
  for (TargetRegisterClass::iterator i = RC->begin(), e = RC->end();
       i != e; i++) {
    if (!MRI.reg_nodbg_empty(*i)) {
      YMMUsed = true;
      break;
    }
  }
  if (!YMMUsed)
    return EverMadeChange;

  // Pre-compute the existence of any live-in YMM registers to this function
  FnHasLiveInYmm = checkFnHasLiveInYmm(MRI);

  assert(BBState.empty());
  BBState.resize(MF.getNumBlockIDs(), 0);
  BBSolved.resize(MF.getNumBlockIDs(), 0);

  // Each BB state depends on all predecessors, loop over until everything
  // converges.  (Once we converge, we can implicitly mark everything that is
  // still ST_UNKNOWN as ST_CLEAN.)
  while (1) {
    bool MadeChange = false;

    // Process all basic blocks.
    for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
      MadeChange |= processBasicBlock(MF, *I);

    // If this iteration over the code changed anything, keep iterating.
    if (!MadeChange) break;
    EverMadeChange = true;
  }

  BBState.clear();
  BBSolved.clear();
  return EverMadeChange;
}

/// processBasicBlock - Loop over all of the instructions in the basic block,
/// inserting vzero upper instructions before function calls.
bool VZeroUpperInserter::processBasicBlock(MachineFunction &MF,
                                           MachineBasicBlock &BB) {
  bool Changed = false;
  unsigned BBNum = BB.getNumber();

  // Don't process already solved BBs
  if (BBSolved[BBNum])
    return false; // No changes

  // Check the state of all predecessors
  unsigned EntryState = ST_INIT;
  for (MachineBasicBlock::const_pred_iterator PI = BB.pred_begin(),
       PE = BB.pred_end(); PI != PE; ++PI) {
    EntryState = computeState(EntryState, BBState[(*PI)->getNumber()]);
    if (EntryState == ST_DIRTY)
      break;
  }


  // The entry MBB for the function may set the initial state to dirty if
  // the function receives any YMM incoming arguments
  if (&BB == MF.begin()) {
    EntryState = ST_CLEAN;
    if (FnHasLiveInYmm)
      EntryState = ST_DIRTY;
  }

  // The current state is initialized according to the predecessors
  unsigned CurState = EntryState;
  bool BBHasCall = false;

  for (MachineBasicBlock::iterator I = BB.begin(); I != BB.end(); ++I) {
    DebugLoc dl = I->getDebugLoc();
    MachineInstr *MI = I;

    bool isControlFlow = MI->isCall() || MI->isReturn();

    // Shortcut: don't need to check regular instructions in dirty state.
    if (!isControlFlow && CurState == ST_DIRTY)
      continue;

    if (hasYmmReg(MI)) {
      // We found a ymm-using instruction; this could be an AVX instruction,
      // or it could be control flow.
      CurState = ST_DIRTY;
      continue;
    }

    // Check for control-flow out of the current function (which might
    // indirectly execute SSE instructions).
    if (!isControlFlow)
      continue;

    // If the call won't clobber any YMM register, skip it as well. It usually
    // happens on helper function calls (such as '_chkstk', '_ftol2') where
    // standard calling convention is not used (RegMask is not used to mark
    // register clobbered and register usage (def/imp-def/use) is well-dfined
    // and explicitly specified.
    if (MI->isCall() && !clobbersAnyYmmReg(MI))
      continue;

    BBHasCall = true;

    // The VZEROUPPER instruction resets the upper 128 bits of all Intel AVX
    // registers. This instruction has zero latency. In addition, the processor
    // changes back to Clean state, after which execution of Intel SSE
    // instructions or Intel AVX instructions has no transition penalty. Add
    // the VZEROUPPER instruction before any function call/return that might
    // execute SSE code.
    // FIXME: In some cases, we may want to move the VZEROUPPER into a
    // predecessor block.
    if (CurState == ST_DIRTY) {
      // Only insert the VZEROUPPER in case the entry state isn't unknown.
      // When unknown, only compute the information within the block to have
      // it available in the exit if possible, but don't change the block.
      if (EntryState != ST_UNKNOWN) {
        BuildMI(BB, I, dl, TII->get(X86::VZEROUPPER));
        ++NumVZU;
      }

      // After the inserted VZEROUPPER the state becomes clean again, but
      // other YMM may appear before other subsequent calls or even before
      // the end of the BB.
      CurState = ST_CLEAN;
    }
  }

  DEBUG(dbgs() << "MBB #" << BBNum
               << ", current state: " << CurState << '\n');

  // A BB can only be considered solved when we both have done all the
  // necessary transformations, and have computed the exit state.  This happens
  // in two cases:
  //  1) We know the entry state: this immediately implies the exit state and
  //     all the necessary transformations.
  //  2) There are no calls, and and a non-call instruction marks this block:
  //     no transformations are necessary, and we know the exit state.
  if (EntryState != ST_UNKNOWN || (!BBHasCall && CurState != ST_UNKNOWN))
    BBSolved[BBNum] = true;

  if (CurState != BBState[BBNum])
    Changed = true;

  BBState[BBNum] = CurState;
  return Changed;
}
