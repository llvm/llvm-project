//===-- PrologEpilogInserter.cpp - Insert Prolog/Epilog code in function --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
// This pass provides an optional shrink wrapping variant of prolog/epilog
// insertion, enabled via --shrink-wrap. See ShrinkWrapping.cpp.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pei"
#include "PrologEpilogInserter.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <climits>

using namespace llvm;

char PEI::ID = 0;
char &llvm::PrologEpilogCodeInserterID = PEI::ID;

INITIALIZE_PASS_BEGIN(PEI, "prologepilog",
                "Prologue/Epilogue Insertion", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(PEI, "prologepilog",
                    "Prologue/Epilogue Insertion & Frame Finalization",
                    false, false)

STATISTIC(NumScavengedRegs, "Number of frame index regs scavenged");
STATISTIC(NumBytesStackSpace,
          "Number of bytes used for stack in all functions");

/// runOnMachineFunction - Insert prolog/epilog code and replace abstract
/// frame indexes with appropriate references.
///
bool PEI::runOnMachineFunction(MachineFunction &Fn) {
  const Function* F = Fn.getFunction();
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
  const TargetFrameLowering *TFI = Fn.getTarget().getFrameLowering();

  assert(!Fn.getRegInfo().getNumVirtRegs() && "Regalloc must assign all vregs");

  RS = TRI->requiresRegisterScavenging(Fn) ? new RegScavenger() : NULL;
  FrameIndexVirtualScavenging = TRI->requiresFrameIndexScavenging(Fn);

  // Calculate the MaxCallFrameSize and AdjustsStack variables for the
  // function's frame information. Also eliminates call frame pseudo
  // instructions.
  calculateCallsInformation(Fn);

  // Allow the target machine to make some adjustments to the function
  // e.g. UsedPhysRegs before calculateCalleeSavedRegisters.
  TFI->processFunctionBeforeCalleeSavedScan(Fn, RS);

  // Scan the function for modified callee saved registers and insert spill code
  // for any callee saved registers that are modified.
  calculateCalleeSavedRegisters(Fn);

  // Determine placement of CSR spill/restore code:
  //  - With shrink wrapping, place spills and restores to tightly
  //    enclose regions in the Machine CFG of the function where
  //    they are used.
  //  - Without shink wrapping (default), place all spills in the
  //    entry block, all restores in return blocks.
  placeCSRSpillsAndRestores(Fn);

  // Add the code to save and restore the callee saved registers
  if (!F->getAttributes().hasAttribute(AttributeSet::FunctionIndex,
                                       Attribute::Naked))
    insertCSRSpillsAndRestores(Fn);

  // Allow the target machine to make final modifications to the function
  // before the frame layout is finalized.
  TFI->processFunctionBeforeFrameFinalized(Fn, RS);

  // Calculate actual frame offsets for all abstract stack objects...
  calculateFrameObjectOffsets(Fn);

  // Add prolog and epilog code to the function.  This function is required
  // to align the stack frame as necessary for any stack variables or
  // called functions.  Because of this, calculateCalleeSavedRegisters()
  // must be called before this function in order to set the AdjustsStack
  // and MaxCallFrameSize variables.
  if (!F->getAttributes().hasAttribute(AttributeSet::FunctionIndex,
                                       Attribute::Naked))
    insertPrologEpilogCode(Fn);

  // Replace all MO_FrameIndex operands with physical register references
  // and actual offsets.
  //
  replaceFrameIndices(Fn);

  // If register scavenging is needed, as we've enabled doing it as a
  // post-pass, scavenge the virtual registers that frame index elimiation
  // inserted.
  if (TRI->requiresRegisterScavenging(Fn) && FrameIndexVirtualScavenging)
    scavengeFrameVirtualRegs(Fn);

  // Clear any vregs created by virtual scavenging.
  Fn.getRegInfo().clearVirtRegs();

  delete RS;
  clearAllSets();
  return true;
}

/// calculateCallsInformation - Calculate the MaxCallFrameSize and AdjustsStack
/// variables for the function's frame information and eliminate call frame
/// pseudo instructions.
void PEI::calculateCallsInformation(MachineFunction &Fn) {
  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();
  const TargetFrameLowering *TFI = Fn.getTarget().getFrameLowering();
  MachineFrameInfo *MFI = Fn.getFrameInfo();

  unsigned MaxCallFrameSize = 0;
  bool AdjustsStack = MFI->adjustsStack();

  // Get the function call frame set-up and tear-down instruction opcode
  int FrameSetupOpcode   = TII.getCallFrameSetupOpcode();
  int FrameDestroyOpcode = TII.getCallFrameDestroyOpcode();

  // Early exit for targets which have no call frame setup/destroy pseudo
  // instructions.
  if (FrameSetupOpcode == -1 && FrameDestroyOpcode == -1)
    return;

  std::vector<MachineBasicBlock::iterator> FrameSDOps;
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB)
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        assert(I->getNumOperands() >= 1 && "Call Frame Setup/Destroy Pseudo"
               " instructions should have a single immediate argument!");
        unsigned Size = I->getOperand(0).getImm();
        if (Size > MaxCallFrameSize) MaxCallFrameSize = Size;
        AdjustsStack = true;
        FrameSDOps.push_back(I);
      } else if (I->isInlineAsm()) {
        // Some inline asm's need a stack frame, as indicated by operand 1.
        unsigned ExtraInfo = I->getOperand(InlineAsm::MIOp_ExtraInfo).getImm();
        if (ExtraInfo & InlineAsm::Extra_IsAlignStack)
          AdjustsStack = true;
      }

  MFI->setAdjustsStack(AdjustsStack);
  MFI->setMaxCallFrameSize(MaxCallFrameSize);

  for (std::vector<MachineBasicBlock::iterator>::iterator
         i = FrameSDOps.begin(), e = FrameSDOps.end(); i != e; ++i) {
    MachineBasicBlock::iterator I = *i;

    // If call frames are not being included as part of the stack frame, and
    // the target doesn't indicate otherwise, remove the call frame pseudos
    // here. The sub/add sp instruction pairs are still inserted, but we don't
    // need to track the SP adjustment for frame index elimination.
    if (TFI->canSimplifyCallFramePseudos(Fn))
      TFI->eliminateCallFramePseudoInstr(Fn, *I->getParent(), I);
  }
}


/// calculateCalleeSavedRegisters - Scan the function for modified callee saved
/// registers.
void PEI::calculateCalleeSavedRegisters(MachineFunction &F) {
  const TargetRegisterInfo *RegInfo = F.getTarget().getRegisterInfo();
  const TargetFrameLowering *TFI = F.getTarget().getFrameLowering();
  MachineFrameInfo *MFI = F.getFrameInfo();

  // Get the callee saved register list...
  const uint16_t *CSRegs = RegInfo->getCalleeSavedRegs(&F);

  // These are used to keep track the callee-save area. Initialize them.
  MinCSFrameIndex = INT_MAX;
  MaxCSFrameIndex = 0;

  // Early exit for targets which have no callee saved registers.
  if (CSRegs == 0 || CSRegs[0] == 0)
    return;

  // In Naked functions we aren't going to save any registers.
  if (F.getFunction()->getAttributes().hasAttribute(AttributeSet::FunctionIndex,
                                                    Attribute::Naked))
    return;

  std::vector<CalleeSavedInfo> CSI;
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    // Functions which call __builtin_unwind_init get all their registers saved.
    if (F.getRegInfo().isPhysRegUsed(Reg) || F.getMMI().callsUnwindInit()) {
      // If the reg is modified, save it!
      CSI.push_back(CalleeSavedInfo(Reg));
    }
  }

  if (CSI.empty())
    return;   // Early exit if no callee saved registers are modified!

  unsigned NumFixedSpillSlots;
  const TargetFrameLowering::SpillSlot *FixedSpillSlots =
    TFI->getCalleeSavedSpillSlots(NumFixedSpillSlots);

  // Now that we know which registers need to be saved and restored, allocate
  // stack slots for them.
  for (std::vector<CalleeSavedInfo>::iterator
         I = CSI.begin(), E = CSI.end(); I != E; ++I) {
    unsigned Reg = I->getReg();
    const TargetRegisterClass *RC = RegInfo->getMinimalPhysRegClass(Reg);

    int FrameIdx;
    if (RegInfo->hasReservedSpillSlot(F, Reg, FrameIdx)) {
      I->setFrameIdx(FrameIdx);
      continue;
    }

    // Check to see if this physreg must be spilled to a particular stack slot
    // on this target.
    const TargetFrameLowering::SpillSlot *FixedSlot = FixedSpillSlots;
    while (FixedSlot != FixedSpillSlots+NumFixedSpillSlots &&
           FixedSlot->Reg != Reg)
      ++FixedSlot;

    if (FixedSlot == FixedSpillSlots + NumFixedSpillSlots) {
      // Nope, just spill it anywhere convenient.
      unsigned Align = RC->getAlignment();
      unsigned StackAlign = TFI->getStackAlignment();

      // We may not be able to satisfy the desired alignment specification of
      // the TargetRegisterClass if the stack alignment is smaller. Use the
      // min.
      Align = std::min(Align, StackAlign);
      FrameIdx = MFI->CreateStackObject(RC->getSize(), Align, true);
      if ((unsigned)FrameIdx < MinCSFrameIndex) MinCSFrameIndex = FrameIdx;
      if ((unsigned)FrameIdx > MaxCSFrameIndex) MaxCSFrameIndex = FrameIdx;
    } else {
      // Spill it to the stack where we must.
      FrameIdx = MFI->CreateFixedObject(RC->getSize(), FixedSlot->Offset, true);
    }

    I->setFrameIdx(FrameIdx);
  }

  MFI->setCalleeSavedInfo(CSI);
}

/// insertCSRSpillsAndRestores - Insert spill and restore code for
/// callee saved registers used in the function, handling shrink wrapping.
///
void PEI::insertCSRSpillsAndRestores(MachineFunction &Fn) {
  // Get callee saved register information.
  MachineFrameInfo *MFI = Fn.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  MFI->setCalleeSavedInfoValid(true);

  // Early exit if no callee saved registers are modified!
  if (CSI.empty())
    return;

  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();
  const TargetFrameLowering *TFI = Fn.getTarget().getFrameLowering();
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
  MachineBasicBlock::iterator I;

  if (!ShrinkWrapThisFunction) {
    // Spill using target interface.
    I = EntryBlock->begin();
    if (!TFI->spillCalleeSavedRegisters(*EntryBlock, I, CSI, TRI)) {
      for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
        // Add the callee-saved register as live-in.
        // It's killed at the spill.
        EntryBlock->addLiveIn(CSI[i].getReg());

        // Insert the spill to the stack frame.
        unsigned Reg = CSI[i].getReg();
        const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
        TII.storeRegToStackSlot(*EntryBlock, I, Reg, true,
                                CSI[i].getFrameIdx(), RC, TRI);
      }
    }

    // Restore using target interface.
    for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri) {
      MachineBasicBlock* MBB = ReturnBlocks[ri];
      I = MBB->end(); --I;

      // Skip over all terminator instructions, which are part of the return
      // sequence.
      MachineBasicBlock::iterator I2 = I;
      while (I2 != MBB->begin() && (--I2)->isTerminator())
        I = I2;

      bool AtStart = I == MBB->begin();
      MachineBasicBlock::iterator BeforeI = I;
      if (!AtStart)
        --BeforeI;

      // Restore all registers immediately before the return and any
      // terminators that precede it.
      if (!TFI->restoreCalleeSavedRegisters(*MBB, I, CSI, TRI)) {
        for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
          unsigned Reg = CSI[i].getReg();
          const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
          TII.loadRegFromStackSlot(*MBB, I, Reg,
                                   CSI[i].getFrameIdx(),
                                   RC, TRI);
          assert(I != MBB->begin() &&
                 "loadRegFromStackSlot didn't insert any code!");
          // Insert in reverse order.  loadRegFromStackSlot can insert
          // multiple instructions.
          if (AtStart)
            I = MBB->begin();
          else {
            I = BeforeI;
            ++I;
          }
        }
      }
    }
    return;
  }

  // Insert spills.
  std::vector<CalleeSavedInfo> blockCSI;
  for (CSRegBlockMap::iterator BI = CSRSave.begin(),
         BE = CSRSave.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet save = BI->second;

    if (save.empty())
      continue;

    blockCSI.clear();
    for (CSRegSet::iterator RI = save.begin(),
           RE = save.end(); RI != RE; ++RI) {
      blockCSI.push_back(CSI[*RI]);
    }
    assert(blockCSI.size() > 0 &&
           "Could not collect callee saved register info");

    I = MBB->begin();

    // When shrink wrapping, use stack slot stores/loads.
    for (unsigned i = 0, e = blockCSI.size(); i != e; ++i) {
      // Add the callee-saved register as live-in.
      // It's killed at the spill.
      MBB->addLiveIn(blockCSI[i].getReg());

      // Insert the spill to the stack frame.
      unsigned Reg = blockCSI[i].getReg();
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.storeRegToStackSlot(*MBB, I, Reg,
                              true,
                              blockCSI[i].getFrameIdx(),
                              RC, TRI);
    }
  }

  for (CSRegBlockMap::iterator BI = CSRRestore.begin(),
         BE = CSRRestore.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet restore = BI->second;

    if (restore.empty())
      continue;

    blockCSI.clear();
    for (CSRegSet::iterator RI = restore.begin(),
           RE = restore.end(); RI != RE; ++RI) {
      blockCSI.push_back(CSI[*RI]);
    }
    assert(blockCSI.size() > 0 &&
           "Could not find callee saved register info");

    // If MBB is empty and needs restores, insert at the _beginning_.
    if (MBB->empty()) {
      I = MBB->begin();
    } else {
      I = MBB->end();
      --I;

      // Skip over all terminator instructions, which are part of the
      // return sequence.
      if (! I->isTerminator()) {
        ++I;
      } else {
        MachineBasicBlock::iterator I2 = I;
        while (I2 != MBB->begin() && (--I2)->isTerminator())
          I = I2;
      }
    }

    bool AtStart = I == MBB->begin();
    MachineBasicBlock::iterator BeforeI = I;
    if (!AtStart)
      --BeforeI;

    // Restore all registers immediately before the return and any
    // terminators that precede it.
    for (unsigned i = 0, e = blockCSI.size(); i != e; ++i) {
      unsigned Reg = blockCSI[i].getReg();
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.loadRegFromStackSlot(*MBB, I, Reg,
                               blockCSI[i].getFrameIdx(),
                               RC, TRI);
      assert(I != MBB->begin() &&
             "loadRegFromStackSlot didn't insert any code!");
      // Insert in reverse order.  loadRegFromStackSlot can insert
      // multiple instructions.
      if (AtStart)
        I = MBB->begin();
      else {
        I = BeforeI;
        ++I;
      }
    }
  }
}

/// AdjustStackOffset - Helper function used to adjust the stack frame offset.
static inline void
AdjustStackOffset(MachineFrameInfo *MFI, int FrameIdx,
                  bool StackGrowsDown, int64_t &Offset,
                  unsigned &MaxAlign) {
  // If the stack grows down, add the object size to find the lowest address.
  if (StackGrowsDown)
    Offset += MFI->getObjectSize(FrameIdx);

  unsigned Align = MFI->getObjectAlignment(FrameIdx);

  // If the alignment of this object is greater than that of the stack, then
  // increase the stack alignment to match.
  MaxAlign = std::max(MaxAlign, Align);

  // Adjust to alignment boundary.
  Offset = (Offset + Align - 1) / Align * Align;

  if (StackGrowsDown) {
    DEBUG(dbgs() << "alloc FI(" << FrameIdx << ") at SP[" << -Offset << "]\n");
    MFI->setObjectOffset(FrameIdx, -Offset); // Set the computed offset
  } else {
    DEBUG(dbgs() << "alloc FI(" << FrameIdx << ") at SP[" << Offset << "]\n");
    MFI->setObjectOffset(FrameIdx, Offset);
    Offset += MFI->getObjectSize(FrameIdx);
  }
}

/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
///
void PEI::calculateFrameObjectOffsets(MachineFunction &Fn) {
  const TargetFrameLowering &TFI = *Fn.getTarget().getFrameLowering();

  bool StackGrowsDown =
    TFI.getStackGrowthDirection() == TargetFrameLowering::StackGrowsDown;

  // Loop over all of the stack objects, assigning sequential addresses...
  MachineFrameInfo *MFI = Fn.getFrameInfo();

  // Start at the beginning of the local area.
  // The Offset is the distance from the stack top in the direction
  // of stack growth -- so it's always nonnegative.
  int LocalAreaOffset = TFI.getOffsetOfLocalArea();
  if (StackGrowsDown)
    LocalAreaOffset = -LocalAreaOffset;
  assert(LocalAreaOffset >= 0
         && "Local area offset should be in direction of stack growth");
  int64_t Offset = LocalAreaOffset;

  // If there are fixed sized objects that are preallocated in the local area,
  // non-fixed objects can't be allocated right at the start of local area.
  // We currently don't support filling in holes in between fixed sized
  // objects, so we adjust 'Offset' to point to the end of last fixed sized
  // preallocated object.
  for (int i = MFI->getObjectIndexBegin(); i != 0; ++i) {
    int64_t FixedOff;
    if (StackGrowsDown) {
      // The maximum distance from the stack pointer is at lower address of
      // the object -- which is given by offset. For down growing stack
      // the offset is negative, so we negate the offset to get the distance.
      FixedOff = -MFI->getObjectOffset(i);
    } else {
      // The maximum distance from the start pointer is at the upper
      // address of the object.
      FixedOff = MFI->getObjectOffset(i) + MFI->getObjectSize(i);
    }
    if (FixedOff > Offset) Offset = FixedOff;
  }

  // First assign frame offsets to stack objects that are used to spill
  // callee saved registers.
  if (StackGrowsDown) {
    for (unsigned i = MinCSFrameIndex; i <= MaxCSFrameIndex; ++i) {
      // If the stack grows down, we need to add the size to find the lowest
      // address of the object.
      Offset += MFI->getObjectSize(i);

      unsigned Align = MFI->getObjectAlignment(i);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      MFI->setObjectOffset(i, -Offset);        // Set the computed offset
    }
  } else {
    int MaxCSFI = MaxCSFrameIndex, MinCSFI = MinCSFrameIndex;
    for (int i = MaxCSFI; i >= MinCSFI ; --i) {
      unsigned Align = MFI->getObjectAlignment(i);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      MFI->setObjectOffset(i, Offset);
      Offset += MFI->getObjectSize(i);
    }
  }

  unsigned MaxAlign = MFI->getMaxAlignment();

  // Make sure the special register scavenging spill slot is closest to the
  // frame pointer if a frame pointer is required.
  const TargetRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();
  if (RS && TFI.hasFP(Fn) && RegInfo->useFPForScavengingIndex(Fn) &&
      !RegInfo->needsStackRealignment(Fn)) {
    SmallVector<int, 2> SFIs;
    RS->getScavengingFrameIndices(SFIs);
    for (SmallVector<int, 2>::iterator I = SFIs.begin(),
         IE = SFIs.end(); I != IE; ++I)
      AdjustStackOffset(MFI, *I, StackGrowsDown, Offset, MaxAlign);
  }

  // FIXME: Once this is working, then enable flag will change to a target
  // check for whether the frame is large enough to want to use virtual
  // frame index registers. Functions which don't want/need this optimization
  // will continue to use the existing code path.
  if (MFI->getUseLocalStackAllocationBlock()) {
    unsigned Align = MFI->getLocalFrameMaxAlign();

    // Adjust to alignment boundary.
    Offset = (Offset + Align - 1) / Align * Align;

    DEBUG(dbgs() << "Local frame base offset: " << Offset << "\n");

    // Resolve offsets for objects in the local block.
    for (unsigned i = 0, e = MFI->getLocalFrameObjectCount(); i != e; ++i) {
      std::pair<int, int64_t> Entry = MFI->getLocalFrameObjectMap(i);
      int64_t FIOffset = (StackGrowsDown ? -Offset : Offset) + Entry.second;
      DEBUG(dbgs() << "alloc FI(" << Entry.first << ") at SP[" <<
            FIOffset << "]\n");
      MFI->setObjectOffset(Entry.first, FIOffset);
    }
    // Allocate the local block
    Offset += MFI->getLocalFrameSize();

    MaxAlign = std::max(Align, MaxAlign);
  }

  // Make sure that the stack protector comes before the local variables on the
  // stack.
  SmallSet<int, 16> LargeStackObjs;
  if (MFI->getStackProtectorIndex() >= 0) {
    AdjustStackOffset(MFI, MFI->getStackProtectorIndex(), StackGrowsDown,
                      Offset, MaxAlign);

    // Assign large stack objects first.
    for (unsigned i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
      if (MFI->isObjectPreAllocated(i) &&
          MFI->getUseLocalStackAllocationBlock())
        continue;
      if (i >= MinCSFrameIndex && i <= MaxCSFrameIndex)
        continue;
      if (RS && RS->isScavengingFrameIndex((int)i))
        continue;
      if (MFI->isDeadObjectIndex(i))
        continue;
      if (MFI->getStackProtectorIndex() == (int)i)
        continue;
      if (!MFI->MayNeedStackProtector(i))
        continue;

      AdjustStackOffset(MFI, i, StackGrowsDown, Offset, MaxAlign);
      LargeStackObjs.insert(i);
    }
  }

  // Then assign frame offsets to stack objects that are not used to spill
  // callee saved registers.
  for (unsigned i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
    if (MFI->isObjectPreAllocated(i) &&
        MFI->getUseLocalStackAllocationBlock())
      continue;
    if (i >= MinCSFrameIndex && i <= MaxCSFrameIndex)
      continue;
    if (RS && RS->isScavengingFrameIndex((int)i))
      continue;
    if (MFI->isDeadObjectIndex(i))
      continue;
    if (MFI->getStackProtectorIndex() == (int)i)
      continue;
    if (LargeStackObjs.count(i))
      continue;

    AdjustStackOffset(MFI, i, StackGrowsDown, Offset, MaxAlign);
  }

  // Make sure the special register scavenging spill slot is closest to the
  // stack pointer.
  if (RS && (!TFI.hasFP(Fn) || RegInfo->needsStackRealignment(Fn) ||
             !RegInfo->useFPForScavengingIndex(Fn))) {
    SmallVector<int, 2> SFIs;
    RS->getScavengingFrameIndices(SFIs);
    for (SmallVector<int, 2>::iterator I = SFIs.begin(),
         IE = SFIs.end(); I != IE; ++I)
      AdjustStackOffset(MFI, *I, StackGrowsDown, Offset, MaxAlign);
  }

  if (!TFI.targetHandlesStackFrameRounding()) {
    // If we have reserved argument space for call sites in the function
    // immediately on entry to the current function, count it as part of the
    // overall stack size.
    if (MFI->adjustsStack() && TFI.hasReservedCallFrame(Fn))
      Offset += MFI->getMaxCallFrameSize();

    // Round up the size to a multiple of the alignment.  If the function has
    // any calls or alloca's, align to the target's StackAlignment value to
    // ensure that the callee's frame or the alloca data is suitably aligned;
    // otherwise, for leaf functions, align to the TransientStackAlignment
    // value.
    unsigned StackAlign;
    if (MFI->adjustsStack() || MFI->hasVarSizedObjects() ||
        (RegInfo->needsStackRealignment(Fn) && MFI->getObjectIndexEnd() != 0))
      StackAlign = TFI.getStackAlignment();
    else
      StackAlign = TFI.getTransientStackAlignment();

    // If the frame pointer is eliminated, all frame offsets will be relative to
    // SP not FP. Align to MaxAlign so this works.
    StackAlign = std::max(StackAlign, MaxAlign);
    unsigned AlignMask = StackAlign - 1;
    Offset = (Offset + AlignMask) & ~uint64_t(AlignMask);
  }

  // Update frame info to pretend that this is part of the stack...
  int64_t StackSize = Offset - LocalAreaOffset;
  MFI->setStackSize(StackSize);
  NumBytesStackSpace += StackSize;
}

/// insertPrologEpilogCode - Scan the function for modified callee saved
/// registers, insert spill code for these callee saved registers, then add
/// prolog and epilog code to the function.
///
void PEI::insertPrologEpilogCode(MachineFunction &Fn) {
  const TargetFrameLowering &TFI = *Fn.getTarget().getFrameLowering();

  // Add prologue to the function...
  TFI.emitPrologue(Fn);

  // Add epilogue to restore the callee-save registers in each exiting block
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    // If last instruction is a return instruction, add an epilogue
    if (!I->empty() && I->back().isReturn())
      TFI.emitEpilogue(Fn, *I);
  }

  // Emit additional code that is required to support segmented stacks, if
  // we've been asked for it.  This, when linked with a runtime with support
  // for segmented stacks (libgcc is one), will result in allocating stack
  // space in small chunks instead of one large contiguous block.
  if (Fn.getTarget().Options.EnableSegmentedStacks)
    TFI.adjustForSegmentedStacks(Fn);

  // Emit additional code that is required to explicitly handle the stack in
  // HiPE native code (if needed) when loaded in the Erlang/OTP runtime. The
  // approach is rather similar to that of Segmented Stacks, but it uses a
  // different conditional check and another BIF for allocating more stack
  // space.
  if (Fn.getFunction()->getCallingConv() == CallingConv::HiPE)
    TFI.adjustForHiPEPrologue(Fn);
}

/// replaceFrameIndices - Replace all MO_FrameIndex operands with physical
/// register references and actual offsets.
///
void PEI::replaceFrameIndices(MachineFunction &Fn) {
  if (!Fn.getFrameInfo()->hasStackObjects()) return; // Nothing to do?

  const TargetMachine &TM = Fn.getTarget();
  assert(TM.getRegisterInfo() && "TM::getRegisterInfo() must be implemented!");
  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *TM.getRegisterInfo();
  const TargetFrameLowering *TFI = TM.getFrameLowering();
  bool StackGrowsDown =
    TFI->getStackGrowthDirection() == TargetFrameLowering::StackGrowsDown;
  int FrameSetupOpcode   = TII.getCallFrameSetupOpcode();
  int FrameDestroyOpcode = TII.getCallFrameDestroyOpcode();

  for (MachineFunction::iterator BB = Fn.begin(),
         E = Fn.end(); BB != E; ++BB) {
#ifndef NDEBUG
    int SPAdjCount = 0; // frame setup / destroy count.
#endif
    int SPAdj = 0;  // SP offset due to call frame setup / destroy.
    if (RS && !FrameIndexVirtualScavenging) RS->enterBasicBlock(BB);

    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {

      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
#ifndef NDEBUG
        // Track whether we see even pairs of them
        SPAdjCount += I->getOpcode() == FrameSetupOpcode ? 1 : -1;
#endif
        // Remember how much SP has been adjusted to create the call
        // frame.
        int Size = I->getOperand(0).getImm();

        if ((!StackGrowsDown && I->getOpcode() == FrameSetupOpcode) ||
            (StackGrowsDown && I->getOpcode() == FrameDestroyOpcode))
          Size = -Size;

        SPAdj += Size;

        MachineBasicBlock::iterator PrevI = BB->end();
        if (I != BB->begin()) PrevI = prior(I);
        TFI->eliminateCallFramePseudoInstr(Fn, *BB, I);

        // Visit the instructions created by eliminateCallFramePseudoInstr().
        if (PrevI == BB->end())
          I = BB->begin();     // The replaced instr was the first in the block.
        else
          I = llvm::next(PrevI);
        continue;
      }

      MachineInstr *MI = I;
      bool DoIncr = true;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        if (!MI->getOperand(i).isFI())
            continue;

        // Some instructions (e.g. inline asm instructions) can have
        // multiple frame indices and/or cause eliminateFrameIndex
        // to insert more than one instruction. We need the register
        // scavenger to go through all of these instructions so that
        // it can update its register information. We keep the
        // iterator at the point before insertion so that we can
        // revisit them in full.
        bool AtBeginning = (I == BB->begin());
        if (!AtBeginning) --I;

        // If this instruction has a FrameIndex operand, we need to
        // use that target machine register info object to eliminate
        // it.
        TRI.eliminateFrameIndex(MI, SPAdj, i,
                                FrameIndexVirtualScavenging ?  NULL : RS);

        // Reset the iterator if we were at the beginning of the BB.
        if (AtBeginning) {
          I = BB->begin();
          DoIncr = false;
        }

        MI = 0;
        break;
      }

      if (DoIncr && I != BB->end()) ++I;

      // Update register states.
      if (RS && !FrameIndexVirtualScavenging && MI) RS->forward(MI);
    }

    // If we have evenly matched pairs of frame setup / destroy instructions,
    // make sure the adjustments come out to zero. If we don't have matched
    // pairs, we can't be sure the missing bit isn't in another basic block
    // due to a custom inserter playing tricks, so just asserting SPAdj==0
    // isn't sufficient. See tMOVCC on Thumb1, for example.
    assert((SPAdjCount || SPAdj == 0) &&
           "Unbalanced call frame setup / destroy pairs?");
  }
}

/// scavengeFrameVirtualRegs - Replace all frame index virtual registers
/// with physical registers. Use the register scavenger to find an
/// appropriate register to use.
///
/// FIXME: Iterating over the instruction stream is unnecessary. We can simply
/// iterate over the vreg use list, which at this point only contains machine
/// operands for which eliminateFrameIndex need a new scratch reg.
void PEI::scavengeFrameVirtualRegs(MachineFunction &Fn) {
  // Run through the instructions and find any virtual registers.
  for (MachineFunction::iterator BB = Fn.begin(),
       E = Fn.end(); BB != E; ++BB) {
    RS->enterBasicBlock(BB);

    int SPAdj = 0;

    // The instruction stream may change in the loop, so check BB->end()
    // directly.
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {
      // We might end up here again with a NULL iterator if we scavenged a
      // register for which we inserted spill code for definition by what was
      // originally the first instruction in BB.
      if (I == MachineBasicBlock::iterator(NULL))
        I = BB->begin();

      MachineInstr *MI = I;
      MachineBasicBlock::iterator J = llvm::next(I);
      MachineBasicBlock::iterator P = I == BB->begin() ?
        MachineBasicBlock::iterator(NULL) : llvm::prior(I);

      // RS should process this instruction before we might scavenge at this
      // location. This is because we might be replacing a virtual register
      // defined by this instruction, and if so, registers killed by this
      // instruction are available, and defined registers are not.
      RS->forward(I);

      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        if (MI->getOperand(i).isReg()) {
          MachineOperand &MO = MI->getOperand(i);
          unsigned Reg = MO.getReg();
          if (Reg == 0)
            continue;
          if (!TargetRegisterInfo::isVirtualRegister(Reg))
            continue;

          // When we first encounter a new virtual register, it
          // must be a definition.
          assert(MI->getOperand(i).isDef() &&
                 "frame index virtual missing def!");
          // Scavenge a new scratch register
          const TargetRegisterClass *RC = Fn.getRegInfo().getRegClass(Reg);
          unsigned ScratchReg = RS->scavengeRegister(RC, J, SPAdj);

          ++NumScavengedRegs;

          // Replace this reference to the virtual register with the
          // scratch register.
          assert (ScratchReg && "Missing scratch register!");
          Fn.getRegInfo().replaceRegWith(Reg, ScratchReg);

          // Because this instruction was processed by the RS before this
          // register was allocated, make sure that the RS now records the
          // register as being used.
          RS->setUsed(ScratchReg);
        }
      }

      // If the scavenger needed to use one of its spill slots, the
      // spill code will have been inserted in between I and J. This is a
      // problem because we need the spill code before I: Move I to just
      // prior to J.
      if (I != llvm::prior(J)) {
        BB->splice(J, BB, I);

        // Before we move I, we need to prepare the RS to visit I again.
        // Specifically, RS will assert if it sees uses of registers that
        // it believes are undefined. Because we have already processed
        // register kills in I, when it visits I again, it will believe that
        // those registers are undefined. To avoid this situation, unprocess
        // the instruction I.
        assert(RS->getCurrentPosition() == I &&
          "The register scavenger has an unexpected position");
        I = P;
        RS->unprocess(P);
      } else
        ++I;
    }
  }
}
