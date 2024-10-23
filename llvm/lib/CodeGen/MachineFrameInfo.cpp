//===-- MachineFrameInfo.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Implements MachineFrameInfo that manages the stack frame.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFrameInfo.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>

#define DEBUG_TYPE "codegen"

using namespace llvm;

void MachineFrameInfo::ensureMaxAlignment(Align Alignment) {
  if (!StackRealignable)
    assert(Alignment <= StackAlignment &&
           "For targets without stack realignment, Alignment is out of limit!");
  if (MaxAlignment < Alignment)
    MaxAlignment = Alignment;
}

/// Clamp the alignment if requested and emit a warning.
static inline Align clampStackAlignment(bool ShouldClamp, Align Alignment,
                                        Align StackAlignment) {
  if (!ShouldClamp || Alignment <= StackAlignment)
    return Alignment;
  LLVM_DEBUG(dbgs() << "Warning: requested alignment " << DebugStr(Alignment)
                    << " exceeds the stack alignment "
                    << DebugStr(StackAlignment)
                    << " when stack realignment is off" << '\n');
  return StackAlignment;
}

int MachineFrameInfo::CreateStackObject(uint64_t Size, Align Alignment,
                                        bool IsSpillSlot,
                                        const AllocaInst *Alloca,
                                        uint8_t StackID) {
  assert(Size != 0 && "Cannot allocate zero size stack objects!");
  Alignment = clampStackAlignment(!StackRealignable, Alignment, StackAlignment);
  Objects.push_back(StackObject(Size, Alignment, 0, false, IsSpillSlot, Alloca,
                                !IsSpillSlot, StackID));
  int Index = (int)Objects.size() - NumFixedObjects - 1;
  assert(Index >= 0 && "Bad frame index!");
  if (contributesToMaxAlignment(StackID))
    ensureMaxAlignment(Alignment);
  return Index;
}

int MachineFrameInfo::CreateSpillStackObject(uint64_t Size, Align Alignment) {
  Alignment = clampStackAlignment(!StackRealignable, Alignment, StackAlignment);
  CreateStackObject(Size, Alignment, true);
  int Index = (int)Objects.size() - NumFixedObjects - 1;
  ensureMaxAlignment(Alignment);
  return Index;
}

int MachineFrameInfo::CreateVariableSizedObject(Align Alignment,
                                                const AllocaInst *Alloca) {
  HasVarSizedObjects = true;
  Alignment = clampStackAlignment(!StackRealignable, Alignment, StackAlignment);
  Objects.push_back(StackObject(0, Alignment, 0, false, false, Alloca, true));
  ensureMaxAlignment(Alignment);
  return (int)Objects.size()-NumFixedObjects-1;
}

int MachineFrameInfo::CreateFixedObject(uint64_t Size, int64_t SPOffset,
                                        bool IsImmutable, bool IsAliased) {
  assert(Size != 0 && "Cannot allocate zero size fixed stack objects!");
  // The alignment of the frame index can be determined from its offset from
  // the incoming frame position.  If the frame object is at offset 32 and
  // the stack is guaranteed to be 16-byte aligned, then we know that the
  // object is 16-byte aligned. Note that unlike the non-fixed case, if the
  // stack needs realignment, we can't assume that the stack will in fact be
  // aligned.
  Align Alignment =
      commonAlignment(ForcedRealign ? Align(1) : StackAlignment, SPOffset);
  Alignment = clampStackAlignment(!StackRealignable, Alignment, StackAlignment);
  Objects.insert(Objects.begin(),
                 StackObject(Size, Alignment, SPOffset, IsImmutable,
                             /*IsSpillSlot=*/false, /*Alloca=*/nullptr,
                             IsAliased));
  return -++NumFixedObjects;
}

int MachineFrameInfo::CreateFixedSpillStackObject(uint64_t Size,
                                                  int64_t SPOffset,
                                                  bool IsImmutable) {
  Align Alignment =
      commonAlignment(ForcedRealign ? Align(1) : StackAlignment, SPOffset);
  Alignment = clampStackAlignment(!StackRealignable, Alignment, StackAlignment);
  Objects.insert(Objects.begin(),
                 StackObject(Size, Alignment, SPOffset, IsImmutable,
                             /*IsSpillSlot=*/true, /*Alloca=*/nullptr,
                             /*IsAliased=*/false));
  return -++NumFixedObjects;
}

BitVector MachineFrameInfo::getPristineRegs(const MachineFunction &MF) const {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  BitVector BV(TRI->getNumRegs());

  // Before CSI is calculated, no registers are considered pristine. They can be
  // freely used and PEI will make sure they are saved.
  if (!isCalleeSavedInfoValid())
    return BV;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (const MCPhysReg *CSR = MRI.getCalleeSavedRegs(); CSR && *CSR;
       ++CSR)
    BV.set(*CSR);

  // Saved CSRs are not pristine.
  for (const auto &I : getCalleeSavedInfo())
    for (MCPhysReg S : TRI->subregs_inclusive(I.getReg()))
      BV.reset(S);

  return BV;
}

uint64_t MachineFrameInfo::estimateStackSize(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  Align MaxAlign = getMaxAlign();
  int64_t Offset = 0;

  // This code is very, very similar to PEI::calculateFrameObjectOffsets().
  // It really should be refactored to share code. Until then, changes
  // should keep in mind that there's tight coupling between the two.

  for (int i = getObjectIndexBegin(); i != 0; ++i) {
    // Only estimate stack size of default stack.
    if (getStackID(i) != TargetStackID::Default)
      continue;
    int64_t FixedOff = -getObjectOffset(i);
    if (FixedOff > Offset) Offset = FixedOff;
  }
  for (unsigned i = 0, e = getObjectIndexEnd(); i != e; ++i) {
    // Only estimate stack size of live objects on default stack.
    if (isDeadObjectIndex(i) || getStackID(i) != TargetStackID::Default)
      continue;
    Offset += getObjectSize(i);
    Align Alignment = getObjectAlign(i);
    // Adjust to alignment boundary
    Offset = alignTo(Offset, Alignment);

    MaxAlign = std::max(Alignment, MaxAlign);
  }

  if (adjustsStack() && TFI->hasReservedCallFrame(MF))
    Offset += getMaxCallFrameSize();

  // Round up the size to a multiple of the alignment.  If the function has
  // any calls or alloca's, align to the target's StackAlignment value to
  // ensure that the callee's frame or the alloca data is suitably aligned;
  // otherwise, for leaf functions, align to the TransientStackAlignment
  // value.
  Align StackAlign;
  if (adjustsStack() || hasVarSizedObjects() ||
      (RegInfo->hasStackRealignment(MF) && getObjectIndexEnd() != 0))
    StackAlign = TFI->getStackAlign();
  else
    StackAlign = TFI->getTransientStackAlign();

  // If the frame pointer is eliminated, all frame offsets will be relative to
  // SP not FP. Align to MaxAlign so this works.
  StackAlign = std::max(StackAlign, MaxAlign);
  return alignTo(Offset, StackAlign);
}

void MachineFrameInfo::computeMaxCallFrameSize(
    MachineFunction &MF, std::vector<MachineBasicBlock::iterator> *FrameSDOps) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  unsigned FrameSetupOpcode = TII.getCallFrameSetupOpcode();
  unsigned FrameDestroyOpcode = TII.getCallFrameDestroyOpcode();
  assert(FrameSetupOpcode != ~0u && FrameDestroyOpcode != ~0u &&
         "Can only compute MaxCallFrameSize if Setup/Destroy opcode are known");

  MaxCallFrameSize = 0;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      if (Opcode == FrameSetupOpcode || Opcode == FrameDestroyOpcode) {
        uint64_t Size = TII.getFrameSize(MI);
        MaxCallFrameSize = std::max(MaxCallFrameSize, Size);
        if (FrameSDOps != nullptr)
          FrameSDOps->push_back(&MI);
      }
    }
  }
}

void MachineFrameInfo::print(const MachineFunction &MF, raw_ostream &OS) const{
  if (Objects.empty()) return;

  const TargetFrameLowering *FI = MF.getSubtarget().getFrameLowering();
  int ValOffset = (FI ? FI->getOffsetOfLocalArea() : 0);

  OS << "Frame Objects:\n";

  for (unsigned i = 0, e = Objects.size(); i != e; ++i) {
    const StackObject &SO = Objects[i];
    OS << "  fi#" << (int)(i-NumFixedObjects) << ": ";

    if (SO.StackID != 0)
      OS << "id=" << static_cast<unsigned>(SO.StackID) << ' ';

    if (SO.Size == ~0ULL) {
      OS << "dead\n";
      continue;
    }
    if (SO.Size == 0)
      OS << "variable sized";
    else
      OS << "size=" << SO.Size;
    OS << ", align=" << SO.Alignment.value();

    if (i < NumFixedObjects)
      OS << ", fixed";
    if (i < NumFixedObjects || SO.SPOffset != -1) {
      int64_t Off = SO.SPOffset - ValOffset;
      OS << ", at location [SP";
      if (Off > 0)
        OS << "+" << Off;
      else if (Off < 0)
        OS << Off;
      OS << "]";
    }
    OS << "\n";
  }
}

std::optional<unsigned>
MachineFrameSizeInfo::getCallFrameSizeAt(MachineInstr &MI) {
  return this->getCallFrameSizeAt(*MI.getParent(), MI.getIterator());
}

std::optional<unsigned>
MachineFrameSizeInfo::getCallFrameSizeAtBegin(MachineBasicBlock &MBB) {
  if (!IsComputed)
    computeSizes();
  if (HasNoBrokenUpCallSeqs || !HasFrameOpcodes)
    return std::nullopt;
  return State[MBB.getNumber()].Entry;
}

std::optional<unsigned>
MachineFrameSizeInfo::getCallFrameSizeAtEnd(MachineBasicBlock &MBB) {
  if (!IsComputed)
    computeSizes();
  if (HasNoBrokenUpCallSeqs || !HasFrameOpcodes)
    return std::nullopt;
  return State[MBB.getNumber()].Exit;
}

std::optional<unsigned>
MachineFrameSizeInfo::getCallFrameSizeAt(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MII) {
  if (!IsComputed)
    computeSizes();

  if (!HasFrameOpcodes)
    return std::nullopt;

  if (MII == MBB.end()) {
    if (HasNoBrokenUpCallSeqs)
      return std::nullopt;
    return State[MBB.getNumber()].Exit;
  }

  if (MII == MBB.begin()) {
    if (HasNoBrokenUpCallSeqs)
      return std::nullopt;
    return State[MBB.getNumber()].Entry;
  }

  // Search backwards from MI for the most recent call frame instruction.
  for (auto &AdjI : reverse(make_range(MBB.begin(), MII))) {
    if (AdjI.getOpcode() == FrameSetupOpcode)
      return TII->getFrameTotalSize(AdjI);
    if (AdjI.getOpcode() == FrameDestroyOpcode)
      return std::nullopt;
  }

  // If none was found, use the call frame size from the start of the basic
  // block.
  if (HasNoBrokenUpCallSeqs)
    return std::nullopt;
  return State[MBB.getNumber()].Entry;
}

void MachineFrameSizeInfo::computeSizes() {
  if (!IsComputed) {
    // Populate fields that are only required once we compute the frame sizes.
    TII = MF.getSubtarget().getInstrInfo();
    FrameSetupOpcode = TII->getCallFrameSetupOpcode();
    FrameDestroyOpcode = TII->getCallFrameDestroyOpcode();
    HasFrameOpcodes = FrameSetupOpcode != ~0u || FrameDestroyOpcode != ~0u;
    assert(!HasFrameOpcodes || FrameSetupOpcode != FrameDestroyOpcode);
    IsComputed = true;
  }
  // If the target has no call frame pseudo instructions, don't compute
  // anything, we always return std::nullopt if queried.
  if (!HasFrameOpcodes)
    return;

  // Returns true if a call sequence in MF is broken up over multiple blocks.
  auto FindBrokenUpCallSeq = [](const MachineFunction &MF,
                                unsigned FrameSetupOpcode,
                                unsigned FrameDestroyOpcode) {
    for (const auto &MBB : MF) {
      for (const auto &I : MBB) {
        unsigned Opcode = I.getOpcode();
        if (Opcode == FrameSetupOpcode)
          break;
        if (Opcode == FrameDestroyOpcode) {
          // A FrameDestroy without a preceeding FrameSetup in the MBB. If
          // FrameInstructions are placed correctly (which we assume), this
          // occurs if and only if a call sequence is broken into multiple
          // blocks.
          return true;
        }
      }
    }
    return false;
  };

  HasNoBrokenUpCallSeqs =
      !FindBrokenUpCallSeq(MF, FrameSetupOpcode, FrameDestroyOpcode);

  // If every call sequence is limited to a single basic block, the frame sizes
  // at entry and exit of each basic block need to be std::nullopt, so there is
  // nothing to compute.
  if (HasNoBrokenUpCallSeqs)
    return;

  State.resize(MF.getNumBlockIDs());

  df_iterator_default_set<const MachineBasicBlock *> Reachable;

  // Visit the MBBs in DFS order.
  for (df_ext_iterator<MachineFunction *,
                       df_iterator_default_set<const MachineBasicBlock *>>
           DFI = df_ext_begin(&MF, Reachable),
           DFE = df_ext_end(&MF, Reachable);
       DFI != DFE; ++DFI) {
    const MachineBasicBlock *MBB = *DFI;

    MachineFrameSizeInfoForBB BBState;

    // Use the exit state of the DFS stack predecessor as entry state for this
    // block. With correctly placed call frame instructions, all other
    // predecessors must have the same call frame size at exit.
    if (DFI.getPathLength() >= 2) {
      const MachineBasicBlock *StackPred = DFI.getPath(DFI.getPathLength() - 2);
      assert(Reachable.count(StackPred) &&
             "DFS stack predecessor is already visited.\n");
      BBState.Entry = State[StackPred->getNumber()].Exit;
      BBState.Exit = BBState.Entry;
    }

    // Search backwards for the last call frame instruction and use its implied
    // state for the block exit. Otherwise, the exit state remains equal to the
    // entry state.
    for (auto &AdjI : reverse(make_range(MBB->begin(), MBB->end()))) {
      if (AdjI.getOpcode() == FrameSetupOpcode) {
        BBState.Exit = TII->getFrameTotalSize(AdjI);
        break;
      }
      if (AdjI.getOpcode() == FrameDestroyOpcode) {
        BBState.Exit = std::nullopt;
        break;
      }
    }
    State[MBB->getNumber()] = BBState;
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MachineFrameInfo::dump(const MachineFunction &MF) const {
  print(MF, dbgs());
}
#endif
