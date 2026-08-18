//===---------------------- MicroOpQueueStage.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the MicroOpQueueStage.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/Stages/MicroOpQueueStage.h"

namespace llvm {
namespace mca {

#define DEBUG_TYPE "llvm-mca"

Error MicroOpQueueStage::moveInstructions() {
  InstRef FirstIR = Buffer[CurrentInstructionSlotIdx];
  while (FirstIR && checkNextStage(FirstIR)) {
    Buffer[CurrentInstructionSlotIdx].invalidate();
    unsigned NormalizedOpcodes = getNormalizedOpcodes(FirstIR);
    CurrentInstructionSlotIdx += NormalizedOpcodes;
    CurrentInstructionSlotIdx %= Buffer.size();
    AvailableEntries += NormalizedOpcodes;

    InstRef SecondIR = Buffer[CurrentInstructionSlotIdx];
    bool Fused = false;
    if (SecondIR && tryFuseInstructions(*FirstIR.getInstruction(),
                                        *SecondIR.getInstruction())) {
      unsigned NormalizedOpcodes = getNormalizedOpcodes(FirstIR);
      CurrentInstructionSlotIdx += NormalizedOpcodes;
      CurrentInstructionSlotIdx %= Buffer.size();
      AvailableEntries += NormalizedOpcodes;
      Fused = true;
    }

    if (llvm::Error Val = moveToTheNextStage(FirstIR))
      return Val;

    if (Fused) {
      // When fusion happens, the second instruction is eliminated,
      // but it should still go through the pipeline to be reflected
      // in the timeline.
      if (llvm::Error Val = moveToTheNextStage(SecondIR))
        return Val;
    }

    FirstIR = Buffer[CurrentInstructionSlotIdx];
  }

  return llvm::ErrorSuccess();
}

MicroOpQueueStage::MicroOpQueueStage(
    const MCSubtargetInfo &STI, unsigned Size, unsigned IPC,
    bool ZeroLatencyStage, ArrayRef<MacroFusionPredicate> FusionPredicates)
    : STI(STI), NextAvailableSlotIdx(0), CurrentInstructionSlotIdx(0),
      MaxIPC(IPC), CurrentIPC(0), IsZeroLatencyStage(ZeroLatencyStage),
      FusionPredicates(FusionPredicates.begin(), FusionPredicates.end()) {
  Buffer.resize(Size ? Size : 1);
  AvailableEntries = Buffer.size();
}

Error MicroOpQueueStage::execute(InstRef &IR) {
  Buffer[NextAvailableSlotIdx] = IR;
  unsigned NormalizedOpcodes = getNormalizedOpcodes(IR);
  NextAvailableSlotIdx += NormalizedOpcodes;
  NextAvailableSlotIdx %= Buffer.size();
  AvailableEntries -= NormalizedOpcodes;
  ++CurrentIPC;
  return llvm::ErrorSuccess();
}

Error MicroOpQueueStage::cycleStart() {
  CurrentIPC = 0;
  if (!IsZeroLatencyStage)
    return moveInstructions();
  return llvm::ErrorSuccess();
}

Error MicroOpQueueStage::cycleEnd() {
  if (IsZeroLatencyStage)
    return moveInstructions();
  return llvm::ErrorSuccess();
}

// Move all writes from Second to First instruction.
static void moveWrites(Instruction &First, Instruction &Second) {
  for (WriteState &SecondDef : Second.getDefs()) {
    bool AlreadyDefined = false;
    for (WriteState &FirstDef : First.getDefs()) {
      if (FirstDef.getRegisterID() == SecondDef.getRegisterID()) {
        AlreadyDefined = true;
        break;
      }
    }

    if (!AlreadyDefined)
      First.getDefs().push_back(SecondDef);

    // Second instruction is going to be eliminated, so it cannot have
    // any active writes.
    SecondDef.setRegisterID(0);
  }
}

// Move all reads from Second to First instruction.
static void moveReads(Instruction &First, const Instruction &Second) {
  for (const ReadState &SecondUse : Second.getUses()) {
    bool AlreadyUsed = false;
    for (ReadState &FirstUse : First.getUses()) {
      if (SecondUse.getRegisterID() == FirstUse.getRegisterID()) {
        AlreadyUsed = true;
        break;
      }
    }

    // Ignore reads of registers which are defined by the instruction
    // we move them to.
    bool IsDef = false;
    for (WriteState &FirstDef : First.getDefs()) {
      if (SecondUse.getRegisterID() == FirstDef.getRegisterID()) {
        IsDef = true;
        break;
      }
    }

    if (!AlreadyUsed && !IsDef)
      First.getUses().push_back(SecondUse);
  }
}

bool MicroOpQueueStage::tryFuseInstructions(Instruction &First,
                                            Instruction &Second) {
  for (MacroFusionPredicate Predicate : FusionPredicates) {
    if (Predicate(STI, First, Second)) {
      moveReads(First, Second);
      moveWrites(First, Second);
      Second.setEliminated();
      return true;
    }
  }
  return false;
}

} // namespace mca
} // namespace llvm
