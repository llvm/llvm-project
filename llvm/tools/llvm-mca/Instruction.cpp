//===--------------------- Instruction.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines abstractions used by the Backend to model register reads,
// register writes and instructions.
//
//===----------------------------------------------------------------------===//

#include "Instruction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mca {

using namespace llvm;

void ReadState::writeStartEvent(unsigned Cycles) {
  assert(DependentWrites);
  assert(CyclesLeft == UNKNOWN_CYCLES);

  // This read may be dependent on more than one write. This typically occurs
  // when a definition is the result of multiple writes where at least one
  // write does a partial register update.
  // The HW is forced to do some extra bookkeeping to track of all the
  // dependent writes, and implement a merging scheme for the partial writes.
  --DependentWrites;
  TotalCycles = std::max(TotalCycles, Cycles);

  if (!DependentWrites)
    CyclesLeft = TotalCycles;
}

void WriteState::onInstructionIssued() {
  assert(CyclesLeft == UNKNOWN_CYCLES);
  // Update the number of cycles left based on the WriteDescriptor info.
  CyclesLeft = WD.Latency;

  // Now that the time left before write-back is know, notify
  // all the users.
  for (const std::pair<ReadState *, int> &User : Users) {
    ReadState *RS = User.first;
    unsigned ReadCycles = std::max(0, CyclesLeft - User.second);
    RS->writeStartEvent(ReadCycles);
  }
}

void WriteState::addUser(ReadState *User, int ReadAdvance) {
  // If CyclesLeft is different than -1, then we don't need to
  // update the list of users. We can just notify the user with
  // the actual number of cycles left (which may be zero).
  if (CyclesLeft != UNKNOWN_CYCLES) {
    unsigned ReadCycles = std::max(0, CyclesLeft - ReadAdvance);
    User->writeStartEvent(ReadCycles);
    return;
  }

  std::pair<ReadState *, int> NewPair(User, ReadAdvance);
  Users.insert(NewPair);
}

void WriteState::cycleEvent() {
  // Note: CyclesLeft can be a negative number. It is an error to
  // make it an unsigned quantity because users of this write may
  // specify a negative ReadAdvance.
  if (CyclesLeft != UNKNOWN_CYCLES)
    CyclesLeft--;
}

void ReadState::cycleEvent() {
  // If CyclesLeft is unknown, then bail out immediately.
  if (CyclesLeft == UNKNOWN_CYCLES)
    return;

  // If there are still dependent writes, or we reached cycle zero,
  // then just exit.
  if (DependentWrites || CyclesLeft == 0)
    return;

  CyclesLeft--;
}

#ifndef NDEBUG
void WriteState::dump() const {
  dbgs() << "{ OpIdx=" << WD.OpIndex << ", Lat=" << WD.Latency << ", RegID "
         << getRegisterID() << ", Cycles Left=" << getCyclesLeft() << " }\n";
}
#endif

bool Instruction::isReady() {
  if (Stage == IS_READY)
    return true;

  assert(Stage == IS_AVAILABLE);
  for (const UniqueUse &Use : Uses)
    if (!Use.get()->isReady())
      return false;

  setReady();
  return true;
}

void Instruction::execute() {
  assert(Stage == IS_READY);
  Stage = IS_EXECUTING;
  for (UniqueDef &Def : Defs)
    Def->onInstructionIssued();
}

bool Instruction::isZeroLatency() const {
  return Desc.MaxLatency == 0 && Defs.size() == 0 && Uses.size() == 0;
}

void Instruction::cycleEvent() {
  if (isDispatched()) {
    for (UniqueUse &Use : Uses)
      Use->cycleEvent();
    return;
  }
  if (isExecuting()) {
    for (UniqueDef &Def : Defs)
      Def->cycleEvent();
    CyclesLeft--;
  }
  if (!CyclesLeft)
    Stage = IS_EXECUTED;
}

} // namespace mca
