//===---------------------- FetchStage.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the Fetch stage of an instruction pipeline.  Its sole
/// purpose in life is to produce instructions for the rest of the pipeline.
///
//===----------------------------------------------------------------------===//

#include "Stages/FetchStage.h"

namespace mca {

bool FetchStage::hasWorkToComplete() const {
  return CurrentInstruction.get() || SM.hasNext();
}

bool FetchStage::isAvailable(const InstRef & /* unused */) const {
  if (!CurrentInstruction)
    return false;
  assert(SM.hasNext() && "Unexpected internal state!");
  const SourceRef SR = SM.peekNext();
  InstRef IR(SR.first, CurrentInstruction.get());
  return checkNextStage(IR);
}

llvm::Error FetchStage::getNextInstruction() {
  assert(!CurrentInstruction && "There is already an instruction to process!");
  if (!SM.hasNext())
    return llvm::ErrorSuccess();
  const SourceRef SR = SM.peekNext();
  llvm::Expected<std::unique_ptr<Instruction>> InstOrErr =
      IB.createInstruction(*SR.second);
  if (!InstOrErr)
    return InstOrErr.takeError();
  CurrentInstruction = std::move(InstOrErr.get());
  return llvm::ErrorSuccess();
}

llvm::Error FetchStage::execute(InstRef & /*unused */) {
  assert(CurrentInstruction && "There is no instruction to process!");
  const SourceRef SR = SM.peekNext();
  InstRef IR(SR.first, CurrentInstruction.get());
  assert(checkNextStage(IR) && "Invalid fetch!");

  Instructions[IR.getSourceIndex()] = std::move(CurrentInstruction);
  if (llvm::Error Val = moveToTheNextStage(IR))
    return Val;

  SM.updateNext();

  // Move the program counter.
  return getNextInstruction();
}

llvm::Error FetchStage::cycleStart() {
  if (!CurrentInstruction && SM.hasNext())
    return getNextInstruction();
  return llvm::ErrorSuccess();
}

llvm::Error FetchStage::cycleEnd() {
  // Find the first instruction which hasn't been retired.
  const InstMap::iterator It =
      llvm::find_if(Instructions, [](const InstMap::value_type &KeyValuePair) {
        return !KeyValuePair.second->isRetired();
      });

  // Erase instructions up to the first that hasn't been retired.
  if (It != Instructions.begin())
    Instructions.erase(Instructions.begin(), It);

  return llvm::ErrorSuccess();
}

} // namespace mca
