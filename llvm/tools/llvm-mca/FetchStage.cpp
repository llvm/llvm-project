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

#include "FetchStage.h"

namespace mca {

bool FetchStage::hasWorkToComplete() const { return SM.hasNext(); }

Stage::Status FetchStage::execute(InstRef &IR) {
  if (!SM.hasNext())
    return Stage::Stop;
  const SourceRef SR = SM.peekNext();
  llvm::Expected<std::unique_ptr<Instruction>> InstOrErr =
      IB.createInstruction(*SR.second);
  if (!InstOrErr)
    return InstOrErr.takeError();
  std::unique_ptr<Instruction> I = std::move(*InstOrErr);
  IR = InstRef(SR.first, I.get());
  Instructions[IR.getSourceIndex()] = std::move(I);
  return Stage::Continue;
}

void FetchStage::postExecute() { SM.updateNext(); }

void FetchStage::cycleEnd() {
  // Find the first instruction which hasn't been retired.
  const InstMap::iterator It =
      llvm::find_if(Instructions, [](const InstMap::value_type &KeyValuePair) {
        return !KeyValuePair.second->isRetired();
      });

  // Erase instructions up to the first that hasn't been retired.
  if (It != Instructions.begin())
    Instructions.erase(Instructions.begin(), It);
}

} // namespace mca
