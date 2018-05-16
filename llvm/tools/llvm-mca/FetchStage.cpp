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
#include "Instruction.h"

using namespace mca;

bool FetchStage::isReady() const { return SM.hasNext(); }

bool FetchStage::execute(InstRef &IR) {
  if (!SM.hasNext())
    return false;
  const SourceRef SR = SM.peekNext();
  std::unique_ptr<Instruction> I = IB.createInstruction(*SR.second);
  IR = InstRef(SR.first, I.get());
  Instructions[IR.getSourceIndex()] = std::move(I);
  return true;
}

void FetchStage::postExecute(const InstRef &IR) {
  // Reclaim instructions that have been retired.
  llvm::remove_if(Instructions,
                  [](InstMapPr &Pr) { return Pr.getSecond()->isRetired(); });
  SM.updateNext();
}
