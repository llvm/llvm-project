//===---------------------- FetchStage.h ------------------------*- C++ -*-===//
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

#ifndef LLVM_TOOLS_LLVM_MCA_FETCH_STAGE_H
#define LLVM_TOOLS_LLVM_MCA_FETCH_STAGE_H

#include "InstrBuilder.h"
#include "SourceMgr.h"
#include "Stages/Stage.h"
#include <map>

namespace mca {

class FetchStage final : public Stage {
  std::unique_ptr<Instruction> CurrentInstruction;
  using InstMap = std::map<unsigned, std::unique_ptr<Instruction>>;
  InstMap Instructions;
  InstrBuilder &IB;
  SourceMgr &SM;

  // Updates the program counter, and sets 'CurrentInstruction'.
  llvm::Error getNextInstruction();

  FetchStage(const FetchStage &Other) = delete;
  FetchStage &operator=(const FetchStage &Other) = delete;

public:
  FetchStage(InstrBuilder &IB, SourceMgr &SM)
      : CurrentInstruction(), IB(IB), SM(SM) {}

  bool isAvailable(const InstRef &IR) const override;
  bool hasWorkToComplete() const override;
  llvm::Error execute(InstRef &IR) override;
  llvm::Error cycleStart() override;
  llvm::Error cycleEnd() override;
};

} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_FETCH_STAGE_H
