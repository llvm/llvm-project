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
#include "Instruction.h"
#include "SourceMgr.h"
#include "Stage.h"
#include "llvm/ADT/DenseMap.h"

namespace mca {

class FetchStage : public Stage {
  using InstMap = llvm::DenseMap<unsigned, std::unique_ptr<Instruction>>;
  using InstMapPr =
      llvm::detail::DenseMapPair<unsigned, std::unique_ptr<Instruction>>;
  InstMap Instructions;
  InstrBuilder &IB;
  SourceMgr &SM;

public:
  FetchStage(InstrBuilder &IB, SourceMgr &SM) : IB(IB), SM(SM) {}
  FetchStage(const FetchStage &Other) = delete;
  FetchStage &operator=(const FetchStage &Other) = delete;

  bool isReady() override final;
  bool execute(InstRef &IR) override final;
  void postExecute(const InstRef &IR) override final;
};

} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_FETCH_STAGE_H
