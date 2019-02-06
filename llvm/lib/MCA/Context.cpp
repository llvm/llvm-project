//===---------------------------- Context.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a class for holding ownership of various simulated
/// hardware units.  A Context also provides a utility routine for constructing
/// a default out-of-order pipeline with fetch, dispatch, execute, and retire
/// stages.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/Context.h"
#include "llvm/MCA/HardwareUnits/RegisterFile.h"
#include "llvm/MCA/HardwareUnits/RetireControlUnit.h"
#include "llvm/MCA/HardwareUnits/Scheduler.h"
#include "llvm/MCA/Stages/DispatchStage.h"
#include "llvm/MCA/Stages/EntryStage.h"
#include "llvm/MCA/Stages/ExecuteStage.h"
#include "llvm/MCA/Stages/RetireStage.h"

namespace llvm {
namespace mca {

std::unique_ptr<Pipeline>
Context::createDefaultPipeline(const PipelineOptions &Opts, InstrBuilder &IB,
                               SourceMgr &SrcMgr) {
  const MCSchedModel &SM = STI.getSchedModel();

  // Create the hardware units defining the backend.
  auto RCU = llvm::make_unique<RetireControlUnit>(SM);
  auto PRF = llvm::make_unique<RegisterFile>(SM, MRI, Opts.RegisterFileSize);
  auto LSU = llvm::make_unique<LSUnit>(SM, Opts.LoadQueueSize,
                                       Opts.StoreQueueSize, Opts.AssumeNoAlias);
  auto HWS = llvm::make_unique<Scheduler>(SM, *LSU);

  // Create the pipeline stages.
  auto Fetch = llvm::make_unique<EntryStage>(SrcMgr);
  auto Dispatch = llvm::make_unique<DispatchStage>(STI, MRI, Opts.DispatchWidth,
                                                   *RCU, *PRF);
  auto Execute = llvm::make_unique<ExecuteStage>(*HWS);
  auto Retire = llvm::make_unique<RetireStage>(*RCU, *PRF);

  // Pass the ownership of all the hardware units to this Context.
  addHardwareUnit(std::move(RCU));
  addHardwareUnit(std::move(PRF));
  addHardwareUnit(std::move(LSU));
  addHardwareUnit(std::move(HWS));

  // Build the pipeline.
  auto StagePipeline = llvm::make_unique<Pipeline>();
  StagePipeline->appendStage(std::move(Fetch));
  StagePipeline->appendStage(std::move(Dispatch));
  StagePipeline->appendStage(std::move(Execute));
  StagePipeline->appendStage(std::move(Retire));
  return StagePipeline;
}

} // namespace mca
} // namespace llvm
