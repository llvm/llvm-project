//===---------------------------- Context.h ---------------------*- C++ -*-===//
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

#ifndef LLVM_MCA_CONTEXT_H
#define LLVM_MCA_CONTEXT_H

#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/HardwareUnits/HardwareUnit.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include <memory>

namespace llvm {
namespace mca {

/// This is a convenience struct to hold the parameters necessary for creating
/// the pre-built "default" out-of-order pipeline.
struct PipelineOptions {
  PipelineOptions(unsigned DW, unsigned RFS, unsigned LQS, unsigned SQS,
                  bool NoAlias, bool ShouldEnableBottleneckAnalysis = false)
      : DispatchWidth(DW), RegisterFileSize(RFS), LoadQueueSize(LQS),
        StoreQueueSize(SQS), AssumeNoAlias(NoAlias),
        EnableBottleneckAnalysis(ShouldEnableBottleneckAnalysis) {}
  unsigned DispatchWidth;
  unsigned RegisterFileSize;
  unsigned LoadQueueSize;
  unsigned StoreQueueSize;
  bool AssumeNoAlias;
  bool EnableBottleneckAnalysis;
};

class Context {
  SmallVector<std::unique_ptr<HardwareUnit>, 4> Hardware;
  const MCRegisterInfo &MRI;
  const MCSubtargetInfo &STI;

public:
  Context(const MCRegisterInfo &R, const MCSubtargetInfo &S) : MRI(R), STI(S) {}
  Context(const Context &C) = delete;
  Context &operator=(const Context &C) = delete;

  void addHardwareUnit(std::unique_ptr<HardwareUnit> H) {
    Hardware.push_back(std::move(H));
  }

  /// Construct a basic pipeline for simulating an out-of-order pipeline.
  /// This pipeline consists of Fetch, Dispatch, Execute, and Retire stages.
  std::unique_ptr<Pipeline> createDefaultPipeline(const PipelineOptions &Opts,
                                                  InstrBuilder &IB,
                                                  SourceMgr &SrcMgr);
};

} // namespace mca
} // namespace llvm
#endif // LLVM_MCA_CONTEXT_H
