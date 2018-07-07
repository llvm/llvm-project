//===---------------------------- Context.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a class for holding ownership of various simulated
/// hardware units.  A Context also provides a utility routine for constructing
/// a default out-of-order pipeline with fetch, dispatch, execute, and retire
/// stages).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_CONTEXT_H
#define LLVM_TOOLS_LLVM_MCA_CONTEXT_H
#include "HardwareUnit.h"
#include "InstrBuilder.h"
#include "Pipeline.h"
#include "SourceMgr.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include <memory>

namespace mca {

/// This is a convenience struct to hold the parameters necessary for creating
/// the pre-built "default" out-of-order pipeline.
struct PipelineOptions {
  PipelineOptions(unsigned DW, unsigned RFS, unsigned LQS, unsigned SQS,
                  bool NoAlias)
      : DispatchWidth(DW), RegisterFileSize(RFS), LoadQueueSize(LQS),
        StoreQueueSize(SQS), AssumeNoAlias(NoAlias) {}
  unsigned DispatchWidth;
  unsigned RegisterFileSize;
  unsigned LoadQueueSize;
  unsigned StoreQueueSize;
  bool AssumeNoAlias;
};

class Context {
  llvm::SmallVector<std::unique_ptr<HardwareUnit>, 4> Hardware;
  const llvm::MCRegisterInfo &MRI;
  const llvm::MCSubtargetInfo &STI;

public:
  Context(const llvm::MCRegisterInfo &R, const llvm::MCSubtargetInfo &S)
      : MRI(R), STI(S) {}
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
#endif // LLVM_TOOLS_LLVM_MCA_CONTEXT_H
