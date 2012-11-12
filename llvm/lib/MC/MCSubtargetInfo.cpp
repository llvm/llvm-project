//===-- MCSubtargetInfo.cpp - Subtarget Information -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

MCSchedModel MCSchedModel::DefaultSchedModel; // For unknown processors.

/// InitMCProcessorInfo - Set or change the CPU (optionally supplemented
/// with feature string). Recompute feature bits and scheduling model.
void
MCSubtargetInfo::InitMCProcessorInfo(StringRef CPU, StringRef FS) {
  SubtargetFeatures Features(FS);
  FeatureBits = Features.getFeatureBits(CPU, ProcDesc, NumProcs,
                                        ProcFeatures, NumFeatures);

  if (!CPU.empty())
    CPUSchedModel = getSchedModelForCPU(CPU);
  else
    CPUSchedModel = &MCSchedModel::DefaultSchedModel;
}

void
MCSubtargetInfo::InitMCSubtargetInfo(StringRef TT, StringRef CPU, StringRef FS,
                                     const SubtargetFeatureKV *PF,
                                     const SubtargetFeatureKV *PD,
                                     const SubtargetInfoKV *ProcSched,
                                     const MCWriteProcResEntry *WPR,
                                     const MCWriteLatencyEntry *WL,
                                     const MCReadAdvanceEntry *RA,
                                     const InstrStage *IS,
                                     const unsigned *OC,
                                     const unsigned *FP,
                                     unsigned NF, unsigned NP) {
  TargetTriple = TT;
  ProcFeatures = PF;
  ProcDesc = PD;
  ProcSchedModels = ProcSched;
  WriteProcResTable = WPR;
  WriteLatencyTable = WL;
  ReadAdvanceTable = RA;

  Stages = IS;
  OperandCycles = OC;
  ForwardingPaths = FP;
  NumFeatures = NF;
  NumProcs = NP;

  InitMCProcessorInfo(CPU, FS);
}

/// ToggleFeature - Toggle a feature and returns the re-computed feature
/// bits. This version does not change the implied bits.
uint64_t MCSubtargetInfo::ToggleFeature(uint64_t FB) {
  FeatureBits ^= FB;
  return FeatureBits;
}

/// ToggleFeature - Toggle a feature and returns the re-computed feature
/// bits. This version will also change all implied bits.
uint64_t MCSubtargetInfo::ToggleFeature(StringRef FS) {
  SubtargetFeatures Features;
  FeatureBits = Features.ToggleFeature(FeatureBits, FS,
                                       ProcFeatures, NumFeatures);
  return FeatureBits;
}


const MCSchedModel *
MCSubtargetInfo::getSchedModelForCPU(StringRef CPU) const {
  assert(ProcSchedModels && "Processor machine model not available!");

#ifndef NDEBUG
  for (size_t i = 1; i < NumProcs; i++) {
    assert(strcmp(ProcSchedModels[i - 1].Key, ProcSchedModels[i].Key) < 0 &&
           "Processor machine model table is not sorted");
  }
#endif

  // Find entry
  SubtargetInfoKV KV;
  KV.Key = CPU.data();
  const SubtargetInfoKV *Found =
    std::lower_bound(ProcSchedModels, ProcSchedModels+NumProcs, KV);
  if (Found == ProcSchedModels+NumProcs || StringRef(Found->Key) != CPU) {
    errs() << "'" << CPU
           << "' is not a recognized processor for this target"
           << " (ignoring processor)\n";
    return &MCSchedModel::DefaultSchedModel;
  }
  assert(Found->Value && "Missing processor SchedModel value");
  return (const MCSchedModel *)Found->Value;
}

InstrItineraryData
MCSubtargetInfo::getInstrItineraryForCPU(StringRef CPU) const {
  const MCSchedModel *SchedModel = getSchedModelForCPU(CPU);
  return InstrItineraryData(SchedModel, Stages, OperandCycles, ForwardingPaths);
}

/// Initialize an InstrItineraryData instance.
void MCSubtargetInfo::initInstrItins(InstrItineraryData &InstrItins) const {
  InstrItins =
    InstrItineraryData(CPUSchedModel, Stages, OperandCycles, ForwardingPaths);
}
