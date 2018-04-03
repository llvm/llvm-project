//===- MCSchedule.cpp - Scheduling ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the default scheduling model.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include <type_traits>

using namespace llvm;

static_assert(std::is_pod<MCSchedModel>::value,
              "We shouldn't have a static constructor here");
const MCSchedModel MCSchedModel::Default = {DefaultIssueWidth,
                                            DefaultMicroOpBufferSize,
                                            DefaultLoopMicroOpBufferSize,
                                            DefaultLoadLatency,
                                            DefaultHighLatency,
                                            DefaultMispredictPenalty,
                                            false,
                                            true,
                                            0,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            nullptr};

int MCSchedModel::computeInstrLatency(const MCSubtargetInfo &STI,
                                      const MCSchedClassDesc &SCDesc) {
  int Latency = 0;
  for (unsigned DefIdx = 0, DefEnd = SCDesc.NumWriteLatencyEntries;
       DefIdx != DefEnd; ++DefIdx) {
    // Lookup the definition's write latency in SubtargetInfo.
    const MCWriteLatencyEntry *WLEntry =
        STI.getWriteLatencyEntry(&SCDesc, DefIdx);
    // Early exit if we found an invalid latency.
    if (WLEntry->Cycles < 0)
      return WLEntry->Cycles;
    Latency = std::max(Latency, static_cast<int>(WLEntry->Cycles));
  }
  return Latency;
}


Optional<double>
MCSchedModel::getReciprocalThroughput(const MCSubtargetInfo &STI,
                                      const MCSchedClassDesc &SCDesc) {
  Optional<double> Throughput;
  const MCSchedModel &SchedModel = STI.getSchedModel();

  for (const MCWriteProcResEntry *WPR = STI.getWriteProcResBegin(&SCDesc),
                                 *WEnd = STI.getWriteProcResEnd(&SCDesc);
       WPR != WEnd; ++WPR) {
    if (WPR->Cycles) {
      unsigned NumUnits =
          SchedModel.getProcResource(WPR->ProcResourceIdx)->NumUnits;
      double Temp = NumUnits * 1.0 / WPR->Cycles;
      Throughput =
          Throughput.hasValue() ? std::min(Throughput.getValue(), Temp) : Temp;
    }
  }

  if (Throughput.hasValue())
    return 1 / Throughput.getValue();
  return Throughput;
}
