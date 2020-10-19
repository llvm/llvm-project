//===--------------------- InstructionInfoView.cpp --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the InstructionInfoView API.
///
//===----------------------------------------------------------------------===//

#include "Views/InstructionInfoView.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {
namespace mca {

void InstructionInfoView::printView(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  ArrayRef<llvm::MCInst> Source = getSource();
  if (!Source.size())
    return;

  IIVDVec IIVD(Source.size());
  collectData(IIVD);

  TempStream << "\n\nInstruction Info:\n";
  TempStream << "[1]: #uOps\n[2]: Latency\n[3]: RThroughput\n"
             << "[4]: MayLoad\n[5]: MayStore\n[6]: HasSideEffects (U)\n";
  if (PrintEncodings) {
    TempStream << "[7]: Encoding Size\n";
    TempStream << "\n[1]    [2]    [3]    [4]    [5]    [6]    [7]    "
               << "Encodings:                    Instructions:\n";
  } else {
    TempStream << "\n[1]    [2]    [3]    [4]    [5]    [6]    Instructions:\n";
  }

  for (auto I : enumerate(zip(IIVD, Source))) {
    const InstructionInfoViewData &IIVDEntry = std::get<0>(I.value());

    TempStream << ' ' << IIVDEntry.NumMicroOpcodes << "    ";
    if (IIVDEntry.NumMicroOpcodes < 10)
      TempStream << "  ";
    else if (IIVDEntry.NumMicroOpcodes < 100)
      TempStream << ' ';
    TempStream << IIVDEntry.Latency << "   ";
    if (IIVDEntry.Latency < 10)
      TempStream << "  ";
    else if (IIVDEntry.Latency < 100)
      TempStream << ' ';

    if (IIVDEntry.RThroughput.hasValue()) {
      double RT = IIVDEntry.RThroughput.getValue();
      TempStream << format("%.2f", RT) << ' ';
      if (RT < 10.0)
        TempStream << "  ";
      else if (RT < 100.0)
        TempStream << ' ';
    } else {
      TempStream << " -     ";
    }
    TempStream << (IIVDEntry.mayLoad ? " *     " : "       ");
    TempStream << (IIVDEntry.mayStore ? " *     " : "       ");
    TempStream << (IIVDEntry.hasUnmodeledSideEffects ? " U     " : "       ");

    if (PrintEncodings) {
      StringRef Encoding(CE.getEncoding(I.index()));
      unsigned EncodingSize = Encoding.size();
      TempStream << " " << EncodingSize
                 << (EncodingSize < 10 ? "     " : "    ");
      TempStream.flush();
      formatted_raw_ostream FOS(TempStream);
      for (unsigned i = 0, e = Encoding.size(); i != e; ++i)
        FOS << format("%02x ", (uint8_t)Encoding[i]);
      FOS.PadToColumn(30);
      FOS.flush();
    }

    const MCInst &Inst = std::get<1>(I.value());
    TempStream << printInstructionString(Inst) << '\n';
  }

  TempStream.flush();
  OS << Buffer;
}

void InstructionInfoView::collectData(
    MutableArrayRef<InstructionInfoViewData> IIVD) const {
  const llvm::MCSubtargetInfo &STI = getSubTargetInfo();
  const MCSchedModel &SM = STI.getSchedModel();
  for (auto I : zip(getSource(), IIVD)) {
    const MCInst &Inst = std::get<0>(I);
    InstructionInfoViewData &IIVDEntry = std::get<1>(I);
    const MCInstrDesc &MCDesc = MCII.get(Inst.getOpcode());

    // Obtain the scheduling class information from the instruction.
    unsigned SchedClassID = MCDesc.getSchedClass();
    unsigned CPUID = SM.getProcessorID();

    // Try to solve variant scheduling classes.
    while (SchedClassID && SM.getSchedClassDesc(SchedClassID)->isVariant())
      SchedClassID =
          STI.resolveVariantSchedClass(SchedClassID, &Inst, &MCII, CPUID);

    const MCSchedClassDesc &SCDesc = *SM.getSchedClassDesc(SchedClassID);
    IIVDEntry.NumMicroOpcodes = SCDesc.NumMicroOps;
    IIVDEntry.Latency = MCSchedModel::computeInstrLatency(STI, SCDesc);
    // Add extra latency due to delays in the forwarding data paths.
    IIVDEntry.Latency += MCSchedModel::getForwardingDelayCycles(
        STI.getReadAdvanceEntries(SCDesc));
    IIVDEntry.RThroughput = MCSchedModel::getReciprocalThroughput(STI, SCDesc);
    IIVDEntry.mayLoad = MCDesc.mayLoad();
    IIVDEntry.mayStore = MCDesc.mayStore();
    IIVDEntry.hasUnmodeledSideEffects = MCDesc.hasUnmodeledSideEffects();
  }
}
} // namespace mca.
} // namespace llvm
