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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/WithColor.h"

namespace llvm {
namespace mca {

void InstructionInfoView::getComment(raw_ostream &OS, const MCInst &MCI) const {
  StringRef S = MCI.getLoc().getPointer();
  size_t Pos = 0, PosCmt = 0;

  // Recognized comments are after assembly instructions on the same line.
  // It is usefull to add in comment scheduling information from architecture
  // specification.
  // '#' comment mark is not supported by llvm-mca

  if (Pos = S.find("\n"); Pos != StringRef::npos) {
    StringRef InstrStr = S.take_front(Pos);
    // C style comment
    if (((PosCmt = InstrStr.find("/*")) != StringRef::npos) &&
        ((Pos = InstrStr.find("*/")) != StringRef::npos)) {
      OS << InstrStr.substr(PosCmt, Pos);
      return;
    }
    // C++ style comment
    if ((PosCmt = InstrStr.find("//")) != StringRef::npos) {
      OS << InstrStr.substr(PosCmt);
    }
  }
}

void InstructionInfoView::printView(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  formatted_raw_ostream FOS(TempStream);

  ArrayRef<llvm::MCInst> Source = getSource();
  if (!Source.size())
    return;

  IIVDVec IIVD(Source.size());
  collectData(IIVD);

  if (PrintFullInfo) {
    FOS << "\n\nResources:\n";
    const MCSchedModel &SM = getSubTargetInfo().getSchedModel();
    for (unsigned I = 1, ResourceIndex = 0, E = SM.getNumProcResourceKinds();
         I < E; ++I) {
      const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
      unsigned NumUnits = ProcResource.NumUnits;
      // Skip invalid resources with zero units.
      if (!NumUnits)
        continue;

      FOS << '[' << ResourceIndex << ']';
      FOS.PadToColumn(6);
      FOS << "- " << ProcResource.Name << ':' << NumUnits;
      if (ProcResource.SubUnitsIdxBegin) {
        FOS.PadToColumn(20);
        for (unsigned U = 0; U < NumUnits; ++U) {
          FOS << SM.getProcResource(ProcResource.SubUnitsIdxBegin[U])->Name;
          if ((U + 1) < NumUnits)
            FOS << ", ";
        }
      }
      FOS << '\n';
      ResourceIndex++;
    }
  }

  SmallVector<unsigned, 16> Paddings = {0, 7, 14, 21, 28, 35, 42};
  SmallVector<StringRef, 16> Fields = {"#uOps",       "Latency",
                                       "RThroughput", "MayLoad",
                                       "MayStore",    "HasSideEffects (U)"};
  SmallVector<StringRef, 8> EndFields;
  unsigned LastPadding = Paddings.back();
  if (PrintFullInfo) {
    Fields.push_back("Bypass Latency");
    // Reserving 7 chars for
    Paddings.push_back(LastPadding += 7);
    Fields.push_back("Resources (<Name> | <Name>[<ReleaseAtCycle>] | "
                     "<Name>[<AcquireAtCycle>,<ReleaseAtCycle])");
    Paddings.push_back(LastPadding += 43);
    Fields.push_back("LLVM Opcode Name");
    Paddings.push_back(LastPadding += 27);
  }
  if (PrintBarriers) {
    Fields.push_back("LoadBarrier");
    Paddings.push_back(LastPadding += 7);
    Fields.push_back("StoreBarrier");
    Paddings.push_back(LastPadding += 7);
  }
  if (PrintEncodings) {
    Fields.push_back("Encoding Size");
    Paddings.push_back(LastPadding += 7);
    EndFields.push_back("Encodings:");
    Paddings.push_back(LastPadding += 30);
  }
  EndFields.push_back("Instructions:");

  FOS << "\n\nInstruction Info:\n";
  for (unsigned i = 0, N = Fields.size(); i < N; i++)
    FOS << "[" << i + 1 << "]: " << Fields[i] << "\n";
  FOS << "\n";

  for (unsigned i = 0, N = Paddings.size(); i < N; i++) {
    if (Paddings[i])
      FOS.PadToColumn(Paddings[i]);
    if (i < Fields.size())
      FOS << "[" << i + 1 << "]";
    else
      FOS << EndFields[i - Fields.size()];
  }
  FOS << "\n";

  for (const auto &[Index, IIVDEntry, Inst] : enumerate(IIVD, Source)) {
    FOS.PadToColumn(Paddings[0] + 1);
    FOS << IIVDEntry.NumMicroOpcodes;
    FOS.PadToColumn(Paddings[1] + 1);
    FOS << IIVDEntry.Latency;
    FOS.PadToColumn(Paddings[2]);
    if (IIVDEntry.RThroughput) {
      double RT = *IIVDEntry.RThroughput;
      FOS << format("%.2f", RT);
    } else {
      FOS << " -";
    }
    FOS.PadToColumn(Paddings[3] + 1);
    FOS << (IIVDEntry.mayLoad ? "*" : " ");
    FOS.PadToColumn(Paddings[4] + 1);
    FOS << (IIVDEntry.mayStore ? "*" : " ");
    FOS.PadToColumn(Paddings[5] + 1);
    FOS << (IIVDEntry.hasUnmodeledSideEffects ? "U" : " ");
    unsigned LastPaddingIdx = 5;

    if (PrintFullInfo) {
      FOS.PadToColumn(Paddings[LastPaddingIdx += 1] + 1);
      FOS << IIVDEntry.Bypass;
      FOS.PadToColumn(Paddings[LastPaddingIdx += 1]);
      FOS << IIVDEntry.Resources;
      FOS.PadToColumn(Paddings[LastPaddingIdx += 1]);
      FOS << IIVDEntry.OpcodeName;
    }

    if (PrintBarriers) {
      FOS.PadToColumn(Paddings[LastPaddingIdx += 1] + 1);
      FOS << (LoweredInsts[Index]->isALoadBarrier() ? "*" : " ");
      FOS.PadToColumn(Paddings[LastPaddingIdx += 1] + 1);
      FOS << (LoweredInsts[Index]->isAStoreBarrier() ? "*" : " ");
    }

    if (PrintEncodings) {
      StringRef Encoding(CE.getEncoding(Index));
      unsigned EncodingSize = Encoding.size();
      FOS.PadToColumn(Paddings[LastPaddingIdx += 1] + 1);
      FOS << EncodingSize;
      FOS.PadToColumn(Paddings[LastPaddingIdx += 1]);
      for (unsigned i = 0, e = Encoding.size(); i != e; ++i)
        FOS << format("%02x ", (uint8_t)Encoding[i]);
    }
    FOS.PadToColumn(Paddings[LastPaddingIdx += 1]);
    FOS << printInstructionString(Inst);
    if (PrintFullInfo) {
      FOS << "\t";
      getComment(FOS, Inst);
    }
    FOS << '\n';
  }

  OS << Buffer;
}

void InstructionInfoView::collectData(
    MutableArrayRef<InstructionInfoViewData> IIVD) const {
  const llvm::MCSubtargetInfo &STI = getSubTargetInfo();
  const MCSchedModel &SM = STI.getSchedModel();
  for (const auto I : zip(getSource(), IIVD)) {
    const MCInst &Inst = std::get<0>(I);
    InstructionInfoViewData &IIVDEntry = std::get<1>(I);
    const MCInstrDesc &MCDesc = MCII.get(Inst.getOpcode());

    // Obtain the scheduling class information from the instruction
    // and instruments.
    auto IVecIt = InstToInstruments.find(&Inst);
    unsigned SchedClassID =
        IVecIt == InstToInstruments.end()
            ? MCDesc.getSchedClass()
            : IM.getSchedClassID(MCII, Inst, IVecIt->second);
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

    if (PrintFullInfo) {
      // Get latency with bypass
      IIVDEntry.Bypass =
          IIVDEntry.Latency - MCSchedModel::getBypassDelayCycles(STI, SCDesc);
      IIVDEntry.OpcodeName = MCII.getName(Inst.getOpcode());
      raw_string_ostream TempStream(IIVDEntry.Resources);
      const MCWriteProcResEntry *Index = STI.getWriteProcResBegin(&SCDesc);
      const MCWriteProcResEntry *Last = STI.getWriteProcResEnd(&SCDesc);
      ListSeparator LS(",");
      for (; Index != Last; ++Index) {
        if (!Index->ReleaseAtCycle)
          continue;
        const MCProcResourceDesc *MCProc =
            SM.getProcResource(Index->ProcResourceIdx);
        if (Index->ReleaseAtCycle > 1) {
          // Output ReleaseAtCycle between [] if not 1 (default)
          // This is to be able to evaluate throughput.
          // See getReciprocalThroughput in MCSchedule.cpp
          if (Index->AcquireAtCycle > 0)
            TempStream << LS
                       << format("%s[%d,%d]", MCProc->Name,
                                 Index->AcquireAtCycle, Index->ReleaseAtCycle);
          else
            TempStream << LS
                       << format("%s[%d]", MCProc->Name, Index->ReleaseAtCycle);
        } else {
          TempStream << LS << MCProc->Name;
        }
      }
    }
  }
}

// Construct a JSON object from a single InstructionInfoViewData object.
json::Object
InstructionInfoView::toJSON(const InstructionInfoViewData &IIVD) const {
  json::Object JO({{"NumMicroOpcodes", IIVD.NumMicroOpcodes},
                   {"Latency", IIVD.Latency},
                   {"mayLoad", IIVD.mayLoad},
                   {"mayStore", IIVD.mayStore},
                   {"hasUnmodeledSideEffects", IIVD.hasUnmodeledSideEffects}});
  JO.try_emplace("RThroughput", IIVD.RThroughput.value_or(0.0));
  return JO;
}

json::Value InstructionInfoView::toJSON() const {
  ArrayRef<llvm::MCInst> Source = getSource();
  if (!Source.size())
    return json::Value(0);

  IIVDVec IIVD(Source.size());
  collectData(IIVD);

  json::Array InstInfo;
  for (const auto &I : enumerate(IIVD)) {
    const InstructionInfoViewData &IIVDEntry = I.value();
    json::Object JO = toJSON(IIVDEntry);
    JO.try_emplace("Instruction", (unsigned)I.index());
    InstInfo.push_back(std::move(JO));
  }
  return json::Object({{"InstructionList", json::Value(std::move(InstInfo))}});
}
} // namespace mca.
} // namespace llvm
