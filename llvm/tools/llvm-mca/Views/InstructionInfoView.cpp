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
#include "llvm/Support/JSON.h"
#include "llvm/Support/WithColor.h"

namespace llvm {
namespace mca {

void InstructionInfoView::getComment(const MCInst &MCI,
                                     std::string &CommentString) const {
  StringRef s = MCI.getLoc().getPointer();
  std::string InstrStr;
  size_t pos = 0, pos_cmt = 0;

  // Recognized comments are after assembly instructions on the same line.
  // It is usefull to add in comment scheduling information from architecture
  // specification.
  // '#' comment mark is not supported by llvm-mca

  CommentString = "";
  if ((pos = s.find("\n")) != std::string::npos) {
    InstrStr = s.substr(0, pos);
    // C style comment
    if (((pos_cmt = InstrStr.find("/*")) != std::string::npos) &&
        ((pos = InstrStr.find("*/")) != std::string::npos)) {
      CommentString = InstrStr.substr(pos_cmt, pos);
      return;
    }
    // C++ style comment
    if ((pos_cmt = InstrStr.find("//")) != std::string::npos) {
      CommentString = InstrStr.substr(pos_cmt, pos);
      return;
    }
  }
  return;
}

void InstructionInfoView::printSchedulingInfoView(raw_ostream &OS) const {
  std::string Buffer;
  std::string CommentString;
  raw_string_ostream TempStream(Buffer);
  formatted_raw_ostream FOS(TempStream);

  ArrayRef<MCInst> Source = getSource();
  if (!Source.size())
    return;

  IIVDVec IIVD(Source.size());
  collectData(IIVD);

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
        if ((U + 1) < NumUnits) {
          FOS << ", ";
        }
      }
    }
    FOS << '\n';
    ResourceIndex++;
  }

  FOS << "\n\nScheduling Info:\n";
  FOS << "[1]: #uOps\n[2]: Latency\n[3]: Bypass Latency\n"
      << "[4]: Throughput\n[5]: Resources\n"
      << "[6]: LLVM OpcodeName\n";

  // paddings for each scheduling info output. Start at [2]
  std::vector<unsigned> paddings = {7, 12, 18, 27, 94, 113, 150};
  for (unsigned i = 0; i < paddings.size() - 1; i++) {
    FOS << "[" << i + 1 << "]";
    FOS.PadToColumn(paddings[i]);
  }
  FOS << "Instructions:\n";

  for (const auto &[Index, IIVDEntry, Inst] : enumerate(IIVD, Source)) {
    getComment(Inst, CommentString);

    FOS << " " << IIVDEntry.NumMicroOpcodes;
    FOS.PadToColumn(paddings[0]);
    FOS << " " << IIVDEntry.Latency;
    FOS.PadToColumn(paddings[1]);
    FOS << " " << IIVDEntry.Bypass;
    FOS.PadToColumn(paddings[2]);
    if (IIVDEntry.RThroughput) {
      double RT = 1.0 / *IIVDEntry.RThroughput;
      FOS << " " << format("%.2f", RT);
    } else {
      FOS << " -";
    }
    FOS.PadToColumn(paddings[3]);
    FOS << " " << IIVDEntry.Resources;
    FOS.PadToColumn(paddings[4]);
    FOS << " " << IIVDEntry.OpcodeName;
    FOS.PadToColumn(paddings[5]);
    FOS << " " << printInstructionString(Inst);
    FOS.PadToColumn(paddings[6]);
    FOS << CommentString << '\n';
  }

  FOS.flush();
  OS << Buffer;
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
    if (PrintEncodings)
      WithColor::warning()
          << "No encodings printed when -scheduling-info option enabled.\n";
    if (PrintBarriers)
      WithColor::warning()
          << "No barrier printed when -scheduling-info option enabled.\n";

    printSchedulingInfoView(OS);
    return;
  }

  std::vector<unsigned> paddings = {0, 7, 14, 21, 28, 35};
  std::vector<std::string>   fields   = {"#uOps", "Latency", "RThroughput",
    "MayLoad", "MayStore", "HasSideEffects (U)"};
  std::vector<std::string>   end_fields;
  unsigned last_padding = 35;
  if (PrintBarriers) {
    paddings.push_back(last_padding + 7);
    paddings.push_back(last_padding + 14);
    last_padding += 14;
    fields.push_back("LoadBarrier"); fields.push_back("StoreBarrier");
  }
  if (PrintEncodings) {
    paddings.push_back(last_padding + 7);
    paddings.push_back(last_padding + 14);
    paddings.push_back(last_padding + 44);
    last_padding += 44;
    fields.push_back("Encoding Size");
    end_fields.push_back("Encodings:");
    end_fields.push_back("Instructions:");
  }
  else {
    paddings.push_back(last_padding + 7);
    last_padding += 7;
    end_fields.push_back("Instructions:");
  }

  FOS << "\n\nInstruction Info:\n";
  for (unsigned i = 0; i < fields.size(); i++) {
    FOS << "[" << i + 1 << "]: " << fields[i] << "\n";
  }
  FOS << "\n";

  for (unsigned i = 0; i < paddings.size(); i++) {
    if (paddings[i] != 0)
      FOS.PadToColumn(paddings[i]);
    if (i < fields.size()) {
      FOS << "[" << i + 1 << "]";
    }
    else {
      FOS << end_fields[i - fields.size()];
    }
  }
  FOS << "\n";

  for (const auto &[Index, IIVDEntry, Inst] : enumerate(IIVD, Source)) {
    FOS.PadToColumn(paddings[0]+1);
    FOS << IIVDEntry.NumMicroOpcodes;
    FOS.PadToColumn(paddings[1]+1);
    FOS << IIVDEntry.Latency;

    FOS.PadToColumn(paddings[2]);
    if (IIVDEntry.RThroughput) {
      double RT = *IIVDEntry.RThroughput;
      FOS << format("%.2f", RT);
    } else {
      FOS << " -";
    }
    FOS.PadToColumn(paddings[3]+1);
    FOS << (IIVDEntry.mayLoad ? "*" : " ");
    FOS.PadToColumn(paddings[4]+1);
    FOS << (IIVDEntry.mayStore ? "*" : " ");
    FOS.PadToColumn(paddings[5]+1);
    FOS << (IIVDEntry.hasUnmodeledSideEffects ? "U" : " ");
    unsigned last_padding_idx = 5;

    if (PrintBarriers) {
      FOS.PadToColumn(paddings[last_padding_idx + 1]+1);
      FOS << (LoweredInsts[Index]->isALoadBarrier() ? "*" : " ");
      FOS.PadToColumn(paddings[last_padding_idx + 2]+1);
      FOS << (LoweredInsts[Index]->isAStoreBarrier() ? "*" : " ");
      last_padding_idx += 2;
    }

    if (PrintEncodings) {
      StringRef Encoding(CE.getEncoding(Index));
      unsigned EncodingSize = Encoding.size();
      FOS.PadToColumn(paddings[last_padding_idx + 1]+1);
      FOS << EncodingSize;
      FOS.PadToColumn(paddings[last_padding_idx + 2]);
      for (unsigned i = 0, e = Encoding.size(); i != e; ++i)
        FOS << format("%02x ", (uint8_t)Encoding[i]);
      last_padding_idx += 2;
    }
    FOS.PadToColumn(paddings[last_padding_idx + 1]);
    FOS << printInstructionString(Inst) << '\n';
  }

  FOS.flush();
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
      IIVDEntry.OpcodeName = (std::string)MCII.getName(Inst.getOpcode());
      raw_string_ostream TempStream(IIVDEntry.Resources);
      const MCWriteProcResEntry *Index = STI.getWriteProcResBegin(&SCDesc);
      const MCWriteProcResEntry *Last = STI.getWriteProcResEnd(&SCDesc);
      auto sep = "";
      for (; Index != Last; ++Index) {
        if (!Index->ReleaseAtCycle)
          continue;
        const MCProcResourceDesc *MCProc =
            SM.getProcResource(Index->ProcResourceIdx);
        if (Index->ReleaseAtCycle > 1) {
          // Output ReleaseAtCycle between [] if not 1 (default)
          // This is to be able to evaluate throughput.
          // See getReciprocalThroughput in MCSchedule.cpp
          TempStream << sep
                     << format("%s[%d]", MCProc->Name, Index->ReleaseAtCycle);
        } else {
          TempStream << sep << format("%s", MCProc->Name);
        }
        sep = ",";
      }
      TempStream.flush();
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
