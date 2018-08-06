//===--------------------- RegisterFileStatistics.cpp -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the RegisterFileStatistics interface.
///
//===----------------------------------------------------------------------===//

#include "RegisterFileStatistics.h"
#include "llvm/Support/Format.h"

using namespace llvm;

namespace mca {

void RegisterFileStatistics::initializeRegisterFileInfo() {
  const MCSchedModel &SM = STI.getSchedModel();
  RegisterFileUsage Empty = {0, 0, 0};
  if (!SM.hasExtraProcessorInfo()) {
    // Assume a single register file.
    RegisterFiles.emplace_back(Empty);
    return;
  }

  // Initialize a RegisterFileUsage for every user defined register file, plus
  // the default register file which is always at index #0.
  const MCExtraProcessorInfo &PI = SM.getExtraProcessorInfo();
  // There is always an "InvalidRegisterFile" entry in tablegen. That entry can
  // be skipped. If there are no user defined register files, then reserve a
  // single entry for the default register file at index #0.
  unsigned NumRegFiles = std::max(PI.NumRegisterFiles, 1U);
  RegisterFiles.resize(NumRegFiles);
  std::fill(RegisterFiles.begin(), RegisterFiles.end(), Empty);
}

void RegisterFileStatistics::onEvent(const HWInstructionEvent &Event) {
  switch (Event.Type) {
  default:
    break;
  case HWInstructionEvent::Retired: {
    const auto &RE = static_cast<const HWInstructionRetiredEvent &>(Event);
    for (unsigned I = 0, E = RegisterFiles.size(); I < E; ++I)
      RegisterFiles[I].CurrentlyUsedMappings -= RE.FreedPhysRegs[I];
    break;
  }
  case HWInstructionEvent::Dispatched: {
    const auto &DE = static_cast<const HWInstructionDispatchedEvent &>(Event);
    for (unsigned I = 0, E = RegisterFiles.size(); I < E; ++I) {
      RegisterFileUsage &RFU = RegisterFiles[I];
      unsigned NumUsedPhysRegs = DE.UsedPhysRegs[I];
      RFU.CurrentlyUsedMappings += NumUsedPhysRegs;
      RFU.TotalMappings += NumUsedPhysRegs;
      RFU.MaxUsedMappings =
          std::max(RFU.MaxUsedMappings, RFU.CurrentlyUsedMappings);
    }
  }
  }
}

void RegisterFileStatistics::printView(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  TempStream << "\n\nRegister File statistics:";
  const RegisterFileUsage &GlobalUsage = RegisterFiles[0];
  TempStream << "\nTotal number of mappings created:    "
             << GlobalUsage.TotalMappings;
  TempStream << "\nMax number of mappings used:         "
             << GlobalUsage.MaxUsedMappings << '\n';

  for (unsigned I = 1, E = RegisterFiles.size(); I < E; ++I) {
    const RegisterFileUsage &RFU = RegisterFiles[I];
    // Obtain the register file descriptor from the scheduling model.
    assert(STI.getSchedModel().hasExtraProcessorInfo() &&
           "Unable to find register file info!");
    const MCExtraProcessorInfo &PI =
        STI.getSchedModel().getExtraProcessorInfo();
    assert(I <= PI.NumRegisterFiles && "Unexpected register file index!");
    const MCRegisterFileDesc &RFDesc = PI.RegisterFiles[I];
    // Skip invalid register files.
    if (!RFDesc.NumPhysRegs)
      continue;

    TempStream << "\n*  Register File #" << I;
    TempStream << " -- " << StringRef(RFDesc.Name) << ':';
    TempStream << "\n   Number of physical registers:     ";
    if (!RFDesc.NumPhysRegs)
      TempStream << "unbounded";
    else
      TempStream << RFDesc.NumPhysRegs;
    TempStream << "\n   Total number of mappings created: "
               << RFU.TotalMappings;
    TempStream << "\n   Max number of mappings used:      "
               << RFU.MaxUsedMappings << '\n';
  }

  TempStream.flush();
  OS << Buffer;
}

} // namespace mca
