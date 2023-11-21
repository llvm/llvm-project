//===- TargetSubtargetInfo.cpp - General Target Information ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file describes the general parts of a Subtarget.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

static cl::list<std::string> MFusions("mfusion", cl::CommaSeparated,
                                      cl::desc("Target specific macro fusions"),
                                      cl::value_desc("a1,+a2,-a3,..."));

TargetSubtargetInfo::TargetSubtargetInfo(
    const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS,
    ArrayRef<SubtargetFeatureKV> PF, ArrayRef<SubtargetSubTypeKV> PD,
    const MCWriteProcResEntry *WPR, const MCWriteLatencyEntry *WL,
    const MCReadAdvanceEntry *RA, const InstrStage *IS, const unsigned *OC,
    const unsigned *FP, ArrayRef<MacroFusionEntry> MF)
    : MCSubtargetInfo(TT, CPU, TuneCPU, FS, PF, PD, WPR, WL, RA, IS, OC, FP),
      MacroFusionTable(MF) {
  // assert if MacroFusionTable is not sorted.
  assert(llvm::is_sorted(MacroFusionTable));
  overrideFusionBits();
}

TargetSubtargetInfo::~TargetSubtargetInfo() = default;

void TargetSubtargetInfo::overrideFusionBits() {
  if (MFusions.getNumOccurrences() != 0) {
    for (std::string &MFusion : MFusions) {
      char Prefix = MFusion[0];
      bool Disable = Prefix == '-';
      if (Prefix == '+' || Prefix == '-')
        MFusion = MFusion.substr(1);

      // MacroFusionTable is sorted.
      const auto *Pos = std::lower_bound(
          MacroFusionTable.begin(), MacroFusionTable.end(), MFusion,
          [](const MacroFusionEntry &LHS, const std::string &RHS) {
            int CmpName = StringRef(LHS.Name).compare(RHS);
            if (CmpName < 0)
              return true;
            if (CmpName > 0)
              return false;
            return false;
          });

      if (Pos == MacroFusionTable.end()) {
        errs() << "'" << MFusion
               << "' is not a recognized macro fusion for this "
               << "target (ignoring it)\n";
        continue;
      }

      // The index is the same as the enum value.
      unsigned Idx = Pos - MacroFusionTable.begin();
      if (Disable)
        disableMacroFusion(Idx);
      else
        enableMacroFusion(Idx);
    }
  }
}

bool TargetSubtargetInfo::enableAtomicExpand() const {
  return true;
}

bool TargetSubtargetInfo::enableIndirectBrExpand() const {
  return false;
}

bool TargetSubtargetInfo::enableMachineScheduler() const {
  return false;
}

bool TargetSubtargetInfo::enableJoinGlobalCopies() const {
  return enableMachineScheduler();
}

bool TargetSubtargetInfo::enableRALocalReassignment(
    CodeGenOptLevel OptLevel) const {
  return true;
}

bool TargetSubtargetInfo::enablePostRAScheduler() const {
  return getSchedModel().PostRAScheduler;
}

bool TargetSubtargetInfo::enablePostRAMachineScheduler() const {
  return enableMachineScheduler() && enablePostRAScheduler();
}

bool TargetSubtargetInfo::useAA() const {
  return false;
}

void TargetSubtargetInfo::mirFileLoaded(MachineFunction &MF) const { }

std::vector<MacroFusionPredTy> TargetSubtargetInfo::getMacroFusions() const {
  std::vector<MacroFusionPredTy> Fusions;
  const MacroFusionBitset &Bits = getMacroFusionBits();
  for (unsigned I = 0; I < MacroFusionTable.size(); I++)
    if (Bits[I])
      Fusions.push_back(MacroFusionTable[I].Pred);

  return Fusions;
}
