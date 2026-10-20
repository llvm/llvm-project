//===--------------------- CustomBehaviour.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods from the CustomBehaviour interface.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/Instruction.h"

namespace llvm {
namespace mca {

CustomBehaviour::~CustomBehaviour() = default;

unsigned CustomBehaviour::checkCustomHazard(ArrayRef<InstRef> IssuedInst,
                                            const InstRef &IR) {
  // 0 signifies that there are no hazards that need to be waited on
  return 0;
}

std::vector<std::unique_ptr<View>>
CustomBehaviour::getStartViews(llvm::MCInstPrinter &IP,
                               llvm::ArrayRef<llvm::MCInst> Insts) {
  return std::vector<std::unique_ptr<View>>();
}

std::vector<std::unique_ptr<View>>
CustomBehaviour::getPostInstrInfoViews(llvm::MCInstPrinter &IP,
                                       llvm::ArrayRef<llvm::MCInst> Insts) {
  return std::vector<std::unique_ptr<View>>();
}

std::vector<std::unique_ptr<View>>
CustomBehaviour::getEndViews(llvm::MCInstPrinter &IP,
                             llvm::ArrayRef<llvm::MCInst> Insts) {
  return std::vector<std::unique_ptr<View>>();
}

const llvm::StringRef LatencyInstrument::DESC_NAME = "LATENCY";

bool InstrumentManager::supportsInstrumentType(StringRef Type) const {
  return EnableInstruments && Type == LatencyInstrument::DESC_NAME;
}

bool InstrumentManager::canCustomize(const ArrayRef<Instrument *> IVec) const {
  for (const auto I : IVec) {
    if (I->getDesc() == LatencyInstrument::DESC_NAME) {
      auto LatInst = static_cast<LatencyInstrument *>(I);
      return LatInst->hasValue();
    }
  }
  return false;
}

void InstrumentManager::customize(const ArrayRef<Instrument *> IVec,
                                  InstrDesc &ID) const {
  for (const auto I : IVec) {
    if (I->getDesc() == LatencyInstrument::DESC_NAME) {
      auto LatInst = static_cast<LatencyInstrument *>(I);
      if (LatInst->hasValue()) {
        unsigned Latency = LatInst->getLatency();
        // TODO Allow to customize a subset of ID.Writes
        for (auto &W : ID.Writes)
          W.Latency = Latency;
        ID.MaxLatency = Latency;
      }
    }
  }
}

UniqueInstrument InstrumentManager::createInstrument(StringRef Desc,
                                                     StringRef Data) {
  if (EnableInstruments) {
    if (Desc == LatencyInstrument::DESC_NAME)
      return std::make_unique<LatencyInstrument>(Data);
  }
  return std::make_unique<Instrument>(Desc, Data);
}

SmallVector<UniqueInstrument>
InstrumentManager::createInstruments(const MCInst &Inst) {
  return SmallVector<UniqueInstrument>();
}

unsigned InstrumentManager::getSchedClassID(
    const MCInstrInfo &MCII, const MCInst &MCI,
    const llvm::SmallVector<Instrument *> &IVec) const {
  return MCII.get(MCI.getOpcode()).getSchedClass();
}

} // namespace mca
} // namespace llvm
