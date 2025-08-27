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

class CustomInstrument : public Instrument {
  std::optional<unsigned> Latency;

public:
  static const llvm::StringRef DESC_NAME;
  explicit CustomInstrument(StringRef Data) : Instrument(DESC_NAME, Data) {
    // Skip spaces and tabs.
    unsigned Position = Data.find_first_not_of(" \t");
    if (Position >= Data.size())
      // We reached the end of the comment. Bail out.
      return;
    Data = Data.drop_front(Position);
    auto [Name, Value] = Data.split(':');
    if (Name.upper() == "LATENCY") {
      Position = Value.find_first_not_of(" \t");
      if (Position >= Value.size())
        return;
      auto Stripped = Value.drop_front(Position);
      unsigned L = 0;
      if (!Stripped.getAsInteger(10, L))
        Latency = L;
    }
  }

  bool canCustomize() const override { return bool(Latency); }

  void customize(InstrDesc &ID) const override {
    if (Latency) {
      for (auto &W : ID.Writes)
        W.Latency = *Latency;
      ID.MaxLatency = *Latency;
    }
  }
};

const llvm::StringRef CustomInstrument::DESC_NAME = "CUSTOMIZE";

bool InstrumentManager::supportsInstrumentType(StringRef Type) const {
  if (EnableDefaults && Type == CustomInstrument::DESC_NAME)
    return true;
  if (TargetIM)
    return TargetIM->supportsInstrumentType(Type);
  return false;
}

bool InstrumentManager::canCustomize(
    const llvm::SmallVector<Instrument *> &IVec) const {
  for (const auto I : IVec) {
    if (I->canCustomize())
      return true;
  }
  return false;
}

void InstrumentManager::customize(const llvm::SmallVector<Instrument *> &IVec,
                                  InstrDesc &ID) const {
  for (const auto I : IVec) {
    if (I->canCustomize())
      I->customize(ID);
  }
}

UniqueInstrument InstrumentManager::createInstrument(llvm::StringRef Desc,
                                                     llvm::StringRef Data) {
  if (!EnableDefaults)
    return std::make_unique<Instrument>(Desc, Data);
  if (Desc == CustomInstrument::DESC_NAME)
    return std::make_unique<CustomInstrument>(Data);
  if (TargetIM && TargetIM->supportsInstrumentType(Desc))
    return TargetIM->createInstrument(Desc, Data);
  return std::make_unique<Instrument>(Desc, Data);
}

SmallVector<UniqueInstrument>
InstrumentManager::createInstruments(const MCInst &Inst) {
  if (TargetIM)
    return TargetIM->createInstruments(Inst);
  return SmallVector<UniqueInstrument>();
}

unsigned InstrumentManager::getSchedClassID(
    const MCInstrInfo &MCII, const MCInst &MCI,
    const llvm::SmallVector<Instrument *> &IVec) const {
  if (TargetIM)
    return TargetIM->getSchedClassID(MCII, MCI, IVec);
  return MCII.get(MCI.getOpcode()).getSchedClass();
}

} // namespace mca
} // namespace llvm
