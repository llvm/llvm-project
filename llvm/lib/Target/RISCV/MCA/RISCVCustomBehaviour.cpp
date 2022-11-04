//===------------------- RISCVCustomBehaviour.cpp ---------------*-C++ -* -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods from the RISCVCustomBehaviour class.
///
//===----------------------------------------------------------------------===//

#include "RISCVCustomBehaviour.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCVInstrInfo.h"
#include "TargetInfo/RISCVTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llvm-mca-riscv-custombehaviour"

// This brings in a table with primary key of
// base instruction opcode and lmul and maps
// to the opcode of the pseudo instruction.
namespace RISCVVInversePseudosTable {
using namespace llvm;
using namespace llvm::RISCV;

struct PseudoInfo {
  uint16_t Pseudo;
  uint16_t BaseInstr;
  uint8_t VLMul;
};

#define GET_RISCVVInversePseudosTable_IMPL
#define GET_RISCVVInversePseudosTable_DECL
#include "RISCVGenSearchableTables.inc"

} // end namespace RISCVVInversePseudosTable

namespace llvm {
namespace mca {

const llvm::StringRef RISCVLMULInstrument::DESC_NAME = "RISCV-LMUL";

bool RISCVLMULInstrument::isDataValid(llvm::StringRef Data) {
  // Return true if not one of the valid LMUL strings
  return StringSwitch<bool>(Data)
      .Cases("M1", "M2", "M4", "M8", "MF2", "MF4", "MF8", true)
      .Default(false);
}

uint8_t RISCVLMULInstrument::getLMUL() const {
  // assertion prevents us from needing llvm_unreachable in the StringSwitch
  // below
  assert(isDataValid(getData()) &&
         "Cannot get LMUL because invalid Data value");
  // These are the LMUL values that are used in RISCV tablegen
  return StringSwitch<uint8_t>(getData())
      .Case("M1", 0b000)
      .Case("M2", 0b001)
      .Case("M4", 0b010)
      .Case("M8", 0b011)
      .Case("MF2", 0b101)
      .Case("MF4", 0b110)
      .Case("MF8", 0b111);
}

bool RISCVInstrumentManager::supportsInstrumentType(
    llvm::StringRef Type) const {
  // Currently, only support for RISCVLMULInstrument type
  return Type == RISCVLMULInstrument::DESC_NAME;
}

SharedInstrument
RISCVInstrumentManager::createInstrument(llvm::StringRef Desc,
                                         llvm::StringRef Data) {
  if (Desc != RISCVLMULInstrument::DESC_NAME) {
    LLVM_DEBUG(dbgs() << "RVCB: Unknown instrumentation Desc: " << Desc
                      << '\n');
    return nullptr;
  }
  if (RISCVLMULInstrument::isDataValid(Data)) {
    LLVM_DEBUG(dbgs() << "RVCB: Bad data for instrument kind " << Desc << ": "
                      << Data << '\n');
    return nullptr;
  }
  return std::make_shared<RISCVLMULInstrument>(Data);
}

unsigned RISCVInstrumentManager::getSchedClassID(
    const MCInstrInfo &MCII, const MCInst &MCI,
    const llvm::SmallVector<SharedInstrument> &IVec) const {
  unsigned short Opcode = MCI.getOpcode();
  unsigned SchedClassID = MCII.get(Opcode).getSchedClass();

  for (const auto &I : IVec) {
    // Unknown Instrument kind
    if (I->getDesc() == RISCVLMULInstrument::DESC_NAME) {
      uint8_t LMUL = static_cast<RISCVLMULInstrument *>(I.get())->getLMUL();
      const RISCVVInversePseudosTable::PseudoInfo *RVV =
          RISCVVInversePseudosTable::getBaseInfo(Opcode, LMUL);
      // Not a RVV instr
      if (!RVV) {
        LLVM_DEBUG(
            dbgs()
            << "RVCB: Could not find PseudoInstruction for Opcode "
            << MCII.getName(Opcode) << ", LMUL=" << I->getData()
            << ". Ignoring instrumentation and using original SchedClassID="
            << SchedClassID << '\n');
        return SchedClassID;
      }

      // Override using pseudo
      LLVM_DEBUG(dbgs() << "RVCB: Found Pseudo Instruction for Opcode "
                        << MCII.getName(Opcode) << ", LMUL=" << I->getData()
                        << ". Overriding original SchedClassID=" << SchedClassID
                        << " with " << MCII.getName(RVV->Pseudo) << '\n');
      return MCII.get(RVV->Pseudo).getSchedClass();
    }
  }

  // Unknown Instrument kind
  LLVM_DEBUG(
      dbgs() << "RVCB: Did not use instrumentation to override Opcode.\n");
  return SchedClassID;
}

} // namespace mca
} // namespace llvm

using namespace llvm;
using namespace mca;

static InstrumentManager *
createRISCVInstrumentManager(const MCSubtargetInfo &STI,
                             const MCInstrInfo &MCII) {
  return new RISCVInstrumentManager(STI, MCII);
}

/// Extern function to initialize the targets for the RISCV backend
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRISCVTargetMCA() {
  TargetRegistry::RegisterInstrumentManager(getTheRISCV32Target(),
                                            createRISCVInstrumentManager);
  TargetRegistry::RegisterInstrumentManager(getTheRISCV64Target(),
                                            createRISCVInstrumentManager);
}
