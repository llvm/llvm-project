//===- JITLinkAArch32InstrInfo.cpp - JITLink AArch32 TableGen backend -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Tablegen backend emits instruction encodings of AArch32 for JITLink.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/Debug.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "jitlink-instr-info"

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

using namespace llvm;

namespace {

struct InstrInfo {
  uint32_t Opcode = 0;
  uint32_t OpcodeMask = 0;
  uint32_t ImmMask = 0;
  uint32_t RegMask = 0;
};

static void extractBits(BitsInit &InstBits, InstrInfo &II) {
  for (unsigned i = 0; i < InstBits.getNumBits(); ++i) {
    Init *Bit = InstBits.getBit(i);

    if (auto *VarBit = dyn_cast<VarBitInit>(Bit)) {
      // Check if the VarBit is for 'imm' or 'Rd'
      std::string VarName = VarBit->getBitVar()->getAsUnquotedString();
      if (VarName == "imm" || VarName == "func") {
        II.ImmMask |= 1 << i;
      } else if (VarName == "Rd") {
        II.RegMask |= 1 << i;
      }
    } else if (auto *TheBit = dyn_cast<BitInit>(Bit)) {
      II.OpcodeMask |= 1 << i;
      if (TheBit->getValue()) {
        II.Opcode |= 1 << i;
      }
    }
  }

  assert((II.OpcodeMask & II.ImmMask & II.RegMask) == 0 &&
         "Masks have intersecting bits");
}

static void writeInstrInfo(raw_ostream &OS, const InstrInfo &II,
                           const std::string &InstName) {
  OS << "GET_INSTR(" << InstName << ", 0x";
  OS.write_hex(II.Opcode) << ", 0x";
  OS.write_hex(II.OpcodeMask) << ", 0x";
  OS.write_hex(II.ImmMask) << ", 0x";
  OS.write_hex(II.RegMask) << ")\n";
}

class JITLinkEmitter {
private:
  RecordKeeper &Records;

public:
  JITLinkEmitter(RecordKeeper &RK) : Records(RK) {}

  void run(raw_ostream &OS);
};

void JITLinkEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Instruction Encoding Information", OS);

  OS << "#ifdef GET_INSTR // (Opc, Opc_Mask, Imm_Mask, Reg_Mask)\n";
  auto RecordsList = Records.getAllDerivedDefinitions("Instruction");
  for (auto *InstRecord : RecordsList) {
    if (InstRecord->getValueAsBit("isPseudo"))
      continue;
    LLVM_DEBUG(dbgs() << "Processing " << InstRecord->getNameInitAsString()
                      << "\n");
    if (InstRecord->getValueAsBit("isMoveImm") ||
        InstRecord->getValueAsBit("isCall") ||
        // FIXME movt for ARM and Thumb2 do not have their isMovImm flags set
        //       so we add these conditionals
        InstRecord->getNameInitAsString() == "MOVTi16" ||
        InstRecord->getNameInitAsString() == "t2MOVTi16") {
      LLVM_DEBUG(for (const auto &Val
                      : InstRecord->getValues()) {
        dbgs() << "Field: " << Val.getNameInitAsString() << " = "
               << Val.getValue()->getAsUnquotedString() << "\n";
      });
      auto *InstBits = InstRecord->getValueAsBitsInit("Inst");
      InstrInfo II;
      extractBits(*InstBits, II);
      writeInstrInfo(OS, II, InstRecord->getNameInitAsString());
    }
  }

  OS << "#endif\n";
}

static TableGen::Emitter::OptClass<JITLinkEmitter>
    X("gen-jitlink-aarch32-instr-info",
      "Generate JITLink Instruction Information");
} // namespace
