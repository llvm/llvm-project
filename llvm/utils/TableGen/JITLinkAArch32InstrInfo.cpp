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
#include "CodeGenDAGPatterns.h"
#include "CodeGenInstruction.h"
#include "CodeGenSchedule.h"
#include "CodeGenTarget.h"
#include "PredicateExpander.h"
#include "SequenceToOffsetTable.h"
#include "SubtargetFeatureInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITLink/aarch32.h"
#include "llvm/Support/Debug.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "jitlink-aarch32-tblgen"

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::aarch32;

namespace {

// Any helper data structures can be defined here. Some backends use
// structs to collect information from the records.

Twine getEdgeKindName(Edge::Kind K) {
#define KIND_NAME_CASE(K)                                                      \
  case K:                                                                      \
    return #K;

  switch (K) {
    KIND_NAME_CASE(Data_Delta32)
    KIND_NAME_CASE(Data_Pointer32)
    KIND_NAME_CASE(Arm_Call)
    KIND_NAME_CASE(Arm_Jump24)
    KIND_NAME_CASE(Arm_MovwAbsNC)
    KIND_NAME_CASE(Arm_MovtAbs)
    KIND_NAME_CASE(Thumb_Call)
    KIND_NAME_CASE(Thumb_Jump24)
    KIND_NAME_CASE(Thumb_MovwAbsNC)
    KIND_NAME_CASE(Thumb_MovtAbs)
    KIND_NAME_CASE(Thumb_MovwPrelNC)
    KIND_NAME_CASE(Thumb_MovtPrel)
  default:
    return "";
  }
#undef KIND_NAME_CASE
}

static StringRef getInstrFromJITLinkEdgeKind(Edge::Kind Kind) {
  /// Translate from JITLink-internal edge kind to TableGen instruction names.
  switch (Kind) {
  case Arm_Call:
    return "";
  case aarch32::Arm_Jump24:
    return "";
  case aarch32::Arm_MovwAbsNC:
    return "MOVi16";
  case aarch32::Arm_MovtAbs:
    return "MOVTi16";
  case aarch32::Thumb_Call:
    return "";
  case aarch32::Thumb_Jump24:
    return "";
  case aarch32::Thumb_MovwAbsNC:
  case aarch32::Thumb_MovwPrelNC:
    return "t2MOVi16";
  case aarch32::Thumb_MovtAbs:
  case aarch32::Thumb_MovtPrel:
    return "t2MOVTi16";
  default:
    return "";
  }
}

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
      if (VarName == "imm") {
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

static void writeArmElement(raw_ostream &OS, Twine InfoName, uint32_t Value,
                            int Indentation = 2) {
  OS.indent(Indentation) << "static constexpr uint32_t " + InfoName + " = 0x";
  OS.write_hex(Value) << ";\n";
}

static void writeArmInfo(raw_ostream &OS, InstrInfo &II, Edge::Kind Kind) {
  OS << "template <> struct FixupInfo<" + getEdgeKindName(Kind) +
            "> : public FixupInfoArm {\n";
  writeArmElement(OS, "Opcode", II.Opcode);
  writeArmElement(OS, "OpcodeMask", II.OpcodeMask);
  writeArmElement(OS, "ImmMask", II.ImmMask);
  writeArmElement(OS, "RegMask", II.RegMask);
  OS << "};\n\n";
}

static void writeThumbElement(raw_ostream &OS, Twine InfoName, uint32_t Value,
                              int Indentation = 2) {
  OS.indent(Indentation) << "static constexpr HalfWords " + InfoName + " {0x";
  OS.write_hex(Value >> 16) << ", 0x";
  OS.write_hex(Value & 0x0000FFFF) << "};\n";
}

static void writeThumbInfo(raw_ostream &OS, InstrInfo &II, Edge::Kind Kind) {
  OS << "template <> struct FixupInfo<" + getEdgeKindName(Kind) +
            "> : public FixupInfoThumb {\n";
  writeThumbElement(OS, "Opcode", II.Opcode);
  writeThumbElement(OS, "OpcodeMask", II.OpcodeMask);
  writeThumbElement(OS, "ImmMask", II.ImmMask);
  writeThumbElement(OS, "RegMask", II.RegMask);
  OS << "};\n\n";
}

class JITLinkAArch32Emitter {
private:
  RecordKeeper &Records;

public:
  JITLinkAArch32Emitter(RecordKeeper &RK) : Records(RK) {}

  void run(raw_ostream &OS);
}; // emitter class

} // anonymous namespace

void JITLinkAArch32Emitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Skeleton data structures", OS);
  OS << "using namespace llvm::jitlink::aarch32;\n\n";
  OS << "namespace llvm {\n";
  OS << "namespace jitlink {\n";
  OS << "namespace aarch32 {\n";
  // const auto &Instructions = Records.getAllDerivedDefinitions("Instruction");
  for (Edge::Kind JITLinkEdgeKind = aarch32::FirstArmRelocation;
       JITLinkEdgeKind <= aarch32::LastThumbRelocation; JITLinkEdgeKind += 1) {
    auto InstName = getInstrFromJITLinkEdgeKind(JITLinkEdgeKind);
    if (InstName.empty())
      continue;
    auto *InstRecord = Records.getDef(InstName);
    auto *InstBits = InstRecord->getValueAsBitsInit("Inst");
    InstrInfo II;
    extractBits(*InstBits, II);
    if (JITLinkEdgeKind > LastArmRelocation)
      writeThumbInfo(OS, II, JITLinkEdgeKind);
    else
      writeArmInfo(OS, II, JITLinkEdgeKind);
  }

  OS << "} //aarch32\n";
  OS << "} //jitlink\n";
  OS << "} //llvm\n";
}

static TableGen::Emitter::OptClass<JITLinkAArch32Emitter>
    X("gen-jitlink-aarch32-instr-info",
      "Generate JITLink AArch32 Instruction Information");
