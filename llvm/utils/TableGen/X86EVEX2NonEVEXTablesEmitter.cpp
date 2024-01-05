//=- utils/TableGen/X86EVEX2NonEVEXTablesEmitter.cpp - X86 backend-*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This tablegen backend is responsible for emitting the X86 backend EVEX2VEX
/// compression tables.
///
//===----------------------------------------------------------------------===//

#include "CodeGenInstruction.h"
#include "CodeGenTarget.h"
#include "X86RecognizableInstr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace X86Disassembler;

namespace {

const std::map<StringRef, StringRef> ManualMap = {
#define EVEXENTRY(EVEX, NonEVEXInstStr) {#EVEX, #NonEVEXInstStr},
#include "X86ManualEVEXCompressTables.def"
};

class X86EVEX2NonEVEXTablesEmitter {
  RecordKeeper &Records;
  CodeGenTarget Target;

  // Hold all non-masked & non-broadcasted EVEX encoded instructions
  std::vector<const CodeGenInstruction *> EVEXInsts;
  // Hold all VEX encoded instructions. Divided into groups with same opcodes
  // to make the search more efficient
  std::map<uint64_t, std::vector<const CodeGenInstruction *>> VEXInsts;

  typedef std::pair<const CodeGenInstruction *, const CodeGenInstruction *>
      Entry;

  // Represent both compress tables
  std::vector<Entry> EVEX2VEX128;
  std::vector<Entry> EVEX2VEX256;

  // Hold all possibly compressed APX instructions, including only ND and EGPR
  // instruction so far
  std::vector<const CodeGenInstruction *> APXInsts;
  // Hold all X86 instructions. Divided into groups with same opcodes
  // to make the search more efficient
  std::map<uint64_t, std::vector<const CodeGenInstruction *>> LegacyInsts;
  // Represent EVEX to Legacy compress tables.
  std::vector<Entry> EVEX2LegacyTable;

public:
  X86EVEX2NonEVEXTablesEmitter(RecordKeeper &R) : Records(R), Target(R) {}

  // run - Output X86 EVEX2VEX tables.
  void run(raw_ostream &OS);

private:
  // Prints the given table as a C++ array of type
  // X86EvexToNonEvexCompressTableEntry
  void printTable(const std::vector<Entry> &Table, raw_ostream &OS);
  // X86EVEXToLegacyCompressTableEntry
  void printEVEX2LegacyTable(const std::vector<Entry> &Table, raw_ostream &OS);
  void addManualEntry(const CodeGenInstruction *EVEXInstr,
                      const CodeGenInstruction *LegacyInstr,
                      const char *TableName);
};

void X86EVEX2NonEVEXTablesEmitter::printTable(const std::vector<Entry> &Table,
                                              raw_ostream &OS) {

  StringRef TargetEnc;
  StringRef TableName;
  StringRef Size;
  if (Table == EVEX2LegacyTable) {
    TargetEnc = "Legacy";
    TableName = "X86EvexToLegacy";
  } else {
    TargetEnc = "VEX";
    TableName = "X86EvexToVex";
    Size = (Table == EVEX2VEX128) ? "128" : "256";
  }

  OS << "// X86 EVEX encoded instructions that have a " << TargetEnc << " "
     << Size << " encoding\n"
     << "// (table format: <EVEX opcode, " << TargetEnc << Size
     << " opcode>).\n"
     << "static const X86EvexToNonEvexCompressTableEntry " << TableName << Size
     << "CompressTable[] = {\n"
     << "  // EVEX scalar with corresponding " << TargetEnc << ".\n";

  // Print all entries added to the table
  for (const auto &Pair : Table) {
    OS << "  { X86::" << Pair.first->TheDef->getName()
       << ", X86::" << Pair.second->TheDef->getName() << " },\n";
  }

  OS << "};\n\n";
}

// Return true if the 2 BitsInits are equal
// Calculates the integer value residing BitsInit object
static inline uint64_t getValueFromBitsInit(const BitsInit *B) {
  uint64_t Value = 0;
  for (unsigned i = 0, e = B->getNumBits(); i != e; ++i) {
    if (BitInit *Bit = dyn_cast<BitInit>(B->getBit(i)))
      Value |= uint64_t(Bit->getValue()) << i;
    else
      PrintFatalError("Invalid VectSize bit");
  }
  return Value;
}

static bool checkMatchable(const CodeGenInstruction *EVEXInst,
                           const CodeGenInstruction *NonEVEXInst) {
  for (unsigned I = 0, E = NonEVEXInst->Operands.size(); I < E; I++) {
    Record *OpRec1 = EVEXInst->Operands[I].Rec;
    Record *OpRec2 = NonEVEXInst->Operands[I].Rec;

    if (OpRec1 == OpRec2)
      continue;

    if (isRegisterOperand(OpRec1) && isRegisterOperand(OpRec2)) {
      if (getRegOperandSize(OpRec1) != getRegOperandSize(OpRec2))
        return false;
    } else if (isMemoryOperand(OpRec1) && isMemoryOperand(OpRec2)) {
      if (getMemOperandSize(OpRec1) != getMemOperandSize(OpRec2))
        return false;
    } else if (isImmediateOperand(OpRec1) && isImmediateOperand(OpRec2)) {
      if (OpRec1->getValueAsDef("Type") != OpRec2->getValueAsDef("Type")) {
        return false;
      }
    } else
      return false;
  }
  return true;
}

// Function object - Operator() returns true if the given VEX instruction
// matches the EVEX instruction of this object.
class IsMatch {
  const CodeGenInstruction *EVEXInst;

public:
  IsMatch(const CodeGenInstruction *EVEXInst) : EVEXInst(EVEXInst) {}

  bool operator()(const CodeGenInstruction *VEXInst) {
    RecognizableInstrBase VEXRI(*VEXInst);
    RecognizableInstrBase EVEXRI(*EVEXInst);
    bool VEX_W = VEXRI.HasREX_W;
    bool EVEX_W = EVEXRI.HasREX_W;
    bool VEX_WIG = VEXRI.IgnoresW;
    bool EVEX_WIG = EVEXRI.IgnoresW;
    bool EVEX_W1_VEX_W0 = EVEXInst->TheDef->getValueAsBit("EVEX_W1_VEX_W0");

    if (VEXRI.IsCodeGenOnly != EVEXRI.IsCodeGenOnly ||
        // VEX/EVEX fields
        VEXRI.OpPrefix != EVEXRI.OpPrefix || VEXRI.OpMap != EVEXRI.OpMap ||
        VEXRI.HasVEX_4V != EVEXRI.HasVEX_4V ||
        VEXRI.HasVEX_L != EVEXRI.HasVEX_L ||
        // Match is allowed if either is VEX_WIG, or they match, or EVEX
        // is VEX_W1X and VEX is VEX_W0.
        (!(VEX_WIG || (!EVEX_WIG && EVEX_W == VEX_W) ||
           (EVEX_W1_VEX_W0 && EVEX_W && !VEX_W))) ||
        // Instruction's format
        VEXRI.Form != EVEXRI.Form)
      return false;

    // This is needed for instructions with intrinsic version (_Int).
    // Where the only difference is the size of the operands.
    // For example: VUCOMISDZrm and Int_VUCOMISDrm
    // Also for instructions that their EVEX version was upgraded to work with
    // k-registers. For example VPCMPEQBrm (xmm output register) and
    // VPCMPEQBZ128rm (k register output register).
    return checkMatchable(EVEXInst, VEXInst);
  }
};

class IsMatchAPX {
  const CodeGenInstruction *EVEXInst;

public:
  IsMatchAPX(const CodeGenInstruction *EVEXInst) : EVEXInst(EVEXInst) {}

  bool operator()(const CodeGenInstruction *LegacyInst) {
    Record *RecEVEX = EVEXInst->TheDef;
    Record *RecLegacy = LegacyInst->TheDef;
    if (RecLegacy->getValueAsDef("OpSize")->getName() == "OpSize16" &&
        RecEVEX->getValueAsDef("OpPrefix")->getName() != "PD")
      return false;

    if (RecLegacy->getValueAsDef("OpSize")->getName() == "OpSize32" &&
        RecEVEX->getValueAsDef("OpPrefix")->getName() != "PS")
      return false;

    if (RecEVEX->getValueAsBit("hasREX_W") !=
        RecLegacy->getValueAsBit("hasREX_W"))
      return false;

    if (RecLegacy->getValueAsDef("Form") != RecEVEX->getValueAsDef("Form"))
      return false;

    if (RecLegacy->getValueAsBit("isCodeGenOnly") !=
        RecEVEX->getValueAsBit("isCodeGenOnly"))
      return false;

    return checkMatchable(EVEXInst, LegacyInst);
  }
};

void X86EVEX2NonEVEXTablesEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("X86 EVEX2VEX tables", OS);

  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    const Record *Def = Inst->TheDef;
    // Filter non-X86 instructions.
    if (!Def->isSubClassOf("X86Inst"))
      continue;
    // _REV instruction should not appear before encoding optimization
    if (Def->getName().ends_with("_REV"))
      continue;
    RecognizableInstrBase RI(*Inst);

    // Add VEX encoded instructions to one of VEXInsts vectors according to
    // it's opcode.
    if (RI.Encoding == X86Local::VEX)
      VEXInsts[RI.Opcode].push_back(Inst);
    // Add relevant EVEX encoded instructions to EVEXInsts
    else if (RI.Encoding == X86Local::EVEX && !RI.HasEVEX_K && !RI.HasEVEX_B &&
             !RI.HasEVEX_L2 && !Def->getValueAsBit("notEVEX2VEXConvertible"))
      EVEXInsts.push_back(Inst);

    if (RI.Encoding == X86Local::EVEX && RI.OpMap == X86Local::T_MAP4 &&
        !RI.HasEVEX_NF &&
        !getValueFromBitsInit(
            Def->getValueAsBitsInit("explicitOpPrefixBits"))) {
      APXInsts.push_back(Inst);
    } else if (Inst->TheDef->getValueAsDef("OpEnc")->getName() == "EncNormal") {
      uint64_t Opcode =
          getValueFromBitsInit(Inst->TheDef->getValueAsBitsInit("Opcode"));
      LegacyInsts[Opcode].push_back(Inst);
    }
  }

  for (const CodeGenInstruction *EVEXInst : EVEXInsts) {
    uint64_t Opcode =
        getValueFromBitsInit(EVEXInst->TheDef->getValueAsBitsInit("Opcode"));
    // For each EVEX instruction look for a VEX match in the appropriate vector
    // (instructions with the same opcode) using function object IsMatch.
    // Allow EVEX2VEXOverride to explicitly specify a match.
    const CodeGenInstruction *VEXInst = nullptr;
    if (!EVEXInst->TheDef->isValueUnset("EVEX2VEXOverride")) {
      StringRef AltInstStr =
          EVEXInst->TheDef->getValueAsString("EVEX2VEXOverride");
      Record *AltInstRec = Records.getDef(AltInstStr);
      assert(AltInstRec && "EVEX2VEXOverride instruction not found!");
      VEXInst = &Target.getInstruction(AltInstRec);
    } else {
      auto Match = llvm::find_if(VEXInsts[Opcode], IsMatch(EVEXInst));
      if (Match != VEXInsts[Opcode].end())
        VEXInst = *Match;
    }

    if (!VEXInst)
      continue;

    // In case a match is found add new entry to the appropriate table
    if (EVEXInst->TheDef->getValueAsBit("hasVEX_L"))
      EVEX2VEX256.push_back(std::make_pair(EVEXInst, VEXInst)); // {0,1}
    else
      EVEX2VEX128.push_back(std::make_pair(EVEXInst, VEXInst)); // {0,0}
  }

  // Print both tables
  printTable(EVEX2VEX128, OS);
  printTable(EVEX2VEX256, OS);

  for (const CodeGenInstruction *EVEXInst : APXInsts) {
    // REV instrs should not appear before encoding optimization.
    if (EVEXInst->TheDef->getName().ends_with("_REV"))
      continue;
    const CodeGenInstruction *LegacyInst = nullptr;
    if (ManualMap.count(EVEXInst->TheDef->getName())) {
      auto NonEVEXInstStr =
          ManualMap.at(StringRef(EVEXInst->TheDef->getName()));
      Record *LegacyRec = Records.getDef(NonEVEXInstStr);
      LegacyInst = &(Target.getInstruction(LegacyRec));
    } else {
      uint64_t Opcode =
          getValueFromBitsInit(EVEXInst->TheDef->getValueAsBitsInit("Opcode"));
      auto Match = llvm::find_if(LegacyInsts[Opcode], IsMatchAPX(EVEXInst));
      if (Match != LegacyInsts[Opcode].end())
        LegacyInst = *Match;
    }
    if (LegacyInst) {
      if (!EVEXInst->TheDef->getValueAsBit("hasEVEX_B"))
        EVEX2LegacyTable.push_back(std::make_pair(EVEXInst, LegacyInst));
    }
  }
  printTable(EVEX2LegacyTable, OS);
}
} // namespace

static TableGen::Emitter::OptClass<X86EVEX2NonEVEXTablesEmitter>
    X("gen-x86-EVEX2NonEVEX-tables",
      "Generate X86 EVEX to NonEVEX compress tables");
