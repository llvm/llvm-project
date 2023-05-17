//===- utils/TableGen/X86EVEX2VEXTablesEmitter.cpp - X86 backend-*- C++ -*-===//
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

class X86EVEX2VEXTablesEmitter {
  RecordKeeper &Records;
  CodeGenTarget Target;

  // Hold all non-masked & non-broadcasted EVEX encoded instructions
  std::vector<const CodeGenInstruction *> EVEXInsts;
  // Hold all VEX encoded instructions. Divided into groups with same opcodes
  // to make the search more efficient
  std::map<uint64_t, std::vector<const CodeGenInstruction *>> VEXInsts;

  typedef std::pair<const CodeGenInstruction *, const CodeGenInstruction *> Entry;
  typedef std::pair<StringRef, StringRef> Predicate;

  // Represent both compress tables
  std::vector<Entry> EVEX2VEX128;
  std::vector<Entry> EVEX2VEX256;
  // Represent predicates of VEX instructions.
  std::vector<Predicate> EVEX2VEXPredicates;

public:
  X86EVEX2VEXTablesEmitter(RecordKeeper &R) : Records(R), Target(R) {}

  // run - Output X86 EVEX2VEX tables.
  void run(raw_ostream &OS);

private:
  // Prints the given table as a C++ array of type
  // X86EvexToVexCompressTableEntry
  void printTable(const std::vector<Entry> &Table, raw_ostream &OS);
  // Prints function which checks target feature specific predicate.
  void printCheckPredicate(const std::vector<Predicate> &Predicates,
                           raw_ostream &OS);
};

void X86EVEX2VEXTablesEmitter::printTable(const std::vector<Entry> &Table,
                                          raw_ostream &OS) {
  StringRef Size = (Table == EVEX2VEX128) ? "128" : "256";

  OS << "// X86 EVEX encoded instructions that have a VEX " << Size
     << " encoding\n"
     << "// (table format: <EVEX opcode, VEX-" << Size << " opcode>).\n"
     << "static const X86EvexToVexCompressTableEntry X86EvexToVex" << Size
     << "CompressTable[] = {\n"
     << "  // EVEX scalar with corresponding VEX.\n";

  // Print all entries added to the table
  for (const auto &Pair : Table) {
    OS << "  { X86::" << Pair.first->TheDef->getName()
       << ", X86::" << Pair.second->TheDef->getName() << " },\n";
  }

  OS << "};\n\n";
}

void X86EVEX2VEXTablesEmitter::printCheckPredicate(
    const std::vector<Predicate> &Predicates, raw_ostream &OS) {
  OS << "static bool CheckVEXInstPredicate"
     << "(MachineInstr &MI, const X86Subtarget *Subtarget) {\n"
     << "  unsigned Opc = MI.getOpcode();\n"
     << "  switch (Opc) {\n"
     << "    default: return true;\n";
  for (const auto &Pair : Predicates)
    OS << "    case X86::" << Pair.first << ": return " << Pair.second << ";\n";
  OS << "  }\n"
     << "}\n\n";
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
    bool VEX_WIG  = VEXRI.IgnoresW;
    bool EVEX_WIG  = EVEXRI.IgnoresW;
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
    for (unsigned i = 0, e = EVEXInst->Operands.size(); i < e; i++) {
      Record *OpRec1 = EVEXInst->Operands[i].Rec;
      Record *OpRec2 = VEXInst->Operands[i].Rec;

      if (OpRec1 == OpRec2)
        continue;

      if (isRegisterOperand(OpRec1) && isRegisterOperand(OpRec2)) {
        if (getRegOperandSize(OpRec1) != getRegOperandSize(OpRec2))
          return false;
      } else if (isMemoryOperand(OpRec1) && isMemoryOperand(OpRec2)) {
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
};

void X86EVEX2VEXTablesEmitter::run(raw_ostream &OS) {
  auto getPredicates = [&](const CodeGenInstruction *Inst) {
    std::vector<Record *> PredicatesRecords =
        Inst->TheDef->getValueAsListOfDefs("Predicates");
    // Currently we only do AVX related checks and assume each instruction
    // has one and only one AVX related predicates.
    for (unsigned i = 0, e = PredicatesRecords.size(); i != e; ++i)
      if (PredicatesRecords[i]->getName().startswith("HasAVX"))
        return PredicatesRecords[i]->getValueAsString("CondString");
    llvm_unreachable(
        "Instruction with checkPredicate set must have one predicate!");
  };

  emitSourceFileHeader("X86 EVEX2VEX tables", OS);

  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    const Record *Def = Inst->TheDef;
    // Filter non-X86 instructions.
    if (!Def->isSubClassOf("X86Inst"))
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
  }

  for (const CodeGenInstruction *EVEXInst : EVEXInsts) {
    uint64_t Opcode = getValueFromBitsInit(EVEXInst->TheDef->
                                           getValueAsBitsInit("Opcode"));
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

    // Adding predicate check to EVEX2VEXPredicates table when needed.
    if (VEXInst->TheDef->getValueAsBit("checkVEXPredicate"))
      EVEX2VEXPredicates.push_back(
          std::make_pair(EVEXInst->TheDef->getName(), getPredicates(VEXInst)));
  }

  // Print both tables
  printTable(EVEX2VEX128, OS);
  printTable(EVEX2VEX256, OS);
  // Print CheckVEXInstPredicate function.
  printCheckPredicate(EVEX2VEXPredicates, OS);
}
} // namespace

static TableGen::Emitter::OptClass<X86EVEX2VEXTablesEmitter>
    X("gen-x86-EVEX2VEX-tables", "Generate X86 EVEX to VEX compress tables");
