//==- utils/TableGen/X86CompressEVEXTablesEmitter.cpp - X86 backend-*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This tablegen backend is responsible for emitting the X86 backend EVEX
/// compression tables.
///
//===----------------------------------------------------------------------===//

#include "CodeGenInstruction.h"
#include "CodeGenTarget.h"
#include "X86RecognizableInstr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <set>

using namespace llvm;
using namespace X86Disassembler;

namespace {

const std::map<StringRef, StringRef> ManualMap = {
#define ENTRY(OLD, NEW) {#OLD, #NEW},
#include "X86ManualCompressEVEXTables.def"
};
const std::set<StringRef> NoCompressSet = {
#define NOCOMP(INSN) #INSN,
#include "X86ManualCompressEVEXTables.def"
};

class X86CompressEVEXTablesEmitter {
  RecordKeeper &Records;
  CodeGenTarget Target;

  // Hold all pontentially compressible EVEX instructions
  std::vector<const CodeGenInstruction *> PreCompressionInsts;
  // Hold all compressed instructions. Divided into groups with same opcodes
  // to make the search more efficient
  std::map<uint64_t, std::vector<const CodeGenInstruction *>> CompressedInsts;

  typedef std::pair<const CodeGenInstruction *, const CodeGenInstruction *>
      Entry;
  typedef std::map<StringRef, std::vector<const CodeGenInstruction *>>
      PredicateInstMap;

  std::vector<Entry> Table;
  // Hold all compressed instructions that need to check predicate
  PredicateInstMap PredicateInsts;

public:
  X86CompressEVEXTablesEmitter(RecordKeeper &R) : Records(R), Target(R) {}

  // run - Output X86 EVEX compression tables.
  void run(raw_ostream &OS);

private:
  // Prints the given table as a C++ array of type X86CompressEVEXTableEntry
  void printTable(const std::vector<Entry> &Table, raw_ostream &OS);
  // Prints function which checks target feature for compressed instructions.
  void printCheckPredicate(const PredicateInstMap &PredicateInsts,
                           raw_ostream &OS);
};

void X86CompressEVEXTablesEmitter::printTable(const std::vector<Entry> &Table,
                                              raw_ostream &OS) {

  OS << "static const X86CompressEVEXTableEntry X86CompressEVEXTable[] = {\n";

  // Print all entries added to the table
  for (const auto &Pair : Table)
    OS << "  { X86::" << Pair.first->TheDef->getName()
       << ", X86::" << Pair.second->TheDef->getName() << " },\n";

  OS << "};\n\n";
}

void X86CompressEVEXTablesEmitter::printCheckPredicate(
    const PredicateInstMap &PredicateInsts, raw_ostream &OS) {

  OS << "static bool checkPredicate(unsigned Opc, const X86Subtarget "
        "*Subtarget) {\n"
     << "  switch (Opc) {\n"
     << "  default: return true;\n";
  for (const auto &[Key, Val] : PredicateInsts) {
    for (const auto &Inst : Val)
      OS << "  case X86::" << Inst->TheDef->getName() << ":\n";
    OS << "    return " << Key << ";\n";
  }

  OS << "  }\n";
  OS << "}\n\n";
}

static uint8_t byteFromBitsInit(const BitsInit *B) {
  unsigned N = B->getNumBits();
  assert(N <= 8 && "Field is too large for uint8_t!");

  uint8_t Value = 0;
  for (unsigned I = 0; I != N; ++I) {
    BitInit *Bit = cast<BitInit>(B->getBit(I));
    Value |= Bit->getValue() << I;
  }
  return Value;
}

class IsMatch {
  const CodeGenInstruction *OldInst;

public:
  IsMatch(const CodeGenInstruction *OldInst) : OldInst(OldInst) {}

  bool operator()(const CodeGenInstruction *NewInst) {
    RecognizableInstrBase NewRI(*NewInst);
    RecognizableInstrBase OldRI(*OldInst);

    // Return false if any of the following fields of does not match.
    if (std::tuple(OldRI.IsCodeGenOnly, OldRI.OpMap, NewRI.OpPrefix,
                   OldRI.HasVEX_4V, OldRI.HasVEX_L, OldRI.HasREX_W,
                   OldRI.Form) !=
        std::tuple(NewRI.IsCodeGenOnly, NewRI.OpMap, OldRI.OpPrefix,
                   NewRI.HasVEX_4V, NewRI.HasVEX_L, NewRI.HasREX_W, NewRI.Form))
      return false;

    for (unsigned I = 0, E = OldInst->Operands.size(); I < E; ++I) {
      Record *OldOpRec = OldInst->Operands[I].Rec;
      Record *NewOpRec = NewInst->Operands[I].Rec;

      if (OldOpRec == NewOpRec)
        continue;

      if (isRegisterOperand(OldOpRec) && isRegisterOperand(NewOpRec)) {
        if (getRegOperandSize(OldOpRec) != getRegOperandSize(NewOpRec))
          return false;
      } else if (isMemoryOperand(OldOpRec) && isMemoryOperand(NewOpRec)) {
        if (getMemOperandSize(OldOpRec) != getMemOperandSize(NewOpRec))
          return false;
      } else if (isImmediateOperand(OldOpRec) && isImmediateOperand(NewOpRec)) {
        if (OldOpRec->getValueAsDef("Type") != NewOpRec->getValueAsDef("Type"))
          return false;
      }
    }

    return true;
  }
};

void X86CompressEVEXTablesEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("X86 EVEX compression tables", OS);

  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    const Record *Rec = Inst->TheDef;
    StringRef Name = Rec->getName();
    // _REV instruction should not appear before encoding optimization
    if (!Rec->isSubClassOf("X86Inst") ||
        Rec->getValueAsBit("isAsmParserOnly") || Name.ends_with("_REV"))
      continue;

    // Promoted legacy instruction is in EVEX space, and has REX2-encoding
    // alternative. It's added due to HW design and never emitted by compiler.
    if (byteFromBitsInit(Rec->getValueAsBitsInit("OpMapBits")) ==
            X86Local::T_MAP4 &&
        byteFromBitsInit(Rec->getValueAsBitsInit("explicitOpPrefixBits")) ==
            X86Local::ExplicitEVEX)
      continue;

    if (NoCompressSet.find(Name) != NoCompressSet.end())
      continue;

    RecognizableInstrBase RI(*Inst);

    bool IsND = RI.OpMap == X86Local::T_MAP4 && RI.HasEVEX_B && RI.HasVEX_4V;
    // Add VEX encoded instructions to one of CompressedInsts vectors according
    // to it's opcode.
    if (RI.Encoding == X86Local::VEX)
      CompressedInsts[RI.Opcode].push_back(Inst);
    // Add relevant EVEX encoded instructions to PreCompressionInsts
    else if (RI.Encoding == X86Local::EVEX && !RI.HasEVEX_K && !RI.HasEVEX_L2 &&
             (!RI.HasEVEX_B || IsND))
      PreCompressionInsts.push_back(Inst);
  }

  for (const CodeGenInstruction *Inst : PreCompressionInsts) {
    const Record *Rec = Inst->TheDef;
    uint8_t Opcode = byteFromBitsInit(Rec->getValueAsBitsInit("Opcode"));
    StringRef Name = Rec->getName();
    const CodeGenInstruction *NewInst = nullptr;
    if (ManualMap.find(Name) != ManualMap.end()) {
      Record *NewRec = Records.getDef(ManualMap.at(Rec->getName()));
      assert(NewRec && "Instruction not found!");
      NewInst = &Target.getInstruction(NewRec);
    } else if (Name.ends_with("_EVEX")) {
      if (auto *NewRec = Records.getDef(Name.drop_back(5)))
        NewInst = &Target.getInstruction(NewRec);
    } else if (Name.ends_with("_ND")) {
      if (auto *NewRec = Records.getDef(Name.drop_back(3))) {
        auto &TempInst = Target.getInstruction(NewRec);
        if (isRegisterOperand(TempInst.Operands[0].Rec))
          NewInst = &TempInst;
      }
    } else {
      // For each pre-compression instruction look for a match in the
      // appropriate vector (instructions with the same opcode) using function
      // object IsMatch.
      auto Match = llvm::find_if(CompressedInsts[Opcode], IsMatch(Inst));
      if (Match != CompressedInsts[Opcode].end())
        NewInst = *Match;
    }

    if (!NewInst)
      continue;

    Table.push_back(std::pair(Inst, NewInst));
    auto Predicates = NewInst->TheDef->getValueAsListOfDefs("Predicates");
    auto It = llvm::find_if(Predicates, [](const Record *R) {
      StringRef Name = R->getName();
      return Name == "HasAVXNECONVERT" || Name == "HasAVXVNNI" ||
             Name == "HasAVXIFMA";
    });
    if (It != Predicates.end())
      PredicateInsts[(*It)->getValueAsString("CondString")].push_back(NewInst);
  }

  printTable(Table, OS);
  printCheckPredicate(PredicateInsts, OS);
}
} // namespace

static TableGen::Emitter::OptClass<X86CompressEVEXTablesEmitter>
    X("gen-x86-compress-evex-tables", "Generate X86 EVEX compression tables");
