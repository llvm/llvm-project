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

  std::vector<Entry> Table;

public:
  X86CompressEVEXTablesEmitter(RecordKeeper &R) : Records(R), Target(R) {}

  // run - Output X86 EVEX compression tables.
  void run(raw_ostream &OS);

private:
  // Prints the given table as a C++ array of type X86CompressEVEXTableEntry
  void printTable(const std::vector<Entry> &Table, raw_ostream &OS);
};

void X86CompressEVEXTablesEmitter::printTable(const std::vector<Entry> &Table,
                                              raw_ostream &OS) {

  OS << "static const X86CompressEVEXTableEntry X86CompressEVEXTable[] = { \n";

  // Print all entries added to the table
  for (const auto &Pair : Table)
    OS << "  { X86::" << Pair.first->TheDef->getName()
       << ", X86::" << Pair.second->TheDef->getName() << " },\n";

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

class IsMatch {
  const CodeGenInstruction *OldInst;

public:
  IsMatch(const CodeGenInstruction *OldInst) : OldInst(OldInst) {}

  bool operator()(const CodeGenInstruction *NewInst) {
    RecognizableInstrBase NewRI(*NewInst);
    RecognizableInstrBase OldRI(*OldInst);

    // Return false if any of the following fields of does not match.
    if (std::make_tuple(OldRI.IsCodeGenOnly, OldRI.OpMap, NewRI.OpPrefix,
                        OldRI.HasVEX_4V, OldRI.HasVEX_L, OldRI.HasREX_W,
                        OldRI.Form) !=
        std::make_tuple(NewRI.IsCodeGenOnly, NewRI.OpMap, OldRI.OpPrefix,
                        NewRI.HasVEX_4V, NewRI.HasVEX_L, NewRI.HasREX_W,
                        NewRI.Form))
      return false;

    // This is needed for instructions with intrinsic version (_Int).
    // Where the only difference is the size of the operands.
    // For example: VUCOMISDZrm and Int_VUCOMISDrm
    // Also for instructions that their EVEX version was upgraded to work with
    // k-registers. For example VPCMPEQBrm (xmm output register) and
    // VPCMPEQBZ128rm (k register output register).
    for (unsigned i = 0, e = OldInst->Operands.size(); i < e; i++) {
      Record *OpRec1 = OldInst->Operands[i].Rec;
      Record *OpRec2 = NewInst->Operands[i].Rec;

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

void X86CompressEVEXTablesEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("X86 EVEX compression tables", OS);

  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    const Record *Rec = Inst->TheDef;
    // _REV instruction should not appear before encoding optimization
    if (!Rec->isSubClassOf("X86Inst") || Rec->getName().ends_with("_REV"))
      continue;

    if (NoCompressSet.find(Rec->getName()) != NoCompressSet.end())
      continue;

    RecognizableInstrBase RI(*Inst);

    // Add VEX encoded instructions to one of CompressedInsts vectors according
    // to it's opcode.
    if (RI.Encoding == X86Local::VEX)
      CompressedInsts[RI.Opcode].push_back(Inst);
    // Add relevant EVEX encoded instructions to PreCompressionInsts
    else if (RI.Encoding == X86Local::EVEX && !RI.HasEVEX_K && !RI.HasEVEX_B &&
             !RI.HasEVEX_L2)
      PreCompressionInsts.push_back(Inst);
  }

  for (const CodeGenInstruction *Inst : PreCompressionInsts) {
    const Record *Rec = Inst->TheDef;
    uint64_t Opcode =
        getValueFromBitsInit(Inst->TheDef->getValueAsBitsInit("Opcode"));
    const CodeGenInstruction *NewInst = nullptr;
    if (ManualMap.find(Rec->getName()) != ManualMap.end()) {
      Record *NewRec = Records.getDef(ManualMap.at(Rec->getName()));
      assert(NewRec && "Instruction not found!");
      NewInst = &Target.getInstruction(NewRec);
    } else {
      // For each pre-compression instruction look for a match in the appropriate
      // vector (instructions with the same opcode) using function object
      // IsMatch.
      auto Match = llvm::find_if(CompressedInsts[Opcode], IsMatch(Inst));
      if (Match != CompressedInsts[Opcode].end())
        NewInst = *Match;
    }

    if (!NewInst)
      continue;

    Table.push_back(std::make_pair(Inst, NewInst));
  }

  printTable(Table, OS);
}
} // namespace

static TableGen::Emitter::OptClass<X86CompressEVEXTablesEmitter>
    X("gen-x86-compress-evex-tables", "Generate X86 EVEX compression tables");
