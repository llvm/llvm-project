//===- InstrInfoEmitter.cpp - Generate a Instruction Set Desc. ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//


#include "CodeGenDAGPatterns.h"
#include "CodeGenSchedule.h"
#include "CodeGenTarget.h"
#include "TableGenBackends.h"
#include "SequenceToOffsetTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>
using namespace llvm;

namespace {
class InstrInfoEmitter {
  RecordKeeper &Records;
  CodeGenDAGPatterns CDP;
  const CodeGenSchedModels &SchedModels;

public:
  InstrInfoEmitter(RecordKeeper &R):
    Records(R), CDP(R), SchedModels(CDP.getTargetInfo().getSchedModels()) {}

  // run - Output the instruction set description.
  void run(raw_ostream &OS);

private:
  void emitEnums(raw_ostream &OS);

  typedef std::map<std::vector<std::string>, unsigned> OperandInfoMapTy;
  void emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                  Record *InstrInfo,
                  std::map<std::vector<Record*>, unsigned> &EL,
                  const OperandInfoMapTy &OpInfo,
                  raw_ostream &OS);

  // Operand information.
  void EmitOperandInfo(raw_ostream &OS, OperandInfoMapTy &OperandInfoIDs);
  std::vector<std::string> GetOperandInfo(const CodeGenInstruction &Inst);
};
} // End anonymous namespace

static void PrintDefList(const std::vector<Record*> &Uses,
                         unsigned Num, raw_ostream &OS) {
  OS << "static const uint16_t ImplicitList" << Num << "[] = { ";
  for (unsigned i = 0, e = Uses.size(); i != e; ++i)
    OS << getQualifiedName(Uses[i]) << ", ";
  OS << "0 };\n";
}

//===----------------------------------------------------------------------===//
// Operand Info Emission.
//===----------------------------------------------------------------------===//

std::vector<std::string>
InstrInfoEmitter::GetOperandInfo(const CodeGenInstruction &Inst) {
  std::vector<std::string> Result;

  for (unsigned i = 0, e = Inst.Operands.size(); i != e; ++i) {
    // Handle aggregate operands and normal operands the same way by expanding
    // either case into a list of operands for this op.
    std::vector<CGIOperandList::OperandInfo> OperandList;

    // This might be a multiple operand thing.  Targets like X86 have
    // registers in their multi-operand operands.  It may also be an anonymous
    // operand, which has a single operand, but no declared class for the
    // operand.
    DagInit *MIOI = Inst.Operands[i].MIOperandInfo;

    if (!MIOI || MIOI->getNumArgs() == 0) {
      // Single, anonymous, operand.
      OperandList.push_back(Inst.Operands[i]);
    } else {
      for (unsigned j = 0, e = Inst.Operands[i].MINumOperands; j != e; ++j) {
        OperandList.push_back(Inst.Operands[i]);

        Record *OpR = cast<DefInit>(MIOI->getArg(j))->getDef();
        OperandList.back().Rec = OpR;
      }
    }

    for (unsigned j = 0, e = OperandList.size(); j != e; ++j) {
      Record *OpR = OperandList[j].Rec;
      std::string Res;

      if (OpR->isSubClassOf("RegisterOperand"))
        OpR = OpR->getValueAsDef("RegClass");
      if (OpR->isSubClassOf("RegisterClass"))
        Res += getQualifiedName(OpR) + "RegClassID, ";
      else if (OpR->isSubClassOf("PointerLikeRegClass"))
        Res += utostr(OpR->getValueAsInt("RegClassKind")) + ", ";
      else
        // -1 means the operand does not have a fixed register class.
        Res += "-1, ";

      // Fill in applicable flags.
      Res += "0";

      // Ptr value whose register class is resolved via callback.
      if (OpR->isSubClassOf("PointerLikeRegClass"))
        Res += "|(1<<MCOI::LookupPtrRegClass)";

      // Predicate operands.  Check to see if the original unexpanded operand
      // was of type PredicateOperand.
      if (Inst.Operands[i].Rec->isSubClassOf("PredicateOperand"))
        Res += "|(1<<MCOI::Predicate)";

      // Optional def operands.  Check to see if the original unexpanded operand
      // was of type OptionalDefOperand.
      if (Inst.Operands[i].Rec->isSubClassOf("OptionalDefOperand"))
        Res += "|(1<<MCOI::OptionalDef)";

      // Fill in operand type.
      Res += ", MCOI::";
      assert(!Inst.Operands[i].OperandType.empty() && "Invalid operand type.");
      Res += Inst.Operands[i].OperandType;

      // Fill in constraint info.
      Res += ", ";

      const CGIOperandList::ConstraintInfo &Constraint =
        Inst.Operands[i].Constraints[j];
      if (Constraint.isNone())
        Res += "0";
      else if (Constraint.isEarlyClobber())
        Res += "(1 << MCOI::EARLY_CLOBBER)";
      else {
        assert(Constraint.isTied());
        Res += "((" + utostr(Constraint.getTiedOperand()) +
                    " << 16) | (1 << MCOI::TIED_TO))";
      }

      Result.push_back(Res);
    }
  }

  return Result;
}

void InstrInfoEmitter::EmitOperandInfo(raw_ostream &OS,
                                       OperandInfoMapTy &OperandInfoIDs) {
  // ID #0 is for no operand info.
  unsigned OperandListNum = 0;
  OperandInfoIDs[std::vector<std::string>()] = ++OperandListNum;

  OS << "\n";
  const CodeGenTarget &Target = CDP.getTargetInfo();
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(),
       E = Target.inst_end(); II != E; ++II) {
    std::vector<std::string> OperandInfo = GetOperandInfo(**II);
    unsigned &N = OperandInfoIDs[OperandInfo];
    if (N != 0) continue;

    N = ++OperandListNum;
    OS << "static const MCOperandInfo OperandInfo" << N << "[] = { ";
    for (unsigned i = 0, e = OperandInfo.size(); i != e; ++i)
      OS << "{ " << OperandInfo[i] << " }, ";
    OS << "};\n";
  }
}

//===----------------------------------------------------------------------===//
// Main Output.
//===----------------------------------------------------------------------===//

// run - Emit the main instruction description records for the target...
void InstrInfoEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Target Instruction Enum Values", OS);
  emitEnums(OS);

  emitSourceFileHeader("Target Instruction Descriptors", OS);

  OS << "\n#ifdef GET_INSTRINFO_MC_DESC\n";
  OS << "#undef GET_INSTRINFO_MC_DESC\n";

  OS << "namespace llvm {\n\n";

  CodeGenTarget &Target = CDP.getTargetInfo();
  const std::string &TargetName = Target.getName();
  Record *InstrInfo = Target.getInstructionSet();

  // Keep track of all of the def lists we have emitted already.
  std::map<std::vector<Record*>, unsigned> EmittedLists;
  unsigned ListNumber = 0;

  // Emit all of the instruction's implicit uses and defs.
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(),
         E = Target.inst_end(); II != E; ++II) {
    Record *Inst = (*II)->TheDef;
    std::vector<Record*> Uses = Inst->getValueAsListOfDefs("Uses");
    if (!Uses.empty()) {
      unsigned &IL = EmittedLists[Uses];
      if (!IL) PrintDefList(Uses, IL = ++ListNumber, OS);
    }
    std::vector<Record*> Defs = Inst->getValueAsListOfDefs("Defs");
    if (!Defs.empty()) {
      unsigned &IL = EmittedLists[Defs];
      if (!IL) PrintDefList(Defs, IL = ++ListNumber, OS);
    }
  }

  OperandInfoMapTy OperandInfoIDs;

  // Emit all of the operand info records.
  EmitOperandInfo(OS, OperandInfoIDs);

  // Emit all of the MCInstrDesc records in their ENUM ordering.
  //
  OS << "\nextern const MCInstrDesc " << TargetName << "Insts[] = {\n";
  const std::vector<const CodeGenInstruction*> &NumberedInstructions =
    Target.getInstructionsByEnumValue();

  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i)
    emitRecord(*NumberedInstructions[i], i, InstrInfo, EmittedLists,
               OperandInfoIDs, OS);
  OS << "};\n\n";

  // Build an array of instruction names
  SequenceToOffsetTable<std::string> InstrNames;
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    const CodeGenInstruction *Instr = NumberedInstructions[i];
    InstrNames.add(Instr->TheDef->getName());
  }

  InstrNames.layout();
  OS << "extern const char " << TargetName << "InstrNameData[] = {\n";
  InstrNames.emit(OS, printChar);
  OS << "};\n\n";

  OS << "extern const unsigned " << TargetName <<"InstrNameIndices[] = {";
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    if (i % 8 == 0)
      OS << "\n    ";
    const CodeGenInstruction *Instr = NumberedInstructions[i];
    OS << InstrNames.get(Instr->TheDef->getName()) << "U, ";
  }

  OS << "\n};\n\n";

  // MCInstrInfo initialization routine.
  OS << "static inline void Init" << TargetName
     << "MCInstrInfo(MCInstrInfo *II) {\n";
  OS << "  II->InitMCInstrInfo(" << TargetName << "Insts, "
     << TargetName << "InstrNameIndices, " << TargetName << "InstrNameData, "
     << NumberedInstructions.size() << ");\n}\n\n";

  OS << "} // End llvm namespace \n";

  OS << "#endif // GET_INSTRINFO_MC_DESC\n\n";

  // Create a TargetInstrInfo subclass to hide the MC layer initialization.
  OS << "\n#ifdef GET_INSTRINFO_HEADER\n";
  OS << "#undef GET_INSTRINFO_HEADER\n";

  std::string ClassName = TargetName + "GenInstrInfo";
  OS << "namespace llvm {\n";
  OS << "struct " << ClassName << " : public TargetInstrInfoImpl {\n"
     << "  explicit " << ClassName << "(int SO = -1, int DO = -1);\n"
     << "};\n";
  OS << "} // End llvm namespace \n";

  OS << "#endif // GET_INSTRINFO_HEADER\n\n";

  OS << "\n#ifdef GET_INSTRINFO_CTOR\n";
  OS << "#undef GET_INSTRINFO_CTOR\n";

  OS << "namespace llvm {\n";
  OS << "extern const MCInstrDesc " << TargetName << "Insts[];\n";
  OS << "extern const unsigned " << TargetName << "InstrNameIndices[];\n";
  OS << "extern const char " << TargetName << "InstrNameData[];\n";
  OS << ClassName << "::" << ClassName << "(int SO, int DO)\n"
     << "  : TargetInstrInfoImpl(SO, DO) {\n"
     << "  InitMCInstrInfo(" << TargetName << "Insts, "
     << TargetName << "InstrNameIndices, " << TargetName << "InstrNameData, "
     << NumberedInstructions.size() << ");\n}\n";
  OS << "} // End llvm namespace \n";

  OS << "#endif // GET_INSTRINFO_CTOR\n\n";
}

void InstrInfoEmitter::emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                                  Record *InstrInfo,
                         std::map<std::vector<Record*>, unsigned> &EmittedLists,
                                  const OperandInfoMapTy &OpInfo,
                                  raw_ostream &OS) {
  int MinOperands = 0;
  if (!Inst.Operands.empty())
    // Each logical operand can be multiple MI operands.
    MinOperands = Inst.Operands.back().MIOperandNo +
                  Inst.Operands.back().MINumOperands;

  OS << "  { ";
  OS << Num << ",\t" << MinOperands << ",\t"
     << Inst.Operands.NumDefs << ",\t"
     << SchedModels.getSchedClassIdx(Inst) << ",\t"
     << Inst.TheDef->getValueAsInt("Size") << ",\t0";

  // Emit all of the target indepedent flags...
  if (Inst.isPseudo)           OS << "|(1<<MCID::Pseudo)";
  if (Inst.isReturn)           OS << "|(1<<MCID::Return)";
  if (Inst.isBranch)           OS << "|(1<<MCID::Branch)";
  if (Inst.isIndirectBranch)   OS << "|(1<<MCID::IndirectBranch)";
  if (Inst.isCompare)          OS << "|(1<<MCID::Compare)";
  if (Inst.isMoveImm)          OS << "|(1<<MCID::MoveImm)";
  if (Inst.isBitcast)          OS << "|(1<<MCID::Bitcast)";
  if (Inst.isSelect)           OS << "|(1<<MCID::Select)";
  if (Inst.isBarrier)          OS << "|(1<<MCID::Barrier)";
  if (Inst.hasDelaySlot)       OS << "|(1<<MCID::DelaySlot)";
  if (Inst.isCall)             OS << "|(1<<MCID::Call)";
  if (Inst.canFoldAsLoad)      OS << "|(1<<MCID::FoldableAsLoad)";
  if (Inst.mayLoad)            OS << "|(1<<MCID::MayLoad)";
  if (Inst.mayStore)           OS << "|(1<<MCID::MayStore)";
  if (Inst.isPredicable)       OS << "|(1<<MCID::Predicable)";
  if (Inst.isConvertibleToThreeAddress) OS << "|(1<<MCID::ConvertibleTo3Addr)";
  if (Inst.isCommutable)       OS << "|(1<<MCID::Commutable)";
  if (Inst.isTerminator)       OS << "|(1<<MCID::Terminator)";
  if (Inst.isReMaterializable) OS << "|(1<<MCID::Rematerializable)";
  if (Inst.isNotDuplicable)    OS << "|(1<<MCID::NotDuplicable)";
  if (Inst.Operands.hasOptionalDef) OS << "|(1<<MCID::HasOptionalDef)";
  if (Inst.usesCustomInserter) OS << "|(1<<MCID::UsesCustomInserter)";
  if (Inst.hasPostISelHook)    OS << "|(1<<MCID::HasPostISelHook)";
  if (Inst.Operands.isVariadic)OS << "|(1<<MCID::Variadic)";
  if (Inst.hasSideEffects)     OS << "|(1<<MCID::UnmodeledSideEffects)";
  if (Inst.isAsCheapAsAMove)   OS << "|(1<<MCID::CheapAsAMove)";
  if (Inst.hasExtraSrcRegAllocReq) OS << "|(1<<MCID::ExtraSrcRegAllocReq)";
  if (Inst.hasExtraDefRegAllocReq) OS << "|(1<<MCID::ExtraDefRegAllocReq)";

  // Emit all of the target-specific flags...
  BitsInit *TSF = Inst.TheDef->getValueAsBitsInit("TSFlags");
  if (!TSF)
    PrintFatalError("no TSFlags?");
  uint64_t Value = 0;
  for (unsigned i = 0, e = TSF->getNumBits(); i != e; ++i) {
    if (BitInit *Bit = dyn_cast<BitInit>(TSF->getBit(i)))
      Value |= uint64_t(Bit->getValue()) << i;
    else
      PrintFatalError("Invalid TSFlags bit in " + Inst.TheDef->getName());
  }
  OS << ", 0x";
  OS.write_hex(Value);
  OS << "ULL, ";

  // Emit the implicit uses and defs lists...
  std::vector<Record*> UseList = Inst.TheDef->getValueAsListOfDefs("Uses");
  if (UseList.empty())
    OS << "NULL, ";
  else
    OS << "ImplicitList" << EmittedLists[UseList] << ", ";

  std::vector<Record*> DefList = Inst.TheDef->getValueAsListOfDefs("Defs");
  if (DefList.empty())
    OS << "NULL, ";
  else
    OS << "ImplicitList" << EmittedLists[DefList] << ", ";

  // Emit the operand info.
  std::vector<std::string> OperandInfo = GetOperandInfo(Inst);
  if (OperandInfo.empty())
    OS << "0";
  else
    OS << "OperandInfo" << OpInfo.find(OperandInfo)->second;

  OS << " },  // Inst #" << Num << " = " << Inst.TheDef->getName() << "\n";
}

// emitEnums - Print out enum values for all of the instructions.
void InstrInfoEmitter::emitEnums(raw_ostream &OS) {

  OS << "\n#ifdef GET_INSTRINFO_ENUM\n";
  OS << "#undef GET_INSTRINFO_ENUM\n";

  OS << "namespace llvm {\n\n";

  CodeGenTarget Target(Records);

  // We must emit the PHI opcode first...
  std::string Namespace = Target.getInstNamespace();

  if (Namespace.empty()) {
    fprintf(stderr, "No instructions defined!\n");
    exit(1);
  }

  const std::vector<const CodeGenInstruction*> &NumberedInstructions =
    Target.getInstructionsByEnumValue();

  OS << "namespace " << Namespace << " {\n";
  OS << "  enum {\n";
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    OS << "    " << NumberedInstructions[i]->TheDef->getName()
       << "\t= " << i << ",\n";
  }
  OS << "    INSTRUCTION_LIST_END = " << NumberedInstructions.size() << "\n";
  OS << "  };\n}\n";
  OS << "} // End llvm namespace \n";

  OS << "#endif // GET_INSTRINFO_ENUM\n\n";
}

namespace llvm {

void EmitInstrInfo(RecordKeeper &RK, raw_ostream &OS) {
  InstrInfoEmitter(RK).run(OS);
  EmitMapTable(RK, OS);
}

} // End llvm namespace
