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

#include "InstrInfoEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include <algorithm>
#include <iostream>
using namespace llvm;

static void PrintDefList(const std::vector<Record*> &Uses,
                         unsigned Num, std::ostream &OS) {
  OS << "static const unsigned ImplicitList" << Num << "[] = { ";
  for (unsigned i = 0, e = Uses.size(); i != e; ++i)
    OS << getQualifiedName(Uses[i]) << ", ";
  OS << "0 };\n";
}

static void PrintBarriers(std::vector<Record*> &Barriers,
                          unsigned Num, std::ostream &OS) {
  OS << "static const TargetRegisterClass* Barriers" << Num << "[] = { ";
  for (unsigned i = 0, e = Barriers.size(); i != e; ++i)
    OS << "&" << getQualifiedName(Barriers[i]) << "RegClass, ";
  OS << "NULL };\n";
}

//===----------------------------------------------------------------------===//
// Instruction Itinerary Information.
//===----------------------------------------------------------------------===//

struct RecordNameComparator {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getName() < Rec2->getName();
  }
};

void InstrInfoEmitter::GatherItinClasses() {
  std::vector<Record*> DefList =
  Records.getAllDerivedDefinitions("InstrItinClass");
  std::sort(DefList.begin(), DefList.end(), RecordNameComparator());
  
  for (unsigned i = 0, N = DefList.size(); i < N; i++)
    ItinClassMap[DefList[i]->getName()] = i;
}  

unsigned InstrInfoEmitter::getItinClassNumber(const Record *InstRec) {
  return ItinClassMap[InstRec->getValueAsDef("Itinerary")->getName()];
}

//===----------------------------------------------------------------------===//
// Operand Info Emission.
//===----------------------------------------------------------------------===//

std::vector<std::string>
InstrInfoEmitter::GetOperandInfo(const CodeGenInstruction &Inst) {
  std::vector<std::string> Result;
  
  for (unsigned i = 0, e = Inst.OperandList.size(); i != e; ++i) {
    // Handle aggregate operands and normal operands the same way by expanding
    // either case into a list of operands for this op.
    std::vector<CodeGenInstruction::OperandInfo> OperandList;

    // This might be a multiple operand thing.  Targets like X86 have
    // registers in their multi-operand operands.  It may also be an anonymous
    // operand, which has a single operand, but no declared class for the
    // operand.
    DagInit *MIOI = Inst.OperandList[i].MIOperandInfo;
    
    if (!MIOI || MIOI->getNumArgs() == 0) {
      // Single, anonymous, operand.
      OperandList.push_back(Inst.OperandList[i]);
    } else {
      for (unsigned j = 0, e = Inst.OperandList[i].MINumOperands; j != e; ++j) {
        OperandList.push_back(Inst.OperandList[i]);

        Record *OpR = dynamic_cast<DefInit*>(MIOI->getArg(j))->getDef();
        OperandList.back().Rec = OpR;
      }
    }

    for (unsigned j = 0, e = OperandList.size(); j != e; ++j) {
      Record *OpR = OperandList[j].Rec;
      std::string Res;
      
      if (OpR->isSubClassOf("RegisterClass"))
        Res += getQualifiedName(OpR) + "RegClassID, ";
      else
        Res += "0, ";
      // Fill in applicable flags.
      Res += "0";
        
      // Ptr value whose register class is resolved via callback.
      if (OpR->getName() == "ptr_rc")
        Res += "|(1<<TOI::LookupPtrRegClass)";

      // Predicate operands.  Check to see if the original unexpanded operand
      // was of type PredicateOperand.
      if (Inst.OperandList[i].Rec->isSubClassOf("PredicateOperand"))
        Res += "|(1<<TOI::Predicate)";
        
      // Optional def operands.  Check to see if the original unexpanded operand
      // was of type OptionalDefOperand.
      if (Inst.OperandList[i].Rec->isSubClassOf("OptionalDefOperand"))
        Res += "|(1<<TOI::OptionalDef)";

      // Fill in constraint info.
      Res += ", " + Inst.OperandList[i].Constraints[j];
      Result.push_back(Res);
    }
  }

  return Result;
}

void InstrInfoEmitter::EmitOperandInfo(std::ostream &OS, 
                                       OperandInfoMapTy &OperandInfoIDs) {
  // ID #0 is for no operand info.
  unsigned OperandListNum = 0;
  OperandInfoIDs[std::vector<std::string>()] = ++OperandListNum;
  
  OS << "\n";
  const CodeGenTarget &Target = CDP.getTargetInfo();
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(),
       E = Target.inst_end(); II != E; ++II) {
    std::vector<std::string> OperandInfo = GetOperandInfo(II->second);
    unsigned &N = OperandInfoIDs[OperandInfo];
    if (N != 0) continue;
    
    N = ++OperandListNum;
    OS << "static const TargetOperandInfo OperandInfo" << N << "[] = { ";
    for (unsigned i = 0, e = OperandInfo.size(); i != e; ++i)
      OS << "{ " << OperandInfo[i] << " }, ";
    OS << "};\n";
  }
}

void InstrInfoEmitter::DetectRegisterClassBarriers(std::vector<Record*> &Defs,
                                  const std::vector<CodeGenRegisterClass> &RCs,
                                  std::vector<Record*> &Barriers) {
  std::set<Record*> DefSet;
  unsigned NumDefs = Defs.size();
  for (unsigned i = 0; i < NumDefs; ++i)
    DefSet.insert(Defs[i]);

  for (unsigned i = 0, e = RCs.size(); i != e; ++i) {
    const CodeGenRegisterClass &RC = RCs[i];
    unsigned NumRegs = RC.Elements.size();
    if (NumRegs > NumDefs)
      continue; // Can't possibly clobber this RC.

    bool Clobber = true;
    for (unsigned j = 0; j < NumRegs; ++j) {
      Record *Reg = RC.Elements[j];
      if (!DefSet.count(Reg)) {
        Clobber = false;
        break;
      }
    }
    if (Clobber)
      Barriers.push_back(RC.TheDef);
  }
}

//===----------------------------------------------------------------------===//
// Main Output.
//===----------------------------------------------------------------------===//

// run - Emit the main instruction description records for the target...
void InstrInfoEmitter::run(std::ostream &OS) {
  GatherItinClasses();

  EmitSourceFileHeader("Target Instruction Descriptors", OS);
  OS << "namespace llvm {\n\n";

  CodeGenTarget &Target = CDP.getTargetInfo();
  const std::string &TargetName = Target.getName();
  Record *InstrInfo = Target.getInstructionSet();
  const std::vector<CodeGenRegisterClass> &RCs = Target.getRegisterClasses();

  // Keep track of all of the def lists we have emitted already.
  std::map<std::vector<Record*>, unsigned> EmittedLists;
  unsigned ListNumber = 0;
  std::map<std::vector<Record*>, unsigned> EmittedBarriers;
  unsigned BarrierNumber = 0;
  std::map<Record*, unsigned> BarriersMap;
 
  // Emit all of the instruction's implicit uses and defs.
  for (CodeGenTarget::inst_iterator II = Target.inst_begin(),
         E = Target.inst_end(); II != E; ++II) {
    Record *Inst = II->second.TheDef;
    std::vector<Record*> Uses = Inst->getValueAsListOfDefs("Uses");
    if (!Uses.empty()) {
      unsigned &IL = EmittedLists[Uses];
      if (!IL) PrintDefList(Uses, IL = ++ListNumber, OS);
    }
    std::vector<Record*> Defs = Inst->getValueAsListOfDefs("Defs");
    if (!Defs.empty()) {
      std::vector<Record*> RCBarriers;
      DetectRegisterClassBarriers(Defs, RCs, RCBarriers);
      if (!RCBarriers.empty()) {
        unsigned &IB = EmittedBarriers[RCBarriers];
        if (!IB) PrintBarriers(RCBarriers, IB = ++BarrierNumber, OS);
        BarriersMap.insert(std::make_pair(Inst, IB));
      }

      unsigned &IL = EmittedLists[Defs];
      if (!IL) PrintDefList(Defs, IL = ++ListNumber, OS);
    }
  }

  OperandInfoMapTy OperandInfoIDs;
  
  // Emit all of the operand info records.
  EmitOperandInfo(OS, OperandInfoIDs);
  
  // Emit all of the TargetInstrDesc records in their ENUM ordering.
  //
  OS << "\nstatic const TargetInstrDesc " << TargetName
     << "Insts[] = {\n";
  std::vector<const CodeGenInstruction*> NumberedInstructions;
  Target.getInstructionsByEnumValue(NumberedInstructions);

  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i)
    emitRecord(*NumberedInstructions[i], i, InstrInfo, EmittedLists,
               BarriersMap, OperandInfoIDs, OS);
  OS << "};\n";
  OS << "} // End llvm namespace \n";
}

void InstrInfoEmitter::emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                                  Record *InstrInfo,
                         std::map<std::vector<Record*>, unsigned> &EmittedLists,
                                  std::map<Record*, unsigned> &BarriersMap,
                                  const OperandInfoMapTy &OpInfo,
                                  std::ostream &OS) {
  int MinOperands = 0;
  if (!Inst.OperandList.empty())
    // Each logical operand can be multiple MI operands.
    MinOperands = Inst.OperandList.back().MIOperandNo +
                  Inst.OperandList.back().MINumOperands;

  OS << "  { ";
  OS << Num << ",\t" << MinOperands << ",\t"
     << Inst.NumDefs << ",\t" << getItinClassNumber(Inst.TheDef)
     << ",\t\"" << Inst.TheDef->getName() << "\", 0";

  // Emit all of the target indepedent flags...
  if (Inst.isReturn)           OS << "|(1<<TID::Return)";
  if (Inst.isBranch)           OS << "|(1<<TID::Branch)";
  if (Inst.isIndirectBranch)   OS << "|(1<<TID::IndirectBranch)";
  if (Inst.isBarrier)          OS << "|(1<<TID::Barrier)";
  if (Inst.hasDelaySlot)       OS << "|(1<<TID::DelaySlot)";
  if (Inst.isCall)             OS << "|(1<<TID::Call)";
  if (Inst.canFoldAsLoad)      OS << "|(1<<TID::FoldableAsLoad)";
  if (Inst.mayLoad)            OS << "|(1<<TID::MayLoad)";
  if (Inst.mayStore)           OS << "|(1<<TID::MayStore)";
  if (Inst.isPredicable)       OS << "|(1<<TID::Predicable)";
  if (Inst.isConvertibleToThreeAddress) OS << "|(1<<TID::ConvertibleTo3Addr)";
  if (Inst.isCommutable)       OS << "|(1<<TID::Commutable)";
  if (Inst.isTerminator)       OS << "|(1<<TID::Terminator)";
  if (Inst.isReMaterializable) OS << "|(1<<TID::Rematerializable)";
  if (Inst.isNotDuplicable)    OS << "|(1<<TID::NotDuplicable)";
  if (Inst.hasOptionalDef)     OS << "|(1<<TID::HasOptionalDef)";
  if (Inst.usesCustomDAGSchedInserter)
    OS << "|(1<<TID::UsesCustomDAGSchedInserter)";
  if (Inst.isVariadic)         OS << "|(1<<TID::Variadic)";
  if (Inst.hasSideEffects)     OS << "|(1<<TID::UnmodeledSideEffects)";
  if (Inst.isAsCheapAsAMove)   OS << "|(1<<TID::CheapAsAMove)";
  OS << ", 0";

  // Emit all of the target-specific flags...
  ListInit *LI    = InstrInfo->getValueAsListInit("TSFlagsFields");
  ListInit *Shift = InstrInfo->getValueAsListInit("TSFlagsShifts");
  if (LI->getSize() != Shift->getSize())
    throw "Lengths of " + InstrInfo->getName() +
          ":(TargetInfoFields, TargetInfoPositions) must be equal!";

  for (unsigned i = 0, e = LI->getSize(); i != e; ++i)
    emitShiftedValue(Inst.TheDef, dynamic_cast<StringInit*>(LI->getElement(i)),
                     dynamic_cast<IntInit*>(Shift->getElement(i)), OS);

  OS << ", ";

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

  std::map<Record*, unsigned>::iterator BI = BarriersMap.find(Inst.TheDef);
  if (BI == BarriersMap.end())
    OS << "NULL, ";
  else
    OS << "Barriers" << BI->second << ", ";

  // Emit the operand info.
  std::vector<std::string> OperandInfo = GetOperandInfo(Inst);
  if (OperandInfo.empty())
    OS << "0";
  else
    OS << "OperandInfo" << OpInfo.find(OperandInfo)->second;
  
  OS << " },  // Inst #" << Num << " = " << Inst.TheDef->getName() << "\n";
}


void InstrInfoEmitter::emitShiftedValue(Record *R, StringInit *Val,
                                        IntInit *ShiftInt, std::ostream &OS) {
  if (Val == 0 || ShiftInt == 0)
    throw std::string("Illegal value or shift amount in TargetInfo*!");
  RecordVal *RV = R->getValue(Val->getValue());
  int Shift = ShiftInt->getValue();

  if (RV == 0 || RV->getValue() == 0) {
    // This isn't an error if this is a builtin instruction.
    if (R->getName() != "PHI" &&
        R->getName() != "INLINEASM" &&
        R->getName() != "DBG_LABEL" &&
        R->getName() != "EH_LABEL" &&
        R->getName() != "GC_LABEL" &&
        R->getName() != "DECLARE" &&
        R->getName() != "EXTRACT_SUBREG" &&
        R->getName() != "INSERT_SUBREG" &&
        R->getName() != "IMPLICIT_DEF" &&
        R->getName() != "SUBREG_TO_REG")
      throw R->getName() + " doesn't have a field named '" + 
            Val->getValue() + "'!";
    return;
  }

  Init *Value = RV->getValue();
  if (BitInit *BI = dynamic_cast<BitInit*>(Value)) {
    if (BI->getValue()) OS << "|(1<<" << Shift << ")";
    return;
  } else if (BitsInit *BI = dynamic_cast<BitsInit*>(Value)) {
    // Convert the Bits to an integer to print...
    Init *I = BI->convertInitializerTo(new IntRecTy());
    if (I)
      if (IntInit *II = dynamic_cast<IntInit*>(I)) {
        if (II->getValue()) {
          if (Shift)
            OS << "|(" << II->getValue() << "<<" << Shift << ")";
          else
            OS << "|" << II->getValue();
        }
        return;
      }

  } else if (IntInit *II = dynamic_cast<IntInit*>(Value)) {
    if (II->getValue()) {
      if (Shift)
        OS << "|(" << II->getValue() << "<<" << Shift << ")";
      else
        OS << II->getValue();
    }
    return;
  }

  std::cerr << "Unhandled initializer: " << *Val << "\n";
  throw "In record '" + R->getName() + "' for TSFlag emission.";
}

