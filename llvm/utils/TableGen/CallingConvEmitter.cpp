//===- CallingConvEmitter.cpp - Generate calling conventions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting descriptions of the calling
// conventions supported by this target.
//
//===----------------------------------------------------------------------===//

#include "CallingConvEmitter.h"
#include "Record.h"
#include "CodeGenTarget.h"
using namespace llvm;

void CallingConvEmitter::run(std::ostream &O) {
  EmitSourceFileHeader("Calling Convention Implementation Fragment", O);

  std::vector<Record*> CCs = Records.getAllDerivedDefinitions("CallingConv");
  
  // Emit prototypes for all of the CC's so that they can forward ref each
  // other.
  for (unsigned i = 0, e = CCs.size(); i != e; ++i) {
    O << "static bool " << CCs[i]->getName()
      << "(unsigned ValNo, MVT ValVT,\n"
      << std::string(CCs[i]->getName().size()+13, ' ')
      << "MVT LocVT, CCValAssign::LocInfo LocInfo,\n"
      << std::string(CCs[i]->getName().size()+13, ' ')
      << "ISD::ArgFlagsTy ArgFlags, CCState &State);\n";
  }
  
  // Emit each calling convention description in full.
  for (unsigned i = 0, e = CCs.size(); i != e; ++i)
    EmitCallingConv(CCs[i], O);
}


void CallingConvEmitter::EmitCallingConv(Record *CC, std::ostream &O) {
  ListInit *CCActions = CC->getValueAsListInit("Actions");
  Counter = 0;

  O << "\n\nstatic bool " << CC->getName()
    << "(unsigned ValNo, MVT ValVT,\n"
    << std::string(CC->getName().size()+13, ' ')
    << "MVT LocVT, CCValAssign::LocInfo LocInfo,\n"
    << std::string(CC->getName().size()+13, ' ')
    << "ISD::ArgFlagsTy ArgFlags, CCState &State) {\n";
  // Emit all of the actions, in order.
  for (unsigned i = 0, e = CCActions->getSize(); i != e; ++i) {
    O << "\n";
    EmitAction(CCActions->getElementAsRecord(i), 2, O);
  }
  
  O << "\n  return true;  // CC didn't match.\n";
  O << "}\n";
}

void CallingConvEmitter::EmitAction(Record *Action,
                                    unsigned Indent, std::ostream &O) {
  std::string IndentStr = std::string(Indent, ' ');
  
  if (Action->isSubClassOf("CCPredicateAction")) {
    O << IndentStr << "if (";
    
    if (Action->isSubClassOf("CCIfType")) {
      ListInit *VTs = Action->getValueAsListInit("VTs");
      for (unsigned i = 0, e = VTs->getSize(); i != e; ++i) {
        Record *VT = VTs->getElementAsRecord(i);
        if (i != 0) O << " ||\n    " << IndentStr;
        O << "LocVT == " << getEnumName(getValueType(VT));
      }

    } else if (Action->isSubClassOf("CCIf")) {
      O << Action->getValueAsString("Predicate");
    } else {
      Action->dump();
      throw "Unknown CCPredicateAction!";
    }
    
    O << ") {\n";
    EmitAction(Action->getValueAsDef("SubAction"), Indent+2, O);
    O << IndentStr << "}\n";
  } else {
    if (Action->isSubClassOf("CCDelegateTo")) {
      Record *CC = Action->getValueAsDef("CC");
      O << IndentStr << "if (!" << CC->getName()
        << "(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State))\n"
        << IndentStr << "  return false;\n";
    } else if (Action->isSubClassOf("CCAssignToReg")) {
      ListInit *RegList = Action->getValueAsListInit("RegList");
      if (RegList->getSize() == 1) {
        O << IndentStr << "if (unsigned Reg = State.AllocateReg(";
        O << getQualifiedName(RegList->getElementAsRecord(0)) << ")) {\n";
      } else {
        O << IndentStr << "static const unsigned RegList" << ++Counter
          << "[] = {\n";
        O << IndentStr << "  ";
        for (unsigned i = 0, e = RegList->getSize(); i != e; ++i) {
          if (i != 0) O << ", ";
          O << getQualifiedName(RegList->getElementAsRecord(i));
        }
        O << "\n" << IndentStr << "};\n";
        O << IndentStr << "if (unsigned Reg = State.AllocateReg(RegList"
          << Counter << ", " << RegList->getSize() << ")) {\n";
      }
      O << IndentStr << "  State.addLoc(CCValAssign::getReg(ValNo, ValVT, "
        << "Reg, LocVT, LocInfo));\n";
      O << IndentStr << "  return false;\n";
      O << IndentStr << "}\n";
    } else if (Action->isSubClassOf("CCAssignToRegWithShadow")) {
      ListInit *RegList = Action->getValueAsListInit("RegList");
      ListInit *ShadowRegList = Action->getValueAsListInit("ShadowRegList");
      if (ShadowRegList->getSize() >0 &&
          ShadowRegList->getSize() != RegList->getSize())
        throw "Invalid length of list of shadowed registers";

      if (RegList->getSize() == 1) {
        O << IndentStr << "if (unsigned Reg = State.AllocateReg(";
        O << getQualifiedName(RegList->getElementAsRecord(0));
        O << ", " << getQualifiedName(ShadowRegList->getElementAsRecord(0));
        O << ")) {\n";
      } else {
        unsigned RegListNumber = ++Counter;
        unsigned ShadowRegListNumber = ++Counter;

        O << IndentStr << "static const unsigned RegList" << RegListNumber
          << "[] = {\n";
        O << IndentStr << "  ";
        for (unsigned i = 0, e = RegList->getSize(); i != e; ++i) {
          if (i != 0) O << ", ";
          O << getQualifiedName(RegList->getElementAsRecord(i));
        }
        O << "\n" << IndentStr << "};\n";

        O << IndentStr << "static const unsigned RegList"
          << ShadowRegListNumber << "[] = {\n";
        O << IndentStr << "  ";
        for (unsigned i = 0, e = ShadowRegList->getSize(); i != e; ++i) {
          if (i != 0) O << ", ";
          O << getQualifiedName(ShadowRegList->getElementAsRecord(i));
        }
        O << "\n" << IndentStr << "};\n";

        O << IndentStr << "if (unsigned Reg = State.AllocateReg(RegList"
          << RegListNumber << ", " << "RegList" << ShadowRegListNumber
          << ", " << RegList->getSize() << ")) {\n";
      }
      O << IndentStr << "  State.addLoc(CCValAssign::getReg(ValNo, ValVT, "
        << "Reg, LocVT, LocInfo));\n";
      O << IndentStr << "  return false;\n";
      O << IndentStr << "}\n";
    } else if (Action->isSubClassOf("CCAssignToStack")) {
      int Size = Action->getValueAsInt("Size");
      int Align = Action->getValueAsInt("Align");

      O << IndentStr << "unsigned Offset" << ++Counter
        << " = State.AllocateStack(";
      if (Size)
        O << Size << ", ";
      else
        O << "\n" << IndentStr << "  State.getTarget().getTargetData()"
          "->getTypePaddedSize(LocVT.getTypeForMVT()), ";
      if (Align)
        O << Align;
      else
        O << "\n" << IndentStr << "  State.getTarget().getTargetData()"
          "->getABITypeAlignment(LocVT.getTypeForMVT())";
      O << ");\n" << IndentStr
        << "State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset"
        << Counter << ", LocVT, LocInfo));\n";
      O << IndentStr << "return false;\n";
    } else if (Action->isSubClassOf("CCPromoteToType")) {
      Record *DestTy = Action->getValueAsDef("DestTy");
      O << IndentStr << "LocVT = " << getEnumName(getValueType(DestTy)) <<";\n";
      O << IndentStr << "if (ArgFlags.isSExt())\n"
        << IndentStr << IndentStr << "LocInfo = CCValAssign::SExt;\n"
        << IndentStr << "else if (ArgFlags.isZExt())\n"
        << IndentStr << IndentStr << "LocInfo = CCValAssign::ZExt;\n"
        << IndentStr << "else\n"
        << IndentStr << IndentStr << "LocInfo = CCValAssign::AExt;\n";
    } else if (Action->isSubClassOf("CCPassByVal")) {
      int Size = Action->getValueAsInt("Size");
      int Align = Action->getValueAsInt("Align");
      O << IndentStr
        << "State.HandleByVal(ValNo, ValVT, LocVT, LocInfo, "
        << Size << ", " << Align << ", ArgFlags);\n";
      O << IndentStr << "return false;\n";
    } else {
      Action->dump();
      throw "Unknown CCAction!";
    }
  }
}
