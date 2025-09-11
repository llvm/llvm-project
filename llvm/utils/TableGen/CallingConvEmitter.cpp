//===- CallingConvEmitter.cpp - Generate calling conventions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting descriptions of the calling
// conventions supported by this target.
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenRegisters.h"
#include "Common/CodeGenTarget.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TGTimer.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <deque>
#include <set>

using namespace llvm;

namespace {
class CallingConvEmitter {
  const RecordKeeper &Records;
  const CodeGenTarget Target;
  unsigned Counter = 0u;
  std::string CurrentAction;
  bool SwiftAction = false;

  std::map<std::string, std::set<std::string>> AssignedRegsMap;
  std::map<std::string, std::set<std::string>> AssignedSwiftRegsMap;
  std::map<std::string, std::set<std::string>> DelegateToMap;

public:
  explicit CallingConvEmitter(const RecordKeeper &R) : Records(R), Target(R) {
    for (const CodeGenRegister &Reg : Target.getRegBank().getRegisters())
      RegistersByDefName.try_emplace(Reg.getName(), &Reg);
  }

  void run(raw_ostream &O);

private:
  void emitCallingConv(const Record *CC, raw_ostream &O);
  void emitAction(const Record *Action, indent Indent, raw_ostream &O);
  void emitArgRegisterLists(raw_ostream &O);

  StringMap<const CodeGenRegister *> RegistersByDefName;
  std::string getQualifiedRegisterName(const Init *I);
};
} // End anonymous namespace

void CallingConvEmitter::run(raw_ostream &O) {
  emitSourceFileHeader("Calling Convention Implementation Fragment", O);

  ArrayRef<const Record *> CCs =
      Records.getAllDerivedDefinitions("CallingConv");

  // Emit prototypes for all of the non-custom CC's so that they can forward ref
  // each other.
  Records.getTimer().startTimer("Emit prototypes");
  O << "#ifndef GET_CC_REGISTER_LISTS\n\n";
  for (const Record *CC : CCs) {
    if (!CC->getValueAsBit("Custom")) {
      unsigned Pad = CC->getName().size();
      if (CC->getValueAsBit("Entry")) {
        O << "bool llvm::";
        Pad += 12;
      } else {
        O << "static bool ";
        Pad += 13;
      }
      O << CC->getName() << "(unsigned ValNo, MVT ValVT,\n"
        << std::string(Pad, ' ') << "MVT LocVT, CCValAssign::LocInfo LocInfo,\n"
        << std::string(Pad, ' ')
        << "ISD::ArgFlagsTy ArgFlags, Type *OrigTy, CCState &State);\n";
    }
  }

  // Emit each non-custom calling convention description in full.
  Records.getTimer().startTimer("Emit full descriptions");
  for (const Record *CC : CCs) {
    if (!CC->getValueAsBit("Custom")) {
      emitCallingConv(CC, O);
    }
  }

  emitArgRegisterLists(O);

  O << "\n#endif // CC_REGISTER_LIST\n";
}

void CallingConvEmitter::emitCallingConv(const Record *CC, raw_ostream &O) {
  const ListInit *CCActions = CC->getValueAsListInit("Actions");
  Counter = 0;

  CurrentAction = CC->getName().str();
  // Call upon the creation of a map entry from the void!
  // We want an entry in AssignedRegsMap for every action, even if that
  // entry is empty.
  AssignedRegsMap[CurrentAction] = {};

  O << "\n\n";
  unsigned Pad = CurrentAction.size();
  if (CC->getValueAsBit("Entry")) {
    O << "bool llvm::";
    Pad += 12;
  } else {
    O << "static bool ";
    Pad += 13;
  }
  O << CurrentAction << "(unsigned ValNo, MVT ValVT,\n"
    << std::string(Pad, ' ') << "MVT LocVT, CCValAssign::LocInfo LocInfo,\n"
    << std::string(Pad, ' ') << "ISD::ArgFlagsTy ArgFlags, Type *OrigTy, "
    << "CCState &State) {\n";
  // Emit all of the actions, in order.
  for (unsigned I = 0, E = CCActions->size(); I != E; ++I) {
    const Record *Action = CCActions->getElementAsRecord(I);
    SwiftAction =
        llvm::any_of(Action->getSuperClasses(), [](const Record *Class) {
          std::string Name = Class->getNameInitAsString();
          return StringRef(Name).starts_with("CCIfSwift");
        });

    O << "\n";
    emitAction(Action, indent(2), O);
  }

  O << "\n  return true; // CC didn't match.\n";
  O << "}\n";
}

// Return the name of the specified Init (DefInit or StringInit), with a
// namespace qualifier if the corresponding record contains one.
std::string CallingConvEmitter::getQualifiedRegisterName(const Init *I) {
  if (const auto *DI = dyn_cast<DefInit>(I))
    return getQualifiedName(DI->getDef());

  const auto *SI = cast<StringInit>(I);
  if (const CodeGenRegister *CGR = RegistersByDefName.lookup(SI->getValue()))
    return getQualifiedName(CGR->TheDef);

  PrintFatalError("register not defined: " + SI->getAsString());
  return "";
}

void CallingConvEmitter::emitAction(const Record *Action, indent Indent,
                                    raw_ostream &O) {

  auto EmitRegList = [&](const ListInit *RL, const StringRef RLName) {
    O << Indent << "static const MCPhysReg " << RLName << "[] = {\n";
    O << Indent << "  ";
    ListSeparator LS;
    for (const Init *V : RL->getElements())
      O << LS << getQualifiedRegisterName(V);
    O << "\n" << Indent << "};\n";
  };

  auto EmitAllocateReg = [&](ArrayRef<const ListInit *> RegLists,
                             ArrayRef<std::string> RLNames) {
    SmallVector<std::string> Parms;
    if (RegLists[0]->size() == 1) {
      for (const ListInit *LI : RegLists)
        Parms.push_back(getQualifiedRegisterName(LI->getElement(0)));
    } else {
      for (const std::string &S : RLNames)
        Parms.push_back(S + utostr(++Counter));
      for (const auto [Idx, LI] : enumerate(RegLists))
        EmitRegList(LI, Parms[Idx]);
    }
    O << formatv("{0}if (MCRegister Reg = State.AllocateReg({1})) {{\n", Indent,
                 make_range(Parms.begin(), Parms.end()));
    O << Indent << "  State.addLoc(CCValAssign::getReg(ValNo, ValVT, "
      << "Reg, LocVT, LocInfo));\n";
  };

  auto EmitAllocateStack = [&](bool EmitOffset = false) {
    int Size = Action->getValueAsInt("Size");
    int Align = Action->getValueAsInt("Align");
    if (EmitOffset)
      O << Indent << "int64_t Offset" << ++Counter << " = ";
    else
      O << Indent << "  (void)";
    O << "State.AllocateStack(";

    const char *Fmt = "  State.getMachineFunction().getDataLayout()."
                      "{0}(EVT(LocVT).getTypeForEVT(State.getContext()))";
    if (Size)
      O << Size << ", ";
    else
      O << "\n" << Indent << formatv(Fmt, "getTypeAllocSize") << ", ";
    if (Align)
      O << "Align(" << Align << ")";
    else
      O << "\n" << Indent << formatv(Fmt, "getABITypeAlign");
    O << ");\n";
  };

  if (Action->isSubClassOf("CCPredicateAction")) {
    O << Indent << "if (";

    if (Action->isSubClassOf("CCIfType")) {
      const ListInit *VTs = Action->getValueAsListInit("VTs");
      for (unsigned I = 0, E = VTs->size(); I != E; ++I) {
        const Record *VT = VTs->getElementAsRecord(I);
        if (I != 0)
          O << " ||\n    " << Indent;
        O << "LocVT == " << getEnumName(getValueType(VT));
      }

    } else if (Action->isSubClassOf("CCIf")) {
      O << Action->getValueAsString("Predicate");
    } else {
      errs() << *Action;
      PrintFatalError(Action->getLoc(), "Unknown CCPredicateAction!");
    }

    O << ") {\n";
    emitAction(Action->getValueAsDef("SubAction"), Indent + 2, O);
    O << Indent << "}\n";
    return;
  }

  if (Action->isSubClassOf("CCDelegateTo")) {
    const Record *CC = Action->getValueAsDef("CC");
    O << Indent << "if (!" << CC->getName()
      << "(ValNo, ValVT, LocVT, LocInfo, ArgFlags, OrigTy, State))\n"
      << Indent + 2 << "return false;\n";
    DelegateToMap[CurrentAction].insert(CC->getName().str());
  } else if (Action->isSubClassOf("CCAssignToReg") ||
             Action->isSubClassOf("CCAssignToRegTuple") ||
             Action->isSubClassOf("CCAssignToRegAndStack")) {
    const ListInit *RegList = Action->getValueAsListInit("RegList");
    for (unsigned I = 0, E = RegList->size(); I != E; ++I) {
      std::string Name = getQualifiedRegisterName(RegList->getElement(I));
      if (SwiftAction)
        AssignedSwiftRegsMap[CurrentAction].insert(std::move(Name));
      else
        AssignedRegsMap[CurrentAction].insert(std::move(Name));
    }
    EmitAllocateReg({RegList}, {"RegList"});

    if (Action->isSubClassOf("CCAssignToRegAndStack"))
      EmitAllocateStack();

    O << Indent << "  return false;\n";
    O << Indent << "}\n";
  } else if (Action->isSubClassOf("CCAssignToRegWithShadow")) {
    const ListInit *RegList = Action->getValueAsListInit("RegList");
    const ListInit *ShadowRegList = Action->getValueAsListInit("ShadowRegList");
    if (!ShadowRegList->empty() && ShadowRegList->size() != RegList->size())
      PrintFatalError(Action->getLoc(),
                      "Invalid length of list of shadowed registers");

    EmitAllocateReg({RegList, ShadowRegList}, {"RegList", "RegList"});

    O << Indent << "  return false;\n";
    O << Indent << "}\n";
  } else if (Action->isSubClassOf("CCAssignToStack")) {
    EmitAllocateStack(/*EmitOffset=*/true);
    O << Indent << "State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset"
      << Counter << ", LocVT, LocInfo));\n";
    O << Indent << "return false;\n";
  } else if (Action->isSubClassOf("CCAssignToStackWithShadow")) {
    int Size = Action->getValueAsInt("Size");
    int Align = Action->getValueAsInt("Align");
    const ListInit *ShadowRegList = Action->getValueAsListInit("ShadowRegList");

    unsigned ShadowRegListNumber = ++Counter;
    EmitRegList(ShadowRegList, "ShadowRegList" + utostr(ShadowRegListNumber));

    O << Indent << "int64_t Offset" << ++Counter << " = State.AllocateStack("
      << Size << ", Align(" << Align << "), "
      << "ShadowRegList" << ShadowRegListNumber << ");\n";
    O << Indent << "State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset"
      << Counter << ", LocVT, LocInfo));\n";
    O << Indent << "return false;\n";
  } else if (Action->isSubClassOf("CCPromoteToType")) {
    const Record *DestTy = Action->getValueAsDef("DestTy");
    MVT::SimpleValueType DestVT = getValueType(DestTy);
    O << Indent << "LocVT = " << getEnumName(DestVT) << ";\n";
    if (MVT(DestVT).isFloatingPoint()) {
      O << Indent << "LocInfo = CCValAssign::FPExt;\n";
    } else {
      O << Indent << "if (ArgFlags.isSExt())\n"
        << Indent << "  LocInfo = CCValAssign::SExt;\n"
        << Indent << "else if (ArgFlags.isZExt())\n"
        << Indent << "  LocInfo = CCValAssign::ZExt;\n"
        << Indent << "else\n"
        << Indent << "  LocInfo = CCValAssign::AExt;\n";
    }
  } else if (Action->isSubClassOf("CCPromoteToUpperBitsInType")) {
    const Record *DestTy = Action->getValueAsDef("DestTy");
    MVT::SimpleValueType DestVT = getValueType(DestTy);
    O << Indent << "LocVT = " << getEnumName(DestVT) << ";\n";
    if (MVT(DestVT).isFloatingPoint()) {
      PrintFatalError(Action->getLoc(),
                      "CCPromoteToUpperBitsInType does not handle floating "
                      "point");
    } else {
      O << Indent << "if (ArgFlags.isSExt())\n"
        << Indent << "  LocInfo = CCValAssign::SExtUpper;\n"
        << Indent << "else if (ArgFlags.isZExt())\n"
        << Indent << "  LocInfo = CCValAssign::ZExtUpper;\n"
        << Indent << "else\n"
        << Indent << "  LocInfo = CCValAssign::AExtUpper;\n";
    }
  } else if (Action->isSubClassOf("CCBitConvertToType")) {
    const Record *DestTy = Action->getValueAsDef("DestTy");
    O << Indent << "LocVT = " << getEnumName(getValueType(DestTy)) << ";\n";
    O << Indent << "LocInfo = CCValAssign::BCvt;\n";
  } else if (Action->isSubClassOf("CCTruncToType")) {
    const Record *DestTy = Action->getValueAsDef("DestTy");
    O << Indent << "LocVT = " << getEnumName(getValueType(DestTy)) << ";\n";
    O << Indent << "LocInfo = CCValAssign::Trunc;\n";
  } else if (Action->isSubClassOf("CCPassIndirect")) {
    const Record *DestTy = Action->getValueAsDef("DestTy");
    O << Indent << "LocVT = " << getEnumName(getValueType(DestTy)) << ";\n";
    O << Indent << "LocInfo = CCValAssign::Indirect;\n";
  } else if (Action->isSubClassOf("CCPassByVal")) {
    int Size = Action->getValueAsInt("Size");
    int Align = Action->getValueAsInt("Align");
    O << Indent << "State.HandleByVal(ValNo, ValVT, LocVT, LocInfo, " << Size
      << ", Align(" << Align << "), ArgFlags);\n";
    O << Indent << "return false;\n";
  } else if (Action->isSubClassOf("CCCustom")) {
    O << Indent << "if (" << Action->getValueAsString("FuncName")
      << "(ValNo, ValVT, "
      << "LocVT, LocInfo, ArgFlags, State))\n";
    O << Indent << "  return false;\n";
  } else {
    errs() << *Action;
    PrintFatalError(Action->getLoc(), "Unknown CCAction!");
  }
}

void CallingConvEmitter::emitArgRegisterLists(raw_ostream &O) {
  // Transitively merge all delegated CCs into AssignedRegsMap.
  using EntryTy = std::pair<std::string, std::set<std::string>>;
  bool Redo;
  do {
    Redo = false;
    std::deque<EntryTy> Worklist(DelegateToMap.begin(), DelegateToMap.end());

    while (!Worklist.empty()) {
      EntryTy Entry = Worklist.front();
      Worklist.pop_front();

      const std::string &CCName = Entry.first;
      std::set<std::string> &Registers = Entry.second;
      if (!Registers.empty())
        continue;

      for (auto &InnerEntry : Worklist) {
        const std::string &InnerCCName = InnerEntry.first;
        std::set<std::string> &InnerRegisters = InnerEntry.second;

        auto It = InnerRegisters.find(CCName);
        if (It != InnerRegisters.end()) {
          const auto &Src = AssignedRegsMap[CCName];
          AssignedRegsMap[InnerCCName].insert(Src.begin(), Src.end());
          InnerRegisters.erase(It);
        }
      }

      DelegateToMap.erase(CCName);
      Redo = true;
    }
  } while (Redo);

  if (AssignedRegsMap.empty())
    return;

  O << "\n#else\n\n";

  for (const auto &[RegName, Registers] : AssignedRegsMap) {
    if (RegName.empty())
      continue;

    O << "const MCRegister " << RegName << "_ArgRegs[] = { ";

    if (Registers.empty())
      O << "0";
    else
      O << llvm::interleaved(Registers);

    O << " };\n";
  }

  if (AssignedSwiftRegsMap.empty())
    return;

  O << "\n// Registers used by Swift.\n";
  for (const auto &[RegName, Registers] : AssignedSwiftRegsMap)
    O << "const MCRegister " << RegName << "_Swift_ArgRegs[] = { "
      << llvm::interleaved(Registers) << " };\n";
}

static TableGen::Emitter::OptClass<CallingConvEmitter>
    X("gen-callingconv", "Generate calling convention descriptions");
