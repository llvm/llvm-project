//===- RegisterBankEmitter.cpp - Generate a Register Bank Desc. -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of a target
// register bank for a code generator.
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenRegisters.h"
#include "Common/CodeGenTarget.h"
#include "Common/InfoByHwMode.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TGTimer.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <set>
#include <unordered_map>

#define DEBUG_TYPE "register-bank-emitter"

using namespace llvm;

namespace {
struct PartialMappingInfo {
  size_t StartIdx;
  size_t Length;

  PartialMappingInfo(size_t Len, size_t StartIdx = 0)
      : StartIdx(StartIdx), Length(Len) {}

  PartialMappingInfo(const Record &Def)
      : StartIdx(Def.getValueAsInt("StartIdx")),
        Length(Def.getValueAsInt("Length")) {}

  bool operator==(const PartialMappingInfo &RHS) const {
    return StartIdx == RHS.StartIdx && Length == RHS.Length;
  }

  bool operator<(const PartialMappingInfo &RHS) const {
    return std::tie(StartIdx, Length) < std::tie(RHS.StartIdx, RHS.Length);
  }

  bool isSimple() const { return StartIdx == 0; }
};

class RegisterBank {

  /// A vector of register classes that are included in the register bank.
  using RegisterClassesTy = std::vector<const CodeGenRegisterClass *>;

private:
  const Record &TheDef;

  /// The register classes that are covered by the register bank.
  RegisterClassesTy RCs;

  std::set<PartialMappingInfo> PartSizeSet;

  /// The register class with the largest register size.
  std::vector<const CodeGenRegisterClass *> RCsWithLargestRegSize;

public:
  RegisterBank(const Record &TheDef, unsigned NumModeIds)
      : TheDef(TheDef), RCsWithLargestRegSize(NumModeIds) {}

  /// Get the human-readable name for the bank.
  StringRef getName() const { return TheDef.getValueAsString("Name"); }

  /// Get the name of the enumerator in the ID enumeration.
  std::string getEnumeratorName() const {
    return (TheDef.getName() + "ID").str();
  }

  /// Get the name of the array holding the register class coverage data;
  std::string getCoverageArrayName() const {
    return (TheDef.getName() + "CoverageData").str();
  }

  /// Get the name of the global instance variable.
  StringRef getInstanceVarName() const { return TheDef.getName(); }

  const Record &getDef() const { return TheDef; }

  /// Get the register classes listed in the RegisterBank.RegisterClasses field.
  std::vector<const CodeGenRegisterClass *>
  getExplicitlySpecifiedRegisterClasses(
      const CodeGenRegBank &RegisterClassHierarchy) const {
    std::vector<const CodeGenRegisterClass *> RCs;
    for (const auto *RCDef : getDef().getValueAsListOfDefs("RegisterClasses"))
      RCs.push_back(RegisterClassHierarchy.getRegClass(RCDef));
    return RCs;
  }

  /// Add a register class to the bank without duplicates.
  void addRegisterClass(const CodeGenRegisterClass *RC) {
    if (llvm::is_contained(RCs, RC))
      return;

    // FIXME? We really want the register size rather than the spill size
    //        since the spill size may be bigger on some targets with
    //        limited load/store instructions. However, we don't store the
    //        register size anywhere (we could sum the sizes of the subregisters
    //        but there may be additional bits too) and we can't derive it from
    //        the VT's reliably due to Untyped.
    unsigned NumModeIds = RCsWithLargestRegSize.size();
    for (unsigned M = 0; M < NumModeIds; ++M) {
      if (RCsWithLargestRegSize[M] == nullptr)
        RCsWithLargestRegSize[M] = RC;
      else if (RCsWithLargestRegSize[M]->RSI.get(M).SpillSize <
               RC->RSI.get(M).SpillSize)
        RCsWithLargestRegSize[M] = RC;
      assert(RCsWithLargestRegSize[M] && "RC was nullptr?");
    }

    RCs.emplace_back(RC);
  }

  std::string getPartialMappingEnumName(const PartialMappingInfo &PM) const {
    std::string Name;
    Name.reserve(getName().size() + 7 + std::size("PMI_"));
    raw_string_ostream OS(Name);
    OS << "PMI_" << getName();
    if (PM.StartIdx != 0)
      OS << PM.StartIdx << '_';
    OS << PM.Length;
    return Name;
  }

  // Initialize partial mapping size info, must be called after
  // RCs is initialized.
  void initPartSizeSet() {
    for (const auto &RC : register_classes()) {
      for (auto &&[HWMode, RegSI] : RC->RSI)
        PartSizeSet.insert(PartialMappingInfo(RegSI.RegSize));
    }

    std::vector<const Record *> ExtraMappings =
        TheDef.getValueAsListOfDefs("ExtraPartialMappings");
    for (const auto *ExtraMapping : ExtraMappings) {
      auto PMI = PartialMappingInfo(*ExtraMapping);
      if (PMI.isSimple())
        PartSizeSet.insert(PartialMappingInfo(*ExtraMapping));
    }

    std::vector<const Record *> IgnoredMappings =
        TheDef.getValueAsListOfDefs("IgnoredPartialMappings");
    for (const auto *IgnoredMapping : IgnoredMappings) {
      PartSizeSet.erase(PartialMappingInfo(*IgnoredMapping));
    }
  }

  const std::set<PartialMappingInfo> getPartSizeSet() const {
    return PartSizeSet;
  }

  const CodeGenRegisterClass *getRCWithLargestRegSize(unsigned HwMode) const {
    return RCsWithLargestRegSize[HwMode];
  }

  unsigned getScalableMappingSize() const {
    return TheDef.getValueAsInt("ScalableMappingSize");
  }

  iterator_range<RegisterClassesTy::const_iterator> register_classes() const {
    return RCs;
  }
};

struct InstrShape {
  std::vector<int> RepresentativeOpIndices;
  std::vector<int> Shape;
  std::vector<unsigned char> IsPtr;

  InstrShape() = default;
  InstrShape(const CodeGenInstruction &CGI) {
    Shape.reserve(CGI.Operands.size());
    RepresentativeOpIndices.resize(CGI.Operands.size(), -1);
    for (const auto &[OpIdx, Op] : llvm::enumerate(CGI.Operands)) {
      if (!Op.Rec->isSubClassOf("TypedOperand")) {
        Shape.push_back(-1);
        continue;
      }
      if (Op.Rec->getValueAsBit("IsImmediate")) {
        Shape.push_back(-1);
        continue;
      }
      int TypeIdx = -1;
      TypeIdx = StringRef(Op.OperandType)
                        .drop_until(llvm::isDigit)
                        .getAsInteger(10, TypeIdx)
                    ? -1
                    : TypeIdx;
      Shape.push_back(TypeIdx);
      if (TypeIdx >= 0) {
        if (RepresentativeOpIndices[TypeIdx] < 0)
          RepresentativeOpIndices[TypeIdx] = OpIdx;
      }
    }
    llvm::erase_if(RepresentativeOpIndices, [](int Id) { return Id < 0; });
    for (int Idx : RepresentativeOpIndices)
      IsPtr.push_back(CGI.Operands[Idx].Rec->getValueAsBit("IsPointer"));
  }
  bool operator<(const InstrShape &RHS) const {
    return std::tie(Shape, IsPtr) < std::tie(RHS.Shape, RHS.IsPtr);
  }
};

class RegisterBankEmitter {
private:
  const CodeGenTarget Target;
  const RecordKeeper &Records;

  // Generic instructions are grouped by shapes, this map owns InstrShape
  std::map<InstrShape, std::set<const CodeGenInstruction *>> InstrShapes;
  std::map<const CodeGenInstruction *, const InstrShape *> CGIToShapeMap;
  std::map<const InstrShape *, UniqueVector<std::vector<const RegisterBank *>>>
      BankAssignments;
  DenseMap<const CodeGenInstruction *, std::vector<const Record *>>
      InstrMappingRules;
  // store start offset of each shape for GeneratedOperandsMappings
  std::vector<size_t> ShapeOffsets;
  // first index is shape index, second index is index of BankAssignment
  // correspond to the current shape index, so we can get start offset of
  // current bank combination
  std::vector<std::vector<size_t>> BankCombOffsets;

  void emitHeader(raw_ostream &OS, StringRef TargetName,
                  ArrayRef<RegisterBank> Banks);
  void emitPartialMapDeclaration(raw_ostream &OS, StringRef TargetName,
                                 ArrayRef<RegisterBank> Banks);
  void emitValueMappingDeclaration(raw_ostream &OS, StringRef TargetName,
                                   ArrayRef<RegisterBank> Banks);
  void emitBaseClassDefinition(raw_ostream &OS, StringRef TargetName,
                               ArrayRef<RegisterBank> Banks);
  void emitBaseClassImplementation(raw_ostream &OS, StringRef TargetName,
                                   ArrayRef<RegisterBank> Banks);
  void emitPartialMapImplementation(raw_ostream &OS, StringRef TargetName,
                                    ArrayRef<RegisterBank> Banks);
  void emitValueMappingImplementation(raw_ostream &OS, StringRef TargetName,
                                      ArrayRef<RegisterBank> Banks);

  void collectInstructionShapes();
  void collectMappingRules(ArrayRef<RegisterBank> Banks);
  void emitInstrShapeDeclaration(raw_ostream &OS, StringRef TargetName);
  void emitInstrShapeImplementation(raw_ostream &OS, StringRef TargetName);
  void emitSearchMethodDeclaration(raw_ostream &OS, StringRef TargetName,
                                   ArrayRef<RegisterBank> Banks);
  void emitSearchMethod(raw_ostream &OS, StringRef TargetName,
                        ArrayRef<RegisterBank> Banks);

public:
  RegisterBankEmitter(const RecordKeeper &R) : Target(R), Records(R) {}

  void run(raw_ostream &OS);
};

} // end anonymous namespace

/// Emit code to declare the ID enumeration and external global instance
/// variables.
void RegisterBankEmitter::emitHeader(raw_ostream &OS, StringRef TargetName,
                                     ArrayRef<RegisterBank> Banks) {
  IfDefEmitter IfDef(OS, "GET_REGBANK_DECLARATIONS");
  NamespaceEmitter NS(OS, ("llvm::" + TargetName).str());

  // <Target>RegisterBankInfo.h
  OS << "enum : unsigned {\n";

  OS << "  InvalidRegBankID = ~0u,\n";
  unsigned ID = 0;
  for (const auto &Bank : Banks)
    OS << "  " << Bank.getEnumeratorName() << " = " << ID++ << ",\n";
  OS << "  NumRegisterBanks,\n"
     << "};\n";
}

/// Emit declarations of the <Target>GenRegisterBankInfo class.
void RegisterBankEmitter::emitBaseClassDefinition(
    raw_ostream &OS, StringRef TargetName, ArrayRef<RegisterBank> Banks) {
  IfDefEmitter IfDef(OS, "GET_TARGET_REGBANK_CLASS");

  OS << "private:\n"
     << "  static const RegisterBank *RegBanks[];\n"
     << "  static const unsigned Sizes[];\n\n"
     << "public:\n"
     << "  const RegisterBank &getRegBankFromRegClass(const "
        "TargetRegisterClass &RC, LLT Ty) const override;\n"
     << "protected:\n"
     << "  " << TargetName << "GenRegisterBankInfo(unsigned HwMode = 0);\n"
     << "\n";

  for (auto *MappingIDDef : Records.getAllDerivedDefinitions("MappingID")) {
    StringRef IDName = MappingIDDef->getName();
    if (IDName == "DefaultMappingID" || IDName == "InvalidMappingID")
      continue;
    OS << "  inline constexpr unsigned " << IDName << " = "
       << MappingIDDef->getValueAsInt("ID") << ";\n";
  }

  emitPartialMapDeclaration(OS, TargetName, Banks);
  emitValueMappingDeclaration(OS, TargetName, Banks);
  emitInstrShapeDeclaration(OS, TargetName);
  emitSearchMethodDeclaration(OS, TargetName, Banks);
}

void RegisterBankEmitter::emitPartialMapDeclaration(
    raw_ostream &OS, StringRef TargetName, ArrayRef<RegisterBank> Banks) {
  OS << "protected:\n"
        "  enum PartialMappingIdx {\n"
        "    PMI_None = -1,\n";
  unsigned Idx = 0;
  for (const auto &Bank : Banks) {
    OS << '\n';
    const std::set<PartialMappingInfo> &PartSizeSet = Bank.getPartSizeSet();
    for (const auto &SI : PartSizeSet) {
      OS << "    // " << Idx << ": " << Bank.getName() << ' ' << SI.Length
         << "-bit value.\n";
      OS << "    " << Bank.getPartialMappingEnumName(SI) << ",\n";
      ++Idx;
    }
    if (!PartSizeSet.empty()) {
      OS << "    PMI_First" << Bank.getName() << " = "
         << Bank.getPartialMappingEnumName(*PartSizeSet.begin()) << ",\n"
         << "    PMI_Last" << Bank.getName() << " = "
         << Bank.getPartialMappingEnumName(*PartSizeSet.rbegin()) << ",\n";
    }
  }
  OS << "  };\n";
  OS << "  static const PartialMapping PartMappings[];\n\n";

  OS << "  static bool checkPartialMap(unsigned Idx, unsigned ValStartIdx, \n"
        "                              unsigned ValLength, const RegisterBank "
        "&RB);\n";
  OS << "  static bool checkPartialMappingIdx(PartialMappingIdx FirstAlias,\n"
        "                                     PartialMappingIdx LastAlias,\n"
        "                                     ArrayRef<PartialMappingIdx> "
        "Order);\n";
}

void RegisterBankEmitter::emitValueMappingDeclaration(
    raw_ostream &OS, StringRef TargetName, ArrayRef<RegisterBank> Banks) {
  OS << "  static const ValueMapping GeneratedValMappings[];\n"
        "  static const ValueMapping *getGeneratedValMappings(unsigned BankID, "
        "const TypeSize &Size);\n"
        "  static int getPartialMappingEnumOffset(unsigned BankID, const "
        "TypeSize &Size);\n"
        "  static const ValueMapping *getGeneratedValueMapping(unsigned Opc, "
        "ArrayRef<unsigned> BankIDs, "
        "ArrayRef<int> Coord);\n"
        "  const InstructionMapping& getGeneratedInstructionMapping(const "
        "MachineInstr &MI) const;\n";
}

void RegisterBankEmitter::emitInstrShapeDeclaration(raw_ostream &OS,
                                                    StringRef TargetName) {
  OS << "\n  static int getInstrShape(unsigned Opc);\n";
  OS << "  static const ValueMapping GeneratedOperandsMappings[];";
}

/// Visit each register class belonging to the given register bank.
///
/// A class belongs to the bank iff any of these apply:
/// * It is explicitly specified
/// * It is a subclass of a class that is a member.
/// * It is a class containing subregisters of the registers of a class that
///   is a member. This is known as a subreg-class.
///
/// This function must be called for each explicitly specified register class.
///
/// \param RC The register class to search.
/// \param Kind A debug string containing the path the visitor took to reach RC.
/// \param VisitFn The action to take for each class visited. It may be called
///                multiple times for a given class if there are multiple paths
///                to the class.
static void visitRegisterBankClasses(
    const CodeGenRegBank &RegisterClassHierarchy,
    const CodeGenRegisterClass *RC, const Twine &Kind,
    std::function<void(const CodeGenRegisterClass *, StringRef)> VisitFn,
    DenseSet<const CodeGenRegisterClass *> &VisitedRCs) {

  // Make sure we only visit each class once to avoid infinite loops.
  if (!VisitedRCs.insert(RC).second)
    return;

  // Visit each explicitly named class.
  VisitFn(RC, Kind.str());

  for (const auto &PossibleSubclass : RegisterClassHierarchy.getRegClasses()) {
    std::string TmpKind =
        (Kind + " (" + PossibleSubclass.getName() + ")").str();

    // Visit each subclass of an explicitly named class.
    if (RC != &PossibleSubclass && RC->hasSubClass(&PossibleSubclass))
      visitRegisterBankClasses(RegisterClassHierarchy, &PossibleSubclass,
                               TmpKind + " " + RC->getName() + " subclass",
                               VisitFn, VisitedRCs);

    // Visit each class that contains only subregisters of RC with a common
    // subregister-index.
    //
    // More precisely, PossibleSubclass is a subreg-class iff Reg:SubIdx is in
    // PossibleSubclass for all registers Reg from RC using any
    // subregister-index SubReg
    if (!PossibleSubclass.hasSuperRegClass(RC))
      continue;

    for (const auto &SubIdx : RegisterClassHierarchy.getSubRegIndices()) {
      if (PossibleSubclass.hasSuperRegClass(&SubIdx, RC)) {
        std::string TmpKind2 = (Twine(TmpKind) + " " + RC->getName() +
                                " class-with-subregs: " + RC->getName())
                                   .str();
        VisitFn(&PossibleSubclass, TmpKind2);
      }
    }
  }
}

void RegisterBankEmitter::emitBaseClassImplementation(
    raw_ostream &OS, StringRef TargetName, ArrayRef<RegisterBank> Banks) {
  const CodeGenRegBank &RegisterClassHierarchy = Target.getRegBank();
  const CodeGenHwModes &CGH = Target.getHwModes();

  IfDefEmitter IfDef(OS, "GET_TARGET_REGBANK_IMPL");
  NamespaceEmitter LlvmNS(OS, "llvm");

  {
    NamespaceEmitter TargetNS(OS, TargetName);
    for (const auto &Bank : Banks) {
      std::vector<std::vector<const CodeGenRegisterClass *>> RCsGroupedByWord(
          (RegisterClassHierarchy.getRegClasses().size() + 31) / 32);

      for (const auto &RC : Bank.register_classes())
        RCsGroupedByWord[RC->EnumValue / 32].push_back(RC);

      OS << "const uint32_t " << Bank.getCoverageArrayName() << "[] = {\n";
      unsigned LowestIdxInWord = 0;
      for (const auto &RCs : RCsGroupedByWord) {
        OS << "    // " << LowestIdxInWord << "-" << (LowestIdxInWord + 31)
           << "\n";
        for (const auto &RC : RCs) {
          OS << "    (1u << (" << RC->getQualifiedIdName() << " - "
             << LowestIdxInWord << ")) |\n";
        }
        OS << "    0,\n";
        LowestIdxInWord += 32;
      }
      OS << "};\n";
    }
    OS << "\n";

    for (const auto &Bank : Banks) {
      std::string QualifiedBankID =
          (TargetName + "::" + Bank.getEnumeratorName()).str();
      OS << "constexpr RegisterBank " << Bank.getInstanceVarName()
         << "(/* ID */ " << QualifiedBankID << ", /* Name */ \""
         << Bank.getName() << "\", " << "/* CoveredRegClasses */ "
         << Bank.getCoverageArrayName() << ", /* NumRegClasses */ "
         << RegisterClassHierarchy.getRegClasses().size() << ");\n";
    }
  } // End target namespace.

  OS << "\nconst RegisterBank *" << TargetName
     << "GenRegisterBankInfo::RegBanks[] = {\n";
  for (const auto &Bank : Banks)
    OS << "    &" << TargetName << "::" << Bank.getInstanceVarName() << ",\n";
  OS << "};\n\n";

  unsigned NumModeIds = CGH.getNumModeIds();
  OS << "const unsigned " << TargetName << "GenRegisterBankInfo::Sizes[] = {\n";
  for (unsigned M = 0; M < NumModeIds; ++M) {
    OS << "    // Mode = " << M << " ("
       << CGH.getModeName(M, /*IncludeDefault=*/true) << ")\n";
    for (const auto &Bank : Banks) {
      const CodeGenRegisterClass &RC = *Bank.getRCWithLargestRegSize(M);
      unsigned Size = RC.RSI.get(M).SpillSize;
      OS << "    " << Size << ",\n";
    }
  }
  OS << "};\n\n";

  OS << TargetName << "GenRegisterBankInfo::" << TargetName
     << "GenRegisterBankInfo(unsigned HwMode)\n"
     << "    : RegisterBankInfo(RegBanks, " << TargetName
     << "::NumRegisterBanks, Sizes, HwMode) {\n"
     << "  // Assert that RegBank indices match their ID's\n"
     << "#ifndef NDEBUG\n"
     << "  for (auto RB : enumerate(RegBanks))\n"
     << "    assert(RB.index() == RB.value()->getID() && \"Index != ID\");\n"
     << "#endif // NDEBUG\n"
     << "}\n";

  uint32_t NumRegBanks = Banks.size();
  uint32_t BitSize = NextPowerOf2(Log2_32(NumRegBanks));
  uint32_t ElemsPerWord = 32 / BitSize;
  uint32_t BitMask = (1 << BitSize) - 1;
  bool HasAmbigousOrMissingEntry = false;
  struct Entry {
    std::string RCIdName;
    std::string RBIdName;
  };
  SmallVector<Entry, 0> Entries;
  for (const auto &Bank : Banks) {
    for (const auto *RC : Bank.register_classes()) {
      if (RC->EnumValue >= Entries.size())
        Entries.resize(RC->EnumValue + 1);
      Entry &E = Entries[RC->EnumValue];
      E.RCIdName = RC->getIdName();
      if (!E.RBIdName.empty()) {
        HasAmbigousOrMissingEntry = true;
        E.RBIdName = "InvalidRegBankID";
      } else {
        E.RBIdName = (TargetName + "::" + Bank.getEnumeratorName()).str();
      }
    }
  }
  for (auto &E : Entries) {
    if (E.RBIdName.empty()) {
      HasAmbigousOrMissingEntry = true;
      E.RBIdName = "InvalidRegBankID";
    }
  }
  OS << "\nconst RegisterBank &\n"
     << TargetName
     << "GenRegisterBankInfo::getRegBankFromRegClass"
        "(const TargetRegisterClass &RC, LLT) const {\n";
  if (HasAmbigousOrMissingEntry) {
    OS << "  constexpr uint32_t InvalidRegBankID = uint32_t("
       << TargetName + "::InvalidRegBankID) & " << BitMask << ";\n";
  }
  unsigned TableSize =
      Entries.size() / ElemsPerWord + ((Entries.size() % ElemsPerWord) > 0);
  OS << "  static const uint32_t RegClass2RegBank[" << TableSize << "] = {\n";
  uint32_t Shift = 32 - BitSize;
  bool First = true;
  std::string TrailingComment;
  for (auto &E : Entries) {
    Shift += BitSize;
    if (Shift == 32) {
      Shift = 0;
      if (First)
        First = false;
      else
        OS << ',' << TrailingComment << '\n';
    } else {
      OS << " |" << TrailingComment << '\n';
    }
    OS << "    ("
       << (E.RBIdName.empty()
               ? "InvalidRegBankID"
               : Twine("uint32_t(").concat(E.RBIdName).concat(")").str())
       << " << " << Shift << ')';
    if (!E.RCIdName.empty())
      TrailingComment = " // " + E.RCIdName;
    else
      TrailingComment = "";
  }
  OS << TrailingComment
     << "\n  };\n"
        "  const unsigned RegClassID = RC.getID();\n"
        "  if (LLVM_LIKELY(RegClassID < "
     << Entries.size()
     << ")) {\n"
        "    unsigned RegBankID = (RegClass2RegBank[RegClassID / "
     << ElemsPerWord << "] >> ((RegClassID % " << ElemsPerWord << ") * "
     << BitSize << ")) & " << BitMask << ";\n";
  if (HasAmbigousOrMissingEntry) {
    OS << "    if (RegBankID != InvalidRegBankID)\n"
          "      return getRegBank(RegBankID);\n";
  } else {
    OS << "    return getRegBank(RegBankID);\n";
  }
  OS << "  }\n"
        "  llvm_unreachable(llvm::Twine(\"Target needs to handle register "
        "class ID "
        "0x\").concat(llvm::Twine::utohexstr(RegClassID)).str().c_str());\n"
        "}\n";

  emitPartialMapImplementation(OS, TargetName, Banks);
  OS << '\n';
  emitValueMappingImplementation(OS, TargetName, Banks);
  OS << '\n';
  emitInstrShapeImplementation(OS, TargetName);
  OS << '\n';
  emitSearchMethod(OS, TargetName, Banks);
}

void RegisterBankEmitter::emitPartialMapImplementation(
    raw_ostream &OS, StringRef TargetName, ArrayRef<RegisterBank> Banks) {
  OS << "\nconst RegisterBankInfo::PartialMapping\n"
     << TargetName
     << "GenRegisterBankInfo::PartMappings[] = {\n"
        "  // StartIdx, Length, RegBank\n";
  for (const auto &Bank : Banks) {
    for (const auto &SI : Bank.getPartSizeSet()) {
      OS << "  {" << SI.StartIdx << ", " << SI.Length << ", " << TargetName
         << "::" << Bank.getInstanceVarName() << "},\n";
    }
    OS << '\n';
  }
  OS << "};\n\n";

  OS << "bool " << TargetName << R"(GenRegisterBankInfo::checkPartialMappingIdx(
    PartialMappingIdx FirstAlias, PartialMappingIdx LastAlias,
    ArrayRef<PartialMappingIdx> Order) {
  if (Order.front() != FirstAlias)
    return false;
  if (Order.back() != LastAlias)
    return false;
  if (Order.front() > Order.back())
    return false;

  return llvm::adjacent_find(Order,
                             [](PartialMappingIdx LHS, PartialMappingIdx RHS) {
                               return RHS != LHS + 1;
                             }) == Order.end();
}
)";

  OS << "bool " << TargetName <<
      R"(GenRegisterBankInfo::checkPartialMap(unsigned Idx,
                                                 unsigned ValStartIdx,
                                                 unsigned ValLength,
                                                 const RegisterBank &RB) {
  const PartialMapping &Map = PartMappings[Idx];
  return Map.StartIdx == ValStartIdx && Map.Length == ValLength &&
         Map.RegBank == &RB;
}
)";
}

void RegisterBankEmitter::emitValueMappingImplementation(
    raw_ostream &OS, StringRef TargetName, ArrayRef<RegisterBank> Banks) {
  DenseMap<const RegisterBank *, DenseMap<size_t, unsigned>> BankValMapIdxMap;
  OS << "const RegisterBankInfo::ValueMapping\n"
     << TargetName
     << "GenRegisterBankInfo::GeneratedValMappings[] = {\n"
        "  // BreakDown, NumBreakDowns\n"
        "  // 0: invalid\n"
        "  {nullptr, 0},\n";
  size_t Idx = 1;
  for (const auto &Bank : Banks) {
    DenseMap<size_t, unsigned> &IdxMap = BankValMapIdxMap[&Bank];
    OS << "\n  // " << Bank.getName() << '\n';
    const std::set<PartialMappingInfo> &PartialSizeSet = Bank.getPartSizeSet();
    for (const auto &PS : PartialSizeSet) {
      if (!PS.isSimple())
        continue;
      IdxMap[PS.Length] = Idx++;
      OS << "  {&PartMappings[" << Bank.getPartialMappingEnumName(PS)
         << "], 1},\n";
    }
  }
  OS << "};\n\n";

  OS << "const RegisterBankInfo::ValueMapping*\n"
     << TargetName
     << "GenRegisterBankInfo::getGeneratedValMappings(unsigned BankID, const "
        "TypeSize &Size) {\n";
  OS << "  switch(BankID) {\n"
        "    default:\n"
        "      break;\n"
        "    case "
     << TargetName
     << "::InvalidRegBankID:\n"
        "      break;\n";
  for (const auto &Bank : Banks) {
    DenseMap<size_t, unsigned> &IdxMap = BankValMapIdxMap[&Bank];
    OS << "\n    case " << TargetName << "::" << Bank.getEnumeratorName()
       << ": {\n";
    const std::set<PartialMappingInfo> &PS = Bank.getPartSizeSet();
    if (size_t ScalableMappingSize = Bank.getScalableMappingSize()) {
      OS << "      if (Size.isScalable())\n"
            "        return &GeneratedValMappings["
         << IdxMap[ScalableMappingSize] << "];\n";
    }
    for (const auto &PM : PS) {
      if (!PM.isSimple())
        continue;
      OS << "      if (Size <= " << PM.Length << ")\n";
      OS << "        return &GeneratedValMappings[" << IdxMap[PM.Length]
         << "];\n";
    }
    OS << "      break;\n"
          "    }\n";
  }
  OS << "  }\n";
  OS << "  return &GeneratedValMappings[0];\n"
        "}\n\n";

  OS << "int\n"
     << TargetName
     << "GenRegisterBankInfo::getPartialMappingEnumOffset(unsigned BankID, "
        "const "
        "TypeSize &Size) {\n";
  OS << "  switch(BankID) {\n"
        "    default:\n"
        "      break;\n"
        "    case "
     << TargetName
     << "::InvalidRegBankID:\n"
        "      break;\n";
  for (const auto &Bank : Banks) {
    OS << "\n    case " << TargetName << "::" << Bank.getEnumeratorName()
       << ": {\n";
    const std::set<PartialMappingInfo> &PS = Bank.getPartSizeSet();
    if (size_t ScalableMappingSize = Bank.getScalableMappingSize()) {
      OS << "      if (Size.isScalable())\n"
            "        return "
         << Bank.getPartialMappingEnumName(ScalableMappingSize) << " - "
         << "PMI_First" << Bank.getName() << ";\n";
    }
    for (const auto &PM : PS) {
      if (!PM.isSimple())
        continue;
      OS << "      if (Size <= " << PM.Length << ")\n";
      OS << "        return " << Bank.getPartialMappingEnumName(PM) << " - "
         << "PMI_First" << Bank.getName() << ";\n";
    }
    OS << "      break;\n"
          "    }\n";
  }
  OS << "  }\n";
  OS << "  return -1;\n"
        "}\n\n";
}

template <typename T>
static void getCartesianProductImpl(
    std::vector<std::vector<T>> &Result, std::vector<T> &Current,
    typename std::vector<std::vector<T>>::const_iterator It,
    typename std::vector<std::vector<T>>::const_iterator End) {
  if (It == End) {
    Result.push_back(Current);
    return;
  }

  // need an easy name for my vector-of-ints
  const auto &ItVec = *It;
  for (auto I = ItVec.begin(), E = ItVec.end(); I != E; ++I) {
    Current.push_back(*I);
    getCartesianProductImpl(Result, Current, It + 1, End);
    Current.pop_back();
  }
}

template <typename T>
static std::vector<std::vector<T>>
getCartesianProduct(const std::vector<std::vector<T>> &Components) {
  std::vector<std::vector<T>> Result;
  std::vector<T> ResultTemp;
  getCartesianProductImpl(Result, ResultTemp, Components.begin(),
                          Components.end());
  return Result;
}

void RegisterBankEmitter::emitInstrShapeImplementation(raw_ostream &OS,
                                                       StringRef TargetName) {

  OS << "const RegisterBankInfo::ValueMapping\n"
     << TargetName << "GenRegisterBankInfo::GeneratedOperandsMappings[] = {\n";
  size_t ShapeOffset = 0;
  BankCombOffsets.resize(InstrShapes.size());
  for (const auto &[Idx, Pair] : llvm::enumerate(InstrShapes)) {
    OS << "\n  // shape" << Idx << '\n';
    ShapeOffsets.push_back(ShapeOffset);
    const auto &BankAssignment = BankAssignments[&Pair.first];
    if (BankAssignment.empty()) {
      for (int I = 0, E = Pair.first.Shape.size(); I != E; ++I)
        OS << "  {nullptr, 0},\n";
      ShapeOffset += Pair.first.Shape.size();
    } else {
      size_t BankCombOffset = 0;
      std::vector<size_t> &CurBankCombOffset = BankCombOffsets[Idx];
      CurBankCombOffset.resize(BankAssignment.size());
      for (const auto &[AssignmentIdx, Banks] :
           llvm::enumerate(BankAssignment)) {
        CurBankCombOffset[AssignmentIdx] = BankCombOffset;
        std::vector<std::vector<std::string>> Comps;
        for (const auto *Bank : Banks) {
          const std::set<PartialMappingInfo> &PSSet = Bank->getPartSizeSet();
          std::vector<std::string> Comp;
          Comp.reserve(PSSet.size());
          for (auto &PS : PSSet)
            Comp.push_back(Bank->getPartialMappingEnumName(PS));
          Comps.push_back(std::move(Comp));
        }

        OS << "  // ";
        ListSeparator LS(" x ");
        for (const auto *Bank : Banks)
          OS << LS << Bank->getName();
        OS << '\n';

        std::vector<std::vector<std::string>> Product =
            getCartesianProduct(Comps);
        BankCombOffset += Product.size();
        for (const auto &Tuple : Product) {
          for (int I = 0, E = Pair.first.Shape.size(); I != E; ++I) {
            auto &Shape = Pair.first.Shape;
            if (Shape[I] < 0)
              OS << "  {nullptr, 0},\n";
            else
              OS << "  {&PartMappings[" << Tuple[Shape[I]] << "], 1},\n";
          }
          ShapeOffset += Pair.first.Shape.size();
        }
      }
    }
  }
  OS << "};\n";

  OS << "int " << TargetName
     << "GenRegisterBankInfo::getInstrShape(unsigned Opc) {\n"
        "  switch(Opc) {\n"
        "    default:\n"
        "      return -1;\n";
  int Idx = 0;
  for (const auto &[Shape, CGIs] : InstrShapes) {
    for (auto *CGI : CGIs) {
      OS << "    case " << TargetName << "::" << CGI->getName() << ":\n";
    }
    OS << "      return " << Idx++ << ";\n";
  }
  OS << "  }\n"
        "}\n";
}

void RegisterBankEmitter::emitSearchMethodDeclaration(
    raw_ostream &OS, StringRef TargetName, ArrayRef<RegisterBank> Banks) {
  OS << '\n';
  OS << "  static size_t decodeCoord(ArrayRef<unsigned> BankIDs, ArrayRef<int> "
        "Coord);\n";
}

void RegisterBankEmitter::emitSearchMethod(raw_ostream &OS,
                                           StringRef TargetName,
                                           ArrayRef<RegisterBank> Banks) {
  OS << "size_t " << TargetName
     << "GenRegisterBankInfo::decodeCoord(ArrayRef<unsigned> BankIDs, "
        "ArrayRef<int> Coord) {\n";
  OS << "  assert(BankIDs.size() == Coord.size() && \"size not match, can't "
        "decode coordination\");\n";
  OS << "  static constexpr size_t BankPartSizeTable[] = { ";
  for (const auto &Bank : Banks)
    OS << Bank.getPartSizeSet().size() << ", ";
  OS << "};\n";
  OS << "  size_t Result = 0, TupleSize = BankIDs.size();\n"
        "  for (auto Idx : llvm::seq(TupleSize)) {\n"
        "    size_t Base = 1;\n"
        "    for (auto BankID : BankIDs.slice(Idx + 1))\n"
        "      Base *= BankPartSizeTable[BankID];\n"
        "    Result += Coord[Idx] * Base;\n"
        "  }\n";

  OS << "  return Result;\n"
        "}\n\n";

  OS << "const RegisterBankInfo::ValueMapping *\n"
     << TargetName
     << "GenRegisterBankInfo::getGeneratedValueMapping(unsigned Opc, "
        "ArrayRef<unsigned>BankIDs, "
        "ArrayRef<int> Coord) {\n";
  OS << "  static constexpr size_t ShapeSizeTable[] = { ";
  for (const auto &KV : InstrShapes)
    OS << KV.first.Shape.size() << ", ";
  OS << " };\n";
  OS << "  static constexpr size_t ShapeOffsetTable[] = { ";
  for (size_t Offset : ShapeOffsets)
    OS << Offset << ", ";
  OS << " };\n\n";

  // Currently the bank count is not so huge, so a multi dimensional array
  // solution is acceptable.
  for (const auto &[ShapeIdx, KV] : llvm::enumerate(InstrShapes)) {
    const InstrShape &Shape = KV.first;
    OS << "  static constexpr size_t BankCombOffsetsShape" << ShapeIdx;
    size_t TensorSize = 1;
    for (size_t _ : llvm::seq(Shape.RepresentativeOpIndices.size())) {
      OS << '[' << TargetName << "::NumRegisterBanks]";
      TensorSize *= Banks.size();
    }
    OS << " = {";
    std::vector<size_t> TensorValues(TensorSize);
    for (const auto &[AssignmentIdx, BankAssignment] :
         llvm::enumerate(BankAssignments[&Shape])) {
      size_t BankCombOffset = BankCombOffsets[ShapeIdx][AssignmentIdx];
      size_t BankNum = Banks.size();
      size_t TensorIdx = 0;
      size_t BankAssignmentSize = BankAssignment.size();
      for (auto &&[BankIdx, Bank] : llvm::enumerate(BankAssignment)) {
        size_t Base = 1;
        for (auto _ : llvm::seq(BankAssignmentSize - BankIdx - 1))
          Base *= BankNum;
        TensorIdx += Base * (Bank - Banks.data());
      }
      TensorValues[TensorIdx] = BankCombOffset + 1;
    }
    for (size_t Val : TensorValues)
      OS << Val << ", ";
    OS << "};\n";
  }
  OS << '\n';

  OS << "  int Shape = getInstrShape(Opc);\n"
        "  if (Shape < 0 || Shape >= "
     << InstrShapes.size() << ")\n";
  OS << "    return nullptr;\n"
        "  size_t Stride = ShapeSizeTable[Shape];\n\n";
  OS << "  size_t BankCombOffset = 0;\n"
        "  switch(Shape) {\n"
        "    default:\n"
        "      llvm_unreachable(\"Unexpected instruction shape!\");\n";
  for (const auto &[ShapeIdx, KV] : llvm::enumerate(InstrShapes)) {
    const InstrShape &Shape = KV.first;
    OS << "    case " << ShapeIdx
       << ":\n"
          "      BankCombOffset = BankCombOffsetsShape"
       << ShapeIdx;
    for (size_t I : llvm::seq(Shape.RepresentativeOpIndices.size()))
      OS << "[BankIDs[" << I << "]]";
    OS << " - 1;\n"
          "      break;\n";
  }
  OS << "  }\n";
  OS << "  const ValueMapping *MappingTable = "
        "&GeneratedOperandsMappings[ShapeOffsetTable[Shape] + "
        "BankCombOffset];\n"
        "  return &MappingTable[decodeCoord(BankIDs, Coord) * Stride];\n";

  OS << "}\n\n";

  OS << "const RegisterBankInfo::InstructionMapping&\n"
     << TargetName
     << "GenRegisterBankInfo::getGeneratedInstructionMapping(const "
        "MachineInstr &MI) const {\n";

  OS << "  const MachineFunction &MF = *MI.getParent()->getParent();\n"
        "  [[maybe_unused]] const MachineRegisterInfo &MRI = "
        "MF.getRegInfo();\n";
  OS << "  const auto &STI = MF.getSubtarget<" << TargetName
     << "Subtarget>();\n";
  OS << "  [[maybe_unused]] const TargetRegisterInfo &TRI = "
        "*STI.getRegisterInfo();\n\n";

  OS << "  const unsigned Opc = MI.getOpcode();\n\n";

  OS << "  switch(Opc) {\n"
        "    default:\n"
        "      break;\n";
  for (const auto &[Shape, CGIs] : InstrShapes) {
    // instruction with same rules, cluster them in the same case body
    std::map<std::vector<const Record *>, std::set<const CodeGenInstruction *>>
        GroupedInstrMappingRules;
    for (auto *CGI : CGIs) {
      auto It = InstrMappingRules.find(CGI);
      if (It == InstrMappingRules.end())
        continue;
      std::vector<const Record *> &Rules = It->second;
      GroupedInstrMappingRules[Rules].insert(CGI);
    }
    for (const auto &[Rules, CGIs] : GroupedInstrMappingRules) {
      for (const auto *CGI : CGIs)
        OS << "\n    case " << TargetName << "::" << CGI->getName() << ":";
      OS << " {\n";

      assert(!CGIs.empty() && "no instructions to emit");
      const CodeGenInstruction *RepCGI = *CGIs.begin();
      const InstrShape &Shape = *CGIToShapeMap[RepCGI];
      SmallVector<StringRef> OperandNames;
      // emit representative operand name and its LLT
      for (auto [Idx, OpIdx] : llvm::enumerate(Shape.RepresentativeOpIndices)) {
        const CGIOperandList::OperandInfo &OPI = RepCGI->Operands[OpIdx];
        OS << "      Register " << OPI.Name << " = MI.getOperand(" << OpIdx
           << ").getReg();\n";
        OS << "      [[maybe_unused]] LLT "
           << (Shape.IsPtr[Idx] ? "PType" : "Type") << Idx << " = MRI.getType("
           << OPI.Name << ");\n";
        OperandNames.push_back(OPI.Name);
      }
      OS << '\n';

      size_t BankTupleSize = Shape.RepresentativeOpIndices.size();
      OS << "      std::array<unsigned, " << BankTupleSize << "> BankIDs = {";
      for (size_t I = 0, E = Shape.RepresentativeOpIndices.size(); I != E; ++I)
        OS << TargetName << "::InvalidRegBankID,";
      OS << "};\n";
      OS << "      std::array TypeSizes = {";
      for (const auto &OperandName : OperandNames)
        OS << "getSizeInBits(" << OperandName << ", MRI, TRI),";
      OS << "};\n";
      for (const auto *RuleDef : Rules) {
        StringRef Pred = RuleDef->getValueAsString("Predicate");
        if (!Pred.empty())
          OS << "      if (" << Pred << ") {\n";

        OS << "        BankIDs = {";
        for (const auto *TypeMapping :
             RuleDef->getValueAsListOfDefs("TypeMappings")) {
          const Record *BankDef = TypeMapping->getValueAsDef("Bank");
          const RegisterBank &Bank =
              *llvm::find_if(Banks, [BankDef](const RegisterBank &Bank) {
                return &Bank.getDef() == BankDef;
              });
          OS << TargetName << "::" << Bank.getEnumeratorName() << ',';
        }
        OS << "};\n\n";

        // now get partial mapping offsets and encode them to shape value
        // mapping offset table

        OS << "        std::array<int, " << BankTupleSize << "> Coord = {";
        for (size_t I = 0; I != BankTupleSize; ++I)
          OS << "-1,";
        OS << "};\n";
        OS << "        for(auto &&[Idx, BankAndSize] : "
              "llvm::enumerate(llvm::zip(BankIDs, TypeSizes))) {\n"
              "          auto &&[BankID, Size] = BankAndSize;\n"
              "          int Offset = getPartialMappingEnumOffset(BankID, "
              "Size);\n"
              "          if (Offset < 0)\n"
              "            return getInvalidInstructionMapping();\n"
              "          Coord[Idx] = Offset;\n"
              "        }\n";
        // now we have bank combination and coordinate, generate instruction
        // value mapping
        OS << "        return getInstructionMapping("
           << RuleDef->getValueAsDef("ID")->getName() << ", "
           << RuleDef->getValueAsInt("Cost")
           << ", "
              "getGeneratedValueMapping(Opc, BankIDs, Coord), "
           << Shape.Shape.size() << ");\n";

        if (!Pred.empty())
          OS << "      }\n";
      }
      OS << "      break;\n";

      OS << "    }\n";
    }
  }
  OS << "  }\n"
        "  return getInvalidInstructionMapping();\n";
  OS << "}\n";
}

void RegisterBankEmitter::collectInstructionShapes() {
  for (const auto *GI :
       Records.getAllDerivedDefinitions("GenericInstruction")) {
    CodeGenInstruction &CGI = Target.getInstruction(GI);
    if (CGI.Operands.isVariadic || CGI.Operands.empty())
      continue;
    InstrShapes[InstrShape(CGI)].insert(&CGI);
  }
  for (auto &[Shape, CGIs] : InstrShapes) {
    for (auto &CGI : CGIs)
      CGIToShapeMap[CGI] = &Shape;
  }
}

void RegisterBankEmitter::collectMappingRules(ArrayRef<RegisterBank> Banks) {
  for (const auto *InstrMapping :
       Records.getAllDerivedDefinitions("RegBankInstrMapping")) {
    std::vector Opcodes = InstrMapping->getValueAsListOfDefs("Opcodes");
    std::vector MappingRules = InstrMapping->getValueAsListOfDefs("Rules");

    const InstrShape *Shape;
    CodeGenInstruction &RepCGI = Target.getInstruction(Opcodes.front());
    // verify all opcodes in list have the same shape
    for (auto *OpcDef : Opcodes) {
      CodeGenInstruction &CGI = Target.getInstruction(OpcDef);
      if (CGIToShapeMap.count(&CGI) == 0)
        PrintFatalError(InstrMapping, CGI.getName() + " is not supported");
      Shape = CGIToShapeMap[&RepCGI];
      if (CGIToShapeMap.at(&CGI) != Shape)
        PrintFatalError(InstrMapping,
                        CGI.getName() + " has different shape with others");

      llvm::append_range(InstrMappingRules[&CGI], MappingRules);
    }

    // verify all typed operands are covered by rules
    for (auto *MappingRule : MappingRules) {
      std::vector TypeMappings =
          MappingRule->getValueAsListOfDefs("TypeMappings");
      if (TypeMappings.size() != Shape->RepresentativeOpIndices.size())
        PrintFatalError(MappingRule->getLoc(),
                        "too many RegBankTypeMapping for these instructions");

      std::vector<const RegisterBank *> BankTuple(
          Shape->RepresentativeOpIndices.size());
      for (auto [Idx, OpIdx] :
           llvm::enumerate(Shape->RepresentativeOpIndices)) {
        const Record *TyDef = RepCGI.Operands[OpIdx].Rec;
        auto TypeMappingIter =
            llvm::find_if(TypeMappings, [TyDef](const Record *Rec) {
              return Rec->getValueAsDef("Type") == TyDef;
            });
        if (TypeMappingIter == TypeMappings.end())
          PrintFatalError(MappingRule->getLoc(),
                          TyDef->getName() + " is not coverted by this rule");
        const Record *TypeMapping = *TypeMappingIter;
        const Record *BankDef = TypeMapping->getValueAsDef("Bank");
        BankTuple[Idx] =
            llvm::find_if(Banks, [BankDef](const RegisterBank &Bank) {
              return &Bank.getDef() == BankDef;
            });
      }
      BankAssignments[Shape].insert(std::move(BankTuple));
    }

    for (auto &[CGI, Rules] : InstrMappingRules) {
      llvm::stable_sort(Rules, [](const Record *LHS, const Record *RHS) {
        return LHS->getValueAsInt("Priority") > RHS->getValueAsInt("Priority");
      });
    }
  }
}

void RegisterBankEmitter::run(raw_ostream &OS) {
  StringRef TargetName = Target.getName();
  const CodeGenRegBank &RegisterClassHierarchy = Target.getRegBank();
  const CodeGenHwModes &CGH = Target.getHwModes();

  TGTimer &Timer = Records.getTimer();
  Timer.startTimer("Analyze records");
  std::vector<RegisterBank> Banks;
  for (const auto &V : Records.getAllDerivedDefinitions("RegisterBank")) {
    DenseSet<const CodeGenRegisterClass *> VisitedRCs;
    RegisterBank Bank(*V, CGH.getNumModeIds());

    for (const CodeGenRegisterClass *RC :
         Bank.getExplicitlySpecifiedRegisterClasses(RegisterClassHierarchy)) {
      visitRegisterBankClasses(
          RegisterClassHierarchy, RC, "explicit",
          [&Bank](const CodeGenRegisterClass *RC, StringRef Kind) {
            LLVM_DEBUG(dbgs()
                       << "Added " << RC->getName() << "(" << Kind << ")\n");
            Bank.addRegisterClass(RC);
          },
          VisitedRCs);
    }

    Bank.initPartSizeSet();
    Banks.push_back(std::move(Bank));
  }

  if (Banks.empty())
    PrintFatalError("No register banks defined");

  // Warn about ambiguous MIR caused by register bank/class name clashes.
  Timer.startTimer("Warn ambiguous");
  for (const auto &Class : RegisterClassHierarchy.getRegClasses()) {
    for (const auto &Bank : Banks) {
      if (Bank.getName().lower() == StringRef(Class.getName()).lower()) {
        PrintWarning(Bank.getDef().getLoc(), "Register bank names should be "
                                             "distinct from register classes "
                                             "to avoid ambiguous MIR");
        PrintNote(Bank.getDef().getLoc(), "RegisterBank was declared here");
        PrintNote(Class.getDef()->getLoc(), "RegisterClass was declared here");
      }
    }
  }

  Timer.startTimer("Emit output");
  collectInstructionShapes();
  collectMappingRules(Banks);
  emitSourceFileHeader("Register Bank Source Fragments", OS);
  emitHeader(OS, TargetName, Banks);
  emitBaseClassDefinition(OS, TargetName, Banks);
  emitBaseClassImplementation(OS, TargetName, Banks);
}

static TableGen::Emitter::OptClass<RegisterBankEmitter>
    X("gen-register-bank", "Generate registers bank descriptions");
