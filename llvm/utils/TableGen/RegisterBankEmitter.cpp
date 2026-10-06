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
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TGTimer.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <set>

#define DEBUG_TYPE "register-bank-emitter"

using namespace llvm;

namespace {
struct PartialMappingInfo {
  size_t StartIdx;
  size_t Length;

  bool operator==(const PartialMappingInfo &RHS) const {
    return StartIdx == RHS.StartIdx && Length == RHS.Length;
  }

  bool operator<(const PartialMappingInfo &RHS) const {
    if (StartIdx < RHS.StartIdx)
      return true;
    if (StartIdx == RHS.StartIdx)
      return Length < RHS.Length;
    return false;
  }
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
      for (auto &&[HWMode, RegSI] : RC->RSI) {
        PartSizeSet.insert({0, RegSI.RegSize});
      }
    }

    std::vector<const Record *> ExtraMappings =
        TheDef.getValueAsListOfDefs("ExtraPartialMappings");
    for (const auto *ExtraMapping : ExtraMappings) {
      PartSizeSet.insert({(size_t)ExtraMapping->getValueAsInt("StartIdx"),
                          (size_t)ExtraMapping->getValueAsInt("Length")});
    }

    std::vector<const Record *> IgnoredMappings =
        TheDef.getValueAsListOfDefs("IgnoredPartialMappings");
    for (const auto *IgnoredMapping : IgnoredMappings) {
      PartSizeSet.erase({(size_t)IgnoredMapping->getValueAsInt("StartIdx"),
                         (size_t)IgnoredMapping->getValueAsInt("Length")});
    }
  }

  const std::set<PartialMappingInfo> getPartSizeSet() const {
    return PartSizeSet;
  }

  const CodeGenRegisterClass *getRCWithLargestRegSize(unsigned HwMode) const {
    return RCsWithLargestRegSize[HwMode];
  }

  iterator_range<RegisterClassesTy::const_iterator> register_classes() const {
    return RCs;
  }
};

class RegisterBankEmitter {
private:
  const CodeGenTarget Target;
  const RecordKeeper &Records;

  void emitHeader(raw_ostream &OS, StringRef TargetName,
                  ArrayRef<RegisterBank> Banks);
  void emitPartialMapDeclaration(raw_ostream &OS, StringRef TargetName,
                                 ArrayRef<RegisterBank> Banks);
  void emitBaseClassDefinition(raw_ostream &OS, StringRef TargetName,
                               ArrayRef<RegisterBank> Banks);
  void emitBaseClassImplementation(raw_ostream &OS, StringRef TargetName,
                                   ArrayRef<RegisterBank> Banks);
  void emitPartialMapImplementation(raw_ostream &OS, StringRef TargetName,
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

  emitPartialMapDeclaration(OS, TargetName, Banks);
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

  PartialMappingIdx Previous = Order.front();
  for (const auto &Current : Order.drop_front()) {
    if (Previous + 1 != Current)
      return false;
    Previous = Current;
  }
  return true;
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
  emitSourceFileHeader("Register Bank Source Fragments", OS);
  emitHeader(OS, TargetName, Banks);
  emitBaseClassDefinition(OS, TargetName, Banks);
  emitBaseClassImplementation(OS, TargetName, Banks);
}

static TableGen::Emitter::OptClass<RegisterBankEmitter>
    X("gen-register-bank", "Generate registers bank descriptions");
