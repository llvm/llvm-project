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
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TGTimer.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "register-bank-emitter"

using namespace llvm;

namespace {

struct PartialMapping {
  size_t Length;
  size_t StartIdx;

  // the index in final table
  size_t Index = 0;

  PartialMapping(const Record &Def)
      : Length(Def.getValueAsInt("Length")),
        StartIdx(Def.getValueAsInt("StartIdx")) {}

  PartialMapping(size_t Length, size_t StartIdx = 0)
      : Length(Length), StartIdx(StartIdx) {}

  bool operator<(const PartialMapping &RHS) const {
    return std::tie(StartIdx, Length) < std::tie(RHS.StartIdx, RHS.Length);
  }

  bool operator==(const PartialMapping &RHS) const {
    return std::tie(StartIdx, Length) == std::tie(RHS.StartIdx, RHS.Length);
  }

  bool isSimple() const { return StartIdx == 0; }
};

struct ValueMapping {
  std::vector<PartialMapping> Parts;

  size_t getNumBreakDowns() const { return Parts.size(); }
  size_t getSize() const {
    assert(getNumBreakDowns() != 0 && "invalid value mapping");
    return Parts.size() * Parts[0].Length;
  }
  size_t getPartialMappingIdx() const {
    assert(getNumBreakDowns() != 0 && "invalid value mapping");
    return Parts[0].Index;
  }
};

class RegisterBank {

  /// A vector of register classes that are included in the register bank.
  using RegisterClassesTy = std::vector<const CodeGenRegisterClass *>;

private:
  const Record &TheDef;

  /// The register classes that are covered by the register bank.
  RegisterClassesTy RCs;

  /// The register class with the largest register size.
  std::vector<const CodeGenRegisterClass *> RCsWithLargestRegSize;

  /// Record all possible partial mappings of this register bank,
  /// currently only mappings with start index 0 are supported.
  std::set<PartialMapping> PartialMappings;
  std::vector<PartialMapping> PartialMappingTable;
  std::vector<ValueMapping> ValueMappings;

public:
  RegisterBank(const Record &TheDef, unsigned NumModeIds)
      : TheDef(TheDef), RCsWithLargestRegSize(NumModeIds) {
    for (const auto *ExtraDef :
         TheDef.getValueAsListOfDefs("ExtraPartialMappings"))
      PartialMappings.emplace(*ExtraDef);
  }

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

  size_t getScalableMappingSize() const {
    return TheDef.getValueAsInt("ScalableMappingSize");
  }

  const Record &getDef() const { return TheDef; }

  const std::set<PartialMapping> &getPartialMappings() const {
    return PartialMappings;
  }
  const std::vector<PartialMapping> &getPartialMappingTable() const {
    return PartialMappingTable;
  }
  const std::vector<ValueMapping> &getValueMappings() const {
    return ValueMappings;
  }

  void collectMappings() {
    for (const auto *RC : register_classes())
      for (const auto &[Mode, RSI] : RC->RSI)
        if (RSI.RegSize != 0)
          PartialMappings.emplace(RSI.RegSize);

    for (const auto *IgnoreDef :
         TheDef.getValueAsListOfDefs("IgnoredPartialMappings"))
      PartialMappings.erase(*IgnoreDef);

    // construct possible value mappings
    // partial mappings in PartialMappings
    size_t PartialMappingIdx = 0;
    for (const auto &PM : PartialMappings) {
      if (!PM.isSimple())
        break;

      ValueMapping VMs;
      VMs.Parts.push_back(PM);
      // find all possible break downs
      size_t NextStartIdx = PM.Length;
      size_t PartSize = PM.Length;
      while (true) {
        auto It = llvm::find_if(
            PartialMappings, [NextStartIdx, PartSize](const PartialMapping PM) {
              return PM.StartIdx == NextStartIdx && PM.Length == PartSize;
            });
        if (It == PartialMappings.end())
          break;
        VMs.Parts.push_back(*It);
        NextStartIdx += PartSize;
      }

      for (auto &PM : VMs.Parts)
        PM.Index = PartialMappingIdx++;
      llvm::append_range(PartialMappingTable, VMs.Parts);
      for (size_t I : llvm::seq(VMs.Parts.size())) {
        ArrayRef<PartialMapping> PMs = VMs.Parts;
        ValueMapping VM;
        VM.Parts.reserve(I + 1);
        llvm::append_range(VM.Parts, PMs.take_front(I + 1));
        ValueMappings.push_back(std::move(VM));
      }
    }
  }

  void setPartialMappingIdxStart(size_t IdxStart) {
    for (auto &PM : PartialMappingTable)
      PM.Index += IdxStart;
    for (auto &VM : ValueMappings)
      for (auto &PM : VM.Parts)
        PM.Index += IdxStart;
  }

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

  const CodeGenRegisterClass *getRCWithLargestRegSize(unsigned HwMode) const {
    return RCsWithLargestRegSize[HwMode];
  }

  iterator_range<RegisterClassesTy::const_iterator> register_classes() const {
    return RCs;
  }
};

struct TypedOperandInfo {
  const Record *Def = nullptr;
  unsigned Index;
  TypedOperandInfo() = default;
  TypedOperandInfo(const Record &Def) : Def(&Def) {
    Def.getValueAsString("OperandType")
        .drop_until(llvm::isDigit)
        .getAsInteger(10, Index);
  }
  bool isPtr() const { return Def->getValueAsBit("IsPointer"); }
  bool isImm() const { return Def->getValueAsBit("IsImmediate"); }

  std::string getLLTName() const {
    assert(!isImm() && "not a reg");
    return (isPtr() ? "PType" : "Type") + llvm::itostr(Index);
  }
};

using TypedOperandInfoMap = DenseMap<const Record *, TypedOperandInfo>;

struct InstrShape {
  // map operand num to type index, -1 means a non TypedOperand or imm
  // TypedOperand
  std::vector<int> Shape;
  std::vector<const Record *> OperandTypes;
  std::vector<int> RepresentativeOperands;
  InstrShape(const CodeGenInstruction &CGI, const TypedOperandInfoMap &TOI) {
    OperandTypes.resize(CGI.Operands.size());
    RepresentativeOperands.resize(CGI.Operands.size(), -1);
    for (const auto &[OperandIdx, Operand] : llvm::enumerate(CGI.Operands)) {
      const Record *OpRec = Operand.Rec;
      auto It = TOI.find(OpRec);
      if (It != TOI.end()) {
        const TypedOperandInfo Info = It->second;
        if (!Info.isImm()) {
          unsigned TypeIdx = Info.Index;
          Shape.push_back(TypeIdx);
          OperandTypes[TypeIdx] = OpRec;
          if (RepresentativeOperands[TypeIdx] == -1)
            RepresentativeOperands[TypeIdx] = OperandIdx;
          continue;
        }
      }
      Shape.push_back(-1);
    }

    while (!OperandTypes.empty() && OperandTypes.back() == nullptr)
      OperandTypes.pop_back();
    while (!RepresentativeOperands.empty() &&
           RepresentativeOperands.back() == -1)
      RepresentativeOperands.pop_back();
  }

  size_t getNumOperands() const { return Shape.size(); }
  size_t getTypeTupleSize() const { return OperandTypes.size(); }

  bool operator==(const InstrShape &RHS) const {
    return std::tie(OperandTypes, Shape) ==
           std::tie(RHS.OperandTypes, RHS.Shape);
  }
  bool operator<(const InstrShape &RHS) const {
    return std::tie(OperandTypes, Shape) <
           std::tie(RHS.OperandTypes, RHS.Shape);
  }
};

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

class RegisterBankEmitter {
private:
  const CodeGenTarget Target;
  const RecordKeeper &Records;

  std::vector<RegisterBank> Banks;

  TypedOperandInfoMap TypedOperands;
  std::map<InstrShape, std::set<const CodeGenInstruction *>> Shapes;
  DenseMap<const InstrShape *, std::set<std::vector<const RegisterBank *>>>
      BankAssignments;
  DenseMap<const CodeGenInstruction *, const InstrShape *> CGIToShapeMap;
  DenseMap<const CodeGenInstruction *, std::vector<const Record *>>
      InstrMappingRules;
  DenseMap<const CodeGenInstruction *, std::vector<const Record *>>
      InstrAltMappingRules;
  DenseMap<const InstrShape *,
           std::map<std::vector<const RegisterBank *>, size_t>>
      ShapeToTupleStart;

  void collectValueMappings();
  void validateMappingRules();
  void collectInstrMappings();

  void emitHeader(raw_ostream &OS, StringRef TargetName,
                  ArrayRef<RegisterBank> Banks);
  void emitBaseClassDefinition(raw_ostream &OS, StringRef TargetName,
                               ArrayRef<RegisterBank> Banks);
  void emitBaseClassImplementation(raw_ostream &OS, StringRef TargetName,
                                   ArrayRef<RegisterBank> Banks);
  std::vector<const RegisterBank *> getBankAssignment(const Record &RuleRec,
                                                      const InstrShape &Shape);
  void emitOperandsMappingRules(raw_ostream &OS, const CodeGenInstruction &CGI,
                                const InstrShape &Shape,
                                const std::vector<const Record *> Rules,
                                bool EmitAlt);
  void emitOperandMappingForShape(raw_ostream &OS, size_t ShapeIdx,
                                  const InstrShape &Shape);
  void emitSearchMethod(raw_ostream &OS, StringRef TargetName, bool EmitAlt);

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
     << "\n"
     << "  const InstructionMapping &getGeneratedInstrMapping(const "
        "MachineInstr &MI) const;\n"
     << "  InstructionMappings getGeneratedInstrAltMappings(const "
        "MachineInstr &MI) const;\n";
  for (const auto &MappingIDDef :
       Records.getAllDerivedDefinitions("RegBankMappingID")) {
    StringRef Name = MappingIDDef->getValueAsString("Name");
    if (Name == "DefaultMappingID" || Name == "InvalidMappingID")
      continue;
    OS << "  static inline constexpr unsigned " << Name << " = "
       << MappingIDDef->getValueAsInt("ID") << ";\n";
  }
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

void RegisterBankEmitter::collectValueMappings() {
  size_t PartialMappingIdxStart = 0;
  for (auto &Bank : Banks) {
    Bank.collectMappings();
    Bank.setPartialMappingIdxStart(PartialMappingIdxStart);
    PartialMappingIdxStart += Bank.getPartialMappingTable().size();
  }
}

void RegisterBankEmitter::validateMappingRules() {
  for (const auto *Rec :
       Records.getAllDerivedDefinitions("RegBankInstrMapping")) {
    std::vector<const Record *> Opcodes = Rec->getValueAsListOfDefs("Opcodes");
    assert(Opcodes.size() > 0 && "size assertion in RegBank.td failed");
    const CodeGenInstruction &CGI = Target.getInstruction(Opcodes[0]);
    if (!CGIToShapeMap.contains(&CGI))
      PrintFatalError("Unsupported instruction " + CGI.getName());
    InstrShape Shape(CGI, TypedOperands);
    if (!llvm::all_of(Opcodes, [&](const Record *OpDef) {
          return Shape ==
                 InstrShape(Target.getInstruction(OpDef), TypedOperands);
        }))
      PrintFatalError(Rec, "Not all instructions have the same shape.");

    for (const auto *RuleRec : Rec->getValueAsListOfDefs("Rules")) {
      std::vector<const Record *> MappingRecs =
          RuleRec->getValueAsListOfDefs("Mappings");
      SetVector<const Record *> MappedTypes;
      for (const auto *MappingRec : MappingRecs) {
        if (!MappingRec->isSubClassOf("RegBankTypeMapping"))
          continue;
        const Record *TypedOperandRec = MappingRec->getValueAsDef("Type");
        if (llvm::find(Shape.OperandTypes, TypedOperandRec) ==
            Shape.OperandTypes.end())
          PrintFatalError(MappingRec,
                          "unexpected TypedOperand for Instruction " +
                              CGI.getName());
        MappedTypes.insert(TypedOperandRec);
      }
      for (const auto *Rec : Shape.OperandTypes)
        if (!MappedTypes.contains(Rec))
          PrintFatalError(RuleRec, Rec->getName() + " is not mapped in rules");
    }
  }
}

void RegisterBankEmitter::collectInstrMappings() {
  // Firstly, collect instruction shapes
  // we are only interested in generic instructions with fixed num of operands.
  for (const auto &CGI : llvm::make_pointee_range(Target.getInstructions())) {
    if (!CGI.TheDef->isSubClassOf("GenericInstruction") ||
        CGI.Operands.empty() || CGI.Operands.isVariadic)
      continue;
    if (llvm::all_of(CGI.Operands, [](const CGIOperandList::OperandInfo &Op) {
          return !Op.Rec->isSubClassOf("TypedOperand");
        }))
      continue;
    InstrShape Shape(CGI, TypedOperands);
    auto It = Shapes.try_emplace(Shape).first;
    It->second.insert(&CGI);
    CGIToShapeMap[&CGI] = &It->first;
  }

  // validate all rules
  validateMappingRules();

  for (const auto *Rec :
       Records.getAllDerivedDefinitions("RegBankInstrMapping")) {
    std::vector<const Record *> Opcodes = Rec->getValueAsListOfDefs("Opcodes");
    const CodeGenInstruction &CGI = Target.getInstruction(Opcodes[0]);
    const InstrShape *Shape = CGIToShapeMap[&CGI];
    for (const auto *Opc : Opcodes) {
      std::vector<const Record *> &Rules =
          InstrMappingRules[&Target.getInstruction(Opc)];
      std::vector<const Record *> &AltRules =
          InstrAltMappingRules[&Target.getInstruction(Opc)];
      for (const auto *RuleDef : Rec->getValueAsListOfDefs("Rules")) {
        if (RuleDef->getValueAsBit("IsAlternative"))
          AltRules.push_back(RuleDef);
        else
          Rules.push_back(RuleDef);
      }
    }
    for (const auto *RuleDef : Rec->getValueAsListOfDefs("Rules"))
      BankAssignments[Shape].insert(getBankAssignment(*RuleDef, *Shape));
  }

  for (auto &KV : InstrMappingRules) {
    llvm::stable_sort(KV.second, [](const Record *LHS, const Record *RHS) {
      return LHS->getValueAsInt("Priority") > RHS->getValueAsInt("Priority");
    });
  }
  for (auto &KV : InstrAltMappingRules) {
    llvm::stable_sort(KV.second, [](const Record *LHS, const Record *RHS) {
      return LHS->getValueAsInt("Priority") > RHS->getValueAsInt("Priority");
    });
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

    // operands mappings
    OS << "\nstatic constexpr RegisterBankInfo::PartialMapping\n"
          "GeneratedPartialMappings[] = {\n"
          "  // StartIdx, Length, RegBank\n";
    for (const auto &Bank : Banks) {
      OS << '\n';
      for (const auto &Mapping : Bank.getPartialMappingTable())
        OS << "  {" << Mapping.StartIdx << ", " << Mapping.Length << ", "
           << Bank.getInstanceVarName() << "}, // " << Mapping.Index << '\n';
    }
    OS << "};\n\n";

    for (const auto &Bank : Banks) {
      const std::vector<ValueMapping> &ValueMappings = Bank.getValueMappings();
      OS << "static constexpr RegisterBankInfo::ValueMapping\n"
         << "Generated" << Bank.getName() << "ValueMappings[] = {\n";
      if (ValueMappings.empty())
        OS << "  {nullptr, 0},\n";
      for (const auto &Mapping : ValueMappings)
        OS << "  {&GeneratedPartialMappings[" << Mapping.getPartialMappingIdx()
           << "], " << Mapping.getNumBreakDowns() << "},\n";
      OS << "};\n\n";
    }

    for (const auto &[ShapeIdx, KV] : llvm::enumerate(Shapes)) {
      const auto &[Shape, CGIs] = KV;

      emitOperandMappingForShape(OS, ShapeIdx, Shape);
    }

    OS << "static size_t getValueMappingIdx(unsigned BankID, const TypeSize "
          "&Size, unsigned NumBreakDowns = 1) {\n";
    OS << "  switch (BankID) {\n";
    for (auto &Bank : Banks) {
      std::vector<ValueMapping> ValueMappings = Bank.getValueMappings();
      OS << "    case " << Bank.getEnumeratorName() << ": {\n";
      size_t ScalableMappingSize = Bank.getScalableMappingSize();
      OS << "      [[maybe_unused]] uint64_t S = Size";
      if (ScalableMappingSize == 0) {
        OS << ";\n";
      } else {
        OS << ".getKnownMinValue();\n";
        OS << "      if (Size.isScalable()) \n"
              "        return ";
        auto It = llvm::find_if(ValueMappings,
                                [ScalableMappingSize](const ValueMapping &VM) {
                                  return VM.getSize() == ScalableMappingSize &&
                                         VM.getNumBreakDowns() == 1;
                                });
        OS << std::distance(ValueMappings.begin(), It) << ";\n";
      }
      for (const auto &[Idx, VM] : llvm::enumerate(ValueMappings)) {
        OS << "      if (S <= " << VM.getSize()
           << " && NumBreakDowns == " << VM.getNumBreakDowns() << ") \n"
           << "        return " << Idx << ";\n";
      }

      OS << "      break;\n"
            "    }\n";
    }
    OS << "  }\n"
          "  return -1;\n"
          "}\n\n";

    OS << "[[maybe_unused]] static const RegisterBankInfo::ValueMapping *\n"
          "getGeneratedValueMapping(unsigned BankID, const TypeSize &Size, "
          "unsigned NumBreakDowns = 1) {\n"
          "  static const RegisterBankInfo::ValueMapping InvalidMapping;\n"
          "  size_t Idx = getValueMappingIdx(BankID, Size, NumBreakDowns);\n"
          "  if (Idx == (size_t)-1) \n"
          "    return &InvalidMapping;\n\n"
          "  switch (BankID) {\n"
          "    default:\n"
          "      return &InvalidMapping;\n";
    for (const auto &Bank : Banks) {
      OS << "    case " << Bank.getEnumeratorName() << ":\n"
         << "      return &Generated" << Bank.getName()
         << "ValueMappings[Idx];\n";
    }
    OS << "  }\n";

    OS << "}\n\n";
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
}

std::vector<const RegisterBank *>
RegisterBankEmitter::getBankAssignment(const Record &RuleRec,
                                       const InstrShape &Shape) {
  std::vector<const RegisterBank *> Assignment(Shape.getTypeTupleSize());
  for (const auto *MappingDef : RuleRec.getValueAsListOfDefs("Mappings")) {
    const Record *BankDef = MappingDef->getValueAsDef("Bank");
    const RegisterBank *Bank =
        &*llvm::find_if(Banks, [BankDef](const RegisterBank &Bank) {
          return &Bank.getDef() == BankDef;
        });

    Assignment[TypedOperands[MappingDef->getValueAsDef("Type")].Index] = Bank;
  }
  return Assignment;
}

void RegisterBankEmitter::emitOperandMappingForShape(raw_ostream &OS,
                                                     size_t ShapeIdx,
                                                     const InstrShape &Shape) {
  OS << "static constexpr RegisterBankInfo::ValueMapping\n"
        "GeneratedOperandsMappingsShape"
     << ShapeIdx << "[] = {\n";
  std::set<std::vector<const RegisterBank *>> &Assignments =
      BankAssignments[&Shape];
  if (Assignments.empty()) {
    for (size_t I : llvm::seq(Shape.Shape.size()))
      OS << "  {nullptr, 0}, " << "// " << I << '\n';
    OS << "};\n\n";
    return;
  }

  size_t Idx = 0;
  std::map<std::vector<const RegisterBank *>, size_t> OffsetMap;
  for (const auto &Assignment : Assignments) {
    std::vector<std::vector<ValueMapping>> Components;
    ListSeparator LS(" x ");
    OS << "  // ";
    for (const auto *Bank : Assignment)
      OS << LS << Bank->getName();
    OS << '\n';

    for (const auto *Bank : Assignment)
      Components.push_back(Bank->getValueMappings());

    std::vector<std::vector<ValueMapping>> Product =
        getCartesianProduct(Components);
    OffsetMap[Assignment] = Idx;
    for (const auto &Tup : Product) {
      for (int I : Shape.Shape) {
        if (I >= 0) {
          const ValueMapping &VM = Tup[I];
          OS << "  {&GeneratedPartialMappings[" << VM.getPartialMappingIdx()
             << "], " << VM.getNumBreakDowns() << "}, // " << Idx++ << '\n';
        } else {
          OS << "  {nullptr, 0}, " << "// " << Idx++ << '\n';
        }
      }
      OS << '\n';
    }
  }
  OS << "};\n";
  ShapeToTupleStart[&Shape] = OffsetMap;
}

void RegisterBankEmitter::emitOperandsMappingRules(
    raw_ostream &OS, const CodeGenInstruction &CGI, const InstrShape &Shape,
    const std::vector<const Record *> Rules, bool EmitAlt) {
  std::vector<std::string> LLTSizeNames;
  for (int I : Shape.RepresentativeOperands) {
    std::string RegName = CGI.Operands[I].Name.str() + "Reg";
    RegName[0] = std::toupper(RegName[0]);
    OS << "      Register " << RegName << " = MI.getOperand(" << I
       << ").getReg();\n";
    const TypedOperandInfo &TyOpInfo = TypedOperands[CGI.Operands[I].Rec];
    std::string LLTName = TyOpInfo.getLLTName();
    OS << "      LLT " << LLTName << " = MRI.getType(" << RegName << ");\n";
    std::string LLTSizeName = LLTName + "Size";
    OS << "      TypeSize " << LLTSizeName << " = " << LLTName
       << ".getSizeInBits();\n";
    LLTSizeNames.push_back(LLTSizeName);
  }
  OS << '\n';

  for (const auto *RuleDef : Rules) {
    if (EmitAlt != RuleDef->getValueAsBit("IsAlternative"))
      continue;

    if (StringRef Pred = RuleDef->getValueAsString("Predicate"); !Pred.empty())
      OS << "      if (" << Pred << ") {\n";
    else
      OS << "      {\n";

    std::vector<const RegisterBank *> BankAssignment =
        getBankAssignment(*RuleDef, Shape);
    OS << "        ArrayRef<ValueMapping> OperandsMappings = "
          "OperandsMappingStart.drop_front("
       << ShapeToTupleStart[&Shape][BankAssignment] << ");\n";
    OS << "        std::array Coord = {";
    StringRef TargetName = Target.getName();
    for (const auto &[Idx, Bank] : llvm::enumerate(BankAssignment)) {
      OS << TargetName << "::getValueMappingIdx(" << TargetName
         << "::" << Bank->getEnumeratorName() << ", " << LLTSizeNames[Idx]
         << "),";
    }
    OS << "};\n";
    OS << "        if (llvm::any_of(Coord, [](size_t S) { return S == "
          "(size_t)-1; }))\n"
          "          break;\n";
    // decode instruction mapping position
    ListSeparator Plus(" + ");
    OS << "        size_t Idx = (";
    for (const auto &[Idx, Bank] : llvm::enumerate(BankAssignment)) {
      OS << Plus << "Coord[" << Idx << "] * "
         << llvm::accumulate(ArrayRef(BankAssignment).drop_front(Idx + 1), 1,
                             [](size_t LHS, const RegisterBank *RHS) {
                               return LHS * RHS->getValueMappings().size();
                             });
    }
    OS << ") * " << Shape.getNumOperands() << ";\n";
    StringRef MapID = RuleDef->getValueAsDef("ID")->getValueAsString("Name");
    unsigned Cost = RuleDef->getValueAsInt("Cost");
    if (!EmitAlt) {
      OS << "        return getInstructionMapping(" << MapID << ", " << Cost
         << ", "
            "&OperandsMappings[Idx], "
         << Shape.getNumOperands() << ");\n";
    } else {
      OS << "        Mappings.push_back(&getInstructionMapping(" << MapID << ", "
         << Cost
         << ", "
            "&OperandsMappings[Idx], "
         << Shape.getNumOperands() << "));\n";
    }

    OS << "      }\n\n";
  }
}

void RegisterBankEmitter::emitSearchMethod(raw_ostream &OS,
                                           StringRef TargetName, bool EmitAlt) {
  if (EmitAlt) {
    OS << "RegisterBankInfo::InstructionMappings\n"
       << TargetName
       << "GenRegisterBankInfo::getGeneratedInstrAltMappings(const "
          "MachineInstr &MI) const {\n";
  } else {
    OS << "const RegisterBankInfo::InstructionMapping &\n"
       << TargetName
       << "GenRegisterBankInfo::getGeneratedInstrMapping(const "
          "MachineInstr &MI) const {\n";
  }

  OS << "  const MachineFunction &MF = *MI.getParent()->getParent();\n"
        "  [[maybe_unused]] const MachineRegisterInfo &MRI = "
        "MF.getRegInfo();\n";
  OS << "  const TargetSubtargetInfo &STI = MF.getSubtarget();\n"
        "  [[maybe_unused]] const TargetRegisterInfo &TRI = "
        "*STI.getRegisterInfo();\n\n";
  if (EmitAlt)
    OS << "  InstructionMappings Mappings;\n";
  OS << "  const unsigned Opc = MI.getOpcode();\n\n";

  OS << "  switch(Opc) {\n"
        "    default:\n"
        "      break;\n";

  for (const auto &[ShapeIdx, ShapeCGIs] : llvm::enumerate(Shapes)) {
    const auto &[Shape, CGIs] = ShapeCGIs;
    // instruction with same rules, cluster them in the same case body
    std::map<std::vector<const Record *>, std::set<const CodeGenInstruction *>>
        GroupedInstrMappingRules;
    DenseMap<const CodeGenInstruction *, std::vector<const Record *>>
        &MappingRules = EmitAlt ? InstrAltMappingRules : InstrMappingRules;
    for (auto *CGI : CGIs) {
      auto It = MappingRules.find(CGI);
      if (It == MappingRules.end())
        continue;
      std::vector<const Record *> &Rules = It->second;
      GroupedInstrMappingRules[Rules].insert(CGI);
    }

    for (const auto &[Rules, CGIs] : GroupedInstrMappingRules) {
      if (Rules.empty())
        continue;
      for (const auto *CGI : CGIs)
        OS << "\n    case " << TargetName << "::" << CGI->getName() << ":";
      OS << " {\n"
            "      ArrayRef<ValueMapping> OperandsMappingStart = "
         << TargetName << "::GeneratedOperandsMappingsShape" << ShapeIdx
         << ";\n";
      emitOperandsMappingRules(OS, **CGIs.begin(), Shape, Rules, EmitAlt);
      OS << "      break;\n"
            "    }\n";
    }
  }

  OS << "  }\n";
  if (EmitAlt)
    OS << "  return Mappings;\n";
  else
    OS << "  return getInvalidInstructionMapping();\n";

  OS << "}\n";
}

void RegisterBankEmitter::run(raw_ostream &OS) {
  StringRef TargetName = Target.getName();
  const CodeGenRegBank &RegisterClassHierarchy = Target.getRegBank();
  const CodeGenHwModes &CGH = Target.getHwModes();

  TGTimer &Timer = Records.getTimer();
  Timer.startTimer("Analyze records");
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

    Banks.push_back(Bank);
  }

  if (Banks.empty())
    PrintFatalError("No register banks defined");

  for (const auto *Op : Records.getAllDerivedDefinitions("TypedOperand"))
    TypedOperands[Op] = TypedOperandInfo(*Op);

  collectValueMappings();
  collectInstrMappings();

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
  {
    IfDefEmitter IfDef(OS, "GET_TARGET_REGBANK_SEARCH_METHOD");
    emitSearchMethod(OS, TargetName, /*EmitAlt=*/false);
    OS << '\n';
    emitSearchMethod(OS, TargetName, /*EmitAlt=*/true);
  }
}

static TableGen::Emitter::OptClass<RegisterBankEmitter>
    X("gen-register-bank", "Generate registers bank descriptions");
