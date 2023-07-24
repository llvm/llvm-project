//===- GlobalISelCombinerMatchTableEmitter.cpp - --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Generate a combiner implementation for GlobalISel from a declarative
/// syntax using GlobalISelMatchTable.
///
//===----------------------------------------------------------------------===//

#include "CodeGenInstruction.h"
#include "CodeGenTarget.h"
#include "GlobalISel/CodeExpander.h"
#include "GlobalISel/CodeExpansions.h"
#include "GlobalISel/CombinerUtils.h"
#include "GlobalISelMatchTable.h"
#include "GlobalISelMatchTableExecutorEmitter.h"
#include "SubtargetFeatureInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::gi;

#define DEBUG_TYPE "gicombiner-matchtable-emitter"

extern cl::list<std::string> SelectedCombiners;
extern cl::opt<bool> StopAfterParse;

namespace {
constexpr StringLiteral CXXApplyPrefix = "GICXXCustomAction_CombineApply";
constexpr StringLiteral CXXPredPrefix = "GICXXPred_MI_Predicate_";

std::string getIsEnabledPredicateEnumName(unsigned CombinerRuleID) {
  return "GICXXPred_Simple_IsRule" + to_string(CombinerRuleID) + "Enabled";
}

void declareInstExpansion(CodeExpansions &CE, const InstructionMatcher &IM,
                          StringRef Name) {
  CE.declare(Name, "State.MIs[" + to_string(IM.getInsnVarID()) + "]");
}

void declareOperandExpansion(CodeExpansions &CE, const OperandMatcher &OM,
                             StringRef Name) {
  CE.declare(Name, "State.MIs[" + to_string(OM.getInsnVarID()) +
                       "]->getOperand(" + to_string(OM.getOpIdx()) + ")");
}

//===- MatchData Handling -------------------------------------------------===//

/// Represents MatchData defined by the match stage and required by the apply
/// stage.
///
/// This allows the plumbing of arbitrary data from C++ predicates between the
/// stages.
///
/// When this class is initially created, it only has a pattern symbol and a
/// type. When all of the MatchDatas declarations of a given pattern have been
/// parsed, `AssignVariables` must be called to assign storage variable names to
/// each MatchDataInfo.
class MatchDataInfo {
  StringRef PatternSymbol;
  StringRef Type;
  std::string VarName;

public:
  static constexpr StringLiteral StructTypeName = "MatchInfosTy";
  static constexpr StringLiteral StructName = "MatchInfos";

  MatchDataInfo(StringRef PatternSymbol, StringRef Type)
      : PatternSymbol(PatternSymbol), Type(Type.trim()) {}

  StringRef getPatternSymbol() const { return PatternSymbol; };
  StringRef getType() const { return Type; };

  bool hasVariableName() const { return !VarName.empty(); }
  void setVariableName(StringRef Name) { VarName = Name; }
  StringRef getVariableName() const;

  std::string getQualifiedVariableName() const {
    return StructName.str() + "." + getVariableName().str();
  }

  void print(raw_ostream &OS) const;
  void dump() const { print(dbgs()); }
};

StringRef MatchDataInfo::getVariableName() const {
  assert(hasVariableName());
  return VarName;
}

void MatchDataInfo::print(raw_ostream &OS) const {
  OS << "(MatchDataInfo pattern_symbol:" << PatternSymbol << " type:'" << Type
     << "' var_name:" << (VarName.empty() ? "<unassigned>" : VarName) << ")";
}

/// Pool of type -> variables used to emit MatchData variables declarations.
///
/// e.g. if the map contains "int64_t" -> ["MD0", "MD1"], then two variable
/// declarations must be emitted: `int64_t MD0` and `int64_t MD1`.
///
/// This has a static lifetime and will outlive all the `MatchDataInfo` objects
/// by design. It needs to persist after all `CombineRuleBuilder` objects died
/// so we can emit the variable declarations.
StringMap<std::vector<std::string>> AllMatchDataVars;

// Assign variable names to all MatchDatas used by a pattern. This must be
// called after all MatchData decls have been parsed inside a rule.
//
// Requires an array of MatchDataInfo so we can handle cases where a pattern
// uses multiple instances of the same MatchData type.
void AssignMatchDataVariables(MutableArrayRef<MatchDataInfo> Infos) {
  static unsigned NextVarID = 0;

  StringMap<unsigned> SeenTypes;
  for (auto &I : Infos) {
    unsigned &NumSeen = SeenTypes[I.getType()];
    auto &ExistingVars = AllMatchDataVars[I.getType()];

    if (NumSeen == ExistingVars.size())
      ExistingVars.push_back("MDInfo" + to_string(NextVarID++));

    I.setVariableName(ExistingVars[NumSeen++]);
  }
}

//===- C++ Predicates Handling --------------------------------------------===//

/// Entry into the static pool of all CXX Predicate code. This contains the
/// fully expanded C++ code.
///
/// Each CXXPattern creates a new entry in the pool to store its data, even
/// after the pattern is destroyed.
///
/// Note that CXXPattern trims C++ code, so the Code is already expected to be
/// free of leading/trailing whitespace.
struct CXXPredicateCode {
  CXXPredicateCode(std::string Code, unsigned ID)
      : Code(Code), ID(ID), BaseEnumName("GICombiner" + to_string(ID)) {
    assert(StringRef(Code).trim() == Code &&
           "Code was expected to be trimmed!");
  }

  const std::string Code;
  const unsigned ID;
  const std::string BaseEnumName;

  bool needsUnreachable() const {
    return !StringRef(Code).starts_with("return");
  }

  std::string getEnumNameWithPrefix(StringRef Prefix) const {
    return Prefix.str() + BaseEnumName;
  }
};

using CXXPredicateCodePool =
    DenseMap<hash_code, std::unique_ptr<CXXPredicateCode>>;
CXXPredicateCodePool AllCXXMatchCode;
CXXPredicateCodePool AllCXXApplyCode;

/// Gets an instance of `CXXPredicateCode` for \p Code, or returns an already
/// existing one.
const CXXPredicateCode &getOrInsert(CXXPredicateCodePool &Pool,
                                    std::string Code) {
  // Check if we already have an identical piece of code, if not, create an
  // entry in the pool.
  const auto CodeHash = hash_value(Code);
  if (auto It = Pool.find(CodeHash); It != Pool.end())
    return *It->second;

  const auto ID = Pool.size();
  auto OwnedData = std::make_unique<CXXPredicateCode>(std::move(Code), ID);
  const auto &DataRef = *OwnedData;
  Pool[CodeHash] = std::move(OwnedData);
  return DataRef;
}

/// Sorts a `CXXPredicateCodePool` by their IDs and returns it.
std::vector<const CXXPredicateCode *>
getSorted(const CXXPredicateCodePool &Pool) {
  std::vector<const CXXPredicateCode *> Out;
  std::transform(Pool.begin(), Pool.end(), std::back_inserter(Out),
                 [&](auto &Elt) { return Elt.second.get(); });
  sort(Out, [](const auto *A, const auto *B) { return A->ID < B->ID; });
  return Out;
}

//===- Pattern Base Class -------------------------------------------------===//

// An abstract pattern found in a combine rule. This can be an apply or match
// pattern.
class Pattern {
public:
  enum {
    K_AnyOpcode,
    K_Inst,
    K_CXX,
  };

  virtual ~Pattern() = default;

  unsigned getKind() const { return Kind; }
  const char *getKindName() const;

  bool hasName() const { return !Name.empty(); }
  StringRef getName() const { return Name; }

  virtual void print(raw_ostream &OS, bool PrintName = true) const = 0;
  void dump() const { return print(dbgs()); }

protected:
  Pattern(unsigned Kind, StringRef Name) : Kind(Kind), Name(Name.str()) {
    assert(!Name.empty() && "unnamed pattern!");
  }

  void printImpl(raw_ostream &OS, bool PrintName,
                 function_ref<void()> ContentPrinter) const;

private:
  unsigned Kind;

  // Note: if this ever changes to a StringRef (e.g. allocated in a pool or
  // something), CombineRuleBuilder::verify() needs to be updated as well.
  // It currently checks that the StringRef in the PatternMap references this.
  std::string Name;
};

const char *Pattern::getKindName() const {
  switch (Kind) {
  case K_AnyOpcode:
    return "AnyOpcodePattern";
  case K_Inst:
    return "InstructionPattern";
  case K_CXX:
    return "CXXPattern";
  }

  llvm_unreachable("unknown pattern kind!");
}

void Pattern::printImpl(raw_ostream &OS, bool PrintName,
                        function_ref<void()> ContentPrinter) const {
  OS << "(" << getKindName() << " ";
  if (PrintName)
    OS << "name:" << getName() << " ";
  ContentPrinter();
  OS << ")";
}

//===- AnyOpcodePattern ---------------------------------------------------===//

/// `wip_match_opcode` patterns.
/// This matches one or more opcodes, and does not check any operands
/// whatsoever.
class AnyOpcodePattern : public Pattern {
public:
  AnyOpcodePattern(StringRef Name) : Pattern(K_AnyOpcode, Name) {}

  static bool classof(const Pattern *P) { return P->getKind() == K_AnyOpcode; }

  void addOpcode(const CodeGenInstruction *I) { Insts.push_back(I); }
  const auto &insts() const { return Insts; }

  void print(raw_ostream &OS, bool PrintName = true) const override;

private:
  SmallVector<const CodeGenInstruction *, 4> Insts;
};

void AnyOpcodePattern::print(raw_ostream &OS, bool PrintName) const {
  printImpl(OS, PrintName, [&OS, this]() {
    OS << "["
       << join(map_range(Insts,
                         [](const auto *I) { return I->TheDef->getName(); }),
               ", ")
       << "]";
  });
}

//===- InstructionPattern -------------------------------------------------===//

/// Matches an instruction, e.g. `G_ADD $x, $y, $z`.
///
/// This pattern is simply CodeGenInstruction + a list of operands.
class InstructionPattern : public Pattern {
public:
  struct Operand {
    std::string Name;
    bool IsDef = false;
  };

  InstructionPattern(const CodeGenInstruction &I, StringRef Name)
      : Pattern(K_Inst, Name), I(I) {}

  static bool classof(const Pattern *P) { return P->getKind() == K_Inst; }

  const auto &operands() const { return Operands; }
  void addOperand(StringRef Name);
  unsigned getNumDefs() const { return I.Operands.NumDefs; }

  const CodeGenInstruction &getInst() const { return I; }
  StringRef getInstName() const { return I.TheDef->getName(); }

  void reportUnreachable(ArrayRef<SMLoc> Locs) const;
  bool checkSemantics(ArrayRef<SMLoc> Loc) const;

  void print(raw_ostream &OS, bool PrintName = true) const override;

private:
  const CodeGenInstruction &I;
  SmallVector<Operand, 4> Operands;
};

void InstructionPattern::addOperand(StringRef Name) {
  const bool IsDef = Operands.size() < getNumDefs();
  Operands.emplace_back(Operand{Name.str(), IsDef});
}

void InstructionPattern::reportUnreachable(ArrayRef<SMLoc> Locs) const {
  PrintError(Locs, "Instruction pattern '" + getName() +
                       "' is unreachable from the pattern root!");
}

bool InstructionPattern::checkSemantics(ArrayRef<SMLoc> Loc) const {
  unsigned NumExpectedOperands = I.Operands.size();
  if (NumExpectedOperands != Operands.size()) {

    PrintError(Loc, "'" + getInstName() + "' expected " +
                        Twine(NumExpectedOperands) + " operands, got " +
                        Twine(Operands.size()));
    return false;
  }
  return true;
}

void InstructionPattern::print(raw_ostream &OS, bool PrintName) const {
  printImpl(OS, PrintName, [&OS, this]() {
    OS << "inst:" << I.TheDef->getName() << " operands:["
       << join(map_range(Operands,
                         [](const auto &O) {
                           return (O.IsDef ? "<def>" : "") + O.Name;
                         }),
               ", ")
       << "]";
  });
}

//===- CXXPattern ---------------------------------------------------------===//

/// Raw C++ code which may need some expansions.
///
///   e.g. [{ return isFooBux(${src}.getReg()); }]
///
/// For the expanded code, \see CXXPredicateCode. CXXPredicateCode objects are
/// created through `expandCode`.
///
/// \see CodeExpander and \see CodeExpansions for more information on code
/// expansions.
///
/// This object has two purposes:
///   - Represent C++ code as a pattern entry.
///   - Be a factory for expanded C++ code.
///     - It's immutable and only holds the raw code so we can expand the same
///       CXX pattern multiple times if we need to.
///
/// Note that the code is always trimmed in the constructor, so leading and
/// trailing whitespaces are removed. This removes bloat in the output, avoids
/// formatting issues, but also allows us to check things like
/// `.startswith("return")` trivially without worrying about spaces.
class CXXPattern : public Pattern {
public:
  CXXPattern(const StringInit &Code, StringRef Name, bool IsApply)
      : CXXPattern(Code.getAsUnquotedString(), Name, IsApply) {}

  CXXPattern(StringRef Code, StringRef Name, bool IsApply)
      : Pattern(K_CXX, Name), IsApply(IsApply), RawCode(Code.trim().str()) {}

  static bool classof(const Pattern *P) { return P->getKind() == K_CXX; }

  bool isApply() const { return IsApply; }
  StringRef getRawCode() const { return RawCode; }

  /// Expands raw code, replacing things such as `${foo}` with their
  /// substitution in \p CE.
  ///
  /// \param CE     Map of Code Expansions
  /// \param Locs   SMLocs for the Code Expander, in case it needs to emit
  ///               diagnostics.
  /// \return A CXXPredicateCode object that contains the expanded code. Note
  /// that this may or may not insert a new object. All CXXPredicateCode objects
  /// are held in a set to avoid emitting duplicate C++ code.
  const CXXPredicateCode &expandCode(const CodeExpansions &CE,
                                     ArrayRef<SMLoc> Locs) const;

  void print(raw_ostream &OS, bool PrintName = true) const override;

private:
  bool IsApply;
  std::string RawCode;
};

const CXXPredicateCode &CXXPattern::expandCode(const CodeExpansions &CE,
                                               ArrayRef<SMLoc> Locs) const {
  std::string Result;
  raw_string_ostream OS(Result);
  CodeExpander Expander(RawCode, CE, Locs, /*ShowExpansions*/ false);
  Expander.emit(OS);
  return getOrInsert(IsApply ? AllCXXApplyCode : AllCXXMatchCode,
                     std::move(Result));
}

void CXXPattern::print(raw_ostream &OS, bool PrintName) const {
  printImpl(OS, PrintName, [&OS, this] {
    OS << (IsApply ? "apply" : "match") << " code:\"";
    printEscapedString(getRawCode(), OS);
    OS << "\"";
  });
}

//===- CombineRuleBuilder -------------------------------------------------===//

/// Helper for CombineRuleBuilder.
///
/// Represents information about an operand.
/// Operands with no MatchPat are considered live-in to the pattern.
struct OperandTableEntry {
  // The matcher pattern that defines this operand.
  // null for live-ins.
  InstructionPattern *MatchPat = nullptr;
  // The apply pattern that (re)defines this operand.
  // This can only be non-null if MatchPat is.
  InstructionPattern *ApplyPat = nullptr;

  bool isLiveIn() const { return !MatchPat; }
};

/// Parses combine rule and builds a small intermediate representation to tie
/// patterns together and emit RuleMatchers to match them. This may emit more
/// than one RuleMatcher, e.g. for `wip_match_opcode`.
///
/// Memory management for `Pattern` objects is done through `std::unique_ptr`.
/// In most cases, there are two stages to a pattern's lifetime:
///   - Creation in a `parse` function
///     - The unique_ptr is stored in a variable, and may be destroyed if the
///       pattern is found to be semantically invalid.
///   - Ownership transfer into a `PatternMap`
///     - Once a pattern is moved into either the map of Match or Apply
///       patterns, it is known to be valid and it never moves back.
class CombineRuleBuilder {
public:
  using PatternMap = MapVector<StringRef, std::unique_ptr<Pattern>>;

  CombineRuleBuilder(const CodeGenTarget &CGT,
                     SubtargetFeatureInfoMap &SubtargetFeatures,
                     Record &RuleDef, unsigned ID,
                     std::vector<RuleMatcher> &OutRMs)
      : CGT(CGT), SubtargetFeatures(SubtargetFeatures), RuleDef(RuleDef),
        RuleID(ID), OutRMs(OutRMs) {}

  /// Parses all fields in the RuleDef record.
  bool parseAll();

  /// Emits all RuleMatchers into the vector of RuleMatchers passed in the
  /// constructor.
  bool emitRuleMatchers();

  void print(raw_ostream &OS) const;
  void dump() const { print(dbgs()); }

  /// Debug-only verification of invariants.
  void verify() const;

private:
  void PrintError(Twine Msg) const { ::PrintError(RuleDef.getLoc(), Msg); }

  /// Adds the expansions from \see MatchDatas to \p CE.
  void declareAllMatchDatasExpansions(CodeExpansions &CE) const;

  /// Adds \p P to \p IM, expanding its code using \p CE.
  void addCXXPredicate(InstructionMatcher &IM, const CodeExpansions &CE,
                       const CXXPattern &P);

  /// Generates a name for anonymous patterns.
  ///
  /// e.g. (G_ADD $x, $y, $z):$foo is a pattern named "foo", but if ":$foo" is
  /// absent, then the pattern is anonymous and this is used to assign it a
  /// name.
  std::string makeAnonPatName(StringRef Prefix) const;
  mutable unsigned AnonIDCnt = 0;

  /// Creates a new RuleMatcher with some boilerplate
  /// settings/actions/predicates, and and adds it to \p OutRMs.
  /// \see addFeaturePredicates too.
  ///
  /// \param AdditionalComment Comment string to be added to the
  ///        `DebugCommentAction`.
  RuleMatcher &addRuleMatcher(Twine AdditionalComment = "");
  bool addFeaturePredicates(RuleMatcher &M);

  bool findRoots();
  bool buildOperandsTable();

  bool parseDefs(DagInit &Def);
  bool parseMatch(DagInit &Match);
  bool parseApply(DagInit &Apply);

  std::unique_ptr<Pattern> parseInstructionMatcher(const Init &Arg,
                                                   StringRef PatName);
  std::unique_ptr<Pattern> parseWipMatchOpcodeMatcher(const Init &Arg,
                                                      StringRef PatName);

  bool emitMatchPattern(CodeExpansions &CE, const InstructionPattern &IP);
  bool emitMatchPattern(CodeExpansions &CE, const AnyOpcodePattern &AOP);

  bool emitApplyPatterns(CodeExpansions &CE, RuleMatcher &M);

  // Recursively visits InstructionPattern from P to build up the
  // RuleMatcher/InstructionMatcher. May create new InstructionMatchers as
  // needed.
  bool emitInstructionMatchPattern(CodeExpansions &CE, RuleMatcher &M,
                                   InstructionMatcher &IM,
                                   const InstructionPattern &P,
                                   DenseSet<const Pattern *> &SeenPats);

  const CodeGenTarget &CGT;
  SubtargetFeatureInfoMap &SubtargetFeatures;
  Record &RuleDef;
  const unsigned RuleID;
  std::vector<RuleMatcher> &OutRMs;

  // For InstructionMatcher::addOperand
  unsigned AllocatedTemporariesBaseID = 0;

  /// The root of the pattern.
  StringRef RootName;

  /// These maps have ownership of the actual Pattern objects.
  /// They both map a Pattern's name to the Pattern instance.
  PatternMap MatchPats;
  PatternMap ApplyPats;

  /// Set by findRoots.
  Pattern *MatchRoot = nullptr;

  MapVector<StringRef, OperandTableEntry> OperandTable;
  SmallVector<MatchDataInfo, 2> MatchDatas;
};

bool CombineRuleBuilder::parseAll() {
  if (!parseDefs(*RuleDef.getValueAsDag("Defs")))
    return false;
  if (!parseMatch(*RuleDef.getValueAsDag("Match")))
    return false;
  if (!parseApply(*RuleDef.getValueAsDag("Apply")))
    return false;
  if (!buildOperandsTable())
    return false;
  if (!findRoots())
    return false;
  LLVM_DEBUG(verify());
  return true;
}

bool CombineRuleBuilder::emitRuleMatchers() {
  assert(MatchRoot);
  CodeExpansions CE;
  declareAllMatchDatasExpansions(CE);

  switch (MatchRoot->getKind()) {
  case Pattern::K_AnyOpcode: {
    if (!emitMatchPattern(CE, *cast<AnyOpcodePattern>(MatchRoot)))
      return false;
    break;
  }
  case Pattern::K_Inst:
    if (!emitMatchPattern(CE, *cast<InstructionPattern>(MatchRoot)))
      return false;
    break;
  case Pattern::K_CXX:
    PrintError("C++ code cannot be the root of a pattern!");
    return false;
  default:
    llvm_unreachable("unknown pattern kind!");
  }

  return true;
}

void CombineRuleBuilder::print(raw_ostream &OS) const {
  OS << "(CombineRule name:" << RuleDef.getName() << " id:" << RuleID
     << " root:" << RootName << "\n";

  OS << "  (MatchDatas ";
  if (MatchDatas.empty())
    OS << "<empty>)\n";
  else {
    OS << "\n";
    for (const auto &MD : MatchDatas) {
      OS << "    ";
      MD.print(OS);
      OS << "\n";
    }
    OS << "  )\n";
  }

  const auto DumpPats = [&](StringRef Name, const PatternMap &Pats) {
    OS << "  (" << Name << " ";
    if (Pats.empty()) {
      OS << "<empty>)\n";
      return;
    }

    OS << "\n";
    for (const auto &[Name, Pat] : Pats) {
      OS << "    ";
      if (Pat.get() == MatchRoot)
        OS << "<root>";
      OS << Name << ":";
      Pat->print(OS, /*PrintName=*/false);
      OS << "\n";
    }
    OS << "  )\n";
  };

  DumpPats("MatchPats", MatchPats);
  DumpPats("ApplyPats", ApplyPats);

  OS << "  (OperandTable ";
  if (OperandTable.empty())
    OS << "<empty>)\n";
  else {
    OS << "\n";
    for (const auto &[Key, Val] : OperandTable) {
      OS << "    [" << Key;
      if (const auto *P = Val.MatchPat)
        OS << " match_pat:" << P->getName();
      if (const auto *P = Val.ApplyPat)
        OS << " apply_pat:" << P->getName();
      if (Val.isLiveIn())
        OS << " live-in";
      OS << "]\n";
    }
    OS << "  )\n";
  }

  OS << ")\n";
}

void CombineRuleBuilder::verify() const {
  const auto VerifyPats = [&](const PatternMap &Pats) {
    for (const auto &[Name, Pat] : Pats) {
      if (!Pat)
        PrintFatalError("null pattern in pattern map!");

      if (Name != Pat->getName()) {
        Pat->dump();
        PrintFatalError("Pattern name mismatch! Map name: " + Name +
                        ", Pat name: " + Pat->getName());
      }

      // As an optimization, the PatternMaps don't re-allocate the PatternName
      // string. They simply reference the std::string inside Pattern. Ensure
      // this is the case to avoid memory issues.
      if (Name.data() != Pat->getName().data()) {
        dbgs() << "Map StringRef: '" << Name << "' @ "
               << (const void *)Name.data() << "\n";
        dbgs() << "Pat String: '" << Pat->getName() << "' @ "
               << (const void *)Pat->getName().data() << "\n";
        PrintFatalError("StringRef stored in the PatternMap is not referencing "
                        "the same string as its Pattern!");
      }
    }
  };

  VerifyPats(MatchPats);
  VerifyPats(ApplyPats);

  for (const auto &[Name, Op] : OperandTable) {
    if (Op.ApplyPat && !Op.MatchPat) {
      dump();
      PrintFatalError("Operand " + Name +
                      " has an apply pattern, but no match pattern!");
    }
  }
}

bool CombineRuleBuilder::addFeaturePredicates(RuleMatcher &M) {
  if (!RuleDef.getValue("Predicates"))
    return true;

  ListInit *Preds = RuleDef.getValueAsListInit("Predicates");
  for (Init *I : Preds->getValues()) {
    if (DefInit *Pred = dyn_cast<DefInit>(I)) {
      Record *Def = Pred->getDef();
      if (!Def->isSubClassOf("Predicate")) {
        ::PrintError(Def->getLoc(), "Unknown 'Predicate' Type");
        return false;
      }

      if (Def->getValueAsString("CondString").empty())
        continue;

      if (SubtargetFeatures.count(Def) == 0) {
        SubtargetFeatures.emplace(
            Def, SubtargetFeatureInfo(Def, SubtargetFeatures.size()));
      }

      M.addRequiredFeature(Def);
    }
  }

  return true;
}

void CombineRuleBuilder::declareAllMatchDatasExpansions(
    CodeExpansions &CE) const {
  for (const auto &MD : MatchDatas)
    CE.declare(MD.getPatternSymbol(), MD.getQualifiedVariableName());
}

void CombineRuleBuilder::addCXXPredicate(InstructionMatcher &IM,
                                         const CodeExpansions &CE,
                                         const CXXPattern &P) {
  const auto &ExpandedCode = P.expandCode(CE, RuleDef.getLoc());
  IM.addPredicate<GenericInstructionPredicateMatcher>(
      ExpandedCode.getEnumNameWithPrefix(CXXPredPrefix));
}

std::string CombineRuleBuilder::makeAnonPatName(StringRef Prefix) const {
  return to_string("__anon_pat_" + Prefix + "_" + to_string(RuleID) + "_" +
                   to_string(AnonIDCnt++));
}

RuleMatcher &CombineRuleBuilder::addRuleMatcher(Twine AdditionalComment) {
  auto &RM = OutRMs.emplace_back(RuleDef.getLoc());
  addFeaturePredicates(RM);
  RM.addRequiredSimplePredicate(getIsEnabledPredicateEnumName(RuleID));
  const std::string AdditionalCommentStr = AdditionalComment.str();
  RM.addAction<DebugCommentAction>(
      "Combiner Rule #" + to_string(RuleID) + ": " + RuleDef.getName().str() +
      (AdditionalCommentStr.empty() ? "" : "; " + AdditionalCommentStr));
  return RM;
}

bool CombineRuleBuilder::findRoots() {
  // Look by pattern name, e.g.
  //    (G_FNEG $x, $y):$root
  if (auto It = MatchPats.find(RootName); It != MatchPats.end()) {
    MatchRoot = It->second.get();
    return true;
  }

  // Look by def:
  //    (G_FNEG $root, $y)
  auto It = OperandTable.find(RootName);
  if (It == OperandTable.end()) {
    PrintError("Cannot find root '" + RootName + "' in match patterns!");
    return false;
  }

  if (!It->second.MatchPat) {
    PrintError("Cannot use live-in operand '" + RootName +
               "' as match pattern root!");
    return false;
  }

  MatchRoot = It->second.MatchPat;
  return true;
}

bool CombineRuleBuilder::buildOperandsTable() {
  // Walk each instruction pattern
  for (auto &[_, P] : MatchPats) {
    auto *IP = dyn_cast<InstructionPattern>(P.get());
    if (!IP)
      continue;
    for (const auto &Operand : IP->operands()) {
      // Create an entry, no matter if it's a use or a def.
      auto &Entry = OperandTable[Operand.Name];

      // We only need to do additional checking on defs, though.
      if (!Operand.IsDef)
        continue;

      if (Entry.MatchPat) {
        PrintError("Operand '" + Operand.Name +
                   "' is defined multiple times in the 'match' patterns");
        return false;
      }
      Entry.MatchPat = IP;
    }
  }

  for (auto &[_, P] : ApplyPats) {
    auto *IP = dyn_cast<InstructionPattern>(P.get());
    if (!IP)
      continue;
    for (const auto &Operand : IP->operands()) {
      // Create an entry, no matter if it's a use or a def.
      auto &Entry = OperandTable[Operand.Name];

      // We only need to do additional checking on defs, though.
      if (!Operand.IsDef)
        continue;

      if (!Entry.MatchPat) {
        PrintError("Cannot define live-in operand '" + Operand.Name +
                   "' in the 'apply' pattern");
        return false;
      }
      if (Entry.ApplyPat) {
        PrintError("Operand '" + Operand.Name +
                   "' is defined multiple times in the 'apply' patterns");
        return false;
      }
      Entry.ApplyPat = IP;
    }
  }

  return true;
}

bool CombineRuleBuilder::parseDefs(DagInit &Def) {
  if (Def.getOperatorAsDef(RuleDef.getLoc())->getName() != "defs") {
    PrintError("Expected defs operator");
    return false;
  }

  SmallVector<StringRef> Roots;
  for (unsigned I = 0, E = Def.getNumArgs(); I < E; ++I) {
    if (isSpecificDef(*Def.getArg(I), "root")) {
      Roots.emplace_back(Def.getArgNameStr(I));
      continue;
    }

    // Subclasses of GIDefMatchData should declare that this rule needs to pass
    // data from the match stage to the apply stage, and ensure that the
    // generated matcher has a suitable variable for it to do so.
    if (Record *MatchDataRec =
            getDefOfSubClass(*Def.getArg(I), "GIDefMatchData")) {
      MatchDatas.emplace_back(Def.getArgNameStr(I),
                              MatchDataRec->getValueAsString("Type"));
      continue;
    }

    // Otherwise emit an appropriate error message.
    if (getDefOfSubClass(*Def.getArg(I), "GIDefKind"))
      PrintError("This GIDefKind not implemented in tablegen");
    else if (getDefOfSubClass(*Def.getArg(I), "GIDefKindWithArgs"))
      PrintError("This GIDefKindWithArgs not implemented in tablegen");
    else
      PrintError("Expected a subclass of GIDefKind or a sub-dag whose "
                 "operator is of type GIDefKindWithArgs");
    return false;
  }

  if (Roots.size() != 1) {
    PrintError("Combine rules must have exactly one root");
    return false;
  }

  RootName = Roots.front();

  // Assign variables to all MatchDatas.
  AssignMatchDataVariables(MatchDatas);
  return true;
}

bool CombineRuleBuilder::parseMatch(DagInit &Match) {
  if (Match.getOperatorAsDef(RuleDef.getLoc())->getName() != "match") {
    PrintError("Expected match operator");
    return false;
  }

  if (Match.getNumArgs() == 0) {
    PrintError("Matcher is empty");
    return false;
  }

  // The match section consists of a list of matchers and predicates. Parse each
  // one and add the equivalent GIMatchDag nodes, predicates, and edges.
  bool HasOpcodeMatcher = false;
  for (unsigned I = 0; I < Match.getNumArgs(); ++I) {
    Init *Arg = Match.getArg(I);
    std::string Name = Match.getArgName(I)
                           ? Match.getArgName(I)->getValue().str()
                           : makeAnonPatName("match");

    if (MatchPats.contains(Name)) {
      PrintError("'" + Name + "' match pattern defined more than once!");
      return false;
    }

    if (auto Pat = parseInstructionMatcher(*Arg, Name)) {
      MatchPats[Pat->getName()] = std::move(Pat);
      continue;
    }

    if (auto Pat = parseWipMatchOpcodeMatcher(*Arg, Name)) {
      if (HasOpcodeMatcher) {
        PrintError("wip_opcode_match can only be present once");
        return false;
      }
      HasOpcodeMatcher = true;
      MatchPats[Pat->getName()] = std::move(Pat);
      continue;
    }

    // Parse arbitrary C++ code
    if (const auto *StringI = dyn_cast<StringInit>(Arg)) {
      auto CXXPat =
          std::make_unique<CXXPattern>(*StringI, Name, /*IsApply*/ false);
      if (!CXXPat->getRawCode().contains("return ")) {
        PrintWarning(RuleDef.getLoc(),
                     "'match' C++ code does not seem to return!");
      }
      MatchPats[CXXPat->getName()] = std::move(CXXPat);
      continue;
    }

    // TODO: don't print this on, e.g. bad operand count in inst pat
    PrintError("Expected a subclass of GIMatchKind or a sub-dag whose "
               "operator is either of a GIMatchKindWithArgs or Instruction");
    PrintNote("Pattern was `" + Arg->getAsString() + "'");
    return false;
  }

  return true;
}

bool CombineRuleBuilder::parseApply(DagInit &Apply) {
  // Currently we only support C++ :(
  if (Apply.getOperatorAsDef(RuleDef.getLoc())->getName() != "apply") {
    PrintError("Expected 'apply' operator in Apply DAG");
    return false;
  }

  if (Apply.getNumArgs() != 1) {
    PrintError("Expected exactly 1 argument in 'apply'");
    return false;
  }

  const StringInit *Code = dyn_cast<StringInit>(Apply.getArg(0));
  auto Pat = std::make_unique<CXXPattern>(*Code, makeAnonPatName("apply"),
                                          /*IsApply*/ true);
  ApplyPats[Pat->getName()] = std::move(Pat);
  return true;
}

std::unique_ptr<Pattern>
CombineRuleBuilder::parseInstructionMatcher(const Init &Arg, StringRef Name) {
  const DagInit *Matcher = getDagWithOperatorOfSubClass(Arg, "Instruction");
  if (!Matcher)
    return nullptr;

  auto &Instr = CGT.getInstruction(Matcher->getOperatorAsDef(RuleDef.getLoc()));
  auto Pat = std::make_unique<InstructionPattern>(Instr, Name);

  for (const auto &NameInit : Matcher->getArgNames())
    Pat->addOperand(NameInit->getAsUnquotedString());

  if (!Pat->checkSemantics(RuleDef.getLoc()))
    return nullptr;

  return std::move(Pat);
}

std::unique_ptr<Pattern>
CombineRuleBuilder::parseWipMatchOpcodeMatcher(const Init &Arg,
                                               StringRef Name) {
  const DagInit *Matcher = getDagWithSpecificOperator(Arg, "wip_match_opcode");
  if (!Matcher)
    return nullptr;

  if (Matcher->getNumArgs() == 0) {
    PrintError("Empty wip_match_opcode");
    return nullptr;
  }

  // Each argument is an opcode that can match.
  auto Result = std::make_unique<AnyOpcodePattern>(Name);
  for (const auto &Arg : Matcher->getArgs()) {
    Record *OpcodeDef = getDefOfSubClass(*Arg, "Instruction");
    if (OpcodeDef) {
      Result->addOpcode(&CGT.getInstruction(OpcodeDef));
      continue;
    }

    PrintError("Arguments to wip_match_opcode must be instructions");
    return nullptr;
  }

  return std::move(Result);
}

bool CombineRuleBuilder::emitMatchPattern(CodeExpansions &CE,
                                          const InstructionPattern &IP) {
  auto &M = addRuleMatcher();
  InstructionMatcher &IM = M.addInstructionMatcher("root");
  declareInstExpansion(CE, IM, IP.getName());

  DenseSet<const Pattern *> SeenPats;
  if (!emitInstructionMatchPattern(CE, M, IM, IP, SeenPats))
    return false;

  // Emit remaining patterns
  for (auto &[_, Pat] : MatchPats) {
    if (SeenPats.contains(Pat.get()))
      continue;

    switch (Pat->getKind()) {
    case Pattern::K_AnyOpcode:
      PrintError("wip_match_opcode can not be used with instruction patterns!");
      return false;
    case Pattern::K_Inst:
      cast<InstructionPattern>(Pat.get())->reportUnreachable(RuleDef.getLoc());
      return false;
    case Pattern::K_CXX: {
      addCXXPredicate(IM, CE, *cast<CXXPattern>(Pat.get()));
      continue;
    }
    default:
      llvm_unreachable("unknown pattern kind!");
    }
  }

  return emitApplyPatterns(CE, M);
}

bool CombineRuleBuilder::emitMatchPattern(CodeExpansions &CE,
                                          const AnyOpcodePattern &AOP) {

  for (const CodeGenInstruction *CGI : AOP.insts()) {
    auto &M = addRuleMatcher("wip_match_opcode alternative '" +
                             CGI->TheDef->getName() + "'");

    InstructionMatcher &IM = M.addInstructionMatcher(AOP.getName());
    declareInstExpansion(CE, IM, AOP.getName());
    // declareInstExpansion needs to be identical, otherwise we need to create a
    // CodeExpansions object here instead.
    assert(IM.getInsnVarID() == 0);

    IM.addPredicate<InstructionOpcodeMatcher>(CGI);

    // Emit remaining patterns.
    for (auto &[_, Pat] : MatchPats) {
      if (Pat.get() == &AOP)
        continue;

      switch (Pat->getKind()) {
      case Pattern::K_AnyOpcode:
        PrintError("wip_match_opcode can only be present once!");
        return false;
      case Pattern::K_Inst:
        cast<InstructionPattern>(Pat.get())->reportUnreachable(
            RuleDef.getLoc());
        return false;
      case Pattern::K_CXX: {
        addCXXPredicate(IM, CE, *cast<CXXPattern>(Pat.get()));
        break;
      }
      default:
        llvm_unreachable("unknown pattern kind!");
      }
    }

    if (!emitApplyPatterns(CE, M))
      return false;
  }

  return true;
}

bool CombineRuleBuilder::emitApplyPatterns(CodeExpansions &CE, RuleMatcher &M) {
  for (auto &[_, Pat] : ApplyPats) {
    switch (Pat->getKind()) {
    case Pattern::K_AnyOpcode:
    case Pattern::K_Inst:
      llvm_unreachable("Unsupported pattern kind in output pattern!");
    case Pattern::K_CXX: {
      CXXPattern *CXXPat = cast<CXXPattern>(Pat.get());
      const auto &ExpandedCode = CXXPat->expandCode(CE, RuleDef.getLoc());
      M.addAction<CustomCXXAction>(
          ExpandedCode.getEnumNameWithPrefix(CXXApplyPrefix));
      continue;
    }
    default:
      llvm_unreachable("Unknown pattern kind!");
    }
  }

  return true;
}

bool CombineRuleBuilder::emitInstructionMatchPattern(
    CodeExpansions &CE, RuleMatcher &M, InstructionMatcher &IM,
    const InstructionPattern &P, DenseSet<const Pattern *> &SeenPats) {
  if (SeenPats.contains(&P))
    return true;

  SeenPats.insert(&P);

  IM.addPredicate<InstructionOpcodeMatcher>(&P.getInst());
  declareInstExpansion(CE, IM, P.getName());

  unsigned OpIdx = 0;
  for (auto &O : P.operands()) {
    auto &OpTableEntry = OperandTable.find(O.Name)->second;

    OperandMatcher &OM =
        IM.addOperand(OpIdx++, O.Name, AllocatedTemporariesBaseID++);
    declareOperandExpansion(CE, OM, O.Name);

    if (O.IsDef)
      continue;

    if (InstructionPattern *DefPat = OpTableEntry.MatchPat) {
      auto InstOpM = OM.addPredicate<InstructionOperandMatcher>(M, O.Name);
      if (!InstOpM) {
        // TODO: copy-pasted from GlobalISelEmitter.cpp. Is it still relevant
        // here?
        PrintError("Nested instruction '" + DefPat->getName() +
                   "' cannot be the same as another operand '" + O.Name + "'");
        return false;
      }

      if (!emitInstructionMatchPattern(CE, M, (*InstOpM)->getInsnMatcher(),
                                       *DefPat, SeenPats))
        return false;
    }
  }

  return true;
}

//===- GICombinerEmitter --------------------------------------------------===//

/// This class is essentially the driver. It fetches all TableGen records, calls
/// CombineRuleBuilder to build the MatchTable's RuleMatchers, then creates the
/// MatchTable & emits it. It also handles emitting all the supporting code such
/// as the list of LLTs, the CXXPredicates, etc.
class GICombinerEmitter final : public GlobalISelMatchTableExecutorEmitter {
  RecordKeeper &Records;
  StringRef Name;
  const CodeGenTarget &Target;
  Record *Combiner;
  unsigned NextRuleID = 0;

  // List all combine rules (ID, name) imported.
  // Note that the combiner rule ID is different from the RuleMatcher ID. The
  // latter is internal to the MatchTable, the former is the canonical ID of the
  // combine rule used to disable/enable it.
  std::vector<std::pair<unsigned, std::string>> AllCombineRules;

  MatchTable buildMatchTable(MutableArrayRef<RuleMatcher> Rules);

  void emitRuleConfigImpl(raw_ostream &OS);

  void emitAdditionalImpl(raw_ostream &OS) override;

  void emitMIPredicateFns(raw_ostream &OS) override;
  void emitI64ImmPredicateFns(raw_ostream &OS) override;
  void emitAPFloatImmPredicateFns(raw_ostream &OS) override;
  void emitAPIntImmPredicateFns(raw_ostream &OS) override;
  void emitTestSimplePredicate(raw_ostream &OS) override;
  void emitRunCustomAction(raw_ostream &OS) override;

  void emitAdditionalTemporariesDecl(raw_ostream &OS,
                                     StringRef Indent) override;

  const CodeGenTarget &getTarget() const override { return Target; }
  StringRef getClassName() const override {
    return Combiner->getValueAsString("Classname");
  }

  std::string getRuleConfigClassName() const {
    return getClassName().str() + "RuleConfig";
  }

  void gatherRules(std::vector<RuleMatcher> &Rules,
                   const std::vector<Record *> &&RulesAndGroups);

public:
  explicit GICombinerEmitter(RecordKeeper &RK, const CodeGenTarget &Target,
                             StringRef Name, Record *Combiner);
  ~GICombinerEmitter() {}

  void run(raw_ostream &OS);
};

void GICombinerEmitter::emitRuleConfigImpl(raw_ostream &OS) {
  OS << "struct " << getRuleConfigClassName() << " {\n"
     << "  SparseBitVector<> DisabledRules;\n\n"
     << "  bool isRuleEnabled(unsigned RuleID) const;\n"
     << "  bool parseCommandLineOption();\n"
     << "  bool setRuleEnabled(StringRef RuleIdentifier);\n"
     << "  bool setRuleDisabled(StringRef RuleIdentifier);\n"
     << "};\n\n";

  std::vector<std::pair<std::string, std::string>> Cases;
  Cases.reserve(AllCombineRules.size());

  for (const auto &[ID, Name] : AllCombineRules)
    Cases.emplace_back(Name, "return " + to_string(ID) + ";\n");

  OS << "static std::optional<uint64_t> getRuleIdxForIdentifier(StringRef "
        "RuleIdentifier) {\n"
     << "  uint64_t I;\n"
     << "  // getAtInteger(...) returns false on success\n"
     << "  bool Parsed = !RuleIdentifier.getAsInteger(0, I);\n"
     << "  if (Parsed)\n"
     << "    return I;\n\n"
     << "#ifndef NDEBUG\n";
  StringMatcher Matcher("RuleIdentifier", Cases, OS);
  Matcher.Emit();
  OS << "#endif // ifndef NDEBUG\n\n"
     << "  return std::nullopt;\n"
     << "}\n";

  OS << "static std::optional<std::pair<uint64_t, uint64_t>> "
        "getRuleRangeForIdentifier(StringRef RuleIdentifier) {\n"
     << "  std::pair<StringRef, StringRef> RangePair = "
        "RuleIdentifier.split('-');\n"
     << "  if (!RangePair.second.empty()) {\n"
     << "    const auto First = "
        "getRuleIdxForIdentifier(RangePair.first);\n"
     << "    const auto Last = "
        "getRuleIdxForIdentifier(RangePair.second);\n"
     << "    if (!First || !Last)\n"
     << "      return std::nullopt;\n"
     << "    if (First >= Last)\n"
     << "      report_fatal_error(\"Beginning of range should be before "
        "end of range\");\n"
     << "    return {{*First, *Last + 1}};\n"
     << "  }\n"
     << "  if (RangePair.first == \"*\") {\n"
     << "    return {{0, " << AllCombineRules.size() << "}};\n"
     << "  }\n"
     << "  const auto I = getRuleIdxForIdentifier(RangePair.first);\n"
     << "  if (!I)\n"
     << "    return std::nullopt;\n"
     << "  return {{*I, *I + 1}};\n"
     << "}\n\n";

  for (bool Enabled : {true, false}) {
    OS << "bool " << getRuleConfigClassName() << "::setRule"
       << (Enabled ? "Enabled" : "Disabled") << "(StringRef RuleIdentifier) {\n"
       << "  auto MaybeRange = getRuleRangeForIdentifier(RuleIdentifier);\n"
       << "  if (!MaybeRange)\n"
       << "    return false;\n"
       << "  for (auto I = MaybeRange->first; I < MaybeRange->second; ++I)\n"
       << "    DisabledRules." << (Enabled ? "reset" : "set") << "(I);\n"
       << "  return true;\n"
       << "}\n\n";
  }

  OS << "static std::vector<std::string> " << Name << "Option;\n"
     << "static cl::list<std::string> " << Name << "DisableOption(\n"
     << "    \"" << Name.lower() << "-disable-rule\",\n"
     << "    cl::desc(\"Disable one or more combiner rules temporarily in "
     << "the " << Name << " pass\"),\n"
     << "    cl::CommaSeparated,\n"
     << "    cl::Hidden,\n"
     << "    cl::cat(GICombinerOptionCategory),\n"
     << "    cl::callback([](const std::string &Str) {\n"
     << "      " << Name << "Option.push_back(Str);\n"
     << "    }));\n"
     << "static cl::list<std::string> " << Name << "OnlyEnableOption(\n"
     << "    \"" << Name.lower() << "-only-enable-rule\",\n"
     << "    cl::desc(\"Disable all rules in the " << Name
     << " pass then re-enable the specified ones\"),\n"
     << "    cl::Hidden,\n"
     << "    cl::cat(GICombinerOptionCategory),\n"
     << "    cl::callback([](const std::string &CommaSeparatedArg) {\n"
     << "      StringRef Str = CommaSeparatedArg;\n"
     << "      " << Name << "Option.push_back(\"*\");\n"
     << "      do {\n"
     << "        auto X = Str.split(\",\");\n"
     << "        " << Name << "Option.push_back((\"!\" + X.first).str());\n"
     << "        Str = X.second;\n"
     << "      } while (!Str.empty());\n"
     << "    }));\n"
     << "\n\n"
     << "bool " << getRuleConfigClassName()
     << "::isRuleEnabled(unsigned RuleID) const {\n"
     << "    return  !DisabledRules.test(RuleID);\n"
     << "}\n"
     << "bool " << getRuleConfigClassName() << "::parseCommandLineOption() {\n"
     << "  for (StringRef Identifier : " << Name << "Option) {\n"
     << "    bool Enabled = Identifier.consume_front(\"!\");\n"
     << "    if (Enabled && !setRuleEnabled(Identifier))\n"
     << "      return false;\n"
     << "    if (!Enabled && !setRuleDisabled(Identifier))\n"
     << "      return false;\n"
     << "  }\n"
     << "  return true;\n"
     << "}\n\n";
}

void GICombinerEmitter::emitAdditionalImpl(raw_ostream &OS) {
  OS << "bool " << getClassName()
     << "::tryCombineAll(MachineInstr &I) const {\n"
     << "  const TargetSubtargetInfo &ST = MF.getSubtarget();\n"
     << "  const PredicateBitset AvailableFeatures = "
        "getAvailableFeatures();\n"
     << "  NewMIVector OutMIs;\n"
     << "  State.MIs.clear();\n"
     << "  State.MIs.push_back(&I);\n"
     << "  " << MatchDataInfo::StructName << " = "
     << MatchDataInfo::StructTypeName << "();\n\n"
     << "  if (executeMatchTable(*this, OutMIs, State, ExecInfo"
     << ", getMatchTable(), *ST.getInstrInfo(), MRI, "
        "*MRI.getTargetRegisterInfo(), *ST.getRegBankInfo(), AvailableFeatures"
     << ", /*CoverageInfo*/ nullptr)) {\n"
     << "    return true;\n"
     << "  }\n\n"
     << "  return false;\n"
     << "}\n\n";
}

void GICombinerEmitter::emitMIPredicateFns(raw_ostream &OS) {
  auto MatchCode = getSorted(AllCXXMatchCode);
  emitMIPredicateFnsImpl<const CXXPredicateCode *>(
      OS, "", ArrayRef<const CXXPredicateCode *>(MatchCode),
      [](const CXXPredicateCode *C) -> StringRef { return C->BaseEnumName; },
      [](const CXXPredicateCode *C) -> StringRef { return C->Code; });
}

void GICombinerEmitter::emitI64ImmPredicateFns(raw_ostream &OS) {
  // Unused, but still needs to be called.
  emitImmPredicateFnsImpl<unsigned>(
      OS, "I64", "int64_t", {}, [](unsigned) { return ""; },
      [](unsigned) { return ""; });
}

void GICombinerEmitter::emitAPFloatImmPredicateFns(raw_ostream &OS) {
  // Unused, but still needs to be called.
  emitImmPredicateFnsImpl<unsigned>(
      OS, "APFloat", "const APFloat &", {}, [](unsigned) { return ""; },
      [](unsigned) { return ""; });
}

void GICombinerEmitter::emitAPIntImmPredicateFns(raw_ostream &OS) {
  // Unused, but still needs to be called.
  emitImmPredicateFnsImpl<unsigned>(
      OS, "APInt", "const APInt &", {}, [](unsigned) { return ""; },
      [](unsigned) { return ""; });
}

void GICombinerEmitter::emitTestSimplePredicate(raw_ostream &OS) {
  if (!AllCombineRules.empty()) {
    OS << "enum {\n";
    std::string EnumeratorSeparator = " = GICXXPred_Invalid + 1,\n";
    // To avoid emitting a switch, we expect that all those rules are in order.
    // That way we can just get the RuleID from the enum by subtracting
    // (GICXXPred_Invalid + 1).
    unsigned ExpectedID = 0;
    (void)ExpectedID;
    for (const auto &[ID, _] : AllCombineRules) {
      assert(ExpectedID++ == ID && "combine rules are not ordered!");
      OS << "  " << getIsEnabledPredicateEnumName(ID) << EnumeratorSeparator;
      EnumeratorSeparator = ",\n";
    }
    OS << "};\n\n";
  }

  OS << "bool " << getClassName()
     << "::testSimplePredicate(unsigned Predicate) const {\n"
     << "    return RuleConfig.isRuleEnabled(Predicate - "
        "GICXXPred_Invalid - "
        "1);\n"
     << "}\n";
}

void GICombinerEmitter::emitRunCustomAction(raw_ostream &OS) {
  const auto ApplyCode = getSorted(AllCXXApplyCode);

  if (!ApplyCode.empty()) {
    OS << "enum {\n";
    std::string EnumeratorSeparator = " = GICXXCustomAction_Invalid + 1,\n";
    for (const auto &Apply : ApplyCode) {
      OS << "  " << Apply->getEnumNameWithPrefix(CXXApplyPrefix)
         << EnumeratorSeparator;
      EnumeratorSeparator = ",\n";
    }
    OS << "};\n";
  }

  OS << "void " << getClassName()
     << "::runCustomAction(unsigned ApplyID, const MatcherState &State) const "
        "{\n";
  if (!ApplyCode.empty()) {
    OS << "  switch(ApplyID) {\n";
    for (const auto &Apply : ApplyCode) {
      OS << "  case " << Apply->getEnumNameWithPrefix(CXXApplyPrefix) << ":{\n"
         << "    " << Apply->Code << "\n"
         << "    return;\n";
      OS << "  }\n";
    }
    OS << "}\n";
  }
  OS << "  llvm_unreachable(\"Unknown Apply Action\");\n"
     << "}\n";
}

void GICombinerEmitter::emitAdditionalTemporariesDecl(raw_ostream &OS,
                                                      StringRef Indent) {
  OS << Indent << "struct " << MatchDataInfo::StructTypeName << " {\n";
  for (const auto &[Type, VarNames] : AllMatchDataVars) {
    assert(!VarNames.empty() && "Cannot have no vars for this type!");
    OS << Indent << "  " << Type << " " << join(VarNames, ", ") << ";\n";
  }
  OS << Indent << "};\n"
     << Indent << "mutable " << MatchDataInfo::StructTypeName << " "
     << MatchDataInfo::StructName << ";\n\n";
}

GICombinerEmitter::GICombinerEmitter(RecordKeeper &RK,
                                     const CodeGenTarget &Target,
                                     StringRef Name, Record *Combiner)
    : Records(RK), Name(Name), Target(Target), Combiner(Combiner) {}

MatchTable
GICombinerEmitter::buildMatchTable(MutableArrayRef<RuleMatcher> Rules) {
  std::vector<Matcher *> InputRules;
  for (Matcher &Rule : Rules)
    InputRules.push_back(&Rule);

  unsigned CurrentOrdering = 0;
  StringMap<unsigned> OpcodeOrder;
  for (RuleMatcher &Rule : Rules) {
    const StringRef Opcode = Rule.getOpcode();
    assert(!Opcode.empty() && "Didn't expect an undefined opcode");
    if (OpcodeOrder.count(Opcode) == 0)
      OpcodeOrder[Opcode] = CurrentOrdering++;
  }

  llvm::stable_sort(InputRules, [&OpcodeOrder](const Matcher *A,
                                               const Matcher *B) {
    auto *L = static_cast<const RuleMatcher *>(A);
    auto *R = static_cast<const RuleMatcher *>(B);
    return std::make_tuple(OpcodeOrder[L->getOpcode()], L->getNumOperands()) <
           std::make_tuple(OpcodeOrder[R->getOpcode()], R->getNumOperands());
  });

  for (Matcher *Rule : InputRules)
    Rule->optimize();

  std::vector<std::unique_ptr<Matcher>> MatcherStorage;
  std::vector<Matcher *> OptRules =
      optimizeRules<GroupMatcher>(InputRules, MatcherStorage);

  for (Matcher *Rule : OptRules)
    Rule->optimize();

  OptRules = optimizeRules<SwitchMatcher>(OptRules, MatcherStorage);

  return MatchTable::buildTable(OptRules, /*WithCoverage*/ false,
                                /*IsCombiner*/ true);
}

/// Recurse into GICombineGroup's and flatten the ruleset into a simple list.
void GICombinerEmitter::gatherRules(
    std::vector<RuleMatcher> &ActiveRules,
    const std::vector<Record *> &&RulesAndGroups) {
  for (Record *R : RulesAndGroups) {
    if (R->isValueUnset("Rules")) {
      AllCombineRules.emplace_back(NextRuleID, R->getName().str());
      CombineRuleBuilder CRB(Target, SubtargetFeatures, *R, NextRuleID++,
                             ActiveRules);

      if (!CRB.parseAll())
        continue;

      if (StopAfterParse) {
        CRB.print(outs());
        continue;
      }

      if (!CRB.emitRuleMatchers())
        continue;
    } else
      gatherRules(ActiveRules, R->getValueAsListOfDefs("Rules"));
  }
}

void GICombinerEmitter::run(raw_ostream &OS) {
  Records.startTimer("Gather rules");
  std::vector<RuleMatcher> Rules;
  gatherRules(Rules, Combiner->getValueAsListOfDefs("Rules"));
  if (ErrorsPrinted)
    PrintFatalError(Combiner->getLoc(), "Failed to parse one or more rules");

  Records.startTimer("Creating Match Table");
  unsigned MaxTemporaries = 0;
  for (const auto &Rule : Rules)
    MaxTemporaries = std::max(MaxTemporaries, Rule.countRendererFns());

  const MatchTable Table = buildMatchTable(Rules);

  Records.startTimer("Emit combiner");

  emitSourceFileHeader(getClassName().str() + " Combiner Match Table", OS);

  // Unused
  std::vector<StringRef> CustomRendererFns;
  // Unused, but hack to avoid empty declarator
  std::vector<LLTCodeGen> TypeObjects = {LLTCodeGen(LLT::scalar(1))};
  // Unused
  std::vector<Record *> ComplexPredicates;

  // GET_GICOMBINER_DEPS, which pulls in extra dependencies.
  OS << "#ifdef GET_GICOMBINER_DEPS\n"
     << "#include \"llvm/ADT/SparseBitVector.h\"\n"
     << "namespace llvm {\n"
     << "extern cl::OptionCategory GICombinerOptionCategory;\n"
     << "} // end namespace llvm\n"
     << "#endif // ifdef GET_GICOMBINER_DEPS\n\n";

  // GET_GICOMBINER_TYPES, which needs to be included before the declaration of
  // the class.
  OS << "#ifdef GET_GICOMBINER_TYPES\n";
  emitRuleConfigImpl(OS);
  OS << "#endif // ifdef GET_GICOMBINER_TYPES\n\n";
  emitPredicateBitset(OS, "GET_GICOMBINER_TYPES");

  // GET_GICOMBINER_CLASS_MEMBERS, which need to be included inside the class.
  emitPredicatesDecl(OS, "GET_GICOMBINER_CLASS_MEMBERS");
  emitTemporariesDecl(OS, "GET_GICOMBINER_CLASS_MEMBERS");

  // GET_GICOMBINER_IMPL, which needs to be included outside the class.
  emitExecutorImpl(OS, Table, TypeObjects, Rules, ComplexPredicates,
                   CustomRendererFns, "GET_GICOMBINER_IMPL");

  // GET_GICOMBINER_CONSTRUCTOR_INITS, which are in the constructor's
  // initializer list.
  emitPredicatesInit(OS, "GET_GICOMBINER_CONSTRUCTOR_INITS");
  emitTemporariesInit(OS, MaxTemporaries, "GET_GICOMBINER_CONSTRUCTOR_INITS");
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//

static void EmitGICombiner(RecordKeeper &RK, raw_ostream &OS) {
  CodeGenTarget Target(RK);

  if (SelectedCombiners.empty())
    PrintFatalError("No combiners selected with -combiners");
  for (const auto &Combiner : SelectedCombiners) {
    Record *CombinerDef = RK.getDef(Combiner);
    if (!CombinerDef)
      PrintFatalError("Could not find " + Combiner);
    GICombinerEmitter(RK, Target, Combiner, CombinerDef).run(OS);
  }
}

static TableGen::Emitter::Opt X("gen-global-isel-combiner-matchtable",
                                EmitGICombiner,
                                "Generate GlobalISel combiner Match Table");
