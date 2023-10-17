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
/// Usually, TableGen backends use "assert is an error" as a means to report
/// invalid input. They try to diagnose common case but don't try very hard and
/// crashes can be common. This backend aims to behave closer to how a language
/// compiler frontend would behave: we try extra hard to diagnose invalid inputs
/// early, and any crash should be considered a bug (= a feature or diagnostic
/// is missing).
///
/// While this can make the backend a bit more complex than it needs to be, it
/// pays off because MIR patterns can get complicated. Giving useful error
/// messages to combine writers can help boost their productivity.
///
/// As with anything, a good balance has to be found. We also don't want to
/// write hundreds of lines of code to detect edge cases. In practice, crashing
/// very occasionally, or giving poor errors in some rare instances, is fine.
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
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::gi;

#define DEBUG_TYPE "gicombiner-emitter"

namespace {
cl::OptionCategory
    GICombinerEmitterCat("Options for -gen-global-isel-combiner");
cl::opt<bool> StopAfterParse(
    "gicombiner-stop-after-parse",
    cl::desc("Stop processing after parsing rules and dump state"),
    cl::cat(GICombinerEmitterCat));
cl::list<std::string>
    SelectedCombiners("combiners", cl::desc("Emit the specified combiners"),
                      cl::cat(GICombinerEmitterCat), cl::CommaSeparated);
cl::opt<bool> DebugCXXPreds(
    "gicombiner-debug-cxxpreds",
    cl::desc("Add Contextual/Debug comments to all C++ predicates"),
    cl::cat(GICombinerEmitterCat));

constexpr StringLiteral CXXApplyPrefix = "GICXXCustomAction_CombineApply";
constexpr StringLiteral CXXPredPrefix = "GICXXPred_MI_Predicate_";
constexpr StringLiteral PatFragClassName = "GICombinePatFrag";
constexpr StringLiteral BuiltinInstClassName = "GIBuiltinInst";

std::string getIsEnabledPredicateEnumName(unsigned CombinerRuleID) {
  return "GICXXPred_Simple_IsRule" + to_string(CombinerRuleID) + "Enabled";
}

/// Copies a StringRef into a static pool to make sure it has a static lifetime.
StringRef insertStrRef(StringRef S) {
  if (S.empty())
    return {};

  static StringSet<> Pool;
  auto [It, Inserted] = Pool.insert(S);
  return It->getKey();
}

void declareInstExpansion(CodeExpansions &CE, const InstructionMatcher &IM,
                          StringRef Name) {
  CE.declare(Name, "State.MIs[" + to_string(IM.getInsnVarID()) + "]");
}

void declareInstExpansion(CodeExpansions &CE, const BuildMIAction &A,
                          StringRef Name) {
  // Note: we use redeclare here because this may overwrite a matcher inst
  // expansion.
  CE.redeclare(Name, "OutMIs[" + to_string(A.getInsnID()) + "]");
}

void declareOperandExpansion(CodeExpansions &CE, const OperandMatcher &OM,
                             StringRef Name) {
  CE.declare(Name, "State.MIs[" + to_string(OM.getInsnVarID()) +
                       "]->getOperand(" + to_string(OM.getOpIdx()) + ")");
}

void declareTempRegExpansion(CodeExpansions &CE, unsigned TempRegID,
                             StringRef Name) {
  CE.declare(Name, "State.TempRegisters[" + to_string(TempRegID) + "]");
}

std::string makeAnonPatName(StringRef Prefix, unsigned Idx) {
  return ("__" + Prefix + "_" + Twine(Idx)).str();
}

template <typename Container> auto keys(Container &&C) {
  return map_range(C, [](auto &Entry) -> auto & { return Entry.first; });
}

template <typename Container> auto values(Container &&C) {
  return map_range(C, [](auto &Entry) -> auto & { return Entry.second; });
}

LLTCodeGen getLLTCodeGenFromRecord(const Record *Ty) {
  assert(Ty->isSubClassOf("ValueType"));
  return LLTCodeGen(*MVTToLLT(getValueType(Ty)));
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
  for (auto &Info : Infos) {
    unsigned &NumSeen = SeenTypes[Info.getType()];
    auto &ExistingVars = AllMatchDataVars[Info.getType()];

    if (NumSeen == ExistingVars.size())
      ExistingVars.push_back("MDInfo" + to_string(NextVarID++));

    Info.setVariableName(ExistingVars[NumSeen++]);
  }
}

//===- C++ Predicates Handling --------------------------------------------===//

/// Entry into the static pool of all CXX Predicate code. This contains
/// fully expanded C++ code.
///
/// The static pool is hidden inside the object and can be accessed through
/// getAllMatchCode/getAllApplyCode
///
/// Note that CXXPattern trims C++ code, so the Code is already expected to be
/// free of leading/trailing whitespace.
class CXXPredicateCode {
  using CXXPredicateCodePool =
      DenseMap<hash_code, std::unique_ptr<CXXPredicateCode>>;
  static CXXPredicateCodePool AllCXXMatchCode;
  static CXXPredicateCodePool AllCXXApplyCode;

  /// Sorts a `CXXPredicateCodePool` by their IDs and returns it.
  static std::vector<const CXXPredicateCode *>
  getSorted(const CXXPredicateCodePool &Pool) {
    std::vector<const CXXPredicateCode *> Out;
    std::transform(Pool.begin(), Pool.end(), std::back_inserter(Out),
                   [&](auto &Elt) { return Elt.second.get(); });
    sort(Out, [](const auto *A, const auto *B) { return A->ID < B->ID; });
    return Out;
  }

  /// Gets an instance of `CXXPredicateCode` for \p Code, or returns an already
  /// existing one.
  static const CXXPredicateCode &get(CXXPredicateCodePool &Pool,
                                     std::string Code) {
    // Check if we already have an identical piece of code, if not, create an
    // entry in the pool.
    const auto CodeHash = hash_value(Code);
    if (auto It = Pool.find(CodeHash); It != Pool.end())
      return *It->second;

    const auto ID = Pool.size();
    auto OwnedData = std::unique_ptr<CXXPredicateCode>(
        new CXXPredicateCode(std::move(Code), ID));
    const auto &DataRef = *OwnedData;
    Pool[CodeHash] = std::move(OwnedData);
    return DataRef;
  }

  CXXPredicateCode(std::string Code, unsigned ID)
      : Code(Code), ID(ID), BaseEnumName("GICombiner" + to_string(ID)) {
    // Don't assert if ErrorsPrinted is set. This may mean CodeExpander failed,
    // and it may add spaces in such cases.
    assert((ErrorsPrinted || StringRef(Code).trim() == Code) &&
           "Code was expected to be trimmed!");
  }

public:
  static const CXXPredicateCode &getMatchCode(std::string Code) {
    return get(AllCXXMatchCode, std::move(Code));
  }

  static const CXXPredicateCode &getApplyCode(std::string Code) {
    return get(AllCXXApplyCode, std::move(Code));
  }

  static std::vector<const CXXPredicateCode *> getAllMatchCode() {
    return getSorted(AllCXXMatchCode);
  }

  static std::vector<const CXXPredicateCode *> getAllApplyCode() {
    return getSorted(AllCXXApplyCode);
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

CXXPredicateCode::CXXPredicateCodePool CXXPredicateCode::AllCXXMatchCode;
CXXPredicateCode::CXXPredicateCodePool CXXPredicateCode::AllCXXApplyCode;

//===- Pattern Base Class -------------------------------------------------===//

/// Base class for all patterns that can be written in an `apply`, `match` or
/// `pattern` DAG operator.
///
/// For example:
///
///     (apply (G_ZEXT $x, $y), (G_ZEXT $y, $z), "return isFoo(${z})")
///
/// Creates 3 Pattern objects:
///   - Two CodeGenInstruction Patterns
///   - A CXXPattern
class Pattern {
public:
  enum {
    K_AnyOpcode,
    K_CXX,

    K_CodeGenInstruction,
    K_PatFrag,
    K_Builtin,
  };

  virtual ~Pattern() = default;

  unsigned getKind() const { return Kind; }
  const char *getKindName() const;

  bool hasName() const { return !Name.empty(); }
  StringRef getName() const { return Name; }

  virtual void print(raw_ostream &OS, bool PrintName = true) const = 0;
  void dump() const { return print(dbgs()); }

protected:
  Pattern(unsigned Kind, StringRef Name)
      : Kind(Kind), Name(insertStrRef(Name)) {
    assert(!Name.empty() && "unnamed pattern!");
  }

  void printImpl(raw_ostream &OS, bool PrintName,
                 function_ref<void()> ContentPrinter) const;

private:
  unsigned Kind;
  StringRef Name;
};

const char *Pattern::getKindName() const {
  switch (Kind) {
  case K_AnyOpcode:
    return "AnyOpcodePattern";
  case K_CXX:
    return "CXXPattern";
  case K_CodeGenInstruction:
    return "CodeGenInstructionPattern";
  case K_PatFrag:
    return "PatFragPattern";
  case K_Builtin:
    return "BuiltinPattern";
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
///
/// TODO: Long-term, this needs to be removed. It's a hack around MIR
///       pattern matching limitations.
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

//===- CXXPattern ---------------------------------------------------------===//

/// Represents raw C++ code which may need some expansions.
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
  CXXPattern(const StringInit &Code, StringRef Name)
      : CXXPattern(Code.getAsUnquotedString(), Name) {}

  CXXPattern(StringRef Code, StringRef Name)
      : Pattern(K_CXX, Name), RawCode(Code.trim().str()) {}

  static bool classof(const Pattern *P) { return P->getKind() == K_CXX; }

  void setIsApply(bool Value = true) { IsApply = Value; }
  StringRef getRawCode() const { return RawCode; }

  /// Expands raw code, replacing things such as `${foo}` with their
  /// substitution in \p CE.
  ///
  /// \param CE     Map of Code Expansions
  /// \param Locs   SMLocs for the Code Expander, in case it needs to emit
  ///               diagnostics.
  /// \param AddComment If DebugCXXPreds is enabled, this is called to emit a
  ///                   comment before the expanded code.
  ///
  /// \return A CXXPredicateCode object that contains the expanded code. Note
  /// that this may or may not insert a new object. All CXXPredicateCode objects
  /// are held in a set to avoid emitting duplicate C++ code.
  const CXXPredicateCode &
  expandCode(const CodeExpansions &CE, ArrayRef<SMLoc> Locs,
             function_ref<void(raw_ostream &)> AddComment = {}) const;

  void print(raw_ostream &OS, bool PrintName = true) const override;

private:
  bool IsApply = false;
  std::string RawCode;
};

const CXXPredicateCode &
CXXPattern::expandCode(const CodeExpansions &CE, ArrayRef<SMLoc> Locs,
                       function_ref<void(raw_ostream &)> AddComment) const {
  std::string Result;
  raw_string_ostream OS(Result);

  if (DebugCXXPreds && AddComment)
    AddComment(OS);

  CodeExpander Expander(RawCode, CE, Locs, /*ShowExpansions*/ false);
  Expander.emit(OS);
  if (IsApply)
    return CXXPredicateCode::getApplyCode(std::move(Result));
  return CXXPredicateCode::getMatchCode(std::move(Result));
}

void CXXPattern::print(raw_ostream &OS, bool PrintName) const {
  printImpl(OS, PrintName, [&OS, this] {
    OS << (IsApply ? "apply" : "match") << " code:\"";
    printEscapedString(getRawCode(), OS);
    OS << "\"";
  });
}

//===- InstructionPattern ---------------------------------------------===//

/// An operand for an InstructionPattern.
///
/// Operands are composed of three elements:
///   - (Optional) Value
///   - (Optional) Name
///   - (Optional) Type
///
/// Some examples:
///   (i32 0):$x -> V=int(0), Name='x', Type=i32
///   0:$x -> V=int(0), Name='x'
///   $x -> Name='x'
///   i32:$x -> Name='x', Type = i32
class InstructionOperand {
public:
  using IntImmTy = int64_t;

  InstructionOperand(IntImmTy Imm, StringRef Name, const Record *Type)
      : Value(Imm), Name(insertStrRef(Name)), Type(Type) {
    assert(!Type || Type->isSubClassOf("ValueType"));
  }

  InstructionOperand(StringRef Name, const Record *Type)
      : Name(insertStrRef(Name)), Type(Type) {}

  bool isNamedImmediate() const { return hasImmValue() && isNamedOperand(); }

  bool hasImmValue() const { return Value.has_value(); }
  IntImmTy getImmValue() const { return *Value; }

  bool isNamedOperand() const { return !Name.empty(); }
  StringRef getOperandName() const {
    assert(isNamedOperand() && "Operand is unnamed");
    return Name;
  }

  InstructionOperand withNewName(StringRef NewName) const {
    InstructionOperand Result = *this;
    Result.Name = insertStrRef(NewName);
    return Result;
  }

  void setIsDef(bool Value = true) { Def = Value; }
  bool isDef() const { return Def; }

  void setType(const Record *R) {
    assert((!Type || (Type == R)) && "Overwriting type!");
    Type = R;
  }
  const Record *getType() const { return Type; }

  std::string describe() const {
    if (!hasImmValue())
      return "MachineOperand $" + getOperandName().str() + "";
    std::string Str = "imm " + to_string(getImmValue());
    if (isNamedImmediate())
      Str += ":$" + getOperandName().str() + "";
    return Str;
  }

  void print(raw_ostream &OS) const {
    if (isDef())
      OS << "<def>";

    bool NeedsColon = true;
    if (const Record *Ty = getType()) {
      if (hasImmValue())
        OS << "(" << Ty->getName() << " " << getImmValue() << ")";
      else
        OS << Ty->getName();
    } else if (hasImmValue())
      OS << getImmValue();
    else
      NeedsColon = false;

    if (isNamedOperand())
      OS << (NeedsColon ? ":" : "") << "$" << getOperandName();
  }

  void dump() const { return print(dbgs()); }

private:
  std::optional<int64_t> Value;
  StringRef Name;
  const Record *Type = nullptr;
  bool Def = false;
};

/// Base class for CodeGenInstructionPattern & PatFragPattern, which handles all
/// the boilerplate for patterns that have a list of operands for some (pseudo)
/// instruction.
class InstructionPattern : public Pattern {
public:
  virtual ~InstructionPattern() = default;

  static bool classof(const Pattern *P) {
    return P->getKind() == K_CodeGenInstruction || P->getKind() == K_PatFrag ||
           P->getKind() == K_Builtin;
  }

  template <typename... Ty> void addOperand(Ty &&...Init) {
    Operands.emplace_back(std::forward<Ty>(Init)...);
  }

  auto &operands() { return Operands; }
  const auto &operands() const { return Operands; }
  unsigned operands_size() const { return Operands.size(); }
  InstructionOperand &getOperand(unsigned K) { return Operands[K]; }
  const InstructionOperand &getOperand(unsigned K) const { return Operands[K]; }

  /// When this InstructionPattern is used as the match root, returns the
  /// operands that must be redefined in the 'apply' pattern for the rule to be
  /// valid.
  ///
  /// For most patterns, this just returns the defs.
  /// For PatFrag this only returns the root of the PF.
  ///
  /// Returns an empty array on error.
  virtual ArrayRef<InstructionOperand> getApplyDefsNeeded() const {
    return {operands().begin(), getNumInstDefs()};
  }

  auto named_operands() {
    return make_filter_range(Operands,
                             [&](auto &O) { return O.isNamedOperand(); });
  }

  auto named_operands() const {
    return make_filter_range(Operands,
                             [&](auto &O) { return O.isNamedOperand(); });
  }

  virtual bool isVariadic() const { return false; }
  virtual unsigned getNumInstOperands() const = 0;
  virtual unsigned getNumInstDefs() const = 0;

  bool hasAllDefs() const { return operands_size() >= getNumInstDefs(); }

  virtual StringRef getInstName() const = 0;

  void reportUnreachable(ArrayRef<SMLoc> Locs) const;
  virtual bool checkSemantics(ArrayRef<SMLoc> Loc);

  void print(raw_ostream &OS, bool PrintName = true) const override;

protected:
  InstructionPattern(unsigned K, StringRef Name) : Pattern(K, Name) {}

  SmallVector<InstructionOperand, 4> Operands;
};

void InstructionPattern::reportUnreachable(ArrayRef<SMLoc> Locs) const {
  PrintError(Locs, "pattern '" + getName() + "' ('" + getInstName() +
                       "') is unreachable from the pattern root!");
}

bool InstructionPattern::checkSemantics(ArrayRef<SMLoc> Loc) {
  unsigned NumExpectedOperands = getNumInstOperands();

  if (isVariadic()) {
    if (Operands.size() < NumExpectedOperands) {
      PrintError(Loc, +"'" + getInstName() + "' expected at least " +
                          Twine(NumExpectedOperands) + " operands, got " +
                          Twine(Operands.size()));
      return false;
    }
  } else if (NumExpectedOperands != Operands.size()) {
    PrintError(Loc, +"'" + getInstName() + "' expected " +
                        Twine(NumExpectedOperands) + " operands, got " +
                        Twine(Operands.size()));
    return false;
  }

  unsigned OpIdx = 0;
  unsigned NumDefs = getNumInstDefs();
  for (auto &Op : Operands)
    Op.setIsDef(OpIdx++ < NumDefs);

  return true;
}

void InstructionPattern::print(raw_ostream &OS, bool PrintName) const {
  printImpl(OS, PrintName, [&OS, this] {
    OS << getInstName() << " operands:[";
    StringRef Sep = "";
    for (const auto &Op : Operands) {
      OS << Sep;
      Op.print(OS);
      Sep = ", ";
    }
    OS << "]";
  });
}

//===- OperandTable -------------------------------------------------------===//

/// Maps InstructionPattern operands to their definitions. This allows us to tie
/// different patterns of a (apply), (match) or (patterns) set of patterns
/// together.
template <typename DefTy = InstructionPattern> class OperandTable {
public:
  static_assert(std::is_base_of_v<InstructionPattern, DefTy>,
                "DefTy should be a derived class from InstructionPattern");

  bool addPattern(DefTy *P, function_ref<void(StringRef)> DiagnoseRedef) {
    for (const auto &Op : P->named_operands()) {
      StringRef OpName = Op.getOperandName();

      // We always create an entry in the OperandTable, even for uses.
      // Uses of operands that don't have a def (= live-ins) will remain with a
      // nullptr as the Def.
      //
      // This allows us tell whether an operand exists in a pattern or not. If
      // there is no entry for it, it doesn't exist, if there is an entry, it's
      // used/def'd at least once.
      auto &Def = Table[OpName];

      if (!Op.isDef())
        continue;

      if (Def) {
        DiagnoseRedef(OpName);
        return false;
      }

      Def = P;
    }

    return true;
  }

  struct LookupResult {
    LookupResult() = default;
    LookupResult(DefTy *Def) : Found(true), Def(Def) {}

    bool Found = false;
    DefTy *Def = nullptr;

    bool isLiveIn() const { return Found && !Def; }
  };

  LookupResult lookup(StringRef OpName) const {
    if (auto It = Table.find(OpName); It != Table.end())
      return LookupResult(It->second);
    return LookupResult();
  }

  DefTy *getDef(StringRef OpName) const { return lookup(OpName).Def; }

  void print(raw_ostream &OS, StringRef Name = "",
             StringRef Indent = "") const {
    OS << Indent << "(OperandTable ";
    if (!Name.empty())
      OS << Name << " ";
    if (Table.empty()) {
      OS << "<empty>)\n";
      return;
    }

    SmallVector<StringRef, 0> Keys(Table.keys());
    sort(Keys);

    OS << "\n";
    for (const auto &Key : Keys) {
      const auto *Def = Table.at(Key);
      OS << Indent << "  " << Key << " -> "
         << (Def ? Def->getName() : "<live-in>") << "\n";
    }
    OS << Indent << ")\n";
  }

  auto begin() const { return Table.begin(); }
  auto end() const { return Table.end(); }

  void dump() const { print(dbgs()); }

private:
  StringMap<DefTy *> Table;
};

//===- CodeGenInstructionPattern ------------------------------------------===//

/// Matches an instruction, e.g. `G_ADD $x, $y, $z`.
class CodeGenInstructionPattern : public InstructionPattern {
public:
  CodeGenInstructionPattern(const CodeGenInstruction &I, StringRef Name)
      : InstructionPattern(K_CodeGenInstruction, Name), I(I) {}

  static bool classof(const Pattern *P) {
    return P->getKind() == K_CodeGenInstruction;
  }

  bool is(StringRef OpcodeName) const {
    return I.TheDef->getName() == OpcodeName;
  }

  bool hasVariadicDefs() const;
  bool isVariadic() const override { return I.Operands.isVariadic; }
  unsigned getNumInstDefs() const override;
  unsigned getNumInstOperands() const override;

  const CodeGenInstruction &getInst() const { return I; }
  StringRef getInstName() const override { return I.TheDef->getName(); }

private:
  const CodeGenInstruction &I;
};

bool CodeGenInstructionPattern::hasVariadicDefs() const {
  // Note: we cannot use variadicOpsAreDefs, it's not set for
  // GenericInstructions.
  if (!isVariadic())
    return false;

  if (I.variadicOpsAreDefs)
    return true;

  DagInit *OutOps = I.TheDef->getValueAsDag("OutOperandList");
  if (OutOps->arg_empty())
    return false;

  auto *LastArgTy = dyn_cast<DefInit>(OutOps->getArg(OutOps->arg_size() - 1));
  return LastArgTy && LastArgTy->getDef()->getName() == "variable_ops";
}

unsigned CodeGenInstructionPattern::getNumInstDefs() const {
  if (!isVariadic() || !hasVariadicDefs())
    return I.Operands.NumDefs;
  unsigned NumOuts = I.Operands.size() - I.Operands.NumDefs;
  assert(Operands.size() > NumOuts);
  return std::max<unsigned>(I.Operands.NumDefs, Operands.size() - NumOuts);
}

unsigned CodeGenInstructionPattern::getNumInstOperands() const {
  unsigned NumCGIOps = I.Operands.size();
  return isVariadic() ? std::max<unsigned>(NumCGIOps, Operands.size())
                      : NumCGIOps;
}

//===- OperandTypeChecker -------------------------------------------------===//

/// This is a trivial type checker for all operands in a set of
/// InstructionPatterns.
///
/// It infers the type of each operand, check it's consistent with the known
/// type of the operand, and then sets all of the types in all operands in
/// setAllOperandTypes.
class OperandTypeChecker {
public:
  OperandTypeChecker(ArrayRef<SMLoc> DiagLoc) : DiagLoc(DiagLoc) {}

  bool check(InstructionPattern *P);

  void setAllOperandTypes();

private:
  struct OpTypeInfo {
    const Record *Type = nullptr;
    InstructionPattern *TypeSrc = nullptr;
  };

  ArrayRef<SMLoc> DiagLoc;
  StringMap<OpTypeInfo> Types;

  SmallVector<InstructionPattern *, 16> Pats;
};

bool OperandTypeChecker::check(InstructionPattern *P) {
  Pats.push_back(P);

  for (auto &Op : P->named_operands()) {
    const Record *Ty = Op.getType();
    if (!Ty)
      continue;

    auto &Info = Types[Op.getOperandName()];

    if (!Info.Type) {
      Info.Type = Ty;
      Info.TypeSrc = P;
      continue;
    }

    if (Info.Type != Ty) {
      PrintError(DiagLoc, "conflicting types for operand '" +
                              Op.getOperandName() + "': first seen with '" +
                              Info.Type->getName() + "' in '" +
                              Info.TypeSrc->getName() + ", now seen with '" +
                              Ty->getName() + "' in '" + P->getName() + "'");
      return false;
    }
  }

  return true;
}

void OperandTypeChecker::setAllOperandTypes() {
  for (auto *Pat : Pats) {
    for (auto &Op : Pat->named_operands()) {
      if (auto &Info = Types[Op.getOperandName()]; Info.Type)
        Op.setType(Info.Type);
    }
  }
}

//===- PatFrag ------------------------------------------------------------===//

/// Represents a parsed GICombinePatFrag. This can be thought of as the
/// equivalent of a CodeGenInstruction, but for PatFragPatterns.
///
/// PatFrags are made of 3 things:
///   - Out parameters (defs)
///   - In parameters
///   - A set of pattern lists (alternatives).
///
/// If the PatFrag uses instruction patterns, the root must be one of the defs.
///
/// Note that this DOES NOT represent the use of the PatFrag, only its
/// definition. The use of the PatFrag in a Pattern is represented by
/// PatFragPattern.
///
/// PatFrags use the term "parameter" instead of operand because they're
/// essentially macros, and using that name avoids confusion. Other than that,
/// they're structured similarly to a MachineInstruction  - all parameters
/// (operands) are in the same list, with defs at the start. This helps mapping
/// parameters to values, because, param N of a PatFrag is always operand N of a
/// PatFragPattern.
class PatFrag {
public:
  enum ParamKind {
    PK_Root,
    PK_MachineOperand,
    PK_Imm,
  };

  struct Param {
    StringRef Name;
    ParamKind Kind;
  };

  using ParamVec = SmallVector<Param, 4>;
  using ParamIt = ParamVec::const_iterator;

  /// Represents an alternative of the PatFrag. When parsing a GICombinePatFrag,
  /// this is created from its "Alternatives" list. Each alternative is a list
  /// of patterns written wrapped in a  `(pattern ...)` dag init.
  ///
  /// Each argument to the `pattern` DAG operator is parsed into a Pattern
  /// instance.
  struct Alternative {
    OperandTable<> OpTable;
    SmallVector<std::unique_ptr<Pattern>, 4> Pats;
  };

  explicit PatFrag(const Record &Def) : Def(Def) {
    assert(Def.isSubClassOf(PatFragClassName));
  }

  static StringRef getParamKindStr(ParamKind OK);

  StringRef getName() const { return Def.getName(); }

  const Record &getDef() const { return Def; }
  ArrayRef<SMLoc> getLoc() const { return Def.getLoc(); }

  Alternative &addAlternative() { return Alts.emplace_back(); }
  const Alternative &getAlternative(unsigned K) const { return Alts[K]; }
  unsigned num_alternatives() const { return Alts.size(); }

  void addInParam(StringRef Name, ParamKind Kind);
  iterator_range<ParamIt> in_params() const;
  unsigned num_in_params() const { return Params.size() - NumOutParams; }

  void addOutParam(StringRef Name, ParamKind Kind);
  iterator_range<ParamIt> out_params() const;
  unsigned num_out_params() const { return NumOutParams; }

  unsigned num_roots() const;
  unsigned num_params() const { return num_in_params() + num_out_params(); }

  /// Finds the operand \p Name and returns its index or -1 if not found.
  /// Remember that all params are part of the same list, with out params at the
  /// start. This means that the index returned can be used to access operands
  /// of InstructionPatterns.
  unsigned getParamIdx(StringRef Name) const;
  const Param &getParam(unsigned K) const { return Params[K]; }

  bool canBeMatchRoot() const { return num_roots() == 1; }

  void print(raw_ostream &OS, StringRef Indent = "") const;
  void dump() const { print(dbgs()); }

  /// Checks if the in-param \p ParamName can be unbound or not.
  /// \p ArgName is the name of the argument passed to the PatFrag.
  ///
  /// An argument can be unbound only if, for all alternatives:
  ///   - There is no CXX pattern, OR:
  ///   - There is an InstructionPattern that binds the parameter.
  ///
  /// e.g. in (MyPatFrag $foo), if $foo has never been seen before (= it's
  /// unbound), this checks if MyPatFrag supports it or not.
  bool handleUnboundInParam(StringRef ParamName, StringRef ArgName,
                            ArrayRef<SMLoc> DiagLoc) const;

  bool checkSemantics();
  bool buildOperandsTables();

private:
  static void printParamsList(raw_ostream &OS, iterator_range<ParamIt> Params);

  void PrintError(Twine Msg) const { ::PrintError(&Def, Msg); }

  const Record &Def;
  unsigned NumOutParams = 0;
  ParamVec Params;
  SmallVector<Alternative, 2> Alts;
};

StringRef PatFrag::getParamKindStr(ParamKind OK) {
  switch (OK) {
  case PK_Root:
    return "root";
  case PK_MachineOperand:
    return "machine_operand";
  case PK_Imm:
    return "imm";
  }

  llvm_unreachable("Unknown operand kind!");
}

void PatFrag::addInParam(StringRef Name, ParamKind Kind) {
  Params.emplace_back(Param{insertStrRef(Name), Kind});
}

iterator_range<PatFrag::ParamIt> PatFrag::in_params() const {
  return {Params.begin() + NumOutParams, Params.end()};
}

void PatFrag::addOutParam(StringRef Name, ParamKind Kind) {
  assert(NumOutParams == Params.size() &&
         "Adding out-param after an in-param!");
  Params.emplace_back(Param{insertStrRef(Name), Kind});
  ++NumOutParams;
}

iterator_range<PatFrag::ParamIt> PatFrag::out_params() const {
  return {Params.begin(), Params.begin() + NumOutParams};
}

unsigned PatFrag::num_roots() const {
  return count_if(out_params(),
                  [&](const auto &P) { return P.Kind == PK_Root; });
}

unsigned PatFrag::getParamIdx(StringRef Name) const {
  for (const auto &[Idx, Op] : enumerate(Params)) {
    if (Op.Name == Name)
      return Idx;
  }

  return -1;
}

bool PatFrag::checkSemantics() {
  for (const auto &Alt : Alts) {
    for (const auto &Pat : Alt.Pats) {
      switch (Pat->getKind()) {
      case Pattern::K_AnyOpcode:
        PrintError("wip_match_opcode cannot be used in " + PatFragClassName);
        return false;
      case Pattern::K_Builtin:
        PrintError("Builtin instructions cannot be used in " +
                   PatFragClassName);
        return false;
      case Pattern::K_CXX:
      case Pattern::K_CodeGenInstruction:
        continue;
      case Pattern::K_PatFrag:
        // TODO: It's just that the emitter doesn't handle it but technically
        // there is no reason why we can't. We just have to be careful with
        // operand mappings, it could get complex.
        PrintError("nested " + PatFragClassName + " are not supported");
        return false;
      }
    }
  }

  StringSet<> SeenOps;
  for (const auto &Op : in_params()) {
    if (SeenOps.count(Op.Name)) {
      PrintError("duplicate parameter '" + Op.Name + "'");
      return false;
    }

    // Check this operand is NOT defined in any alternative's patterns.
    for (const auto &Alt : Alts) {
      if (Alt.OpTable.lookup(Op.Name).Def) {
        PrintError("input parameter '" + Op.Name + "' cannot be redefined!");
        return false;
      }
    }

    if (Op.Kind == PK_Root) {
      PrintError("input parameterr '" + Op.Name + "' cannot be a root!");
      return false;
    }

    SeenOps.insert(Op.Name);
  }

  for (const auto &Op : out_params()) {
    if (Op.Kind != PK_Root && Op.Kind != PK_MachineOperand) {
      PrintError("output parameter '" + Op.Name +
                 "' must be 'root' or 'gi_mo'");
      return false;
    }

    if (SeenOps.count(Op.Name)) {
      PrintError("duplicate parameter '" + Op.Name + "'");
      return false;
    }

    // Check this operand is defined in all alternative's patterns.
    for (const auto &Alt : Alts) {
      const auto *OpDef = Alt.OpTable.getDef(Op.Name);
      if (!OpDef) {
        PrintError("output parameter '" + Op.Name +
                   "' must be defined by all alternative patterns in '" +
                   Def.getName() + "'");
        return false;
      }

      if (Op.Kind == PK_Root && OpDef->getNumInstDefs() != 1) {
        // The instruction that defines the root must have a single def.
        // Otherwise we'd need to support multiple roots and it gets messy.
        //
        // e.g. this is not supported:
        //   (pattern (G_UNMERGE_VALUES $x, $root, $vec))
        PrintError("all instructions that define root '" + Op.Name + "' in '" +
                   Def.getName() + "' can only have a single output operand");
        return false;
      }
    }

    SeenOps.insert(Op.Name);
  }

  if (num_out_params() != 0 && num_roots() == 0) {
    PrintError(PatFragClassName + " must have one root in its 'out' operands");
    return false;
  }

  if (num_roots() > 1) {
    PrintError(PatFragClassName + " can only have one root");
    return false;
  }

  // TODO: find unused params

  // Now, typecheck all alternatives.
  for (auto &Alt : Alts) {
    OperandTypeChecker OTC(Def.getLoc());
    for (auto &Pat : Alt.Pats) {
      if (auto *IP = dyn_cast<InstructionPattern>(Pat.get())) {
        if (!OTC.check(IP))
          return false;
      }
    }
    OTC.setAllOperandTypes();
  }

  return true;
}

bool PatFrag::handleUnboundInParam(StringRef ParamName, StringRef ArgName,
                                   ArrayRef<SMLoc> DiagLoc) const {
  // The parameter must be a live-in of all alternatives for this to work.
  // Otherwise, we risk having unbound parameters being used (= crashes).
  //
  // Examples:
  //
  // in (ins $y), (patterns (G_FNEG $dst, $y), "return matchFnegOp(${y})")
  //    even if $y is unbound, we'll lazily bind it when emitting the G_FNEG.
  //
  // in (ins $y), (patterns "return matchFnegOp(${y})")
  //    if $y is unbound when this fragment is emitted, C++ code expansion will
  //    fail.
  for (const auto &Alt : Alts) {
    auto &OT = Alt.OpTable;
    if (!OT.lookup(ParamName).Found) {
      ::PrintError(DiagLoc, "operand '" + ArgName + "' (for parameter '" +
                                ParamName + "' of '" + getName() +
                                "') cannot be unbound");
      PrintNote(
          DiagLoc,
          "one or more alternatives of '" + getName() + "' do not bind '" +
              ParamName +
              "' to an instruction operand; either use a bound operand or "
              "ensure '" +
              Def.getName() + "' binds '" + ParamName +
              "' in all alternatives");
      return false;
    }
  }

  return true;
}

bool PatFrag::buildOperandsTables() {
  // enumerate(...) doesn't seem to allow lvalues so we need to count the old
  // way.
  unsigned Idx = 0;

  const auto DiagnoseRedef = [this, &Idx](StringRef OpName) {
    PrintError("Operand '" + OpName +
               "' is defined multiple times in patterns of alternative #" +
               to_string(Idx));
  };

  for (auto &Alt : Alts) {
    for (auto &Pat : Alt.Pats) {
      auto *IP = dyn_cast<InstructionPattern>(Pat.get());
      if (!IP)
        continue;

      if (!Alt.OpTable.addPattern(IP, DiagnoseRedef))
        return false;
    }

    ++Idx;
  }

  return true;
}

void PatFrag::print(raw_ostream &OS, StringRef Indent) const {
  OS << Indent << "(PatFrag name:" << getName() << "\n";
  if (!in_params().empty()) {
    OS << Indent << "  (ins ";
    printParamsList(OS, in_params());
    OS << ")\n";
  }

  if (!out_params().empty()) {
    OS << Indent << "  (outs ";
    printParamsList(OS, out_params());
    OS << ")\n";
  }

  // TODO: Dump OperandTable as well.
  OS << Indent << "  (alternatives [\n";
  for (const auto &Alt : Alts) {
    OS << Indent << "    [\n";
    for (const auto &Pat : Alt.Pats) {
      OS << Indent << "      ";
      Pat->print(OS, /*PrintName=*/true);
      OS << ",\n";
    }
    OS << Indent << "    ],\n";
  }
  OS << Indent << "  ])\n";

  OS << Indent << ')';
}

void PatFrag::printParamsList(raw_ostream &OS, iterator_range<ParamIt> Params) {
  OS << '['
     << join(map_range(Params,
                       [](auto &O) {
                         return (O.Name + ":" + getParamKindStr(O.Kind)).str();
                       }),
             ", ")
     << ']';
}

//===- PatFragPattern -----------------------------------------------------===//

class PatFragPattern : public InstructionPattern {
public:
  PatFragPattern(const PatFrag &PF, StringRef Name)
      : InstructionPattern(K_PatFrag, Name), PF(PF) {}

  static bool classof(const Pattern *P) { return P->getKind() == K_PatFrag; }

  const PatFrag &getPatFrag() const { return PF; }
  StringRef getInstName() const override { return PF.getName(); }

  unsigned getNumInstDefs() const override { return PF.num_out_params(); }
  unsigned getNumInstOperands() const override { return PF.num_params(); }

  ArrayRef<InstructionOperand> getApplyDefsNeeded() const override;

  bool checkSemantics(ArrayRef<SMLoc> DiagLoc) override;

  /// Before emitting the patterns inside the PatFrag, add all necessary code
  /// expansions to \p PatFragCEs imported from \p ParentCEs.
  ///
  /// For a MachineOperand PatFrag parameter, this will fetch the expansion for
  /// that operand from \p ParentCEs and add it to \p PatFragCEs. Errors can be
  /// emitted if the MachineOperand reference is unbound.
  ///
  /// For an Immediate PatFrag parameter this simply adds the integer value to
  /// \p PatFragCEs as an expansion.
  ///
  /// \param ParentCEs Contains all of the code expansions declared by the other
  ///                  patterns emitted so far in the pattern list containing
  ///                  this PatFragPattern.
  /// \param PatFragCEs Output Code Expansions (usually empty)
  /// \param DiagLoc    Diagnostic loc in case an error occurs.
  /// \return `true` on success, `false` on failure.
  bool mapInputCodeExpansions(const CodeExpansions &ParentCEs,
                              CodeExpansions &PatFragCEs,
                              ArrayRef<SMLoc> DiagLoc) const;

private:
  const PatFrag &PF;
};

ArrayRef<InstructionOperand> PatFragPattern::getApplyDefsNeeded() const {
  assert(PF.num_roots() == 1);
  // Only roots need to be redef.
  for (auto [Idx, Param] : enumerate(PF.out_params())) {
    if (Param.Kind == PatFrag::PK_Root)
      return getOperand(Idx);
  }
  llvm_unreachable("root not found!");
}

bool PatFragPattern::checkSemantics(ArrayRef<SMLoc> DiagLoc) {
  if (!InstructionPattern::checkSemantics(DiagLoc))
    return false;

  for (const auto &[Idx, Op] : enumerate(Operands)) {
    switch (PF.getParam(Idx).Kind) {
    case PatFrag::PK_Imm:
      if (!Op.hasImmValue()) {
        PrintError(DiagLoc, "expected operand " + to_string(Idx) + " of '" +
                                getInstName() + "' to be an immediate; got " +
                                Op.describe());
        return false;
      }
      if (Op.isNamedImmediate()) {
        PrintError(DiagLoc, "operand " + to_string(Idx) + " of '" +
                                getInstName() +
                                "' cannot be a named immediate");
        return false;
      }
      break;
    case PatFrag::PK_Root:
    case PatFrag::PK_MachineOperand:
      if (!Op.isNamedOperand() || Op.isNamedImmediate()) {
        PrintError(DiagLoc, "expected operand " + to_string(Idx) + " of '" +
                                getInstName() +
                                "' to be a MachineOperand; got " +
                                Op.describe());
        return false;
      }
      break;
    }
  }

  return true;
}

bool PatFragPattern::mapInputCodeExpansions(const CodeExpansions &ParentCEs,
                                            CodeExpansions &PatFragCEs,
                                            ArrayRef<SMLoc> DiagLoc) const {
  for (const auto &[Idx, Op] : enumerate(operands())) {
    StringRef ParamName = PF.getParam(Idx).Name;

    // Operands to a PFP can only be named, or be an immediate, but not a named
    // immediate.
    assert(!Op.isNamedImmediate());

    if (Op.isNamedOperand()) {
      StringRef ArgName = Op.getOperandName();
      // Map it only if it's been defined.
      auto It = ParentCEs.find(ArgName);
      if (It == ParentCEs.end()) {
        if (!PF.handleUnboundInParam(ParamName, ArgName, DiagLoc))
          return false;
      } else
        PatFragCEs.declare(ParamName, It->second);
      continue;
    }

    if (Op.hasImmValue()) {
      PatFragCEs.declare(ParamName, to_string(Op.getImmValue()));
      continue;
    }

    llvm_unreachable("Unknown Operand Type!");
  }

  return true;
}

//===- BuiltinPattern -----------------------------------------------------===//

enum BuiltinKind {
  BI_ReplaceReg,
  BI_EraseRoot,
};

class BuiltinPattern : public InstructionPattern {
  struct BuiltinInfo {
    StringLiteral DefName;
    BuiltinKind Kind;
    unsigned NumOps;
    unsigned NumDefs;
  };

  static constexpr std::array<BuiltinInfo, 2> KnownBuiltins = {{
      {"GIReplaceReg", BI_ReplaceReg, 2, 1},
      {"GIEraseRoot", BI_EraseRoot, 0, 0},
  }};

public:
  BuiltinPattern(const Record &Def, StringRef Name)
      : InstructionPattern(K_Builtin, Name), I(getBuiltinInfo(Def)) {}

  static bool classof(const Pattern *P) { return P->getKind() == K_Builtin; }

  unsigned getNumInstOperands() const override { return I.NumOps; }
  unsigned getNumInstDefs() const override { return I.NumDefs; }
  StringRef getInstName() const override { return I.DefName; }
  BuiltinKind getBuiltinKind() const { return I.Kind; }

  bool checkSemantics(ArrayRef<SMLoc> Loc) override;

private:
  static BuiltinInfo getBuiltinInfo(const Record &Def);

  BuiltinInfo I;
};

BuiltinPattern::BuiltinInfo BuiltinPattern::getBuiltinInfo(const Record &Def) {
  assert(Def.isSubClassOf(BuiltinInstClassName));

  StringRef Name = Def.getName();
  for (const auto &KBI : KnownBuiltins) {
    if (KBI.DefName == Name)
      return KBI;
  }

  PrintFatalError(Def.getLoc(), "Unimplemented " + BuiltinInstClassName +
                                    " def '" + Name + "'");
}

bool BuiltinPattern::checkSemantics(ArrayRef<SMLoc> Loc) {
  if (!InstructionPattern::checkSemantics(Loc))
    return false;

  // For now all builtins just take names, no immediates.
  for (const auto &[Idx, Op] : enumerate(operands())) {
    if (!Op.isNamedOperand() || Op.isNamedImmediate()) {
      PrintError(Loc, "expected operand " + to_string(Idx) + " of '" +
                          getInstName() + "' to be a name");
      return false;
    }
  }

  return true;
}

//===- PrettyStackTrace Helpers  ------------------------------------------===//

class PrettyStackTraceParse : public PrettyStackTraceEntry {
  const Record &Def;

public:
  PrettyStackTraceParse(const Record &Def) : Def(Def) {}

  void print(raw_ostream &OS) const override {
    if (Def.isSubClassOf("GICombineRule"))
      OS << "Parsing GICombineRule '" << Def.getName() << "'";
    else if (Def.isSubClassOf(PatFragClassName))
      OS << "Parsing " << PatFragClassName << " '" << Def.getName() << "'";
    else
      OS << "Parsing '" << Def.getName() << "'";
    OS << "\n";
  }
};

class PrettyStackTraceEmit : public PrettyStackTraceEntry {
  const Record &Def;
  const Pattern *Pat = nullptr;

public:
  PrettyStackTraceEmit(const Record &Def, const Pattern *Pat = nullptr)
      : Def(Def), Pat(Pat) {}

  void print(raw_ostream &OS) const override {
    if (Def.isSubClassOf("GICombineRule"))
      OS << "Emitting GICombineRule '" << Def.getName() << "'";
    else if (Def.isSubClassOf(PatFragClassName))
      OS << "Emitting " << PatFragClassName << " '" << Def.getName() << "'";
    else
      OS << "Emitting '" << Def.getName() << "'";

    if (Pat)
      OS << " [" << Pat->getKindName() << " '" << Pat->getName() << "']";
    OS << "\n";
  }
};

//===- CombineRuleBuilder -------------------------------------------------===//

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
  using PatternAlternatives = DenseMap<const Pattern *, unsigned>;

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
#ifndef NDEBUG
  void verify() const;
#endif

private:
  const CodeGenInstruction &getGConstant() const {
    return CGT.getInstruction(RuleDef.getRecords().getDef("G_CONSTANT"));
  }

  void PrintError(Twine Msg) const { ::PrintError(&RuleDef, Msg); }
  void PrintWarning(Twine Msg) const { ::PrintWarning(RuleDef.getLoc(), Msg); }
  void PrintNote(Twine Msg) const { ::PrintNote(RuleDef.getLoc(), Msg); }

  void print(raw_ostream &OS, const PatternAlternatives &Alts) const;

  bool addApplyPattern(std::unique_ptr<Pattern> Pat);
  bool addMatchPattern(std::unique_ptr<Pattern> Pat);

  /// Adds the expansions from \see MatchDatas to \p CE.
  void declareAllMatchDatasExpansions(CodeExpansions &CE) const;

  /// Adds a matcher \p P to \p IM, expanding its code using \p CE.
  /// Note that the predicate is added on the last InstructionMatcher.
  ///
  /// \p Alts is only used if DebugCXXPreds is enabled.
  void addCXXPredicate(RuleMatcher &M, const CodeExpansions &CE,
                       const CXXPattern &P, const PatternAlternatives &Alts);

  /// Adds an apply \p P to \p IM, expanding its code using \p CE.
  void addCXXAction(RuleMatcher &M, const CodeExpansions &CE,
                    const CXXPattern &P);

  bool hasOnlyCXXApplyPatterns() const;
  bool hasEraseRoot() const;

  // Infer machine operand types and check their consistency.
  bool typecheckPatterns();

  /// For all PatFragPatterns, add a new entry in PatternAlternatives for each
  /// PatternList it contains. This is multiplicative, so if we have 2
  /// PatFrags with 3 alternatives each, we get 2*3 permutations added to
  /// PermutationsToEmit. The "MaxPermutations" field controls how many
  /// permutations are allowed before an error is emitted and this function
  /// returns false. This is a simple safeguard to prevent combination of
  /// PatFrags from generating enormous amounts of rules.
  bool buildPermutationsToEmit();

  /// Checks additional semantics of the Patterns.
  bool checkSemantics();

  /// Creates a new RuleMatcher with some boilerplate
  /// settings/actions/predicates, and and adds it to \p OutRMs.
  /// \see addFeaturePredicates too.
  ///
  /// \param Alts Current set of alternatives, for debug comment.
  /// \param AdditionalComment Comment string to be added to the
  ///        `DebugCommentAction`.
  RuleMatcher &addRuleMatcher(const PatternAlternatives &Alts,
                              Twine AdditionalComment = "");
  bool addFeaturePredicates(RuleMatcher &M);

  bool findRoots();
  bool buildRuleOperandsTable();

  bool parseDefs(const DagInit &Def);
  bool
  parsePatternList(const DagInit &List,
                   function_ref<bool(std::unique_ptr<Pattern>)> ParseAction,
                   StringRef Operator, ArrayRef<SMLoc> DiagLoc,
                   StringRef AnonPatNamePrefix) const;

  std::unique_ptr<Pattern> parseInstructionPattern(const Init &Arg,
                                                   StringRef PatName) const;
  std::unique_ptr<Pattern> parseWipMatchOpcodeMatcher(const Init &Arg,
                                                      StringRef PatName) const;
  bool parseInstructionPatternOperand(InstructionPattern &IP,
                                      const Init *OpInit,
                                      const StringInit *OpName) const;
  std::unique_ptr<PatFrag> parsePatFragImpl(const Record *Def) const;
  bool parsePatFragParamList(
      ArrayRef<SMLoc> DiagLoc, const DagInit &OpsList,
      function_ref<bool(StringRef, PatFrag::ParamKind)> ParseAction) const;
  const PatFrag *parsePatFrag(const Record *Def) const;

  bool emitMatchPattern(CodeExpansions &CE, const PatternAlternatives &Alts,
                        const InstructionPattern &IP);
  bool emitMatchPattern(CodeExpansions &CE, const PatternAlternatives &Alts,
                        const AnyOpcodePattern &AOP);

  bool emitPatFragMatchPattern(CodeExpansions &CE,
                               const PatternAlternatives &Alts, RuleMatcher &RM,
                               InstructionMatcher *IM,
                               const PatFragPattern &PFP,
                               DenseSet<const Pattern *> &SeenPats);

  bool emitApplyPatterns(CodeExpansions &CE, RuleMatcher &M);

  // Recursively visits InstructionPatterns from P to build up the
  // RuleMatcher actions.
  bool emitInstructionApplyPattern(CodeExpansions &CE, RuleMatcher &M,
                                   const InstructionPattern &P,
                                   DenseSet<const Pattern *> &SeenPats,
                                   StringMap<unsigned> &OperandToTempRegID);

  bool emitCodeGenInstructionApplyImmOperand(RuleMatcher &M,
                                             BuildMIAction &DstMI,
                                             const CodeGenInstructionPattern &P,
                                             const InstructionOperand &O);

  bool emitBuiltinApplyPattern(CodeExpansions &CE, RuleMatcher &M,
                               const BuiltinPattern &P,
                               StringMap<unsigned> &OperandToTempRegID);

  // Recursively visits CodeGenInstructionPattern from P to build up the
  // RuleMatcher/InstructionMatcher. May create new InstructionMatchers as
  // needed.
  using OperandMapperFnRef =
      function_ref<InstructionOperand(const InstructionOperand &)>;
  using OperandDefLookupFn =
      function_ref<const InstructionPattern *(StringRef)>;
  bool emitCodeGenInstructionMatchPattern(
      CodeExpansions &CE, const PatternAlternatives &Alts, RuleMatcher &M,
      InstructionMatcher &IM, const CodeGenInstructionPattern &P,
      DenseSet<const Pattern *> &SeenPats, OperandDefLookupFn LookupOperandDef,
      OperandMapperFnRef OperandMapper = [](const auto &O) { return O; });

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

  /// Operand tables to tie match/apply patterns together.
  OperandTable<> MatchOpTable;
  OperandTable<> ApplyOpTable;

  /// Set by findRoots.
  Pattern *MatchRoot = nullptr;
  SmallDenseSet<InstructionPattern *, 2> ApplyRoots;

  SmallVector<MatchDataInfo, 2> MatchDatas;
  SmallVector<PatternAlternatives, 1> PermutationsToEmit;

  // print()/debug-only members.
  mutable SmallPtrSet<const PatFrag *, 2> SeenPatFrags;
};

bool CombineRuleBuilder::parseAll() {
  auto StackTrace = PrettyStackTraceParse(RuleDef);

  if (!parseDefs(*RuleDef.getValueAsDag("Defs")))
    return false;

  if (!parsePatternList(
          *RuleDef.getValueAsDag("Match"),
          [this](auto Pat) { return addMatchPattern(std::move(Pat)); }, "match",
          RuleDef.getLoc(), (RuleDef.getName() + "_match").str()))
    return false;

  if (!parsePatternList(
          *RuleDef.getValueAsDag("Apply"),
          [this](auto Pat) { return addApplyPattern(std::move(Pat)); }, "apply",
          RuleDef.getLoc(), (RuleDef.getName() + "_apply").str()))
    return false;

  if (!buildRuleOperandsTable() || !typecheckPatterns() || !findRoots() ||
      !checkSemantics() || !buildPermutationsToEmit())
    return false;
  LLVM_DEBUG(verify());
  return true;
}

bool CombineRuleBuilder::emitRuleMatchers() {
  auto StackTrace = PrettyStackTraceEmit(RuleDef);

  assert(MatchRoot);
  CodeExpansions CE;
  declareAllMatchDatasExpansions(CE);

  assert(!PermutationsToEmit.empty());
  for (const auto &Alts : PermutationsToEmit) {
    switch (MatchRoot->getKind()) {
    case Pattern::K_AnyOpcode: {
      if (!emitMatchPattern(CE, Alts, *cast<AnyOpcodePattern>(MatchRoot)))
        return false;
      break;
    }
    case Pattern::K_PatFrag:
    case Pattern::K_Builtin:
    case Pattern::K_CodeGenInstruction:
      if (!emitMatchPattern(CE, Alts, *cast<InstructionPattern>(MatchRoot)))
        return false;
      break;
    case Pattern::K_CXX:
      PrintError("C++ code cannot be the root of a rule!");
      return false;
    default:
      llvm_unreachable("unknown pattern kind!");
    }
  }

  return true;
}

void CombineRuleBuilder::print(raw_ostream &OS) const {
  OS << "(CombineRule name:" << RuleDef.getName() << " id:" << RuleID
     << " root:" << RootName << "\n";

  if (!MatchDatas.empty()) {
    OS << "  (MatchDatas\n";
    for (const auto &MD : MatchDatas) {
      OS << "    ";
      MD.print(OS);
      OS << "\n";
    }
    OS << "  )\n";
  }

  if (!SeenPatFrags.empty()) {
    OS << "  (PatFrags\n";
    for (const auto *PF : SeenPatFrags) {
      PF->print(OS, /*Indent=*/"    ");
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
        OS << "<match_root>";
      if (isa<InstructionPattern>(Pat.get()) &&
          ApplyRoots.contains(cast<InstructionPattern>(Pat.get())))
        OS << "<apply_root>";
      OS << Name << ":";
      Pat->print(OS, /*PrintName=*/false);
      OS << "\n";
    }
    OS << "  )\n";
  };

  DumpPats("MatchPats", MatchPats);
  DumpPats("ApplyPats", ApplyPats);

  MatchOpTable.print(OS, "MatchPats", /*Indent*/ "  ");
  ApplyOpTable.print(OS, "ApplyPats", /*Indent*/ "  ");

  if (PermutationsToEmit.size() > 1) {
    OS << "  (PermutationsToEmit\n";
    for (const auto &Perm : PermutationsToEmit) {
      OS << "    ";
      print(OS, Perm);
      OS << ",\n";
    }
    OS << "  )\n";
  }

  OS << ")\n";
}

#ifndef NDEBUG
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

      // Sanity check: the map should point to the same data as the Pattern.
      // Both strings are allocated in the pool using insertStrRef.
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

  // Check there are no wip_match_opcode patterns in the "apply" patterns.
  if (any_of(ApplyPats,
             [&](auto &E) { return isa<AnyOpcodePattern>(E.second.get()); })) {
    dump();
    PrintFatalError(
        "illegal wip_match_opcode pattern in the 'apply' patterns!");
  }

  // Check there are no nullptrs in ApplyRoots.
  if (ApplyRoots.contains(nullptr)) {
    PrintFatalError(
        "CombineRuleBuilder's ApplyRoots set contains a null pointer!");
  }
}
#endif

void CombineRuleBuilder::print(raw_ostream &OS,
                               const PatternAlternatives &Alts) const {
  SmallVector<std::string, 1> Strings(
      map_range(Alts, [](const auto &PatAndPerm) {
        return PatAndPerm.first->getName().str() + "[" +
               to_string(PatAndPerm.second) + "]";
      }));
  // Sort so output is deterministic for tests. Otherwise it's sorted by pointer
  // values.
  sort(Strings);
  OS << "[" << join(Strings, ", ") << "]";
}

bool CombineRuleBuilder::addApplyPattern(std::unique_ptr<Pattern> Pat) {
  StringRef Name = Pat->getName();
  if (ApplyPats.contains(Name)) {
    PrintError("'" + Name + "' apply pattern defined more than once!");
    return false;
  }

  if (isa<AnyOpcodePattern>(Pat.get())) {
    PrintError("'" + Name +
               "': wip_match_opcode is not supported in apply patterns");
    return false;
  }

  if (isa<PatFragPattern>(Pat.get())) {
    PrintError("'" + Name + "': using " + PatFragClassName +
               " is not supported in apply patterns");
    return false;
  }

  if (auto *CXXPat = dyn_cast<CXXPattern>(Pat.get()))
    CXXPat->setIsApply();

  ApplyPats[Name] = std::move(Pat);
  return true;
}

bool CombineRuleBuilder::addMatchPattern(std::unique_ptr<Pattern> Pat) {
  StringRef Name = Pat->getName();
  if (MatchPats.contains(Name)) {
    PrintError("'" + Name + "' match pattern defined more than once!");
    return false;
  }

  // For now, none of the builtins can appear in 'match'.
  if (const auto *BP = dyn_cast<BuiltinPattern>(Pat.get())) {
    PrintError("'" + BP->getInstName() +
               "' cannot be used in a 'match' pattern");
    return false;
  }

  MatchPats[Name] = std::move(Pat);
  return true;
}

void CombineRuleBuilder::declareAllMatchDatasExpansions(
    CodeExpansions &CE) const {
  for (const auto &MD : MatchDatas)
    CE.declare(MD.getPatternSymbol(), MD.getQualifiedVariableName());
}

void CombineRuleBuilder::addCXXPredicate(RuleMatcher &M,
                                         const CodeExpansions &CE,
                                         const CXXPattern &P,
                                         const PatternAlternatives &Alts) {
  // FIXME: Hack so C++ code is executed last. May not work for more complex
  // patterns.
  auto &IM = *std::prev(M.insnmatchers().end());
  const auto &ExpandedCode =
      P.expandCode(CE, RuleDef.getLoc(), [&](raw_ostream &OS) {
        OS << "// Pattern Alternatives: ";
        print(OS, Alts);
        OS << "\n";
      });
  IM->addPredicate<GenericInstructionPredicateMatcher>(
      ExpandedCode.getEnumNameWithPrefix(CXXPredPrefix));
}

void CombineRuleBuilder::addCXXAction(RuleMatcher &M, const CodeExpansions &CE,
                                      const CXXPattern &P) {
  const auto &ExpandedCode = P.expandCode(CE, RuleDef.getLoc());
  M.addAction<CustomCXXAction>(
      ExpandedCode.getEnumNameWithPrefix(CXXApplyPrefix));
}

bool CombineRuleBuilder::hasOnlyCXXApplyPatterns() const {
  return all_of(ApplyPats, [&](auto &Entry) {
    return isa<CXXPattern>(Entry.second.get());
  });
}

bool CombineRuleBuilder::hasEraseRoot() const {
  return any_of(ApplyPats, [&](auto &Entry) {
    if (const auto *BP = dyn_cast<BuiltinPattern>(Entry.second.get()))
      return BP->getBuiltinKind() == BI_EraseRoot;
    return false;
  });
}

bool CombineRuleBuilder::typecheckPatterns() {
  OperandTypeChecker OTC(RuleDef.getLoc());

  for (auto &Pat : values(MatchPats)) {
    if (auto *IP = dyn_cast<InstructionPattern>(Pat.get())) {
      if (!OTC.check(IP))
        return false;
    }
  }

  for (auto &Pat : values(ApplyPats)) {
    if (auto *IP = dyn_cast<InstructionPattern>(Pat.get())) {
      if (!OTC.check(IP))
        return false;
    }
  }

  OTC.setAllOperandTypes();
  return true;
}

bool CombineRuleBuilder::buildPermutationsToEmit() {
  PermutationsToEmit.clear();

  // Start with one empty set of alternatives.
  PermutationsToEmit.emplace_back();
  for (const auto &Pat : values(MatchPats)) {
    unsigned NumAlts = 0;
    // Note: technically, AnyOpcodePattern also needs permutations, but:
    //    - We only allow a single one of them in the root.
    //    - They cannot be mixed with any other pattern other than C++ code.
    // So we don't really need to take them into account here. We could, but
    // that pattern is a hack anyway and the less it's involved, the better.
    if (const auto *PFP = dyn_cast<PatFragPattern>(Pat.get()))
      NumAlts = PFP->getPatFrag().num_alternatives();
    else
      continue;

    // For each pattern that needs permutations, multiply the current set of
    // alternatives.
    auto CurPerms = PermutationsToEmit;
    PermutationsToEmit.clear();

    for (const auto &Perm : CurPerms) {
      assert(!Perm.count(Pat.get()) && "Pattern already emitted?");
      for (unsigned K = 0; K < NumAlts; ++K) {
        PatternAlternatives NewPerm = Perm;
        NewPerm[Pat.get()] = K;
        PermutationsToEmit.emplace_back(std::move(NewPerm));
      }
    }
  }

  if (int64_t MaxPerms = RuleDef.getValueAsInt("MaxPermutations");
      MaxPerms > 0) {
    if ((int64_t)PermutationsToEmit.size() > MaxPerms) {
      PrintError("cannot emit rule '" + RuleDef.getName() + "'; " +
                 Twine(PermutationsToEmit.size()) +
                 " permutations would be emitted, but the max is " +
                 Twine(MaxPerms));
      return false;
    }
  }

  // Ensure we always have a single empty entry, it simplifies the emission
  // logic so it doesn't need to handle the case where there are no perms.
  if (PermutationsToEmit.empty()) {
    PermutationsToEmit.emplace_back();
    return true;
  }

  return true;
}

bool CombineRuleBuilder::checkSemantics() {
  assert(MatchRoot && "Cannot call this before findRoots()");

  bool UsesWipMatchOpcode = false;
  for (const auto &Match : MatchPats) {
    const auto *Pat = Match.second.get();

    if (const auto *CXXPat = dyn_cast<CXXPattern>(Pat)) {
      if (!CXXPat->getRawCode().contains("return "))
        PrintWarning("'match' C++ code does not seem to return!");
      continue;
    }

    const auto *AOP = dyn_cast<AnyOpcodePattern>(Pat);
    if (!AOP)
      continue;

    if (UsesWipMatchOpcode) {
      PrintError("wip_opcode_match can only be present once");
      return false;
    }

    UsesWipMatchOpcode = true;
  }

  for (const auto &Apply : ApplyPats) {
    assert(Apply.second.get());
    const auto *IP = dyn_cast<InstructionPattern>(Apply.second.get());
    if (!IP)
      continue;

    if (UsesWipMatchOpcode) {
      PrintError("cannot use wip_match_opcode in combination with apply "
                 "instruction patterns!");
      return false;
    }

    const auto *BIP = dyn_cast<BuiltinPattern>(IP);
    if (!BIP)
      continue;
    StringRef Name = BIP->getInstName();

    // (GIEraseInst) has to be the only apply pattern, or it can not be used at
    // all. The root cannot have any defs either.
    switch (BIP->getBuiltinKind()) {
    case BI_EraseRoot: {
      if (ApplyPats.size() > 1) {
        PrintError(Name + " must be the only 'apply' pattern");
        return false;
      }

      const auto *IRoot = dyn_cast<CodeGenInstructionPattern>(MatchRoot);
      if (!IRoot) {
        PrintError(Name +
                   " can only be used if the root is a CodeGenInstruction");
        return false;
      }

      if (IRoot->getNumInstDefs() != 0) {
        PrintError(Name + " can only be used if on roots that do "
                          "not have any output operand");
        PrintNote("'" + IRoot->getInstName() + "' has " +
                  Twine(IRoot->getNumInstDefs()) + " output operands");
        return false;
      }
      break;
    }
    case BI_ReplaceReg: {
      // (GIReplaceReg can only be used on the root instruction)
      // TODO: When we allow rewriting non-root instructions, also allow this.
      StringRef OldRegName = BIP->getOperand(0).getOperandName();
      auto *Def = MatchOpTable.getDef(OldRegName);
      if (!Def) {
        PrintError(Name + " cannot find a matched pattern that defines '" +
                   OldRegName + "'");
        return false;
      }
      if (MatchOpTable.getDef(OldRegName) != MatchRoot) {
        PrintError(Name + " cannot replace '" + OldRegName +
                   "': this builtin can only replace a register defined by the "
                   "match root");
        return false;
      }
      break;
    }
    }
  }

  return true;
}

RuleMatcher &CombineRuleBuilder::addRuleMatcher(const PatternAlternatives &Alts,
                                                Twine AdditionalComment) {
  auto &RM = OutRMs.emplace_back(RuleDef.getLoc());
  addFeaturePredicates(RM);
  RM.setPermanentGISelFlags(GISF_IgnoreCopies);
  RM.addRequiredSimplePredicate(getIsEnabledPredicateEnumName(RuleID));

  std::string Comment;
  raw_string_ostream CommentOS(Comment);
  CommentOS << "Combiner Rule #" << RuleID << ": " << RuleDef.getName();
  if (!Alts.empty()) {
    CommentOS << " @ ";
    print(CommentOS, Alts);
  }
  if (!AdditionalComment.isTriviallyEmpty())
    CommentOS << "; " << AdditionalComment;
  RM.addAction<DebugCommentAction>(Comment);
  return RM;
}

bool CombineRuleBuilder::addFeaturePredicates(RuleMatcher &M) {
  if (!RuleDef.getValue("Predicates"))
    return true;

  ListInit *Preds = RuleDef.getValueAsListInit("Predicates");
  for (Init *PI : Preds->getValues()) {
    DefInit *Pred = dyn_cast<DefInit>(PI);
    if (!Pred)
      continue;

    Record *Def = Pred->getDef();
    if (!Def->isSubClassOf("Predicate")) {
      ::PrintError(Def, "Unknown 'Predicate' Type");
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

  return true;
}

bool CombineRuleBuilder::findRoots() {
  const auto Finish = [&]() {
    assert(MatchRoot);

    if (hasOnlyCXXApplyPatterns() || hasEraseRoot())
      return true;

    auto *IPRoot = dyn_cast<InstructionPattern>(MatchRoot);
    if (!IPRoot)
      return true;

    if (IPRoot->getNumInstDefs() == 0) {
      // No defs to work with -> find the root using the pattern name.
      auto It = ApplyPats.find(RootName);
      if (It == ApplyPats.end()) {
        PrintError("Cannot find root '" + RootName + "' in apply patterns!");
        return false;
      }

      auto *ApplyRoot = dyn_cast<InstructionPattern>(It->second.get());
      if (!ApplyRoot) {
        PrintError("apply pattern root '" + RootName +
                   "' must be an instruction pattern");
        return false;
      }

      ApplyRoots.insert(ApplyRoot);
      return true;
    }

    // Collect all redefinitions of the MatchRoot's defs and put them in
    // ApplyRoots.
    const auto DefsNeeded = IPRoot->getApplyDefsNeeded();
    for (auto &Op : DefsNeeded) {
      assert(Op.isDef() && Op.isNamedOperand());
      StringRef Name = Op.getOperandName();

      auto *ApplyRedef = ApplyOpTable.getDef(Name);
      if (!ApplyRedef) {
        PrintError("'" + Name + "' must be redefined in the 'apply' pattern");
        return false;
      }

      ApplyRoots.insert((InstructionPattern *)ApplyRedef);
    }

    if (auto It = ApplyPats.find(RootName); It != ApplyPats.end()) {
      if (find(ApplyRoots, It->second.get()) == ApplyRoots.end()) {
        PrintError("apply pattern '" + RootName +
                   "' is supposed to be a root but it does not redefine any of "
                   "the defs of the match root");
        return false;
      }
    }

    return true;
  };

  // Look by pattern name, e.g.
  //    (G_FNEG $x, $y):$root
  if (auto MatchPatIt = MatchPats.find(RootName);
      MatchPatIt != MatchPats.end()) {
    MatchRoot = MatchPatIt->second.get();
    return Finish();
  }

  // Look by def:
  //    (G_FNEG $root, $y)
  auto LookupRes = MatchOpTable.lookup(RootName);
  if (!LookupRes.Found) {
    PrintError("Cannot find root '" + RootName + "' in match patterns!");
    return false;
  }

  MatchRoot = LookupRes.Def;
  if (!MatchRoot) {
    PrintError("Cannot use live-in operand '" + RootName +
               "' as match pattern root!");
    return false;
  }

  return Finish();
}

bool CombineRuleBuilder::buildRuleOperandsTable() {
  const auto DiagnoseRedefMatch = [&](StringRef OpName) {
    PrintError("Operand '" + OpName +
               "' is defined multiple times in the 'match' patterns");
  };

  const auto DiagnoseRedefApply = [&](StringRef OpName) {
    PrintError("Operand '" + OpName +
               "' is defined multiple times in the 'apply' patterns");
  };

  for (auto &Pat : values(MatchPats)) {
    auto *IP = dyn_cast<InstructionPattern>(Pat.get());
    if (IP && !MatchOpTable.addPattern(IP, DiagnoseRedefMatch))
      return false;
  }

  for (auto &Pat : values(ApplyPats)) {
    auto *IP = dyn_cast<InstructionPattern>(Pat.get());
    if (IP && !ApplyOpTable.addPattern(IP, DiagnoseRedefApply))
      return false;
  }

  return true;
}

bool CombineRuleBuilder::parseDefs(const DagInit &Def) {
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

bool CombineRuleBuilder::parsePatternList(
    const DagInit &List,
    function_ref<bool(std::unique_ptr<Pattern>)> ParseAction,
    StringRef Operator, ArrayRef<SMLoc> DiagLoc,
    StringRef AnonPatNamePrefix) const {
  if (List.getOperatorAsDef(RuleDef.getLoc())->getName() != Operator) {
    ::PrintError(DiagLoc, "Expected " + Operator + " operator");
    return false;
  }

  if (List.getNumArgs() == 0) {
    ::PrintError(DiagLoc, Operator + " pattern list is empty");
    return false;
  }

  // The match section consists of a list of matchers and predicates. Parse each
  // one and add the equivalent GIMatchDag nodes, predicates, and edges.
  for (unsigned I = 0; I < List.getNumArgs(); ++I) {
    Init *Arg = List.getArg(I);
    std::string Name = List.getArgName(I)
                           ? List.getArgName(I)->getValue().str()
                           : makeAnonPatName(AnonPatNamePrefix, I);

    if (auto Pat = parseInstructionPattern(*Arg, Name)) {
      if (!ParseAction(std::move(Pat)))
        return false;
      continue;
    }

    if (auto Pat = parseWipMatchOpcodeMatcher(*Arg, Name)) {
      if (!ParseAction(std::move(Pat)))
        return false;
      continue;
    }

    // Parse arbitrary C++ code
    if (const auto *StringI = dyn_cast<StringInit>(Arg)) {
      auto CXXPat = std::make_unique<CXXPattern>(*StringI, Name);
      if (!ParseAction(std::move(CXXPat)))
        return false;
      continue;
    }

    ::PrintError(DiagLoc,
                 "Failed to parse pattern: '" + Arg->getAsString() + "'");
    return false;
  }

  return true;
}

std::unique_ptr<Pattern>
CombineRuleBuilder::parseInstructionPattern(const Init &Arg,
                                            StringRef Name) const {
  const DagInit *DagPat = dyn_cast<DagInit>(&Arg);
  if (!DagPat)
    return nullptr;

  std::unique_ptr<InstructionPattern> Pat;
  if (const DagInit *IP = getDagWithOperatorOfSubClass(Arg, "Instruction")) {
    auto &Instr = CGT.getInstruction(IP->getOperatorAsDef(RuleDef.getLoc()));
    Pat = std::make_unique<CodeGenInstructionPattern>(Instr, Name);
  } else if (const DagInit *PFP =
                 getDagWithOperatorOfSubClass(Arg, PatFragClassName)) {
    const Record *Def = PFP->getOperatorAsDef(RuleDef.getLoc());
    const PatFrag *PF = parsePatFrag(Def);
    if (!PF)
      return nullptr; // Already diagnosed by parsePatFrag
    Pat = std::make_unique<PatFragPattern>(*PF, Name);
  } else if (const DagInit *BP =
                 getDagWithOperatorOfSubClass(Arg, BuiltinInstClassName)) {
    Pat = std::make_unique<BuiltinPattern>(
        *BP->getOperatorAsDef(RuleDef.getLoc()), Name);
  } else {
    return nullptr;
  }

  for (unsigned K = 0; K < DagPat->getNumArgs(); ++K) {
    if (!parseInstructionPatternOperand(*Pat, DagPat->getArg(K),
                                        DagPat->getArgName(K)))
      return nullptr;
  }

  if (!Pat->checkSemantics(RuleDef.getLoc()))
    return nullptr;

  return std::move(Pat);
}

std::unique_ptr<Pattern>
CombineRuleBuilder::parseWipMatchOpcodeMatcher(const Init &Arg,
                                               StringRef Name) const {
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

bool CombineRuleBuilder::parseInstructionPatternOperand(
    InstructionPattern &IP, const Init *OpInit,
    const StringInit *OpName) const {
  const auto ParseErr = [&]() {
    PrintError("cannot parse operand '" + OpInit->getAsUnquotedString() + "' ");
    if (OpName)
      PrintNote("operand name is '" + OpName->getAsUnquotedString() + "'");
    return false;
  };

  // untyped immediate, e.g. 0
  if (const auto *IntImm = dyn_cast<IntInit>(OpInit)) {
    std::string Name = OpName ? OpName->getAsUnquotedString() : "";
    IP.addOperand(IntImm->getValue(), Name, /*Type=*/nullptr);
    return true;
  }

  // typed immediate, e.g. (i32 0)
  if (const auto *DagOp = dyn_cast<DagInit>(OpInit)) {
    if (DagOp->getNumArgs() != 1)
      return ParseErr();

    Record *ImmTy = DagOp->getOperatorAsDef(RuleDef.getLoc());
    if (!ImmTy->isSubClassOf("ValueType")) {
      PrintError("cannot parse immediate '" + OpInit->getAsUnquotedString() +
                 "', '" + ImmTy->getName() + "' is not a ValueType!");
      return false;
    }

    if (!IP.hasAllDefs()) {
      PrintError("out operand of '" + IP.getInstName() +
                 "' cannot be an immediate");
      return false;
    }

    const auto *Val = dyn_cast<IntInit>(DagOp->getArg(0));
    if (!Val)
      return ParseErr();

    std::string Name = OpName ? OpName->getAsUnquotedString() : "";
    IP.addOperand(Val->getValue(), Name, ImmTy);
    return true;
  }

  // Typed operand e.g. $x/$z in (G_FNEG $x, $z)
  if (auto *DefI = dyn_cast<DefInit>(OpInit)) {
    if (!OpName) {
      PrintError("expected an operand name after '" + OpInit->getAsString() +
                 "'");
      return false;
    }
    const Record *Def = DefI->getDef();
    if (!Def->isSubClassOf("ValueType")) {
      PrintError("invalid operand type: '" + Def->getName() +
                 "' is not a ValueType");
      return false;
    }
    IP.addOperand(OpName->getAsUnquotedString(), Def);
    return true;
  }

  // Untyped operand e.g. $x/$z in (G_FNEG $x, $z)
  if (isa<UnsetInit>(OpInit)) {
    assert(OpName && "Unset w/ no OpName?");
    IP.addOperand(OpName->getAsUnquotedString(), /*Type=*/nullptr);
    return true;
  }

  return ParseErr();
}

std::unique_ptr<PatFrag>
CombineRuleBuilder::parsePatFragImpl(const Record *Def) const {
  auto StackTrace = PrettyStackTraceParse(*Def);
  if (!Def->isSubClassOf(PatFragClassName))
    return nullptr;

  const DagInit *Ins = Def->getValueAsDag("InOperands");
  if (Ins->getOperatorAsDef(Def->getLoc())->getName() != "ins") {
    ::PrintError(Def, "expected 'ins' operator for " + PatFragClassName +
                          " in operands list");
    return nullptr;
  }

  const DagInit *Outs = Def->getValueAsDag("OutOperands");
  if (Outs->getOperatorAsDef(Def->getLoc())->getName() != "outs") {
    ::PrintError(Def, "expected 'outs' operator for " + PatFragClassName +
                          " out operands list");
    return nullptr;
  }

  auto Result = std::make_unique<PatFrag>(*Def);
  if (!parsePatFragParamList(Def->getLoc(), *Outs,
                             [&](StringRef Name, PatFrag::ParamKind Kind) {
                               Result->addOutParam(Name, Kind);
                               return true;
                             }))
    return nullptr;

  if (!parsePatFragParamList(Def->getLoc(), *Ins,
                             [&](StringRef Name, PatFrag::ParamKind Kind) {
                               Result->addInParam(Name, Kind);
                               return true;
                             }))
    return nullptr;

  const ListInit *Alts = Def->getValueAsListInit("Alternatives");
  unsigned AltIdx = 0;
  for (const Init *Alt : *Alts) {
    const auto *PatDag = dyn_cast<DagInit>(Alt);
    if (!PatDag) {
      ::PrintError(Def, "expected dag init for PatFrag pattern alternative");
      return nullptr;
    }

    PatFrag::Alternative &A = Result->addAlternative();
    const auto AddPat = [&](std::unique_ptr<Pattern> Pat) {
      A.Pats.push_back(std::move(Pat));
      return true;
    };

    if (!parsePatternList(
            *PatDag, AddPat, "pattern", Def->getLoc(),
            /*AnonPatPrefix*/
            (Def->getName() + "_alt" + Twine(AltIdx++) + "_pattern").str()))
      return nullptr;
  }

  if (!Result->buildOperandsTables() || !Result->checkSemantics())
    return nullptr;

  return Result;
}

bool CombineRuleBuilder::parsePatFragParamList(
    ArrayRef<SMLoc> DiagLoc, const DagInit &OpsList,
    function_ref<bool(StringRef, PatFrag::ParamKind)> ParseAction) const {
  for (unsigned K = 0; K < OpsList.getNumArgs(); ++K) {
    const StringInit *Name = OpsList.getArgName(K);
    const Init *Ty = OpsList.getArg(K);

    if (!Name) {
      ::PrintError(DiagLoc, "all operands must be named'");
      return false;
    }
    const std::string NameStr = Name->getAsUnquotedString();

    PatFrag::ParamKind OpKind;
    if (isSpecificDef(*Ty, "gi_imm"))
      OpKind = PatFrag::PK_Imm;
    else if (isSpecificDef(*Ty, "root"))
      OpKind = PatFrag::PK_Root;
    else if (isa<UnsetInit>(Ty) ||
             isSpecificDef(*Ty, "gi_mo")) // no type = gi_mo.
      OpKind = PatFrag::PK_MachineOperand;
    else {
      ::PrintError(
          DiagLoc,
          "'" + NameStr +
              "' operand type was expected to be 'root', 'gi_imm' or 'gi_mo'");
      return false;
    }

    if (!ParseAction(NameStr, OpKind))
      return false;
  }

  return true;
}

const PatFrag *CombineRuleBuilder::parsePatFrag(const Record *Def) const {
  // Cache already parsed PatFrags to avoid doing extra work.
  static DenseMap<const Record *, std::unique_ptr<PatFrag>> ParsedPatFrags;

  auto It = ParsedPatFrags.find(Def);
  if (It != ParsedPatFrags.end()) {
    SeenPatFrags.insert(It->second.get());
    return It->second.get();
  }

  std::unique_ptr<PatFrag> NewPatFrag = parsePatFragImpl(Def);
  if (!NewPatFrag) {
    ::PrintError(Def, "Could not parse " + PatFragClassName + " '" +
                          Def->getName() + "'");
    // Put a nullptr in the map so we don't attempt parsing this again.
    ParsedPatFrags[Def] = nullptr;
    return nullptr;
  }

  const auto *Res = NewPatFrag.get();
  ParsedPatFrags[Def] = std::move(NewPatFrag);
  SeenPatFrags.insert(Res);
  return Res;
}

bool CombineRuleBuilder::emitMatchPattern(CodeExpansions &CE,
                                          const PatternAlternatives &Alts,
                                          const InstructionPattern &IP) {
  auto StackTrace = PrettyStackTraceEmit(RuleDef, &IP);

  auto &M = addRuleMatcher(Alts);
  InstructionMatcher &IM = M.addInstructionMatcher("root");
  declareInstExpansion(CE, IM, IP.getName());

  DenseSet<const Pattern *> SeenPats;

  const auto FindOperandDef = [&](StringRef Op) -> InstructionPattern * {
    return MatchOpTable.getDef(Op);
  };

  if (const auto *CGP = dyn_cast<CodeGenInstructionPattern>(&IP)) {
    if (!emitCodeGenInstructionMatchPattern(CE, Alts, M, IM, *CGP, SeenPats,
                                            FindOperandDef))
      return false;
  } else if (const auto *PFP = dyn_cast<PatFragPattern>(&IP)) {
    if (!PFP->getPatFrag().canBeMatchRoot()) {
      PrintError("cannot use '" + PFP->getInstName() + " as match root");
      return false;
    }

    if (!emitPatFragMatchPattern(CE, Alts, M, &IM, *PFP, SeenPats))
      return false;
  } else if (isa<BuiltinPattern>(&IP)) {
    llvm_unreachable("No match builtins known!");
  } else
    llvm_unreachable("Unknown kind of InstructionPattern!");

  // Emit remaining patterns
  for (auto &Pat : values(MatchPats)) {
    if (SeenPats.contains(Pat.get()))
      continue;

    switch (Pat->getKind()) {
    case Pattern::K_AnyOpcode:
      PrintError("wip_match_opcode can not be used with instruction patterns!");
      return false;
    case Pattern::K_PatFrag: {
      if (!emitPatFragMatchPattern(CE, Alts, M, /*IM*/ nullptr,
                                   *cast<PatFragPattern>(Pat.get()), SeenPats))
        return false;
      continue;
    }
    case Pattern::K_Builtin:
      PrintError("No known match builtins");
      return false;
    case Pattern::K_CodeGenInstruction:
      cast<InstructionPattern>(Pat.get())->reportUnreachable(RuleDef.getLoc());
      return false;
    case Pattern::K_CXX: {
      addCXXPredicate(M, CE, *cast<CXXPattern>(Pat.get()), Alts);
      continue;
    }
    default:
      llvm_unreachable("unknown pattern kind!");
    }
  }

  return emitApplyPatterns(CE, M);
}

bool CombineRuleBuilder::emitMatchPattern(CodeExpansions &CE,
                                          const PatternAlternatives &Alts,
                                          const AnyOpcodePattern &AOP) {
  auto StackTrace = PrettyStackTraceEmit(RuleDef, &AOP);

  for (const CodeGenInstruction *CGI : AOP.insts()) {
    auto &M = addRuleMatcher(Alts, "wip_match_opcode '" +
                                       CGI->TheDef->getName() + "'");

    InstructionMatcher &IM = M.addInstructionMatcher(AOP.getName());
    declareInstExpansion(CE, IM, AOP.getName());
    // declareInstExpansion needs to be identical, otherwise we need to create a
    // CodeExpansions object here instead.
    assert(IM.getInsnVarID() == 0);

    IM.addPredicate<InstructionOpcodeMatcher>(CGI);

    // Emit remaining patterns.
    for (auto &Pat : values(MatchPats)) {
      if (Pat.get() == &AOP)
        continue;

      switch (Pat->getKind()) {
      case Pattern::K_AnyOpcode:
        PrintError("wip_match_opcode can only be present once!");
        return false;
      case Pattern::K_PatFrag: {
        DenseSet<const Pattern *> SeenPats;
        if (!emitPatFragMatchPattern(CE, Alts, M, /*IM*/ nullptr,
                                     *cast<PatFragPattern>(Pat.get()),
                                     SeenPats))
          return false;
        continue;
      }
      case Pattern::K_Builtin:
        PrintError("No known match builtins");
        return false;
      case Pattern::K_CodeGenInstruction:
        cast<InstructionPattern>(Pat.get())->reportUnreachable(
            RuleDef.getLoc());
        return false;
      case Pattern::K_CXX: {
        addCXXPredicate(M, CE, *cast<CXXPattern>(Pat.get()), Alts);
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

bool CombineRuleBuilder::emitPatFragMatchPattern(
    CodeExpansions &CE, const PatternAlternatives &Alts, RuleMatcher &RM,
    InstructionMatcher *IM, const PatFragPattern &PFP,
    DenseSet<const Pattern *> &SeenPats) {
  auto StackTrace = PrettyStackTraceEmit(RuleDef, &PFP);

  if (SeenPats.contains(&PFP))
    return true;
  SeenPats.insert(&PFP);

  const auto &PF = PFP.getPatFrag();

  if (!IM) {
    // When we don't have an IM, this means this PatFrag isn't reachable from
    // the root. This is only acceptable if it doesn't define anything (e.g. a
    // pure C++ PatFrag).
    if (PF.num_out_params() != 0) {
      PFP.reportUnreachable(RuleDef.getLoc());
      return false;
    }
  } else {
    // When an IM is provided, this is reachable from the root, and we're
    // expecting to have output operands.
    // TODO: If we want to allow for multiple roots we'll need a map of IMs
    // then, and emission becomes a bit more complicated.
    assert(PF.num_roots() == 1);
  }

  CodeExpansions PatFragCEs;
  if (!PFP.mapInputCodeExpansions(CE, PatFragCEs, RuleDef.getLoc()))
    return false;

  // List of {ParamName, ArgName}.
  // When all patterns have been emitted, find expansions in PatFragCEs named
  // ArgName and add their expansion to CE using ParamName as the key.
  SmallVector<std::pair<std::string, std::string>, 4> CEsToImport;

  // Map parameter names to the actual argument.
  const auto OperandMapper =
      [&](const InstructionOperand &O) -> InstructionOperand {
    if (!O.isNamedOperand())
      return O;

    StringRef ParamName = O.getOperandName();

    // Not sure what to do with those tbh. They should probably never be here.
    assert(!O.isNamedImmediate() && "TODO: handle named imms");
    unsigned PIdx = PF.getParamIdx(ParamName);

    // Map parameters to the argument values.
    if (PIdx == (unsigned)-1) {
      // This is a temp of the PatFragPattern, prefix the name to avoid
      // conflicts.
      return O.withNewName((PFP.getName() + "." + ParamName).str());
    }

    // The operand will be added to PatFragCEs's code expansions using the
    // parameter's name. If it's bound to some operand during emission of the
    // patterns, we'll want to add it to CE.
    auto ArgOp = PFP.getOperand(PIdx);
    if (ArgOp.isNamedOperand())
      CEsToImport.emplace_back(ArgOp.getOperandName().str(), ParamName);

    if (ArgOp.getType() && O.getType() && ArgOp.getType() != O.getType()) {
      StringRef PFName = PF.getName();
      PrintWarning("impossible type constraints: operand " + Twine(PIdx) +
                   " of '" + PFP.getName() + "' has type '" +
                   ArgOp.getType()->getName() + "', but '" + PFName +
                   "' constrains it to '" + O.getType()->getName() + "'");
      if (ArgOp.isNamedOperand())
        PrintNote("operand " + Twine(PIdx) + " of '" + PFP.getName() +
                  "' is '" + ArgOp.getOperandName() + "'");
      if (O.isNamedOperand())
        PrintNote("argument " + Twine(PIdx) + " of '" + PFName + "' is '" +
                  ParamName + "'");
    }

    return ArgOp;
  };

  // PatFragPatterns are only made of InstructionPatterns or CXXPatterns.
  // Emit instructions from the root.
  const auto &FragAlt = PF.getAlternative(Alts.lookup(&PFP));
  const auto &FragAltOT = FragAlt.OpTable;
  const auto LookupOperandDef =
      [&](StringRef Op) -> const InstructionPattern * {
    return FragAltOT.getDef(Op);
  };

  DenseSet<const Pattern *> PatFragSeenPats;
  for (const auto &[Idx, InOp] : enumerate(PF.out_params())) {
    if (InOp.Kind != PatFrag::PK_Root)
      continue;

    StringRef ParamName = InOp.Name;
    const auto *Def = FragAltOT.getDef(ParamName);
    assert(Def && "PatFrag::checkSemantics should have emitted an error if "
                  "an out operand isn't defined!");
    assert(isa<CodeGenInstructionPattern>(Def) &&
           "Nested PatFrags not supported yet");

    if (!emitCodeGenInstructionMatchPattern(
            PatFragCEs, Alts, RM, *IM, *cast<CodeGenInstructionPattern>(Def),
            PatFragSeenPats, LookupOperandDef, OperandMapper))
      return false;
  }

  // Emit leftovers.
  for (const auto &Pat : FragAlt.Pats) {
    if (PatFragSeenPats.contains(Pat.get()))
      continue;

    if (const auto *CXXPat = dyn_cast<CXXPattern>(Pat.get())) {
      addCXXPredicate(RM, PatFragCEs, *CXXPat, Alts);
      continue;
    }

    if (const auto *IP = dyn_cast<InstructionPattern>(Pat.get())) {
      IP->reportUnreachable(PF.getLoc());
      return false;
    }

    llvm_unreachable("Unexpected pattern kind in PatFrag");
  }

  for (const auto &[ParamName, ArgName] : CEsToImport) {
    // Note: we're find if ParamName already exists. It just means it's been
    // bound before, so we prefer to keep the first binding.
    CE.declare(ParamName, PatFragCEs.lookup(ArgName));
  }

  return true;
}

bool CombineRuleBuilder::emitApplyPatterns(CodeExpansions &CE, RuleMatcher &M) {
  if (hasOnlyCXXApplyPatterns()) {
    for (auto &Pat : values(ApplyPats))
      addCXXAction(M, CE, *cast<CXXPattern>(Pat.get()));
    return true;
  }

  DenseSet<const Pattern *> SeenPats;
  StringMap<unsigned> OperandToTempRegID;

  for (auto *ApplyRoot : ApplyRoots) {
    assert(isa<InstructionPattern>(ApplyRoot) &&
           "Root can only be a InstructionPattern!");
    if (!emitInstructionApplyPattern(CE, M,
                                     cast<InstructionPattern>(*ApplyRoot),
                                     SeenPats, OperandToTempRegID))
      return false;
  }

  for (auto &Pat : values(ApplyPats)) {
    if (SeenPats.contains(Pat.get()))
      continue;

    switch (Pat->getKind()) {
    case Pattern::K_AnyOpcode:
      llvm_unreachable("Unexpected pattern in apply!");
    case Pattern::K_PatFrag:
      // TODO: We could support pure C++ PatFrags as a temporary thing.
      llvm_unreachable("Unexpected pattern in apply!");
    case Pattern::K_Builtin:
      if (!emitInstructionApplyPattern(CE, M, cast<BuiltinPattern>(*Pat),
                                       SeenPats, OperandToTempRegID))
        return false;
      break;
    case Pattern::K_CodeGenInstruction:
      cast<CodeGenInstructionPattern>(*Pat).reportUnreachable(RuleDef.getLoc());
      return false;
    case Pattern::K_CXX: {
      addCXXAction(M, CE, *cast<CXXPattern>(Pat.get()));
      continue;
    }
    default:
      llvm_unreachable("unknown pattern kind!");
    }
  }

  return true;
}

bool CombineRuleBuilder::emitInstructionApplyPattern(
    CodeExpansions &CE, RuleMatcher &M, const InstructionPattern &P,
    DenseSet<const Pattern *> &SeenPats,
    StringMap<unsigned> &OperandToTempRegID) {
  auto StackTrace = PrettyStackTraceEmit(RuleDef, &P);

  if (SeenPats.contains(&P))
    return true;

  SeenPats.insert(&P);

  // First, render the uses.
  for (auto &Op : P.named_operands()) {
    if (Op.isDef())
      continue;

    StringRef OpName = Op.getOperandName();
    if (const auto *DefPat = ApplyOpTable.getDef(OpName)) {
      if (!emitInstructionApplyPattern(CE, M, *DefPat, SeenPats,
                                       OperandToTempRegID))
        return false;
    } else {
      // If we have no def, check this exists in the MatchRoot.
      if (!Op.isNamedImmediate() && !MatchOpTable.lookup(OpName).Found) {
        PrintError("invalid output operand '" + OpName +
                   "': operand is not a live-in of the match pattern, and it "
                   "has no definition");
        return false;
      }
    }
  }

  if (const auto *BP = dyn_cast<BuiltinPattern>(&P))
    return emitBuiltinApplyPattern(CE, M, *BP, OperandToTempRegID);

  if (isa<PatFragPattern>(&P))
    llvm_unreachable("PatFragPatterns is not supported in 'apply'!");

  auto &CGIP = cast<CodeGenInstructionPattern>(P);

  // Now render this inst.
  auto &DstMI =
      M.addAction<BuildMIAction>(M.allocateOutputInsnID(), &CGIP.getInst());

  for (auto &Op : P.operands()) {
    if (Op.isNamedImmediate()) {
      PrintError("invalid output operand '" + Op.getOperandName() +
                 "': output immediates cannot be named");
      PrintNote("while emitting pattern '" + P.getName() + "' (" +
                P.getInstName() + ")");
      return false;
    }

    if (Op.hasImmValue()) {
      if (!emitCodeGenInstructionApplyImmOperand(M, DstMI, CGIP, Op))
        return false;
      continue;
    }

    StringRef OpName = Op.getOperandName();

    // Uses of operand.
    if (!Op.isDef()) {
      if (auto It = OperandToTempRegID.find(OpName);
          It != OperandToTempRegID.end()) {
        assert(!MatchOpTable.lookup(OpName).Found &&
               "Temp reg is also from match pattern?");
        DstMI.addRenderer<TempRegRenderer>(It->second);
      } else {
        // This should be a match live in or a redef of a matched instr.
        // If it's a use of a temporary register, then we messed up somewhere -
        // the previous condition should have passed.
        assert(MatchOpTable.lookup(OpName).Found &&
               !ApplyOpTable.getDef(OpName) && "Temp reg not emitted yet!");
        DstMI.addRenderer<CopyRenderer>(OpName);
      }
      continue;
    }

    // Determine what we're dealing with. Are we replace a matched instruction?
    // Creating a new one?
    auto OpLookupRes = MatchOpTable.lookup(OpName);
    if (OpLookupRes.Found) {
      if (OpLookupRes.isLiveIn()) {
        // live-in of the match pattern.
        PrintError("Cannot define live-in operand '" + OpName +
                   "' in the 'apply' pattern");
        return false;
      }
      assert(OpLookupRes.Def);

      // TODO: Handle this. We need to mutate the instr, or delete the old
      // one.
      //       Likewise, we also need to ensure we redef everything, if the
      //       instr has more than one def, we need to redef all or nothing.
      if (OpLookupRes.Def != MatchRoot) {
        PrintError("redefining an instruction other than the root is not "
                   "supported (operand '" +
                   OpName + "')");
        return false;
      }
      // redef of a match
      DstMI.addRenderer<CopyRenderer>(OpName);
      continue;
    }

    // Define a new register unique to the apply patterns (AKA a "temp"
    // register).
    unsigned TempRegID;
    if (auto It = OperandToTempRegID.find(OpName);
        It != OperandToTempRegID.end()) {
      TempRegID = It->second;
    } else {
      // This is a brand new register.
      TempRegID = M.allocateTempRegID();
      OperandToTempRegID[OpName] = TempRegID;
      const Record *Ty = Op.getType();
      if (!Ty) {
        PrintError("def of a new register '" + OpName +
                   "' in the apply patterns must have a type");
        return false;
      }
      declareTempRegExpansion(CE, TempRegID, OpName);
      // Always insert the action at the beginning, otherwise we may end up
      // using the temp reg before it's available.
      M.insertAction<MakeTempRegisterAction>(
          M.actions_begin(), getLLTCodeGenFromRecord(Ty), TempRegID);
    }

    DstMI.addRenderer<TempRegRenderer>(TempRegID);
  }

  // TODO: works?
  DstMI.chooseInsnToMutate(M);
  declareInstExpansion(CE, DstMI, P.getName());

  return true;
}

bool CombineRuleBuilder::emitCodeGenInstructionApplyImmOperand(
    RuleMatcher &M, BuildMIAction &DstMI, const CodeGenInstructionPattern &P,
    const InstructionOperand &O) {
  // If we have a type, we implicitly emit a G_CONSTANT, except for G_CONSTANT
  // itself where we emit a CImm.
  //
  // No type means we emit a simple imm.
  // G_CONSTANT is a special case and needs a CImm though so this is likely a
  // mistake.
  const bool isGConstant = P.is("G_CONSTANT");
  const Record *Ty = O.getType();
  if (!Ty) {
    if (isGConstant) {
      PrintError("'G_CONSTANT' immediate must be typed!");
      PrintNote("while emitting pattern '" + P.getName() + "' (" +
                P.getInstName() + ")");
      return false;
    }

    DstMI.addRenderer<ImmRenderer>(O.getImmValue());
    return true;
  }

  LLTCodeGen LLT = getLLTCodeGenFromRecord(Ty);
  if (isGConstant) {
    DstMI.addRenderer<ImmRenderer>(O.getImmValue(), LLT);
    return true;
  }

  unsigned TempRegID = M.allocateTempRegID();
  auto ActIt = M.insertAction<BuildMIAction>(
      M.actions_begin(), M.allocateOutputInsnID(), &getGConstant());
  // Ensure MakeTempReg occurs before the BuildMI of th G_CONSTANT.
  M.insertAction<MakeTempRegisterAction>(ActIt, LLT, TempRegID);
  auto &ConstantMI = *static_cast<BuildMIAction *>(ActIt->get());
  ConstantMI.addRenderer<TempRegRenderer>(TempRegID);
  ConstantMI.addRenderer<ImmRenderer>(O.getImmValue(), LLT);
  DstMI.addRenderer<TempRegRenderer>(TempRegID);
  return true;
}

bool CombineRuleBuilder::emitBuiltinApplyPattern(
    CodeExpansions &CE, RuleMatcher &M, const BuiltinPattern &P,
    StringMap<unsigned> &OperandToTempRegID) {
  const auto Error = [&](Twine Reason) {
    PrintError("cannot emit '" + P.getInstName() + "' builtin: " + Reason);
    return false;
  };

  switch (P.getBuiltinKind()) {
  case BI_EraseRoot: {
    // Root is always inst 0.
    M.addAction<EraseInstAction>(/*InsnID*/ 0);
    return true;
  }
  case BI_ReplaceReg: {
    StringRef Old = P.getOperand(0).getOperandName();
    StringRef New = P.getOperand(1).getOperandName();

    if (!ApplyOpTable.lookup(New).Found && !MatchOpTable.lookup(New).Found)
      return Error("unknown operand '" + Old + "'");

    auto &OldOM = M.getOperandMatcher(Old);
    if (auto It = OperandToTempRegID.find(New);
        It != OperandToTempRegID.end()) {
      // Replace with temp reg.
      M.addAction<ReplaceRegAction>(OldOM.getInsnVarID(), OldOM.getOpIdx(),
                                    It->second);
    } else {
      // Replace with matched reg.
      auto &NewOM = M.getOperandMatcher(New);
      M.addAction<ReplaceRegAction>(OldOM.getInsnVarID(), OldOM.getOpIdx(),
                                    NewOM.getInsnVarID(), NewOM.getOpIdx());
    }
    // checkSemantics should have ensured that we can only rewrite the root.
    // Ensure we're deleting it.
    assert(MatchOpTable.getDef(Old) == MatchRoot);
    // TODO: We could avoid adding the action again if it's already in. The
    // MatchTable is smart enough to only emit one opcode even if
    // EraseInstAction is present multiple times. I think searching for a copy
    // is more expensive than just blindly adding it though.
    M.addAction<EraseInstAction>(/*InsnID*/ 0);

    return true;
  }
  }

  llvm_unreachable("Unknown BuiltinKind!");
}

bool isLiteralImm(const InstructionPattern &P, unsigned OpIdx) {
  if (const auto *CGP = dyn_cast<CodeGenInstructionPattern>(&P)) {
    StringRef InstName = CGP->getInst().TheDef->getName();
    return (InstName == "G_CONSTANT" || InstName == "G_FCONSTANT") &&
           OpIdx == 1;
  }

  llvm_unreachable("TODO");
}

bool CombineRuleBuilder::emitCodeGenInstructionMatchPattern(
    CodeExpansions &CE, const PatternAlternatives &Alts, RuleMatcher &M,
    InstructionMatcher &IM, const CodeGenInstructionPattern &P,
    DenseSet<const Pattern *> &SeenPats, OperandDefLookupFn LookupOperandDef,
    OperandMapperFnRef OperandMapper) {
  auto StackTrace = PrettyStackTraceEmit(RuleDef, &P);

  if (SeenPats.contains(&P))
    return true;

  SeenPats.insert(&P);

  IM.addPredicate<InstructionOpcodeMatcher>(&P.getInst());
  declareInstExpansion(CE, IM, P.getName());

  for (const auto &[Idx, OriginalO] : enumerate(P.operands())) {
    // Remap the operand. This is used when emitting InstructionPatterns inside
    // PatFrags, so it can remap them to the arguments passed to the pattern.
    //
    // We use the remapped operand to emit immediates, and for the symbolic
    // operand names (in IM.addOperand). CodeExpansions and OperandTable lookups
    // still use the original name.
    //
    // The "def" flag on the remapped operand is always ignored.
    auto RemappedO = OperandMapper(OriginalO);
    assert(RemappedO.isNamedOperand() == OriginalO.isNamedOperand() &&
           "Cannot remap an unnamed operand to a named one!");

    const auto OpName =
        RemappedO.isNamedOperand() ? RemappedO.getOperandName().str() : "";
    OperandMatcher &OM =
        IM.addOperand(Idx, OpName, AllocatedTemporariesBaseID++);
    if (!OpName.empty())
      declareOperandExpansion(CE, OM, OriginalO.getOperandName());

    // Handle immediates.
    if (RemappedO.hasImmValue()) {
      if (isLiteralImm(P, Idx))
        OM.addPredicate<LiteralIntOperandMatcher>(RemappedO.getImmValue());
      else
        OM.addPredicate<ConstantIntOperandMatcher>(RemappedO.getImmValue());
    }

    // Handle typed operands, but only bother to check if it hasn't been done
    // before.
    //
    // getOperandMatcher will always return the first OM to have been created
    // for that Operand. "OM" here is always a new OperandMatcher.
    //
    // Always emit a check for unnamed operands.
    if (OpName.empty() ||
        !M.getOperandMatcher(OpName).contains<LLTOperandMatcher>()) {
      if (const Record *Ty = RemappedO.getType())
        OM.addPredicate<LLTOperandMatcher>(getLLTCodeGenFromRecord(Ty));
    }

    // Stop here if the operand is a def, or if it had no name.
    if (OriginalO.isDef() || !OriginalO.isNamedOperand())
      continue;

    const auto *DefPat = LookupOperandDef(OriginalO.getOperandName());
    if (!DefPat)
      continue;

    if (OriginalO.hasImmValue()) {
      assert(!OpName.empty());
      // This is a named immediate that also has a def, that's not okay.
      // e.g.
      //    (G_SEXT $y, (i32 0))
      //    (COPY $x, 42:$y)
      PrintError("'" + OpName +
                 "' is a named immediate, it cannot be defined by another "
                 "instruction");
      PrintNote("'" + OpName + "' is defined by '" + DefPat->getName() + "'");
      return false;
    }

    // From here we know that the operand defines an instruction, and we need to
    // emit it.
    auto InstOpM =
        OM.addPredicate<InstructionOperandMatcher>(M, DefPat->getName());
    if (!InstOpM) {
      // TODO: copy-pasted from GlobalISelEmitter.cpp. Is it still relevant
      // here?
      PrintError("Nested instruction '" + DefPat->getName() +
                 "' cannot be the same as another operand '" +
                 OriginalO.getOperandName() + "'");
      return false;
    }

    auto &IM = (*InstOpM)->getInsnMatcher();
    if (const auto *CGIDef = dyn_cast<CodeGenInstructionPattern>(DefPat)) {
      if (!emitCodeGenInstructionMatchPattern(CE, Alts, M, IM, *CGIDef,
                                              SeenPats, LookupOperandDef,
                                              OperandMapper))
        return false;
      continue;
    }

    if (const auto *PFPDef = dyn_cast<PatFragPattern>(DefPat)) {
      if (!emitPatFragMatchPattern(CE, Alts, M, &IM, *PFPDef, SeenPats))
        return false;
      continue;
    }

    llvm_unreachable("unknown type of InstructionPattern");
  }

  return true;
}

//===- GICombinerEmitter --------------------------------------------------===//

/// Main implementation class. This emits the tablegenerated output.
///
/// It collects rules, uses `CombineRuleBuilder` to parse them and accumulate
/// RuleMatchers, then takes all the necessary state/data from the various
/// static storage pools and wires them together to emit the match table &
/// associated function/data structures.
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

  // Keep track of all rules we've seen so far to ensure we don't process
  // the same rule twice.
  StringSet<> RulesSeen;

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

  StringRef getCombineAllMethodName() const {
    return Combiner->getValueAsString("CombineAllMethodName");
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
  OS << "bool " << getClassName() << "::" << getCombineAllMethodName()
     << "(MachineInstr &I) const {\n"
     << "  const TargetSubtargetInfo &ST = MF.getSubtarget();\n"
     << "  const PredicateBitset AvailableFeatures = "
        "getAvailableFeatures();\n"
     << "  B.setInstrAndDebugLoc(I);\n"
     << "  State.MIs.clear();\n"
     << "  State.MIs.push_back(&I);\n"
     << "  " << MatchDataInfo::StructName << " = "
     << MatchDataInfo::StructTypeName << "();\n\n"
     << "  if (executeMatchTable(*this, State, ExecInfo, B"
     << ", getMatchTable(), *ST.getInstrInfo(), MRI, "
        "*MRI.getTargetRegisterInfo(), *ST.getRegBankInfo(), AvailableFeatures"
     << ", /*CoverageInfo*/ nullptr)) {\n"
     << "    return true;\n"
     << "  }\n\n"
     << "  return false;\n"
     << "}\n\n";
}

void GICombinerEmitter::emitMIPredicateFns(raw_ostream &OS) {
  auto MatchCode = CXXPredicateCode::getAllMatchCode();
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
    for (const auto &ID : keys(AllCombineRules)) {
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
  const auto ApplyCode = CXXPredicateCode::getAllApplyCode();

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
     << "::runCustomAction(unsigned ApplyID, const MatcherState &State, "
        "NewMIVector &OutMIs) const "
        "{\n";
  if (!ApplyCode.empty()) {
    OS << "  switch(ApplyID) {\n";
    for (const auto &Apply : ApplyCode) {
      OS << "  case " << Apply->getEnumNameWithPrefix(CXXApplyPrefix) << ":{\n"
         << "    " << join(split(Apply->Code, "\n"), "\n    ") << "\n"
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
  for (Record *Rec : RulesAndGroups) {
    if (!Rec->isValueUnset("Rules")) {
      gatherRules(ActiveRules, Rec->getValueAsListOfDefs("Rules"));
      continue;
    }

    StringRef RuleName = Rec->getName();
    if (!RulesSeen.insert(RuleName).second) {
      PrintWarning(Rec->getLoc(),
                   "skipping rule '" + Rec->getName() +
                       "' because it has already been processed");
      continue;
    }

    AllCombineRules.emplace_back(NextRuleID, Rec->getName().str());
    CombineRuleBuilder CRB(Target, SubtargetFeatures, *Rec, NextRuleID++,
                           ActiveRules);

    if (!CRB.parseAll()) {
      assert(ErrorsPrinted && "Parsing failed without errors!");
      continue;
    }

    if (StopAfterParse) {
      CRB.print(outs());
      continue;
    }

    if (!CRB.emitRuleMatchers()) {
      assert(ErrorsPrinted && "Emission failed without errors!");
      continue;
    }
  }
}

void GICombinerEmitter::run(raw_ostream &OS) {
  InstructionOpcodeMatcher::initOpcodeValuesMap(Target);
  LLTOperandMatcher::initTypeIDValuesMap();

  Records.startTimer("Gather rules");
  std::vector<RuleMatcher> Rules;
  gatherRules(Rules, Combiner->getValueAsListOfDefs("Rules"));
  if (ErrorsPrinted)
    PrintFatalError(Combiner->getLoc(), "Failed to parse one or more rules");

  if (StopAfterParse)
    return;

  Records.startTimer("Creating Match Table");
  unsigned MaxTemporaries = 0;
  for (const auto &Rule : Rules)
    MaxTemporaries = std::max(MaxTemporaries, Rule.countRendererFns());

  llvm::stable_sort(Rules, [&](const RuleMatcher &A, const RuleMatcher &B) {
    if (A.isHigherPriorityThan(B)) {
      assert(!B.isHigherPriorityThan(A) && "Cannot be more important "
                                           "and less important at "
                                           "the same time");
      return true;
    }
    return false;
  });

  const MatchTable Table = buildMatchTable(Rules);

  Records.startTimer("Emit combiner");

  emitSourceFileHeader(getClassName().str() + " Combiner Match Table", OS);

  // Unused
  std::vector<StringRef> CustomRendererFns;
  // Unused
  std::vector<Record *> ComplexPredicates;

  SmallVector<LLTCodeGen, 16> TypeObjects;
  append_range(TypeObjects, KnownTypes);
  llvm::sort(TypeObjects);

  // Hack: Avoid empty declarator.
  if (TypeObjects.empty())
    TypeObjects.push_back(LLT::scalar(1));

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
  EnablePrettyStackTrace();
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

static TableGen::Emitter::Opt X("gen-global-isel-combiner", EmitGICombiner,
                                "Generate GlobalISel Combiner");
