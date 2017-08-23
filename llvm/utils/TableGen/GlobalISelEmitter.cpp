//===- GlobalISelEmitter.cpp - Generate an instruction selector -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This tablegen backend emits code for use by the GlobalISel instruction
/// selector. See include/llvm/CodeGen/TargetGlobalISel.td.
///
/// This file analyzes the patterns recognized by the SelectionDAGISel tablegen
/// backend, filters out the ones that are unsupported, maps
/// SelectionDAG-specific constructs to their GlobalISel counterpart
/// (when applicable: MVT to LLT;  SDNode to generic Instruction).
///
/// Not all patterns are supported: pass the tablegen invocation
/// "-warn-on-skipped-patterns" to emit a warning when a pattern is skipped,
/// as well as why.
///
/// The generated file defines a single method:
///     bool <Target>InstructionSelector::selectImpl(MachineInstr &I) const;
/// intended to be used in InstructionSelector::select as the first-step
/// selector for the patterns that don't require complex C++.
///
/// FIXME: We'll probably want to eventually define a base
/// "TargetGenInstructionSelector" class.
///
//===----------------------------------------------------------------------===//

#include "CodeGenDAGPatterns.h"
#include "SubtargetFeatureInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LowLevelTypeImpl.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
#include <numeric>
using namespace llvm;

#define DEBUG_TYPE "gisel-emitter"

STATISTIC(NumPatternTotal, "Total number of patterns");
STATISTIC(NumPatternImported, "Number of patterns imported from SelectionDAG");
STATISTIC(NumPatternImportsSkipped, "Number of SelectionDAG imports skipped");
STATISTIC(NumPatternEmitted, "Number of patterns emitted");

cl::OptionCategory GlobalISelEmitterCat("Options for -gen-global-isel");

static cl::opt<bool> WarnOnSkippedPatterns(
    "warn-on-skipped-patterns",
    cl::desc("Explain why a pattern was skipped for inclusion "
             "in the GlobalISel selector"),
    cl::init(false), cl::cat(GlobalISelEmitterCat));

namespace {
//===- Helper functions ---------------------------------------------------===//

/// This class stands in for LLT wherever we want to tablegen-erate an
/// equivalent at compiler run-time.
class LLTCodeGen {
private:
  LLT Ty;

public:
  LLTCodeGen(const LLT &Ty) : Ty(Ty) {}

  std::string getCxxEnumValue() const {
    std::string Str;
    raw_string_ostream OS(Str);

    emitCxxEnumValue(OS);
    return OS.str();
  }

  void emitCxxEnumValue(raw_ostream &OS) const {
    if (Ty.isScalar()) {
      OS << "GILLT_s" << Ty.getSizeInBits();
      return;
    }
    if (Ty.isVector()) {
      OS << "GILLT_v" << Ty.getNumElements() << "s" << Ty.getScalarSizeInBits();
      return;
    }
    llvm_unreachable("Unhandled LLT");
  }

  void emitCxxConstructorCall(raw_ostream &OS) const {
    if (Ty.isScalar()) {
      OS << "LLT::scalar(" << Ty.getSizeInBits() << ")";
      return;
    }
    if (Ty.isVector()) {
      OS << "LLT::vector(" << Ty.getNumElements() << ", "
         << Ty.getScalarSizeInBits() << ")";
      return;
    }
    llvm_unreachable("Unhandled LLT");
  }

  const LLT &get() const { return Ty; }

  /// This ordering is used for std::unique() and std::sort(). There's no
  /// particular logic behind the order but either A < B or B < A must be
  /// true if A != B.
  bool operator<(const LLTCodeGen &Other) const {
    if (Ty.isValid() != Other.Ty.isValid())
      return Ty.isValid() < Other.Ty.isValid();
    if (!Ty.isValid())
      return false;

    if (Ty.isVector() != Other.Ty.isVector())
      return Ty.isVector() < Other.Ty.isVector();
    if (Ty.isScalar() != Other.Ty.isScalar())
      return Ty.isScalar() < Other.Ty.isScalar();
    if (Ty.isPointer() != Other.Ty.isPointer())
      return Ty.isPointer() < Other.Ty.isPointer();

    if (Ty.isPointer() && Ty.getAddressSpace() != Other.Ty.getAddressSpace())
      return Ty.getAddressSpace() < Other.Ty.getAddressSpace();

    if (Ty.isVector() && Ty.getNumElements() != Other.Ty.getNumElements())
      return Ty.getNumElements() < Other.Ty.getNumElements();

    return Ty.getSizeInBits() < Other.Ty.getSizeInBits();
  }
};

class InstructionMatcher;
/// Convert an MVT to an equivalent LLT if possible, or the invalid LLT() for
/// MVTs that don't map cleanly to an LLT (e.g., iPTR, *any, ...).
static Optional<LLTCodeGen> MVTToLLT(MVT::SimpleValueType SVT) {
  MVT VT(SVT);
  if (VT.isVector() && VT.getVectorNumElements() != 1)
    return LLTCodeGen(
        LLT::vector(VT.getVectorNumElements(), VT.getScalarSizeInBits()));
  if (VT.isInteger() || VT.isFloatingPoint())
    return LLTCodeGen(LLT::scalar(VT.getSizeInBits()));
  return None;
}

static std::string explainPredicates(const TreePatternNode *N) {
  std::string Explanation = "";
  StringRef Separator = "";
  for (const auto &P : N->getPredicateFns()) {
    Explanation +=
        (Separator + P.getOrigPatFragRecord()->getRecord()->getName()).str();
    if (P.isAlwaysTrue())
      Explanation += " always-true";
    if (P.isImmediatePattern())
      Explanation += " immediate";
  }
  return Explanation;
}

std::string explainOperator(Record *Operator) {
  if (Operator->isSubClassOf("SDNode"))
    return (" (" + Operator->getValueAsString("Opcode") + ")").str();

  if (Operator->isSubClassOf("Intrinsic"))
    return (" (Operator is an Intrinsic, " + Operator->getName() + ")").str();

  return " (Operator not understood)";
}

/// Helper function to let the emitter report skip reason error messages.
static Error failedImport(const Twine &Reason) {
  return make_error<StringError>(Reason, inconvertibleErrorCode());
}

static Error isTrivialOperatorNode(const TreePatternNode *N) {
  std::string Explanation = "";
  std::string Separator = "";

  bool HasUnsupportedPredicate = false;
  for (const auto &Predicate : N->getPredicateFns()) {
    if (Predicate.isAlwaysTrue())
      continue;

    if (Predicate.isImmediatePattern())
      continue;

    HasUnsupportedPredicate = true;
    Explanation = Separator + "Has a predicate (" + explainPredicates(N) + ")";
    Separator = ", ";
    break;
  }

  if (N->getTransformFn()) {
    Explanation += Separator + "Has a transform function";
    Separator = ", ";
  }

  if (!HasUnsupportedPredicate && !N->getTransformFn())
    return Error::success();

  return failedImport(Explanation);
}

static Record *getInitValueAsRegClass(Init *V) {
  if (DefInit *VDefInit = dyn_cast<DefInit>(V)) {
    if (VDefInit->getDef()->isSubClassOf("RegisterOperand"))
      return VDefInit->getDef()->getValueAsDef("RegClass");
    if (VDefInit->getDef()->isSubClassOf("RegisterClass"))
      return VDefInit->getDef();
  }
  return nullptr;
}

std::string
getNameForFeatureBitset(const std::vector<Record *> &FeatureBitset) {
  std::string Name = "GIFBS";
  for (const auto &Feature : FeatureBitset)
    Name += ("_" + Feature->getName()).str();
  return Name;
}

//===- MatchTable Helpers -------------------------------------------------===//

class MatchTable;

/// A record to be stored in a MatchTable.
///
/// This class represents any and all output that may be required to emit the
/// MatchTable. Instances  are most often configured to represent an opcode or
/// value that will be emitted to the table with some formatting but it can also
/// represent commas, comments, and other formatting instructions.
struct MatchTableRecord {
  enum RecordFlagsBits {
    MTRF_None = 0x0,
    /// Causes EmitStr to be formatted as comment when emitted.
    MTRF_Comment = 0x1,
    /// Causes the record value to be followed by a comma when emitted.
    MTRF_CommaFollows = 0x2,
    /// Causes the record value to be followed by a line break when emitted.
    MTRF_LineBreakFollows = 0x4,
    /// Indicates that the record defines a label and causes an additional
    /// comment to be emitted containing the index of the label.
    MTRF_Label = 0x8,
    /// Causes the record to be emitted as the index of the label specified by
    /// LabelID along with a comment indicating where that label is.
    MTRF_JumpTarget = 0x10,
    /// Causes the formatter to add a level of indentation before emitting the
    /// record.
    MTRF_Indent = 0x20,
    /// Causes the formatter to remove a level of indentation after emitting the
    /// record.
    MTRF_Outdent = 0x40,
  };

  /// When MTRF_Label or MTRF_JumpTarget is used, indicates a label id to
  /// reference or define.
  unsigned LabelID;
  /// The string to emit. Depending on the MTRF_* flags it may be a comment, a
  /// value, a label name.
  std::string EmitStr;

private:
  /// The number of MatchTable elements described by this record. Comments are 0
  /// while values are typically 1. Values >1 may occur when we need to emit
  /// values that exceed the size of a MatchTable element.
  unsigned NumElements;

public:
  /// A bitfield of RecordFlagsBits flags.
  unsigned Flags;

  MatchTableRecord(Optional<unsigned> LabelID_, StringRef EmitStr,
                   unsigned NumElements, unsigned Flags)
      : LabelID(LabelID_.hasValue() ? LabelID_.getValue() : ~0u),
        EmitStr(EmitStr), NumElements(NumElements), Flags(Flags) {
    assert((!LabelID_.hasValue() || LabelID != ~0u) &&
           "This value is reserved for non-labels");
  }

  void emit(raw_ostream &OS, bool LineBreakNextAfterThis,
            const MatchTable &Table) const;
  unsigned size() const { return NumElements; }
};

/// Holds the contents of a generated MatchTable to enable formatting and the
/// necessary index tracking needed to support GIM_Try.
class MatchTable {
  /// An unique identifier for the table. The generated table will be named
  /// MatchTable${ID}.
  unsigned ID;
  /// The records that make up the table. Also includes comments describing the
  /// values being emitted and line breaks to format it.
  std::vector<MatchTableRecord> Contents;
  /// The currently defined labels.
  DenseMap<unsigned, unsigned> LabelMap;
  /// Tracks the sum of MatchTableRecord::NumElements as the table is built.
  unsigned CurrentSize;

  /// A unique identifier for a MatchTable label.
  static unsigned CurrentLabelID;

public:
  static MatchTableRecord LineBreak;
  static MatchTableRecord Comment(StringRef Comment) {
    return MatchTableRecord(None, Comment, 0, MatchTableRecord::MTRF_Comment);
  }
  static MatchTableRecord Opcode(StringRef Opcode, int IndentAdjust = 0) {
    unsigned ExtraFlags = 0;
    if (IndentAdjust > 0)
      ExtraFlags |= MatchTableRecord::MTRF_Indent;
    if (IndentAdjust < 0)
      ExtraFlags |= MatchTableRecord::MTRF_Outdent;

    return MatchTableRecord(None, Opcode, 1,
                            MatchTableRecord::MTRF_CommaFollows | ExtraFlags);
  }
  static MatchTableRecord NamedValue(StringRef NamedValue) {
    return MatchTableRecord(None, NamedValue, 1,
                            MatchTableRecord::MTRF_CommaFollows);
  }
  static MatchTableRecord NamedValue(StringRef Namespace,
                                     StringRef NamedValue) {
    return MatchTableRecord(None, (Namespace + "::" + NamedValue).str(), 1,
                            MatchTableRecord::MTRF_CommaFollows);
  }
  static MatchTableRecord IntValue(int64_t IntValue) {
    return MatchTableRecord(None, llvm::to_string(IntValue), 1,
                            MatchTableRecord::MTRF_CommaFollows);
  }
  static MatchTableRecord Label(unsigned LabelID) {
    return MatchTableRecord(LabelID, "Label " + llvm::to_string(LabelID), 0,
                            MatchTableRecord::MTRF_Label |
                                MatchTableRecord::MTRF_Comment |
                                MatchTableRecord::MTRF_LineBreakFollows);
  }
  static MatchTableRecord JumpTarget(unsigned LabelID) {
    return MatchTableRecord(LabelID, "Label " + llvm::to_string(LabelID), 1,
                            MatchTableRecord::MTRF_JumpTarget |
                                MatchTableRecord::MTRF_Comment |
                                MatchTableRecord::MTRF_CommaFollows);
  }

  MatchTable(unsigned ID) : ID(ID), CurrentSize(0) {}

  void push_back(const MatchTableRecord &Value) {
    if (Value.Flags & MatchTableRecord::MTRF_Label)
      defineLabel(Value.LabelID);
    Contents.push_back(Value);
    CurrentSize += Value.size();
  }

  unsigned allocateLabelID() const { return CurrentLabelID++; }

  void defineLabel(unsigned LabelID) {
    LabelMap.insert(std::make_pair(LabelID, CurrentSize));
  }

  unsigned getLabelIndex(unsigned LabelID) const {
    const auto I = LabelMap.find(LabelID);
    assert(I != LabelMap.end() && "Use of undeclared label");
    return I->second;
  }

  void emitUse(raw_ostream &OS) const { OS << "MatchTable" << ID; }

  void emitDeclaration(raw_ostream &OS) const {
    unsigned Indentation = 4;
    OS << "  constexpr static int64_t MatchTable" << ID << "[] = {";
    LineBreak.emit(OS, true, *this);
    OS << std::string(Indentation, ' ');

    for (auto I = Contents.begin(), E = Contents.end(); I != E;
         ++I) {
      bool LineBreakIsNext = false;
      const auto &NextI = std::next(I);

      if (NextI != E) {
        if (NextI->EmitStr == "" &&
            NextI->Flags == MatchTableRecord::MTRF_LineBreakFollows)
          LineBreakIsNext = true;
      }

      if (I->Flags & MatchTableRecord::MTRF_Indent)
        Indentation += 2;

      I->emit(OS, LineBreakIsNext, *this);
      if (I->Flags & MatchTableRecord::MTRF_LineBreakFollows)
        OS << std::string(Indentation, ' ');

      if (I->Flags & MatchTableRecord::MTRF_Outdent)
        Indentation -= 2;
    }
    OS << "};\n";
  }
};

unsigned MatchTable::CurrentLabelID = 0;

MatchTableRecord MatchTable::LineBreak = {
    None, "" /* Emit String */, 0 /* Elements */,
    MatchTableRecord::MTRF_LineBreakFollows};

void MatchTableRecord::emit(raw_ostream &OS, bool LineBreakIsNextAfterThis,
                            const MatchTable &Table) const {
  bool UseLineComment =
      LineBreakIsNextAfterThis | (Flags & MTRF_LineBreakFollows);
  if (Flags & (MTRF_JumpTarget | MTRF_CommaFollows))
    UseLineComment = false;

  if (Flags & MTRF_Comment)
    OS << (UseLineComment ? "// " : "/*");

  OS << EmitStr;
  if (Flags & MTRF_Label)
    OS << ": @" << Table.getLabelIndex(LabelID);

  if (Flags & MTRF_Comment && !UseLineComment)
    OS << "*/";

  if (Flags & MTRF_JumpTarget) {
    if (Flags & MTRF_Comment)
      OS << " ";
    OS << Table.getLabelIndex(LabelID);
  }

  if (Flags & MTRF_CommaFollows) {
    OS << ",";
    if (!LineBreakIsNextAfterThis && !(Flags & MTRF_LineBreakFollows))
      OS << " ";
  }

  if (Flags & MTRF_LineBreakFollows)
    OS << "\n";
}

MatchTable &operator<<(MatchTable &Table, const MatchTableRecord &Value) {
  Table.push_back(Value);
  return Table;
}

//===- Matchers -----------------------------------------------------------===//

class OperandMatcher;
class MatchAction;

/// Generates code to check that a match rule matches.
class RuleMatcher {
  /// A list of matchers that all need to succeed for the current rule to match.
  /// FIXME: This currently supports a single match position but could be
  /// extended to support multiple positions to support div/rem fusion or
  /// load-multiple instructions.
  std::vector<std::unique_ptr<InstructionMatcher>> Matchers;

  /// A list of actions that need to be taken when all predicates in this rule
  /// have succeeded.
  std::vector<std::unique_ptr<MatchAction>> Actions;

  typedef std::map<const InstructionMatcher *, unsigned>
      DefinedInsnVariablesMap;
  /// A map of instruction matchers to the local variables created by
  /// emitCaptureOpcodes().
  DefinedInsnVariablesMap InsnVariableIDs;

  /// ID for the next instruction variable defined with defineInsnVar()
  unsigned NextInsnVarID;

  std::vector<Record *> RequiredFeatures;

public:
  RuleMatcher()
      : Matchers(), Actions(), InsnVariableIDs(), NextInsnVarID(0) {}
  RuleMatcher(RuleMatcher &&Other) = default;
  RuleMatcher &operator=(RuleMatcher &&Other) = default;

  InstructionMatcher &addInstructionMatcher(StringRef SymbolicName);
  void addRequiredFeature(Record *Feature);
  const std::vector<Record *> &getRequiredFeatures() const;

  template <class Kind, class... Args> Kind &addAction(Args &&... args);

  /// Define an instruction without emitting any code to do so.
  /// This is used for the root of the match.
  unsigned implicitlyDefineInsnVar(const InstructionMatcher &Matcher);
  /// Define an instruction and emit corresponding state-machine opcodes.
  unsigned defineInsnVar(MatchTable &Table, const InstructionMatcher &Matcher,
                         unsigned InsnVarID, unsigned OpIdx);
  unsigned getInsnVarID(const InstructionMatcher &InsnMatcher) const;
  DefinedInsnVariablesMap::const_iterator defined_insn_vars_begin() const {
    return InsnVariableIDs.begin();
  }
  DefinedInsnVariablesMap::const_iterator defined_insn_vars_end() const {
    return InsnVariableIDs.end();
  }
  iterator_range<typename DefinedInsnVariablesMap::const_iterator>
  defined_insn_vars() const {
    return make_range(defined_insn_vars_begin(), defined_insn_vars_end());
  }

  const InstructionMatcher &getInstructionMatcher(StringRef SymbolicName) const;

  void emitCaptureOpcodes(MatchTable &Table);

  void emit(MatchTable &Table);

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  bool isHigherPriorityThan(const RuleMatcher &B) const;

  /// Report the maximum number of temporary operands needed by the rule
  /// matcher.
  unsigned countRendererFns() const;

  // FIXME: Remove this as soon as possible
  InstructionMatcher &insnmatcher_front() const { return *Matchers.front(); }
};

template <class PredicateTy> class PredicateListMatcher {
private:
  typedef std::vector<std::unique_ptr<PredicateTy>> PredicateVec;
  PredicateVec Predicates;

  /// Template instantiations should specialize this to return a string to use
  /// for the comment emitted when there are no predicates.
  std::string getNoPredicateComment() const;

public:
  /// Construct a new operand predicate and add it to the matcher.
  template <class Kind, class... Args>
  Kind &addPredicate(Args&&... args) {
    Predicates.emplace_back(
        llvm::make_unique<Kind>(std::forward<Args>(args)...));
    return *static_cast<Kind *>(Predicates.back().get());
  }

  typename PredicateVec::const_iterator predicates_begin() const {
    return Predicates.begin();
  }
  typename PredicateVec::const_iterator predicates_end() const {
    return Predicates.end();
  }
  iterator_range<typename PredicateVec::const_iterator> predicates() const {
    return make_range(predicates_begin(), predicates_end());
  }
  typename PredicateVec::size_type predicates_size() const {
    return Predicates.size();
  }

  /// Emit MatchTable opcodes that tests whether all the predicates are met.
  template <class... Args>
  void emitPredicateListOpcodes(MatchTable &Table, Args &&... args) const {
    if (Predicates.empty()) {
      Table << MatchTable::Comment(getNoPredicateComment())
            << MatchTable::LineBreak;
      return;
    }

    for (const auto &Predicate : predicates())
      Predicate->emitPredicateOpcodes(Table, std::forward<Args>(args)...);
  }
};

/// Generates code to check a predicate of an operand.
///
/// Typical predicates include:
/// * Operand is a particular register.
/// * Operand is assigned a particular register bank.
/// * Operand is an MBB.
class OperandPredicateMatcher {
public:
  /// This enum is used for RTTI and also defines the priority that is given to
  /// the predicate when generating the matcher code. Kinds with higher priority
  /// must be tested first.
  ///
  /// The relative priority of OPM_LLT, OPM_RegBank, and OPM_MBB do not matter
  /// but OPM_Int must have priority over OPM_RegBank since constant integers
  /// are represented by a virtual register defined by a G_CONSTANT instruction.
  enum PredicateKind {
    OPM_ComplexPattern,
    OPM_IntrinsicID,
    OPM_Instruction,
    OPM_Int,
    OPM_LiteralInt,
    OPM_LLT,
    OPM_RegBank,
    OPM_MBB,
  };

protected:
  PredicateKind Kind;

public:
  OperandPredicateMatcher(PredicateKind Kind) : Kind(Kind) {}
  virtual ~OperandPredicateMatcher() {}

  PredicateKind getKind() const { return Kind; }

  /// Return the OperandMatcher for the specified operand or nullptr if there
  /// isn't one by that name in this operand predicate matcher.
  ///
  /// InstructionOperandMatcher is the only subclass that can return non-null
  /// for this.
  virtual Optional<const OperandMatcher *>
  getOptionalOperand(StringRef SymbolicName) const {
    assert(!SymbolicName.empty() && "Cannot lookup unnamed operand");
    return None;
  }

  /// Emit MatchTable opcodes to capture instructions into the MIs table.
  ///
  /// Only InstructionOperandMatcher needs to do anything for this method the
  /// rest just walk the tree.
  virtual void emitCaptureOpcodes(MatchTable &Table, RuleMatcher &Rule,
                                  unsigned InsnVarID, unsigned OpIdx) const {}

  /// Emit MatchTable opcodes that check the predicate for the given operand.
  virtual void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                                    unsigned InsnVarID,
                                    unsigned OpIdx) const = 0;

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  virtual bool isHigherPriorityThan(const OperandPredicateMatcher &B) const;

  /// Report the maximum number of temporary operands needed by the predicate
  /// matcher.
  virtual unsigned countRendererFns() const { return 0; }
};

template <>
std::string
PredicateListMatcher<OperandPredicateMatcher>::getNoPredicateComment() const {
  return "No operand predicates";
}

/// Generates code to check that an operand is a particular LLT.
class LLTOperandMatcher : public OperandPredicateMatcher {
protected:
  LLTCodeGen Ty;

public:
  static std::set<LLTCodeGen> KnownTypes;

  LLTOperandMatcher(const LLTCodeGen &Ty)
      : OperandPredicateMatcher(OPM_LLT), Ty(Ty) {
    KnownTypes.insert(Ty);
  }

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_LLT;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID, unsigned OpIdx) const override {
    Table << MatchTable::Opcode("GIM_CheckType") << MatchTable::Comment("MI")
          << MatchTable::IntValue(InsnVarID) << MatchTable::Comment("Op")
          << MatchTable::IntValue(OpIdx) << MatchTable::Comment("Type")
          << MatchTable::NamedValue(Ty.getCxxEnumValue())
          << MatchTable::LineBreak;
  }
};

std::set<LLTCodeGen> LLTOperandMatcher::KnownTypes;

/// Generates code to check that an operand is a particular target constant.
class ComplexPatternOperandMatcher : public OperandPredicateMatcher {
protected:
  const OperandMatcher &Operand;
  const Record &TheDef;

  unsigned getAllocatedTemporariesBaseID() const;

public:
  ComplexPatternOperandMatcher(const OperandMatcher &Operand,
                               const Record &TheDef)
      : OperandPredicateMatcher(OPM_ComplexPattern), Operand(Operand),
        TheDef(TheDef) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_ComplexPattern;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID, unsigned OpIdx) const override {
    unsigned ID = getAllocatedTemporariesBaseID();
    Table << MatchTable::Opcode("GIM_CheckComplexPattern")
          << MatchTable::Comment("MI") << MatchTable::IntValue(InsnVarID)
          << MatchTable::Comment("Op") << MatchTable::IntValue(OpIdx)
          << MatchTable::Comment("Renderer") << MatchTable::IntValue(ID)
          << MatchTable::NamedValue(("GICP_" + TheDef.getName()).str())
          << MatchTable::LineBreak;
  }

  unsigned countRendererFns() const override {
    return 1;
  }
};

/// Generates code to check that an operand is in a particular register bank.
class RegisterBankOperandMatcher : public OperandPredicateMatcher {
protected:
  const CodeGenRegisterClass &RC;

public:
  RegisterBankOperandMatcher(const CodeGenRegisterClass &RC)
      : OperandPredicateMatcher(OPM_RegBank), RC(RC) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_RegBank;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID, unsigned OpIdx) const override {
    Table << MatchTable::Opcode("GIM_CheckRegBankForClass")
          << MatchTable::Comment("MI") << MatchTable::IntValue(InsnVarID)
          << MatchTable::Comment("Op") << MatchTable::IntValue(OpIdx)
          << MatchTable::Comment("RC")
          << MatchTable::NamedValue(RC.getQualifiedName() + "RegClassID")
          << MatchTable::LineBreak;
  }
};

/// Generates code to check that an operand is a basic block.
class MBBOperandMatcher : public OperandPredicateMatcher {
public:
  MBBOperandMatcher() : OperandPredicateMatcher(OPM_MBB) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_MBB;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID, unsigned OpIdx) const override {
    Table << MatchTable::Opcode("GIM_CheckIsMBB") << MatchTable::Comment("MI")
          << MatchTable::IntValue(InsnVarID) << MatchTable::Comment("Op")
          << MatchTable::IntValue(OpIdx) << MatchTable::LineBreak;
  }
};

/// Generates code to check that an operand is a G_CONSTANT with a particular
/// int.
class ConstantIntOperandMatcher : public OperandPredicateMatcher {
protected:
  int64_t Value;

public:
  ConstantIntOperandMatcher(int64_t Value)
      : OperandPredicateMatcher(OPM_Int), Value(Value) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_Int;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID, unsigned OpIdx) const override {
    Table << MatchTable::Opcode("GIM_CheckConstantInt")
          << MatchTable::Comment("MI") << MatchTable::IntValue(InsnVarID)
          << MatchTable::Comment("Op") << MatchTable::IntValue(OpIdx)
          << MatchTable::IntValue(Value) << MatchTable::LineBreak;
  }
};

/// Generates code to check that an operand is a raw int (where MO.isImm() or
/// MO.isCImm() is true).
class LiteralIntOperandMatcher : public OperandPredicateMatcher {
protected:
  int64_t Value;

public:
  LiteralIntOperandMatcher(int64_t Value)
      : OperandPredicateMatcher(OPM_LiteralInt), Value(Value) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_LiteralInt;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID, unsigned OpIdx) const override {
    Table << MatchTable::Opcode("GIM_CheckLiteralInt")
          << MatchTable::Comment("MI") << MatchTable::IntValue(InsnVarID)
          << MatchTable::Comment("Op") << MatchTable::IntValue(OpIdx)
          << MatchTable::IntValue(Value) << MatchTable::LineBreak;
  }
};

/// Generates code to check that an operand is an intrinsic ID.
class IntrinsicIDOperandMatcher : public OperandPredicateMatcher {
protected:
  const CodeGenIntrinsic *II;

public:
  IntrinsicIDOperandMatcher(const CodeGenIntrinsic *II)
      : OperandPredicateMatcher(OPM_IntrinsicID), II(II) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_IntrinsicID;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID, unsigned OpIdx) const override {
    Table << MatchTable::Opcode("GIM_CheckIntrinsicID")
          << MatchTable::Comment("MI") << MatchTable::IntValue(InsnVarID)
          << MatchTable::Comment("Op") << MatchTable::IntValue(OpIdx)
          << MatchTable::NamedValue("Intrinsic::" + II->EnumName)
          << MatchTable::LineBreak;
  }
};

/// Generates code to check that a set of predicates match for a particular
/// operand.
class OperandMatcher : public PredicateListMatcher<OperandPredicateMatcher> {
protected:
  InstructionMatcher &Insn;
  unsigned OpIdx;
  std::string SymbolicName;

  /// The index of the first temporary variable allocated to this operand. The
  /// number of allocated temporaries can be found with
  /// countRendererFns().
  unsigned AllocatedTemporariesBaseID;

public:
  OperandMatcher(InstructionMatcher &Insn, unsigned OpIdx,
                 const std::string &SymbolicName,
                 unsigned AllocatedTemporariesBaseID)
      : Insn(Insn), OpIdx(OpIdx), SymbolicName(SymbolicName),
        AllocatedTemporariesBaseID(AllocatedTemporariesBaseID) {}

  bool hasSymbolicName() const { return !SymbolicName.empty(); }
  const StringRef getSymbolicName() const { return SymbolicName; }
  void setSymbolicName(StringRef Name) {
    assert(SymbolicName.empty() && "Operand already has a symbolic name");
    SymbolicName = Name;
  }
  unsigned getOperandIndex() const { return OpIdx; }

  std::string getOperandExpr(unsigned InsnVarID) const {
    return "State.MIs[" + llvm::to_string(InsnVarID) + "]->getOperand(" +
           llvm::to_string(OpIdx) + ")";
  }

  Optional<const OperandMatcher *>
  getOptionalOperand(StringRef DesiredSymbolicName) const {
    assert(!DesiredSymbolicName.empty() && "Cannot lookup unnamed operand");
    if (DesiredSymbolicName == SymbolicName)
      return this;
    for (const auto &OP : predicates()) {
      const auto &MaybeOperand = OP->getOptionalOperand(DesiredSymbolicName);
      if (MaybeOperand.hasValue())
        return MaybeOperand.getValue();
    }
    return None;
  }

  InstructionMatcher &getInstructionMatcher() const { return Insn; }

  /// Emit MatchTable opcodes to capture instructions into the MIs table.
  void emitCaptureOpcodes(MatchTable &Table, RuleMatcher &Rule,
                          unsigned InsnVarID) const {
    for (const auto &Predicate : predicates())
      Predicate->emitCaptureOpcodes(Table, Rule, InsnVarID, OpIdx);
  }

  /// Emit MatchTable opcodes that test whether the instruction named in
  /// InsnVarID matches all the predicates and all the operands.
  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID) const {
    std::string Comment;
    raw_string_ostream CommentOS(Comment);
    CommentOS << "MIs[" << InsnVarID << "] ";
    if (SymbolicName.empty())
      CommentOS << "Operand " << OpIdx;
    else
      CommentOS << SymbolicName;
    Table << MatchTable::Comment(CommentOS.str()) << MatchTable::LineBreak;

    emitPredicateListOpcodes(Table, Rule, InsnVarID, OpIdx);
  }

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  bool isHigherPriorityThan(const OperandMatcher &B) const {
    // Operand matchers involving more predicates have higher priority.
    if (predicates_size() > B.predicates_size())
      return true;
    if (predicates_size() < B.predicates_size())
      return false;

    // This assumes that predicates are added in a consistent order.
    for (const auto &Predicate : zip(predicates(), B.predicates())) {
      if (std::get<0>(Predicate)->isHigherPriorityThan(*std::get<1>(Predicate)))
        return true;
      if (std::get<1>(Predicate)->isHigherPriorityThan(*std::get<0>(Predicate)))
        return false;
    }

    return false;
  };

  /// Report the maximum number of temporary operands needed by the operand
  /// matcher.
  unsigned countRendererFns() const {
    return std::accumulate(
        predicates().begin(), predicates().end(), 0,
        [](unsigned A,
           const std::unique_ptr<OperandPredicateMatcher> &Predicate) {
          return A + Predicate->countRendererFns();
        });
  }

  unsigned getAllocatedTemporariesBaseID() const {
    return AllocatedTemporariesBaseID;
  }
};

unsigned ComplexPatternOperandMatcher::getAllocatedTemporariesBaseID() const {
  return Operand.getAllocatedTemporariesBaseID();
}

/// Generates code to check a predicate on an instruction.
///
/// Typical predicates include:
/// * The opcode of the instruction is a particular value.
/// * The nsw/nuw flag is/isn't set.
class InstructionPredicateMatcher {
protected:
  /// This enum is used for RTTI and also defines the priority that is given to
  /// the predicate when generating the matcher code. Kinds with higher priority
  /// must be tested first.
  enum PredicateKind {
    IPM_Opcode,
    IPM_ImmPredicate,
  };

  PredicateKind Kind;

public:
  InstructionPredicateMatcher(PredicateKind Kind) : Kind(Kind) {}
  virtual ~InstructionPredicateMatcher() {}

  PredicateKind getKind() const { return Kind; }

  /// Emit MatchTable opcodes that test whether the instruction named in
  /// InsnVarID matches the predicate.
  virtual void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                                    unsigned InsnVarID) const = 0;

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  virtual bool
  isHigherPriorityThan(const InstructionPredicateMatcher &B) const {
    return Kind < B.Kind;
  };

  /// Report the maximum number of temporary operands needed by the predicate
  /// matcher.
  virtual unsigned countRendererFns() const { return 0; }
};

template <>
std::string
PredicateListMatcher<InstructionPredicateMatcher>::getNoPredicateComment() const {
  return "No instruction predicates";
}

/// Generates code to check the opcode of an instruction.
class InstructionOpcodeMatcher : public InstructionPredicateMatcher {
protected:
  const CodeGenInstruction *I;

public:
  InstructionOpcodeMatcher(const CodeGenInstruction *I)
      : InstructionPredicateMatcher(IPM_Opcode), I(I) {}

  static bool classof(const InstructionPredicateMatcher *P) {
    return P->getKind() == IPM_Opcode;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID) const override {
    Table << MatchTable::Opcode("GIM_CheckOpcode") << MatchTable::Comment("MI")
          << MatchTable::IntValue(InsnVarID)
          << MatchTable::NamedValue(I->Namespace, I->TheDef->getName())
          << MatchTable::LineBreak;
  }

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  bool
  isHigherPriorityThan(const InstructionPredicateMatcher &B) const override {
    if (InstructionPredicateMatcher::isHigherPriorityThan(B))
      return true;
    if (B.InstructionPredicateMatcher::isHigherPriorityThan(*this))
      return false;

    // Prioritize opcodes for cosmetic reasons in the generated source. Although
    // this is cosmetic at the moment, we may want to drive a similar ordering
    // using instruction frequency information to improve compile time.
    if (const InstructionOpcodeMatcher *BO =
            dyn_cast<InstructionOpcodeMatcher>(&B))
      return I->TheDef->getName() < BO->I->TheDef->getName();

    return false;
  };

  bool isConstantInstruction() const {
    return I->TheDef->getName() == "G_CONSTANT";
  }
};

/// Generates code to check that this instruction is a constant whose value
/// meets an immediate predicate.
///
/// Immediates are slightly odd since they are typically used like an operand
/// but are represented as an operator internally. We typically write simm8:$src
/// in a tablegen pattern, but this is just syntactic sugar for
/// (imm:i32)<<P:Predicate_simm8>>:$imm which more directly describes the nodes
/// that will be matched and the predicate (which is attached to the imm
/// operator) that will be tested. In SelectionDAG this describes a
/// ConstantSDNode whose internal value will be tested using the simm8 predicate.
///
/// The corresponding GlobalISel representation is %1 = G_CONSTANT iN Value. In
/// this representation, the immediate could be tested with an
/// InstructionMatcher, InstructionOpcodeMatcher, OperandMatcher, and a
/// OperandPredicateMatcher-subclass to check the Value meets the predicate but
/// there are two implementation issues with producing that matcher
/// configuration from the SelectionDAG pattern:
/// * ImmLeaf is a PatFrag whose root is an InstructionMatcher. This means that
///   were we to sink the immediate predicate to the operand we would have to
///   have two partial implementations of PatFrag support, one for immediates
///   and one for non-immediates.
/// * At the point we handle the predicate, the OperandMatcher hasn't been
///   created yet. If we were to sink the predicate to the OperandMatcher we
///   would also have to complicate (or duplicate) the code that descends and
///   creates matchers for the subtree.
/// Overall, it's simpler to handle it in the place it was found.
class InstructionImmPredicateMatcher : public InstructionPredicateMatcher {
protected:
  TreePredicateFn Predicate;

public:
  InstructionImmPredicateMatcher(const TreePredicateFn &Predicate)
      : InstructionPredicateMatcher(IPM_ImmPredicate), Predicate(Predicate) {}

  static bool classof(const InstructionPredicateMatcher *P) {
    return P->getKind() == IPM_ImmPredicate;
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID) const override {
    Table << MatchTable::Opcode("GIM_CheckImmPredicate")
          << MatchTable::Comment("MI") << MatchTable::IntValue(InsnVarID)
          << MatchTable::Comment("Predicate")
          << MatchTable::NamedValue("GIPFP_" + Predicate.getFnName())
          << MatchTable::LineBreak;
  }
};

/// Generates code to check that a set of predicates and operands match for a
/// particular instruction.
///
/// Typical predicates include:
/// * Has a specific opcode.
/// * Has an nsw/nuw flag or doesn't.
class InstructionMatcher
    : public PredicateListMatcher<InstructionPredicateMatcher> {
protected:
  typedef std::vector<std::unique_ptr<OperandMatcher>> OperandVec;

  /// The operands to match. All rendered operands must be present even if the
  /// condition is always true.
  OperandVec Operands;

  std::string SymbolicName;

public:
  InstructionMatcher(StringRef SymbolicName) : SymbolicName(SymbolicName) {}

  /// Add an operand to the matcher.
  OperandMatcher &addOperand(unsigned OpIdx, const std::string &SymbolicName,
                             unsigned AllocatedTemporariesBaseID) {
    Operands.emplace_back(new OperandMatcher(*this, OpIdx, SymbolicName,
                                             AllocatedTemporariesBaseID));
    return *Operands.back();
  }

  OperandMatcher &getOperand(unsigned OpIdx) {
    auto I = std::find_if(Operands.begin(), Operands.end(),
                          [&OpIdx](const std::unique_ptr<OperandMatcher> &X) {
                            return X->getOperandIndex() == OpIdx;
                          });
    if (I != Operands.end())
      return **I;
    llvm_unreachable("Failed to lookup operand");
  }

  Optional<const OperandMatcher *>
  getOptionalOperand(StringRef SymbolicName) const {
    assert(!SymbolicName.empty() && "Cannot lookup unnamed operand");
    for (const auto &Operand : Operands) {
      const auto &OM = Operand->getOptionalOperand(SymbolicName);
      if (OM.hasValue())
        return OM.getValue();
    }
    return None;
  }

  const OperandMatcher &getOperand(StringRef SymbolicName) const {
    Optional<const OperandMatcher *>OM = getOptionalOperand(SymbolicName);
    if (OM.hasValue())
      return *OM.getValue();
    llvm_unreachable("Failed to lookup operand");
  }

  StringRef getSymbolicName() const { return SymbolicName; }
  unsigned getNumOperands() const { return Operands.size(); }
  OperandVec::iterator operands_begin() { return Operands.begin(); }
  OperandVec::iterator operands_end() { return Operands.end(); }
  iterator_range<OperandVec::iterator> operands() {
    return make_range(operands_begin(), operands_end());
  }
  OperandVec::const_iterator operands_begin() const { return Operands.begin(); }
  OperandVec::const_iterator operands_end() const { return Operands.end(); }
  iterator_range<OperandVec::const_iterator> operands() const {
    return make_range(operands_begin(), operands_end());
  }

  /// Emit MatchTable opcodes to check the shape of the match and capture
  /// instructions into the MIs table.
  void emitCaptureOpcodes(MatchTable &Table, RuleMatcher &Rule,
                          unsigned InsnID) {
    Table << MatchTable::Opcode("GIM_CheckNumOperands")
          << MatchTable::Comment("MI") << MatchTable::IntValue(InsnID)
          << MatchTable::Comment("Expected")
          << MatchTable::IntValue(getNumOperands()) << MatchTable::LineBreak;
    for (const auto &Operand : Operands)
      Operand->emitCaptureOpcodes(Table, Rule, InsnID);
  }

  /// Emit MatchTable opcodes that test whether the instruction named in
  /// InsnVarName matches all the predicates and all the operands.
  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID) const {
    emitPredicateListOpcodes(Table, Rule, InsnVarID);
    for (const auto &Operand : Operands)
      Operand->emitPredicateOpcodes(Table, Rule, InsnVarID);
  }

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  bool isHigherPriorityThan(const InstructionMatcher &B) const {
    // Instruction matchers involving more operands have higher priority.
    if (Operands.size() > B.Operands.size())
      return true;
    if (Operands.size() < B.Operands.size())
      return false;

    for (const auto &Predicate : zip(predicates(), B.predicates())) {
      if (std::get<0>(Predicate)->isHigherPriorityThan(*std::get<1>(Predicate)))
        return true;
      if (std::get<1>(Predicate)->isHigherPriorityThan(*std::get<0>(Predicate)))
        return false;
    }

    for (const auto &Operand : zip(Operands, B.Operands)) {
      if (std::get<0>(Operand)->isHigherPriorityThan(*std::get<1>(Operand)))
        return true;
      if (std::get<1>(Operand)->isHigherPriorityThan(*std::get<0>(Operand)))
        return false;
    }

    return false;
  };

  /// Report the maximum number of temporary operands needed by the instruction
  /// matcher.
  unsigned countRendererFns() const {
    return std::accumulate(predicates().begin(), predicates().end(), 0,
                           [](unsigned A,
                              const std::unique_ptr<InstructionPredicateMatcher>
                                  &Predicate) {
                             return A + Predicate->countRendererFns();
                           }) +
           std::accumulate(
               Operands.begin(), Operands.end(), 0,
               [](unsigned A, const std::unique_ptr<OperandMatcher> &Operand) {
                 return A + Operand->countRendererFns();
               });
  }

  bool isConstantInstruction() const {
    for (const auto &P : predicates())
      if (const InstructionOpcodeMatcher *Opcode =
              dyn_cast<InstructionOpcodeMatcher>(P.get()))
        return Opcode->isConstantInstruction();
    return false;
  }
};

/// Generates code to check that the operand is a register defined by an
/// instruction that matches the given instruction matcher.
///
/// For example, the pattern:
///   (set $dst, (G_MUL (G_ADD $src1, $src2), $src3))
/// would use an InstructionOperandMatcher for operand 1 of the G_MUL to match
/// the:
///   (G_ADD $src1, $src2)
/// subpattern.
class InstructionOperandMatcher : public OperandPredicateMatcher {
protected:
  std::unique_ptr<InstructionMatcher> InsnMatcher;

public:
  InstructionOperandMatcher(StringRef SymbolicName)
      : OperandPredicateMatcher(OPM_Instruction),
        InsnMatcher(new InstructionMatcher(SymbolicName)) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_Instruction;
  }

  InstructionMatcher &getInsnMatcher() const { return *InsnMatcher; }

  Optional<const OperandMatcher *>
  getOptionalOperand(StringRef SymbolicName) const override {
    assert(!SymbolicName.empty() && "Cannot lookup unnamed operand");
    return InsnMatcher->getOptionalOperand(SymbolicName);
  }

  void emitCaptureOpcodes(MatchTable &Table, RuleMatcher &Rule,
                          unsigned InsnID, unsigned OpIdx) const override {
    unsigned InsnVarID = Rule.defineInsnVar(Table, *InsnMatcher, InsnID, OpIdx);
    InsnMatcher->emitCaptureOpcodes(Table, Rule, InsnVarID);
  }

  void emitPredicateOpcodes(MatchTable &Table, RuleMatcher &Rule,
                            unsigned InsnVarID_,
                            unsigned OpIdx_) const override {
    unsigned InsnVarID = Rule.getInsnVarID(*InsnMatcher);
    InsnMatcher->emitPredicateOpcodes(Table, Rule, InsnVarID);
  }
};

//===- Actions ------------------------------------------------------------===//
class OperandRenderer {
public:
  enum RendererKind {
    OR_Copy,
    OR_CopySubReg,
    OR_CopyConstantAsImm,
    OR_Imm,
    OR_Register,
    OR_ComplexPattern
  };

protected:
  RendererKind Kind;

public:
  OperandRenderer(RendererKind Kind) : Kind(Kind) {}
  virtual ~OperandRenderer() {}

  RendererKind getKind() const { return Kind; }

  virtual void emitRenderOpcodes(MatchTable &Table,
                                 RuleMatcher &Rule) const = 0;
};

/// A CopyRenderer emits code to copy a single operand from an existing
/// instruction to the one being built.
class CopyRenderer : public OperandRenderer {
protected:
  unsigned NewInsnID;
  /// The matcher for the instruction that this operand is copied from.
  /// This provides the facility for looking up an a operand by it's name so
  /// that it can be used as a source for the instruction being built.
  const InstructionMatcher &Matched;
  /// The name of the operand.
  const StringRef SymbolicName;

public:
  CopyRenderer(unsigned NewInsnID, const InstructionMatcher &Matched,
               StringRef SymbolicName)
      : OperandRenderer(OR_Copy), NewInsnID(NewInsnID), Matched(Matched),
        SymbolicName(SymbolicName) {
    assert(!SymbolicName.empty() && "Cannot copy from an unspecified source");
  }

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_Copy;
  }

  const StringRef getSymbolicName() const { return SymbolicName; }

  void emitRenderOpcodes(MatchTable &Table, RuleMatcher &Rule) const override {
    const OperandMatcher &Operand = Matched.getOperand(SymbolicName);
    unsigned OldInsnVarID = Rule.getInsnVarID(Operand.getInstructionMatcher());
    Table << MatchTable::Opcode("GIR_Copy") << MatchTable::Comment("NewInsnID")
          << MatchTable::IntValue(NewInsnID) << MatchTable::Comment("OldInsnID")
          << MatchTable::IntValue(OldInsnVarID) << MatchTable::Comment("OpIdx")
          << MatchTable::IntValue(Operand.getOperandIndex())
          << MatchTable::Comment(SymbolicName) << MatchTable::LineBreak;
  }
};

/// A CopyConstantAsImmRenderer emits code to render a G_CONSTANT instruction to
/// an extended immediate operand.
class CopyConstantAsImmRenderer : public OperandRenderer {
protected:
  unsigned NewInsnID;
  /// The name of the operand.
  const std::string SymbolicName;
  bool Signed;

public:
  CopyConstantAsImmRenderer(unsigned NewInsnID, StringRef SymbolicName)
      : OperandRenderer(OR_CopyConstantAsImm), NewInsnID(NewInsnID),
        SymbolicName(SymbolicName), Signed(true) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_CopyConstantAsImm;
  }

  const StringRef getSymbolicName() const { return SymbolicName; }

  void emitRenderOpcodes(MatchTable &Table, RuleMatcher &Rule) const override {
    const InstructionMatcher &InsnMatcher = Rule.getInstructionMatcher(SymbolicName);
    unsigned OldInsnVarID = Rule.getInsnVarID(InsnMatcher);
    Table << MatchTable::Opcode(Signed ? "GIR_CopyConstantAsSImm"
                                       : "GIR_CopyConstantAsUImm")
          << MatchTable::Comment("NewInsnID") << MatchTable::IntValue(NewInsnID)
          << MatchTable::Comment("OldInsnID")
          << MatchTable::IntValue(OldInsnVarID)
          << MatchTable::Comment(SymbolicName) << MatchTable::LineBreak;
  }
};

/// A CopySubRegRenderer emits code to copy a single register operand from an
/// existing instruction to the one being built and indicate that only a
/// subregister should be copied.
class CopySubRegRenderer : public OperandRenderer {
protected:
  unsigned NewInsnID;
  /// The matcher for the instruction that this operand is copied from.
  /// This provides the facility for looking up an a operand by it's name so
  /// that it can be used as a source for the instruction being built.
  const InstructionMatcher &Matched;
  /// The name of the operand.
  const StringRef SymbolicName;
  /// The subregister to extract.
  const CodeGenSubRegIndex *SubReg;

public:
  CopySubRegRenderer(unsigned NewInsnID, const InstructionMatcher &Matched,
                     StringRef SymbolicName, const CodeGenSubRegIndex *SubReg)
      : OperandRenderer(OR_CopySubReg), NewInsnID(NewInsnID), Matched(Matched),
        SymbolicName(SymbolicName), SubReg(SubReg) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_CopySubReg;
  }

  const StringRef getSymbolicName() const { return SymbolicName; }

  void emitRenderOpcodes(MatchTable &Table, RuleMatcher &Rule) const override {
    const OperandMatcher &Operand = Matched.getOperand(SymbolicName);
    unsigned OldInsnVarID = Rule.getInsnVarID(Operand.getInstructionMatcher());
    Table << MatchTable::Opcode("GIR_CopySubReg")
          << MatchTable::Comment("NewInsnID") << MatchTable::IntValue(NewInsnID)
          << MatchTable::Comment("OldInsnID")
          << MatchTable::IntValue(OldInsnVarID) << MatchTable::Comment("OpIdx")
          << MatchTable::IntValue(Operand.getOperandIndex())
          << MatchTable::Comment("SubRegIdx")
          << MatchTable::IntValue(SubReg->EnumValue)
          << MatchTable::Comment(SymbolicName) << MatchTable::LineBreak;
  }
};

/// Adds a specific physical register to the instruction being built.
/// This is typically useful for WZR/XZR on AArch64.
class AddRegisterRenderer : public OperandRenderer {
protected:
  unsigned InsnID;
  const Record *RegisterDef;

public:
  AddRegisterRenderer(unsigned InsnID, const Record *RegisterDef)
      : OperandRenderer(OR_Register), InsnID(InsnID), RegisterDef(RegisterDef) {
  }

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_Register;
  }

  void emitRenderOpcodes(MatchTable &Table, RuleMatcher &Rule) const override {
    Table << MatchTable::Opcode("GIR_AddRegister")
          << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
          << MatchTable::NamedValue(
                 (RegisterDef->getValue("Namespace")
                      ? RegisterDef->getValueAsString("Namespace")
                      : ""),
                 RegisterDef->getName())
          << MatchTable::LineBreak;
  }
};

/// Adds a specific immediate to the instruction being built.
class ImmRenderer : public OperandRenderer {
protected:
  unsigned InsnID;
  int64_t Imm;

public:
  ImmRenderer(unsigned InsnID, int64_t Imm)
      : OperandRenderer(OR_Imm), InsnID(InsnID), Imm(Imm) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_Imm;
  }

  void emitRenderOpcodes(MatchTable &Table, RuleMatcher &Rule) const override {
    Table << MatchTable::Opcode("GIR_AddImm") << MatchTable::Comment("InsnID")
          << MatchTable::IntValue(InsnID) << MatchTable::Comment("Imm")
          << MatchTable::IntValue(Imm) << MatchTable::LineBreak;
  }
};

/// Adds operands by calling a renderer function supplied by the ComplexPattern
/// matcher function.
class RenderComplexPatternOperand : public OperandRenderer {
private:
  unsigned InsnID;
  const Record &TheDef;
  /// The name of the operand.
  const StringRef SymbolicName;
  /// The renderer number. This must be unique within a rule since it's used to
  /// identify a temporary variable to hold the renderer function.
  unsigned RendererID;

  unsigned getNumOperands() const {
    return TheDef.getValueAsDag("Operands")->getNumArgs();
  }

public:
  RenderComplexPatternOperand(unsigned InsnID, const Record &TheDef,
                              StringRef SymbolicName, unsigned RendererID)
      : OperandRenderer(OR_ComplexPattern), InsnID(InsnID), TheDef(TheDef),
        SymbolicName(SymbolicName), RendererID(RendererID) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_ComplexPattern;
  }

  void emitRenderOpcodes(MatchTable &Table, RuleMatcher &Rule) const override {
    Table << MatchTable::Opcode("GIR_ComplexRenderer")
          << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
          << MatchTable::Comment("RendererID")
          << MatchTable::IntValue(RendererID) << MatchTable::LineBreak;
  }
};

/// An action taken when all Matcher predicates succeeded for a parent rule.
///
/// Typical actions include:
/// * Changing the opcode of an instruction.
/// * Adding an operand to an instruction.
class MatchAction {
public:
  virtual ~MatchAction() {}

  /// Emit the MatchTable opcodes to implement the action.
  ///
  /// \param RecycleInsnID If given, it's an instruction to recycle. The
  ///                      requirements on the instruction vary from action to
  ///                      action.
  virtual void emitActionOpcodes(MatchTable &Table, RuleMatcher &Rule,
                                 unsigned RecycleInsnID) const = 0;
};

/// Generates a comment describing the matched rule being acted upon.
class DebugCommentAction : public MatchAction {
private:
  const PatternToMatch &P;

public:
  DebugCommentAction(const PatternToMatch &P) : P(P) {}

  void emitActionOpcodes(MatchTable &Table, RuleMatcher &Rule,
                         unsigned RecycleInsnID) const override {
    Table << MatchTable::Comment(llvm::to_string(*P.getSrcPattern()) + "  =>  " +
                               llvm::to_string(*P.getDstPattern()))
          << MatchTable::LineBreak;
  }
};

/// Generates code to build an instruction or mutate an existing instruction
/// into the desired instruction when this is possible.
class BuildMIAction : public MatchAction {
private:
  unsigned InsnID;
  const CodeGenInstruction *I;
  const InstructionMatcher &Matched;
  std::vector<std::unique_ptr<OperandRenderer>> OperandRenderers;

  /// True if the instruction can be built solely by mutating the opcode.
  bool canMutate() const {
    if (OperandRenderers.size() != Matched.getNumOperands())
      return false;

    for (const auto &Renderer : enumerate(OperandRenderers)) {
      if (const auto *Copy = dyn_cast<CopyRenderer>(&*Renderer.value())) {
        const OperandMatcher &OM = Matched.getOperand(Copy->getSymbolicName());
        if (&Matched != &OM.getInstructionMatcher() ||
            OM.getOperandIndex() != Renderer.index())
          return false;
      } else
        return false;
    }

    return true;
  }

public:
  BuildMIAction(unsigned InsnID, const CodeGenInstruction *I,
                const InstructionMatcher &Matched)
      : InsnID(InsnID), I(I), Matched(Matched) {}

  template <class Kind, class... Args>
  Kind &addRenderer(Args&&... args) {
    OperandRenderers.emplace_back(
        llvm::make_unique<Kind>(std::forward<Args>(args)...));
    return *static_cast<Kind *>(OperandRenderers.back().get());
  }

  void emitActionOpcodes(MatchTable &Table, RuleMatcher &Rule,
                         unsigned RecycleInsnID) const override {
    if (canMutate()) {
      Table << MatchTable::Opcode("GIR_MutateOpcode")
            << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
            << MatchTable::Comment("RecycleInsnID")
            << MatchTable::IntValue(RecycleInsnID)
            << MatchTable::Comment("Opcode")
            << MatchTable::NamedValue(I->Namespace, I->TheDef->getName())
            << MatchTable::LineBreak;

      if (!I->ImplicitDefs.empty() || !I->ImplicitUses.empty()) {
        for (auto Def : I->ImplicitDefs) {
          auto Namespace = Def->getValue("Namespace")
                               ? Def->getValueAsString("Namespace")
                               : "";
          Table << MatchTable::Opcode("GIR_AddImplicitDef")
                << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
                << MatchTable::NamedValue(Namespace, Def->getName())
                << MatchTable::LineBreak;
        }
        for (auto Use : I->ImplicitUses) {
          auto Namespace = Use->getValue("Namespace")
                               ? Use->getValueAsString("Namespace")
                               : "";
          Table << MatchTable::Opcode("GIR_AddImplicitUse")
                << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
                << MatchTable::NamedValue(Namespace, Use->getName())
                << MatchTable::LineBreak;
        }
      }
      return;
    }

    // TODO: Simple permutation looks like it could be almost as common as
    //       mutation due to commutative operations.

    Table << MatchTable::Opcode("GIR_BuildMI") << MatchTable::Comment("InsnID")
          << MatchTable::IntValue(InsnID) << MatchTable::Comment("Opcode")
          << MatchTable::NamedValue(I->Namespace, I->TheDef->getName())
          << MatchTable::LineBreak;
    for (const auto &Renderer : OperandRenderers)
      Renderer->emitRenderOpcodes(Table, Rule);

    if (I->mayLoad || I->mayStore) {
      Table << MatchTable::Opcode("GIR_MergeMemOperands")
            << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
            << MatchTable::Comment("MergeInsnID's");
      // Emit the ID's for all the instructions that are matched by this rule.
      // TODO: Limit this to matched instructions that mayLoad/mayStore or have
      //       some other means of having a memoperand. Also limit this to
      //       emitted instructions that expect to have a memoperand too. For
      //       example, (G_SEXT (G_LOAD x)) that results in separate load and
      //       sign-extend instructions shouldn't put the memoperand on the
      //       sign-extend since it has no effect there.
      std::vector<unsigned> MergeInsnIDs;
      for (const auto &IDMatcherPair : Rule.defined_insn_vars())
        MergeInsnIDs.push_back(IDMatcherPair.second);
      std::sort(MergeInsnIDs.begin(), MergeInsnIDs.end());
      for (const auto &MergeInsnID : MergeInsnIDs)
        Table << MatchTable::IntValue(MergeInsnID);
      Table << MatchTable::NamedValue("GIU_MergeMemOperands_EndOfList")
            << MatchTable::LineBreak;
    }

    Table << MatchTable::Opcode("GIR_EraseFromParent")
          << MatchTable::Comment("InsnID")
          << MatchTable::IntValue(RecycleInsnID) << MatchTable::LineBreak;
  }
};

/// Generates code to constrain the operands of an output instruction to the
/// register classes specified by the definition of that instruction.
class ConstrainOperandsToDefinitionAction : public MatchAction {
  unsigned InsnID;

public:
  ConstrainOperandsToDefinitionAction(unsigned InsnID) : InsnID(InsnID) {}

  void emitActionOpcodes(MatchTable &Table, RuleMatcher &Rule,
                         unsigned RecycleInsnID) const override {
    Table << MatchTable::Opcode("GIR_ConstrainSelectedInstOperands")
          << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
          << MatchTable::LineBreak;
  }
};

/// Generates code to constrain the specified operand of an output instruction
/// to the specified register class.
class ConstrainOperandToRegClassAction : public MatchAction {
  unsigned InsnID;
  unsigned OpIdx;
  const CodeGenRegisterClass &RC;

public:
  ConstrainOperandToRegClassAction(unsigned InsnID, unsigned OpIdx,
                                   const CodeGenRegisterClass &RC)
      : InsnID(InsnID), OpIdx(OpIdx), RC(RC) {}

  void emitActionOpcodes(MatchTable &Table, RuleMatcher &Rule,
                         unsigned RecycleInsnID) const override {
    Table << MatchTable::Opcode("GIR_ConstrainOperandRC")
          << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
          << MatchTable::Comment("Op") << MatchTable::IntValue(OpIdx)
          << MatchTable::Comment("RC " + RC.getName())
          << MatchTable::IntValue(RC.EnumValue) << MatchTable::LineBreak;
  }
};

InstructionMatcher &RuleMatcher::addInstructionMatcher(StringRef SymbolicName) {
  Matchers.emplace_back(new InstructionMatcher(SymbolicName));
  return *Matchers.back();
}

void RuleMatcher::addRequiredFeature(Record *Feature) {
  RequiredFeatures.push_back(Feature);
}

const std::vector<Record *> &RuleMatcher::getRequiredFeatures() const {
  return RequiredFeatures;
}

template <class Kind, class... Args>
Kind &RuleMatcher::addAction(Args &&... args) {
  Actions.emplace_back(llvm::make_unique<Kind>(std::forward<Args>(args)...));
  return *static_cast<Kind *>(Actions.back().get());
}

unsigned
RuleMatcher::implicitlyDefineInsnVar(const InstructionMatcher &Matcher) {
  unsigned NewInsnVarID = NextInsnVarID++;
  InsnVariableIDs[&Matcher] = NewInsnVarID;
  return NewInsnVarID;
}

unsigned RuleMatcher::defineInsnVar(MatchTable &Table,
                                    const InstructionMatcher &Matcher,
                                    unsigned InsnID, unsigned OpIdx) {
  unsigned NewInsnVarID = implicitlyDefineInsnVar(Matcher);
  Table << MatchTable::Opcode("GIM_RecordInsn")
        << MatchTable::Comment("DefineMI") << MatchTable::IntValue(NewInsnVarID)
        << MatchTable::Comment("MI") << MatchTable::IntValue(InsnID)
        << MatchTable::Comment("OpIdx") << MatchTable::IntValue(OpIdx)
        << MatchTable::Comment("MIs[" + llvm::to_string(NewInsnVarID) + "]")
        << MatchTable::LineBreak;
  return NewInsnVarID;
}

unsigned RuleMatcher::getInsnVarID(const InstructionMatcher &InsnMatcher) const {
  const auto &I = InsnVariableIDs.find(&InsnMatcher);
  if (I != InsnVariableIDs.end())
    return I->second;
  llvm_unreachable("Matched Insn was not captured in a local variable");
}

const InstructionMatcher &
RuleMatcher::getInstructionMatcher(StringRef SymbolicName) const {
  for (const auto &I : InsnVariableIDs)
    if (I.first->getSymbolicName() == SymbolicName)
      return *I.first;
  llvm_unreachable(
      ("Failed to lookup instruction " + SymbolicName).str().c_str());
}

/// Emit MatchTable opcodes to check the shape of the match and capture
/// instructions into local variables.
void RuleMatcher::emitCaptureOpcodes(MatchTable &Table) {
  assert(Matchers.size() == 1 && "Cannot handle multi-root matchers yet");
  unsigned InsnVarID = implicitlyDefineInsnVar(*Matchers.front());
  Matchers.front()->emitCaptureOpcodes(Table, *this, InsnVarID);
}

void RuleMatcher::emit(MatchTable &Table) {
  if (Matchers.empty())
    llvm_unreachable("Unexpected empty matcher!");

  // The representation supports rules that require multiple roots such as:
  //    %ptr(p0) = ...
  //    %elt0(s32) = G_LOAD %ptr
  //    %1(p0) = G_ADD %ptr, 4
  //    %elt1(s32) = G_LOAD p0 %1
  // which could be usefully folded into:
  //    %ptr(p0) = ...
  //    %elt0(s32), %elt1(s32) = TGT_LOAD_PAIR %ptr
  // on some targets but we don't need to make use of that yet.
  assert(Matchers.size() == 1 && "Cannot handle multi-root matchers yet");

  unsigned LabelID = Table.allocateLabelID();
  Table << MatchTable::Opcode("GIM_Try", +1)
        << MatchTable::Comment("On fail goto") << MatchTable::JumpTarget(LabelID)
        << MatchTable::LineBreak;

  if (!RequiredFeatures.empty()) {
    Table << MatchTable::Opcode("GIM_CheckFeatures")
          << MatchTable::NamedValue(getNameForFeatureBitset(RequiredFeatures))
          << MatchTable::LineBreak;
  }

  emitCaptureOpcodes(Table);

  Matchers.front()->emitPredicateOpcodes(Table, *this,
                                         getInsnVarID(*Matchers.front()));

  // We must also check if it's safe to fold the matched instructions.
  if (InsnVariableIDs.size() >= 2) {
    // Invert the map to create stable ordering (by var names)
    SmallVector<unsigned, 2> InsnIDs;
    for (const auto &Pair : InsnVariableIDs) {
      // Skip the root node since it isn't moving anywhere. Everything else is
      // sinking to meet it.
      if (Pair.first == Matchers.front().get())
        continue;

      InsnIDs.push_back(Pair.second);
    }
    std::sort(InsnIDs.begin(), InsnIDs.end());

    for (const auto &InsnID : InsnIDs) {
      // Reject the difficult cases until we have a more accurate check.
      Table << MatchTable::Opcode("GIM_CheckIsSafeToFold")
            << MatchTable::Comment("InsnID") << MatchTable::IntValue(InsnID)
            << MatchTable::LineBreak;

      // FIXME: Emit checks to determine it's _actually_ safe to fold and/or
      //        account for unsafe cases.
      //
      //        Example:
      //          MI1--> %0 = ...
      //                 %1 = ... %0
      //          MI0--> %2 = ... %0
      //          It's not safe to erase MI1. We currently handle this by not
      //          erasing %0 (even when it's dead).
      //
      //        Example:
      //          MI1--> %0 = load volatile @a
      //                 %1 = load volatile @a
      //          MI0--> %2 = ... %0
      //          It's not safe to sink %0's def past %1. We currently handle
      //          this by rejecting all loads.
      //
      //        Example:
      //          MI1--> %0 = load @a
      //                 %1 = store @a
      //          MI0--> %2 = ... %0
      //          It's not safe to sink %0's def past %1. We currently handle
      //          this by rejecting all loads.
      //
      //        Example:
      //                   G_CONDBR %cond, @BB1
      //                 BB0:
      //          MI1-->   %0 = load @a
      //                   G_BR @BB1
      //                 BB1:
      //          MI0-->   %2 = ... %0
      //          It's not always safe to sink %0 across control flow. In this
      //          case it may introduce a memory fault. We currentl handle this
      //          by rejecting all loads.
    }
  }

  for (const auto &MA : Actions)
    MA->emitActionOpcodes(Table, *this, 0);
  Table << MatchTable::Opcode("GIR_Done", -1) << MatchTable::LineBreak
        << MatchTable::Label(LabelID);
}

bool RuleMatcher::isHigherPriorityThan(const RuleMatcher &B) const {
  // Rules involving more match roots have higher priority.
  if (Matchers.size() > B.Matchers.size())
    return true;
  if (Matchers.size() < B.Matchers.size())
    return false;

  for (const auto &Matcher : zip(Matchers, B.Matchers)) {
    if (std::get<0>(Matcher)->isHigherPriorityThan(*std::get<1>(Matcher)))
      return true;
    if (std::get<1>(Matcher)->isHigherPriorityThan(*std::get<0>(Matcher)))
      return false;
  }

  return false;
}

unsigned RuleMatcher::countRendererFns() const {
  return std::accumulate(
      Matchers.begin(), Matchers.end(), 0,
      [](unsigned A, const std::unique_ptr<InstructionMatcher> &Matcher) {
        return A + Matcher->countRendererFns();
      });
}

bool OperandPredicateMatcher::isHigherPriorityThan(
    const OperandPredicateMatcher &B) const {
  // Generally speaking, an instruction is more important than an Int or a
  // LiteralInt because it can cover more nodes but theres an exception to
  // this. G_CONSTANT's are less important than either of those two because they
  // are more permissive.

  const InstructionOperandMatcher *AOM =
      dyn_cast<InstructionOperandMatcher>(this);
  const InstructionOperandMatcher *BOM =
      dyn_cast<InstructionOperandMatcher>(&B);
  bool AIsConstantInsn = AOM && AOM->getInsnMatcher().isConstantInstruction();
  bool BIsConstantInsn = BOM && BOM->getInsnMatcher().isConstantInstruction();

  if (AOM && BOM) {
    // The relative priorities between a G_CONSTANT and any other instruction
    // don't actually matter but this code is needed to ensure a strict weak
    // ordering. This is particularly important on Windows where the rules will
    // be incorrectly sorted without it.
    if (AIsConstantInsn != BIsConstantInsn)
      return AIsConstantInsn < BIsConstantInsn;
    return false;
  }

  if (AOM && AIsConstantInsn && (B.Kind == OPM_Int || B.Kind == OPM_LiteralInt))
    return false;
  if (BOM && BIsConstantInsn && (Kind == OPM_Int || Kind == OPM_LiteralInt))
    return true;

  return Kind < B.Kind;
}

//===- GlobalISelEmitter class --------------------------------------------===//

class GlobalISelEmitter {
public:
  explicit GlobalISelEmitter(RecordKeeper &RK);
  void run(raw_ostream &OS);

private:
  const RecordKeeper &RK;
  const CodeGenDAGPatterns CGP;
  const CodeGenTarget &Target;
  CodeGenRegBank CGRegs;

  /// Keep track of the equivalence between SDNodes and Instruction.
  /// This is defined using 'GINodeEquiv' in the target description.
  DenseMap<Record *, const CodeGenInstruction *> NodeEquivs;

  /// Keep track of the equivalence between ComplexPattern's and
  /// GIComplexOperandMatcher. Map entries are specified by subclassing
  /// GIComplexPatternEquiv.
  DenseMap<const Record *, const Record *> ComplexPatternEquivs;

  // Map of predicates to their subtarget features.
  SubtargetFeatureInfoMap SubtargetFeatures;

  void gatherNodeEquivs();
  const CodeGenInstruction *findNodeEquiv(Record *N) const;

  Error importRulePredicates(RuleMatcher &M, ArrayRef<Init *> Predicates);
  Expected<InstructionMatcher &>
  createAndImportSelDAGMatcher(InstructionMatcher &InsnMatcher,
                               const TreePatternNode *Src,
                               unsigned &TempOpIdx) const;
  Error importChildMatcher(InstructionMatcher &InsnMatcher,
                           const TreePatternNode *SrcChild, unsigned OpIdx,
                           unsigned &TempOpIdx) const;
  Expected<BuildMIAction &>
  createAndImportInstructionRenderer(RuleMatcher &M, const TreePatternNode *Dst,
                                     const InstructionMatcher &InsnMatcher);
  Error importExplicitUseRenderer(BuildMIAction &DstMIBuilder,
                                  TreePatternNode *DstChild,
                                  const InstructionMatcher &InsnMatcher) const;
  Error importDefaultOperandRenderers(BuildMIAction &DstMIBuilder,
                                      DagInit *DefaultOps) const;
  Error
  importImplicitDefRenderers(BuildMIAction &DstMIBuilder,
                             const std::vector<Record *> &ImplicitDefs) const;

  /// Analyze pattern \p P, returning a matcher for it if possible.
  /// Otherwise, return an Error explaining why we don't support it.
  Expected<RuleMatcher> runOnPattern(const PatternToMatch &P);

  void declareSubtargetFeature(Record *Predicate);
};

void GlobalISelEmitter::gatherNodeEquivs() {
  assert(NodeEquivs.empty());
  for (Record *Equiv : RK.getAllDerivedDefinitions("GINodeEquiv"))
    NodeEquivs[Equiv->getValueAsDef("Node")] =
        &Target.getInstruction(Equiv->getValueAsDef("I"));

  assert(ComplexPatternEquivs.empty());
  for (Record *Equiv : RK.getAllDerivedDefinitions("GIComplexPatternEquiv")) {
    Record *SelDAGEquiv = Equiv->getValueAsDef("SelDAGEquivalent");
    if (!SelDAGEquiv)
      continue;
    ComplexPatternEquivs[SelDAGEquiv] = Equiv;
 }
}

const CodeGenInstruction *GlobalISelEmitter::findNodeEquiv(Record *N) const {
  return NodeEquivs.lookup(N);
}

GlobalISelEmitter::GlobalISelEmitter(RecordKeeper &RK)
    : RK(RK), CGP(RK), Target(CGP.getTargetInfo()), CGRegs(RK) {}

//===- Emitter ------------------------------------------------------------===//

Error
GlobalISelEmitter::importRulePredicates(RuleMatcher &M,
                                        ArrayRef<Init *> Predicates) {
  for (const Init *Predicate : Predicates) {
    const DefInit *PredicateDef = static_cast<const DefInit *>(Predicate);
    declareSubtargetFeature(PredicateDef->getDef());
    M.addRequiredFeature(PredicateDef->getDef());
  }

  return Error::success();
}

Expected<InstructionMatcher &>
GlobalISelEmitter::createAndImportSelDAGMatcher(InstructionMatcher &InsnMatcher,
                                                const TreePatternNode *Src,
                                                unsigned &TempOpIdx) const {
  const CodeGenInstruction *SrcGIOrNull = nullptr;

  // Start with the defined operands (i.e., the results of the root operator).
  if (Src->getExtTypes().size() > 1)
    return failedImport("Src pattern has multiple results");

  if (Src->isLeaf()) {
    Init *SrcInit = Src->getLeafValue();
    if (isa<IntInit>(SrcInit)) {
      InsnMatcher.addPredicate<InstructionOpcodeMatcher>(
          &Target.getInstruction(RK.getDef("G_CONSTANT")));
    } else
      return failedImport(
          "Unable to deduce gMIR opcode to handle Src (which is a leaf)");
  } else {
    SrcGIOrNull = findNodeEquiv(Src->getOperator());
    if (!SrcGIOrNull)
      return failedImport("Pattern operator lacks an equivalent Instruction" +
                          explainOperator(Src->getOperator()));
    auto &SrcGI = *SrcGIOrNull;

    // The operators look good: match the opcode
    InsnMatcher.addPredicate<InstructionOpcodeMatcher>(&SrcGI);
  }

  unsigned OpIdx = 0;
  for (const EEVT::TypeSet &Ty : Src->getExtTypes()) {
    auto OpTyOrNone = MVTToLLT(Ty.getConcrete());

    if (!OpTyOrNone)
      return failedImport(
          "Result of Src pattern operator has an unsupported type");

    // Results don't have a name unless they are the root node. The caller will
    // set the name if appropriate.
    OperandMatcher &OM = InsnMatcher.addOperand(OpIdx++, "", TempOpIdx);
    OM.addPredicate<LLTOperandMatcher>(*OpTyOrNone);
  }

  for (const auto &Predicate : Src->getPredicateFns()) {
    if (Predicate.isAlwaysTrue())
      continue;

    if (Predicate.isImmediatePattern()) {
      InsnMatcher.addPredicate<InstructionImmPredicateMatcher>(Predicate);
      continue;
    }

    return failedImport("Src pattern child has predicate (" +
                        explainPredicates(Src) + ")");
  }

  if (Src->isLeaf()) {
    Init *SrcInit = Src->getLeafValue();
    if (IntInit *SrcIntInit = dyn_cast<IntInit>(SrcInit)) {
      OperandMatcher &OM = InsnMatcher.addOperand(OpIdx++, "", TempOpIdx);
      OM.addPredicate<LiteralIntOperandMatcher>(SrcIntInit->getValue());
    } else
      return failedImport(
          "Unable to deduce gMIR opcode to handle Src (which is a leaf)");
  } else {
    assert(SrcGIOrNull &&
           "Expected to have already found an equivalent Instruction");
    if (SrcGIOrNull->TheDef->getName() == "G_CONSTANT") {
      // imm still has an operand but we don't need to do anything with it
      // here since we don't support ImmLeaf predicates yet. However, we still
      // need to note the hidden operand to get GIM_CheckNumOperands correct.
      InsnMatcher.addOperand(OpIdx++, "", TempOpIdx);
      return InsnMatcher;
    }

    // Match the used operands (i.e. the children of the operator).
    for (unsigned i = 0, e = Src->getNumChildren(); i != e; ++i) {
      TreePatternNode *SrcChild = Src->getChild(i);

      // For G_INTRINSIC, the operand immediately following the defs is an
      // intrinsic ID.
      if (SrcGIOrNull->TheDef->getName() == "G_INTRINSIC" && i == 0) {
        if (const CodeGenIntrinsic *II = Src->getIntrinsicInfo(CGP)) {
          OperandMatcher &OM =
              InsnMatcher.addOperand(OpIdx++, SrcChild->getName(), TempOpIdx);
          OM.addPredicate<IntrinsicIDOperandMatcher>(II);
          continue;
        }

        return failedImport("Expected IntInit containing instrinsic ID)");
      }

      if (auto Error =
              importChildMatcher(InsnMatcher, SrcChild, OpIdx++, TempOpIdx))
        return std::move(Error);
    }
  }

  return InsnMatcher;
}

Error GlobalISelEmitter::importChildMatcher(InstructionMatcher &InsnMatcher,
                                            const TreePatternNode *SrcChild,
                                            unsigned OpIdx,
                                            unsigned &TempOpIdx) const {
  OperandMatcher &OM =
      InsnMatcher.addOperand(OpIdx, SrcChild->getName(), TempOpIdx);

  ArrayRef<EEVT::TypeSet> ChildTypes = SrcChild->getExtTypes();
  if (ChildTypes.size() != 1)
    return failedImport("Src pattern child has multiple results");

  // Check MBB's before the type check since they are not a known type.
  if (!SrcChild->isLeaf()) {
    if (SrcChild->getOperator()->isSubClassOf("SDNode")) {
      auto &ChildSDNI = CGP.getSDNodeInfo(SrcChild->getOperator());
      if (ChildSDNI.getSDClassName() == "BasicBlockSDNode") {
        OM.addPredicate<MBBOperandMatcher>();
        return Error::success();
      }
    }
  }

  auto OpTyOrNone = MVTToLLT(ChildTypes.front().getConcrete());
  if (!OpTyOrNone)
    return failedImport("Src operand has an unsupported type (" + to_string(*SrcChild) + ")");
  OM.addPredicate<LLTOperandMatcher>(*OpTyOrNone);

  // Check for nested instructions.
  if (!SrcChild->isLeaf()) {
    // Map the node to a gMIR instruction.
    InstructionOperandMatcher &InsnOperand =
        OM.addPredicate<InstructionOperandMatcher>(SrcChild->getName());
    auto InsnMatcherOrError = createAndImportSelDAGMatcher(
        InsnOperand.getInsnMatcher(), SrcChild, TempOpIdx);
    if (auto Error = InsnMatcherOrError.takeError())
      return Error;

    return Error::success();
  }

  // Check for constant immediates.
  if (auto *ChildInt = dyn_cast<IntInit>(SrcChild->getLeafValue())) {
    OM.addPredicate<ConstantIntOperandMatcher>(ChildInt->getValue());
    return Error::success();
  }

  // Check for def's like register classes or ComplexPattern's.
  if (auto *ChildDefInit = dyn_cast<DefInit>(SrcChild->getLeafValue())) {
    auto *ChildRec = ChildDefInit->getDef();

    // Check for register classes.
    if (ChildRec->isSubClassOf("RegisterClass") ||
        ChildRec->isSubClassOf("RegisterOperand")) {
      OM.addPredicate<RegisterBankOperandMatcher>(
          Target.getRegisterClass(getInitValueAsRegClass(ChildDefInit)));
      return Error::success();
    }

    // Check for ComplexPattern's.
    if (ChildRec->isSubClassOf("ComplexPattern")) {
      const auto &ComplexPattern = ComplexPatternEquivs.find(ChildRec);
      if (ComplexPattern == ComplexPatternEquivs.end())
        return failedImport("SelectionDAG ComplexPattern (" +
                            ChildRec->getName() + ") not mapped to GlobalISel");

      OM.addPredicate<ComplexPatternOperandMatcher>(OM,
                                                    *ComplexPattern->second);
      TempOpIdx++;
      return Error::success();
    }

    if (ChildRec->isSubClassOf("ImmLeaf")) {
      return failedImport(
          "Src pattern child def is an unsupported tablegen class (ImmLeaf)");
    }

    return failedImport(
        "Src pattern child def is an unsupported tablegen class");
  }

  return failedImport("Src pattern child is an unsupported kind");
}

Error GlobalISelEmitter::importExplicitUseRenderer(
    BuildMIAction &DstMIBuilder, TreePatternNode *DstChild,
    const InstructionMatcher &InsnMatcher) const {
  if (DstChild->getTransformFn() != nullptr) {
    return failedImport("Dst pattern child has transform fn " +
                        DstChild->getTransformFn()->getName());
  }

  if (!DstChild->isLeaf()) {
    // We accept 'bb' here. It's an operator because BasicBlockSDNode isn't
    // inline, but in MI it's just another operand.
    if (DstChild->getOperator()->isSubClassOf("SDNode")) {
      auto &ChildSDNI = CGP.getSDNodeInfo(DstChild->getOperator());
      if (ChildSDNI.getSDClassName() == "BasicBlockSDNode") {
        DstMIBuilder.addRenderer<CopyRenderer>(0, InsnMatcher,
                                               DstChild->getName());
        return Error::success();
      }
    }

    // Similarly, imm is an operator in TreePatternNode's view but must be
    // rendered as operands.
    // FIXME: The target should be able to choose sign-extended when appropriate
    //        (e.g. on Mips).
    if (DstChild->getOperator()->getName() == "imm") {
      DstMIBuilder.addRenderer<CopyConstantAsImmRenderer>(0,
                                                          DstChild->getName());
      return Error::success();
    }

    return failedImport("Dst pattern child isn't a leaf node or an MBB" + llvm::to_string(*DstChild));
  }

  // Otherwise, we're looking for a bog-standard RegisterClass operand.
  if (auto *ChildDefInit = dyn_cast<DefInit>(DstChild->getLeafValue())) {
    auto *ChildRec = ChildDefInit->getDef();

    ArrayRef<EEVT::TypeSet> ChildTypes = DstChild->getExtTypes();
    if (ChildTypes.size() != 1)
      return failedImport("Dst pattern child has multiple results");

    auto OpTyOrNone = MVTToLLT(ChildTypes.front().getConcrete());
    if (!OpTyOrNone)
      return failedImport("Dst operand has an unsupported type");

    if (ChildRec->isSubClassOf("Register")) {
      DstMIBuilder.addRenderer<AddRegisterRenderer>(0, ChildRec);
      return Error::success();
    }

    if (ChildRec->isSubClassOf("RegisterClass") ||
        ChildRec->isSubClassOf("RegisterOperand")) {
      DstMIBuilder.addRenderer<CopyRenderer>(0, InsnMatcher,
                                             DstChild->getName());
      return Error::success();
    }

    if (ChildRec->isSubClassOf("ComplexPattern")) {
      const auto &ComplexPattern = ComplexPatternEquivs.find(ChildRec);
      if (ComplexPattern == ComplexPatternEquivs.end())
        return failedImport(
            "SelectionDAG ComplexPattern not mapped to GlobalISel");

      const OperandMatcher &OM = InsnMatcher.getOperand(DstChild->getName());
      DstMIBuilder.addRenderer<RenderComplexPatternOperand>(
          0, *ComplexPattern->second, DstChild->getName(),
          OM.getAllocatedTemporariesBaseID());
      return Error::success();
    }

    if (ChildRec->isSubClassOf("SDNodeXForm"))
      return failedImport("Dst pattern child def is an unsupported tablegen "
                          "class (SDNodeXForm)");

    return failedImport(
        "Dst pattern child def is an unsupported tablegen class");
  }

  return failedImport("Dst pattern child is an unsupported kind");
}

Expected<BuildMIAction &> GlobalISelEmitter::createAndImportInstructionRenderer(
    RuleMatcher &M, const TreePatternNode *Dst,
    const InstructionMatcher &InsnMatcher) {
  Record *DstOp = Dst->getOperator();
  if (!DstOp->isSubClassOf("Instruction")) {
    if (DstOp->isSubClassOf("ValueType"))
      return failedImport(
          "Pattern operator isn't an instruction (it's a ValueType)");
    return failedImport("Pattern operator isn't an instruction");
  }
  CodeGenInstruction *DstI = &Target.getInstruction(DstOp);

  unsigned DstINumUses = DstI->Operands.size() - DstI->Operands.NumDefs;
  unsigned ExpectedDstINumUses = Dst->getNumChildren();
  bool IsExtractSubReg = false;

  // COPY_TO_REGCLASS is just a copy with a ConstrainOperandToRegClassAction
  // attached. Similarly for EXTRACT_SUBREG except that's a subregister copy.
  if (DstI->TheDef->getName() == "COPY_TO_REGCLASS") {
    DstI = &Target.getInstruction(RK.getDef("COPY"));
    DstINumUses--; // Ignore the class constraint.
    ExpectedDstINumUses--;
  } else if (DstI->TheDef->getName() == "EXTRACT_SUBREG") {
    DstI = &Target.getInstruction(RK.getDef("COPY"));
    IsExtractSubReg = true;
  }

  auto &DstMIBuilder = M.addAction<BuildMIAction>(0, DstI, InsnMatcher);

  // Render the explicit defs.
  for (unsigned I = 0; I < DstI->Operands.NumDefs; ++I) {
    const CGIOperandList::OperandInfo &DstIOperand = DstI->Operands[I];
    DstMIBuilder.addRenderer<CopyRenderer>(0, InsnMatcher, DstIOperand.Name);
  }

  // EXTRACT_SUBREG needs to use a subregister COPY.
  if (IsExtractSubReg) {
    if (!Dst->getChild(0)->isLeaf())
      return failedImport("EXTRACT_SUBREG child #1 is not a leaf");

    if (DefInit *SubRegInit =
            dyn_cast<DefInit>(Dst->getChild(1)->getLeafValue())) {
      CodeGenRegisterClass *RC = CGRegs.getRegClass(
          getInitValueAsRegClass(Dst->getChild(0)->getLeafValue()));
      CodeGenSubRegIndex *SubIdx = CGRegs.getSubRegIdx(SubRegInit->getDef());

      const auto &SrcRCDstRCPair =
          RC->getMatchingSubClassWithSubRegs(CGRegs, SubIdx);
      if (SrcRCDstRCPair.hasValue()) {
        assert(SrcRCDstRCPair->second && "Couldn't find a matching subclass");
        if (SrcRCDstRCPair->first != RC)
          return failedImport("EXTRACT_SUBREG requires an additional COPY");
      }

      DstMIBuilder.addRenderer<CopySubRegRenderer>(
          0, InsnMatcher, Dst->getChild(0)->getName(), SubIdx);
      return DstMIBuilder;
    }

    return failedImport("EXTRACT_SUBREG child #1 is not a subreg index");
  }

  // Render the explicit uses.
  unsigned Child = 0;
  unsigned NumDefaultOps = 0;
  for (unsigned I = 0; I != DstINumUses; ++I) {
    const CGIOperandList::OperandInfo &DstIOperand =
        DstI->Operands[DstI->Operands.NumDefs + I];

    // If the operand has default values, introduce them now.
    // FIXME: Until we have a decent test case that dictates we should do
    // otherwise, we're going to assume that operands with default values cannot
    // be specified in the patterns. Therefore, adding them will not cause us to
    // end up with too many rendered operands.
    if (DstIOperand.Rec->isSubClassOf("OperandWithDefaultOps")) {
      DagInit *DefaultOps = DstIOperand.Rec->getValueAsDag("DefaultOps");
      if (auto Error = importDefaultOperandRenderers(DstMIBuilder, DefaultOps))
        return std::move(Error);
      ++NumDefaultOps;
      continue;
    }

    if (auto Error = importExplicitUseRenderer(
            DstMIBuilder, Dst->getChild(Child), InsnMatcher))
      return std::move(Error);
    ++Child;
  }

  if (NumDefaultOps + ExpectedDstINumUses != DstINumUses)
    return failedImport("Expected " + llvm::to_string(DstINumUses) +
                        " used operands but found " +
                        llvm::to_string(ExpectedDstINumUses) +
                        " explicit ones and " + llvm::to_string(NumDefaultOps) +
                        " default ones");

  return DstMIBuilder;
}

Error GlobalISelEmitter::importDefaultOperandRenderers(
    BuildMIAction &DstMIBuilder, DagInit *DefaultOps) const {
  for (const auto *DefaultOp : DefaultOps->getArgs()) {
    // Look through ValueType operators.
    if (const DagInit *DefaultDagOp = dyn_cast<DagInit>(DefaultOp)) {
      if (const DefInit *DefaultDagOperator =
              dyn_cast<DefInit>(DefaultDagOp->getOperator())) {
        if (DefaultDagOperator->getDef()->isSubClassOf("ValueType"))
          DefaultOp = DefaultDagOp->getArg(0);
      }
    }

    if (const DefInit *DefaultDefOp = dyn_cast<DefInit>(DefaultOp)) {
      DstMIBuilder.addRenderer<AddRegisterRenderer>(0, DefaultDefOp->getDef());
      continue;
    }

    if (const IntInit *DefaultIntOp = dyn_cast<IntInit>(DefaultOp)) {
      DstMIBuilder.addRenderer<ImmRenderer>(0, DefaultIntOp->getValue());
      continue;
    }

    return failedImport("Could not add default op");
  }

  return Error::success();
}

Error GlobalISelEmitter::importImplicitDefRenderers(
    BuildMIAction &DstMIBuilder,
    const std::vector<Record *> &ImplicitDefs) const {
  if (!ImplicitDefs.empty())
    return failedImport("Pattern defines a physical register");
  return Error::success();
}

Expected<RuleMatcher> GlobalISelEmitter::runOnPattern(const PatternToMatch &P) {
  // Keep track of the matchers and actions to emit.
  RuleMatcher M;
  M.addAction<DebugCommentAction>(P);

  if (auto Error = importRulePredicates(M, P.getPredicates()->getValues()))
    return std::move(Error);

  // Next, analyze the pattern operators.
  TreePatternNode *Src = P.getSrcPattern();
  TreePatternNode *Dst = P.getDstPattern();

  // If the root of either pattern isn't a simple operator, ignore it.
  if (auto Err = isTrivialOperatorNode(Dst))
    return failedImport("Dst pattern root isn't a trivial operator (" +
                        toString(std::move(Err)) + ")");
  if (auto Err = isTrivialOperatorNode(Src))
    return failedImport("Src pattern root isn't a trivial operator (" +
                        toString(std::move(Err)) + ")");

  InstructionMatcher &InsnMatcherTemp = M.addInstructionMatcher(Src->getName());
  unsigned TempOpIdx = 0;
  auto InsnMatcherOrError =
      createAndImportSelDAGMatcher(InsnMatcherTemp, Src, TempOpIdx);
  if (auto Error = InsnMatcherOrError.takeError())
    return std::move(Error);
  InstructionMatcher &InsnMatcher = InsnMatcherOrError.get();

  if (Dst->isLeaf()) {
    Record *RCDef = getInitValueAsRegClass(Dst->getLeafValue());

    const CodeGenRegisterClass &RC = Target.getRegisterClass(RCDef);
    if (RCDef) {
      // We need to replace the def and all its uses with the specified
      // operand. However, we must also insert COPY's wherever needed.
      // For now, emit a copy and let the register allocator clean up.
      auto &DstI = Target.getInstruction(RK.getDef("COPY"));
      const auto &DstIOperand = DstI.Operands[0];

      OperandMatcher &OM0 = InsnMatcher.getOperand(0);
      OM0.setSymbolicName(DstIOperand.Name);
      OM0.addPredicate<RegisterBankOperandMatcher>(RC);

      auto &DstMIBuilder = M.addAction<BuildMIAction>(0, &DstI, InsnMatcher);
      DstMIBuilder.addRenderer<CopyRenderer>(0, InsnMatcher, DstIOperand.Name);
      DstMIBuilder.addRenderer<CopyRenderer>(0, InsnMatcher, Dst->getName());
      M.addAction<ConstrainOperandToRegClassAction>(0, 0, RC);

      // We're done with this pattern!  It's eligible for GISel emission; return
      // it.
      ++NumPatternImported;
      return std::move(M);
    }

    return failedImport("Dst pattern root isn't a known leaf");
  }

  // Start with the defined operands (i.e., the results of the root operator).
  Record *DstOp = Dst->getOperator();
  if (!DstOp->isSubClassOf("Instruction"))
    return failedImport("Pattern operator isn't an instruction");

  auto &DstI = Target.getInstruction(DstOp);
  if (DstI.Operands.NumDefs != Src->getExtTypes().size())
    return failedImport("Src pattern results and dst MI defs are different (" +
                        to_string(Src->getExtTypes().size()) + " def(s) vs " +
                        to_string(DstI.Operands.NumDefs) + " def(s))");

  // The root of the match also has constraints on the register bank so that it
  // matches the result instruction.
  unsigned OpIdx = 0;
  for (const EEVT::TypeSet &Ty : Src->getExtTypes()) {
    (void)Ty;

    const auto &DstIOperand = DstI.Operands[OpIdx];
    Record *DstIOpRec = DstIOperand.Rec;
    if (DstI.TheDef->getName() == "COPY_TO_REGCLASS") {
      DstIOpRec = getInitValueAsRegClass(Dst->getChild(1)->getLeafValue());

      if (DstIOpRec == nullptr)
        return failedImport(
            "COPY_TO_REGCLASS operand #1 isn't a register class");
    } else if (DstI.TheDef->getName() == "EXTRACT_SUBREG") {
      if (!Dst->getChild(0)->isLeaf())
        return failedImport("EXTRACT_SUBREG operand #0 isn't a leaf");

      // We can assume that a subregister is in the same bank as it's super
      // register.
      DstIOpRec = getInitValueAsRegClass(Dst->getChild(0)->getLeafValue());

      if (DstIOpRec == nullptr)
        return failedImport(
            "EXTRACT_SUBREG operand #0 isn't a register class");
    } else if (DstIOpRec->isSubClassOf("RegisterOperand"))
      DstIOpRec = DstIOpRec->getValueAsDef("RegClass");
    else if (!DstIOpRec->isSubClassOf("RegisterClass"))
      return failedImport("Dst MI def isn't a register class" +
                          to_string(*Dst));

    OperandMatcher &OM = InsnMatcher.getOperand(OpIdx);
    OM.setSymbolicName(DstIOperand.Name);
    OM.addPredicate<RegisterBankOperandMatcher>(
        Target.getRegisterClass(DstIOpRec));
    ++OpIdx;
  }

  auto DstMIBuilderOrError =
      createAndImportInstructionRenderer(M, Dst, InsnMatcher);
  if (auto Error = DstMIBuilderOrError.takeError())
    return std::move(Error);
  BuildMIAction &DstMIBuilder = DstMIBuilderOrError.get();

  // Render the implicit defs.
  // These are only added to the root of the result.
  if (auto Error = importImplicitDefRenderers(DstMIBuilder, P.getDstRegs()))
    return std::move(Error);

  // Constrain the registers to classes. This is normally derived from the
  // emitted instruction but a few instructions require special handling.
  if (DstI.TheDef->getName() == "COPY_TO_REGCLASS") {
    // COPY_TO_REGCLASS does not provide operand constraints itself but the
    // result is constrained to the class given by the second child.
    Record *DstIOpRec =
        getInitValueAsRegClass(Dst->getChild(1)->getLeafValue());

    if (DstIOpRec == nullptr)
      return failedImport("COPY_TO_REGCLASS operand #1 isn't a register class");

    M.addAction<ConstrainOperandToRegClassAction>(
        0, 0, Target.getRegisterClass(DstIOpRec));

    // We're done with this pattern!  It's eligible for GISel emission; return
    // it.
    ++NumPatternImported;
    return std::move(M);
  }

  if (DstI.TheDef->getName() == "EXTRACT_SUBREG") {
    // EXTRACT_SUBREG selects into a subregister COPY but unlike most
    // instructions, the result register class is controlled by the
    // subregisters of the operand. As a result, we must constrain the result
    // class rather than check that it's already the right one.
    if (!Dst->getChild(0)->isLeaf())
      return failedImport("EXTRACT_SUBREG child #1 is not a leaf");

    DefInit *SubRegInit = dyn_cast<DefInit>(Dst->getChild(1)->getLeafValue());
    if (!SubRegInit)
      return failedImport("EXTRACT_SUBREG child #1 is not a subreg index");

    // Constrain the result to the same register bank as the operand.
    Record *DstIOpRec =
        getInitValueAsRegClass(Dst->getChild(0)->getLeafValue());

    if (DstIOpRec == nullptr)
      return failedImport("EXTRACT_SUBREG operand #1 isn't a register class");

    CodeGenSubRegIndex *SubIdx = CGRegs.getSubRegIdx(SubRegInit->getDef());
    CodeGenRegisterClass *SrcRC = CGRegs.getRegClass(DstIOpRec);

    // It would be nice to leave this constraint implicit but we're required
    // to pick a register class so constrain the result to a register class
    // that can hold the correct MVT.
    //
    // FIXME: This may introduce an extra copy if the chosen class doesn't
    //        actually contain the subregisters.
    assert(Src->getExtTypes().size() == 1 &&
             "Expected Src of EXTRACT_SUBREG to have one result type");

    const auto &SrcRCDstRCPair =
        SrcRC->getMatchingSubClassWithSubRegs(CGRegs, SubIdx);
    assert(SrcRCDstRCPair->second && "Couldn't find a matching subclass");
    M.addAction<ConstrainOperandToRegClassAction>(0, 0, *SrcRCDstRCPair->second);
    M.addAction<ConstrainOperandToRegClassAction>(0, 1, *SrcRCDstRCPair->first);

    // We're done with this pattern!  It's eligible for GISel emission; return
    // it.
    ++NumPatternImported;
    return std::move(M);
  }

  M.addAction<ConstrainOperandsToDefinitionAction>(0);

  // We're done with this pattern!  It's eligible for GISel emission; return it.
  ++NumPatternImported;
  return std::move(M);
}

void GlobalISelEmitter::run(raw_ostream &OS) {
  // Track the GINodeEquiv definitions.
  gatherNodeEquivs();

  emitSourceFileHeader(("Global Instruction Selector for the " +
                       Target.getName() + " target").str(), OS);
  std::vector<RuleMatcher> Rules;
  // Look through the SelectionDAG patterns we found, possibly emitting some.
  for (const PatternToMatch &Pat : CGP.ptms()) {
    ++NumPatternTotal;
    auto MatcherOrErr = runOnPattern(Pat);

    // The pattern analysis can fail, indicating an unsupported pattern.
    // Report that if we've been asked to do so.
    if (auto Err = MatcherOrErr.takeError()) {
      if (WarnOnSkippedPatterns) {
        PrintWarning(Pat.getSrcRecord()->getLoc(),
                     "Skipped pattern: " + toString(std::move(Err)));
      } else {
        consumeError(std::move(Err));
      }
      ++NumPatternImportsSkipped;
      continue;
    }

    Rules.push_back(std::move(MatcherOrErr.get()));
  }

  std::stable_sort(Rules.begin(), Rules.end(),
            [&](const RuleMatcher &A, const RuleMatcher &B) {
              if (A.isHigherPriorityThan(B)) {
                assert(!B.isHigherPriorityThan(A) && "Cannot be more important "
                                                     "and less important at "
                                                     "the same time");
                return true;
              }
              return false;
            });

  std::vector<Record *> ComplexPredicates =
      RK.getAllDerivedDefinitions("GIComplexOperandMatcher");
  std::sort(ComplexPredicates.begin(), ComplexPredicates.end(),
            [](const Record *A, const Record *B) {
              if (A->getName() < B->getName())
                return true;
              return false;
            });
  unsigned MaxTemporaries = 0;
  for (const auto &Rule : Rules)
    MaxTemporaries = std::max(MaxTemporaries, Rule.countRendererFns());

  OS << "#ifdef GET_GLOBALISEL_PREDICATE_BITSET\n"
     << "const unsigned MAX_SUBTARGET_PREDICATES = " << SubtargetFeatures.size()
     << ";\n"
     << "using PredicateBitset = "
        "llvm::PredicateBitsetImpl<MAX_SUBTARGET_PREDICATES>;\n"
     << "#endif // ifdef GET_GLOBALISEL_PREDICATE_BITSET\n\n";

  OS << "#ifdef GET_GLOBALISEL_TEMPORARIES_DECL\n"
     << "  mutable MatcherState State;\n"
     << "  typedef "
        "ComplexRendererFn("
     << Target.getName()
     << "InstructionSelector::*ComplexMatcherMemFn)(MachineOperand &) const;\n"
     << "const MatcherInfoTy<PredicateBitset, ComplexMatcherMemFn> "
        "MatcherInfo;\n"
     << "#endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL\n\n";

  OS << "#ifdef GET_GLOBALISEL_TEMPORARIES_INIT\n"
     << ", State(" << MaxTemporaries << "),\n"
     << "MatcherInfo({TypeObjects, FeatureBitsets, ImmPredicateFns, {\n"
     << "  nullptr, // GICP_Invalid\n";
  for (const auto &Record : ComplexPredicates)
    OS << "  &" << Target.getName()
       << "InstructionSelector::" << Record->getValueAsString("MatcherFn")
       << ", // " << Record->getName() << "\n";
  OS << "}})\n"
     << "#endif // ifdef GET_GLOBALISEL_TEMPORARIES_INIT\n\n";

  OS << "#ifdef GET_GLOBALISEL_IMPL\n";
  SubtargetFeatureInfo::emitSubtargetFeatureBitEnumeration(SubtargetFeatures,
                                                           OS);

  // Separate subtarget features by how often they must be recomputed.
  SubtargetFeatureInfoMap ModuleFeatures;
  std::copy_if(SubtargetFeatures.begin(), SubtargetFeatures.end(),
               std::inserter(ModuleFeatures, ModuleFeatures.end()),
               [](const SubtargetFeatureInfoMap::value_type &X) {
                 return !X.second.mustRecomputePerFunction();
               });
  SubtargetFeatureInfoMap FunctionFeatures;
  std::copy_if(SubtargetFeatures.begin(), SubtargetFeatures.end(),
               std::inserter(FunctionFeatures, FunctionFeatures.end()),
               [](const SubtargetFeatureInfoMap::value_type &X) {
                 return X.second.mustRecomputePerFunction();
               });

  SubtargetFeatureInfo::emitComputeAvailableFeatures(
      Target.getName(), "InstructionSelector", "computeAvailableModuleFeatures",
      ModuleFeatures, OS);
  SubtargetFeatureInfo::emitComputeAvailableFeatures(
      Target.getName(), "InstructionSelector",
      "computeAvailableFunctionFeatures", FunctionFeatures, OS,
      "const MachineFunction *MF");

  // Emit a table containing the LLT objects needed by the matcher and an enum
  // for the matcher to reference them with.
  std::vector<LLTCodeGen> TypeObjects;
  for (const auto &Ty : LLTOperandMatcher::KnownTypes)
    TypeObjects.push_back(Ty);
  std::sort(TypeObjects.begin(), TypeObjects.end());
  OS << "// LLT Objects.\n"
     << "enum {\n";
  for (const auto &TypeObject : TypeObjects) {
    OS << "  ";
    TypeObject.emitCxxEnumValue(OS);
    OS << ",\n";
  }
  OS << "};\n"
     << "const static LLT TypeObjects[] = {\n";
  for (const auto &TypeObject : TypeObjects) {
    OS << "  ";
    TypeObject.emitCxxConstructorCall(OS);
    OS << ",\n";
  }
  OS << "};\n\n";

  // Emit a table containing the PredicateBitsets objects needed by the matcher
  // and an enum for the matcher to reference them with.
  std::vector<std::vector<Record *>> FeatureBitsets;
  for (auto &Rule : Rules)
    FeatureBitsets.push_back(Rule.getRequiredFeatures());
  std::sort(
      FeatureBitsets.begin(), FeatureBitsets.end(),
      [&](const std::vector<Record *> &A, const std::vector<Record *> &B) {
        if (A.size() < B.size())
          return true;
        if (A.size() > B.size())
          return false;
        for (const auto &Pair : zip(A, B)) {
          if (std::get<0>(Pair)->getName() < std::get<1>(Pair)->getName())
            return true;
          if (std::get<0>(Pair)->getName() > std::get<1>(Pair)->getName())
            return false;
        }
        return false;
      });
  FeatureBitsets.erase(
      std::unique(FeatureBitsets.begin(), FeatureBitsets.end()),
      FeatureBitsets.end());
  OS << "// Feature bitsets.\n"
     << "enum {\n"
     << "  GIFBS_Invalid,\n";
  for (const auto &FeatureBitset : FeatureBitsets) {
    if (FeatureBitset.empty())
      continue;
    OS << "  " << getNameForFeatureBitset(FeatureBitset) << ",\n";
  }
  OS << "};\n"
     << "const static PredicateBitset FeatureBitsets[] {\n"
     << "  {}, // GIFBS_Invalid\n";
  for (const auto &FeatureBitset : FeatureBitsets) {
    if (FeatureBitset.empty())
      continue;
    OS << "  {";
    for (const auto &Feature : FeatureBitset) {
      const auto &I = SubtargetFeatures.find(Feature);
      assert(I != SubtargetFeatures.end() && "Didn't import predicate?");
      OS << I->second.getEnumBitName() << ", ";
    }
    OS << "},\n";
  }
  OS << "};\n\n";

  // Emit complex predicate table and an enum to reference them with.
  OS << "// ComplexPattern predicates.\n"
     << "enum {\n"
     << "  GICP_Invalid,\n";
  for (const auto &Record : ComplexPredicates)
    OS << "  GICP_" << Record->getName() << ",\n";
  OS << "};\n"
     << "// See constructor for table contents\n\n";

  // Emit imm predicate table and an enum to reference them with.
  // The 'Predicate_' part of the name is redundant but eliminating it is more
  // trouble than it's worth.
  {
    OS << "// PatFrag predicates.\n"
       << "enum {\n";
    StringRef EnumeratorSeparator = " = GIPFP_Invalid,\n";
    for (const auto *Record : RK.getAllDerivedDefinitions("PatFrag")) {
      if (!Record->getValueAsString("ImmediateCode").empty()) {
        OS << "  GIPFP_Predicate_" << Record->getName() << EnumeratorSeparator;
        EnumeratorSeparator = ",\n";
      }
    }
    OS << "};\n";
  }
  for (const auto *Record : RK.getAllDerivedDefinitions("PatFrag"))
    if (!Record->getValueAsString("ImmediateCode").empty())
      OS << "  static bool Predicate_" << Record->getName() << "(int64_t Imm) {"
         << Record->getValueAsString("ImmediateCode") << "  }\n";
  OS << "static InstructionSelector::ImmediatePredicateFn ImmPredicateFns[] = {\n";
  for (const auto *Record : RK.getAllDerivedDefinitions("PatFrag"))
    if (!Record->getValueAsString("ImmediateCode").empty())
      OS << "  Predicate_" << Record->getName() << ",\n";
  OS << "};\n";

  OS << "bool " << Target.getName()
     << "InstructionSelector::selectImpl(MachineInstr &I) const {\n"
     << "  MachineFunction &MF = *I.getParent()->getParent();\n"
     << "  MachineRegisterInfo &MRI = MF.getRegInfo();\n"
     << "  // FIXME: This should be computed on a per-function basis rather "
        "than per-insn.\n"
     << "  AvailableFunctionFeatures = computeAvailableFunctionFeatures(&STI, "
        "&MF);\n"
     << "  const PredicateBitset AvailableFeatures = getAvailableFeatures();\n"
     << "  NewMIVector OutMIs;\n"
     << "  State.MIs.clear();\n"
     << "  State.MIs.push_back(&I);\n\n";

  MatchTable Table(0);
  for (auto &Rule : Rules) {
    Rule.emit(Table);
    ++NumPatternEmitted;
  }
  Table << MatchTable::Opcode("GIM_Reject") << MatchTable::LineBreak;
  Table.emitDeclaration(OS);
  OS << "  if (executeMatchTable(*this, OutMIs, State, MatcherInfo, ";
  Table.emitUse(OS);
  OS << ", TII, MRI, TRI, RBI, AvailableFeatures)) {\n"
     << "    return true;\n"
     << "  }\n\n";

  OS << "  return false;\n"
     << "}\n"
     << "#endif // ifdef GET_GLOBALISEL_IMPL\n";

  OS << "#ifdef GET_GLOBALISEL_PREDICATES_DECL\n"
     << "PredicateBitset AvailableModuleFeatures;\n"
     << "mutable PredicateBitset AvailableFunctionFeatures;\n"
     << "PredicateBitset getAvailableFeatures() const {\n"
     << "  return AvailableModuleFeatures | AvailableFunctionFeatures;\n"
     << "}\n"
     << "PredicateBitset\n"
     << "computeAvailableModuleFeatures(const " << Target.getName()
     << "Subtarget *Subtarget) const;\n"
     << "PredicateBitset\n"
     << "computeAvailableFunctionFeatures(const " << Target.getName()
     << "Subtarget *Subtarget,\n"
     << "                                 const MachineFunction *MF) const;\n"
     << "#endif // ifdef GET_GLOBALISEL_PREDICATES_DECL\n";

  OS << "#ifdef GET_GLOBALISEL_PREDICATES_INIT\n"
     << "AvailableModuleFeatures(computeAvailableModuleFeatures(&STI)),\n"
     << "AvailableFunctionFeatures()\n"
     << "#endif // ifdef GET_GLOBALISEL_PREDICATES_INIT\n";
}

void GlobalISelEmitter::declareSubtargetFeature(Record *Predicate) {
  if (SubtargetFeatures.count(Predicate) == 0)
    SubtargetFeatures.emplace(
        Predicate, SubtargetFeatureInfo(Predicate, SubtargetFeatures.size()));
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//

namespace llvm {
void EmitGlobalISel(RecordKeeper &RK, raw_ostream &OS) {
  GlobalISelEmitter(RK).run(OS);
}
} // End llvm namespace
