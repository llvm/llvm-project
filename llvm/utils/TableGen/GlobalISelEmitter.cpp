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

  void emitCxxConstructorCall(raw_ostream &OS) const {
    if (Ty.isScalar()) {
      OS << "LLT::scalar(" << Ty.getSizeInBits() << ")";
      return;
    }
    if (Ty.isVector()) {
      OS << "LLT::vector(" << Ty.getNumElements() << ", " << Ty.getScalarSizeInBits()
         << ")";
      return;
    }
    llvm_unreachable("Unhandled LLT");
  }

  const LLT &get() const { return Ty; }
};

class InstructionMatcher;
/// Convert an MVT to an equivalent LLT if possible, or the invalid LLT() for
/// MVTs that don't map cleanly to an LLT (e.g., iPTR, *any, ...).
static Optional<LLTCodeGen> MVTToLLT(MVT::SimpleValueType SVT) {
  MVT VT(SVT);
  if (VT.isVector() && VT.getVectorNumElements() != 1)
    return LLTCodeGen(LLT::vector(VT.getVectorNumElements(), VT.getScalarSizeInBits()));
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
    return " (" + Operator->getValueAsString("Opcode") + ")";

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
  if (N->isLeaf()) {
    Explanation = "Is a leaf";
    Separator = ", ";
  }

  if (N->hasAnyPredicate()) {
    Explanation = Separator + "Has a predicate (" + explainPredicates(N) + ")";
    Separator = ", ";
  }

  if (N->getTransformFn()) {
    Explanation += Separator + "Has a transform function";
    Separator = ", ";
  }

  if (!N->isLeaf() && !N->hasAnyPredicate() && !N->getTransformFn())
    return Error::success();

  return failedImport(Explanation);
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

  /// A map of instruction matchers to the local variables created by
  /// emitCxxCaptureStmts().
  std::map<const InstructionMatcher *, std::string> InsnVariableNames;

  /// ID for the next instruction variable defined with defineInsnVar()
  unsigned NextInsnVarID;

  std::vector<Record *> RequiredFeatures;

public:
  RuleMatcher()
      : Matchers(), Actions(), InsnVariableNames(), NextInsnVarID(0) {}
  RuleMatcher(RuleMatcher &&Other) = default;
  RuleMatcher &operator=(RuleMatcher &&Other) = default;

  InstructionMatcher &addInstructionMatcher();
  void addRequiredFeature(Record *Feature);

  template <class Kind, class... Args> Kind &addAction(Args &&... args);

  std::string defineInsnVar(raw_ostream &OS, const InstructionMatcher &Matcher,
                            StringRef Value);
  StringRef getInsnVarName(const InstructionMatcher &InsnMatcher) const;

  void emitCxxCapturedInsnList(raw_ostream &OS);
  void emitCxxCaptureStmts(raw_ostream &OS, StringRef Expr);

  void emit(raw_ostream &OS,
            std::map<Record *, SubtargetFeatureInfo, LessRecordByID>
                SubtargetFeatures);

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

public:
  /// Construct a new operand predicate and add it to the matcher.
  template <class Kind, class... Args>
  Kind &addPredicate(Args&&... args) {
    Predicates.emplace_back(
        llvm::make_unique<Kind>(std::forward<Args>(args)...));
    return *static_cast<Kind *>(Predicates.back().get());
  }

  typename PredicateVec::const_iterator predicates_begin() const { return Predicates.begin(); }
  typename PredicateVec::const_iterator predicates_end() const { return Predicates.end(); }
  iterator_range<typename PredicateVec::const_iterator> predicates() const {
    return make_range(predicates_begin(), predicates_end());
  }
  typename PredicateVec::size_type predicates_size() const { return Predicates.size(); }

  /// Emit a C++ expression that tests whether all the predicates are met.
  template <class... Args>
  void emitCxxPredicateListExpr(raw_ostream &OS, Args &&... args) const {
    if (Predicates.empty()) {
      OS << "true";
      return;
    }

    StringRef Separator = "";
    for (const auto &Predicate : predicates()) {
      OS << Separator << "(";
      Predicate->emitCxxPredicateExpr(OS, std::forward<Args>(args)...);
      OS << ")";
      Separator = " &&\n";
    }
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
    OPM_Instruction,
    OPM_Int,
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

  /// Emit C++ statements to capture instructions into local variables.
  ///
  /// Only InstructionOperandMatcher needs to do anything for this method.
  virtual void emitCxxCaptureStmts(raw_ostream &OS, RuleMatcher &Rule,
                                   StringRef Expr) const {}

  /// Emit a C++ expression that checks the predicate for the given operand.
  virtual void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                                    StringRef OperandExpr) const = 0;

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  virtual bool isHigherPriorityThan(const OperandPredicateMatcher &B) const {
    return Kind < B.Kind;
  };

  /// Report the maximum number of temporary operands needed by the predicate
  /// matcher.
  virtual unsigned countRendererFns() const { return 0; }
};

/// Generates code to check that an operand is a particular LLT.
class LLTOperandMatcher : public OperandPredicateMatcher {
protected:
  LLTCodeGen Ty;

public:
  LLTOperandMatcher(const LLTCodeGen &Ty)
      : OperandPredicateMatcher(OPM_LLT), Ty(Ty) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_LLT;
  }

  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef OperandExpr) const override {
    OS << "MRI.getType(" << OperandExpr << ".getReg()) == (";
    Ty.emitCxxConstructorCall(OS);
    OS << ")";
  }
};

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

  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef OperandExpr) const override {
    unsigned ID = getAllocatedTemporariesBaseID();
    OS << "(Renderer" << ID << " = " << TheDef.getValueAsString("MatcherFn")
       << "(" << OperandExpr << "))";
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

  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef OperandExpr) const override {
    OS << "(&RBI.getRegBankFromRegClass(" << RC.getQualifiedName()
       << "RegClass) == RBI.getRegBank(" << OperandExpr
       << ".getReg(), MRI, TRI))";
  }
};

/// Generates code to check that an operand is a basic block.
class MBBOperandMatcher : public OperandPredicateMatcher {
public:
  MBBOperandMatcher() : OperandPredicateMatcher(OPM_MBB) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_MBB;
  }

  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef OperandExpr) const override {
    OS << OperandExpr << ".isMBB()";
  }
};

/// Generates code to check that an operand is a particular int.
class IntOperandMatcher : public OperandPredicateMatcher {
protected:
  int64_t Value;

public:
  IntOperandMatcher(int64_t Value)
      : OperandPredicateMatcher(OPM_Int), Value(Value) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_Int;
  }

  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef OperandExpr) const override {
    OS << "isOperandImmEqual(" << OperandExpr << ", " << Value << ", MRI)";
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

  std::string getOperandExpr(StringRef InsnVarName) const {
    return (InsnVarName + ".getOperand(" + llvm::to_string(OpIdx) + ")").str();
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

  /// Emit C++ statements to capture instructions into local variables.
  void emitCxxCaptureStmts(raw_ostream &OS, RuleMatcher &Rule,
                           StringRef OperandExpr) const {
    for (const auto &Predicate : predicates())
      Predicate->emitCxxCaptureStmts(OS, Rule, OperandExpr);
  }

  /// Emit a C++ expression that tests whether the instruction named in
  /// InsnVarName matches all the predicate and all the operands.
  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef InsnVarName) const {
    OS << "(/* ";
    if (SymbolicName.empty())
      OS << "Operand " << OpIdx;
    else
      OS << SymbolicName;
    OS << " */ ";
    emitCxxPredicateListExpr(OS, Rule, getOperandExpr(InsnVarName));
    OS << ")";
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
  };

  PredicateKind Kind;

public:
  InstructionPredicateMatcher(PredicateKind Kind) : Kind(Kind) {}
  virtual ~InstructionPredicateMatcher() {}

  PredicateKind getKind() const { return Kind; }

  /// Emit a C++ expression that tests whether the instruction named in
  /// InsnVarName matches the predicate.
  virtual void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                                    StringRef InsnVarName) const = 0;

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  virtual bool isHigherPriorityThan(const InstructionPredicateMatcher &B) const {
    return Kind < B.Kind;
  };

  /// Report the maximum number of temporary operands needed by the predicate
  /// matcher.
  virtual unsigned countRendererFns() const { return 0; }
};

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

  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef InsnVarName) const override {
    OS << InsnVarName << ".getOpcode() == " << I->Namespace
       << "::" << I->TheDef->getName();
  }

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  bool isHigherPriorityThan(const InstructionPredicateMatcher &B) const override {
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

public:
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

  /// Emit C++ statements to check the shape of the match and capture
  /// instructions into local variables.
  void emitCxxCaptureStmts(raw_ostream &OS, RuleMatcher &Rule, StringRef Expr) {
    OS << "if (" << Expr << ".getNumOperands() < " << getNumOperands() << ")\n"
       << "  return false;\n";
    for (const auto &Operand : Operands) {
      Operand->emitCxxCaptureStmts(OS, Rule, Operand->getOperandExpr(Expr));
    }
  }

  /// Emit a C++ expression that tests whether the instruction named in
  /// InsnVarName matches all the predicates and all the operands.
  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef InsnVarName) const {
    emitCxxPredicateListExpr(OS, Rule, InsnVarName);
    for (const auto &Operand : Operands) {
      OS << " &&\n(";
      Operand->emitCxxPredicateExpr(OS, Rule, InsnVarName);
      OS << ")";
    }
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
  InstructionOperandMatcher()
      : OperandPredicateMatcher(OPM_Instruction),
        InsnMatcher(new InstructionMatcher()) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_Instruction;
  }

  InstructionMatcher &getInsnMatcher() const { return *InsnMatcher; }

  Optional<const OperandMatcher *>
  getOptionalOperand(StringRef SymbolicName) const override {
    assert(!SymbolicName.empty() && "Cannot lookup unnamed operand");
    return InsnMatcher->getOptionalOperand(SymbolicName);
  }

  void emitCxxCaptureStmts(raw_ostream &OS, RuleMatcher &Rule,
                           StringRef OperandExpr) const override {
    OS << "if (!" << OperandExpr + ".isReg())\n"
       << "  return false;\n";
    std::string InsnVarName = Rule.defineInsnVar(
        OS, *InsnMatcher,
        ("*MRI.getVRegDef(" + OperandExpr + ".getReg())").str());
    InsnMatcher->emitCxxCaptureStmts(OS, Rule, InsnVarName);
  }

  void emitCxxPredicateExpr(raw_ostream &OS, RuleMatcher &Rule,
                            StringRef OperandExpr) const override {
    OperandExpr = Rule.getInsnVarName(*InsnMatcher);
    OS << "(";
    InsnMatcher->emitCxxPredicateExpr(OS, Rule, OperandExpr);
    OS << ")\n";
  }
};

//===- Actions ------------------------------------------------------------===//
class OperandRenderer {
public:
  enum RendererKind { OR_Copy, OR_Imm, OR_Register, OR_ComplexPattern };

protected:
  RendererKind Kind;

public:
  OperandRenderer(RendererKind Kind) : Kind(Kind) {}
  virtual ~OperandRenderer() {}

  RendererKind getKind() const { return Kind; }

  virtual void emitCxxRenderStmts(raw_ostream &OS, RuleMatcher &Rule) const = 0;
};

/// A CopyRenderer emits code to copy a single operand from an existing
/// instruction to the one being built.
class CopyRenderer : public OperandRenderer {
protected:
  /// The matcher for the instruction that this operand is copied from.
  /// This provides the facility for looking up an a operand by it's name so
  /// that it can be used as a source for the instruction being built.
  const InstructionMatcher &Matched;
  /// The name of the operand.
  const StringRef SymbolicName;

public:
  CopyRenderer(const InstructionMatcher &Matched, StringRef SymbolicName)
      : OperandRenderer(OR_Copy), Matched(Matched), SymbolicName(SymbolicName) {
  }

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_Copy;
  }

  const StringRef getSymbolicName() const { return SymbolicName; }

  void emitCxxRenderStmts(raw_ostream &OS, RuleMatcher &Rule) const override {
    const OperandMatcher &Operand = Matched.getOperand(SymbolicName);
    StringRef InsnVarName =
        Rule.getInsnVarName(Operand.getInstructionMatcher());
    std::string OperandExpr = Operand.getOperandExpr(InsnVarName);
    OS << "    MIB.add(" << OperandExpr << "/*" << SymbolicName << "*/);\n";
  }
};

/// Adds a specific physical register to the instruction being built.
/// This is typically useful for WZR/XZR on AArch64.
class AddRegisterRenderer : public OperandRenderer {
protected:
  const Record *RegisterDef;

public:
  AddRegisterRenderer(const Record *RegisterDef)
      : OperandRenderer(OR_Register), RegisterDef(RegisterDef) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_Register;
  }

  void emitCxxRenderStmts(raw_ostream &OS, RuleMatcher &Rule) const override {
    OS << "    MIB.addReg(" << RegisterDef->getValueAsString("Namespace")
       << "::" << RegisterDef->getName() << ");\n";
  }
};

/// Adds a specific immediate to the instruction being built.
class ImmRenderer : public OperandRenderer {
protected:
  int64_t Imm;

public:
  ImmRenderer(int64_t Imm)
      : OperandRenderer(OR_Imm), Imm(Imm) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_Imm;
  }

  void emitCxxRenderStmts(raw_ostream &OS, RuleMatcher &Rule) const override {
    OS << "    MIB.addImm(" << Imm << ");\n";
  }
};

/// Adds operands by calling a renderer function supplied by the ComplexPattern
/// matcher function.
class RenderComplexPatternOperand : public OperandRenderer {
private:
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
  RenderComplexPatternOperand(const Record &TheDef, StringRef SymbolicName,
                              unsigned RendererID)
      : OperandRenderer(OR_ComplexPattern), TheDef(TheDef),
        SymbolicName(SymbolicName), RendererID(RendererID) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_ComplexPattern;
  }

  void emitCxxRenderStmts(raw_ostream &OS, RuleMatcher &Rule) const override {
    OS << "Renderer" << RendererID << "(MIB);\n";
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

  /// Emit the C++ statements to implement the action.
  ///
  /// \param RecycleVarName If given, it's an instruction to recycle. The
  ///                       requirements on the instruction vary from action to
  ///                       action.
  virtual void emitCxxActionStmts(raw_ostream &OS, RuleMatcher &Rule,
                                  StringRef RecycleVarName) const = 0;
};

/// Generates a comment describing the matched rule being acted upon.
class DebugCommentAction : public MatchAction {
private:
  const PatternToMatch &P;

public:
  DebugCommentAction(const PatternToMatch &P) : P(P) {}

  void emitCxxActionStmts(raw_ostream &OS, RuleMatcher &Rule,
                          StringRef RecycleVarName) const override {
    OS << "// " << *P.getSrcPattern() << "  =>  " << *P.getDstPattern() << "\n";
  }
};

/// Generates code to build an instruction or mutate an existing instruction
/// into the desired instruction when this is possible.
class BuildMIAction : public MatchAction {
private:
  const CodeGenInstruction *I;
  const InstructionMatcher &Matched;
  std::vector<std::unique_ptr<OperandRenderer>> OperandRenderers;

  /// True if the instruction can be built solely by mutating the opcode.
  bool canMutate() const {
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
  BuildMIAction(const CodeGenInstruction *I, const InstructionMatcher &Matched)
      : I(I), Matched(Matched) {}

  template <class Kind, class... Args>
  Kind &addRenderer(Args&&... args) {
    OperandRenderers.emplace_back(
        llvm::make_unique<Kind>(std::forward<Args>(args)...));
    return *static_cast<Kind *>(OperandRenderers.back().get());
  }

  void emitCxxActionStmts(raw_ostream &OS, RuleMatcher &Rule,
                          StringRef RecycleVarName) const override {
    if (canMutate()) {
      OS << "    " << RecycleVarName << ".setDesc(TII.get(" << I->Namespace
         << "::" << I->TheDef->getName() << "));\n";

      if (!I->ImplicitDefs.empty() || !I->ImplicitUses.empty()) {
        OS << "    auto MIB = MachineInstrBuilder(MF, &" << RecycleVarName
           << ");\n";

        for (auto Def : I->ImplicitDefs) {
          auto Namespace = Def->getValueAsString("Namespace");
          OS << "    MIB.addDef(" << Namespace << "::" << Def->getName()
             << ", RegState::Implicit);\n";
        }
        for (auto Use : I->ImplicitUses) {
          auto Namespace = Use->getValueAsString("Namespace");
          OS << "    MIB.addUse(" << Namespace << "::" << Use->getName()
             << ", RegState::Implicit);\n";
        }
      }

      OS << "    MachineInstr &NewI = " << RecycleVarName << ";\n";
      return;
    }

    // TODO: Simple permutation looks like it could be almost as common as
    //       mutation due to commutative operations.

    OS << "MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, "
          "I.getDebugLoc(), TII.get("
       << I->Namespace << "::" << I->TheDef->getName() << "));\n";
    for (const auto &Renderer : OperandRenderers)
      Renderer->emitCxxRenderStmts(OS, Rule);
    OS << "    for (const auto *FromMI : ";
    Rule.emitCxxCapturedInsnList(OS);
    OS << ")\n";
    OS << "      for (const auto &MMO : FromMI->memoperands())\n";
    OS << "        MIB.addMemOperand(MMO);\n";
    OS << "    " << RecycleVarName << ".eraseFromParent();\n";
    OS << "    MachineInstr &NewI = *MIB;\n";
  }
};

InstructionMatcher &RuleMatcher::addInstructionMatcher() {
  Matchers.emplace_back(new InstructionMatcher());
  return *Matchers.back();
}

void RuleMatcher::addRequiredFeature(Record *Feature) {
  RequiredFeatures.push_back(Feature);
}

template <class Kind, class... Args>
Kind &RuleMatcher::addAction(Args &&... args) {
  Actions.emplace_back(llvm::make_unique<Kind>(std::forward<Args>(args)...));
  return *static_cast<Kind *>(Actions.back().get());
}

std::string RuleMatcher::defineInsnVar(raw_ostream &OS,
                                       const InstructionMatcher &Matcher,
                                       StringRef Value) {
  std::string InsnVarName = "MI" + llvm::to_string(NextInsnVarID++);
  OS << "MachineInstr &" << InsnVarName << " = " << Value << ";\n";
  InsnVariableNames[&Matcher] = InsnVarName;
  return InsnVarName;
}

StringRef RuleMatcher::getInsnVarName(const InstructionMatcher &InsnMatcher) const {
  const auto &I = InsnVariableNames.find(&InsnMatcher);
  if (I != InsnVariableNames.end())
    return I->second;
  llvm_unreachable("Matched Insn was not captured in a local variable");
}

/// Emit a C++ initializer_list containing references to every matched instruction.
void RuleMatcher::emitCxxCapturedInsnList(raw_ostream &OS) {
  SmallVector<StringRef, 2> Names;
  for (const auto &Pair : InsnVariableNames)
    Names.push_back(Pair.second);
  std::sort(Names.begin(), Names.end());

  OS << "{";
  for (const auto &Name : Names)
    OS << "&" << Name << ", ";
  OS << "}";
}

/// Emit C++ statements to check the shape of the match and capture
/// instructions into local variables.
void RuleMatcher::emitCxxCaptureStmts(raw_ostream &OS, StringRef Expr) {
  assert(Matchers.size() == 1 && "Cannot handle multi-root matchers yet");
  std::string InsnVarName = defineInsnVar(OS, *Matchers.front(), Expr);
  Matchers.front()->emitCxxCaptureStmts(OS, *this, InsnVarName);
}

void RuleMatcher::emit(raw_ostream &OS,
                       std::map<Record *, SubtargetFeatureInfo, LessRecordByID>
                           SubtargetFeatures) {
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

  OS << "if (";
  OS << "[&]() {\n";
  if (!RequiredFeatures.empty()) {
    OS << "  PredicateBitset ExpectedFeatures = {";
    StringRef Separator = "";
    for (const auto &Predicate : RequiredFeatures) {
      const auto &I = SubtargetFeatures.find(Predicate);
      assert(I != SubtargetFeatures.end() && "Didn't import predicate?");
      OS << Separator << I->second.getEnumBitName();
      Separator = ", ";
    }
    OS << "};\n";
    OS << "if ((AvailableFeatures & ExpectedFeatures) != ExpectedFeatures)\n"
       << "  return false;\n";
  }

  emitCxxCaptureStmts(OS, "I");

  OS << "    if (";
  Matchers.front()->emitCxxPredicateExpr(OS, *this,
                                         getInsnVarName(*Matchers.front()));
  OS << ") {\n";

  // We must also check if it's safe to fold the matched instructions.
  if (InsnVariableNames.size() >= 2) {
    for (const auto &Pair : InsnVariableNames) {
      // Skip the root node since it isn't moving anywhere. Everything else is
      // sinking to meet it.
      if (Pair.first == Matchers.front().get())
        continue;

      // Reject the difficult cases until we have a more accurate check.
      OS << "      if (!isObviouslySafeToFold(" << Pair.second
         << ")) return false;\n";

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

  for (const auto &MA : Actions) {
    MA->emitCxxActionStmts(OS, *this, "I");
  }

  OS << "      constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);\n";
  OS << "      return true;\n";
  OS << "    }\n";
  OS << "    return false;\n";
  OS << "  }()) { return true; }\n\n";
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

//===- GlobalISelEmitter class --------------------------------------------===//

class GlobalISelEmitter {
public:
  explicit GlobalISelEmitter(RecordKeeper &RK);
  void run(raw_ostream &OS);

private:
  const RecordKeeper &RK;
  const CodeGenDAGPatterns CGP;
  const CodeGenTarget &Target;

  /// Keep track of the equivalence between SDNodes and Instruction.
  /// This is defined using 'GINodeEquiv' in the target description.
  DenseMap<Record *, const CodeGenInstruction *> NodeEquivs;

  /// Keep track of the equivalence between ComplexPattern's and
  /// GIComplexOperandMatcher. Map entries are specified by subclassing
  /// GIComplexPatternEquiv.
  DenseMap<const Record *, const Record *> ComplexPatternEquivs;

  // Map of predicates to their subtarget features.
  std::map<Record *, SubtargetFeatureInfo, LessRecordByID> SubtargetFeatures;

  void gatherNodeEquivs();
  const CodeGenInstruction *findNodeEquiv(Record *N) const;

  Error importRulePredicates(RuleMatcher &M, ArrayRef<Init *> Predicates);
  Expected<InstructionMatcher &>
  createAndImportSelDAGMatcher(InstructionMatcher &InsnMatcher,
                               const TreePatternNode *Src) const;
  Error importChildMatcher(InstructionMatcher &InsnMatcher,
                           TreePatternNode *SrcChild, unsigned OpIdx,
                           unsigned &TempOpIdx) const;
  Expected<BuildMIAction &> createAndImportInstructionRenderer(
      RuleMatcher &M, const TreePatternNode *Dst,
      const InstructionMatcher &InsnMatcher) const;
  Error importExplicitUseRenderer(BuildMIAction &DstMIBuilder,
                                  TreePatternNode *DstChild,
                                  const InstructionMatcher &InsnMatcher) const;
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
    : RK(RK), CGP(RK), Target(CGP.getTargetInfo()) {}

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

Expected<InstructionMatcher &> GlobalISelEmitter::createAndImportSelDAGMatcher(
    InstructionMatcher &InsnMatcher, const TreePatternNode *Src) const {
  // Start with the defined operands (i.e., the results of the root operator).
  if (Src->getExtTypes().size() > 1)
    return failedImport("Src pattern has multiple results");

  auto SrcGIOrNull = findNodeEquiv(Src->getOperator());
  if (!SrcGIOrNull)
    return failedImport("Pattern operator lacks an equivalent Instruction" +
                        explainOperator(Src->getOperator()));
  auto &SrcGI = *SrcGIOrNull;

  // The operators look good: match the opcode and mutate it to the new one.
  InsnMatcher.addPredicate<InstructionOpcodeMatcher>(&SrcGI);

  unsigned OpIdx = 0;
  unsigned TempOpIdx = 0;
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

  // Match the used operands (i.e. the children of the operator).
  for (unsigned i = 0, e = Src->getNumChildren(); i != e; ++i) {
    if (auto Error = importChildMatcher(InsnMatcher, Src->getChild(i), OpIdx++,
                                        TempOpIdx))
      return std::move(Error);
  }

  return InsnMatcher;
}

Error GlobalISelEmitter::importChildMatcher(InstructionMatcher &InsnMatcher,
                                            TreePatternNode *SrcChild,
                                            unsigned OpIdx,
                                            unsigned &TempOpIdx) const {
  OperandMatcher &OM =
      InsnMatcher.addOperand(OpIdx, SrcChild->getName(), TempOpIdx);

  if (SrcChild->hasAnyPredicate())
    return failedImport("Src pattern child has predicate (" +
                        explainPredicates(SrcChild) + ")");

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
    return failedImport("Src operand has an unsupported type");
  OM.addPredicate<LLTOperandMatcher>(*OpTyOrNone);

  // Check for nested instructions.
  if (!SrcChild->isLeaf()) {
    // Map the node to a gMIR instruction.
    InstructionOperandMatcher &InsnOperand =
        OM.addPredicate<InstructionOperandMatcher>();
    auto InsnMatcherOrError =
        createAndImportSelDAGMatcher(InsnOperand.getInsnMatcher(), SrcChild);
    if (auto Error = InsnMatcherOrError.takeError())
      return Error;

    return Error::success();
  }

  // Check for constant immediates.
  if (auto *ChildInt = dyn_cast<IntInit>(SrcChild->getLeafValue())) {
    OM.addPredicate<IntOperandMatcher>(ChildInt->getValue());
    return Error::success();
  }

  // Check for def's like register classes or ComplexPattern's.
  if (auto *ChildDefInit = dyn_cast<DefInit>(SrcChild->getLeafValue())) {
    auto *ChildRec = ChildDefInit->getDef();

    // Check for register classes.
    if (ChildRec->isSubClassOf("RegisterClass")) {
      OM.addPredicate<RegisterBankOperandMatcher>(
          Target.getRegisterClass(ChildRec));
      return Error::success();
    }

    if (ChildRec->isSubClassOf("RegisterOperand")) {
      OM.addPredicate<RegisterBankOperandMatcher>(
          Target.getRegisterClass(ChildRec->getValueAsDef("RegClass")));
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
  // The only non-leaf child we accept is 'bb': it's an operator because
  // BasicBlockSDNode isn't inline, but in MI it's just another operand.
  if (!DstChild->isLeaf()) {
    if (DstChild->getOperator()->isSubClassOf("SDNode")) {
      auto &ChildSDNI = CGP.getSDNodeInfo(DstChild->getOperator());
      if (ChildSDNI.getSDClassName() == "BasicBlockSDNode") {
        DstMIBuilder.addRenderer<CopyRenderer>(InsnMatcher,
                                               DstChild->getName());
        return Error::success();
      }
    }
    return failedImport("Dst pattern child isn't a leaf node or an MBB");
  }

  // Otherwise, we're looking for a bog-standard RegisterClass operand.
  if (DstChild->hasAnyPredicate())
    return failedImport("Dst pattern child has predicate (" +
                        explainPredicates(DstChild) + ")");

  if (auto *ChildDefInit = dyn_cast<DefInit>(DstChild->getLeafValue())) {
    auto *ChildRec = ChildDefInit->getDef();

    ArrayRef<EEVT::TypeSet> ChildTypes = DstChild->getExtTypes();
    if (ChildTypes.size() != 1)
      return failedImport("Dst pattern child has multiple results");

    auto OpTyOrNone = MVTToLLT(ChildTypes.front().getConcrete());
    if (!OpTyOrNone)
      return failedImport("Dst operand has an unsupported type");

    if (ChildRec->isSubClassOf("Register")) {
      DstMIBuilder.addRenderer<AddRegisterRenderer>(ChildRec);
      return Error::success();
    }

    if (ChildRec->isSubClassOf("RegisterClass") ||
        ChildRec->isSubClassOf("RegisterOperand")) {
      DstMIBuilder.addRenderer<CopyRenderer>(InsnMatcher, DstChild->getName());
      return Error::success();
    }

    if (ChildRec->isSubClassOf("ComplexPattern")) {
      const auto &ComplexPattern = ComplexPatternEquivs.find(ChildRec);
      if (ComplexPattern == ComplexPatternEquivs.end())
        return failedImport(
            "SelectionDAG ComplexPattern not mapped to GlobalISel");

      const OperandMatcher &OM = InsnMatcher.getOperand(DstChild->getName());
      DstMIBuilder.addRenderer<RenderComplexPatternOperand>(
          *ComplexPattern->second, DstChild->getName(),
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
    const InstructionMatcher &InsnMatcher) const {
  Record *DstOp = Dst->getOperator();
  if (!DstOp->isSubClassOf("Instruction")) {
    if (DstOp->isSubClassOf("ValueType"))
      return failedImport(
          "Pattern operator isn't an instruction (it's a ValueType)");
    return failedImport("Pattern operator isn't an instruction");
  }
  auto &DstI = Target.getInstruction(DstOp);

  auto &DstMIBuilder = M.addAction<BuildMIAction>(&DstI, InsnMatcher);

  // Render the explicit defs.
  for (unsigned I = 0; I < DstI.Operands.NumDefs; ++I) {
    const auto &DstIOperand = DstI.Operands[I];
    DstMIBuilder.addRenderer<CopyRenderer>(InsnMatcher, DstIOperand.Name);
  }

  // Figure out which operands need defaults inserted. Operands that subclass
  // OperandWithDefaultOps are considered from left to right until we have
  // enough operands to render the instruction.
  SmallSet<unsigned, 2> DefaultOperands;
  unsigned DstINumUses = DstI.Operands.size() - DstI.Operands.NumDefs;
  unsigned NumDefaultOperands = 0;
  for (unsigned I = 0; I < DstINumUses &&
                       DstINumUses > Dst->getNumChildren() + NumDefaultOperands;
       ++I) {
    const auto &DstIOperand = DstI.Operands[DstI.Operands.NumDefs + I];
    if (DstIOperand.Rec->isSubClassOf("OperandWithDefaultOps")) {
      DefaultOperands.insert(I);
      NumDefaultOperands +=
          DstIOperand.Rec->getValueAsDag("DefaultOps")->getNumArgs();
    }
  }
  if (DstINumUses > Dst->getNumChildren() + DefaultOperands.size())
    return failedImport("Insufficient operands supplied and default ops "
                        "couldn't make up the shortfall");
  if (DstINumUses < Dst->getNumChildren() + DefaultOperands.size())
    return failedImport("Too many operands supplied");

  // Render the explicit uses.
  unsigned Child = 0;
  for (unsigned I = 0; I != DstINumUses; ++I) {
    // If we need to insert default ops here, then do so.
    if (DefaultOperands.count(I)) {
      const auto &DstIOperand = DstI.Operands[DstI.Operands.NumDefs + I];

      DagInit *DefaultOps = DstIOperand.Rec->getValueAsDag("DefaultOps");
      for (const auto *DefaultOp : DefaultOps->args()) {
        // Look through ValueType operators.
        if (const DagInit *DefaultDagOp = dyn_cast<DagInit>(DefaultOp)) {
          if (const DefInit *DefaultDagOperator =
                  dyn_cast<DefInit>(DefaultDagOp->getOperator())) {
            if (DefaultDagOperator->getDef()->isSubClassOf("ValueType"))
              DefaultOp = DefaultDagOp->getArg(0);
          }
        }

        if (const DefInit *DefaultDefOp = dyn_cast<DefInit>(DefaultOp)) {
          DstMIBuilder.addRenderer<AddRegisterRenderer>(DefaultDefOp->getDef());
          continue;
        }

        if (const IntInit *DefaultIntOp = dyn_cast<IntInit>(DefaultOp)) {
          DstMIBuilder.addRenderer<ImmRenderer>(DefaultIntOp->getValue());
          continue;
        }

        return failedImport("Could not add default op");
      }

      continue;
    }

    if (auto Error = importExplicitUseRenderer(
            DstMIBuilder, Dst->getChild(Child), InsnMatcher))
      return std::move(Error);
    ++Child;
  }

  return DstMIBuilder;
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

  // Start with the defined operands (i.e., the results of the root operator).
  Record *DstOp = Dst->getOperator();
  if (!DstOp->isSubClassOf("Instruction"))
    return failedImport("Pattern operator isn't an instruction");

  auto &DstI = Target.getInstruction(DstOp);
  if (DstI.Operands.NumDefs != Src->getExtTypes().size())
    return failedImport("Src pattern results and dst MI defs are different (" +
                        to_string(Src->getExtTypes().size()) + " def(s) vs " +
                        to_string(DstI.Operands.NumDefs) + " def(s))");

  InstructionMatcher &InsnMatcherTemp = M.addInstructionMatcher();
  auto InsnMatcherOrError = createAndImportSelDAGMatcher(InsnMatcherTemp, Src);
  if (auto Error = InsnMatcherOrError.takeError())
    return std::move(Error);
  InstructionMatcher &InsnMatcher = InsnMatcherOrError.get();

  // The root of the match also has constraints on the register bank so that it
  // matches the result instruction.
  unsigned OpIdx = 0;
  for (const EEVT::TypeSet &Ty : Src->getExtTypes()) {
    (void)Ty;

    const auto &DstIOperand = DstI.Operands[OpIdx];
    Record *DstIOpRec = DstIOperand.Rec;
    if (DstIOpRec->isSubClassOf("RegisterOperand"))
      DstIOpRec = DstIOpRec->getValueAsDef("RegClass");
    if (!DstIOpRec->isSubClassOf("RegisterClass"))
      return failedImport("Dst MI def isn't a register class");

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

  unsigned MaxTemporaries = 0;
  for (const auto &Rule : Rules)
    MaxTemporaries = std::max(MaxTemporaries, Rule.countRendererFns());

  OS << "#ifdef GET_GLOBALISEL_PREDICATE_BITSET\n"
     << "const unsigned MAX_SUBTARGET_PREDICATES = " << SubtargetFeatures.size()
     << ";\n"
     << "using PredicateBitset = "
        "llvm::PredicateBitsetImpl<MAX_SUBTARGET_PREDICATES>;\n"
     << "#endif // ifdef GET_GLOBALISEL_PREDICATE_BITSET\n\n";

  OS << "#ifdef GET_GLOBALISEL_TEMPORARIES_DECL\n";
  for (unsigned I = 0; I < MaxTemporaries; ++I)
    OS << "  mutable ComplexRendererFn Renderer" << I << ";\n";
  OS << "#endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL\n\n";

  OS << "#ifdef GET_GLOBALISEL_TEMPORARIES_INIT\n";
  for (unsigned I = 0; I < MaxTemporaries; ++I)
    OS << ", Renderer" << I << "(nullptr)\n";
  OS << "#endif // ifdef GET_GLOBALISEL_TEMPORARIES_INIT\n\n";

  OS << "#ifdef GET_GLOBALISEL_IMPL\n";
  SubtargetFeatureInfo::emitSubtargetFeatureBitEnumeration(SubtargetFeatures,
                                                           OS);
  SubtargetFeatureInfo::emitNameTable(SubtargetFeatures, OS);
  SubtargetFeatureInfo::emitComputeAvailableFeatures(
      Target.getName(), "InstructionSelector", "computeAvailableFeatures",
      SubtargetFeatures, OS);

  OS << "bool " << Target.getName()
     << "InstructionSelector::selectImpl(MachineInstr &I) const {\n"
     << "  MachineFunction &MF = *I.getParent()->getParent();\n"
     << "  const MachineRegisterInfo &MRI = MF.getRegInfo();\n";

  for (auto &Rule : Rules) {
    Rule.emit(OS, SubtargetFeatures);
    ++NumPatternEmitted;
  }

  OS << "  return false;\n"
     << "}\n"
     << "#endif // ifdef GET_GLOBALISEL_IMPL\n";
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
