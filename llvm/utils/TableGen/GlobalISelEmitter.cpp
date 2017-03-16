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
#include "llvm/ADT/Optional.h"
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

static cl::opt<bool> WarnOnSkippedPatterns(
    "warn-on-skipped-patterns",
    cl::desc("Explain why a pattern was skipped for inclusion "
             "in the GlobalISel selector"),
    cl::init(false));

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
      OS << "LLT::vector(" << Ty.getNumElements() << ", " << Ty.getSizeInBits()
         << ")";
      return;
    }
    llvm_unreachable("Unhandled LLT");
  }

  const LLT &get() const { return Ty; }
};

class InstructionMatcher;
class OperandPlaceholder {
private:
  enum PlaceholderKind {
    OP_MatchReference,
    OP_Temporary,
  } Kind;

  struct MatchReferenceData {
    InstructionMatcher *InsnMatcher;
    StringRef InsnVarName;
    StringRef SymbolicName;
  };

  struct TemporaryData {
    unsigned OpIdx;
  };

  union {
    struct MatchReferenceData MatchReference;
    struct TemporaryData Temporary;
  };

  OperandPlaceholder(PlaceholderKind Kind) : Kind(Kind) {}

public:
  ~OperandPlaceholder() {}

  static OperandPlaceholder
  CreateMatchReference(InstructionMatcher *InsnMatcher,
                       const StringRef InsnVarName, const StringRef SymbolicName) {
    OperandPlaceholder Result(OP_MatchReference);
    Result.MatchReference.InsnMatcher = InsnMatcher;
    Result.MatchReference.InsnVarName = InsnVarName;
    Result.MatchReference.SymbolicName = SymbolicName;
    return Result;
  }

  static OperandPlaceholder CreateTemporary(unsigned OpIdx) {
    OperandPlaceholder Result(OP_Temporary);
    Result.Temporary.OpIdx = OpIdx;
    return Result;
  }

  void emitCxxValueExpr(raw_ostream &OS) const;
};

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

static bool isTrivialOperatorNode(const TreePatternNode *N) {
  return !N->isLeaf() && !N->hasAnyPredicate() && !N->getTransformFn();
}

//===- Matchers -----------------------------------------------------------===//

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

public:
  RuleMatcher() {}

  InstructionMatcher &addInstructionMatcher();

  template <class Kind, class... Args> Kind &addAction(Args &&... args);

  void emit(raw_ostream &OS) const;

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  bool isHigherPriorityThan(const RuleMatcher &B) const;

  /// Report the maximum number of temporary operands needed by the rule
  /// matcher.
  unsigned countTemporaryOperands() const;
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

  /// Emit a C++ expression that checks the predicate for the given operand.
  virtual void emitCxxPredicateExpr(raw_ostream &OS,
                                    StringRef OperandExpr) const = 0;

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  virtual bool isHigherPriorityThan(const OperandPredicateMatcher &B) const {
    return Kind < B.Kind;
  };

  /// Report the maximum number of temporary operands needed by the predicate
  /// matcher.
  virtual unsigned countTemporaryOperands() const { return 0; }
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

  void emitCxxPredicateExpr(raw_ostream &OS,
                            StringRef OperandExpr) const override {
    OS << "MRI.getType(" << OperandExpr << ".getReg()) == (";
    Ty.emitCxxConstructorCall(OS);
    OS << ")";
  }
};

/// Generates code to check that an operand is a particular target constant.
class ComplexPatternOperandMatcher : public OperandPredicateMatcher {
protected:
  const Record &TheDef;
  /// The index of the first temporary operand to allocate to this
  /// ComplexPattern.
  unsigned BaseTemporaryID;

  unsigned getNumOperands() const {
    return TheDef.getValueAsDag("Operands")->getNumArgs();
  }

public:
  ComplexPatternOperandMatcher(const Record &TheDef, unsigned BaseTemporaryID)
      : OperandPredicateMatcher(OPM_ComplexPattern), TheDef(TheDef),
        BaseTemporaryID(BaseTemporaryID) {}

  void emitCxxPredicateExpr(raw_ostream &OS,
                            StringRef OperandExpr) const override {
    OS << TheDef.getValueAsString("MatcherFn") << "(" << OperandExpr;
    for (unsigned I = 0; I < getNumOperands(); ++I) {
      OS << ", ";
      OperandPlaceholder::CreateTemporary(BaseTemporaryID + I)
          .emitCxxValueExpr(OS);
    }
    OS << ")";
  }

  unsigned countTemporaryOperands() const override {
    return getNumOperands();
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

  void emitCxxPredicateExpr(raw_ostream &OS,
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

  void emitCxxPredicateExpr(raw_ostream &OS,
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

  void emitCxxPredicateExpr(raw_ostream &OS,
                            StringRef OperandExpr) const override {
    OS << "isOperandImmEqual(" << OperandExpr << ", " << Value << ", MRI)";
  }
};

/// Generates code to check that a set of predicates match for a particular
/// operand.
class OperandMatcher : public PredicateListMatcher<OperandPredicateMatcher> {
protected:
  unsigned OpIdx;
  std::string SymbolicName;

public:
  OperandMatcher(unsigned OpIdx, const std::string &SymbolicName)
      : OpIdx(OpIdx), SymbolicName(SymbolicName) {}

  bool hasSymbolicName() const { return !SymbolicName.empty(); }
  const StringRef getSymbolicName() const { return SymbolicName; }
  unsigned getOperandIndex() const { return OpIdx; }

  std::string getOperandExpr(const StringRef InsnVarName) const {
    return (InsnVarName + ".getOperand(" + llvm::to_string(OpIdx) + ")").str();
  }

  /// Emit a C++ expression that tests whether the instruction named in
  /// InsnVarName matches all the predicate and all the operands.
  void emitCxxPredicateExpr(raw_ostream &OS, const StringRef InsnVarName) const {
    OS << "(/* ";
    if (SymbolicName.empty())
      OS << "Operand " << OpIdx;
    else
      OS << SymbolicName;
    OS << " */ ";
    emitCxxPredicateListExpr(OS, getOperandExpr(InsnVarName));
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
  unsigned countTemporaryOperands() const {
    return std::accumulate(
        predicates().begin(), predicates().end(), 0,
        [](unsigned A,
           const std::unique_ptr<OperandPredicateMatcher> &Predicate) {
          return A + Predicate->countTemporaryOperands();
        });
  }
};

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
  virtual void emitCxxPredicateExpr(raw_ostream &OS,
                                    StringRef InsnVarName) const = 0;

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  virtual bool isHigherPriorityThan(const InstructionPredicateMatcher &B) const {
    return Kind < B.Kind;
  };

  /// Report the maximum number of temporary operands needed by the predicate
  /// matcher.
  virtual unsigned countTemporaryOperands() const { return 0; }
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

  void emitCxxPredicateExpr(raw_ostream &OS,
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
  typedef std::vector<OperandMatcher> OperandVec;

  /// The operands to match. All rendered operands must be present even if the
  /// condition is always true.
  OperandVec Operands;

public:
  /// Add an operand to the matcher.
  OperandMatcher &addOperand(unsigned OpIdx, const std::string &SymbolicName) {
    Operands.emplace_back(OpIdx, SymbolicName);
    return Operands.back();
  }

  const OperandMatcher &getOperand(const StringRef SymbolicName) const {
    assert(!SymbolicName.empty() && "Cannot lookup unnamed operand");
    const auto &I = std::find_if(Operands.begin(), Operands.end(),
                                 [&SymbolicName](const OperandMatcher &X) {
                                   return X.getSymbolicName() == SymbolicName;
                                 });
    if (I != Operands.end())
      return *I;
    llvm_unreachable("Failed to lookup operand");
  }

  unsigned getNumOperands() const { return Operands.size(); }
  OperandVec::const_iterator operands_begin() const {
    return Operands.begin();
  }
  OperandVec::const_iterator operands_end() const {
    return Operands.end();
  }
  iterator_range<OperandVec::const_iterator> operands() const {
    return make_range(operands_begin(), operands_end());
  }

  /// Emit a C++ expression that tests whether the instruction named in
  /// InsnVarName matches all the predicates and all the operands.
  void emitCxxPredicateExpr(raw_ostream &OS, StringRef InsnVarName) const {
    emitCxxPredicateListExpr(OS, InsnVarName);
    for (const auto &Operand : Operands) {
      OS << " &&\n(";
      Operand.emitCxxPredicateExpr(OS, InsnVarName);
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
      if (std::get<0>(Operand).isHigherPriorityThan(std::get<1>(Operand)))
        return true;
      if (std::get<1>(Operand).isHigherPriorityThan(std::get<0>(Operand)))
        return false;
    }

    return false;
  };

  /// Report the maximum number of temporary operands needed by the instruction
  /// matcher.
  unsigned countTemporaryOperands() const {
    return std::accumulate(predicates().begin(), predicates().end(), 0,
                           [](unsigned A,
                              const std::unique_ptr<InstructionPredicateMatcher>
                                  &Predicate) {
                             return A + Predicate->countTemporaryOperands();
                           }) +
           std::accumulate(Operands.begin(), Operands.end(), 0,
                           [](unsigned A, const OperandMatcher &Operand) {
                             return A + Operand.countTemporaryOperands();
                           });
  }
};

//===- Actions ------------------------------------------------------------===//
void OperandPlaceholder::emitCxxValueExpr(raw_ostream &OS) const {
  switch (Kind) {
  case OP_MatchReference:
    OS << MatchReference.InsnMatcher->getOperand(MatchReference.SymbolicName)
              .getOperandExpr(MatchReference.InsnVarName);
    break;
  case OP_Temporary:
    OS << "TempOp" << Temporary.OpIdx;
    break;
  }
}

class OperandRenderer {
public:
  enum RendererKind { OR_Copy, OR_Register, OR_ComplexPattern };

protected:
  RendererKind Kind;

public:
  OperandRenderer(RendererKind Kind) : Kind(Kind) {}
  virtual ~OperandRenderer() {}

  RendererKind getKind() const { return Kind; }

  virtual void emitCxxRenderStmts(raw_ostream &OS) const = 0;
};

/// A CopyRenderer emits code to copy a single operand from an existing
/// instruction to the one being built.
class CopyRenderer : public OperandRenderer {
protected:
  /// The matcher for the instruction that this operand is copied from.
  /// This provides the facility for looking up an a operand by it's name so
  /// that it can be used as a source for the instruction being built.
  const InstructionMatcher &Matched;
  /// The name of the instruction to copy from.
  const StringRef InsnVarName;
  /// The name of the operand.
  const StringRef SymbolicName;

public:
  CopyRenderer(const InstructionMatcher &Matched, const StringRef InsnVarName,
               const StringRef SymbolicName)
      : OperandRenderer(OR_Copy), Matched(Matched), InsnVarName(InsnVarName),
        SymbolicName(SymbolicName) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_Copy;
  }

  const StringRef getSymbolicName() const { return SymbolicName; }

  void emitCxxRenderStmts(raw_ostream &OS) const override {
    std::string OperandExpr =
        Matched.getOperand(SymbolicName).getOperandExpr(InsnVarName);
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

  void emitCxxRenderStmts(raw_ostream &OS) const override {
    OS << "    MIB.addReg(" << RegisterDef->getValueAsString("Namespace")
       << "::" << RegisterDef->getName() << ");\n";
  }
};

class RenderComplexPatternOperand : public OperandRenderer {
private:
  const Record &TheDef;
  std::vector<OperandPlaceholder> Sources;

  unsigned getNumOperands() const {
    return TheDef.getValueAsDag("Operands")->getNumArgs();
  }

public:
  RenderComplexPatternOperand(const Record &TheDef,
                              const ArrayRef<OperandPlaceholder> Sources)
      : OperandRenderer(OR_ComplexPattern), TheDef(TheDef), Sources(Sources) {}

  static bool classof(const OperandRenderer *R) {
    return R->getKind() == OR_ComplexPattern;
  }

  void emitCxxRenderStmts(raw_ostream &OS) const override {
    assert(Sources.size() == getNumOperands() && "Inconsistent number of operands");
    for (const auto &Source : Sources) {
      OS << "MIB.add(";
      Source.emitCxxValueExpr(OS);
      OS << ");\n";
    }
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
  /// \param InsnVarName If given, it's an instruction to recycle. The
  ///                    requirements on the instruction vary from action to
  ///                    action.
  virtual void emitCxxActionStmts(raw_ostream &OS,
                                  const StringRef InsnVarName) const = 0;
};

/// Generates a comment describing the matched rule being acted upon.
class DebugCommentAction : public MatchAction {
private:
  const PatternToMatch &P;

public:
  DebugCommentAction(const PatternToMatch &P) : P(P) {}

  void emitCxxActionStmts(raw_ostream &OS,
                          const StringRef InsnVarName) const override {
    OS << "// " << *P.getSrcPattern() << "  =>  " << *P.getDstPattern();
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
        if (Matched.getOperand(Copy->getSymbolicName()).getOperandIndex() !=
            Renderer.index())
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

  virtual void emitCxxActionStmts(raw_ostream &OS,
                                  const StringRef InsnVarName) const {
    if (canMutate()) {
      OS << "I.setDesc(TII.get(" << I->Namespace << "::" << I->TheDef->getName()
         << "));\n";
      OS << "    MachineInstr &NewI = I;\n";
      return;
    }

    // TODO: Simple permutation looks like it could be almost as common as
    //       mutation due to commutative operations.

    OS << "MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, "
          "I.getDebugLoc(), TII.get("
       << I->Namespace << "::" << I->TheDef->getName() << "));\n";
    for (const auto &Renderer : OperandRenderers)
      Renderer->emitCxxRenderStmts(OS);
    OS << "    MIB.setMemRefs(I.memoperands_begin(), I.memoperands_end());\n";
    OS << "    " << InsnVarName << ".eraseFromParent();\n";
    OS << "    MachineInstr &NewI = *MIB;\n";
  }
};

InstructionMatcher &RuleMatcher::addInstructionMatcher() {
  Matchers.emplace_back(new InstructionMatcher());
  return *Matchers.back();
}

template <class Kind, class... Args>
Kind &RuleMatcher::addAction(Args &&... args) {
  Actions.emplace_back(llvm::make_unique<Kind>(std::forward<Args>(args)...));
  return *static_cast<Kind *>(Actions.back().get());
}

void RuleMatcher::emit(raw_ostream &OS) const {
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
  OS << "  if (";
  Matchers.front()->emitCxxPredicateExpr(OS, "I");
  OS << ") {\n";

  for (const auto &MA : Actions) {
    OS << "    ";
    MA->emitCxxActionStmts(OS, "I");
    OS << "\n";
  }

  OS << "    constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);\n";
  OS << "    return true;\n";
  OS << "  }\n\n";
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

unsigned RuleMatcher::countTemporaryOperands() const {
  return std::accumulate(
      Matchers.begin(), Matchers.end(), 0,
      [](unsigned A, const std::unique_ptr<InstructionMatcher> &Matcher) {
        return A + Matcher->countTemporaryOperands();
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

  void gatherNodeEquivs();
  const CodeGenInstruction *findNodeEquiv(Record *N) const;

  /// Analyze pattern \p P, returning a matcher for it if possible.
  /// Otherwise, return an Error explaining why we don't support it.
  Expected<RuleMatcher> runOnPattern(const PatternToMatch &P);
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

/// Helper function to let the emitter report skip reason error messages.
static Error failedImport(const Twine &Reason) {
  return make_error<StringError>(Reason, inconvertibleErrorCode());
}

Expected<RuleMatcher> GlobalISelEmitter::runOnPattern(const PatternToMatch &P) {
  unsigned TempOpIdx = 0;

  // Keep track of the matchers and actions to emit.
  RuleMatcher M;
  M.addAction<DebugCommentAction>(P);

  // First, analyze the whole pattern.
  // If the entire pattern has a predicate (e.g., target features), ignore it.
  if (!P.getPredicates()->getValues().empty())
    return failedImport("Pattern has a predicate");

  // Physreg imp-defs require additional logic.  Ignore the pattern.
  if (!P.getDstRegs().empty())
    return failedImport("Pattern defines a physical register");

  // Next, analyze the pattern operators.
  TreePatternNode *Src = P.getSrcPattern();
  TreePatternNode *Dst = P.getDstPattern();

  // If the root of either pattern isn't a simple operator, ignore it.
  if (!isTrivialOperatorNode(Dst))
    return failedImport("Dst pattern root isn't a trivial operator");
  if (!isTrivialOperatorNode(Src))
    return failedImport("Src pattern root isn't a trivial operator");

  Record *DstOp = Dst->getOperator();
  if (!DstOp->isSubClassOf("Instruction"))
    return failedImport("Pattern operator isn't an instruction");

  auto &DstI = Target.getInstruction(DstOp);

  auto SrcGIOrNull = findNodeEquiv(Src->getOperator());
  if (!SrcGIOrNull)
    return failedImport("Pattern operator lacks an equivalent Instruction");
  auto &SrcGI = *SrcGIOrNull;

  // The operators look good: match the opcode and mutate it to the new one.
  InstructionMatcher &InsnMatcher = M.addInstructionMatcher();
  InsnMatcher.addPredicate<InstructionOpcodeMatcher>(&SrcGI);
  auto &DstMIBuilder = M.addAction<BuildMIAction>(&DstI, InsnMatcher);

  // Next, analyze the children, only accepting patterns that don't require
  // any change to operands.
  if (Src->getNumChildren() != Dst->getNumChildren())
    return failedImport("Src/dst patterns have a different # of children");

  unsigned OpIdx = 0;

  // Start with the defined operands (i.e., the results of the root operator).
  if (DstI.Operands.NumDefs != Src->getExtTypes().size())
    return failedImport("Src pattern results and dst MI defs are different");

  for (const EEVT::TypeSet &Ty : Src->getExtTypes()) {
    const auto &DstIOperand = DstI.Operands[OpIdx];
    Record *DstIOpRec = DstIOperand.Rec;
    if (!DstIOpRec->isSubClassOf("RegisterClass"))
      return failedImport("Dst MI def isn't a register class");

    auto OpTyOrNone = MVTToLLT(Ty.getConcrete());
    if (!OpTyOrNone)
      return failedImport("Dst operand has an unsupported type");

    OperandMatcher &OM = InsnMatcher.addOperand(OpIdx, DstIOperand.Name);
    OM.addPredicate<LLTOperandMatcher>(*OpTyOrNone);
    OM.addPredicate<RegisterBankOperandMatcher>(
        Target.getRegisterClass(DstIOpRec));
    DstMIBuilder.addRenderer<CopyRenderer>(InsnMatcher, "I", DstIOperand.Name);
    ++OpIdx;
  }

  // Finally match the used operands (i.e., the children of the root operator).
  for (unsigned i = 0, e = Src->getNumChildren(); i != e; ++i) {
    auto *SrcChild = Src->getChild(i);

    OperandMatcher &OM = InsnMatcher.addOperand(OpIdx++, SrcChild->getName());

    // The only non-leaf child we accept is 'bb': it's an operator because
    // BasicBlockSDNode isn't inline, but in MI it's just another operand.
    if (!SrcChild->isLeaf()) {
      if (SrcChild->getOperator()->isSubClassOf("SDNode")) {
        auto &ChildSDNI = CGP.getSDNodeInfo(SrcChild->getOperator());
        if (ChildSDNI.getSDClassName() == "BasicBlockSDNode") {
          OM.addPredicate<MBBOperandMatcher>();
          continue;
        }
      }
      return failedImport("Src pattern child isn't a leaf node or an MBB");
    }

    if (SrcChild->hasAnyPredicate())
      return failedImport("Src pattern child has predicate");

    ArrayRef<EEVT::TypeSet> ChildTypes = SrcChild->getExtTypes();
    if (ChildTypes.size() != 1)
      return failedImport("Src pattern child has multiple results");

    auto OpTyOrNone = MVTToLLT(ChildTypes.front().getConcrete());
    if (!OpTyOrNone)
      return failedImport("Src operand has an unsupported type");
    OM.addPredicate<LLTOperandMatcher>(*OpTyOrNone);

    if (auto *ChildInt = dyn_cast<IntInit>(SrcChild->getLeafValue())) {
      OM.addPredicate<IntOperandMatcher>(ChildInt->getValue());
      continue;
    }

    if (auto *ChildDefInit = dyn_cast<DefInit>(SrcChild->getLeafValue())) {
      auto *ChildRec = ChildDefInit->getDef();

      if (ChildRec->isSubClassOf("RegisterClass")) {
        OM.addPredicate<RegisterBankOperandMatcher>(
            Target.getRegisterClass(ChildRec));
        continue;
      }

      if (ChildRec->isSubClassOf("ComplexPattern")) {
        const auto &ComplexPattern = ComplexPatternEquivs.find(ChildRec);
        if (ComplexPattern == ComplexPatternEquivs.end())
          return failedImport(
              "SelectionDAG ComplexPattern not mapped to GlobalISel");

        const auto &Predicate = OM.addPredicate<ComplexPatternOperandMatcher>(
            *ComplexPattern->second, TempOpIdx);
        TempOpIdx += Predicate.countTemporaryOperands();
        continue;
      }

      return failedImport(
          "Src pattern child def is an unsupported tablegen class");
    }

    return failedImport("Src pattern child is an unsupported kind");
  }

  TempOpIdx = 0;
  // Finally render the used operands (i.e., the children of the root operator).
  for (unsigned i = 0, e = Dst->getNumChildren(); i != e; ++i) {
    auto *DstChild = Dst->getChild(i);

    // The only non-leaf child we accept is 'bb': it's an operator because
    // BasicBlockSDNode isn't inline, but in MI it's just another operand.
    if (!DstChild->isLeaf()) {
      if (DstChild->getOperator()->isSubClassOf("SDNode")) {
        auto &ChildSDNI = CGP.getSDNodeInfo(DstChild->getOperator());
        if (ChildSDNI.getSDClassName() == "BasicBlockSDNode") {
          DstMIBuilder.addRenderer<CopyRenderer>(InsnMatcher, "I",
                                                 DstChild->getName());
          continue;
        }
      }
      return failedImport("Dst pattern child isn't a leaf node or an MBB");
    }

    // Otherwise, we're looking for a bog-standard RegisterClass operand.
    if (DstChild->hasAnyPredicate())
      return failedImport("Dst pattern child has predicate");

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
        continue;
      }

      if (ChildRec->isSubClassOf("RegisterClass")) {
        DstMIBuilder.addRenderer<CopyRenderer>(InsnMatcher, "I",
                                               DstChild->getName());
        continue;
      }

      if (ChildRec->isSubClassOf("ComplexPattern")) {
        const auto &ComplexPattern = ComplexPatternEquivs.find(ChildRec);
        if (ComplexPattern == ComplexPatternEquivs.end())
          return failedImport(
              "SelectionDAG ComplexPattern not mapped to GlobalISel");

        SmallVector<OperandPlaceholder, 2> RenderedOperands;
        for (unsigned I = 0; I < InsnMatcher.getOperand(DstChild->getName())
                                     .countTemporaryOperands();
             ++I) {
          RenderedOperands.push_back(OperandPlaceholder::CreateTemporary(I));
          TempOpIdx++;
        }
        DstMIBuilder.addRenderer<RenderComplexPatternOperand>(
            *ComplexPattern->second,
            RenderedOperands);
        continue;
      }

      return failedImport(
          "Dst pattern child def is an unsupported tablegen class");
    }

    return failedImport("Dst pattern child is an unsupported kind");
  }

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
    MaxTemporaries = std::max(MaxTemporaries, Rule.countTemporaryOperands());

  OS << "#ifdef GET_GLOBALISEL_TEMPORARIES_DECL\n";
  for (unsigned I = 0; I < MaxTemporaries; ++I)
    OS << "  mutable MachineOperand TempOp" << I << ";\n";
  OS << "#endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL\n\n";

  OS << "#ifdef GET_GLOBALISEL_TEMPORARIES_INIT\n";
  for (unsigned I = 0; I < MaxTemporaries; ++I)
    OS << ", TempOp" << I << "(MachineOperand::CreatePlaceholder())\n";
  OS << "#endif // ifdef GET_GLOBALISEL_TEMPORARIES_INIT\n\n";

  OS << "#ifdef GET_GLOBALISEL_IMPL\n"
     << "bool " << Target.getName()
     << "InstructionSelector::selectImpl(MachineInstr &I) const {\n"
     << "  MachineFunction &MF = *I.getParent()->getParent();\n"
     << "  const MachineRegisterInfo &MRI = MF.getRegInfo();\n";

  for (const auto &Rule : Rules) {
    Rule.emit(OS);
    ++NumPatternEmitted;
  }

  OS << "  return false;\n"
     << "}\n"
     << "#endif // ifdef GET_GLOBALISEL_IMPL\n";
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//

namespace llvm {
void EmitGlobalISel(RecordKeeper &RK, raw_ostream &OS) {
  GlobalISelEmitter(RK).run(OS);
}
} // End llvm namespace
