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
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
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

/// Convert an MVT to an equivalent LLT if possible, or the invalid LLT() for
/// MVTs that don't map cleanly to an LLT (e.g., iPTR, *any, ...).
static Optional<std::string> MVTToLLT(MVT::SimpleValueType SVT) {
  std::string TyStr;
  raw_string_ostream OS(TyStr);
  MVT VT(SVT);
  if (VT.isVector() && VT.getVectorNumElements() != 1) {
    OS << "LLT::vector(" << VT.getVectorNumElements() << ", "
       << VT.getScalarSizeInBits() << ")";
  } else if (VT.isInteger() || VT.isFloatingPoint()) {
    OS << "LLT::scalar(" << VT.getSizeInBits() << ")";
  } else {
    return None;
  }
  OS.flush();
  return TyStr;
}

static bool isTrivialOperatorNode(const TreePatternNode *N) {
  return !N->isLeaf() && !N->hasAnyPredicate() && !N->getTransformFn();
}

//===- Matchers -----------------------------------------------------------===//

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
  /// The relative priority of OPM_LLT, OPM_RegBank, and OPM_MBB do not matter.
  enum PredicateKind {
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
  virtual bool isHigherPriorityThan(const OperandPredicateMatcher &B) {
    return false;
  };
};

/// Generates code to check that an operand is a particular LLT.
class LLTOperandMatcher : public OperandPredicateMatcher {
protected:
  std::string Ty;

public:
  LLTOperandMatcher(std::string Ty)
      : OperandPredicateMatcher(OPM_LLT), Ty(Ty) {}

  static bool classof(const OperandPredicateMatcher *P) {
    return P->getKind() == OPM_LLT;
  }

  void emitCxxPredicateExpr(raw_ostream &OS,
                            StringRef OperandExpr) const override {
    OS << "MRI.getType(" << OperandExpr << ".getReg()) == (" << Ty << ")";
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

/// Generates code to check that a set of predicates match for a particular
/// operand.
class OperandMatcher : public PredicateListMatcher<OperandPredicateMatcher> {
protected:
  unsigned OpIdx;

public:
  OperandMatcher(unsigned OpIdx) : OpIdx(OpIdx) {}
  std::string getOperandExpr(StringRef InsnVarName) const {
    return (InsnVarName + ".getOperand(" + llvm::to_string(OpIdx) + ")").str();
  }

  /// Emit a C++ expression that tests whether the instruction named in
  /// InsnVarName matches all the predicate and all the operands.
  void emitCxxPredicateExpr(raw_ostream &OS, StringRef InsnVarName) const {
    OS << "(/* Operand " << OpIdx << " */ ";
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
  bool isHigherPriorityThan(const InstructionPredicateMatcher &B) const {
    return Kind < B.Kind;
  };
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
  /// Returns true if  is more important than B.
  bool isHigherPriorityThan(const InstructionPredicateMatcher &B) const {
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
  std::vector<OperandMatcher> Operands;

public:
  /// Add an operand to the matcher.
  OperandMatcher &addOperand(unsigned OpIdx) {
    Operands.emplace_back(OpIdx);
    return Operands.back();
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
};

//===- Actions ------------------------------------------------------------===//

/// An action taken when all Matcher predicates succeeded for a parent rule.
///
/// Typical actions include:
/// * Changing the opcode of an instruction.
/// * Adding an operand to an instruction.
class MatchAction {
public:
  virtual ~MatchAction() {}
  virtual void emitCxxActionStmts(raw_ostream &OS) const = 0;
};

/// Generates a comment describing the matched rule being acted upon.
class DebugCommentAction : public MatchAction {
private:
  const PatternToMatch &P;

public:
  DebugCommentAction(const PatternToMatch &P) : P(P) {}

  virtual void emitCxxActionStmts(raw_ostream &OS) const {
    OS << "// " << *P.getSrcPattern() << "  =>  " << *P.getDstPattern();
  }
};

/// Generates code to set the opcode (really, the MCInstrDesc) of a matched
/// instruction to a given Instruction.
class MutateOpcodeAction : public MatchAction {
private:
  const CodeGenInstruction *I;

public:
  MutateOpcodeAction(const CodeGenInstruction *I) : I(I) {}

  virtual void emitCxxActionStmts(raw_ostream &OS) const {
    OS << "I.setDesc(TII.get(" << I->Namespace << "::" << I->TheDef->getName()
       << "));";
  }
};

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

  InstructionMatcher &addInstructionMatcher() {
    Matchers.emplace_back(new InstructionMatcher());
    return *Matchers.back();
  }

  template <class Kind, class... Args>
  Kind &addAction(Args&&... args) {
    Actions.emplace_back(llvm::make_unique<Kind>(std::forward<Args>(args)...));
    return *static_cast<Kind *>(Actions.back().get());
  }

  void emit(raw_ostream &OS) const {
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
      MA->emitCxxActionStmts(OS);
      OS << "\n";
    }

    OS << "    constrainSelectedInstRegOperands(I, TII, TRI, RBI);\n";
    OS << "    return true;\n";
    OS << "  }\n\n";
  }

  /// Compare the priority of this object and B.
  ///
  /// Returns true if this object is more important than B.
  bool isHigherPriorityThan(const RuleMatcher &B) const {
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
  };
};

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

  void gatherNodeEquivs();
  const CodeGenInstruction *findNodeEquiv(Record *N);

  /// Analyze pattern \p P, returning a matcher for it if possible.
  /// Otherwise, return an Error explaining why we don't support it.
  Expected<RuleMatcher> runOnPattern(const PatternToMatch &P);
};

void GlobalISelEmitter::gatherNodeEquivs() {
  assert(NodeEquivs.empty());
  for (Record *Equiv : RK.getAllDerivedDefinitions("GINodeEquiv"))
    NodeEquivs[Equiv->getValueAsDef("Node")] =
        &Target.getInstruction(Equiv->getValueAsDef("I"));
}

const CodeGenInstruction *GlobalISelEmitter::findNodeEquiv(Record *N) {
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
  M.addAction<MutateOpcodeAction>(&DstI);

  // Next, analyze the children, only accepting patterns that don't require
  // any change to operands.
  if (Src->getNumChildren() != Dst->getNumChildren())
    return failedImport("Src/dst patterns have a different # of children");

  unsigned OpIdx = 0;

  // Start with the defined operands (i.e., the results of the root operator).
  if (DstI.Operands.NumDefs != Src->getExtTypes().size())
    return failedImport("Src pattern results and dst MI defs are different");

  for (const EEVT::TypeSet &Ty : Src->getExtTypes()) {
    Record *DstIOpRec = DstI.Operands[OpIdx].Rec;
    if (!DstIOpRec->isSubClassOf("RegisterClass"))
      return failedImport("Dst MI def isn't a register class");

    auto OpTyOrNone = MVTToLLT(Ty.getConcrete());
    if (!OpTyOrNone)
      return failedImport("Dst operand has an unsupported type");

    OperandMatcher &OM = InsnMatcher.addOperand(OpIdx);
    OM.addPredicate<LLTOperandMatcher>(*OpTyOrNone);
    OM.addPredicate<RegisterBankOperandMatcher>(
        Target.getRegisterClass(DstIOpRec));
    ++OpIdx;
  }

  // Finally match the used operands (i.e., the children of the root operator).
  for (unsigned i = 0, e = Src->getNumChildren(); i != e; ++i) {
    auto *SrcChild = Src->getChild(i);
    auto *DstChild = Dst->getChild(i);

    // Patterns can reorder operands.  Ignore those for now.
    if (SrcChild->getName() != DstChild->getName())
      return failedImport("Src/dst pattern children not in same order");

    // The only non-leaf child we accept is 'bb': it's an operator because
    // BasicBlockSDNode isn't inline, but in MI it's just another operand.
    if (!SrcChild->isLeaf()) {
      if (DstChild->isLeaf() ||
          SrcChild->getOperator() != DstChild->getOperator())
        return failedImport("Src/dst pattern child operator mismatch");

      if (SrcChild->getOperator()->isSubClassOf("SDNode")) {
        auto &ChildSDNI = CGP.getSDNodeInfo(SrcChild->getOperator());
        if (ChildSDNI.getSDClassName() == "BasicBlockSDNode") {
          InsnMatcher.addOperand(OpIdx++).addPredicate<MBBOperandMatcher>();
          continue;
        }
      }
      return failedImport("Src pattern child isn't a leaf node");
    }

    if (SrcChild->getLeafValue() != DstChild->getLeafValue())
      return failedImport("Src/dst pattern child leaf mismatch");

    // Otherwise, we're looking for a bog-standard RegisterClass operand.
    if (SrcChild->hasAnyPredicate())
      return failedImport("Src pattern child has predicate");
    auto *ChildRec = cast<DefInit>(SrcChild->getLeafValue())->getDef();
    if (!ChildRec->isSubClassOf("RegisterClass"))
      return failedImport("Src pattern child isn't a RegisterClass");

    ArrayRef<EEVT::TypeSet> ChildTypes = SrcChild->getExtTypes();
    if (ChildTypes.size() != 1)
      return failedImport("Src pattern child has multiple results");

    auto OpTyOrNone = MVTToLLT(ChildTypes.front().getConcrete());
    if (!OpTyOrNone)
      return failedImport("Src operand has an unsupported type");

    OperandMatcher &OM = InsnMatcher.addOperand(OpIdx);
    OM.addPredicate<LLTOperandMatcher>(*OpTyOrNone);
    OM.addPredicate<RegisterBankOperandMatcher>(
        Target.getRegisterClass(ChildRec));
    ++OpIdx;
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
  OS << "bool " << Target.getName()
     << "InstructionSelector::selectImpl"
        "(MachineInstr &I) const {\n  const MachineRegisterInfo &MRI = "
        "I.getParent()->getParent()->getRegInfo();\n\n";

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

  std::sort(Rules.begin(), Rules.end(),
            [](const RuleMatcher &A, const RuleMatcher &B) {
              if (A.isHigherPriorityThan(B)) {
                assert(!B.isHigherPriorityThan(A) && "Cannot be more important "
                                                     "and less important at "
                                                     "the same time");
                return true;
              }
              return false;
            });

  for (const auto &Rule : Rules) {
    Rule.emit(OS);
    ++NumPatternEmitted;
  }

  OS << "  return false;\n}\n";
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//

namespace llvm {
void EmitGlobalISel(RecordKeeper &RK, raw_ostream &OS) {
  GlobalISelEmitter(RK).run(OS);
}
} // End llvm namespace
