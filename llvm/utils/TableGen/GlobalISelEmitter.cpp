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
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
using namespace llvm;

#define DEBUG_TYPE "gisel-emitter"

STATISTIC(NumPatternTotal, "Total number of patterns");
STATISTIC(NumPatternSkipped, "Number of patterns skipped");
STATISTIC(NumPatternEmitted, "Number of patterns emitted");

static cl::opt<bool> WarnOnSkippedPatterns(
    "warn-on-skipped-patterns",
    cl::desc("Explain why a pattern was skipped for inclusion "
             "in the GlobalISel selector"),
    cl::init(false));

namespace {

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

  struct SkipReason {
    std::string Reason;
  };

  /// Analyze pattern \p P, possibly emitting matching code for it to \p OS.
  /// Otherwise, return a reason why this pattern was skipped for emission.
  Optional<SkipReason> runOnPattern(const PatternToMatch &P,
                                    raw_ostream &OS);
};

} // end anonymous namespace

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

struct Matcher {
  virtual ~Matcher() {}
  virtual void emit(raw_ostream &OS) const = 0;
};

raw_ostream &operator<<(raw_ostream &S, const Matcher &M) {
  M.emit(S);
  return S;
}

struct MatchAction {
  virtual ~MatchAction() {}
  virtual void emit(raw_ostream &OS) const = 0;
};

raw_ostream &operator<<(raw_ostream &S, const MatchAction &A) {
  A.emit(S);
  return S;
}

struct MatchOpcode : public Matcher {
  MatchOpcode(const CodeGenInstruction *I) : I(I) {}
  const CodeGenInstruction *I;

  virtual void emit(raw_ostream &OS) const {
    OS << "I.getOpcode() == " << I->Namespace << "::" << I->TheDef->getName();
  }
};

struct MatchRegOpType : public Matcher {
  MatchRegOpType(unsigned OpIdx, std::string Ty)
      : OpIdx(OpIdx), Ty(Ty) {}
  unsigned OpIdx;
  std::string Ty;

  virtual void emit(raw_ostream &OS) const {
    OS << "MRI.getType(I.getOperand(" << OpIdx << ").getReg()) == (" << Ty
       << ")";
  }
};

struct MatchRegOpBank : public Matcher {
  MatchRegOpBank(unsigned OpIdx, const CodeGenRegisterClass &RC)
      : OpIdx(OpIdx), RC(RC) {}
  unsigned OpIdx;
  const CodeGenRegisterClass &RC;

  virtual void emit(raw_ostream &OS) const {
    OS << "(&RBI.getRegBankFromRegClass(" << RC.getQualifiedName()
       << "RegClass) == RBI.getRegBank(I.getOperand(" << OpIdx
       << ").getReg(), MRI, TRI))";
  }
};

struct MatchMBBOp : public Matcher {
  MatchMBBOp(unsigned OpIdx) : OpIdx(OpIdx) {}
  unsigned OpIdx;

  virtual void emit(raw_ostream &OS) const {
    OS << "I.getOperand(" << OpIdx << ").isMBB()";
  }
};

struct MutateOpcode : public MatchAction {
  MutateOpcode(const CodeGenInstruction *I) : I(I) {}
  const CodeGenInstruction *I;

  virtual void emit(raw_ostream &OS) const {
    OS << "I.setDesc(TII.get(" << I->Namespace << "::" << I->TheDef->getName()
       << "));";
  }
};

class MatcherEmitter {
  const PatternToMatch &P;

public:
  std::vector<std::unique_ptr<Matcher>> Matchers;
  std::vector<std::unique_ptr<MatchAction>> Actions;

  MatcherEmitter(const PatternToMatch &P) : P(P) {}

  void emit(raw_ostream &OS) {
    if (Matchers.empty())
      llvm_unreachable("Unexpected empty matcher!");

    OS << "  // Src: " << *P.getSrcPattern() << "\n"
       << "  // Dst: " << *P.getDstPattern() << "\n";

    OS << "  if ((" << *Matchers.front() << ")";
    for (auto &MA : makeArrayRef(Matchers).drop_front())
      OS << " &&\n      (" << *MA << ")";
    OS << ") {\n";

    for (auto &MA : Actions)
      OS << "    " << *MA << "\n";

    OS << "    constrainSelectedInstRegOperands(I, TII, TRI, RBI);\n";
    OS << "    return true;\n";
    OS << "  }\n";
  }
};

//===- GlobalISelEmitter class --------------------------------------------===//

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

Optional<GlobalISelEmitter::SkipReason>
GlobalISelEmitter::runOnPattern(const PatternToMatch &P, raw_ostream &OS) {

  // Keep track of the matchers and actions to emit.
  MatcherEmitter M(P);

  // First, analyze the whole pattern.
  // If the entire pattern has a predicate (e.g., target features), ignore it.
  if (!P.getPredicates()->getValues().empty())
    return SkipReason{"Pattern has a predicate"};

  // Physreg imp-defs require additional logic.  Ignore the pattern.
  if (!P.getDstRegs().empty())
    return SkipReason{"Pattern defines a physical register"};

  // Next, analyze the pattern operators.
  TreePatternNode *Src = P.getSrcPattern();
  TreePatternNode *Dst = P.getDstPattern();

  // If the root of either pattern isn't a simple operator, ignore it.
  if (!isTrivialOperatorNode(Dst))
    return SkipReason{"Dst pattern root isn't a trivial operator"};
  if (!isTrivialOperatorNode(Src))
    return SkipReason{"Src pattern root isn't a trivial operator"};

  Record *DstOp = Dst->getOperator();
  if (!DstOp->isSubClassOf("Instruction"))
    return SkipReason{"Pattern operator isn't an instruction"};

  auto &DstI = Target.getInstruction(DstOp);

  auto SrcGIOrNull = findNodeEquiv(Src->getOperator());
  if (!SrcGIOrNull)
    return SkipReason{"Pattern operator lacks an equivalent Instruction"};
  auto &SrcGI = *SrcGIOrNull;

  // The operators look good: match the opcode and mutate it to the new one.
  M.Matchers.emplace_back(new MatchOpcode(&SrcGI));
  M.Actions.emplace_back(new MutateOpcode(&DstI));

  // Next, analyze the children, only accepting patterns that don't require
  // any change to operands.
  if (Src->getNumChildren() != Dst->getNumChildren())
    return SkipReason{"Src/dst patterns have a different # of children"};

  unsigned OpIdx = 0;

  // Start with the defined operands (i.e., the results of the root operator).
  if (DstI.Operands.NumDefs != Src->getExtTypes().size())
    return SkipReason{"Src pattern results and dst MI defs are different"};

  for (const EEVT::TypeSet &Ty : Src->getExtTypes()) {
    Record *DstIOpRec = DstI.Operands[OpIdx].Rec;
    if (!DstIOpRec->isSubClassOf("RegisterClass"))
      return SkipReason{"Dst MI def isn't a register class"};

    auto OpTyOrNone = MVTToLLT(Ty.getConcrete());
    if (!OpTyOrNone)
      return SkipReason{"Dst operand has an unsupported type"};

    M.Matchers.emplace_back(new MatchRegOpType(OpIdx, *OpTyOrNone));
    M.Matchers.emplace_back(
        new MatchRegOpBank(OpIdx, Target.getRegisterClass(DstIOpRec)));
    ++OpIdx;
  }

  // Finally match the used operands (i.e., the children of the root operator).
  for (unsigned i = 0, e = Src->getNumChildren(); i != e; ++i) {
    auto *SrcChild = Src->getChild(i);
    auto *DstChild = Dst->getChild(i);

    // Patterns can reorder operands.  Ignore those for now.
    if (SrcChild->getName() != DstChild->getName())
      return SkipReason{"Src/dst pattern children not in same order"};

    // The only non-leaf child we accept is 'bb': it's an operator because
    // BasicBlockSDNode isn't inline, but in MI it's just another operand.
    if (!SrcChild->isLeaf()) {
      if (DstChild->isLeaf() ||
          SrcChild->getOperator() != DstChild->getOperator())
        return SkipReason{"Src/dst pattern child operator mismatch"};

      if (SrcChild->getOperator()->isSubClassOf("SDNode")) {
        auto &ChildSDNI = CGP.getSDNodeInfo(SrcChild->getOperator());
        if (ChildSDNI.getSDClassName() == "BasicBlockSDNode") {
          M.Matchers.emplace_back(new MatchMBBOp(OpIdx++));
          continue;
        }
      }
      return SkipReason{"Src pattern child isn't a leaf node"};
    }

    if (SrcChild->getLeafValue() != DstChild->getLeafValue())
      return SkipReason{"Src/dst pattern child leaf mismatch"};

    // Otherwise, we're looking for a bog-standard RegisterClass operand.
    if (SrcChild->hasAnyPredicate())
      return SkipReason{"Src pattern child has predicate"};
    auto *ChildRec = cast<DefInit>(SrcChild->getLeafValue())->getDef();
    if (!ChildRec->isSubClassOf("RegisterClass"))
      return SkipReason{"Src pattern child isn't a RegisterClass"};

    ArrayRef<EEVT::TypeSet> ChildTypes = SrcChild->getExtTypes();
    if (ChildTypes.size() != 1)
      return SkipReason{"Src pattern child has multiple results"};

    auto OpTyOrNone = MVTToLLT(ChildTypes.front().getConcrete());
    if (!OpTyOrNone)
      return SkipReason{"Src operand has an unsupported type"};

    M.Matchers.emplace_back(new MatchRegOpType(OpIdx, *OpTyOrNone));
    M.Matchers.emplace_back(
        new MatchRegOpBank(OpIdx, Target.getRegisterClass(ChildRec)));
    ++OpIdx;
  }

  // We're done with this pattern!  Emit the processed result.
  M.emit(OS);
  ++NumPatternEmitted;
  return None;
}

void GlobalISelEmitter::run(raw_ostream &OS) {
  // Track the GINodeEquiv definitions.
  gatherNodeEquivs();

  emitSourceFileHeader(("Global Instruction Selector for the " +
                       Target.getName() + " target").str(), OS);
  OS << "bool " << Target.getName()
     << "InstructionSelector::selectImpl"
        "(MachineInstr &I) const {\n  const MachineRegisterInfo &MRI = "
        "I.getParent()->getParent()->getRegInfo();\n";

  // Look through the SelectionDAG patterns we found, possibly emitting some.
  for (const PatternToMatch &Pat : CGP.ptms()) {
    ++NumPatternTotal;
    if (auto SkipReason = runOnPattern(Pat, OS)) {
      if (WarnOnSkippedPatterns) {
        PrintWarning(Pat.getSrcRecord()->getLoc(),
                     "Skipped pattern: " + SkipReason->Reason);
      }
      ++NumPatternSkipped;
    }
  }

  OS << "  return false;\n}\n";
}

//===----------------------------------------------------------------------===//

namespace llvm {
void EmitGlobalISel(RecordKeeper &RK, raw_ostream &OS) {
  GlobalISelEmitter(RK).run(OS);
}
} // End llvm namespace
