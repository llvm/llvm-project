
//===- GlobalISelEmitter.cpp - Generate an instruction selector -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This tablegen backend emits code for use by the GlobalISel instruction
/// selector. See include/llvm/Target/GlobalISel/Target.td.
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

#include "Basic/CodeGenIntrinsics.h"
#include "Common/CodeGenDAGPatterns.h"
#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenRegisters.h"
#include "Common/CodeGenTarget.h"
#include "Common/GlobalISel/GlobalISelMatchTable.h"
#include "Common/GlobalISel/GlobalISelMatchTableExecutorEmitter.h"
#include "Common/InfoByHwMode.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/Support/CodeGenCoverage.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>

using namespace llvm;
using namespace llvm::gi;

using action_iterator = RuleMatcher::action_iterator;

#define DEBUG_TYPE "gisel-emitter"

STATISTIC(NumPatternTotal, "Total number of patterns");
STATISTIC(NumPatternImported, "Number of patterns imported from SelectionDAG");
STATISTIC(NumPatternImportsSkipped, "Number of SelectionDAG imports skipped");
STATISTIC(NumPatternsTested,
          "Number of patterns executed according to coverage information");

static cl::OptionCategory GlobalISelEmitterCat("Options for -gen-global-isel");

static cl::opt<bool> WarnOnSkippedPatterns(
    "warn-on-skipped-patterns",
    cl::desc("Explain why a pattern was skipped for inclusion "
             "in the GlobalISel selector"),
    cl::init(false), cl::cat(GlobalISelEmitterCat));

static cl::opt<bool> GenerateCoverage(
    "instrument-gisel-coverage",
    cl::desc("Generate coverage instrumentation for GlobalISel"),
    cl::init(false), cl::cat(GlobalISelEmitterCat));

static cl::opt<std::string> UseCoverageFile(
    "gisel-coverage-file", cl::init(""),
    cl::desc("Specify file to retrieve coverage information from"),
    cl::cat(GlobalISelEmitterCat));

static cl::opt<bool> OptimizeMatchTable(
    "optimize-match-table",
    cl::desc("Generate an optimized version of the match table"),
    cl::init(true), cl::cat(GlobalISelEmitterCat));

namespace {

static std::string explainPredicates(const TreePatternNode &N) {
  std::string Explanation;
  StringRef Separator = "";
  for (const TreePredicateCall &Call : N.getPredicateCalls()) {
    const TreePredicateFn &P = Call.Fn;
    Explanation +=
        (Separator + P.getOrigPatFragRecord()->getRecord()->getName()).str();
    Separator = ", ";

    if (P.isAlwaysTrue())
      Explanation += " always-true";
    if (P.isImmediatePattern())
      Explanation += " immediate";

    if (P.isUnindexed())
      Explanation += " unindexed";

    if (P.isNonExtLoad())
      Explanation += " non-extload";
    if (P.isAnyExtLoad())
      Explanation += " extload";
    if (P.isSignExtLoad())
      Explanation += " sextload";
    if (P.isZeroExtLoad())
      Explanation += " zextload";

    if (P.isNonTruncStore())
      Explanation += " non-truncstore";
    if (P.isTruncStore())
      Explanation += " truncstore";

    if (const Record *VT = P.getMemoryVT())
      Explanation += (" MemVT=" + VT->getName()).str();
    if (const Record *VT = P.getScalarMemoryVT())
      Explanation += (" ScalarVT(MemVT)=" + VT->getName()).str();

    if (const ListInit *AddrSpaces = P.getAddressSpaces()) {
      raw_string_ostream OS(Explanation);
      OS << " AddressSpaces=[";

      StringRef AddrSpaceSeparator;
      for (const Init *Val : AddrSpaces->getElements()) {
        const IntInit *IntVal = dyn_cast<IntInit>(Val);
        if (!IntVal)
          continue;

        OS << AddrSpaceSeparator << IntVal->getValue();
        AddrSpaceSeparator = ", ";
      }

      OS << ']';
    }

    int64_t MinAlign = P.getMinAlignment();
    if (MinAlign > 0)
      Explanation += " MinAlign=" + utostr(MinAlign);

    if (P.isAtomicOrderingMonotonic())
      Explanation += " monotonic";
    if (P.isAtomicOrderingAcquire())
      Explanation += " acquire";
    if (P.isAtomicOrderingRelease())
      Explanation += " release";
    if (P.isAtomicOrderingAcquireRelease())
      Explanation += " acq_rel";
    if (P.isAtomicOrderingSequentiallyConsistent())
      Explanation += " seq_cst";
    if (P.isAtomicOrderingAcquireOrStronger())
      Explanation += " >=acquire";
    if (P.isAtomicOrderingWeakerThanAcquire())
      Explanation += " <acquire";
    if (P.isAtomicOrderingReleaseOrStronger())
      Explanation += " >=release";
    if (P.isAtomicOrderingWeakerThanRelease())
      Explanation += " <release";
  }
  return Explanation;
}

std::string explainOperator(const Record *Operator) {
  if (Operator->isSubClassOf("SDNode"))
    return (" (" + Operator->getValueAsString("Opcode") + ")").str();

  if (Operator->isSubClassOf("Intrinsic"))
    return (" (Operator is an Intrinsic, " + Operator->getName() + ")").str();

  if (Operator->isSubClassOf("ComplexPattern"))
    return (" (Operator is an unmapped ComplexPattern, " + Operator->getName() +
            ")")
        .str();

  if (Operator->isSubClassOf("SDNodeXForm"))
    return (" (Operator is an unmapped SDNodeXForm, " + Operator->getName() +
            ")")
        .str();

  return (" (Operator " + Operator->getName() + " not understood)").str();
}

/// Helper function to let the emitter report skip reason error messages.
static Error failedImport(const Twine &Reason) {
  return make_error<StringError>(Reason, inconvertibleErrorCode());
}

static Error isTrivialOperatorNode(const TreePatternNode &N) {
  std::string Explanation;
  std::string Separator;

  bool HasUnsupportedPredicate = false;
  for (const TreePredicateCall &Call : N.getPredicateCalls()) {
    const TreePredicateFn &Predicate = Call.Fn;

    if (Predicate.isAlwaysTrue())
      continue;

    if (Predicate.isImmediatePattern())
      continue;

    if (Predicate.hasNoUse() || Predicate.hasOneUse())
      continue;

    if (Predicate.isNonExtLoad() || Predicate.isAnyExtLoad() ||
        Predicate.isSignExtLoad() || Predicate.isZeroExtLoad())
      continue;

    if (Predicate.isNonTruncStore() || Predicate.isTruncStore())
      continue;

    if (Predicate.isLoad() && Predicate.getMemoryVT())
      continue;

    if (Predicate.isStore() && Predicate.getMemoryVT())
      continue;

    if (Predicate.isLoad() || Predicate.isStore()) {
      if (Predicate.isUnindexed())
        continue;
    }

    if (Predicate.isLoad() || Predicate.isStore() || Predicate.isAtomic()) {
      const ListInit *AddrSpaces = Predicate.getAddressSpaces();
      if (AddrSpaces && !AddrSpaces->empty())
        continue;

      if (Predicate.getMinAlignment() > 0)
        continue;
    }

    if (Predicate.isAtomic() && Predicate.getMemoryVT())
      continue;

    if (Predicate.isAtomic() &&
        (Predicate.isAtomicOrderingMonotonic() ||
         Predicate.isAtomicOrderingAcquire() ||
         Predicate.isAtomicOrderingRelease() ||
         Predicate.isAtomicOrderingAcquireRelease() ||
         Predicate.isAtomicOrderingSequentiallyConsistent() ||
         Predicate.isAtomicOrderingAcquireOrStronger() ||
         Predicate.isAtomicOrderingWeakerThanAcquire() ||
         Predicate.isAtomicOrderingReleaseOrStronger() ||
         Predicate.isAtomicOrderingWeakerThanRelease()))
      continue;

    if (Predicate.hasGISelPredicateCode())
      continue;

    HasUnsupportedPredicate = true;
    Explanation = Separator + "Has a predicate (" + explainPredicates(N) + ")";
    Separator = ", ";
    Explanation += (Separator + "first-failing:" +
                    Predicate.getOrigPatFragRecord()->getRecord()->getName())
                       .str();
    break;
  }

  if (!HasUnsupportedPredicate)
    return Error::success();

  return failedImport(Explanation);
}

static const Record *getInitValueAsRegClass(const Init *V) {
  if (const DefInit *VDefInit = dyn_cast<DefInit>(V)) {
    if (VDefInit->getDef()->isSubClassOf("RegisterOperand"))
      return VDefInit->getDef()->getValueAsDef("RegClass");
    if (VDefInit->getDef()->isSubClassOf("RegisterClass"))
      return VDefInit->getDef();
  }
  return nullptr;
}

static std::string getScopedName(unsigned Scope, const std::string &Name) {
  return ("pred:" + Twine(Scope) + ":" + Name).str();
}

static std::string getMangledRootDefName(StringRef DefOperandName) {
  return ("DstI[" + DefOperandName + "]").str();
}

//===- GlobalISelEmitter class --------------------------------------------===//

static Expected<LLTCodeGen> getInstResultType(const TreePatternNode &Dst,
                                              const CodeGenTarget &Target) {
  // While we allow more than one output (both implicit and explicit defs)
  // below, we only expect one explicit def here.
  assert(Dst.getOperator()->isSubClassOf("Instruction"));
  CodeGenInstruction &InstInfo = Target.getInstruction(Dst.getOperator());
  if (!InstInfo.Operands.NumDefs)
    return failedImport("Dst pattern child needs a def");

  ArrayRef<TypeSetByHwMode> ChildTypes = Dst.getExtTypes();
  if (ChildTypes.size() < 1)
    return failedImport("Dst pattern child has no result");

  // If there are multiple results, just take the first one (this is how
  // SelectionDAG does it).
  std::optional<LLTCodeGen> MaybeOpTy;
  if (ChildTypes.front().isMachineValueType()) {
    MaybeOpTy = MVTToLLT(ChildTypes.front().getMachineValueType().SimpleTy);
  }

  if (!MaybeOpTy)
    return failedImport("Dst operand has an unsupported type");
  return *MaybeOpTy;
}

class GlobalISelEmitter final : public GlobalISelMatchTableExecutorEmitter {
public:
  explicit GlobalISelEmitter(const RecordKeeper &RK);

  void emitAdditionalImpl(raw_ostream &OS) override;

  void emitMIPredicateFns(raw_ostream &OS) override;
  void emitLeafPredicateFns(raw_ostream &OS) override;
  void emitI64ImmPredicateFns(raw_ostream &OS) override;
  void emitAPFloatImmPredicateFns(raw_ostream &OS) override;
  void emitAPIntImmPredicateFns(raw_ostream &OS) override;
  void emitTestSimplePredicate(raw_ostream &OS) override;
  void emitRunCustomAction(raw_ostream &OS) override;

  const CodeGenTarget &getTarget() const override { return Target; }
  StringRef getClassName() const override { return ClassName; }

  void run(raw_ostream &OS);

private:
  std::string ClassName;

  const RecordKeeper &RK;
  const CodeGenDAGPatterns CGP;
  const CodeGenTarget &Target;
  CodeGenRegBank &CGRegs;

  ArrayRef<const Record *> AllPatFrags;

  /// Keep track of the equivalence between SDNodes and Instruction by mapping
  /// SDNodes to the GINodeEquiv mapping. We need to map to the GINodeEquiv to
  /// check for attributes on the relation such as CheckMMOIsNonAtomic.
  /// This is defined using 'GINodeEquiv' in the target description.
  DenseMap<const Record *, const Record *> NodeEquivs;

  /// Keep track of the equivalence between ComplexPattern's and
  /// GIComplexOperandMatcher. Map entries are specified by subclassing
  /// GIComplexPatternEquiv.
  DenseMap<const Record *, const Record *> ComplexPatternEquivs;

  /// Keep track of the equivalence between SDNodeXForm's and
  /// GICustomOperandRenderer. Map entries are specified by subclassing
  /// GISDNodeXFormEquiv.
  DenseMap<const Record *, const Record *> SDNodeXFormEquivs;

  /// Keep track of Scores of PatternsToMatch similar to how the DAG does.
  /// This adds compatibility for RuleMatchers to use this for ordering rules.
  DenseMap<uint64_t, int> RuleMatcherScores;

  // Rule coverage information.
  std::optional<CodeGenCoverage> RuleCoverage;

  /// Variables used to help with collecting of named operands for predicates
  /// with 'let PredicateCodeUsesOperands = 1'. WaitingForNamedOperands is set
  /// to the number of named operands that predicate expects. Store locations in
  /// StoreIdxForName correspond to the order in which operand names appear in
  /// predicate's argument list.
  /// When we visit named operand and WaitingForNamedOperands is not zero, add
  /// matcher that will record operand and decrease counter.
  unsigned WaitingForNamedOperands = 0;
  StringMap<unsigned> StoreIdxForName;

  void gatherOpcodeValues();
  void gatherTypeIDValues();
  void gatherNodeEquivs();

  const Record *findNodeEquiv(const Record *N) const;
  const CodeGenInstruction *getEquivNode(const Record &Equiv,
                                         const TreePatternNode &N) const;

  Error importRulePredicates(RuleMatcher &M,
                             ArrayRef<const Record *> Predicates);
  Expected<InstructionMatcher &>
  createAndImportSelDAGMatcher(RuleMatcher &Rule,
                               InstructionMatcher &InsnMatcher,
                               const TreePatternNode &Src, unsigned &TempOpIdx);
  Error importComplexPatternOperandMatcher(OperandMatcher &OM, const Record *R,
                                           unsigned &TempOpIdx) const;
  Error importChildMatcher(RuleMatcher &Rule, InstructionMatcher &InsnMatcher,
                           const TreePatternNode &SrcChild,
                           bool OperandIsAPointer, bool OperandIsImmArg,
                           unsigned OpIdx, unsigned &TempOpIdx);

  Expected<BuildMIAction &>
  createAndImportInstructionRenderer(RuleMatcher &M,
                                     InstructionMatcher &InsnMatcher,
                                     const TreePatternNode &Dst) const;
  Expected<action_iterator> createAndImportSubInstructionRenderer(
      action_iterator InsertPt, RuleMatcher &M, const TreePatternNode &Dst,
      unsigned TempReg) const;
  Expected<action_iterator>
  createInstructionRenderer(action_iterator InsertPt, RuleMatcher &M,
                            const TreePatternNode &Dst) const;

  Expected<action_iterator>
  importExplicitDefRenderers(action_iterator InsertPt, RuleMatcher &M,
                             BuildMIAction &DstMIBuilder,
                             const TreePatternNode &Dst, bool IsRoot) const;

  Expected<action_iterator>
  importExplicitUseRenderers(action_iterator InsertPt, RuleMatcher &M,
                             BuildMIAction &DstMIBuilder,
                             const TreePatternNode &Dst) const;

  Error importNamedNodeRenderer(RuleMatcher &M, BuildMIAction &MIBuilder,
                                const TreePatternNode &N) const;

  Error importLeafNodeRenderer(RuleMatcher &M, BuildMIAction &MIBuilder,
                               const TreePatternNode &N,
                               action_iterator InsertPt) const;

  Error importXFormNodeRenderer(RuleMatcher &M, BuildMIAction &MIBuilder,
                                const TreePatternNode &N) const;

  Error importInstructionNodeRenderer(RuleMatcher &M, BuildMIAction &MIBuilder,
                                      const TreePatternNode &N,
                                      action_iterator &InsertPt) const;

  Error importNodeRenderer(RuleMatcher &M, BuildMIAction &MIBuilder,
                           const TreePatternNode &N,
                           action_iterator &InsertPt) const;

  Error importImplicitDefRenderers(BuildMIAction &DstMIBuilder,
                                   ArrayRef<const Record *> ImplicitDefs) const;

  /// Analyze pattern \p P, returning a matcher for it if possible.
  /// Otherwise, return an Error explaining why we don't support it.
  Expected<RuleMatcher> runOnPattern(const PatternToMatch &P);

  void declareSubtargetFeature(const Record *Predicate);

  unsigned declareHwModeCheck(StringRef HwModeFeatures);

  MatchTable buildMatchTable(MutableArrayRef<RuleMatcher> Rules, bool Optimize,
                             bool WithCoverage);

  /// Infer a CodeGenRegisterClass for the type of \p SuperRegNode. The returned
  /// CodeGenRegisterClass will support the CodeGenRegisterClass of
  /// \p SubRegNode, and the subregister index defined by \p SubRegIdxNode.
  /// If no register class is found, return nullptr.
  const CodeGenRegisterClass *
  inferSuperRegisterClassForNode(const TypeSetByHwMode &Ty,
                                 const TreePatternNode &SuperRegNode,
                                 const TreePatternNode &SubRegIdxNode) const;
  const CodeGenSubRegIndex *
  inferSubRegIndexForNode(const TreePatternNode &SubRegIdxNode) const;

  /// Infer a CodeGenRegisterClass which suppoorts \p Ty and \p SubRegIdxNode.
  /// Return nullptr if no such class exists.
  const CodeGenRegisterClass *
  inferSuperRegisterClass(const TypeSetByHwMode &Ty,
                          const TreePatternNode &SubRegIdxNode) const;

  /// Return the CodeGenRegisterClass associated with \p Leaf if it has one.
  const CodeGenRegisterClass *
  getRegClassFromLeaf(const TreePatternNode &Leaf) const;

  /// Return a CodeGenRegisterClass for \p N if one can be found. Return
  /// nullptr otherwise.
  const CodeGenRegisterClass *
  inferRegClassFromPattern(const TreePatternNode &N) const;

  const CodeGenRegisterClass *
  inferRegClassFromInstructionPattern(const TreePatternNode &N,
                                      unsigned ResIdx) const;

  Error constrainOperands(action_iterator InsertPt, RuleMatcher &M,
                          unsigned InsnID, const TreePatternNode &Dst) const;

  /// Return the size of the MemoryVT in this predicate, if possible.
  std::optional<unsigned>
  getMemSizeBitsFromPredicate(const TreePredicateFn &Predicate);

  // Add builtin predicates.
  Expected<InstructionMatcher &>
  addBuiltinPredicates(const Record *SrcGIEquivOrNull,
                       const TreePredicateFn &Predicate,
                       InstructionMatcher &InsnMatcher, bool &HasAddedMatcher);
};

StringRef getPatFragPredicateEnumName(const Record *R) { return R->getName(); }

void GlobalISelEmitter::gatherOpcodeValues() {
  InstructionOpcodeMatcher::initOpcodeValuesMap(Target);
}

void GlobalISelEmitter::gatherTypeIDValues() {
  LLTOperandMatcher::initTypeIDValuesMap();
}

void GlobalISelEmitter::gatherNodeEquivs() {
  assert(NodeEquivs.empty());
  for (const Record *Equiv : RK.getAllDerivedDefinitions("GINodeEquiv"))
    NodeEquivs[Equiv->getValueAsDef("Node")] = Equiv;

  assert(ComplexPatternEquivs.empty());
  for (const Record *Equiv :
       RK.getAllDerivedDefinitions("GIComplexPatternEquiv")) {
    const Record *SelDAGEquiv = Equiv->getValueAsDef("SelDAGEquivalent");
    if (!SelDAGEquiv)
      continue;
    ComplexPatternEquivs[SelDAGEquiv] = Equiv;
  }

  assert(SDNodeXFormEquivs.empty());
  for (const Record *Equiv :
       RK.getAllDerivedDefinitions("GISDNodeXFormEquiv")) {
    const Record *SelDAGEquiv = Equiv->getValueAsDef("SelDAGEquivalent");
    if (!SelDAGEquiv)
      continue;
    SDNodeXFormEquivs[SelDAGEquiv] = Equiv;
  }
}

const Record *GlobalISelEmitter::findNodeEquiv(const Record *N) const {
  return NodeEquivs.lookup(N);
}

const CodeGenInstruction *
GlobalISelEmitter::getEquivNode(const Record &Equiv,
                                const TreePatternNode &N) const {
  if (N.getNumChildren() >= 1) {
    // setcc operation maps to two different G_* instructions based on the type.
    if (!Equiv.isValueUnset("IfFloatingPoint") &&
        MVT(N.getChild(0).getSimpleType(0)).isFloatingPoint())
      return &Target.getInstruction(Equiv.getValueAsDef("IfFloatingPoint"));
  }

  if (!Equiv.isValueUnset("IfConvergent") &&
      N.getIntrinsicInfo(CGP)->isConvergent)
    return &Target.getInstruction(Equiv.getValueAsDef("IfConvergent"));

  for (const TreePredicateCall &Call : N.getPredicateCalls()) {
    const TreePredicateFn &Predicate = Call.Fn;
    if (!Equiv.isValueUnset("IfSignExtend") &&
        (Predicate.isLoad() || Predicate.isAtomic()) &&
        Predicate.isSignExtLoad())
      return &Target.getInstruction(Equiv.getValueAsDef("IfSignExtend"));
    if (!Equiv.isValueUnset("IfZeroExtend") &&
        (Predicate.isLoad() || Predicate.isAtomic()) &&
        Predicate.isZeroExtLoad())
      return &Target.getInstruction(Equiv.getValueAsDef("IfZeroExtend"));
  }

  return &Target.getInstruction(Equiv.getValueAsDef("I"));
}

GlobalISelEmitter::GlobalISelEmitter(const RecordKeeper &RK)
    : GlobalISelMatchTableExecutorEmitter(), RK(RK), CGP(RK),
      Target(CGP.getTargetInfo()), CGRegs(Target.getRegBank()) {
  ClassName = Target.getName().str() + "InstructionSelector";
}

//===- Emitter ------------------------------------------------------------===//

Error GlobalISelEmitter::importRulePredicates(
    RuleMatcher &M, ArrayRef<const Record *> Predicates) {
  for (const Record *Pred : Predicates) {
    if (Pred->getValueAsString("CondString").empty())
      continue;
    declareSubtargetFeature(Pred);
    M.addRequiredFeature(Pred);
  }

  return Error::success();
}

std::optional<unsigned> GlobalISelEmitter::getMemSizeBitsFromPredicate(
    const TreePredicateFn &Predicate) {
  std::optional<LLTCodeGen> MemTyOrNone =
      MVTToLLT(getValueType(Predicate.getMemoryVT()));

  if (!MemTyOrNone)
    return std::nullopt;

  // Align so unusual types like i1 don't get rounded down.
  return llvm::alignTo(
      static_cast<unsigned>(MemTyOrNone->get().getSizeInBits()), 8);
}

Expected<InstructionMatcher &> GlobalISelEmitter::addBuiltinPredicates(
    const Record *SrcGIEquivOrNull, const TreePredicateFn &Predicate,
    InstructionMatcher &InsnMatcher, bool &HasAddedMatcher) {
  if (Predicate.isLoad() || Predicate.isStore() || Predicate.isAtomic()) {
    if (const ListInit *AddrSpaces = Predicate.getAddressSpaces()) {
      SmallVector<unsigned, 4> ParsedAddrSpaces;

      for (const Init *Val : AddrSpaces->getElements()) {
        const IntInit *IntVal = dyn_cast<IntInit>(Val);
        if (!IntVal)
          return failedImport("Address space is not an integer");
        ParsedAddrSpaces.push_back(IntVal->getValue());
      }

      if (!ParsedAddrSpaces.empty()) {
        InsnMatcher.addPredicate<MemoryAddressSpacePredicateMatcher>(
            0, ParsedAddrSpaces);
        return InsnMatcher;
      }
    }

    int64_t MinAlign = Predicate.getMinAlignment();
    if (MinAlign > 0) {
      InsnMatcher.addPredicate<MemoryAlignmentPredicateMatcher>(0, MinAlign);
      return InsnMatcher;
    }
  }

  // G_LOAD is used for both non-extending and any-extending loads.
  if (Predicate.isLoad() || Predicate.isAtomic()) {
    if (Predicate.isNonExtLoad()) {
      InsnMatcher.addPredicate<MemoryVsLLTSizePredicateMatcher>(
          0, MemoryVsLLTSizePredicateMatcher::EqualTo, 0);
      return InsnMatcher;
    }
    if (Predicate.isAnyExtLoad()) {
      InsnMatcher.addPredicate<MemoryVsLLTSizePredicateMatcher>(
          0, MemoryVsLLTSizePredicateMatcher::LessThan, 0);
      return InsnMatcher;
    }
  }

  if (Predicate.isStore()) {
    if (Predicate.isTruncStore()) {
      if (Predicate.getMemoryVT() != nullptr) {
        // FIXME: If MemoryVT is set, we end up with 2 checks for the MMO size.
        auto MemSizeInBits = getMemSizeBitsFromPredicate(Predicate);
        if (!MemSizeInBits)
          return failedImport("MemVT could not be converted to LLT");

        InsnMatcher.addPredicate<MemorySizePredicateMatcher>(0, *MemSizeInBits /
                                                                    8);
      } else {
        InsnMatcher.addPredicate<MemoryVsLLTSizePredicateMatcher>(
            0, MemoryVsLLTSizePredicateMatcher::LessThan, 0);
      }
      return InsnMatcher;
    }
    if (Predicate.isNonTruncStore()) {
      // We need to check the sizes match here otherwise we could incorrectly
      // match truncating stores with non-truncating ones.
      InsnMatcher.addPredicate<MemoryVsLLTSizePredicateMatcher>(
          0, MemoryVsLLTSizePredicateMatcher::EqualTo, 0);
    }
  }

  assert(SrcGIEquivOrNull != nullptr && "Invalid SrcGIEquivOrNull value");
  // No check required. We already did it by swapping the opcode.
  if (!SrcGIEquivOrNull->isValueUnset("IfSignExtend") &&
      Predicate.isSignExtLoad())
    return InsnMatcher;

  // No check required. We already did it by swapping the opcode.
  if (!SrcGIEquivOrNull->isValueUnset("IfZeroExtend") &&
      Predicate.isZeroExtLoad())
    return InsnMatcher;

  // No check required. G_STORE by itself is a non-extending store.
  if (Predicate.isNonTruncStore())
    return InsnMatcher;

  if (Predicate.isLoad() || Predicate.isStore() || Predicate.isAtomic()) {
    if (Predicate.getMemoryVT() != nullptr) {
      auto MemSizeInBits = getMemSizeBitsFromPredicate(Predicate);
      if (!MemSizeInBits)
        return failedImport("MemVT could not be converted to LLT");

      InsnMatcher.addPredicate<MemorySizePredicateMatcher>(0,
                                                           *MemSizeInBits / 8);
      return InsnMatcher;
    }
  }

  if (Predicate.isLoad() || Predicate.isStore()) {
    // No check required. A G_LOAD/G_STORE is an unindexed load.
    if (Predicate.isUnindexed())
      return InsnMatcher;
  }

  if (Predicate.isAtomic()) {
    if (Predicate.isAtomicOrderingMonotonic()) {
      InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>("Monotonic");
      return InsnMatcher;
    }
    if (Predicate.isAtomicOrderingAcquire()) {
      InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>("Acquire");
      return InsnMatcher;
    }
    if (Predicate.isAtomicOrderingRelease()) {
      InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>("Release");
      return InsnMatcher;
    }
    if (Predicate.isAtomicOrderingAcquireRelease()) {
      InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>(
          "AcquireRelease");
      return InsnMatcher;
    }
    if (Predicate.isAtomicOrderingSequentiallyConsistent()) {
      InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>(
          "SequentiallyConsistent");
      return InsnMatcher;
    }
  }

  if (Predicate.isAtomicOrderingAcquireOrStronger()) {
    InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>(
        "Acquire", AtomicOrderingMMOPredicateMatcher::AO_OrStronger);
    return InsnMatcher;
  }
  if (Predicate.isAtomicOrderingWeakerThanAcquire()) {
    InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>(
        "Acquire", AtomicOrderingMMOPredicateMatcher::AO_WeakerThan);
    return InsnMatcher;
  }

  if (Predicate.isAtomicOrderingReleaseOrStronger()) {
    InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>(
        "Release", AtomicOrderingMMOPredicateMatcher::AO_OrStronger);
    return InsnMatcher;
  }
  if (Predicate.isAtomicOrderingWeakerThanRelease()) {
    InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>(
        "Release", AtomicOrderingMMOPredicateMatcher::AO_WeakerThan);
    return InsnMatcher;
  }
  HasAddedMatcher = false;
  return InsnMatcher;
}

Expected<InstructionMatcher &> GlobalISelEmitter::createAndImportSelDAGMatcher(
    RuleMatcher &Rule, InstructionMatcher &InsnMatcher,
    const TreePatternNode &Src, unsigned &TempOpIdx) {
  const auto SavedFlags = Rule.setGISelFlags(Src.getGISelFlagsRecord());

  const Record *SrcGIEquivOrNull = nullptr;
  const CodeGenInstruction *SrcGIOrNull = nullptr;

  // Start with the defined operands (i.e., the results of the root operator).
  if (Src.isLeaf()) {
    const Init *SrcInit = Src.getLeafValue();
    if (isa<IntInit>(SrcInit)) {
      InsnMatcher.addPredicate<InstructionOpcodeMatcher>(
          &Target.getInstruction(RK.getDef("G_CONSTANT")));
    } else {
      return failedImport(
          "Unable to deduce gMIR opcode to handle Src (which is a leaf)");
    }
  } else {
    SrcGIEquivOrNull = findNodeEquiv(Src.getOperator());
    if (!SrcGIEquivOrNull)
      return failedImport("Pattern operator lacks an equivalent Instruction" +
                          explainOperator(Src.getOperator()));
    SrcGIOrNull = getEquivNode(*SrcGIEquivOrNull, Src);

    // The operators look good: match the opcode
    InsnMatcher.addPredicate<InstructionOpcodeMatcher>(SrcGIOrNull);
  }

  // Since there are no opcodes for atomic loads and stores comparing to
  // SelectionDAG, we add CheckMMOIsNonAtomic predicate immediately after the
  // opcode predicate to make a logical combination of them.
  if (SrcGIEquivOrNull &&
      SrcGIEquivOrNull->getValueAsBit("CheckMMOIsNonAtomic"))
    InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>("NotAtomic");
  else if (SrcGIEquivOrNull &&
           SrcGIEquivOrNull->getValueAsBit("CheckMMOIsAtomic")) {
    InsnMatcher.addPredicate<AtomicOrderingMMOPredicateMatcher>(
        "Unordered", AtomicOrderingMMOPredicateMatcher::AO_OrStronger);
  }

  unsigned OpIdx = 0;
  for (const TypeSetByHwMode &VTy : Src.getExtTypes()) {
    // Results don't have a name unless they are the root node. The caller will
    // set the name if appropriate.
    const bool OperandIsAPointer =
        SrcGIOrNull && SrcGIOrNull->isOutOperandAPointer(OpIdx);
    OperandMatcher &OM = InsnMatcher.addOperand(OpIdx++, "", TempOpIdx);
    if (auto Error = OM.addTypeCheckPredicate(VTy, OperandIsAPointer))
      return failedImport(toString(std::move(Error)) +
                          " for result of Src pattern operator");
  }

  for (const TreePredicateCall &Call : Src.getPredicateCalls()) {
    const TreePredicateFn &Predicate = Call.Fn;
    bool HasAddedBuiltinMatcher = true;
    if (Predicate.isAlwaysTrue())
      continue;

    if (Predicate.isImmediatePattern()) {
      InsnMatcher.addPredicate<InstructionImmPredicateMatcher>(Predicate);
      continue;
    }

    auto InsnMatcherOrError = addBuiltinPredicates(
        SrcGIEquivOrNull, Predicate, InsnMatcher, HasAddedBuiltinMatcher);
    if (auto Error = InsnMatcherOrError.takeError())
      return std::move(Error);

    // FIXME: This should be part of addBuiltinPredicates(). If we add this at
    // the start of addBuiltinPredicates() without returning, then there might
    // be cases where we hit the last return before which the
    // HasAddedBuiltinMatcher will be set to false. The predicate could be
    // missed if we add it in the middle or at the end due to return statements
    // after the addPredicate<>() calls.
    if (Predicate.hasNoUse()) {
      InsnMatcher.addPredicate<NoUsePredicateMatcher>();
      HasAddedBuiltinMatcher = true;
    }
    if (Predicate.hasOneUse()) {
      InsnMatcher.addPredicate<OneUsePredicateMatcher>();
      HasAddedBuiltinMatcher = true;
    }

    if (Predicate.hasGISelPredicateCode()) {
      if (Predicate.usesOperands()) {
        assert(WaitingForNamedOperands == 0 &&
               "previous predicate didn't find all operands or "
               "nested predicate that uses operands");
        TreePattern *TP = Predicate.getOrigPatFragRecord();
        WaitingForNamedOperands = TP->getNumArgs();
        for (unsigned I = 0; I < WaitingForNamedOperands; ++I)
          StoreIdxForName[getScopedName(Call.Scope, TP->getArgName(I))] = I;
      }
      InsnMatcher.addPredicate<GenericInstructionPredicateMatcher>(Predicate);
      continue;
    }
    if (!HasAddedBuiltinMatcher) {
      return failedImport("Src pattern child has predicate (" +
                          explainPredicates(Src) + ")");
    }
  }

  if (Src.isLeaf()) {
    const Init *SrcInit = Src.getLeafValue();
    if (const IntInit *SrcIntInit = dyn_cast<IntInit>(SrcInit)) {
      OperandMatcher &OM =
          InsnMatcher.addOperand(OpIdx++, Src.getName().str(), TempOpIdx);
      OM.addPredicate<LiteralIntOperandMatcher>(SrcIntInit->getValue());
    } else {
      return failedImport(
          "Unable to deduce gMIR opcode to handle Src (which is a leaf)");
    }
  } else {
    assert(SrcGIOrNull &&
           "Expected to have already found an equivalent Instruction");
    if (SrcGIOrNull->getName() == "G_CONSTANT" ||
        SrcGIOrNull->getName() == "G_FCONSTANT" ||
        SrcGIOrNull->getName() == "G_FRAME_INDEX") {
      // imm/fpimm still have operands but we don't need to do anything with it
      // here since we don't support ImmLeaf predicates yet. However, we still
      // need to note the hidden operand to get GIM_CheckNumOperands correct.
      InsnMatcher.addOperand(OpIdx++, "", TempOpIdx);
      return InsnMatcher;
    }

    // Special case because the operand order is changed from setcc. The
    // predicate operand needs to be swapped from the last operand to the first
    // source.

    unsigned NumChildren = Src.getNumChildren();
    bool IsFCmp = SrcGIOrNull->getName() == "G_FCMP";

    if (IsFCmp || SrcGIOrNull->getName() == "G_ICMP") {
      const TreePatternNode &SrcChild = Src.getChild(NumChildren - 1);
      if (SrcChild.isLeaf()) {
        const DefInit *DI = dyn_cast<DefInit>(SrcChild.getLeafValue());
        const Record *CCDef = DI ? DI->getDef() : nullptr;
        if (!CCDef || !CCDef->isSubClassOf("CondCode"))
          return failedImport("Unable to handle CondCode");

        OperandMatcher &OM = InsnMatcher.addOperand(
            OpIdx++, SrcChild.getName().str(), TempOpIdx);
        StringRef PredType = IsFCmp ? CCDef->getValueAsString("FCmpPredicate")
                                    : CCDef->getValueAsString("ICmpPredicate");

        if (!PredType.empty()) {
          OM.addPredicate<CmpPredicateOperandMatcher>(PredType.str());
          // Process the other 2 operands normally.
          --NumChildren;
        }
      }
    }

    // Match the used operands (i.e. the children of the operator).
    bool IsIntrinsic =
        SrcGIOrNull->getName() == "G_INTRINSIC" ||
        SrcGIOrNull->getName() == "G_INTRINSIC_W_SIDE_EFFECTS" ||
        SrcGIOrNull->getName() == "G_INTRINSIC_CONVERGENT" ||
        SrcGIOrNull->getName() == "G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS";
    const CodeGenIntrinsic *II = Src.getIntrinsicInfo(CGP);
    if (IsIntrinsic && !II)
      return failedImport("Expected IntInit containing intrinsic ID)");

    for (unsigned I = 0; I != NumChildren; ++I) {
      const TreePatternNode &SrcChild = Src.getChild(I);

      // We need to determine the meaning of a literal integer based on the
      // context. If this is a field required to be an immediate (such as an
      // immarg intrinsic argument), the required predicates are different than
      // a constant which may be materialized in a register. If we have an
      // argument that is required to be an immediate, we should not emit an LLT
      // type check, and should not be looking for a G_CONSTANT defined
      // register.
      bool OperandIsImmArg = SrcGIOrNull->isInOperandImmArg(I);

      // SelectionDAG allows pointers to be represented with iN since it doesn't
      // distinguish between pointers and integers but they are different types
      // in GlobalISel. Coerce integers to pointers to address space 0 if the
      // context indicates a pointer.
      //
      bool OperandIsAPointer = SrcGIOrNull->isInOperandAPointer(I);

      if (IsIntrinsic) {
        // For G_INTRINSIC/G_INTRINSIC_W_SIDE_EFFECTS, the operand immediately
        // following the defs is an intrinsic ID.
        if (I == 0) {
          OperandMatcher &OM = InsnMatcher.addOperand(
              OpIdx++, SrcChild.getName().str(), TempOpIdx);
          OM.addPredicate<IntrinsicIDOperandMatcher>(II);
          continue;
        }

        // We have to check intrinsics for llvm_anyptr_ty and immarg parameters.
        //
        // Note that we have to look at the i-1th parameter, because we don't
        // have the intrinsic ID in the intrinsic's parameter list.
        OperandIsAPointer |= II->isParamAPointer(I - 1);
        OperandIsImmArg |= II->isParamImmArg(I - 1);
      }

      if (auto Error =
              importChildMatcher(Rule, InsnMatcher, SrcChild, OperandIsAPointer,
                                 OperandIsImmArg, OpIdx++, TempOpIdx))
        return std::move(Error);
    }
  }

  return InsnMatcher;
}

Error GlobalISelEmitter::importComplexPatternOperandMatcher(
    OperandMatcher &OM, const Record *R, unsigned &TempOpIdx) const {
  const auto &ComplexPattern = ComplexPatternEquivs.find(R);
  if (ComplexPattern == ComplexPatternEquivs.end())
    return failedImport("SelectionDAG ComplexPattern (" + R->getName() +
                        ") not mapped to GlobalISel");

  OM.addPredicate<ComplexPatternOperandMatcher>(OM, *ComplexPattern->second);
  TempOpIdx++;
  return Error::success();
}

// Get the name to use for a pattern operand. For an anonymous physical register
// input, this should use the register name.
static StringRef getSrcChildName(const TreePatternNode &SrcChild,
                                 const Record *&PhysReg) {
  StringRef SrcChildName = SrcChild.getName();
  if (SrcChildName.empty() && SrcChild.isLeaf()) {
    if (auto *ChildDefInit = dyn_cast<DefInit>(SrcChild.getLeafValue())) {
      auto *ChildRec = ChildDefInit->getDef();
      if (ChildRec->isSubClassOf("Register")) {
        SrcChildName = ChildRec->getName();
        PhysReg = ChildRec;
      }
    }
  }

  return SrcChildName;
}

Error GlobalISelEmitter::importChildMatcher(
    RuleMatcher &Rule, InstructionMatcher &InsnMatcher,
    const TreePatternNode &SrcChild, bool OperandIsAPointer,
    bool OperandIsImmArg, unsigned OpIdx, unsigned &TempOpIdx) {

  const Record *PhysReg = nullptr;
  std::string SrcChildName = getSrcChildName(SrcChild, PhysReg).str();
  if (!SrcChild.isLeaf() &&
      SrcChild.getOperator()->isSubClassOf("ComplexPattern")) {
    // The "name" of a non-leaf complex pattern (MY_PAT $op1, $op2) is
    // "MY_PAT:op1:op2" and the ones with same "name" represent same operand.
    std::string PatternName = SrcChild.getOperator()->getName().str();
    for (const TreePatternNode &Child : SrcChild.children()) {
      PatternName += ":";
      PatternName += Child.getName();
    }
    SrcChildName = PatternName;
  }

  OperandMatcher &OM =
      PhysReg ? InsnMatcher.addPhysRegInput(PhysReg, OpIdx, TempOpIdx)
              : InsnMatcher.addOperand(OpIdx, SrcChildName, TempOpIdx);
  if (OM.isSameAsAnotherOperand())
    return Error::success();

  ArrayRef<TypeSetByHwMode> ChildTypes = SrcChild.getExtTypes();
  if (ChildTypes.size() != 1)
    return failedImport("Src pattern child has multiple results");

  // Check MBB's before the type check since they are not a known type.
  if (!SrcChild.isLeaf()) {
    if (SrcChild.getOperator()->getName() == "bb") {
      OM.addPredicate<MBBOperandMatcher>();
      return Error::success();
    }
    if (SrcChild.getOperator()->getName() == "timm") {
      OM.addPredicate<ImmOperandMatcher>();

      // Add predicates, if any
      for (const TreePredicateCall &Call : SrcChild.getPredicateCalls()) {
        const TreePredicateFn &Predicate = Call.Fn;

        // Only handle immediate patterns for now
        if (Predicate.isImmediatePattern()) {
          OM.addPredicate<OperandImmPredicateMatcher>(Predicate);
        }
      }

      return Error::success();
    }
  } else if (auto *ChildDefInit = dyn_cast<DefInit>(SrcChild.getLeafValue())) {
    auto *ChildRec = ChildDefInit->getDef();
    if (ChildRec->isSubClassOf("ValueType") && !SrcChild.hasName()) {
      // An unnamed ValueType as in (sext_inreg GPR:$foo, i8). GISel represents
      // this as a literal constant with the scalar size.
      MVT::SimpleValueType VT = llvm::getValueType(ChildRec);
      OM.addPredicate<LiteralIntOperandMatcher>(MVT(VT).getScalarSizeInBits());
      return Error::success();
    }
  }

  // Immediate arguments have no meaningful type to check as they don't have
  // registers.
  if (!OperandIsImmArg) {
    if (auto Error =
            OM.addTypeCheckPredicate(ChildTypes.front(), OperandIsAPointer))
      return failedImport(toString(std::move(Error)) + " for Src operand (" +
                          to_string(SrcChild) + ")");
  }

  // Try look up SrcChild for a (named) predicate operand if there is any.
  if (WaitingForNamedOperands) {
    auto &ScopedNames = SrcChild.getNamesAsPredicateArg();
    if (!ScopedNames.empty()) {
      auto PA = ScopedNames.begin();
      std::string Name = getScopedName(PA->getScope(), PA->getIdentifier());
      OM.addPredicate<RecordNamedOperandMatcher>(StoreIdxForName[Name], Name);
      --WaitingForNamedOperands;
    }
  }

  // Check for nested instructions.
  if (!SrcChild.isLeaf()) {
    if (SrcChild.getOperator()->isSubClassOf("ComplexPattern")) {
      // When a ComplexPattern is used as an operator, it should do the same
      // thing as when used as a leaf. However, the children of the operator
      // name the sub-operands that make up the complex operand and we must
      // prepare to reference them in the renderer too.
      unsigned RendererID = TempOpIdx;
      if (auto Error = importComplexPatternOperandMatcher(
              OM, SrcChild.getOperator(), TempOpIdx))
        return Error;

      for (unsigned I = 0, E = SrcChild.getNumChildren(); I != E; ++I) {
        auto &SubOperand = SrcChild.getChild(I);
        if (!SubOperand.getName().empty()) {
          if (auto Error = Rule.defineComplexSubOperand(
                  SubOperand.getName(), SrcChild.getOperator(), RendererID, I,
                  SrcChildName))
            return Error;
        }
      }

      return Error::success();
    }

    auto MaybeInsnOperand = OM.addPredicate<InstructionOperandMatcher>(
        InsnMatcher.getRuleMatcher(), SrcChild.getName());
    if (!MaybeInsnOperand) {
      // This isn't strictly true. If the user were to provide exactly the same
      // matchers as the original operand then we could allow it. However, it's
      // simpler to not permit the redundant specification.
      return failedImport(
          "Nested instruction cannot be the same as another operand");
    }

    // Map the node to a gMIR instruction.
    InstructionOperandMatcher &InsnOperand = **MaybeInsnOperand;
    auto InsnMatcherOrError = createAndImportSelDAGMatcher(
        Rule, InsnOperand.getInsnMatcher(), SrcChild, TempOpIdx);
    if (auto Error = InsnMatcherOrError.takeError())
      return Error;

    return Error::success();
  }

  if (SrcChild.hasAnyPredicate()) {
    for (const TreePredicateCall &Call : SrcChild.getPredicateCalls()) {
      const TreePredicateFn &Predicate = Call.Fn;

      if (!Predicate.hasGISelLeafPredicateCode())
        return failedImport("Src pattern child has unsupported predicate");
      OM.addPredicate<OperandLeafPredicateMatcher>(Predicate);
    }
    return Error::success();
  }

  // Check for constant immediates.
  if (auto *ChildInt = dyn_cast<IntInit>(SrcChild.getLeafValue())) {
    if (OperandIsImmArg) {
      // Checks for argument directly in operand list
      OM.addPredicate<LiteralIntOperandMatcher>(ChildInt->getValue());
    } else {
      // Checks for materialized constant
      OM.addPredicate<ConstantIntOperandMatcher>(ChildInt->getValue());
    }
    return Error::success();
  }

  // Check for def's like register classes or ComplexPattern's.
  if (auto *ChildDefInit = dyn_cast<DefInit>(SrcChild.getLeafValue())) {
    auto *ChildRec = ChildDefInit->getDef();

    // Check for register classes.
    if (ChildRec->isSubClassOf("RegisterClass") ||
        ChildRec->isSubClassOf("RegisterOperand")) {
      OM.addPredicate<RegisterBankOperandMatcher>(
          Target.getRegisterClass(getInitValueAsRegClass(ChildDefInit)));
      return Error::success();
    }

    if (ChildRec->isSubClassOf("Register")) {
      // This just be emitted as a copy to the specific register.
      ValueTypeByHwMode VT = ChildTypes.front().getValueTypeByHwMode();
      const CodeGenRegisterClass *RC =
          CGRegs.getMinimalPhysRegClass(ChildRec, &VT);
      if (!RC) {
        return failedImport(
            "Could not determine physical register class of pattern source");
      }

      OM.addPredicate<RegisterBankOperandMatcher>(*RC);
      return Error::success();
    }

    // Check for ValueType.
    if (ChildRec->isSubClassOf("ValueType")) {
      // We already added a type check as standard practice so this doesn't need
      // to do anything.
      return Error::success();
    }

    // Check for ComplexPattern's.
    if (ChildRec->isSubClassOf("ComplexPattern"))
      return importComplexPatternOperandMatcher(OM, ChildRec, TempOpIdx);

    if (ChildRec->isSubClassOf("ImmLeaf")) {
      return failedImport(
          "Src pattern child def is an unsupported tablegen class (ImmLeaf)");
    }

    // Place holder for SRCVALUE nodes. Nothing to do here.
    if (ChildRec->getName() == "srcvalue")
      return Error::success();

    const bool ImmAllOnesV = ChildRec->getName() == "immAllOnesV";
    if (ImmAllOnesV || ChildRec->getName() == "immAllZerosV") {
      auto MaybeInsnOperand = OM.addPredicate<InstructionOperandMatcher>(
          InsnMatcher.getRuleMatcher(), SrcChild.getName(), false);
      InstructionOperandMatcher &InsnOperand = **MaybeInsnOperand;

      ValueTypeByHwMode VTy = ChildTypes.front().getValueTypeByHwMode();

      const CodeGenInstruction &BuildVector =
          Target.getInstruction(RK.getDef("G_BUILD_VECTOR"));
      const CodeGenInstruction &BuildVectorTrunc =
          Target.getInstruction(RK.getDef("G_BUILD_VECTOR_TRUNC"));

      // Treat G_BUILD_VECTOR as the canonical opcode, and G_BUILD_VECTOR_TRUNC
      // as an alternative.
      InsnOperand.getInsnMatcher().addPredicate<InstructionOpcodeMatcher>(
          ArrayRef({&BuildVector, &BuildVectorTrunc}));

      // TODO: Handle both G_BUILD_VECTOR and G_BUILD_VECTOR_TRUNC We could
      // theoretically not emit any opcode check, but getOpcodeMatcher currently
      // has to succeed.
      OperandMatcher &OM =
          InsnOperand.getInsnMatcher().addOperand(0, "", TempOpIdx);
      if (auto Error = OM.addTypeCheckPredicate(TypeSetByHwMode(VTy),
                                                /*OperandIsAPointer=*/false))
        return failedImport(toString(std::move(Error)) +
                            " for result of Src pattern operator");

      InsnOperand.getInsnMatcher().addPredicate<VectorSplatImmPredicateMatcher>(
          ImmAllOnesV ? VectorSplatImmPredicateMatcher::AllOnes
                      : VectorSplatImmPredicateMatcher::AllZeros);
      return Error::success();
    }

    return failedImport(
        "Src pattern child def is an unsupported tablegen class");
  }

  return failedImport("Src pattern child is an unsupported kind");
}

// Equivalent of MatcherGen::EmitResultOfNamedOperand.
Error GlobalISelEmitter::importNamedNodeRenderer(
    RuleMatcher &M, BuildMIAction &MIBuilder, const TreePatternNode &N) const {
  StringRef NodeName = N.getName();

  if (auto SubOperand = M.getComplexSubOperand(NodeName)) {
    auto [ComplexPatternRec, RendererID, SubOperandIdx] = *SubOperand;
    MIBuilder.addRenderer<RenderComplexPatternOperand>(
        *ComplexPatternRec, NodeName, RendererID, SubOperandIdx);
    return Error::success();
  }

  if (!N.isLeaf()) {
    StringRef OperatorName = N.getOperator()->getName();

    if (OperatorName == "imm") {
      MIBuilder.addRenderer<CopyConstantAsImmRenderer>(NodeName);
      return Error::success();
    }

    if (OperatorName == "fpimm") {
      MIBuilder.addRenderer<CopyFConstantAsFPImmRenderer>(NodeName);
      return Error::success();
    }

    // TODO: 'imm' and 'fpimm' are the only nodes that need special treatment.
    //   Remove this check and add CopyRenderer unconditionally for other nodes.
    if (OperatorName == "bb" || OperatorName == "timm" ||
        OperatorName == "tframeindex") {
      MIBuilder.addRenderer<CopyRenderer>(NodeName);
      return Error::success();
    }

    return failedImport("node has unsupported operator " + to_string(N));
  }

  if (const auto *DI = dyn_cast<DefInit>(N.getLeafValue())) {
    const Record *R = DI->getDef();

    if (N.getNumResults() != 1)
      return failedImport("node does not have one result " + to_string(N));

    if (R->isSubClassOf("ComplexPattern")) {
      auto I = ComplexPatternEquivs.find(R);
      if (I == ComplexPatternEquivs.end())
        return failedImport("ComplexPattern " + R->getName() +
                            " does not have GISel equivalent");

      const OperandMatcher &OM = M.getOperandMatcher(NodeName);
      MIBuilder.addRenderer<RenderComplexPatternOperand>(
          *I->second, NodeName, OM.getAllocatedTemporariesBaseID());
      return Error::success();
    }

    if (R->isSubClassOf("RegisterOperand") &&
        !R->isValueUnset("GIZeroRegister")) {
      MIBuilder.addRenderer<CopyOrAddZeroRegRenderer>(
          NodeName, R->getValueAsDef("GIZeroRegister"));
      return Error::success();
    }

    // TODO: All special cases are handled above. Remove this check and add
    //   CopyRenderer unconditionally.
    if (R->isSubClassOf("RegisterClass") ||
        R->isSubClassOf("RegisterOperand") || R->isSubClassOf("ValueType")) {
      MIBuilder.addRenderer<CopyRenderer>(NodeName);
      return Error::success();
    }
  }

  // TODO: Change this to assert and move to the beginning of the function.
  if (!M.hasOperand(NodeName))
    return failedImport("could not find node $" + NodeName +
                        " in the source DAG");

  // TODO: Remove this check and add CopyRenderer unconditionally.
  // TODO: Handle nodes with multiple results (provided they can reach here).
  if (isa<UnsetInit>(N.getLeafValue())) {
    MIBuilder.addRenderer<CopyRenderer>(NodeName);
    return Error::success();
  }

  return failedImport("unsupported node " + to_string(N));
}

// Equivalent of MatcherGen::EmitResultLeafAsOperand.
Error GlobalISelEmitter::importLeafNodeRenderer(
    RuleMatcher &M, BuildMIAction &MIBuilder, const TreePatternNode &N,
    action_iterator InsertPt) const {
  if (const auto *II = dyn_cast<IntInit>(N.getLeafValue())) {
    MIBuilder.addRenderer<ImmRenderer>(II->getValue());
    return Error::success();
  }

  if (const auto *DI = dyn_cast<DefInit>(N.getLeafValue())) {
    const Record *R = DI->getDef();

    if (R->isSubClassOf("Register") || R->getName() == "zero_reg") {
      MIBuilder.addRenderer<AddRegisterRenderer>(Target, R);
      return Error::success();
    }

    if (R->getName() == "undef_tied_input") {
      std::optional<LLTCodeGen> OpTyOrNone = MVTToLLT(N.getSimpleType(0));
      if (!OpTyOrNone)
        return failedImport("unsupported type");

      unsigned TempRegID = M.allocateTempRegID();
      M.insertAction<MakeTempRegisterAction>(InsertPt, *OpTyOrNone, TempRegID);

      auto I = M.insertAction<BuildMIAction>(
          InsertPt, M.allocateOutputInsnID(),
          &Target.getInstruction(RK.getDef("IMPLICIT_DEF")));
      auto &ImpDefBuilder = static_cast<BuildMIAction &>(**I);
      ImpDefBuilder.addRenderer<TempRegRenderer>(TempRegID, /*IsDef=*/true);

      MIBuilder.addRenderer<TempRegRenderer>(TempRegID);
      return Error::success();
    }

    if (R->isSubClassOf("SubRegIndex")) {
      const CodeGenSubRegIndex *SubRegIndex = CGRegs.findSubRegIdx(R);
      MIBuilder.addRenderer<ImmRenderer>(SubRegIndex->EnumValue);
      return Error::success();
    }

    // There are also RegisterClass / RegisterOperand operands of REG_SEQUENCE /
    // COPY_TO_REGCLASS, but these instructions are currently handled elsewhere.
  }

  return failedImport("unrecognized node " + to_string(N));
}

// Equivalent of MatcherGen::EmitResultSDNodeXFormAsOperand.
Error GlobalISelEmitter::importXFormNodeRenderer(
    RuleMatcher &M, BuildMIAction &MIBuilder, const TreePatternNode &N) const {
  const Record *XFormRec = N.getOperator();
  auto I = SDNodeXFormEquivs.find(XFormRec);
  if (I == SDNodeXFormEquivs.end())
    return failedImport("SDNodeXForm " + XFormRec->getName() +
                        " does not have GISel equivalent");

  // TODO: Fail to import if GISDNodeXForm does not have RendererFn.
  //   This currently results in a fatal error in emitRenderOpcodes.
  const Record *XFormEquivRec = I->second;

  // The node to apply the transformation function to.
  // FIXME: The node may not have a name and may be a leaf. It should be
  //   rendered first, like any other nodes. This may or may not require
  //   introducing a temporary register, and we can't tell that without
  //   inspecting the node (possibly recursively). This is a general drawback
  //   of appending renderers directly to BuildMIAction.
  const TreePatternNode &Node = N.getChild(0);
  StringRef NodeName = Node.getName();

  const Record *XFormOpc = CGP.getSDNodeTransform(XFormRec).first;
  if (XFormOpc->getName() == "timm") {
    // If this is a TargetConstant, there won't be a corresponding
    // instruction to transform. Instead, this will refer directly to an
    // operand in an instruction's operand list.
    MIBuilder.addRenderer<CustomOperandRenderer>(*XFormEquivRec, NodeName);
  } else {
    MIBuilder.addRenderer<CustomRenderer>(*XFormEquivRec, NodeName);
  }

  return Error::success();
}

// Equivalent of MatcherGen::EmitResultInstructionAsOperand.
Error GlobalISelEmitter::importInstructionNodeRenderer(
    RuleMatcher &M, BuildMIAction &MIBuilder, const TreePatternNode &N,
    action_iterator &InsertPt) const {
  Expected<LLTCodeGen> OpTy = getInstResultType(N, Target);
  if (!OpTy)
    return OpTy.takeError();

  // TODO: See the comment in importXFormNodeRenderer. We rely on the node
  //   requiring a temporary register, which prevents us from using this
  //   function on the root of the destination DAG.
  unsigned TempRegID = M.allocateTempRegID();
  InsertPt = M.insertAction<MakeTempRegisterAction>(InsertPt, *OpTy, TempRegID);
  MIBuilder.addRenderer<TempRegRenderer>(TempRegID);

  auto InsertPtOrError =
      createAndImportSubInstructionRenderer(++InsertPt, M, N, TempRegID);
  if (!InsertPtOrError)
    return InsertPtOrError.takeError();

  InsertPt = *InsertPtOrError;
  return Error::success();
}

// Equivalent of MatcherGen::EmitResultOperand.
Error GlobalISelEmitter::importNodeRenderer(RuleMatcher &M,
                                            BuildMIAction &MIBuilder,
                                            const TreePatternNode &N,
                                            action_iterator &InsertPt) const {
  if (N.hasName())
    return importNamedNodeRenderer(M, MIBuilder, N);

  if (N.isLeaf())
    return importLeafNodeRenderer(M, MIBuilder, N, InsertPt);

  if (N.getOperator()->isSubClassOf("SDNodeXForm"))
    return importXFormNodeRenderer(M, MIBuilder, N);

  if (N.getOperator()->isSubClassOf("Instruction"))
    return importInstructionNodeRenderer(M, MIBuilder, N, InsertPt);

  // Should not reach here.
  return failedImport("unrecognized node " + llvm::to_string(N));
}

/// Generates code that builds the resulting instruction(s) from the destination
/// DAG. Note that to do this we do not and should not need the source DAG.
/// We do need to know whether a generated instruction defines a result of the
/// source DAG; this information is available via RuleMatcher::hasOperand.
Expected<BuildMIAction &> GlobalISelEmitter::createAndImportInstructionRenderer(
    RuleMatcher &M, InstructionMatcher &InsnMatcher,
    const TreePatternNode &Dst) const {
  auto InsertPtOrError = createInstructionRenderer(M.actions_end(), M, Dst);
  if (auto Error = InsertPtOrError.takeError())
    return std::move(Error);

  action_iterator InsertPt = InsertPtOrError.get();
  BuildMIAction &DstMIBuilder = *static_cast<BuildMIAction *>(InsertPt->get());

  for (auto PhysOp : M.physoperands()) {
    InsertPt = M.insertAction<BuildMIAction>(
        InsertPt, M.allocateOutputInsnID(),
        &Target.getInstruction(RK.getDef("COPY")));
    BuildMIAction &CopyToPhysRegMIBuilder =
        *static_cast<BuildMIAction *>(InsertPt->get());
    CopyToPhysRegMIBuilder.addRenderer<AddRegisterRenderer>(Target,
                                                            PhysOp.first, true);
    CopyToPhysRegMIBuilder.addRenderer<CopyPhysRegRenderer>(PhysOp.first);
  }

  if (auto Error = importExplicitDefRenderers(InsertPt, M, DstMIBuilder, Dst,
                                              /*IsRoot=*/true)
                       .takeError())
    return std::move(Error);

  if (auto Error = importExplicitUseRenderers(InsertPt, M, DstMIBuilder, Dst)
                       .takeError())
    return std::move(Error);

  return DstMIBuilder;
}

Expected<action_iterator>
GlobalISelEmitter::createAndImportSubInstructionRenderer(
    action_iterator InsertPt, RuleMatcher &M, const TreePatternNode &Dst,
    unsigned TempRegID) const {
  auto InsertPtOrError = createInstructionRenderer(InsertPt, M, Dst);

  // TODO: Assert there's exactly one result.

  if (auto Error = InsertPtOrError.takeError())
    return std::move(Error);

  BuildMIAction &DstMIBuilder =
      *static_cast<BuildMIAction *>(InsertPtOrError.get()->get());

  // Assign the result to TempReg.
  DstMIBuilder.addRenderer<TempRegRenderer>(TempRegID, true);

  // Handle additional (ignored) results.
  InsertPtOrError = importExplicitDefRenderers(
      std::prev(*InsertPtOrError), M, DstMIBuilder, Dst, /*IsRoot=*/false);
  if (auto Error = InsertPtOrError.takeError())
    return std::move(Error);

  InsertPtOrError =
      importExplicitUseRenderers(InsertPtOrError.get(), M, DstMIBuilder, Dst);
  if (auto Error = InsertPtOrError.takeError())
    return std::move(Error);

  if (auto Error =
          constrainOperands(InsertPt, M, DstMIBuilder.getInsnID(), Dst))
    return std::move(Error);

  return InsertPtOrError.get();
}

Expected<action_iterator>
GlobalISelEmitter::createInstructionRenderer(action_iterator InsertPt,
                                             RuleMatcher &M,
                                             const TreePatternNode &Dst) const {
  const Record *DstOp = Dst.getOperator();
  if (!DstOp->isSubClassOf("Instruction")) {
    if (DstOp->isSubClassOf("ValueType"))
      return failedImport(
          "Pattern operator isn't an instruction (it's a ValueType)");
    return failedImport("Pattern operator isn't an instruction");
  }
  CodeGenInstruction *DstI = &Target.getInstruction(DstOp);

  // COPY_TO_REGCLASS is just a copy with a ConstrainOperandToRegClassAction
  // attached. Similarly for EXTRACT_SUBREG except that's a subregister copy.
  StringRef Name = DstI->getName();
  if (Name == "COPY_TO_REGCLASS" || Name == "EXTRACT_SUBREG")
    DstI = &Target.getInstruction(RK.getDef("COPY"));

  return M.insertAction<BuildMIAction>(InsertPt, M.allocateOutputInsnID(),
                                       DstI);
}

Expected<action_iterator> GlobalISelEmitter::importExplicitDefRenderers(
    action_iterator InsertPt, RuleMatcher &M, BuildMIAction &DstMIBuilder,
    const TreePatternNode &Dst, bool IsRoot) const {
  const CodeGenInstruction *DstI = DstMIBuilder.getCGI();

  // Process explicit defs. The caller may have already handled the first def.
  for (unsigned I = IsRoot ? 0 : 1, E = DstI->Operands.NumDefs; I != E; ++I) {
    const CGIOperandList::OperandInfo &OpInfo = DstI->Operands[I];
    std::string OpName = getMangledRootDefName(OpInfo.Name);

    // If the def is used in the source DAG, forward it.
    if (IsRoot && M.hasOperand(OpName)) {
      // CopyRenderer saves a StringRef, so cannot pass OpName itself -
      // let's use a string with an appropriate lifetime.
      StringRef PermanentRef = M.getOperandMatcher(OpName).getSymbolicName();
      DstMIBuilder.addRenderer<CopyRenderer>(PermanentRef);
      continue;
    }

    // A discarded explicit def may be an optional physical register.
    // If this is the case, add the default register and mark it as dead.
    if (OpInfo.Rec->isSubClassOf("OptionalDefOperand")) {
      for (const TreePatternNode &DefaultOp :
           make_pointee_range(CGP.getDefaultOperand(OpInfo.Rec).DefaultOps)) {
        // TODO: Do these checks in ParseDefaultOperands.
        if (!DefaultOp.isLeaf())
          return failedImport("optional def is not a leaf");

        const auto *RegDI = dyn_cast<DefInit>(DefaultOp.getLeafValue());
        if (!RegDI)
          return failedImport("optional def is not a record");

        const Record *Reg = RegDI->getDef();
        if (!Reg->isSubClassOf("Register") && Reg->getName() != "zero_reg")
          return failedImport("optional def is not a register");

        DstMIBuilder.addRenderer<AddRegisterRenderer>(
            Target, Reg, /*IsDef=*/true, /*IsDead=*/true);
      }
      continue;
    }

    // The def is discarded, create a dead virtual register for it.
    const TypeSetByHwMode &ExtTy = Dst.getExtType(I);
    if (!ExtTy.isMachineValueType())
      return failedImport("unsupported typeset");

    auto OpTy = MVTToLLT(ExtTy.getMachineValueType().SimpleTy);
    if (!OpTy)
      return failedImport("unsupported type");

    unsigned TempRegID = M.allocateTempRegID();
    InsertPt =
        M.insertAction<MakeTempRegisterAction>(InsertPt, *OpTy, TempRegID);
    DstMIBuilder.addRenderer<TempRegRenderer>(
        TempRegID, /*IsDef=*/true, /*SubReg=*/nullptr, /*IsDead=*/true);
  }

  // Implicit defs are not currently supported, mark all of them as dead.
  for (const Record *Reg : DstI->ImplicitDefs) {
    std::string OpName = getMangledRootDefName(Reg->getName());
    assert(!M.hasOperand(OpName) && "The pattern should've been rejected");
    DstMIBuilder.setDeadImplicitDef(Reg);
  }

  return InsertPt;
}

Expected<action_iterator> GlobalISelEmitter::importExplicitUseRenderers(
    action_iterator InsertPt, RuleMatcher &M, BuildMIAction &DstMIBuilder,
    const TreePatternNode &Dst) const {
  const CodeGenInstruction *DstI = DstMIBuilder.getCGI();
  CodeGenInstruction *OrigDstI = &Target.getInstruction(Dst.getOperator());

  StringRef Name = OrigDstI->getName();
  unsigned ExpectedDstINumUses = Dst.getNumChildren();

  // EXTRACT_SUBREG needs to use a subregister COPY.
  if (Name == "EXTRACT_SUBREG") {
    if (!Dst.getChild(1).isLeaf())
      return failedImport("EXTRACT_SUBREG child #1 is not a leaf");
    const DefInit *SubRegInit =
        dyn_cast<DefInit>(Dst.getChild(1).getLeafValue());
    if (!SubRegInit)
      return failedImport("EXTRACT_SUBREG child #1 is not a subreg index");

    const CodeGenSubRegIndex *SubIdx =
        CGRegs.findSubRegIdx(SubRegInit->getDef());
    const TreePatternNode &ValChild = Dst.getChild(0);
    if (!ValChild.isLeaf()) {
      // We really have to handle the source instruction, and then insert a
      // copy from the subregister.
      auto ExtractSrcTy = getInstResultType(ValChild, Target);
      if (!ExtractSrcTy)
        return ExtractSrcTy.takeError();

      unsigned TempRegID = M.allocateTempRegID();
      InsertPt = M.insertAction<MakeTempRegisterAction>(InsertPt, *ExtractSrcTy,
                                                        TempRegID);

      auto InsertPtOrError = createAndImportSubInstructionRenderer(
          ++InsertPt, M, ValChild, TempRegID);
      if (auto Error = InsertPtOrError.takeError())
        return std::move(Error);

      DstMIBuilder.addRenderer<TempRegRenderer>(TempRegID, false, SubIdx);
      return InsertPt;
    }

    // If this is a source operand, this is just a subregister copy.
    const Record *RCDef = getInitValueAsRegClass(ValChild.getLeafValue());
    if (!RCDef)
      return failedImport("EXTRACT_SUBREG child #0 could not "
                          "be coerced to a register class");

    CodeGenRegisterClass *RC = CGRegs.getRegClass(RCDef);

    const auto SrcRCDstRCPair =
        RC->getMatchingSubClassWithSubRegs(CGRegs, SubIdx);
    if (SrcRCDstRCPair) {
      assert(SrcRCDstRCPair->second && "Couldn't find a matching subclass");
      if (SrcRCDstRCPair->first != RC)
        return failedImport("EXTRACT_SUBREG requires an additional COPY");
    }

    StringRef RegOperandName = Dst.getChild(0).getName();
    if (const auto &SubOperand = M.getComplexSubOperand(RegOperandName)) {
      DstMIBuilder.addRenderer<RenderComplexPatternOperand>(
          *std::get<0>(*SubOperand), RegOperandName, std::get<1>(*SubOperand),
          std::get<2>(*SubOperand), SubIdx);
      return InsertPt;
    }

    DstMIBuilder.addRenderer<CopySubRegRenderer>(RegOperandName, SubIdx);
    return InsertPt;
  }

  if (Name == "REG_SEQUENCE") {
    if (!Dst.getChild(0).isLeaf())
      return failedImport("REG_SEQUENCE child #0 is not a leaf");

    const Record *RCDef =
        getInitValueAsRegClass(Dst.getChild(0).getLeafValue());
    if (!RCDef)
      return failedImport("REG_SEQUENCE child #0 could not "
                          "be coerced to a register class");

    if ((ExpectedDstINumUses - 1) % 2 != 0)
      return failedImport("Malformed REG_SEQUENCE");

    for (unsigned I = 1; I != ExpectedDstINumUses; I += 2) {
      const TreePatternNode &ValChild = Dst.getChild(I);
      const TreePatternNode &SubRegChild = Dst.getChild(I + 1);

      if (const DefInit *SubRegInit =
              dyn_cast<DefInit>(SubRegChild.getLeafValue())) {
        const CodeGenSubRegIndex *SubIdx =
            CGRegs.findSubRegIdx(SubRegInit->getDef());

        if (Error Err = importNodeRenderer(M, DstMIBuilder, ValChild, InsertPt))
          return Err;

        DstMIBuilder.addRenderer<SubRegIndexRenderer>(SubIdx);
      }
    }

    return InsertPt;
  }

  // Render the explicit uses.
  unsigned DstINumUses = OrigDstI->Operands.size() - OrigDstI->Operands.NumDefs;
  if (Name == "COPY_TO_REGCLASS") {
    DstINumUses--; // Ignore the class constraint.
    ExpectedDstINumUses--;
  }

  // NumResults - This is the number of results produced by the instruction in
  // the "outs" list.
  unsigned NumResults = OrigDstI->Operands.NumDefs;

  // Number of operands we know the output instruction must have. If it is
  // variadic, we could have more operands.
  unsigned NumFixedOperands = DstI->Operands.size();

  // Loop over all of the fixed operands of the instruction pattern, emitting
  // code to fill them all in. The node 'N' usually has number children equal to
  // the number of input operands of the instruction.  However, in cases where
  // there are predicate operands for an instruction, we need to fill in the
  // 'execute always' values. Match up the node operands to the instruction
  // operands to do this.
  unsigned Child = 0;

  // Similarly to the code in TreePatternNode::ApplyTypeConstraints, count the
  // number of operands at the end of the list which have default values.
  // Those can come from the pattern if it provides enough arguments, or be
  // filled in with the default if the pattern hasn't provided them. But any
  // operand with a default value _before_ the last mandatory one will be
  // filled in with their defaults unconditionally.
  unsigned NonOverridableOperands = NumFixedOperands;
  while (NonOverridableOperands > NumResults &&
         CGP.operandHasDefault(DstI->Operands[NonOverridableOperands - 1].Rec))
    --NonOverridableOperands;

  unsigned NumDefaultOps = 0;
  for (unsigned I = 0; I != DstINumUses; ++I) {
    unsigned InstOpNo = DstI->Operands.NumDefs + I;

    // Determine what to emit for this operand.
    const Record *OperandNode = DstI->Operands[InstOpNo].Rec;

    // If the operand has default values, introduce them now.
    if (CGP.operandHasDefault(OperandNode) &&
        (InstOpNo < NonOverridableOperands || Child >= Dst.getNumChildren())) {
      // This is a predicate or optional def operand which the pattern has not
      // overridden, or which we aren't letting it override; emit the 'default
      // ops' operands.
      for (const TreePatternNode &OpNode :
           make_pointee_range(CGP.getDefaultOperand(OperandNode).DefaultOps)) {
        if (Error Err = importNodeRenderer(M, DstMIBuilder, OpNode, InsertPt))
          return Err;
      }

      ++NumDefaultOps;
      continue;
    }

    if (Error Err =
            importNodeRenderer(M, DstMIBuilder, Dst.getChild(Child), InsertPt))
      return Err;

    ++Child;
  }

  if (NumDefaultOps + ExpectedDstINumUses != DstINumUses)
    return failedImport("Expected " + llvm::to_string(DstINumUses) +
                        " used operands but found " +
                        llvm::to_string(ExpectedDstINumUses) +
                        " explicit ones and " + llvm::to_string(NumDefaultOps) +
                        " default ones");

  return InsertPt;
}

Error GlobalISelEmitter::importImplicitDefRenderers(
    BuildMIAction &DstMIBuilder, ArrayRef<const Record *> ImplicitDefs) const {
  if (!ImplicitDefs.empty())
    return failedImport("Pattern defines a physical register");
  return Error::success();
}

Error GlobalISelEmitter::constrainOperands(action_iterator InsertPt,
                                           RuleMatcher &M, unsigned InsnID,
                                           const TreePatternNode &Dst) const {
  const Record *DstOp = Dst.getOperator();
  const CodeGenInstruction &DstI = Target.getInstruction(DstOp);
  StringRef DstIName = DstI.getName();

  if (DstIName == "COPY_TO_REGCLASS") {
    // COPY_TO_REGCLASS does not provide operand constraints itself but the
    // result is constrained to the class given by the second child.
    const Record *DstIOpRec =
        getInitValueAsRegClass(Dst.getChild(1).getLeafValue());

    if (DstIOpRec == nullptr)
      return failedImport("COPY_TO_REGCLASS operand #1 isn't a register class");

    M.insertAction<ConstrainOperandToRegClassAction>(
        InsertPt, InsnID, 0, Target.getRegisterClass(DstIOpRec));
  } else if (DstIName == "EXTRACT_SUBREG") {
    const CodeGenRegisterClass *SuperClass =
        inferRegClassFromPattern(Dst.getChild(0));
    if (!SuperClass)
      return failedImport(
          "Cannot infer register class from EXTRACT_SUBREG operand #0");

    const CodeGenSubRegIndex *SubIdx = inferSubRegIndexForNode(Dst.getChild(1));
    if (!SubIdx)
      return failedImport("EXTRACT_SUBREG child #1 is not a subreg index");

    // It would be nice to leave this constraint implicit but we're required
    // to pick a register class so constrain the result to a register class
    // that can hold the correct MVT.
    //
    // FIXME: This may introduce an extra copy if the chosen class doesn't
    //        actually contain the subregisters.
    const auto SrcRCDstRCPair =
        SuperClass->getMatchingSubClassWithSubRegs(CGRegs, SubIdx);
    if (!SrcRCDstRCPair) {
      return failedImport("subreg index is incompatible "
                          "with inferred reg class");
    }

    assert(SrcRCDstRCPair->second && "Couldn't find a matching subclass");
    M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, 0,
                                                     *SrcRCDstRCPair->second);
    M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, 1,
                                                     *SrcRCDstRCPair->first);
  } else if (DstIName == "INSERT_SUBREG") {
    // We need to constrain the destination, a super regsister source, and a
    // subregister source.
    const CodeGenRegisterClass *SubClass =
        inferRegClassFromPattern(Dst.getChild(1));
    if (!SubClass)
      return failedImport(
          "Cannot infer register class from INSERT_SUBREG operand #1");
    const CodeGenRegisterClass *SuperClass = inferSuperRegisterClassForNode(
        Dst.getExtType(0), Dst.getChild(0), Dst.getChild(2));
    if (!SuperClass)
      return failedImport(
          "Cannot infer register class for INSERT_SUBREG operand #0");
    M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, 0,
                                                     *SuperClass);
    M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, 1,
                                                     *SuperClass);
    M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, 2,
                                                     *SubClass);
  } else if (DstIName == "SUBREG_TO_REG") {
    // We need to constrain the destination and subregister source.
    // Attempt to infer the subregister source from the first child. If it has
    // an explicitly given register class, we'll use that. Otherwise, we will
    // fail.
    const CodeGenRegisterClass *SubClass =
        inferRegClassFromPattern(Dst.getChild(1));
    if (!SubClass)
      return failedImport(
          "Cannot infer register class from SUBREG_TO_REG child #1");
    // We don't have a child to look at that might have a super register node.
    const CodeGenRegisterClass *SuperClass =
        inferSuperRegisterClass(Dst.getExtType(0), Dst.getChild(2));
    if (!SuperClass)
      return failedImport(
          "Cannot infer register class for SUBREG_TO_REG operand #0");
    M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, 0,
                                                     *SuperClass);
    M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, 2,
                                                     *SubClass);
  } else if (DstIName == "REG_SEQUENCE") {
    const CodeGenRegisterClass *SuperClass =
        inferRegClassFromPattern(Dst.getChild(0));

    M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, 0,
                                                     *SuperClass);

    unsigned Num = Dst.getNumChildren();
    for (unsigned I = 1; I != Num; I += 2) {
      const TreePatternNode &SubRegChild = Dst.getChild(I + 1);

      const CodeGenSubRegIndex *SubIdx = inferSubRegIndexForNode(SubRegChild);
      if (!SubIdx)
        return failedImport("REG_SEQUENCE child is not a subreg index");

      const auto SrcRCDstRCPair =
          SuperClass->getMatchingSubClassWithSubRegs(CGRegs, SubIdx);

      M.insertAction<ConstrainOperandToRegClassAction>(InsertPt, InsnID, I,
                                                       *SrcRCDstRCPair->second);
    }
  } else {
    M.insertAction<ConstrainOperandsToDefinitionAction>(InsertPt, InsnID);
  }

  return Error::success();
}

const CodeGenRegisterClass *
GlobalISelEmitter::getRegClassFromLeaf(const TreePatternNode &Leaf) const {
  assert(Leaf.isLeaf() && "Expected leaf?");
  const Record *RCRec = getInitValueAsRegClass(Leaf.getLeafValue());
  if (!RCRec)
    return nullptr;
  return CGRegs.getRegClass(RCRec);
}

const CodeGenRegisterClass *
GlobalISelEmitter::inferRegClassFromPattern(const TreePatternNode &N) const {
  if (N.isLeaf())
    return getRegClassFromLeaf(N);

  // We don't have a leaf node, so we have to try and infer something. Check
  // that we have an instruction that we can infer something from.

  // Only handle things that produce at least one value (if multiple values,
  // just take the first one).
  if (N.getNumTypes() < 1)
    return nullptr;
  const Record *OpRec = N.getOperator();

  // We only want instructions.
  if (!OpRec->isSubClassOf("Instruction"))
    return nullptr;

  // Don't want to try and infer things when there could potentially be more
  // than one candidate register class.
  return inferRegClassFromInstructionPattern(N, /*ResIdx=*/0);
}

const CodeGenRegisterClass *
GlobalISelEmitter::inferRegClassFromInstructionPattern(const TreePatternNode &N,
                                                       unsigned ResIdx) const {
  const CodeGenInstruction &Inst = Target.getInstruction(N.getOperator());
  assert(ResIdx < Inst.Operands.NumDefs &&
         "Can only infer register class for explicit defs");

  // Handle any special-case instructions which we can safely infer register
  // classes from.
  StringRef InstName = Inst.getName();
  if (InstName == "REG_SEQUENCE") {
    // (outs $super_dst), (ins $dst_regclass, variable_ops)
    // Destination register class is explicitly specified by the first operand.
    const TreePatternNode &RCChild = N.getChild(0);
    if (!RCChild.isLeaf())
      return nullptr;
    return getRegClassFromLeaf(RCChild);
  }

  if (InstName == "COPY_TO_REGCLASS") {
    // (outs $dst), (ins $src, $dst_regclass)
    // Destination register class is explicitly specified by the second operand.
    const TreePatternNode &RCChild = N.getChild(1);
    if (!RCChild.isLeaf())
      return nullptr;
    return getRegClassFromLeaf(RCChild);
  }

  if (InstName == "INSERT_SUBREG") {
    // (outs $super_dst), (ins $super_src, $sub_src, $sub_idx);
    // If we can infer the register class for the first operand, use that.
    // Otherwise, find a register class that supports both the specified
    // sub-register index and the type of the instruction's result.
    const TreePatternNode &Child0 = N.getChild(0);
    assert(Child0.getNumTypes() == 1 && "Unexpected number of types!");
    return inferSuperRegisterClassForNode(N.getExtType(0), Child0,
                                          N.getChild(2));
  }

  if (InstName == "EXTRACT_SUBREG") {
    // (outs $sub_dst), (ins $super_src, $sub_idx)
    // Find a register class that can be used for a sub-register copy from
    // the specified source at the specified sub-register index.
    const CodeGenRegisterClass *SuperRC =
        inferRegClassFromPattern(N.getChild(0));
    if (!SuperRC)
      return nullptr;

    const CodeGenSubRegIndex *SubIdx = inferSubRegIndexForNode(N.getChild(1));
    if (!SubIdx)
      return nullptr;

    const auto SubRCAndSubRegRC =
        SuperRC->getMatchingSubClassWithSubRegs(CGRegs, SubIdx);
    if (!SubRCAndSubRegRC)
      return nullptr;

    return SubRCAndSubRegRC->second;
  }

  if (InstName == "SUBREG_TO_REG") {
    // (outs $super_dst), (ins $super_src, $sub_src, $sub_idx)
    // Find a register class that supports both the specified sub-register
    // index and the type of the instruction's result.
    return inferSuperRegisterClass(N.getExtType(0), N.getChild(2));
  }

  // Handle destination record types that we can safely infer a register class
  // from.
  const auto &DstIOperand = Inst.Operands[ResIdx];
  const Record *DstIOpRec = DstIOperand.Rec;
  if (DstIOpRec->isSubClassOf("RegisterOperand"))
    return &Target.getRegisterClass(DstIOpRec->getValueAsDef("RegClass"));

  if (DstIOpRec->isSubClassOf("RegisterClass"))
    return &Target.getRegisterClass(DstIOpRec);

  return nullptr;
}

const CodeGenRegisterClass *GlobalISelEmitter::inferSuperRegisterClass(
    const TypeSetByHwMode &Ty, const TreePatternNode &SubRegIdxNode) const {
  // We need a ValueTypeByHwMode for getSuperRegForSubReg.
  if (!Ty.isValueTypeByHwMode(false))
    return nullptr;
  if (!SubRegIdxNode.isLeaf())
    return nullptr;
  const DefInit *SubRegInit = dyn_cast<DefInit>(SubRegIdxNode.getLeafValue());
  if (!SubRegInit)
    return nullptr;
  const CodeGenSubRegIndex *SubIdx = CGRegs.findSubRegIdx(SubRegInit->getDef());

  // Use the information we found above to find a minimal register class which
  // supports the subregister and type we want.
  return CGRegs.getSuperRegForSubReg(Ty.getValueTypeByHwMode(), SubIdx,
                                     /*MustBeAllocatable=*/true);
}

const CodeGenRegisterClass *GlobalISelEmitter::inferSuperRegisterClassForNode(
    const TypeSetByHwMode &Ty, const TreePatternNode &SuperRegNode,
    const TreePatternNode &SubRegIdxNode) const {
  // Check if we already have a defined register class for the super register
  // node. If we do, then we should preserve that rather than inferring anything
  // from the subregister index node. We can assume that whoever wrote the
  // pattern in the first place made sure that the super register and
  // subregister are compatible.
  if (const CodeGenRegisterClass *SuperRegisterClass =
          inferRegClassFromPattern(SuperRegNode))
    return SuperRegisterClass;
  return inferSuperRegisterClass(Ty, SubRegIdxNode);
}

const CodeGenSubRegIndex *GlobalISelEmitter::inferSubRegIndexForNode(
    const TreePatternNode &SubRegIdxNode) const {
  if (!SubRegIdxNode.isLeaf())
    return nullptr;

  const DefInit *SubRegInit = dyn_cast<DefInit>(SubRegIdxNode.getLeafValue());
  if (!SubRegInit)
    return nullptr;
  return CGRegs.findSubRegIdx(SubRegInit->getDef());
}

Expected<RuleMatcher> GlobalISelEmitter::runOnPattern(const PatternToMatch &P) {
  // Keep track of the matchers and actions to emit.
  int Score = P.getPatternComplexity(CGP);
  RuleMatcher M(P.getSrcRecord()->getLoc());
  RuleMatcherScores[M.getRuleID()] = Score;
  M.addAction<DebugCommentAction>(llvm::to_string(P.getSrcPattern()) +
                                  "  =>  " +
                                  llvm::to_string(P.getDstPattern()));

  SmallVector<const Record *, 4> Predicates;
  P.getPredicateRecords(Predicates);
  if (auto Error = importRulePredicates(M, Predicates))
    return std::move(Error);

  if (!P.getHwModeFeatures().empty())
    M.addHwModeIdx(declareHwModeCheck(P.getHwModeFeatures()));

  // Next, analyze the pattern operators.
  TreePatternNode &Src = P.getSrcPattern();
  TreePatternNode &Dst = P.getDstPattern();

  // If the root of either pattern isn't a simple operator, ignore it.
  if (auto Err = isTrivialOperatorNode(Dst))
    return failedImport("Dst pattern root isn't a trivial operator (" +
                        toString(std::move(Err)) + ")");
  if (auto Err = isTrivialOperatorNode(Src))
    return failedImport("Src pattern root isn't a trivial operator (" +
                        toString(std::move(Err)) + ")");

  // The different predicates and matchers created during
  // addInstructionMatcher use the RuleMatcher M to set up their
  // instruction ID (InsnVarID) that are going to be used when
  // M is going to be emitted.
  // However, the code doing the emission still relies on the IDs
  // returned during that process by the RuleMatcher when issuing
  // the recordInsn opcodes.
  // Because of that:
  // 1. The order in which we created the predicates
  //    and such must be the same as the order in which we emit them,
  //    and
  // 2. We need to reset the generation of the IDs in M somewhere between
  //    addInstructionMatcher and emit
  //
  // FIXME: Long term, we don't want to have to rely on this implicit
  // naming being the same. One possible solution would be to have
  // explicit operator for operation capture and reference those.
  // The plus side is that it would expose opportunities to share
  // the capture accross rules. The downside is that it would
  // introduce a dependency between predicates (captures must happen
  // before their first use.)
  InstructionMatcher &InsnMatcherTemp = M.addInstructionMatcher(Src.getName());
  unsigned TempOpIdx = 0;

  const auto SavedFlags = M.setGISelFlags(P.getSrcRecord());

  auto InsnMatcherOrError =
      createAndImportSelDAGMatcher(M, InsnMatcherTemp, Src, TempOpIdx);
  if (auto Error = InsnMatcherOrError.takeError())
    return std::move(Error);
  InstructionMatcher &InsnMatcher = InsnMatcherOrError.get();

  if (Dst.isLeaf()) {
    if (const Record *RCDef = getInitValueAsRegClass(Dst.getLeafValue())) {
      const CodeGenRegisterClass &RC = Target.getRegisterClass(RCDef);

      // We need to replace the def and all its uses with the specified
      // operand. However, we must also insert COPY's wherever needed.
      // For now, emit a copy and let the register allocator clean up.
      auto &DstI = Target.getInstruction(RK.getDef("COPY"));
      const auto &DstIOperand = DstI.Operands[0];

      OperandMatcher &OM0 = InsnMatcher.getOperand(0);
      OM0.setSymbolicName(DstIOperand.Name);
      M.defineOperand(OM0.getSymbolicName(), OM0);
      OM0.addPredicate<RegisterBankOperandMatcher>(RC);

      auto &DstMIBuilder =
          M.addAction<BuildMIAction>(M.allocateOutputInsnID(), &DstI);
      DstMIBuilder.addRenderer<CopyRenderer>(DstIOperand.Name);
      DstMIBuilder.addRenderer<CopyRenderer>(Dst.getName());
      M.addAction<ConstrainOperandToRegClassAction>(0, 0, RC);

      // Erase the root.
      unsigned RootInsnID = M.getInsnVarID(InsnMatcher);
      M.addAction<EraseInstAction>(RootInsnID);

      // We're done with this pattern!  It's eligible for GISel emission; return
      // it.
      ++NumPatternImported;
      return std::move(M);
    }

    return failedImport("Dst pattern root isn't a known leaf");
  }

  // Start with the defined operands (i.e., the results of the root operator).
  const Record *DstOp = Dst.getOperator();
  if (!DstOp->isSubClassOf("Instruction"))
    return failedImport("Pattern operator isn't an instruction");

  const CodeGenInstruction &DstI = Target.getInstruction(DstOp);

  // Count both implicit and explicit defs in the dst instruction.
  // This avoids errors importing patterns that have inherent implicit defs.
  unsigned DstExpDefs = DstI.Operands.NumDefs,
           DstNumDefs = DstI.ImplicitDefs.size() + DstExpDefs,
           SrcNumDefs = Src.getExtTypes().size();
  if (DstNumDefs < SrcNumDefs) {
    if (DstNumDefs != 0)
      return failedImport("Src pattern result has more defs than dst MI (" +
                          to_string(SrcNumDefs) + " def(s) vs " +
                          to_string(DstNumDefs) + " def(s))");

    bool FoundNoUsePred = false;
    for (const auto &Pred : InsnMatcher.predicates()) {
      if ((FoundNoUsePred = isa<NoUsePredicateMatcher>(Pred.get())))
        break;
    }
    if (!FoundNoUsePred)
      return failedImport("Src pattern result has " + to_string(SrcNumDefs) +
                          " def(s) without the HasNoUse predicate set to true "
                          "but Dst MI has no def");
  }

  // The root of the match also has constraints on the register bank so that it
  // matches the result instruction.
  unsigned N = std::min(DstExpDefs, SrcNumDefs);
  for (unsigned I = 0; I < N; ++I) {
    const auto &DstIOperand = DstI.Operands[I];

    OperandMatcher &OM = InsnMatcher.getOperand(I);
    // The operand names declared in the DstI instruction are unrelated to
    // those used in pattern's source and destination DAGs, so mangle the
    // former to prevent implicitly adding unexpected
    // GIM_CheckIsSameOperand predicates by the defineOperand method.
    OM.setSymbolicName(getMangledRootDefName(DstIOperand.Name));
    M.defineOperand(OM.getSymbolicName(), OM);

    const CodeGenRegisterClass *RC =
        inferRegClassFromInstructionPattern(Dst, I);
    if (!RC)
      return failedImport("Could not infer register class for result #" +
                          Twine(I) + " from pattern " + to_string(Dst));
    OM.addPredicate<RegisterBankOperandMatcher>(*RC);
  }

  auto DstMIBuilderOrError =
      createAndImportInstructionRenderer(M, InsnMatcher, Dst);
  if (auto Error = DstMIBuilderOrError.takeError())
    return std::move(Error);
  BuildMIAction &DstMIBuilder = DstMIBuilderOrError.get();

  // Render the implicit defs.
  // These are only added to the root of the result.
  if (auto Error = importImplicitDefRenderers(DstMIBuilder, P.getDstRegs()))
    return std::move(Error);

  DstMIBuilder.chooseInsnToMutate(M);

  // Constrain the registers to classes. This is normally derived from the
  // emitted instruction but a few instructions require special handling.
  if (auto Error =
          constrainOperands(M.actions_end(), M, DstMIBuilder.getInsnID(), Dst))
    return std::move(Error);

  // Erase the root.
  unsigned RootInsnID = M.getInsnVarID(InsnMatcher);
  M.addAction<EraseInstAction>(RootInsnID);

  // We're done with this pattern!  It's eligible for GISel emission; return it.
  ++NumPatternImported;
  return std::move(M);
}

MatchTable
GlobalISelEmitter::buildMatchTable(MutableArrayRef<RuleMatcher> Rules,
                                   bool Optimize, bool WithCoverage) {
  std::vector<Matcher *> InputRules;
  for (Matcher &Rule : Rules)
    InputRules.push_back(&Rule);

  if (!Optimize)
    return MatchTable::buildTable(InputRules, WithCoverage);

  unsigned CurrentOrdering = 0;
  StringMap<unsigned> OpcodeOrder;
  for (RuleMatcher &Rule : Rules) {
    const StringRef Opcode = Rule.getOpcode();
    assert(!Opcode.empty() && "Didn't expect an undefined opcode");
    if (OpcodeOrder.try_emplace(Opcode, CurrentOrdering).second)
      ++CurrentOrdering;
  }

  llvm::stable_sort(
      InputRules, [&OpcodeOrder](const Matcher *A, const Matcher *B) {
        auto *L = static_cast<const RuleMatcher *>(A);
        auto *R = static_cast<const RuleMatcher *>(B);
        return std::tuple(OpcodeOrder[L->getOpcode()],
                          L->insnmatchers_front().getNumOperandMatchers()) <
               std::tuple(OpcodeOrder[R->getOpcode()],
                          R->insnmatchers_front().getNumOperandMatchers());
      });

  for (Matcher *Rule : InputRules)
    Rule->optimize();

  std::vector<std::unique_ptr<Matcher>> MatcherStorage;
  std::vector<Matcher *> OptRules =
      optimizeRules<GroupMatcher>(InputRules, MatcherStorage);

  for (Matcher *Rule : OptRules)
    Rule->optimize();

  OptRules = optimizeRules<SwitchMatcher>(OptRules, MatcherStorage);

  return MatchTable::buildTable(OptRules, WithCoverage);
}

void GlobalISelEmitter::emitAdditionalImpl(raw_ostream &OS) {
  OS << "bool " << getClassName()
     << "::selectImpl(MachineInstr &I, CodeGenCoverage "
        "&CoverageInfo) const {\n"
     << "  const PredicateBitset AvailableFeatures = "
        "getAvailableFeatures();\n"
     << "  MachineIRBuilder B(I);\n"
     << "  State.MIs.clear();\n"
     << "  State.MIs.push_back(&I);\n\n"
     << "  if (executeMatchTable(*this, State, ExecInfo, B"
     << ", getMatchTable(), TII, MF->getRegInfo(), TRI, RBI, AvailableFeatures"
     << ", &CoverageInfo)) {\n"
     << "    return true;\n"
     << "  }\n\n"
     << "  return false;\n"
     << "}\n\n";
}

void GlobalISelEmitter::emitMIPredicateFns(raw_ostream &OS) {
  std::vector<const Record *> MatchedRecords;
  llvm::copy_if(AllPatFrags, std::back_inserter(MatchedRecords),
                [](const Record *R) {
                  return !R->getValueAsString("GISelPredicateCode").empty();
                });
  emitMIPredicateFnsImpl<const Record *>(
      OS,
      "  const MachineFunction &MF = *MI.getParent()->getParent();\n"
      "  const MachineRegisterInfo &MRI = MF.getRegInfo();\n"
      "  const auto &Operands = State.RecordedOperands;\n"
      "  (void)Operands;\n"
      "  (void)MRI;",
      ArrayRef<const Record *>(MatchedRecords), &getPatFragPredicateEnumName,
      [](const Record *R) { return R->getValueAsString("GISelPredicateCode"); },
      "PatFrag predicates.");
}

void GlobalISelEmitter::emitLeafPredicateFns(raw_ostream &OS) {
  std::vector<const Record *> MatchedRecords;
  llvm::copy_if(AllPatFrags, std::back_inserter(MatchedRecords),
                [](const Record *R) {
                  return (!R->getValueAsOptionalString("GISelLeafPredicateCode")
                               .value_or(std::string())
                               .empty());
                });
  emitLeafPredicateFnsImpl<const Record *>(
      OS,
      "  const auto &Operands = State.RecordedOperands;\n"
      "  Register Reg = MO.getReg();\n"
      "  (void)Operands;\n"
      "  (void)Reg;",
      ArrayRef<const Record *>(MatchedRecords), &getPatFragPredicateEnumName,
      [](const Record *R) {
        return R->getValueAsString("GISelLeafPredicateCode");
      },
      "PatFrag predicates.");
}

void GlobalISelEmitter::emitI64ImmPredicateFns(raw_ostream &OS) {
  std::vector<const Record *> MatchedRecords;
  llvm::copy_if(AllPatFrags, std::back_inserter(MatchedRecords),
                [](const Record *R) {
                  bool Unset;
                  return !R->getValueAsString("ImmediateCode").empty() &&
                         !R->getValueAsBitOrUnset("IsAPFloat", Unset) &&
                         !R->getValueAsBit("IsAPInt");
                });
  emitImmPredicateFnsImpl<const Record *>(
      OS, "I64", "int64_t", ArrayRef<const Record *>(MatchedRecords),
      &getPatFragPredicateEnumName,
      [](const Record *R) { return R->getValueAsString("ImmediateCode"); },
      "PatFrag predicates.");
}

void GlobalISelEmitter::emitAPFloatImmPredicateFns(raw_ostream &OS) {
  std::vector<const Record *> MatchedRecords;
  llvm::copy_if(AllPatFrags, std::back_inserter(MatchedRecords),
                [](const Record *R) {
                  bool Unset;
                  return !R->getValueAsString("ImmediateCode").empty() &&
                         R->getValueAsBitOrUnset("IsAPFloat", Unset);
                });
  emitImmPredicateFnsImpl<const Record *>(
      OS, "APFloat", "const APFloat &",
      ArrayRef<const Record *>(MatchedRecords), &getPatFragPredicateEnumName,
      [](const Record *R) { return R->getValueAsString("ImmediateCode"); },
      "PatFrag predicates.");
}

void GlobalISelEmitter::emitAPIntImmPredicateFns(raw_ostream &OS) {
  std::vector<const Record *> MatchedRecords;
  llvm::copy_if(AllPatFrags, std::back_inserter(MatchedRecords),
                [](const Record *R) {
                  return !R->getValueAsString("ImmediateCode").empty() &&
                         R->getValueAsBit("IsAPInt");
                });
  emitImmPredicateFnsImpl<const Record *>(
      OS, "APInt", "const APInt &", ArrayRef<const Record *>(MatchedRecords),
      &getPatFragPredicateEnumName,
      [](const Record *R) { return R->getValueAsString("ImmediateCode"); },
      "PatFrag predicates.");
}

void GlobalISelEmitter::emitTestSimplePredicate(raw_ostream &OS) {
  OS << "bool " << getClassName() << "::testSimplePredicate(unsigned) const {\n"
     << "    llvm_unreachable(\"" + getClassName() +
            " does not support simple predicates!\");\n"
     << "  return false;\n"
     << "}\n";
}

void GlobalISelEmitter::emitRunCustomAction(raw_ostream &OS) {
  OS << "bool " << getClassName()
     << "::runCustomAction(unsigned, const MatcherState&, NewMIVector &) const "
        "{\n"
     << "    llvm_unreachable(\"" + getClassName() +
            " does not support custom C++ actions!\");\n"
     << "}\n";
}

bool hasBFloatType(const TreePatternNode &Node) {
  for (unsigned I = 0, E = Node.getNumTypes(); I < E; I++) {
    auto Ty = Node.getType(I);
    for (auto T : Ty)
      if (T.second == MVT::bf16 ||
          (T.second.isVector() && T.second.getScalarType() == MVT::bf16))
        return true;
  }
  for (const TreePatternNode &C : Node.children())
    if (hasBFloatType(C))
      return true;
  return false;
}

void GlobalISelEmitter::run(raw_ostream &OS) {
  if (!UseCoverageFile.empty()) {
    RuleCoverage = CodeGenCoverage();
    auto RuleCoverageBufOrErr = MemoryBuffer::getFile(UseCoverageFile);
    if (!RuleCoverageBufOrErr) {
      PrintWarning(SMLoc(), "Missing rule coverage data");
      RuleCoverage = std::nullopt;
    } else {
      if (!RuleCoverage->parse(*RuleCoverageBufOrErr.get(), Target.getName())) {
        PrintWarning(SMLoc(), "Ignoring invalid or missing rule coverage data");
        RuleCoverage = std::nullopt;
      }
    }
  }

  // Track the run-time opcode values
  gatherOpcodeValues();
  // Track the run-time LLT ID values
  gatherTypeIDValues();

  // Track the GINodeEquiv definitions.
  gatherNodeEquivs();

  AllPatFrags = RK.getAllDerivedDefinitions("PatFrags");

  emitSourceFileHeader(
      ("Global Instruction Selector for the " + Target.getName() + " target")
          .str(),
      OS);
  std::vector<RuleMatcher> Rules;
  // Look through the SelectionDAG patterns we found, possibly emitting some.
  for (const PatternToMatch &Pat : CGP.ptms()) {
    ++NumPatternTotal;

    if (Pat.getGISelShouldIgnore())
      continue; // skip without warning

    // Skip any patterns containing BF16 types, as GISel cannot currently tell
    // the difference between fp16 and bf16. FIXME: This can be removed once
    // BF16 is supported properly.
    if (hasBFloatType(Pat.getSrcPattern()))
      continue;

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

    if (RuleCoverage) {
      if (RuleCoverage->isCovered(MatcherOrErr->getRuleID()))
        ++NumPatternsTested;
      else
        PrintWarning(Pat.getSrcRecord()->getLoc(),
                     "Pattern is not covered by a test");
    }
    Rules.push_back(std::move(MatcherOrErr.get()));
  }

  // Comparison function to order records by name.
  auto OrderByName = [](const Record *A, const Record *B) {
    return A->getName() < B->getName();
  };

  std::vector<const Record *> ComplexPredicates =
      RK.getAllDerivedDefinitions("GIComplexOperandMatcher");
  llvm::sort(ComplexPredicates, OrderByName);

  std::vector<StringRef> CustomRendererFns;
  transform(RK.getAllDerivedDefinitions("GICustomOperandRenderer"),
            std::back_inserter(CustomRendererFns), [](const auto &Record) {
              return Record->getValueAsString("RendererFn");
            });
  // Sort and remove duplicates to get a list of unique renderer functions, in
  // case some were mentioned more than once.
  llvm::sort(CustomRendererFns);
  CustomRendererFns.erase(llvm::unique(CustomRendererFns),
                          CustomRendererFns.end());

  // Create a table containing the LLT objects needed by the matcher and an enum
  // for the matcher to reference them with.
  std::vector<LLTCodeGen> TypeObjects;
  append_range(TypeObjects, KnownTypes);
  llvm::sort(TypeObjects);

  // Sort rules.
  llvm::stable_sort(Rules, [&](const RuleMatcher &A, const RuleMatcher &B) {
    int ScoreA = RuleMatcherScores[A.getRuleID()];
    int ScoreB = RuleMatcherScores[B.getRuleID()];
    if (ScoreA > ScoreB)
      return true;
    if (ScoreB > ScoreA)
      return false;
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

  // Build match table
  const MatchTable Table =
      buildMatchTable(Rules, OptimizeMatchTable, GenerateCoverage);

  emitPredicateBitset(OS, "GET_GLOBALISEL_PREDICATE_BITSET");
  emitTemporariesDecl(OS, "GET_GLOBALISEL_TEMPORARIES_DECL");
  emitTemporariesInit(OS, MaxTemporaries, "GET_GLOBALISEL_TEMPORARIES_INIT");
  emitExecutorImpl(OS, Table, TypeObjects, Rules, ComplexPredicates,
                   CustomRendererFns, "GET_GLOBALISEL_IMPL");
  emitPredicatesDecl(OS, "GET_GLOBALISEL_PREDICATES_DECL");
  emitPredicatesInit(OS, "GET_GLOBALISEL_PREDICATES_INIT");
}

void GlobalISelEmitter::declareSubtargetFeature(const Record *Predicate) {
  SubtargetFeatures.try_emplace(Predicate, Predicate, SubtargetFeatures.size());
}

unsigned GlobalISelEmitter::declareHwModeCheck(StringRef HwModeFeatures) {
  return HwModes.emplace(HwModeFeatures.str(), HwModes.size()).first->second;
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//

static TableGen::Emitter::OptClass<GlobalISelEmitter>
    X("gen-global-isel", "Generate GlobalISel selector");
