//===- GlobalCombinerEmitter.cpp - Generate a combiner --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Generate a combiner implementation for GlobalISel from a declarative
/// syntax
///
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "GlobalISel/CodeExpander.h"
#include "GlobalISel/CodeExpansions.h"
#include "GlobalISel/GIMatchDag.h"
#include "GlobalISel/GIMatchDagPredicate.h"
#include "GlobalISel/GIMatchTree.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "gicombiner-emitter"

// FIXME: Use ALWAYS_ENABLED_STATISTIC once it's available.
unsigned NumPatternTotal = 0;
STATISTIC(NumPatternTotalStatistic, "Total number of patterns");

cl::OptionCategory
    GICombinerEmitterCat("Options for -gen-global-isel-combiner");
static cl::list<std::string>
    SelectedCombiners("combiners", cl::desc("Emit the specified combiners"),
                      cl::cat(GICombinerEmitterCat), cl::CommaSeparated);
static cl::opt<bool> ShowExpansions(
    "gicombiner-show-expansions",
    cl::desc("Use C++ comments to indicate occurence of code expansion"),
    cl::cat(GICombinerEmitterCat));
static cl::opt<bool> StopAfterParse(
    "gicombiner-stop-after-parse",
    cl::desc("Stop processing after parsing rules and dump state"),
    cl::cat(GICombinerEmitterCat));
static cl::opt<bool> StopAfterBuild(
    "gicombiner-stop-after-build",
    cl::desc("Stop processing after building the match tree"),
    cl::cat(GICombinerEmitterCat));

namespace {
typedef uint64_t RuleID;

// We're going to be referencing the same small strings quite a lot for operand
// names and the like. Make their lifetime management simple with a global
// string table.
StringSet<> StrTab;

StringRef insertStrTab(StringRef S) {
  if (S.empty())
    return S;
  return StrTab.insert(S).first->first();
}

class format_partition_name {
  const GIMatchTree &Tree;
  unsigned Idx;

public:
  format_partition_name(const GIMatchTree &Tree, unsigned Idx)
      : Tree(Tree), Idx(Idx) {}
  void print(raw_ostream &OS) const {
    Tree.getPartitioner()->emitPartitionName(OS, Idx);
  }
};
raw_ostream &operator<<(raw_ostream &OS, const format_partition_name &Fmt) {
  Fmt.print(OS);
  return OS;
}

/// Declares data that is passed from the match stage to the apply stage.
class MatchDataInfo {
  /// The symbol used in the tablegen patterns
  StringRef PatternSymbol;
  /// The data type for the variable
  StringRef Type;
  /// The name of the variable as declared in the generated matcher.
  std::string VariableName;

public:
  MatchDataInfo(StringRef PatternSymbol, StringRef Type, StringRef VariableName)
      : PatternSymbol(PatternSymbol), Type(Type), VariableName(VariableName) {}

  StringRef getPatternSymbol() const { return PatternSymbol; };
  StringRef getType() const { return Type; };
  StringRef getVariableName() const { return VariableName; };
};

class RootInfo {
  StringRef PatternSymbol;

public:
  RootInfo(StringRef PatternSymbol) : PatternSymbol(PatternSymbol) {}

  StringRef getPatternSymbol() const { return PatternSymbol; }
};

class CombineRule {
public:

  using const_matchdata_iterator = std::vector<MatchDataInfo>::const_iterator;

  struct VarInfo {
    const GIMatchDagInstr *N;
    const GIMatchDagOperand *Op;
    const DagInit *Matcher;

  public:
    VarInfo(const GIMatchDagInstr *N, const GIMatchDagOperand *Op,
            const DagInit *Matcher)
        : N(N), Op(Op), Matcher(Matcher) {}
  };

protected:
  /// A unique ID for this rule
  /// ID's are used for debugging and run-time disabling of rules among other
  /// things.
  RuleID ID;

  /// A unique ID that can be used for anonymous objects belonging to this rule.
  /// Used to create unique names in makeNameForAnon*() without making tests
  /// overly fragile.
  unsigned UID = 0;

  /// The record defining this rule.
  const Record &TheDef;

  /// The roots of a match. These are the leaves of the DAG that are closest to
  /// the end of the function. I.e. the nodes that are encountered without
  /// following any edges of the DAG described by the pattern as we work our way
  /// from the bottom of the function to the top.
  std::vector<RootInfo> Roots;

  GIMatchDag MatchDag;

  /// A block of arbitrary C++ to finish testing the match.
  /// FIXME: This is a temporary measure until we have actual pattern matching
  const StringInit *MatchingFixupCode = nullptr;

  /// The MatchData defined by the match stage and required by the apply stage.
  /// This allows the plumbing of arbitrary data from C++ predicates between the
  /// stages.
  ///
  /// For example, suppose you have:
  ///   %A = <some-constant-expr>
  ///   %0 = G_ADD %1, %A
  /// you could define a GIMatchPredicate that walks %A, constant folds as much
  /// as possible and returns an APInt containing the discovered constant. You
  /// could then declare:
  ///   def apint : GIDefMatchData<"APInt">;
  /// add it to the rule with:
  ///   (defs root:$root, apint:$constant)
  /// evaluate it in the pattern with a C++ function that takes a
  /// MachineOperand& and an APInt& with:
  ///   (match [{MIR %root = G_ADD %0, %A }],
  ///             (constantfold operand:$A, apint:$constant))
  /// and finally use it in the apply stage with:
  ///   (apply (create_operand
  ///                [{ MachineOperand::CreateImm(${constant}.getZExtValue());
  ///                ]}, apint:$constant),
  ///             [{MIR %root = FOO %0, %constant }])
  std::vector<MatchDataInfo> MatchDataDecls;

  void declareMatchData(StringRef PatternSymbol, StringRef Type,
                        StringRef VarName);

  bool parseInstructionMatcher(const CodeGenTarget &Target, StringInit *ArgName,
                               const Init &Arg,
                               StringMap<std::vector<VarInfo>> &NamedEdgeDefs,
                               StringMap<std::vector<VarInfo>> &NamedEdgeUses);
  bool parseWipMatchOpcodeMatcher(const CodeGenTarget &Target,
                                  StringInit *ArgName, const Init &Arg);

public:
  CombineRule(const CodeGenTarget &Target, GIMatchDagContext &Ctx, RuleID ID,
              const Record &R)
      : ID(ID), TheDef(R), MatchDag(Ctx) {}
  CombineRule(const CombineRule &) = delete;

  bool parseDefs();
  bool parseMatcher(const CodeGenTarget &Target);

  RuleID getID() const { return ID; }
  unsigned allocUID() { return UID++; }
  StringRef getName() const { return TheDef.getName(); }
  const Record &getDef() const { return TheDef; }
  const StringInit *getMatchingFixupCode() const { return MatchingFixupCode; }
  size_t getNumRoots() const { return Roots.size(); }

  GIMatchDag &getMatchDag() { return MatchDag; }
  const GIMatchDag &getMatchDag() const { return MatchDag; }

  using const_root_iterator = std::vector<RootInfo>::const_iterator;
  const_root_iterator roots_begin() const { return Roots.begin(); }
  const_root_iterator roots_end() const { return Roots.end(); }
  iterator_range<const_root_iterator> roots() const {
    return llvm::make_range(Roots.begin(), Roots.end());
  }

  iterator_range<const_matchdata_iterator> matchdata_decls() const {
    return make_range(MatchDataDecls.begin(), MatchDataDecls.end());
  }

  /// Export expansions for this rule
  void declareExpansions(CodeExpansions &Expansions) const {
    for (const auto &I : matchdata_decls())
      Expansions.declare(I.getPatternSymbol(), I.getVariableName());
  }

  /// The matcher will begin from the roots and will perform the match by
  /// traversing the edges to cover the whole DAG. This function reverses DAG
  /// edges such that everything is reachable from a root. This is part of the
  /// preparation work for flattening the DAG into a tree.
  void reorientToRoots() {
    SmallSet<const GIMatchDagInstr *, 5> Roots;
    SmallSet<const GIMatchDagInstr *, 5> Visited;
    SmallSet<GIMatchDagEdge *, 20> EdgesRemaining;

    for (auto &I : MatchDag.roots()) {
      Roots.insert(I);
      Visited.insert(I);
    }
    for (auto &I : MatchDag.edges())
      EdgesRemaining.insert(I);

    bool Progressed = false;
    SmallSet<GIMatchDagEdge *, 20> EdgesToRemove;
    while (!EdgesRemaining.empty()) {
      for (auto *EI : EdgesRemaining) {
        if (Visited.count(EI->getFromMI())) {
          if (Roots.count(EI->getToMI()))
            PrintError(TheDef.getLoc(), "One or more roots are unnecessary");
          Visited.insert(EI->getToMI());
          EdgesToRemove.insert(EI);
          Progressed = true;
        }
      }
      for (GIMatchDagEdge *ToRemove : EdgesToRemove)
        EdgesRemaining.erase(ToRemove);
      EdgesToRemove.clear();

      for (auto EI = EdgesRemaining.begin(), EE = EdgesRemaining.end();
           EI != EE; ++EI) {
        if (Visited.count((*EI)->getToMI())) {
          (*EI)->reverse();
          Visited.insert((*EI)->getToMI());
          EdgesToRemove.insert(*EI);
          Progressed = true;
        }
        for (GIMatchDagEdge *ToRemove : EdgesToRemove)
          EdgesRemaining.erase(ToRemove);
        EdgesToRemove.clear();
      }

      if (!Progressed) {
        LLVM_DEBUG(dbgs() << "No progress\n");
        return;
      }
      Progressed = false;
    }
  }
};

/// A convenience function to check that an Init refers to a specific def. This
/// is primarily useful for testing for defs and similar in DagInit's since
/// DagInit's support any type inside them.
static bool isSpecificDef(const Init &N, StringRef Def) {
  if (const DefInit *OpI = dyn_cast<DefInit>(&N))
    if (OpI->getDef()->getName() == Def)
      return true;
  return false;
}

/// A convenience function to check that an Init refers to a def that is a
/// subclass of the given class and coerce it to a def if it is. This is
/// primarily useful for testing for subclasses of GIMatchKind and similar in
/// DagInit's since DagInit's support any type inside them.
static Record *getDefOfSubClass(const Init &N, StringRef Cls) {
  if (const DefInit *OpI = dyn_cast<DefInit>(&N))
    if (OpI->getDef()->isSubClassOf(Cls))
      return OpI->getDef();
  return nullptr;
}

/// A convenience function to check that an Init refers to a dag whose operator
/// is a specific def and coerce it to a dag if it is. This is primarily useful
/// for testing for subclasses of GIMatchKind and similar in DagInit's since
/// DagInit's support any type inside them.
static const DagInit *getDagWithSpecificOperator(const Init &N,
                                                 StringRef Name) {
  if (const DagInit *I = dyn_cast<DagInit>(&N))
    if (I->getNumArgs() > 0)
      if (const DefInit *OpI = dyn_cast<DefInit>(I->getOperator()))
        if (OpI->getDef()->getName() == Name)
          return I;
  return nullptr;
}

/// A convenience function to check that an Init refers to a dag whose operator
/// is a def that is a subclass of the given class and coerce it to a dag if it
/// is. This is primarily useful for testing for subclasses of GIMatchKind and
/// similar in DagInit's since DagInit's support any type inside them.
static const DagInit *getDagWithOperatorOfSubClass(const Init &N,
                                                   StringRef Cls) {
  if (const DagInit *I = dyn_cast<DagInit>(&N))
    if (I->getNumArgs() > 0)
      if (const DefInit *OpI = dyn_cast<DefInit>(I->getOperator()))
        if (OpI->getDef()->isSubClassOf(Cls))
          return I;
  return nullptr;
}

StringRef makeNameForAnonInstr(CombineRule &Rule) {
  return insertStrTab(to_string(
      format("__anon%" PRIu64 "_%u", Rule.getID(), Rule.allocUID())));
}

StringRef makeDebugName(CombineRule &Rule, StringRef Name) {
  return insertStrTab(Name.empty() ? makeNameForAnonInstr(Rule) : StringRef(Name));
}

StringRef makeNameForAnonPredicate(CombineRule &Rule) {
  return insertStrTab(to_string(
      format("__anonpred%" PRIu64 "_%u", Rule.getID(), Rule.allocUID())));
}

void CombineRule::declareMatchData(StringRef PatternSymbol, StringRef Type,
                                   StringRef VarName) {
  MatchDataDecls.emplace_back(PatternSymbol, Type, VarName);
}

bool CombineRule::parseDefs() {
  DagInit *Defs = TheDef.getValueAsDag("Defs");

  if (Defs->getOperatorAsDef(TheDef.getLoc())->getName() != "defs") {
    PrintError(TheDef.getLoc(), "Expected defs operator");
    return false;
  }

  for (unsigned I = 0, E = Defs->getNumArgs(); I < E; ++I) {
    // Roots should be collected into Roots
    if (isSpecificDef(*Defs->getArg(I), "root")) {
      Roots.emplace_back(Defs->getArgNameStr(I));
      continue;
    }

    // Subclasses of GIDefMatchData should declare that this rule needs to pass
    // data from the match stage to the apply stage, and ensure that the
    // generated matcher has a suitable variable for it to do so.
    if (Record *MatchDataRec =
            getDefOfSubClass(*Defs->getArg(I), "GIDefMatchData")) {
      declareMatchData(Defs->getArgNameStr(I),
                       MatchDataRec->getValueAsString("Type"),
                       llvm::to_string(llvm::format("MatchData%" PRIu64, ID)));
      continue;
    }

    // Otherwise emit an appropriate error message.
    if (getDefOfSubClass(*Defs->getArg(I), "GIDefKind"))
      PrintError(TheDef.getLoc(),
                 "This GIDefKind not implemented in tablegen");
    else if (getDefOfSubClass(*Defs->getArg(I), "GIDefKindWithArgs"))
      PrintError(TheDef.getLoc(),
                 "This GIDefKindWithArgs not implemented in tablegen");
    else
      PrintError(TheDef.getLoc(),
                 "Expected a subclass of GIDefKind or a sub-dag whose "
                 "operator is of type GIDefKindWithArgs");
    return false;
  }

  if (Roots.empty()) {
    PrintError(TheDef.getLoc(), "Combine rules must have at least one root");
    return false;
  }
  return true;
}

// Parse an (Instruction $a:Arg1, $b:Arg2, ...) matcher. Edges are formed
// between matching operand names between different matchers.
bool CombineRule::parseInstructionMatcher(
    const CodeGenTarget &Target, StringInit *ArgName, const Init &Arg,
    StringMap<std::vector<VarInfo>> &NamedEdgeDefs,
    StringMap<std::vector<VarInfo>> &NamedEdgeUses) {
  if (const DagInit *Matcher =
          getDagWithOperatorOfSubClass(Arg, "Instruction")) {
    auto &Instr =
        Target.getInstruction(Matcher->getOperatorAsDef(TheDef.getLoc()));

    StringRef Name = ArgName ? ArgName->getValue() : "";

    GIMatchDagInstr *N =
        MatchDag.addInstrNode(makeDebugName(*this, Name), insertStrTab(Name),
                              MatchDag.getContext().makeOperandList(Instr));

    N->setOpcodeAnnotation(&Instr);
    const auto &P = MatchDag.addPredicateNode<GIMatchDagOpcodePredicate>(
        makeNameForAnonPredicate(*this), Instr);
    MatchDag.addPredicateDependency(N, nullptr, P, &P->getOperandInfo()["mi"]);
    unsigned OpIdx = 0;
    for (const auto &NameInit : Matcher->getArgNames()) {
      StringRef Name = insertStrTab(NameInit->getAsUnquotedString());
      if (Name.empty())
        continue;
      N->assignNameToOperand(OpIdx, Name);

      // Record the endpoints of any named edges. We'll add the cartesian
      // product of edges later.
      const auto &InstrOperand = N->getOperandInfo()[OpIdx];
      if (InstrOperand.isDef()) {
        NamedEdgeDefs.try_emplace(Name);
        NamedEdgeDefs[Name].emplace_back(N, &InstrOperand, Matcher);
      } else {
        NamedEdgeUses.try_emplace(Name);
        NamedEdgeUses[Name].emplace_back(N, &InstrOperand, Matcher);
      }

      if (InstrOperand.isDef()) {
        if (any_of(Roots, [&](const RootInfo &X) {
              return X.getPatternSymbol() == Name;
            })) {
          N->setMatchRoot();
        }
      }

      OpIdx++;
    }

    return true;
  }
  return false;
}

// Parse the wip_match_opcode placeholder that's temporarily present in lieu of
// implementing macros or choices between two matchers.
bool CombineRule::parseWipMatchOpcodeMatcher(const CodeGenTarget &Target,
                                             StringInit *ArgName,
                                             const Init &Arg) {
  if (const DagInit *Matcher =
          getDagWithSpecificOperator(Arg, "wip_match_opcode")) {
    StringRef Name = ArgName ? ArgName->getValue() : "";

    GIMatchDagInstr *N =
        MatchDag.addInstrNode(makeDebugName(*this, Name), insertStrTab(Name),
                              MatchDag.getContext().makeEmptyOperandList());

    if (any_of(Roots, [&](const RootInfo &X) {
          return ArgName && X.getPatternSymbol() == ArgName->getValue();
        })) {
      N->setMatchRoot();
    }

    const auto &P = MatchDag.addPredicateNode<GIMatchDagOneOfOpcodesPredicate>(
        makeNameForAnonPredicate(*this));
    MatchDag.addPredicateDependency(N, nullptr, P, &P->getOperandInfo()["mi"]);
    // Each argument is an opcode that will pass this predicate. Add them all to
    // the predicate implementation
    for (const auto &Arg : Matcher->getArgs()) {
      Record *OpcodeDef = getDefOfSubClass(*Arg, "Instruction");
      if (OpcodeDef) {
        P->addOpcode(&Target.getInstruction(OpcodeDef));
        continue;
      }
      PrintError(TheDef.getLoc(),
                 "Arguments to wip_match_opcode must be instructions");
      return false;
    }
    return true;
  }
  return false;
}
bool CombineRule::parseMatcher(const CodeGenTarget &Target) {
  StringMap<std::vector<VarInfo>> NamedEdgeDefs;
  StringMap<std::vector<VarInfo>> NamedEdgeUses;
  DagInit *Matchers = TheDef.getValueAsDag("Match");

  if (Matchers->getOperatorAsDef(TheDef.getLoc())->getName() != "match") {
    PrintError(TheDef.getLoc(), "Expected match operator");
    return false;
  }

  if (Matchers->getNumArgs() == 0) {
    PrintError(TheDef.getLoc(), "Matcher is empty");
    return false;
  }

  // The match section consists of a list of matchers and predicates. Parse each
  // one and add the equivalent GIMatchDag nodes, predicates, and edges.
  for (unsigned I = 0; I < Matchers->getNumArgs(); ++I) {
    if (parseInstructionMatcher(Target, Matchers->getArgName(I),
                                *Matchers->getArg(I), NamedEdgeDefs,
                                NamedEdgeUses))
      continue;

    if (parseWipMatchOpcodeMatcher(Target, Matchers->getArgName(I),
                                   *Matchers->getArg(I)))
      continue;


    // Parse arbitrary C++ code we have in lieu of supporting MIR matching
    if (const StringInit *StringI = dyn_cast<StringInit>(Matchers->getArg(I))) {
      assert(!MatchingFixupCode &&
             "Only one block of arbitrary code is currently permitted");
      MatchingFixupCode = StringI;
      MatchDag.setHasPostMatchPredicate(true);
      continue;
    }

    PrintError(TheDef.getLoc(),
               "Expected a subclass of GIMatchKind or a sub-dag whose "
               "operator is either of a GIMatchKindWithArgs or Instruction");
    PrintNote("Pattern was `" + Matchers->getArg(I)->getAsString() + "'");
    return false;
  }

  // Add the cartesian product of use -> def edges.
  bool FailedToAddEdges = false;
  for (const auto &NameAndDefs : NamedEdgeDefs) {
    if (NameAndDefs.getValue().size() > 1) {
      PrintError(TheDef.getLoc(),
                 "Two different MachineInstrs cannot def the same vreg");
      for (const auto &NameAndDefOp : NameAndDefs.getValue())
        PrintNote("in " + to_string(*NameAndDefOp.N) + " created from " +
                  to_string(*NameAndDefOp.Matcher) + "");
      FailedToAddEdges = true;
    }
    const auto &Uses = NamedEdgeUses[NameAndDefs.getKey()];
    for (const VarInfo &DefVar : NameAndDefs.getValue()) {
      for (const VarInfo &UseVar : Uses) {
        MatchDag.addEdge(insertStrTab(NameAndDefs.getKey()), UseVar.N, UseVar.Op,
                         DefVar.N, DefVar.Op);
      }
    }
  }
  if (FailedToAddEdges)
    return false;

  // If a variable is referenced in multiple use contexts then we need a
  // predicate to confirm they are the same operand. We can elide this if it's
  // also referenced in a def context and we're traversing the def-use chain
  // from the def to the uses but we can't know which direction we're going
  // until after reorientToRoots().
  for (const auto &NameAndUses : NamedEdgeUses) {
    const auto &Uses = NameAndUses.getValue();
    if (Uses.size() > 1) {
      const auto &LeadingVar = Uses.front();
      for (const auto &Var : ArrayRef<VarInfo>(Uses).drop_front()) {
        // Add a predicate for each pair until we've covered the whole
        // equivalence set. We could test the whole set in a single predicate
        // but that means we can't test any equivalence until all the MO's are
        // available which can lead to wasted work matching the DAG when this
        // predicate can already be seen to have failed.
        //
        // We have a similar problem due to the need to wait for a particular MO
        // before being able to test any of them. However, that is mitigated by
        // the order in which we build the DAG. We build from the roots outwards
        // so by using the first recorded use in all the predicates, we are
        // making the dependency on one of the earliest visited references in
        // the DAG. It's not guaranteed once the generated matcher is optimized
        // (because the factoring the common portions of rules might change the
        // visit order) but this should mean that these predicates depend on the
        // first MO to become available.
        const auto &P = MatchDag.addPredicateNode<GIMatchDagSameMOPredicate>(
            makeNameForAnonPredicate(*this));
        MatchDag.addPredicateDependency(LeadingVar.N, LeadingVar.Op, P,
                                        &P->getOperandInfo()["mi0"]);
        MatchDag.addPredicateDependency(Var.N, Var.Op, P,
                                        &P->getOperandInfo()["mi1"]);
      }
    }
  }
  return true;
}

class GICombinerEmitter {
  RecordKeeper &Records;
  StringRef Name;
  const CodeGenTarget &Target;
  Record *Combiner;
  std::vector<std::unique_ptr<CombineRule>> Rules;
  GIMatchDagContext MatchDagCtx;

  std::unique_ptr<CombineRule> makeCombineRule(const Record &R);

  void gatherRules(std::vector<std::unique_ptr<CombineRule>> &ActiveRules,
                   const std::vector<Record *> &&RulesAndGroups);

public:
  explicit GICombinerEmitter(RecordKeeper &RK, const CodeGenTarget &Target,
                             StringRef Name, Record *Combiner);
  ~GICombinerEmitter() {}

  StringRef getClassName() const {
    return Combiner->getValueAsString("Classname");
  }
  void run(raw_ostream &OS);

  /// Emit the name matcher (guarded by #ifndef NDEBUG) used to disable rules in
  /// response to the generated cl::opt.
  void emitNameMatcher(raw_ostream &OS) const;

  void generateCodeForTree(raw_ostream &OS, const GIMatchTree &Tree,
                           StringRef Indent) const;
};

GICombinerEmitter::GICombinerEmitter(RecordKeeper &RK,
                                     const CodeGenTarget &Target,
                                     StringRef Name, Record *Combiner)
    : Records(RK), Name(Name), Target(Target), Combiner(Combiner) {}

void GICombinerEmitter::emitNameMatcher(raw_ostream &OS) const {
  std::vector<std::pair<std::string, std::string>> Cases;
  Cases.reserve(Rules.size());

  for (const CombineRule &EnumeratedRule : make_pointee_range(Rules)) {
    std::string Code;
    raw_string_ostream SS(Code);
    SS << "return " << EnumeratedRule.getID() << ";\n";
    Cases.push_back(
        std::make_pair(std::string(EnumeratedRule.getName()), Code));
  }

  OS << "static Optional<uint64_t> getRuleIdxForIdentifier(StringRef "
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
     << "  return None;\n"
     << "}\n";
}

std::unique_ptr<CombineRule>
GICombinerEmitter::makeCombineRule(const Record &TheDef) {
  std::unique_ptr<CombineRule> Rule =
      std::make_unique<CombineRule>(Target, MatchDagCtx, NumPatternTotal, TheDef);

  if (!Rule->parseDefs())
    return nullptr;
  if (!Rule->parseMatcher(Target))
    return nullptr;

  Rule->reorientToRoots();

  LLVM_DEBUG({
    dbgs() << "Parsed rule defs/match for '" << Rule->getName() << "'\n";
    Rule->getMatchDag().dump();
    Rule->getMatchDag().writeDOTGraph(dbgs(), Rule->getName());
  });
  if (StopAfterParse)
    return Rule;

  // For now, don't support traversing from def to use. We'll come back to
  // this later once we have the algorithm changes to support it.
  bool EmittedDefToUseError = false;
  for (const auto &E : Rule->getMatchDag().edges()) {
    if (E->isDefToUse()) {
      if (!EmittedDefToUseError) {
        PrintError(
            TheDef.getLoc(),
            "Generated state machine cannot lookup uses from a def (yet)");
        EmittedDefToUseError = true;
      }
      PrintNote("Node " + to_string(*E->getFromMI()));
      PrintNote("Node " + to_string(*E->getToMI()));
      PrintNote("Edge " + to_string(*E));
    }
  }
  if (EmittedDefToUseError)
    return nullptr;

  // For now, don't support multi-root rules. We'll come back to this later
  // once we have the algorithm changes to support it.
  if (Rule->getNumRoots() > 1) {
    PrintError(TheDef.getLoc(), "Multi-root matches are not supported (yet)");
    return nullptr;
  }
  return Rule;
}

/// Recurse into GICombineGroup's and flatten the ruleset into a simple list.
void GICombinerEmitter::gatherRules(
    std::vector<std::unique_ptr<CombineRule>> &ActiveRules,
    const std::vector<Record *> &&RulesAndGroups) {
  for (Record *R : RulesAndGroups) {
    if (R->isValueUnset("Rules")) {
      std::unique_ptr<CombineRule> Rule = makeCombineRule(*R);
      if (Rule == nullptr) {
        PrintError(R->getLoc(), "Failed to parse rule");
        continue;
      }
      ActiveRules.emplace_back(std::move(Rule));
      ++NumPatternTotal;
    } else
      gatherRules(ActiveRules, R->getValueAsListOfDefs("Rules"));
  }
}

void GICombinerEmitter::generateCodeForTree(raw_ostream &OS,
                                            const GIMatchTree &Tree,
                                            StringRef Indent) const {
  if (Tree.getPartitioner() != nullptr) {
    Tree.getPartitioner()->generatePartitionSelectorCode(OS, Indent);
    for (const auto &EnumChildren : enumerate(Tree.children())) {
      OS << Indent << "if (Partition == " << EnumChildren.index() << " /* "
         << format_partition_name(Tree, EnumChildren.index()) << " */) {\n";
      generateCodeForTree(OS, EnumChildren.value(), (Indent + "  ").str());
      OS << Indent << "}\n";
    }
    return;
  }

  bool AnyFullyTested = false;
  for (const auto &Leaf : Tree.possible_leaves()) {
    OS << Indent << "// Leaf name: " << Leaf.getName() << "\n";

    const CombineRule *Rule = Leaf.getTargetData<CombineRule>();
    const Record &RuleDef = Rule->getDef();

    OS << Indent << "// Rule: " << RuleDef.getName() << "\n"
       << Indent << "if (!RuleConfig->isRuleDisabled(" << Rule->getID()
       << ")) {\n";

    CodeExpansions Expansions;
    for (const auto &VarBinding : Leaf.var_bindings()) {
      if (VarBinding.isInstr())
        Expansions.declare(VarBinding.getName(),
                           "MIs[" + to_string(VarBinding.getInstrID()) + "]");
      else
        Expansions.declare(VarBinding.getName(),
                           "MIs[" + to_string(VarBinding.getInstrID()) +
                               "]->getOperand(" +
                               to_string(VarBinding.getOpIdx()) + ")");
    }
    Rule->declareExpansions(Expansions);

    DagInit *Applyer = RuleDef.getValueAsDag("Apply");
    if (Applyer->getOperatorAsDef(RuleDef.getLoc())->getName() !=
        "apply") {
      PrintError(RuleDef.getLoc(), "Expected 'apply' operator in Apply DAG");
      return;
    }

    OS << Indent << "  if (1\n";

    // Attempt to emit code for any untested predicates left over. Note that
    // isFullyTested() will remain false even if we succeed here and therefore
    // combine rule elision will not be performed. This is because we do not
    // know if there's any connection between the predicates for each leaf and
    // therefore can't tell if one makes another unreachable. Ideally, the
    // partitioner(s) would be sufficiently complete to prevent us from having
    // untested predicates left over.
    for (const GIMatchDagPredicate *Predicate : Leaf.untested_predicates()) {
      if (Predicate->generateCheckCode(OS, (Indent + "      ").str(),
                                       Expansions))
        continue;
      PrintError(RuleDef.getLoc(),
                 "Unable to test predicate used in rule");
      PrintNote(SMLoc(),
                "This indicates an incomplete implementation in tablegen");
      Predicate->print(errs());
      errs() << "\n";
      OS << Indent
         << "llvm_unreachable(\"TableGen did not emit complete code for this "
            "path\");\n";
      break;
    }

    if (Rule->getMatchingFixupCode() &&
        !Rule->getMatchingFixupCode()->getValue().empty()) {
      // FIXME: Single-use lambda's like this are a serious compile-time
      // performance and memory issue. It's convenient for this early stage to
      // defer some work to successive patches but we need to eliminate this
      // before the ruleset grows to small-moderate size. Last time, it became
      // a big problem for low-mem systems around the 500 rule mark but by the
      // time we grow that large we should have merged the ISel match table
      // mechanism with the Combiner.
      OS << Indent << "      && [&]() {\n"
         << Indent << "      "
         << CodeExpander(Rule->getMatchingFixupCode()->getValue(), Expansions,
                         RuleDef.getLoc(), ShowExpansions)
         << "\n"
         << Indent << "      return true;\n"
         << Indent << "  }()";
    }
    OS << ") {\n" << Indent << "   ";

    if (const StringInit *Code = dyn_cast<StringInit>(Applyer->getArg(0))) {
      OS << CodeExpander(Code->getAsUnquotedString(), Expansions,
                         RuleDef.getLoc(), ShowExpansions)
         << "\n"
         << Indent << "    return true;\n"
         << Indent << "  }\n";
    } else {
      PrintError(RuleDef.getLoc(), "Expected apply code block");
      return;
    }

    OS << Indent << "}\n";

    assert(Leaf.isFullyTraversed());

    // If we didn't have any predicates left over and we're not using the
    // trap-door we have to support arbitrary C++ code while we're migrating to
    // the declarative style then we know that subsequent leaves are
    // unreachable.
    if (Leaf.isFullyTested() &&
        (!Rule->getMatchingFixupCode() ||
         Rule->getMatchingFixupCode()->getValue().empty())) {
      AnyFullyTested = true;
      OS << Indent
         << "llvm_unreachable(\"Combine rule elision was incorrect\");\n"
         << Indent << "return false;\n";
    }
  }
  if (!AnyFullyTested)
    OS << Indent << "return false;\n";
}

static void emitAdditionalHelperMethodArguments(raw_ostream &OS,
                                                Record *Combiner) {
  for (Record *Arg : Combiner->getValueAsListOfDefs("AdditionalArguments"))
    OS << ",\n    " << Arg->getValueAsString("Type")
       << Arg->getValueAsString("Name");
}

void GICombinerEmitter::run(raw_ostream &OS) {
  Records.startTimer("Gather rules");
  gatherRules(Rules, Combiner->getValueAsListOfDefs("Rules"));
  if (StopAfterParse) {
    MatchDagCtx.print(errs());
    PrintNote(Combiner->getLoc(),
              "Terminating due to -gicombiner-stop-after-parse");
    return;
  }
  if (ErrorsPrinted)
    PrintFatalError(Combiner->getLoc(), "Failed to parse one or more rules");
  LLVM_DEBUG(dbgs() << "Optimizing tree for " << Rules.size() << " rules\n");
  std::unique_ptr<GIMatchTree> Tree;
  Records.startTimer("Optimize combiner");
  {
    GIMatchTreeBuilder TreeBuilder(0);
    for (const auto &Rule : Rules) {
      bool HadARoot = false;
      for (const auto &Root : enumerate(Rule->getMatchDag().roots())) {
        TreeBuilder.addLeaf(Rule->getName(), Root.index(), Rule->getMatchDag(),
                            Rule.get());
        HadARoot = true;
      }
      if (!HadARoot)
        PrintFatalError(Rule->getDef().getLoc(), "All rules must have a root");
    }

    Tree = TreeBuilder.run();
  }
  if (StopAfterBuild) {
    Tree->writeDOTGraph(outs());
    PrintNote(Combiner->getLoc(),
              "Terminating due to -gicombiner-stop-after-build");
    return;
  }

  Records.startTimer("Emit combiner");
  OS << "#ifdef " << Name.upper() << "_GENCOMBINERHELPER_DEPS\n"
     << "#include \"llvm/ADT/SparseBitVector.h\"\n"
     << "namespace llvm {\n"
     << "extern cl::OptionCategory GICombinerOptionCategory;\n"
     << "} // end namespace llvm\n"
     << "#endif // ifdef " << Name.upper() << "_GENCOMBINERHELPER_DEPS\n\n";

  OS << "#ifdef " << Name.upper() << "_GENCOMBINERHELPER_H\n"
     << "class " << getClassName() << "RuleConfig {\n"
     << "  SparseBitVector<> DisabledRules;\n"
     << "\n"
     << "public:\n"
     << "  bool parseCommandLineOption();\n"
     << "  bool isRuleDisabled(unsigned ID) const;\n"
     << "  bool setRuleEnabled(StringRef RuleIdentifier);\n"
     << "  bool setRuleDisabled(StringRef RuleIdentifier);\n"
     << "};\n"
     << "\n"
     << "class " << getClassName();
  StringRef StateClass = Combiner->getValueAsString("StateClass");
  if (!StateClass.empty())
    OS << " : public " << StateClass;
  OS << " {\n"
     << "  const " << getClassName() << "RuleConfig *RuleConfig;\n"
     << "\n"
     << "public:\n"
     << "  template <typename... Args>" << getClassName() << "(const "
     << getClassName() << "RuleConfig &RuleConfig, Args &&... args) : ";
  if (!StateClass.empty())
    OS << StateClass << "(std::forward<Args>(args)...), ";
  OS << "RuleConfig(&RuleConfig) {}\n"
     << "\n"
     << "  bool tryCombineAll(\n"
     << "    GISelChangeObserver &Observer,\n"
     << "    MachineInstr &MI,\n"
     << "    MachineIRBuilder &B";
  emitAdditionalHelperMethodArguments(OS, Combiner);
  OS << ") const;\n";
  OS << "};\n\n";

  emitNameMatcher(OS);

  OS << "static Optional<std::pair<uint64_t, uint64_t>> "
        "getRuleRangeForIdentifier(StringRef RuleIdentifier) {\n"
     << "  std::pair<StringRef, StringRef> RangePair = "
        "RuleIdentifier.split('-');\n"
     << "  if (!RangePair.second.empty()) {\n"
     << "    const auto First = "
        "getRuleIdxForIdentifier(RangePair.first);\n"
     << "    const auto Last = "
        "getRuleIdxForIdentifier(RangePair.second);\n"
     << "    if (!First.hasValue() || !Last.hasValue())\n"
     << "      return None;\n"
     << "    if (First >= Last)\n"
     << "      report_fatal_error(\"Beginning of range should be before "
        "end of range\");\n"
     << "    return {{*First, *Last + 1}};\n"
     << "  } else if (RangePair.first == \"*\") {\n"
     << "    return {{0, " << Rules.size() << "}};\n"
     << "  } else {\n"
     << "    const auto I = getRuleIdxForIdentifier(RangePair.first);\n"
     << "    if (!I.hasValue())\n"
     << "      return None;\n"
     << "    return {{*I, *I + 1}};\n"
     << "  }\n"
     << "  return None;\n"
     << "}\n\n";

  for (bool Enabled : {true, false}) {
    OS << "bool " << getClassName() << "RuleConfig::setRule"
       << (Enabled ? "Enabled" : "Disabled") << "(StringRef RuleIdentifier) {\n"
       << "  auto MaybeRange = getRuleRangeForIdentifier(RuleIdentifier);\n"
       << "  if (!MaybeRange.hasValue())\n"
       << "    return false;\n"
       << "  for (auto I = MaybeRange->first; I < MaybeRange->second; ++I)\n"
       << "    DisabledRules." << (Enabled ? "reset" : "set") << "(I);\n"
       << "  return true;\n"
       << "}\n\n";
  }

  OS << "bool " << getClassName()
     << "RuleConfig::isRuleDisabled(unsigned RuleID) const {\n"
     << "  return DisabledRules.test(RuleID);\n"
     << "}\n";
  OS << "#endif // ifdef " << Name.upper() << "_GENCOMBINERHELPER_H\n\n";

  OS << "#ifdef " << Name.upper() << "_GENCOMBINERHELPER_CPP\n"
     << "\n"
     << "std::vector<std::string> " << Name << "Option;\n"
     << "cl::list<std::string> " << Name << "DisableOption(\n"
     << "    \"" << Name.lower() << "-disable-rule\",\n"
     << "    cl::desc(\"Disable one or more combiner rules temporarily in "
     << "the " << Name << " pass\"),\n"
     << "    cl::CommaSeparated,\n"
     << "    cl::Hidden,\n"
     << "    cl::cat(GICombinerOptionCategory),\n"
     << "    cl::callback([](const std::string &Str) {\n"
     << "      " << Name << "Option.push_back(Str);\n"
     << "    }));\n"
     << "cl::list<std::string> " << Name << "OnlyEnableOption(\n"
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
     << "\n"
     << "bool " << getClassName() << "RuleConfig::parseCommandLineOption() {\n"
     << "  for (StringRef Identifier : " << Name << "Option) {\n"
     << "    bool Enabled = Identifier.consume_front(\"!\");\n"
     << "    if (Enabled && !setRuleEnabled(Identifier))\n"
     << "      return false;\n"
     << "    if (!Enabled && !setRuleDisabled(Identifier))\n"
     << "      return false;\n"
     << "  }\n"
     << "  return true;\n"
     << "}\n\n";

  OS << "bool " << getClassName() << "::tryCombineAll(\n"
     << "    GISelChangeObserver &Observer,\n"
     << "    MachineInstr &MI,\n"
     << "    MachineIRBuilder &B";
  emitAdditionalHelperMethodArguments(OS, Combiner);
  OS << ") const {\n"
     << "  MachineBasicBlock *MBB = MI.getParent();\n"
     << "  MachineFunction *MF = MBB->getParent();\n"
     << "  MachineRegisterInfo &MRI = MF->getRegInfo();\n"
     << "  SmallVector<MachineInstr *, 8> MIs = {&MI};\n\n"
     << "  (void)MBB; (void)MF; (void)MRI; (void)RuleConfig;\n\n";

  OS << "  // Match data\n";
  for (const auto &Rule : Rules)
    for (const auto &I : Rule->matchdata_decls())
      OS << "  " << I.getType() << " " << I.getVariableName() << ";\n";
  OS << "\n";

  OS << "  int Partition = -1;\n";
  generateCodeForTree(OS, *Tree, "  ");
  OS << "\n  return false;\n"
     << "}\n"
     << "#endif // ifdef " << Name.upper() << "_GENCOMBINERHELPER_CPP\n";
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//

namespace llvm {
void EmitGICombiner(RecordKeeper &RK, raw_ostream &OS) {
  CodeGenTarget Target(RK);
  emitSourceFileHeader("Global Combiner", OS);

  if (SelectedCombiners.empty())
    PrintFatalError("No combiners selected with -combiners");
  for (const auto &Combiner : SelectedCombiners) {
    Record *CombinerDef = RK.getDef(Combiner);
    if (!CombinerDef)
      PrintFatalError("Could not find " + Combiner);
    GICombinerEmitter(RK, Target, Combiner, CombinerDef).run(OS);
  }
  NumPatternTotalStatistic = NumPatternTotal;
}

} // namespace llvm
