//===- ForwardOpTree.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Move instructions between statements.
//
//===----------------------------------------------------------------------===//

#include "polly/ForwardOpTree.h"
#include "polly/Options.h"
#include "polly/ScopBuilder.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLOStream.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/VirtualInstruction.h"
#include "polly/ZoneAlgo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/ctx.h"
#include "isl/isl-noexceptions.h"
#include <cassert>
#include <memory>

#define DEBUG_TYPE "polly-optree"

using namespace llvm;
using namespace polly;

static cl::opt<bool>
    AnalyzeKnown("polly-optree-analyze-known",
                 cl::desc("Analyze array contents for load forwarding"),
                 cl::cat(PollyCategory), cl::init(true), cl::Hidden);

static cl::opt<bool>
    NormalizePHIs("polly-optree-normalize-phi",
                  cl::desc("Replace PHIs by their incoming values"),
                  cl::cat(PollyCategory), cl::init(false), cl::Hidden);

static cl::opt<unsigned>
    MaxOps("polly-optree-max-ops",
           cl::desc("Maximum number of ISL operations to invest for known "
                    "analysis; 0=no limit"),
           cl::init(1000000), cl::cat(PollyCategory), cl::Hidden);

STATISTIC(KnownAnalyzed, "Number of successfully analyzed SCoPs");
STATISTIC(KnownOutOfQuota,
          "Analyses aborted because max_operations was reached");

STATISTIC(TotalInstructionsCopied, "Number of copied instructions");
STATISTIC(TotalKnownLoadsForwarded,
          "Number of forwarded loads because their value was known");
STATISTIC(TotalReloads, "Number of reloaded values");
STATISTIC(TotalReadOnlyCopied, "Number of copied read-only accesses");
STATISTIC(TotalForwardedTrees, "Number of forwarded operand trees");
STATISTIC(TotalModifiedStmts,
          "Number of statements with at least one forwarded tree");

STATISTIC(ScopsModified, "Number of SCoPs with at least one forwarded tree");

STATISTIC(NumValueWrites, "Number of scalar value writes after OpTree");
STATISTIC(NumValueWritesInLoops,
          "Number of scalar value writes nested in affine loops after OpTree");
STATISTIC(NumPHIWrites, "Number of scalar phi writes after OpTree");
STATISTIC(NumPHIWritesInLoops,
          "Number of scalar phi writes nested in affine loops after OpTree");
STATISTIC(NumSingletonWrites, "Number of singleton writes after OpTree");
STATISTIC(NumSingletonWritesInLoops,
          "Number of singleton writes nested in affine loops after OpTree");

namespace {

/// The state of whether an operand tree was/can be forwarded.
///
/// The items apply to an instructions and its operand tree with the instruction
/// as the root element. If the value in question is not an instruction in the
/// SCoP, it can be a leaf of an instruction's operand tree.
enum ForwardingDecision {
  /// An uninitialized value.
  FD_Unknown,

  /// The root instruction or value cannot be forwarded at all.
  FD_CannotForward,

  /// The root instruction or value can be forwarded as a leaf of a larger
  /// operand tree.
  /// It does not make sense to move the value itself, it would just replace it
  /// by a use of itself. For instance, a constant "5" used in a statement can
  /// be forwarded, but it would just replace it by the same constant "5".
  /// However, it makes sense to move as an operand of
  ///
  ///   %add = add 5, 5
  ///
  /// where "5" is moved as part of a larger operand tree. "5" would be placed
  /// (disregarding for a moment that literal constants don't have a location
  /// and can be used anywhere) into the same statement as %add would.
  FD_CanForwardLeaf,

  /// The root instruction can be forwarded and doing so avoids a scalar
  /// dependency.
  ///
  /// This can be either because the operand tree can be moved to the target
  /// statement, or a memory access is redirected to read from a different
  /// location.
  FD_CanForwardProfitably,

  /// A forwarding method cannot be applied to the operand tree.
  /// The difference to FD_CannotForward is that there might be other methods
  /// that can handle it.
  FD_NotApplicable
};

/// Represents the evaluation of and action to taken when forwarding a value
/// from an operand tree.
struct ForwardingAction {
  using KeyTy = std::pair<Value *, ScopStmt *>;

  /// Evaluation of forwarding a value.
  ForwardingDecision Decision = FD_Unknown;

  /// Callback to execute the forwarding.
  /// Returning true allows deleting the polly::MemoryAccess if the value is the
  /// root of the operand tree (and its elimination the reason why the
  /// forwarding is done). Return false if the MemoryAccess is reused or there
  /// might be other users of the read accesses. In the letter case the
  /// polly::SimplifyPass can remove dead MemoryAccesses.
  std::function<bool()> Execute = []() -> bool {
    llvm_unreachable("unspecified how to forward");
  };

  /// Other values that need to be forwarded if this action is executed. Their
  /// actions are executed after this one.
  SmallVector<KeyTy, 4> Depends;

  /// Named ctor: The method creating this object does not apply to the kind of
  /// value, but other methods may.
  static ForwardingAction notApplicable() {
    ForwardingAction Result;
    Result.Decision = FD_NotApplicable;
    return Result;
  }

  /// Named ctor: The value cannot be forwarded.
  static ForwardingAction cannotForward() {
    ForwardingAction Result;
    Result.Decision = FD_CannotForward;
    return Result;
  }

  /// Named ctor: The value can just be used without any preparation.
  static ForwardingAction triviallyForwardable(bool IsProfitable, Value *Val) {
    ForwardingAction Result;
    Result.Decision =
        IsProfitable ? FD_CanForwardProfitably : FD_CanForwardLeaf;
    Result.Execute = [=]() {
      LLVM_DEBUG(dbgs() << "    trivially forwarded: " << *Val << "\n");
      return true;
    };
    return Result;
  }

  /// Name ctor: The value can be forwarded by executing an action.
  static ForwardingAction canForward(std::function<bool()> Execute,
                                     ArrayRef<KeyTy> Depends,
                                     bool IsProfitable) {
    ForwardingAction Result;
    Result.Decision =
        IsProfitable ? FD_CanForwardProfitably : FD_CanForwardLeaf;
    Result.Execute = std::move(Execute);
    Result.Depends.append(Depends.begin(), Depends.end());
    return Result;
  }
};

/// Implementation of operand tree forwarding for a specific SCoP.
///
/// For a statement that requires a scalar value (through a value read
/// MemoryAccess), see if its operand can be moved into the statement. If so,
/// the MemoryAccess is removed and the all the operand tree instructions are
/// moved into the statement. All original instructions are left in the source
/// statements. The simplification pass can clean these up.
class ForwardOpTreeImpl final : ZoneAlgorithm {
private:
  using MemoizationTy = DenseMap<ForwardingAction::KeyTy, ForwardingAction>;

  /// Scope guard to limit the number of isl operations for this pass.
  IslMaxOperationsGuard &MaxOpGuard;

  /// How many instructions have been copied to other statements.
  int NumInstructionsCopied = 0;

  /// Number of loads forwarded because their value was known.
  int NumKnownLoadsForwarded = 0;

  /// Number of values reloaded from known array elements.
  int NumReloads = 0;

  /// How many read-only accesses have been copied.
  int NumReadOnlyCopied = 0;

  /// How many operand trees have been forwarded.
  int NumForwardedTrees = 0;

  /// Number of statements with at least one forwarded operand tree.
  int NumModifiedStmts = 0;

  /// Whether we carried out at least one change to the SCoP.
  bool Modified = false;

  /// Cache of how to forward values.
  /// The key of this map is the llvm::Value to be forwarded and the
  /// polly::ScopStmt it is forwarded from. This is because the same llvm::Value
  /// can evaluate differently depending on where it is evaluate. For instance,
  /// a synthesizable Scev represents a recurrence with an loop but the loop's
  /// exit value if evaluated after the loop.
  /// The cached results are only valid for the current TargetStmt.
  /// CHECKME: ScalarEvolution::getScevAtScope should take care for getting the
  /// exit value when instantiated outside of the loop. The primary concern is
  /// ambiguity when crossing PHI nodes, which currently is not supported.
  MemoizationTy ForwardingActions;

  /// Contains the zones where array elements are known to contain a specific
  /// value.
  /// { [Element[] -> Zone[]] -> ValInst[] }
  /// @see computeKnown()
  isl::union_map Known;

  /// Translator for newly introduced ValInsts to already existing ValInsts such
  /// that new introduced load instructions can reuse the Known analysis of its
  /// original load. { ValInst[] -> ValInst[] }
  isl::union_map Translator;

  /// Get list of array elements that do contain the same ValInst[] at Domain[].
  ///
  /// @param ValInst { Domain[] -> ValInst[] }
  ///                The values for which we search for alternative locations,
  ///                per statement instance.
  ///
  /// @return { Domain[] -> Element[] }
  ///         For each statement instance, the array elements that contain the
  ///         same ValInst.
  isl::union_map findSameContentElements(isl::union_map ValInst) {
    assert(!ValInst.is_single_valued().is_false());

    // { Domain[] }
    isl::union_set Domain = ValInst.domain();

    // { Domain[] -> Scatter[] }
    isl::union_map Schedule = getScatterFor(Domain);

    // { Element[] -> [Scatter[] -> ValInst[]] }
    isl::union_map MustKnownCurried =
        convertZoneToTimepoints(Known, isl::dim::in, false, true).curry();

    // { [Domain[] -> ValInst[]] -> Scatter[] }
    isl::union_map DomValSched = ValInst.domain_map().apply_range(Schedule);

    // { [Scatter[] -> ValInst[]] -> [Domain[] -> ValInst[]] }
    isl::union_map SchedValDomVal =
        DomValSched.range_product(ValInst.range_map()).reverse();

    // { Element[] -> [Domain[] -> ValInst[]] }
    isl::union_map MustKnownInst = MustKnownCurried.apply_range(SchedValDomVal);

    // { Domain[] -> Element[] }
    isl::union_map MustKnownMap =
        MustKnownInst.uncurry().domain().unwrap().reverse();
    simplify(MustKnownMap);

    return MustKnownMap;
  }

  /// Find a single array element for each statement instance, within a single
  /// array.
  ///
  /// @param MustKnown { Domain[] -> Element[] }
  ///                  Set of candidate array elements.
  /// @param Domain    { Domain[] }
  ///                  The statement instance for which we need elements for.
  ///
  /// @return { Domain[] -> Element[] }
  ///         For each statement instance, an array element out of @p MustKnown.
  ///         All array elements must be in the same array (Polly does not yet
  ///         support reading from different accesses using the same
  ///         MemoryAccess). If no mapping for all of @p Domain exists, returns
  ///         null.
  isl::map singleLocation(isl::union_map MustKnown, isl::set Domain) {
    // { Domain[] -> Element[] }
    isl::map Result;

    // Make irrelevant elements not interfere.
    Domain = Domain.intersect_params(S->getContext());

    // MemoryAccesses can read only elements from a single array
    // (i.e. not: { Dom[0] -> A[0]; Dom[1] -> B[1] }).
    // Look through all spaces until we find one that contains at least the
    // wanted statement instance.s
    for (isl::map Map : MustKnown.get_map_list()) {
      // Get the array this is accessing.
      isl::id ArrayId = Map.get_tuple_id(isl::dim::out);
      ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(ArrayId.get_user());

      // No support for generation of indirect array accesses.
      if (SAI->getBasePtrOriginSAI())
        continue;

      // Determine whether this map contains all wanted values.
      isl::set MapDom = Map.domain();
      if (!Domain.is_subset(MapDom).is_true())
        continue;

      // There might be multiple array elements that contain the same value, but
      // choose only one of them. lexmin is used because it returns a one-value
      // mapping, we do not care about which one.
      // TODO: Get the simplest access function.
      Result = Map.lexmin();
      break;
    }

    return Result;
  }

public:
  ForwardOpTreeImpl(Scop *S, LoopInfo *LI, IslMaxOperationsGuard &MaxOpGuard)
      : ZoneAlgorithm("polly-optree", S, LI), MaxOpGuard(MaxOpGuard) {}

  /// Compute the zones of known array element contents.
  ///
  /// @return True if the computed #Known is usable.
  bool computeKnownValues() {
    isl::union_map MustKnown, KnownFromLoad, KnownFromInit;

    // Check that nothing strange occurs.
    collectCompatibleElts();

    {
      IslQuotaScope QuotaScope = MaxOpGuard.enter();

      computeCommon();
      if (NormalizePHIs)
        computeNormalizedPHIs();
      Known = computeKnown(true, true);

      // Preexisting ValInsts use the known content analysis of themselves.
      Translator = makeIdentityMap(Known.range(), false);
    }

    if (Known.is_null() || Translator.is_null() || NormalizeMap.is_null()) {
      assert(isl_ctx_last_error(IslCtx.get()) == isl_error_quota);
      Known = {};
      Translator = {};
      NormalizeMap = {};
      LLVM_DEBUG(dbgs() << "Known analysis exceeded max_operations\n");
      return false;
    }

    KnownAnalyzed++;
    LLVM_DEBUG(dbgs() << "All known: " << Known << "\n");

    return true;
  }

  void printStatistics(raw_ostream &OS, int Indent = 0) {
    OS.indent(Indent) << "Statistics {\n";
    OS.indent(Indent + 4) << "Instructions copied: " << NumInstructionsCopied
                          << '\n';
    OS.indent(Indent + 4) << "Known loads forwarded: " << NumKnownLoadsForwarded
                          << '\n';
    OS.indent(Indent + 4) << "Reloads: " << NumReloads << '\n';
    OS.indent(Indent + 4) << "Read-only accesses copied: " << NumReadOnlyCopied
                          << '\n';
    OS.indent(Indent + 4) << "Operand trees forwarded: " << NumForwardedTrees
                          << '\n';
    OS.indent(Indent + 4) << "Statements with forwarded operand trees: "
                          << NumModifiedStmts << '\n';
    OS.indent(Indent) << "}\n";
  }

  void printStatements(raw_ostream &OS, int Indent = 0) const {
    OS.indent(Indent) << "After statements {\n";
    for (auto &Stmt : *S) {
      OS.indent(Indent + 4) << Stmt.getBaseName() << "\n";
      for (auto *MA : Stmt)
        MA->print(OS);

      OS.indent(Indent + 12);
      Stmt.printInstructions(OS);
    }
    OS.indent(Indent) << "}\n";
  }

  /// Create a new MemoryAccess of type read and MemoryKind::Array.
  ///
  /// @param Stmt           The statement in which the access occurs.
  /// @param LI             The instruction that does the access.
  /// @param AccessRelation The array element that each statement instance
  ///                       accesses.
  ///
  /// @param The newly created access.
  MemoryAccess *makeReadArrayAccess(ScopStmt *Stmt, LoadInst *LI,
                                    isl::map AccessRelation) {
    isl::id ArrayId = AccessRelation.get_tuple_id(isl::dim::out);
    ScopArrayInfo *SAI = reinterpret_cast<ScopArrayInfo *>(ArrayId.get_user());

    // Create a dummy SCEV access, to be replaced anyway.
    SmallVector<const SCEV *, 4> Sizes;
    Sizes.reserve(SAI->getNumberOfDimensions());
    SmallVector<const SCEV *, 4> Subscripts;
    Subscripts.reserve(SAI->getNumberOfDimensions());
    for (unsigned i = 0; i < SAI->getNumberOfDimensions(); i += 1) {
      Sizes.push_back(SAI->getDimensionSize(i));
      Subscripts.push_back(nullptr);
    }

    MemoryAccess *Access =
        new MemoryAccess(Stmt, LI, MemoryAccess::READ, SAI->getBasePtr(),
                         LI->getType(), true, {}, Sizes, LI, MemoryKind::Array);
    S->addAccessFunction(Access);
    Stmt->addAccess(Access, true);

    Access->setNewAccessRelation(AccessRelation);

    return Access;
  }

  /// Forward a load by reading from an array element that contains the same
  /// value. Typically the location it was loaded from.
  ///
  /// @param TargetStmt  The statement the operand tree will be copied to.
  /// @param Inst        The (possibly speculatable) instruction to forward.
  /// @param UseStmt     The statement that uses @p Inst.
  /// @param UseLoop     The loop @p Inst is used in.
  /// @param DefStmt     The statement @p Inst is defined in.
  /// @param DefLoop     The loop which contains @p Inst.
  ///
  /// @return A ForwardingAction object describing the feasibility and
  ///         profitability evaluation and the callback carrying-out the value
  ///         forwarding.
  ForwardingAction forwardKnownLoad(ScopStmt *TargetStmt, Instruction *Inst,
                                    ScopStmt *UseStmt, Loop *UseLoop,
                                    ScopStmt *DefStmt, Loop *DefLoop) {
    // Cannot do anything without successful known analysis.
    if (Known.is_null() || Translator.is_null() ||
        MaxOpGuard.hasQuotaExceeded())
      return ForwardingAction::notApplicable();

    LoadInst *LI = dyn_cast<LoadInst>(Inst);
    if (!LI)
      return ForwardingAction::notApplicable();

    ForwardingDecision OpDecision =
        forwardTree(TargetStmt, LI->getPointerOperand(), DefStmt, DefLoop);
    switch (OpDecision) {
    case FD_CanForwardProfitably:
    case FD_CanForwardLeaf:
      break;
    case FD_CannotForward:
      return ForwardingAction::cannotForward();
    default:
      llvm_unreachable("Shouldn't return this");
    }

    MemoryAccess *Access = TargetStmt->getArrayAccessOrNULLFor(LI);
    if (Access) {
      // If the load is already in the statement, no forwarding is necessary.
      // However, it might happen that the LoadInst is already present in the
      // statement's instruction list. In that case we do as follows:
      // - For the evaluation, we can trivially forward it as it is
      //   benefit of forwarding an already present instruction.
      // - For the execution, prepend the instruction (to make it
      //   available to all instructions following in the instruction list), but
      //   do not add another MemoryAccess.
      auto ExecAction = [this, TargetStmt, LI, Access]() -> bool {
        TargetStmt->prependInstruction(LI);
        LLVM_DEBUG(
            dbgs() << "    forwarded known load with preexisting MemoryAccess"
                   << Access << "\n");
        (void)Access;

        NumKnownLoadsForwarded++;
        TotalKnownLoadsForwarded++;
        return true;
      };
      return ForwardingAction::canForward(
          ExecAction, {{LI->getPointerOperand(), DefStmt}}, true);
    }

    // Allow the following Isl calculations (until we return the
    // ForwardingAction, excluding the code inside the lambda that will be
    // executed later) to fail.
    IslQuotaScope QuotaScope = MaxOpGuard.enter();

    // { DomainDef[] -> ValInst[] }
    isl::map ExpectedVal = makeValInst(Inst, UseStmt, UseLoop);
    assert(!isNormalized(ExpectedVal).is_false() &&
           "LoadInsts are always normalized");

    // { DomainUse[] -> DomainTarget[] }
    isl::map UseToTarget = getDefToTarget(UseStmt, TargetStmt);

    // { DomainTarget[] -> ValInst[] }
    isl::map TargetExpectedVal = ExpectedVal.apply_domain(UseToTarget);
    isl::union_map TranslatedExpectedVal =
        isl::union_map(TargetExpectedVal).apply_range(Translator);

    // { DomainTarget[] -> Element[] }
    isl::union_map Candidates = findSameContentElements(TranslatedExpectedVal);

    isl::map SameVal = singleLocation(Candidates, getDomainFor(TargetStmt));
    if (SameVal.is_null())
      return ForwardingAction::notApplicable();

    LLVM_DEBUG(dbgs() << "      expected values where " << TargetExpectedVal
                      << "\n");
    LLVM_DEBUG(dbgs() << "      candidate elements where " << Candidates
                      << "\n");

    // { ValInst[] }
    isl::space ValInstSpace = ExpectedVal.get_space().range();

    // After adding a new load to the SCoP, also update the Known content
    // about it. The new load will have a known ValInst of
    // { [DomainTarget[] -> Value[]] }
    // but which -- because it is a copy of it -- has same value as the
    // { [DomainDef[] -> Value[]] }
    // that it replicates. Instead of  cloning the known content of
    // [DomainDef[] -> Value[]]
    // for DomainTarget[], we add a 'translator' that maps
    // [DomainTarget[] -> Value[]] to [DomainDef[] -> Value[]]
    // before comparing to the known content.
    // TODO: 'Translator' could also be used to map PHINodes to their incoming
    // ValInsts.
    isl::map LocalTranslator;
    if (!ValInstSpace.is_wrapping().is_false()) {
      // { DefDomain[] -> Value[] }
      isl::map ValInsts = ExpectedVal.range().unwrap();

      // { DefDomain[] }
      isl::set DefDomain = ValInsts.domain();

      // { Value[] }
      isl::space ValSpace = ValInstSpace.unwrap().range();

      // { Value[] -> Value[] }
      isl::map ValToVal =
          isl::map::identity(ValSpace.map_from_domain_and_range(ValSpace));

      // { DomainDef[] -> DomainTarget[] }
      isl::map DefToTarget = getDefToTarget(DefStmt, TargetStmt);

      // { [TargetDomain[] -> Value[]] -> [DefDomain[] -> Value] }
      LocalTranslator = DefToTarget.reverse().product(ValToVal);
      LLVM_DEBUG(dbgs() << "      local translator is " << LocalTranslator
                        << "\n");

      if (LocalTranslator.is_null())
        return ForwardingAction::notApplicable();
    }

    auto ExecAction = [this, TargetStmt, LI, SameVal,
                       LocalTranslator]() -> bool {
      TargetStmt->prependInstruction(LI);
      MemoryAccess *Access = makeReadArrayAccess(TargetStmt, LI, SameVal);
      LLVM_DEBUG(dbgs() << "    forwarded known load with new MemoryAccess"
                        << Access << "\n");
      (void)Access;

      if (!LocalTranslator.is_null())
        Translator = Translator.unite(LocalTranslator);

      NumKnownLoadsForwarded++;
      TotalKnownLoadsForwarded++;
      return true;
    };
    return ForwardingAction::canForward(
        ExecAction, {{LI->getPointerOperand(), DefStmt}}, true);
  }

  /// Forward a scalar by redirecting the access to an array element that stores
  /// the same value.
  ///
  /// @param TargetStmt  The statement the operand tree will be copied to.
  /// @param Inst        The scalar to forward.
  /// @param UseStmt     The statement that uses @p Inst.
  /// @param UseLoop     The loop @p Inst is used in.
  /// @param DefStmt     The statement @p Inst is defined in.
  /// @param DefLoop     The loop which contains @p Inst.
  ///
  /// @return A ForwardingAction object describing the feasibility and
  ///         profitability evaluation and the callback carrying-out the value
  ///         forwarding.
  ForwardingAction reloadKnownContent(ScopStmt *TargetStmt, Instruction *Inst,
                                      ScopStmt *UseStmt, Loop *UseLoop,
                                      ScopStmt *DefStmt, Loop *DefLoop) {
    // Cannot do anything without successful known analysis.
    if (Known.is_null() || Translator.is_null() ||
        MaxOpGuard.hasQuotaExceeded())
      return ForwardingAction::notApplicable();

    // Don't spend too much time analyzing whether it can be reloaded.
    IslQuotaScope QuotaScope = MaxOpGuard.enter();

    // { DomainDef[] -> ValInst[] }
    isl::union_map ExpectedVal = makeNormalizedValInst(Inst, UseStmt, UseLoop);

    // { DomainUse[] -> DomainTarget[] }
    isl::map UseToTarget = getDefToTarget(UseStmt, TargetStmt);

    // { DomainTarget[] -> ValInst[] }
    isl::union_map TargetExpectedVal = ExpectedVal.apply_domain(UseToTarget);
    isl::union_map TranslatedExpectedVal =
        TargetExpectedVal.apply_range(Translator);

    // { DomainTarget[] -> Element[] }
    isl::union_map Candidates = findSameContentElements(TranslatedExpectedVal);

    isl::map SameVal = singleLocation(Candidates, getDomainFor(TargetStmt));
    simplify(SameVal);
    if (SameVal.is_null())
      return ForwardingAction::notApplicable();

    auto ExecAction = [this, TargetStmt, Inst, SameVal]() {
      MemoryAccess *Access = TargetStmt->lookupInputAccessOf(Inst);
      if (!Access)
        Access = TargetStmt->ensureValueRead(Inst);
      Access->setNewAccessRelation(SameVal);

      LLVM_DEBUG(dbgs() << "    forwarded known content of " << *Inst
                        << " which is " << SameVal << "\n");
      TotalReloads++;
      NumReloads++;
      return false;
    };

    return ForwardingAction::canForward(ExecAction, {}, true);
  }

  /// Forwards a speculatively executable instruction.
  ///
  /// @param TargetStmt  The statement the operand tree will be copied to.
  /// @param UseInst     The (possibly speculatable) instruction to forward.
  /// @param DefStmt     The statement @p UseInst is defined in.
  /// @param DefLoop     The loop which contains @p UseInst.
  ///
  /// @return A ForwardingAction object describing the feasibility and
  ///         profitability evaluation and the callback carrying-out the value
  ///         forwarding.
  ForwardingAction forwardSpeculatable(ScopStmt *TargetStmt,
                                       Instruction *UseInst, ScopStmt *DefStmt,
                                       Loop *DefLoop) {
    // PHIs, unless synthesizable, are not yet supported.
    if (isa<PHINode>(UseInst))
      return ForwardingAction::notApplicable();

    // Compatible instructions must satisfy the following conditions:
    // 1. Idempotent (instruction will be copied, not moved; although its
    //    original instance might be removed by simplification)
    // 2. Not access memory (There might be memory writes between)
    // 3. Not cause undefined behaviour (we might copy to a location when the
    //    original instruction was no executed; this is currently not possible
    //    because we do not forward PHINodes)
    // 4. Not leak memory if executed multiple times (i.e. malloc)
    //
    // Instruction::mayHaveSideEffects is not sufficient because it considers
    // malloc to not have side-effects. llvm::isSafeToSpeculativelyExecute is
    // not sufficient because it allows memory accesses.
    if (mayHaveNonDefUseDependency(*UseInst))
      return ForwardingAction::notApplicable();

    SmallVector<ForwardingAction::KeyTy, 4> Depends;
    Depends.reserve(UseInst->getNumOperands());
    for (Value *OpVal : UseInst->operand_values()) {
      ForwardingDecision OpDecision =
          forwardTree(TargetStmt, OpVal, DefStmt, DefLoop);
      switch (OpDecision) {
      case FD_CannotForward:
        return ForwardingAction::cannotForward();

      case FD_CanForwardLeaf:
      case FD_CanForwardProfitably:
        Depends.emplace_back(OpVal, DefStmt);
        break;

      case FD_NotApplicable:
      case FD_Unknown:
        llvm_unreachable(
            "forwardTree should never return FD_NotApplicable/FD_Unknown");
      }
    }

    auto ExecAction = [this, TargetStmt, UseInst]() {
      // To ensure the right order, prepend this instruction before its
      // operands. This ensures that its operands are inserted before the
      // instruction using them.
      TargetStmt->prependInstruction(UseInst);

      LLVM_DEBUG(dbgs() << "    forwarded speculable instruction: " << *UseInst
                        << "\n");
      NumInstructionsCopied++;
      TotalInstructionsCopied++;
      return true;
    };
    return ForwardingAction::canForward(ExecAction, Depends, true);
  }

  /// Determines whether an operand tree can be forwarded and returns
  /// instructions how to do so in the form of a ForwardingAction object.
  ///
  /// @param TargetStmt  The statement the operand tree will be copied to.
  /// @param UseVal      The value (usually an instruction) which is root of an
  ///                    operand tree.
  /// @param UseStmt     The statement that uses @p UseVal.
  /// @param UseLoop     The loop @p UseVal is used in.
  ///
  /// @return A ForwardingAction object describing the feasibility and
  ///         profitability evaluation and the callback carrying-out the value
  ///         forwarding.
  ForwardingAction forwardTreeImpl(ScopStmt *TargetStmt, Value *UseVal,
                                   ScopStmt *UseStmt, Loop *UseLoop) {
    ScopStmt *DefStmt = nullptr;
    Loop *DefLoop = nullptr;

    // { DefDomain[] -> TargetDomain[] }
    isl::map DefToTarget;

    VirtualUse VUse = VirtualUse::create(UseStmt, UseLoop, UseVal, true);
    switch (VUse.getKind()) {
    case VirtualUse::Constant:
    case VirtualUse::Block:
    case VirtualUse::Hoisted:
      // These can be used anywhere without special considerations.
      return ForwardingAction::triviallyForwardable(false, UseVal);

    case VirtualUse::Synthesizable: {
      // Check if the value is synthesizable at the new location as well. This
      // might be possible when leaving a loop for which ScalarEvolution is
      // unable to derive the exit value for.
      // TODO: If there is a LCSSA PHI at the loop exit, use that one.
      // If the SCEV contains a SCEVAddRecExpr, we currently depend on that we
      // do not forward past its loop header. This would require us to use a
      // previous loop induction variable instead the current one. We currently
      // do not allow forwarding PHI nodes, thus this should never occur (the
      // only exception where no phi is necessary being an unreachable loop
      // without edge from the outside).
      VirtualUse TargetUse = VirtualUse::create(
          S, TargetStmt, TargetStmt->getSurroundingLoop(), UseVal, true);
      if (TargetUse.getKind() == VirtualUse::Synthesizable)
        return ForwardingAction::triviallyForwardable(false, UseVal);

      LLVM_DEBUG(
          dbgs() << "    Synthesizable would not be synthesizable anymore: "
                 << *UseVal << "\n");
      return ForwardingAction::cannotForward();
    }

    case VirtualUse::ReadOnly: {
      if (!ModelReadOnlyScalars)
        return ForwardingAction::triviallyForwardable(false, UseVal);

      // If we model read-only scalars, we need to create a MemoryAccess for it.
      auto ExecAction = [this, TargetStmt, UseVal]() {
        TargetStmt->ensureValueRead(UseVal);

        LLVM_DEBUG(dbgs() << "    forwarded read-only value " << *UseVal
                          << "\n");
        NumReadOnlyCopied++;
        TotalReadOnlyCopied++;

        // Note that we cannot return true here. With a operand tree
        // depth of 0, UseVal is the use in TargetStmt that we try to replace.
        // With -polly-analyze-read-only-scalars=true we would ensure the
        // existence of a MemoryAccess (which already exists for a leaf) and be
        // removed again by tryForwardTree because it's goal is to remove this
        // scalar MemoryAccess. It interprets FD_CanForwardTree as the
        // permission to do so.
        return false;
      };
      return ForwardingAction::canForward(ExecAction, {}, false);
    }

    case VirtualUse::Intra:
      // Knowing that UseStmt and DefStmt are the same statement instance, just
      // reuse the information about UseStmt for DefStmt
      DefStmt = UseStmt;

      LLVM_FALLTHROUGH;
    case VirtualUse::Inter:
      Instruction *Inst = cast<Instruction>(UseVal);

      if (!DefStmt) {
        DefStmt = S->getStmtFor(Inst);
        if (!DefStmt)
          return ForwardingAction::cannotForward();
      }

      DefLoop = LI->getLoopFor(Inst->getParent());

      ForwardingAction SpeculativeResult =
          forwardSpeculatable(TargetStmt, Inst, DefStmt, DefLoop);
      if (SpeculativeResult.Decision != FD_NotApplicable)
        return SpeculativeResult;

      ForwardingAction KnownResult = forwardKnownLoad(
          TargetStmt, Inst, UseStmt, UseLoop, DefStmt, DefLoop);
      if (KnownResult.Decision != FD_NotApplicable)
        return KnownResult;

      ForwardingAction ReloadResult = reloadKnownContent(
          TargetStmt, Inst, UseStmt, UseLoop, DefStmt, DefLoop);
      if (ReloadResult.Decision != FD_NotApplicable)
        return ReloadResult;

      // When no method is found to forward the operand tree, we effectively
      // cannot handle it.
      LLVM_DEBUG(dbgs() << "    Cannot forward instruction: " << *Inst << "\n");
      return ForwardingAction::cannotForward();
    }

    llvm_unreachable("Case unhandled");
  }

  /// Determines whether an operand tree can be forwarded. Previous evaluations
  /// are cached.
  ///
  /// @param TargetStmt  The statement the operand tree will be copied to.
  /// @param UseVal      The value (usually an instruction) which is root of an
  ///                    operand tree.
  /// @param UseStmt     The statement that uses @p UseVal.
  /// @param UseLoop     The loop @p UseVal is used in.
  ///
  /// @return FD_CannotForward        if @p UseVal cannot be forwarded.
  ///         FD_CanForwardLeaf       if @p UseVal is forwardable, but not
  ///                                 profitable.
  ///         FD_CanForwardProfitably if @p UseVal is forwardable and useful to
  ///                                 do.
  ForwardingDecision forwardTree(ScopStmt *TargetStmt, Value *UseVal,
                                 ScopStmt *UseStmt, Loop *UseLoop) {
    // Lookup any cached evaluation.
    auto It = ForwardingActions.find({UseVal, UseStmt});
    if (It != ForwardingActions.end())
      return It->second.Decision;

    // Make a new evaluation.
    ForwardingAction Action =
        forwardTreeImpl(TargetStmt, UseVal, UseStmt, UseLoop);
    ForwardingDecision Result = Action.Decision;

    // Remember for the next time.
    assert(!ForwardingActions.count({UseVal, UseStmt}) &&
           "circular dependency?");
    ForwardingActions.insert({{UseVal, UseStmt}, std::move(Action)});

    return Result;
  }

  /// Forward an operand tree using cached actions.
  ///
  /// @param Stmt   Statement the operand tree is moved into.
  /// @param UseVal Root of the operand tree within @p Stmt.
  /// @param RA     The MemoryAccess for @p UseVal that the forwarding intends
  ///               to remove.
  void applyForwardingActions(ScopStmt *Stmt, Value *UseVal, MemoryAccess *RA) {
    using ChildItTy =
        decltype(std::declval<ForwardingAction>().Depends.begin());
    using EdgeTy = std::pair<ForwardingAction *, ChildItTy>;

    DenseSet<ForwardingAction::KeyTy> Visited;
    SmallVector<EdgeTy, 32> Stack;
    SmallVector<ForwardingAction *, 32> Ordered;

    // Seed the tree search using the root value.
    assert(ForwardingActions.count({UseVal, Stmt}));
    ForwardingAction *RootAction = &ForwardingActions[{UseVal, Stmt}];
    Stack.emplace_back(RootAction, RootAction->Depends.begin());

    // Compute the postorder of the operand tree: all operands of an instruction
    // must be visited before the instruction itself. As an additional
    // requirement, the topological ordering must be 'compact': Any subtree node
    // must not be interleaved with nodes from a non-shared subtree. This is
    // because the same llvm::Instruction can be materialized multiple times as
    // used at different ScopStmts which might be different values. Intersecting
    // these lifetimes may result in miscompilations.
    // FIXME: Intersecting lifetimes might still be possible for the roots
    // themselves, since instructions are just prepended to a ScopStmt's
    // instruction list.
    while (!Stack.empty()) {
      EdgeTy &Top = Stack.back();
      ForwardingAction *TopAction = Top.first;
      ChildItTy &TopEdge = Top.second;

      if (TopEdge == TopAction->Depends.end()) {
        // Postorder sorting
        Ordered.push_back(TopAction);
        Stack.pop_back();
        continue;
      }
      ForwardingAction::KeyTy Key = *TopEdge;

      // Next edge for this level
      ++TopEdge;

      auto VisitIt = Visited.insert(Key);
      if (!VisitIt.second)
        continue;

      assert(ForwardingActions.count(Key) &&
             "Must not insert new actions during execution phase");
      ForwardingAction *ChildAction = &ForwardingActions[Key];
      Stack.emplace_back(ChildAction, ChildAction->Depends.begin());
    }

    // Actually, we need the reverse postorder because actions prepend new
    // instructions. Therefore, the first one will always be the action for the
    // operand tree's root.
    assert(Ordered.back() == RootAction);
    if (RootAction->Execute())
      Stmt->removeSingleMemoryAccess(RA);
    Ordered.pop_back();
    for (auto DepAction : reverse(Ordered)) {
      assert(DepAction->Decision != FD_Unknown &&
             DepAction->Decision != FD_CannotForward);
      assert(DepAction != RootAction);
      DepAction->Execute();
    }
  }

  /// Try to forward an operand tree rooted in @p RA.
  bool tryForwardTree(MemoryAccess *RA) {
    assert(RA->isLatestScalarKind());
    LLVM_DEBUG(dbgs() << "Trying to forward operand tree " << RA << "...\n");

    ScopStmt *Stmt = RA->getStatement();
    Loop *InLoop = Stmt->getSurroundingLoop();

    isl::map TargetToUse;
    if (!Known.is_null()) {
      isl::space DomSpace = Stmt->getDomainSpace();
      TargetToUse =
          isl::map::identity(DomSpace.map_from_domain_and_range(DomSpace));
    }

    ForwardingDecision Assessment =
        forwardTree(Stmt, RA->getAccessValue(), Stmt, InLoop);

    // If considered feasible and profitable, forward it.
    bool Changed = false;
    if (Assessment == FD_CanForwardProfitably) {
      applyForwardingActions(Stmt, RA->getAccessValue(), RA);
      Changed = true;
    }

    ForwardingActions.clear();
    return Changed;
  }

  /// Return which SCoP this instance is processing.
  Scop *getScop() const { return S; }

  /// Run the algorithm: Use value read accesses as operand tree roots and try
  /// to forward them into the statement.
  bool forwardOperandTrees() {
    for (ScopStmt &Stmt : *S) {
      bool StmtModified = false;

      // Because we are modifying the MemoryAccess list, collect them first to
      // avoid iterator invalidation.
      SmallVector<MemoryAccess *, 16> Accs(Stmt.begin(), Stmt.end());

      for (MemoryAccess *RA : Accs) {
        if (!RA->isRead())
          continue;
        if (!RA->isLatestScalarKind())
          continue;

        if (tryForwardTree(RA)) {
          Modified = true;
          StmtModified = true;
          NumForwardedTrees++;
          TotalForwardedTrees++;
        }
      }

      if (StmtModified) {
        NumModifiedStmts++;
        TotalModifiedStmts++;
      }
    }

    if (Modified) {
      ScopsModified++;
      S->realignParams();
    }
    return Modified;
  }

  /// Print the pass result, performed transformations and the SCoP after the
  /// transformation.
  void print(raw_ostream &OS, int Indent = 0) {
    printStatistics(OS, Indent);

    if (!Modified) {
      // This line can easily be checked in regression tests.
      OS << "ForwardOpTree executed, but did not modify anything\n";
      return;
    }

    printStatements(OS, Indent);
  }

  bool isModified() const { return Modified; }
};

static std::unique_ptr<ForwardOpTreeImpl> runForwardOpTree(Scop &S,
                                                           LoopInfo &LI) {
  std::unique_ptr<ForwardOpTreeImpl> Impl;
  {
    IslMaxOperationsGuard MaxOpGuard(S.getIslCtx().get(), MaxOps, false);
    Impl = std::make_unique<ForwardOpTreeImpl>(&S, &LI, MaxOpGuard);

    if (AnalyzeKnown) {
      LLVM_DEBUG(dbgs() << "Prepare forwarders...\n");
      Impl->computeKnownValues();
    }

    LLVM_DEBUG(dbgs() << "Forwarding operand trees...\n");
    Impl->forwardOperandTrees();

    if (MaxOpGuard.hasQuotaExceeded()) {
      LLVM_DEBUG(dbgs() << "Not all operations completed because of "
                           "max_operations exceeded\n");
      KnownOutOfQuota++;
    }
  }

  LLVM_DEBUG(dbgs() << "\nFinal Scop:\n");
  LLVM_DEBUG(dbgs() << S);

  // Update statistics
  Scop::ScopStatistics ScopStats = S.getStatistics();
  NumValueWrites += ScopStats.NumValueWrites;
  NumValueWritesInLoops += ScopStats.NumValueWritesInLoops;
  NumPHIWrites += ScopStats.NumPHIWrites;
  NumPHIWritesInLoops += ScopStats.NumPHIWritesInLoops;
  NumSingletonWrites += ScopStats.NumSingletonWrites;
  NumSingletonWritesInLoops += ScopStats.NumSingletonWritesInLoops;

  return Impl;
}

static PreservedAnalyses
runForwardOpTreeUsingNPM(Scop &S, ScopAnalysisManager &SAM,
                         ScopStandardAnalysisResults &SAR, SPMUpdater &U,
                         raw_ostream *OS) {
  LoopInfo &LI = SAR.LI;

  std::unique_ptr<ForwardOpTreeImpl> Impl = runForwardOpTree(S, LI);
  if (OS) {
    *OS << "Printing analysis 'Polly - Forward operand tree' for region: '"
        << S.getName() << "' in function '" << S.getFunction().getName()
        << "':\n";
    if (Impl) {
      assert(Impl->getScop() == &S);

      Impl->print(*OS);
    }
  }

  if (!Impl->isModified())
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<AllAnalysesOn<Module>>();
  PA.preserveSet<AllAnalysesOn<Function>>();
  PA.preserveSet<AllAnalysesOn<Loop>>();
  return PA;
}

/// Pass that redirects scalar reads to array elements that are known to contain
/// the same value.
///
/// This reduces the number of scalar accesses and therefore potentially
/// increases the freedom of the scheduler. In the ideal case, all reads of a
/// scalar definition are redirected (We currently do not care about removing
/// the write in this case).  This is also useful for the main DeLICM pass as
/// there are less scalars to be mapped.
class ForwardOpTreeWrapperPass final : public ScopPass {
private:
  /// The pass implementation, also holding per-scop data.
  std::unique_ptr<ForwardOpTreeImpl> Impl;

public:
  static char ID;

  explicit ForwardOpTreeWrapperPass() : ScopPass(ID) {}
  ForwardOpTreeWrapperPass(const ForwardOpTreeWrapperPass &) = delete;
  ForwardOpTreeWrapperPass &
  operator=(const ForwardOpTreeWrapperPass &) = delete;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitive<ScopInfoRegionPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesAll();
  }

  bool runOnScop(Scop &S) override {
    // Free resources for previous SCoP's computation, if not yet done.
    releaseMemory();

    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

    Impl = runForwardOpTree(S, LI);

    return false;
  }

  void printScop(raw_ostream &OS, Scop &S) const override {
    if (!Impl)
      return;

    assert(Impl->getScop() == &S);
    Impl->print(OS);
  }

  void releaseMemory() override { Impl.reset(); }
}; // class ForwardOpTree

char ForwardOpTreeWrapperPass::ID;

/// Print result from ForwardOpTreeWrapperPass.
class ForwardOpTreePrinterLegacyPass final : public ScopPass {
public:
  static char ID;

  ForwardOpTreePrinterLegacyPass() : ForwardOpTreePrinterLegacyPass(outs()){};
  explicit ForwardOpTreePrinterLegacyPass(llvm::raw_ostream &OS)
      : ScopPass(ID), OS(OS) {}

  bool runOnScop(Scop &S) override {
    ForwardOpTreeWrapperPass &P = getAnalysis<ForwardOpTreeWrapperPass>();

    OS << "Printing analysis '" << P.getPassName() << "' for region: '"
       << S.getRegion().getNameStr() << "' in function '"
       << S.getFunction().getName() << "':\n";
    P.printScop(OS, S);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ScopPass::getAnalysisUsage(AU);
    AU.addRequired<ForwardOpTreeWrapperPass>();
    AU.setPreservesAll();
  }

private:
  llvm::raw_ostream &OS;
};

char ForwardOpTreePrinterLegacyPass::ID = 0;
} // namespace

Pass *polly::createForwardOpTreeWrapperPass() {
  return new ForwardOpTreeWrapperPass();
}

Pass *polly::createForwardOpTreePrinterLegacyPass(llvm::raw_ostream &OS) {
  return new ForwardOpTreePrinterLegacyPass(OS);
}

llvm::PreservedAnalyses ForwardOpTreePass::run(Scop &S,
                                               ScopAnalysisManager &SAM,
                                               ScopStandardAnalysisResults &SAR,
                                               SPMUpdater &U) {
  return runForwardOpTreeUsingNPM(S, SAM, SAR, U, nullptr);
}

llvm::PreservedAnalyses
ForwardOpTreePrinterPass::run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &SAR, SPMUpdater &U) {
  return runForwardOpTreeUsingNPM(S, SAM, SAR, U, &OS);
}

INITIALIZE_PASS_BEGIN(ForwardOpTreeWrapperPass, "polly-optree",
                      "Polly - Forward operand tree", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(ForwardOpTreeWrapperPass, "polly-optree",
                    "Polly - Forward operand tree", false, false)

INITIALIZE_PASS_BEGIN(ForwardOpTreePrinterLegacyPass, "polly-print-optree",
                      "Polly - Print forward operand tree result", false, false)
INITIALIZE_PASS_DEPENDENCY(ForwardOpTreeWrapperPass)
INITIALIZE_PASS_END(ForwardOpTreePrinterLegacyPass, "polly-print-optree",
                    "Polly - Print forward operand tree result", false, false)
