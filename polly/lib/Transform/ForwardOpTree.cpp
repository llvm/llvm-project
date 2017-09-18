//===- ForwardOpTree.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Pass.h"
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

static cl::opt<unsigned long>
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

  /// The root instruction can be forwarded in a non-trivial way. This requires
  /// the operand tree root to be an instruction in some statement.
  FD_CanForwardTree,

  /// Used to indicate that a forwarding has be carried out successfully.
  FD_DidForward,

  /// A forwarding method cannot be applied to the operand tree.
  /// The difference to FD_CannotForward is that there might be other methods
  /// that can handle it.
  /// The conditions that make an operand tree applicable must be checked even
  /// with DoIt==true because a method following the one that returned
  /// FD_NotApplicable might have returned FD_CanForwardTree.
  FD_NotApplicable
};

/// Implementation of operand tree forwarding for a specific SCoP.
///
/// For a statement that requires a scalar value (through a value read
/// MemoryAccess), see if its operand can be moved into the statement. If so,
/// the MemoryAccess is removed and the all the operand tree instructions are
/// moved into the statement. All original instructions are left in the source
/// statements. The simplification pass can clean these up.
class ForwardOpTreeImpl : ZoneAlgorithm {
private:
  /// How many instructions have been copied to other statements.
  int NumInstructionsCopied = 0;

  /// Number of loads forwarded because their value was known.
  int NumKnownLoadsForwarded = 0;

  /// How many read-only accesses have been copied.
  int NumReadOnlyCopied = 0;

  /// How many operand trees have been forwarded.
  int NumForwardedTrees = 0;

  /// Number of statements with at least one forwarded operand tree.
  int NumModifiedStmts = 0;

  /// Whether we carried out at least one change to the SCoP.
  bool Modified = false;

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
    assert(ValInst.is_single_valued().is_true());

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

    // MemoryAccesses can read only elements from a single array
    // (i.e. not: { Dom[0] -> A[0]; Dom[1] -> B[1] }).
    // Look through all spaces until we find one that contains at least the
    // wanted statement instance.s
    MustKnown.foreach_map([&](isl::map Map) -> isl::stat {
      // Get the array this is accessing.
      isl::id ArrayId = Map.get_tuple_id(isl::dim::out);
      ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(ArrayId.get_user());

      // No support for generation of indirect array accesses.
      if (SAI->getBasePtrOriginSAI())
        return isl::stat::ok; // continue

      // Determine whether this map contains all wanted values.
      isl::set MapDom = Map.domain();
      if (!Domain.is_subset(MapDom).is_true())
        return isl::stat::ok; // continue

      // There might be multiple array elements that contain the same value, but
      // choose only one of them. lexmin is used because it returns a one-value
      // mapping, we do not care about which one.
      // TODO: Get the simplest access function.
      Result = Map.lexmin();
      return isl::stat::error; // break
    });

    return Result;
  }

public:
  ForwardOpTreeImpl(Scop *S, LoopInfo *LI)
      : ZoneAlgorithm("polly-optree", S, LI) {}

  /// Compute the zones of known array element contents.
  ///
  /// @return True if the computed #Known is usable.
  bool computeKnownValues() {
    isl::union_map MustKnown, KnownFromLoad, KnownFromInit;

    // Check that nothing strange occurs.
    collectCompatibleElts();

    isl_ctx_reset_error(IslCtx.get());
    {
      IslMaxOperationsGuard MaxOpGuard(IslCtx.get(), MaxOps);

      computeCommon();
      Known = computeKnown(true, true);

      // Preexisting ValInsts use the known content analysis of themselves.
      Translator = makeIdentityMap(Known.range(), false);
    }

    if (!Known || !Translator) {
      assert(isl_ctx_last_error(IslCtx.get()) == isl_error_quota);
      KnownOutOfQuota++;
      Known = nullptr;
      Translator = nullptr;
      DEBUG(dbgs() << "Known analysis exceeded max_operations\n");
      return false;
    }

    KnownAnalyzed++;
    DEBUG(dbgs() << "All known: " << Known << "\n");

    return true;
  }

  void printStatistics(raw_ostream &OS, int Indent = 0) {
    OS.indent(Indent) << "Statistics {\n";
    OS.indent(Indent + 4) << "Instructions copied: " << NumInstructionsCopied
                          << '\n';
    OS.indent(Indent + 4) << "Known loads forwarded: " << NumKnownLoadsForwarded
                          << '\n';
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

  /// For an llvm::Value defined in @p DefStmt, compute the RAW dependency for a
  /// use in every instance of @p UseStmt.
  ///
  /// @param UseStmt Statement a scalar is used in.
  /// @param DefStmt Statement a scalar is defined in.
  ///
  /// @return { DomainUse[] -> DomainDef[] }
  isl::map computeUseToDefFlowDependency(ScopStmt *UseStmt, ScopStmt *DefStmt) {
    // { DomainUse[] -> Scatter[] }
    isl::map UseScatter = getScatterFor(UseStmt);

    // { Zone[] -> DomainDef[] }
    isl::map ReachDefZone = getScalarReachingDefinition(DefStmt);

    // { Scatter[] -> DomainDef[] }
    isl::map ReachDefTimepoints =
        convertZoneToTimepoints(ReachDefZone, isl::dim::in, false, true);

    // { DomainUse[] -> DomainDef[] }
    return UseScatter.apply_range(ReachDefTimepoints);
  }

  /// Forward a load by reading from an array element that contains the same
  /// value. Typically the location it was loaded from.
  ///
  /// @param TargetStmt  The statement the operand tree will be copied to.
  /// @param Inst        The (possibly speculatable) instruction to forward.
  /// @param UseStmt     The statement that uses @p Inst.
  /// @param UseLoop     The loop @p Inst is used in.
  /// @param UseToTarget { DomainUse[] -> DomainTarget[] }
  ///                    A mapping from the statement instance @p Inst is used
  ///                    to the statement instance it is forwarded to.
  /// @param DefStmt     The statement @p Inst is defined in.
  /// @param DefLoop     The loop which contains @p Inst.
  /// @param DefToTarget { DomainDef[] -> DomainTarget[] }
  ///                    A mapping from the statement instance @p Inst is
  ///                    defined to the statement instance it is forwarded to.
  /// @param DoIt        If false, only determine whether an operand tree can be
  ///                    forwarded. If true, carry out the forwarding. Do not
  ///                    use DoIt==true if an operand tree is not known to be
  ///                    forwardable.
  ///
  /// @return FD_NotApplicable  if @p Inst is not a LoadInst.
  ///         FD_CannotForward  if no array element to load from was found.
  ///         FD_CanForwardLeaf if the load is already in the target statement
  ///                           instance.
  ///         FD_CanForwardTree if @p Inst is forwardable.
  ///         FD_DidForward     if @p DoIt was true.
  ForwardingDecision forwardKnownLoad(ScopStmt *TargetStmt, Instruction *Inst,
                                      ScopStmt *UseStmt, Loop *UseLoop,
                                      isl::map UseToTarget, ScopStmt *DefStmt,
                                      Loop *DefLoop, isl::map DefToTarget,
                                      bool DoIt) {
    // Cannot do anything without successful known analysis.
    if (Known.is_null())
      return FD_NotApplicable;

    LoadInst *LI = dyn_cast<LoadInst>(Inst);
    if (!LI)
      return FD_NotApplicable;

    // If the load is already in the statement, no forwarding is necessary.
    // However, it might happen that the LoadInst is already present in the
    // statement's instruction list. In that case we do as follows:
    // - For the evaluation (DoIt==false), we can trivially forward it as it is
    //   benefit of forwarding an already present instruction.
    // - For the execution (DoIt==true), prepend the instruction (to make it
    //   available to all instructions following in the instruction list), but
    //   do not add another MemoryAccess.
    MemoryAccess *Access = TargetStmt->getArrayAccessOrNULLFor(LI);
    if (Access && !DoIt)
      return FD_CanForwardTree;

    if (DoIt)
      TargetStmt->prependInstruction(LI);

    ForwardingDecision OpDecision =
        forwardTree(TargetStmt, LI->getPointerOperand(), DefStmt, DefLoop,
                    DefToTarget, DoIt);
    switch (OpDecision) {
    case FD_CannotForward:
      assert(!DoIt);
      return OpDecision;

    case FD_CanForwardLeaf:
    case FD_CanForwardTree:
      assert(!DoIt);
      break;

    case FD_DidForward:
      assert(DoIt);
      break;

    default:
      llvm_unreachable("Shouldn't return this");
    }

    // { DomainDef[] -> ValInst[] }
    isl::map ExpectedVal = makeValInst(Inst, UseStmt, UseLoop);

    // { DomainTarget[] -> ValInst[] }
    isl::map TargetExpectedVal = ExpectedVal.apply_domain(UseToTarget);
    isl::union_map TranslatedExpectedVal =
        isl::union_map(TargetExpectedVal).apply_range(Translator);

    // { DomainTarget[] -> Element[] }
    isl::union_map Candidates = findSameContentElements(TranslatedExpectedVal);

    isl::map SameVal = singleLocation(Candidates, getDomainFor(TargetStmt));
    if (!SameVal)
      return FD_CannotForward;

    if (!DoIt)
      return FD_CanForwardTree;

    if (Access) {
      DEBUG(dbgs() << "    forwarded known load with preexisting MemoryAccess"
                   << Access << "\n");
    } else {
      Access = makeReadArrayAccess(TargetStmt, LI, SameVal);
      DEBUG(dbgs() << "    forwarded known load with new MemoryAccess" << Access
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
      if (ValInstSpace.is_wrapping()) {
        // { DefDomain[] -> Value[] }
        isl::map ValInsts = ExpectedVal.range().unwrap();

        // { DefDomain[] }
        isl::set DefDomain = ValInsts.domain();

        // { Value[] }
        isl::space ValSpace = ValInstSpace.unwrap().range();

        // { Value[] -> Value[] }
        isl::map ValToVal =
            isl::map::identity(ValSpace.map_from_domain_and_range(ValSpace));

        // { [TargetDomain[] -> Value[]] -> [DefDomain[] -> Value] }
        isl::map LocalTranslator = DefToTarget.reverse().product(ValToVal);

        Translator = Translator.add_map(LocalTranslator);
        DEBUG(dbgs() << "      local translator is " << LocalTranslator
                     << "\n");
      }
    }
    DEBUG(dbgs() << "      expected values where " << TargetExpectedVal
                 << "\n");
    DEBUG(dbgs() << "      candidate elements where " << Candidates << "\n");
    assert(Access);

    NumKnownLoadsForwarded++;
    TotalKnownLoadsForwarded++;
    return FD_DidForward;
  }

  /// Forwards a speculatively executable instruction.
  ///
  /// @param TargetStmt  The statement the operand tree will be copied to.
  /// @param UseInst     The (possibly speculatable) instruction to forward.
  /// @param DefStmt     The statement @p UseInst is defined in.
  /// @param DefLoop     The loop which contains @p UseInst.
  /// @param DefToTarget { DomainDef[] -> DomainTarget[] }
  ///                    A mapping from the statement instance @p UseInst is
  ///                    defined to the statement instance it is forwarded to.
  /// @param DoIt        If false, only determine whether an operand tree can be
  ///                    forwarded. If true, carry out the forwarding. Do not
  ///                    use DoIt==true if an operand tree is not known to be
  ///                    forwardable.
  ///
  /// @return FD_NotApplicable  if @p UseInst is not speculatable.
  ///         FD_CannotForward  if one of @p UseInst's operands is not
  ///                           forwardable.
  ///         FD_CanForwardTree if @p UseInst is forwardable.
  ///         FD_DidForward     if @p DoIt was true.
  ForwardingDecision forwardSpeculatable(ScopStmt *TargetStmt,
                                         Instruction *UseInst,
                                         ScopStmt *DefStmt, Loop *DefLoop,
                                         isl::map DefToTarget, bool DoIt) {
    // PHIs, unless synthesizable, are not yet supported.
    if (isa<PHINode>(UseInst))
      return FD_NotApplicable;

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
    if (mayBeMemoryDependent(*UseInst))
      return FD_NotApplicable;

    if (DoIt) {
      // To ensure the right order, prepend this instruction before its
      // operands. This ensures that its operands are inserted before the
      // instruction using them.
      // TODO: The operand tree is not really a tree, but a DAG. We should be
      // able to handle DAGs without duplication.
      TargetStmt->prependInstruction(UseInst);
      NumInstructionsCopied++;
      TotalInstructionsCopied++;
    }

    for (Value *OpVal : UseInst->operand_values()) {
      ForwardingDecision OpDecision =
          forwardTree(TargetStmt, OpVal, DefStmt, DefLoop, DefToTarget, DoIt);
      switch (OpDecision) {
      case FD_CannotForward:
        assert(!DoIt);
        return FD_CannotForward;

      case FD_CanForwardLeaf:
      case FD_CanForwardTree:
        assert(!DoIt);
        break;

      case FD_DidForward:
        assert(DoIt);
        break;

      case FD_NotApplicable:
        llvm_unreachable("forwardTree should never return FD_NotApplicable");
      }
    }

    if (DoIt)
      return FD_DidForward;
    return FD_CanForwardTree;
  }

  /// Determines whether an operand tree can be forwarded or carries out a
  /// forwarding, depending on the @p DoIt flag.
  ///
  /// @param TargetStmt  The statement the operand tree will be copied to.
  /// @param UseVal      The value (usually an instruction) which is root of an
  ///                    operand tree.
  /// @param UseStmt     The statement that uses @p UseVal.
  /// @param UseLoop     The loop @p UseVal is used in.
  /// @param UseToTarget { DomainUse[] -> DomainTarget[] }
  ///                    A mapping from the statement instance @p UseVal is used
  ///                    to the statement instance it is forwarded to.
  /// @param DoIt        If false, only determine whether an operand tree can be
  ///                    forwarded. If true, carry out the forwarding. Do not
  ///                    use DoIt==true if an operand tree is not known to be
  ///                    forwardable.
  ///
  /// @return If DoIt==false, return whether the operand tree can be forwarded.
  ///         If DoIt==true, return FD_DidForward.
  ForwardingDecision forwardTree(ScopStmt *TargetStmt, Value *UseVal,
                                 ScopStmt *UseStmt, Loop *UseLoop,
                                 isl::map UseToTarget, bool DoIt) {
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
      if (DoIt)
        return FD_DidForward;
      return FD_CanForwardLeaf;

    case VirtualUse::Synthesizable: {
      // ScopExpander will take care for of generating the code at the new
      // location.
      if (DoIt)
        return FD_DidForward;

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
        return FD_CanForwardLeaf;

      DEBUG(dbgs() << "    Synthesizable would not be synthesizable anymore: "
                   << *UseVal << "\n");
      return FD_CannotForward;
    }

    case VirtualUse::ReadOnly:
      // Note that we cannot return FD_CanForwardTree here. With a operand tree
      // depth of 0, UseVal is the use in TargetStmt that we try to replace.
      // With -polly-analyze-read-only-scalars=true we would ensure the
      // existence of a MemoryAccess (which already exists for a leaf) and be
      // removed again by tryForwardTree because it's goal is to remove this
      // scalar MemoryAccess. It interprets FD_CanForwardTree as the permission
      // to do so.
      if (!DoIt)
        return FD_CanForwardLeaf;

      // If we model read-only scalars, we need to create a MemoryAccess for it.
      if (ModelReadOnlyScalars)
        TargetStmt->ensureValueRead(UseVal);

      NumReadOnlyCopied++;
      TotalReadOnlyCopied++;
      return FD_DidForward;

    case VirtualUse::Intra:
      // Knowing that UseStmt and DefStmt are the same statement instance, just
      // reuse the information about UseStmt for DefStmt
      DefStmt = UseStmt;
      DefToTarget = UseToTarget;

      LLVM_FALLTHROUGH;
    case VirtualUse::Inter:
      Instruction *Inst = cast<Instruction>(UseVal);

      if (!DefStmt) {
        DefStmt = S->getStmtFor(Inst);
        if (!DefStmt)
          return FD_CannotForward;
      }

      DefLoop = LI->getLoopFor(Inst->getParent());

      if (DefToTarget.is_null() && !Known.is_null()) {
        // { UseDomain[] -> DefDomain[] }
        isl::map UseToDef = computeUseToDefFlowDependency(UseStmt, DefStmt);

        // { DefDomain[] -> UseDomain[] -> TargetDomain[] } shortened to
        // { DefDomain[] -> TargetDomain[] }
        DefToTarget = UseToTarget.apply_domain(UseToDef);
        simplify(DefToTarget);
      }

      ForwardingDecision SpeculativeResult = forwardSpeculatable(
          TargetStmt, Inst, DefStmt, DefLoop, DefToTarget, DoIt);
      if (SpeculativeResult != FD_NotApplicable)
        return SpeculativeResult;

      ForwardingDecision KnownResult =
          forwardKnownLoad(TargetStmt, Inst, UseStmt, UseLoop, UseToTarget,
                           DefStmt, DefLoop, DefToTarget, DoIt);
      if (KnownResult != FD_NotApplicable)
        return KnownResult;

      // When no method is found to forward the operand tree, we effectively
      // cannot handle it.
      DEBUG(dbgs() << "    Cannot forward instruction: " << *Inst << "\n");
      return FD_CannotForward;
    }

    llvm_unreachable("Case unhandled");
  }

  /// Try to forward an operand tree rooted in @p RA.
  bool tryForwardTree(MemoryAccess *RA) {
    assert(RA->isLatestScalarKind());
    DEBUG(dbgs() << "Trying to forward operand tree " << RA << "...\n");

    ScopStmt *Stmt = RA->getStatement();
    Loop *InLoop = Stmt->getSurroundingLoop();

    isl::map TargetToUse;
    if (!Known.is_null()) {
      isl::space DomSpace = Stmt->getDomainSpace();
      TargetToUse =
          isl::map::identity(DomSpace.map_from_domain_and_range(DomSpace));
    }

    ForwardingDecision Assessment = forwardTree(
        Stmt, RA->getAccessValue(), Stmt, InLoop, TargetToUse, false);
    assert(Assessment != FD_DidForward);
    if (Assessment != FD_CanForwardTree)
      return false;

    ForwardingDecision Execution = forwardTree(Stmt, RA->getAccessValue(), Stmt,
                                               InLoop, TargetToUse, true);
    assert(Execution == FD_DidForward &&
           "A previous positive assessment must also be executable");
    (void)Execution;

    Stmt->removeSingleMemoryAccess(RA);
    return true;
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
      SmallVector<MemoryAccess *, 16> Accs;
      for (MemoryAccess *RA : Stmt) {
        if (!RA->isRead())
          continue;
        if (!RA->isLatestScalarKind())
          continue;

        Accs.push_back(RA);
      }

      for (MemoryAccess *RA : Accs) {
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

    if (Modified)
      ScopsModified++;
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
};

/// Pass that redirects scalar reads to array elements that are known to contain
/// the same value.
///
/// This reduces the number of scalar accesses and therefore potentially
/// increases the freedom of the scheduler. In the ideal case, all reads of a
/// scalar definition are redirected (We currently do not care about removing
/// the write in this case).  This is also useful for the main DeLICM pass as
/// there are less scalars to be mapped.
class ForwardOpTree : public ScopPass {
private:
  /// The pass implementation, also holding per-scop data.
  std::unique_ptr<ForwardOpTreeImpl> Impl;

public:
  static char ID;

  explicit ForwardOpTree() : ScopPass(ID) {}
  ForwardOpTree(const ForwardOpTree &) = delete;
  ForwardOpTree &operator=(const ForwardOpTree &) = delete;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitive<ScopInfoRegionPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesAll();
  }

  bool runOnScop(Scop &S) override {
    // Free resources for previous SCoP's computation, if not yet done.
    releaseMemory();

    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    Impl = llvm::make_unique<ForwardOpTreeImpl>(&S, &LI);

    if (AnalyzeKnown) {
      DEBUG(dbgs() << "Prepare forwarders...\n");
      Impl->computeKnownValues();
    }

    DEBUG(dbgs() << "Forwarding operand trees...\n");
    Impl->forwardOperandTrees();

    DEBUG(dbgs() << "\nFinal Scop:\n");
    DEBUG(dbgs() << S);

    // Update statistics
    auto ScopStats = S.getStatistics();
    NumValueWrites += ScopStats.NumValueWrites;
    NumValueWritesInLoops += ScopStats.NumValueWritesInLoops;
    NumPHIWrites += ScopStats.NumPHIWrites;
    NumPHIWritesInLoops += ScopStats.NumPHIWritesInLoops;
    NumSingletonWrites += ScopStats.NumSingletonWrites;
    NumSingletonWritesInLoops += ScopStats.NumSingletonWritesInLoops;

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

char ForwardOpTree::ID;

} // namespace

ScopPass *polly::createForwardOpTreePass() { return new ForwardOpTree(); }

INITIALIZE_PASS_BEGIN(ForwardOpTree, "polly-optree",
                      "Polly - Forward operand tree", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(ForwardOpTree, "polly-optree",
                    "Polly - Forward operand tree", false, false)
