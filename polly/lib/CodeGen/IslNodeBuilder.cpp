//===- IslNodeBuilder.cpp - Translate an isl AST into a LLVM-IR AST -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the IslNodeBuilder, a class to translate an isl AST into
// a LLVM-IR AST.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/CodeGen/LoopGeneratorsGOMP.h"
#include "polly/CodeGen/LoopGeneratorsKMP.h"
#include "polly/CodeGen/RuntimeDebugBuilder.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Support/VirtualInstruction.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "isl/aff.h"
#include "isl/aff_type.h"
#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include "isl/val.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-codegen"

STATISTIC(VersionedScops, "Number of SCoPs that required versioning.");

STATISTIC(SequentialLoops, "Number of generated sequential for-loops");
STATISTIC(ParallelLoops, "Number of generated parallel for-loops");
STATISTIC(IfConditions, "Number of generated if-conditions");

/// OpenMP backend options
enum class OpenMPBackend { GNU, LLVM };

static cl::opt<bool> PollyGenerateRTCPrint(
    "polly-codegen-emit-rtc-print",
    cl::desc("Emit code that prints the runtime check result dynamically."),
    cl::Hidden, cl::cat(PollyCategory));

// If this option is set we always use the isl AST generator to regenerate
// memory accesses. Without this option set we regenerate expressions using the
// original SCEV expressions and only generate new expressions in case the
// access relation has been changed and consequently must be regenerated.
static cl::opt<bool> PollyGenerateExpressions(
    "polly-codegen-generate-expressions",
    cl::desc("Generate AST expressions for unmodified and modified accesses"),
    cl::Hidden, cl::cat(PollyCategory));

static cl::opt<int> PollyTargetFirstLevelCacheLineSize(
    "polly-target-first-level-cache-line-size",
    cl::desc("The size of the first level cache line size specified in bytes."),
    cl::Hidden, cl::init(64), cl::cat(PollyCategory));

static cl::opt<OpenMPBackend> PollyOmpBackend(
    "polly-omp-backend", cl::desc("Choose the OpenMP library to use:"),
    cl::values(clEnumValN(OpenMPBackend::GNU, "GNU", "GNU OpenMP"),
               clEnumValN(OpenMPBackend::LLVM, "LLVM", "LLVM OpenMP")),
    cl::Hidden, cl::init(OpenMPBackend::GNU), cl::cat(PollyCategory));

isl::ast_expr IslNodeBuilder::getUpperBound(isl::ast_node_for For,
                                            ICmpInst::Predicate &Predicate) {
  isl::ast_expr Cond = For.cond();
  isl::ast_expr Iterator = For.iterator();
  assert(isl_ast_expr_get_type(Cond.get()) == isl_ast_expr_op &&
         "conditional expression is not an atomic upper bound");

  isl_ast_op_type OpType = isl_ast_expr_get_op_type(Cond.get());

  switch (OpType) {
  case isl_ast_op_le:
    Predicate = ICmpInst::ICMP_SLE;
    break;
  case isl_ast_op_lt:
    Predicate = ICmpInst::ICMP_SLT;
    break;
  default:
    llvm_unreachable("Unexpected comparison type in loop condition");
  }

  isl::ast_expr Arg0 = Cond.get_op_arg(0);

  assert(isl_ast_expr_get_type(Arg0.get()) == isl_ast_expr_id &&
         "conditional expression is not an atomic upper bound");

  isl::id UBID = Arg0.get_id();

  assert(isl_ast_expr_get_type(Iterator.get()) == isl_ast_expr_id &&
         "Could not get the iterator");

  isl::id IteratorID = Iterator.get_id();

  assert(UBID.get() == IteratorID.get() &&
         "conditional expression is not an atomic upper bound");

  return Cond.get_op_arg(1);
}

int IslNodeBuilder::getNumberOfIterations(isl::ast_node_for For) {
  assert(isl_ast_node_get_type(For.get()) == isl_ast_node_for);
  isl::ast_node Body = For.body();

  // First, check if we can actually handle this code.
  switch (isl_ast_node_get_type(Body.get())) {
  case isl_ast_node_user:
    break;
  case isl_ast_node_block: {
    isl::ast_node_block BodyBlock = Body.as<isl::ast_node_block>();
    isl::ast_node_list List = BodyBlock.children();
    for (isl::ast_node Node : List) {
      isl_ast_node_type NodeType = isl_ast_node_get_type(Node.get());
      if (NodeType != isl_ast_node_user)
        return -1;
    }
    break;
  }
  default:
    return -1;
  }

  isl::ast_expr Init = For.init();
  if (!Init.isa<isl::ast_expr_int>() || !Init.val().is_zero())
    return -1;
  isl::ast_expr Inc = For.inc();
  if (!Inc.isa<isl::ast_expr_int>() || !Inc.val().is_one())
    return -1;
  CmpInst::Predicate Predicate;
  isl::ast_expr UB = getUpperBound(For, Predicate);
  if (!UB.isa<isl::ast_expr_int>())
    return -1;
  isl::val UpVal = UB.get_val();
  int NumberIterations = UpVal.get_num_si();
  if (NumberIterations < 0)
    return -1;
  if (Predicate == CmpInst::ICMP_SLT)
    return NumberIterations;
  else
    return NumberIterations + 1;
}

static void findReferencesByUse(Value *SrcVal, ScopStmt *UserStmt,
                                Loop *UserScope, const ValueMapT &GlobalMap,
                                SetVector<Value *> &Values,
                                SetVector<const SCEV *> &SCEVs) {
  VirtualUse VUse = VirtualUse::create(UserStmt, UserScope, SrcVal, true);
  switch (VUse.getKind()) {
  case VirtualUse::Constant:
    // When accelerator-offloading, GlobalValue is a host address whose content
    // must still be transferred to the GPU.
    if (isa<GlobalValue>(SrcVal))
      Values.insert(SrcVal);
    break;

  case VirtualUse::Synthesizable:
    SCEVs.insert(VUse.getScevExpr());
    return;

  case VirtualUse::Block:
  case VirtualUse::ReadOnly:
  case VirtualUse::Hoisted:
  case VirtualUse::Intra:
  case VirtualUse::Inter:
    break;
  }

  if (Value *NewVal = GlobalMap.lookup(SrcVal))
    Values.insert(NewVal);
}

static void findReferencesInInst(Instruction *Inst, ScopStmt *UserStmt,
                                 Loop *UserScope, const ValueMapT &GlobalMap,
                                 SetVector<Value *> &Values,
                                 SetVector<const SCEV *> &SCEVs) {
  for (Use &U : Inst->operands())
    findReferencesByUse(U.get(), UserStmt, UserScope, GlobalMap, Values, SCEVs);
}

static void findReferencesInStmt(ScopStmt *Stmt, SetVector<Value *> &Values,
                                 ValueMapT &GlobalMap,
                                 SetVector<const SCEV *> &SCEVs) {
  LoopInfo *LI = Stmt->getParent()->getLI();

  BasicBlock *BB = Stmt->getBasicBlock();
  Loop *Scope = LI->getLoopFor(BB);
  for (Instruction *Inst : Stmt->getInstructions())
    findReferencesInInst(Inst, Stmt, Scope, GlobalMap, Values, SCEVs);

  if (Stmt->isRegionStmt()) {
    for (BasicBlock *BB : Stmt->getRegion()->blocks()) {
      Loop *Scope = LI->getLoopFor(BB);
      for (Instruction &Inst : *BB)
        findReferencesInInst(&Inst, Stmt, Scope, GlobalMap, Values, SCEVs);
    }
  }
}

void polly::addReferencesFromStmt(ScopStmt *Stmt, void *UserPtr,
                                  bool CreateScalarRefs) {
  auto &References = *static_cast<SubtreeReferences *>(UserPtr);

  findReferencesInStmt(Stmt, References.Values, References.GlobalMap,
                       References.SCEVs);

  for (auto &Access : *Stmt) {
    if (References.ParamSpace) {
      isl::space ParamSpace = Access->getLatestAccessRelation().get_space();
      (*References.ParamSpace) =
          References.ParamSpace->align_params(ParamSpace);
    }

    if (Access->isLatestArrayKind()) {
      auto *BasePtr = Access->getLatestScopArrayInfo()->getBasePtr();
      if (Instruction *OpInst = dyn_cast<Instruction>(BasePtr))
        if (Stmt->getParent()->contains(OpInst))
          continue;

      References.Values.insert(BasePtr);
      continue;
    }

    if (CreateScalarRefs)
      References.Values.insert(References.BlockGen.getOrCreateAlloca(*Access));
  }
}

/// Extract the out-of-scop values and SCEVs referenced from a set describing
/// a ScopStmt.
///
/// This includes the SCEVUnknowns referenced by the SCEVs used in the
/// statement and the base pointers of the memory accesses. For scalar
/// statements we force the generation of alloca memory locations and list
/// these locations in the set of out-of-scop values as well.
///
/// @param Set     A set which references the ScopStmt we are interested in.
/// @param UserPtr A void pointer that can be casted to a SubtreeReferences
///                structure.
static void addReferencesFromStmtSet(isl::set Set, SubtreeReferences *UserPtr) {
  isl::id Id = Set.get_tuple_id();
  auto *Stmt = static_cast<ScopStmt *>(Id.get_user());
  addReferencesFromStmt(Stmt, UserPtr);
}

/// Extract the out-of-scop values and SCEVs referenced from a union set
/// referencing multiple ScopStmts.
///
/// This includes the SCEVUnknowns referenced by the SCEVs used in the
/// statement and the base pointers of the memory accesses. For scalar
/// statements we force the generation of alloca memory locations and list
/// these locations in the set of out-of-scop values as well.
///
/// @param USet       A union set referencing the ScopStmts we are interested
///                   in.
/// @param References The SubtreeReferences data structure through which
///                   results are returned and further information is
///                   provided.
static void addReferencesFromStmtUnionSet(isl::union_set USet,
                                          SubtreeReferences &References) {

  for (isl::set Set : USet.get_set_list())
    addReferencesFromStmtSet(Set, &References);
}

isl::union_map
IslNodeBuilder::getScheduleForAstNode(const isl::ast_node &Node) {
  return IslAstInfo::getSchedule(Node);
}

void IslNodeBuilder::getReferencesInSubtree(const isl::ast_node &For,
                                            SetVector<Value *> &Values,
                                            SetVector<const Loop *> &Loops) {
  SetVector<const SCEV *> SCEVs;
  SubtreeReferences References = {
      LI, SE, S, ValueMap, Values, SCEVs, getBlockGenerator(), nullptr};

  for (const auto &I : IDToValue)
    Values.insert(I.second);

  // NOTE: this is populated in IslNodeBuilder::addParameters
  for (const auto &I : OutsideLoopIterations)
    Values.insert(cast<SCEVUnknown>(I.second)->getValue());

  isl::union_set Schedule = getScheduleForAstNode(For).domain();
  addReferencesFromStmtUnionSet(Schedule, References);

  for (const SCEV *Expr : SCEVs) {
    findValues(Expr, SE, Values);
    findLoops(Expr, Loops);
  }

  Values.remove_if([](const Value *V) { return isa<GlobalValue>(V); });

  /// Note: Code generation of induction variables of loops outside Scops
  ///
  /// Remove loops that contain the scop or that are part of the scop, as they
  /// are considered local. This leaves only loops that are before the scop, but
  /// do not contain the scop itself.
  /// We ignore loops perfectly contained in the Scop because these are already
  /// generated at `IslNodeBuilder::addParameters`. These `Loops` are loops
  /// whose induction variables are referred to by the Scop, but the Scop is not
  /// fully contained in these Loops. Since there can be many of these,
  /// we choose to codegen these on-demand.
  /// @see IslNodeBuilder::materializeNonScopLoopInductionVariable.
  Loops.remove_if([this](const Loop *L) {
    return S.contains(L) || L->contains(S.getEntry());
  });

  // Contains Values that may need to be replaced with other values
  // due to replacements from the ValueMap. We should make sure
  // that we return correctly remapped values.
  // NOTE: this code path is tested by:
  //     1.  test/Isl/CodeGen/OpenMP/single_loop_with_loop_invariant_baseptr.ll
  //     2.  test/Isl/CodeGen/OpenMP/loop-body-references-outer-values-3.ll
  SetVector<Value *> ReplacedValues;
  for (Value *V : Values) {
    ReplacedValues.insert(getLatestValue(V));
  }
  Values = ReplacedValues;
}

Value *IslNodeBuilder::getLatestValue(Value *Original) const {
  auto It = ValueMap.find(Original);
  if (It == ValueMap.end())
    return Original;
  return It->second;
}

void IslNodeBuilder::createMark(__isl_take isl_ast_node *Node) {
  auto *Id = isl_ast_node_mark_get_id(Node);
  auto Child = isl_ast_node_mark_get_node(Node);
  isl_ast_node_free(Node);
  // If a child node of a 'SIMD mark' is a loop that has a single iteration,
  // it will be optimized away and we should skip it.
  if (strcmp(isl_id_get_name(Id), "SIMD") == 0 &&
      isl_ast_node_get_type(Child) == isl_ast_node_for) {
    createForSequential(isl::manage(Child).as<isl::ast_node_for>(), true);
    isl_id_free(Id);
    return;
  }

  BandAttr *ChildLoopAttr = getLoopAttr(isl::manage_copy(Id));
  BandAttr *AncestorLoopAttr;
  if (ChildLoopAttr) {
    // Save current LoopAttr environment to restore again when leaving this
    // subtree. This means there was no loop between the ancestor LoopAttr and
    // this mark, i.e. the ancestor LoopAttr did not directly mark a loop. This
    // can happen e.g. if the AST build peeled or unrolled the loop.
    AncestorLoopAttr = Annotator.getStagingAttrEnv();

    Annotator.getStagingAttrEnv() = ChildLoopAttr;
  }

  create(Child);

  if (ChildLoopAttr) {
    assert(Annotator.getStagingAttrEnv() == ChildLoopAttr &&
           "Nest must not overwrite loop attr environment");
    Annotator.getStagingAttrEnv() = AncestorLoopAttr;
  }

  isl_id_free(Id);
}

/// Restore the initial ordering of dimensions of the band node
///
/// In case the band node represents all the dimensions of the iteration
/// domain, recreate the band node to restore the initial ordering of the
/// dimensions.
///
/// @param Node The band node to be modified.
/// @return The modified schedule node.
static bool IsLoopVectorizerDisabled(isl::ast_node_for Node) {
  assert(isl_ast_node_get_type(Node.get()) == isl_ast_node_for);
  isl::ast_node Body = Node.body();
  if (isl_ast_node_get_type(Body.get()) != isl_ast_node_mark)
    return false;

  isl::ast_node_mark BodyMark = Body.as<isl::ast_node_mark>();
  auto Id = BodyMark.id();
  if (strcmp(Id.get_name().c_str(), "Loop Vectorizer Disabled") == 0)
    return true;
  return false;
}

void IslNodeBuilder::createForSequential(isl::ast_node_for For,
                                         bool MarkParallel) {
  Value *ValueLB, *ValueUB, *ValueInc;
  Type *MaxType;
  BasicBlock *ExitBlock;
  Value *IV;
  CmpInst::Predicate Predicate;

  bool LoopVectorizerDisabled = IsLoopVectorizerDisabled(For);

  isl::ast_node Body = For.body();

  // isl_ast_node_for_is_degenerate(For)
  //
  // TODO: For degenerated loops we could generate a plain assignment.
  //       However, for now we just reuse the logic for normal loops, which will
  //       create a loop with a single iteration.

  isl::ast_expr Init = For.init();
  isl::ast_expr Inc = For.inc();
  isl::ast_expr Iterator = For.iterator();
  isl::id IteratorID = Iterator.get_id();
  isl::ast_expr UB = getUpperBound(For, Predicate);

  ValueLB = ExprBuilder.create(Init.release());
  ValueUB = ExprBuilder.create(UB.release());
  ValueInc = ExprBuilder.create(Inc.release());

  MaxType = ExprBuilder.getType(Iterator.get());
  MaxType = ExprBuilder.getWidestType(MaxType, ValueLB->getType());
  MaxType = ExprBuilder.getWidestType(MaxType, ValueUB->getType());
  MaxType = ExprBuilder.getWidestType(MaxType, ValueInc->getType());

  if (MaxType != ValueLB->getType())
    ValueLB = Builder.CreateSExt(ValueLB, MaxType);
  if (MaxType != ValueUB->getType())
    ValueUB = Builder.CreateSExt(ValueUB, MaxType);
  if (MaxType != ValueInc->getType())
    ValueInc = Builder.CreateSExt(ValueInc, MaxType);

  // If we can show that LB <Predicate> UB holds at least once, we can
  // omit the GuardBB in front of the loop.
  bool UseGuardBB = !GenSE->isKnownPredicate(Predicate, GenSE->getSCEV(ValueLB),
                                             GenSE->getSCEV(ValueUB));
  IV = createLoop(ValueLB, ValueUB, ValueInc, Builder, *GenLI, *GenDT,
                  ExitBlock, Predicate, &Annotator, MarkParallel, UseGuardBB,
                  LoopVectorizerDisabled);
  IDToValue[IteratorID.get()] = IV;

  create(Body.release());

  Annotator.popLoop(MarkParallel);

  IDToValue.erase(IDToValue.find(IteratorID.get()));

  Builder.SetInsertPoint(&ExitBlock->front());

  SequentialLoops++;
}

void IslNodeBuilder::createForParallel(__isl_take isl_ast_node *For) {
  isl_ast_node *Body;
  isl_ast_expr *Init, *Inc, *Iterator, *UB;
  isl_id *IteratorID;
  Value *ValueLB, *ValueUB, *ValueInc;
  Type *MaxType;
  Value *IV;
  CmpInst::Predicate Predicate;

  // The preamble of parallel code interacts different than normal code with
  // e.g., scalar initialization. Therefore, we ensure the parallel code is
  // separated from the last basic block.
  BasicBlock *ParBB = SplitBlock(Builder.GetInsertBlock(),
                                 &*Builder.GetInsertPoint(), &DT, &LI);
  ParBB->setName("polly.parallel.for");
  Builder.SetInsertPoint(&ParBB->front());

  Body = isl_ast_node_for_get_body(For);
  Init = isl_ast_node_for_get_init(For);
  Inc = isl_ast_node_for_get_inc(For);
  Iterator = isl_ast_node_for_get_iterator(For);
  IteratorID = isl_ast_expr_get_id(Iterator);
  UB = getUpperBound(isl::manage_copy(For).as<isl::ast_node_for>(), Predicate)
           .release();

  ValueLB = ExprBuilder.create(Init);
  ValueUB = ExprBuilder.create(UB);
  ValueInc = ExprBuilder.create(Inc);

  // OpenMP always uses SLE. In case the isl generated AST uses a SLT
  // expression, we need to adjust the loop bound by one.
  if (Predicate == CmpInst::ICMP_SLT)
    ValueUB = Builder.CreateAdd(
        ValueUB, Builder.CreateSExt(Builder.getTrue(), ValueUB->getType()));

  MaxType = ExprBuilder.getType(Iterator);
  MaxType = ExprBuilder.getWidestType(MaxType, ValueLB->getType());
  MaxType = ExprBuilder.getWidestType(MaxType, ValueUB->getType());
  MaxType = ExprBuilder.getWidestType(MaxType, ValueInc->getType());

  if (MaxType != ValueLB->getType())
    ValueLB = Builder.CreateSExt(ValueLB, MaxType);
  if (MaxType != ValueUB->getType())
    ValueUB = Builder.CreateSExt(ValueUB, MaxType);
  if (MaxType != ValueInc->getType())
    ValueInc = Builder.CreateSExt(ValueInc, MaxType);

  BasicBlock::iterator LoopBody;

  SetVector<Value *> SubtreeValues;
  SetVector<const Loop *> Loops;

  getReferencesInSubtree(isl::manage_copy(For), SubtreeValues, Loops);

  // Create for all loops we depend on values that contain the current loop
  // iteration. These values are necessary to generate code for SCEVs that
  // depend on such loops. As a result we need to pass them to the subfunction.
  // See [Code generation of induction variables of loops outside Scops]
  for (const Loop *L : Loops) {
    Value *LoopInductionVar = materializeNonScopLoopInductionVariable(L);
    SubtreeValues.insert(LoopInductionVar);
  }

  ValueMapT NewValues;

  std::unique_ptr<ParallelLoopGenerator> ParallelLoopGenPtr;

  switch (PollyOmpBackend) {
  case OpenMPBackend::GNU:
    ParallelLoopGenPtr.reset(new ParallelLoopGeneratorGOMP(Builder, DL));
    break;
  case OpenMPBackend::LLVM:
    ParallelLoopGenPtr.reset(new ParallelLoopGeneratorKMP(Builder, DL));
    break;
  }

  IV = ParallelLoopGenPtr->createParallelLoop(
      ValueLB, ValueUB, ValueInc, SubtreeValues, NewValues, &LoopBody);
  BasicBlock::iterator AfterLoop = Builder.GetInsertPoint();

  // Remember the parallel subfunction
  Function *SubFn = LoopBody->getFunction();
  ParallelSubfunctions.push_back(SubFn);

  // We start working on the outlined function. Since DominatorTree/LoopInfo are
  // not an inter-procedural passes, we temporarily switch them out. Save the
  // old ones first.
  Function *CallerFn = Builder.GetInsertBlock()->getParent();
  DominatorTree *CallerDT = GenDT;
  LoopInfo *CallerLI = GenLI;
  ScalarEvolution *CallerSE = GenSE;
  ValueMapT CallerGlobals = ValueMap;
  IslExprBuilder::IDToValueTy IDToValueCopy = IDToValue;

  // Get the analyses for the subfunction. ParallelLoopGenerator already create
  // DominatorTree and LoopInfo for us.
  DominatorTree *SubDT = ParallelLoopGenPtr->getCalleeDominatorTree();
  LoopInfo *SubLI = ParallelLoopGenPtr->getCalleeLoopInfo();

  // Create TargetLibraryInfo, AssumptionCachem and ScalarEvolution ourselves.
  // TODO: Ideally, we would use the pass manager's TargetLibraryInfoPass and
  // AssumptionAnalysis instead of our own. They contain more target-specific
  // information than we have available here: TargetLibraryInfoImpl can be a
  // derived class determined by TargetMachine, AssumptionCache can be
  // configured using a TargetTransformInfo object also derived from
  // TargetMachine.
  TargetLibraryInfoImpl BaselineInfoImpl(
      Triple(SubFn->getParent()->getTargetTriple()));
  TargetLibraryInfo CalleeTLI(BaselineInfoImpl, SubFn);
  AssumptionCache CalleeAC(*SubFn);
  std::unique_ptr<ScalarEvolution> SubSE = std::make_unique<ScalarEvolution>(
      *SubFn, CalleeTLI, CalleeAC, *SubDT, *SubLI);

  // Switch to the subfunction
  GenDT = SubDT;
  GenLI = SubLI;
  GenSE = SubSE.get();
  BlockGen.switchGeneratedFunc(SubFn, GenDT, GenLI, GenSE);
  RegionGen.switchGeneratedFunc(SubFn, GenDT, GenLI, GenSE);
  ExprBuilder.switchGeneratedFunc(SubFn, GenDT, GenLI, GenSE);
  Builder.SetInsertPoint(&*LoopBody);

  // Update the ValueMap to use instructions in the subfunction. Note that
  // "GlobalMap" used in BlockGenerator/IslExprBuilder is a reference to this
  // ValueMap.
  for (auto &[OldVal, NewVal] : ValueMap) {
    NewVal = NewValues.lookup(NewVal);

    // Clean-up any value that getReferencesInSubtree thinks we do not need.
    // DenseMap::erase only writes a tombstone (and destroys OldVal/NewVal), so
    // does not invalidate our iterator.
    if (!NewVal)
      ValueMap.erase(OldVal);
  }

  // This is for NewVals that do not appear in ValueMap (such as SCoP-invariant
  // values whose original value can be reused as long as we are in the same
  // function). No need to map the others.
  for (auto &[NewVal, NewNewVal] : NewValues) {
    if (Instruction *NewValInst = dyn_cast<Instruction>((Value *)NewVal)) {
      if (S.contains(NewValInst))
        continue;
      assert(NewValInst->getFunction() == &S.getFunction());
    }
    assert(!ValueMap.contains(NewVal));
    ValueMap[NewVal] = NewNewVal;
  }

  // Also update the IDToValue map to use instructions from the subfunction.
  for (auto &[OldVal, NewVal] : IDToValue) {
    NewVal = NewValues.lookup(NewVal);
    assert(NewVal);
  }
  IDToValue[IteratorID] = IV;

#ifndef NDEBUG
  // Check whether the maps now exclusively refer to SubFn values.
  for (auto &[OldVal, SubVal] : ValueMap) {
    Instruction *SubInst = dyn_cast<Instruction>((Value *)SubVal);
    assert(SubInst->getFunction() == SubFn &&
           "Instructions from outside the subfn cannot be accessed within the "
           "subfn");
  }
  for (auto &[Id, SubVal] : IDToValue) {
    Instruction *SubInst = dyn_cast<Instruction>((Value *)SubVal);
    assert(SubInst->getFunction() == SubFn &&
           "Instructions from outside the subfn cannot be accessed within the "
           "subfn");
  }
#endif

  ValueMapT NewValuesReverse;
  for (auto P : NewValues)
    NewValuesReverse[P.second] = P.first;

  Annotator.addAlternativeAliasBases(NewValuesReverse);

  create(Body);

  Annotator.resetAlternativeAliasBases();

  // Resume working on the caller function.
  GenDT = CallerDT;
  GenLI = CallerLI;
  GenSE = CallerSE;
  IDToValue = std::move(IDToValueCopy);
  ValueMap = std::move(CallerGlobals);
  ExprBuilder.switchGeneratedFunc(CallerFn, CallerDT, CallerLI, CallerSE);
  RegionGen.switchGeneratedFunc(CallerFn, CallerDT, CallerLI, CallerSE);
  BlockGen.switchGeneratedFunc(CallerFn, CallerDT, CallerLI, CallerSE);
  Builder.SetInsertPoint(&*AfterLoop);

  for (const Loop *L : Loops)
    OutsideLoopIterations.erase(L);

  isl_ast_node_free(For);
  isl_ast_expr_free(Iterator);
  isl_id_free(IteratorID);

  ParallelLoops++;
}

void IslNodeBuilder::createFor(__isl_take isl_ast_node *For) {
  if (IslAstInfo::isExecutedInParallel(isl::manage_copy(For))) {
    createForParallel(For);
    return;
  }
  bool Parallel = (IslAstInfo::isParallel(isl::manage_copy(For)) &&
                   !IslAstInfo::isReductionParallel(isl::manage_copy(For)));
  createForSequential(isl::manage(For).as<isl::ast_node_for>(), Parallel);
}

void IslNodeBuilder::createIf(__isl_take isl_ast_node *If) {
  isl_ast_expr *Cond = isl_ast_node_if_get_cond(If);

  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *CondBB = SplitBlock(Builder.GetInsertBlock(),
                                  &*Builder.GetInsertPoint(), GenDT, GenLI);
  CondBB->setName("polly.cond");
  BasicBlock *MergeBB = SplitBlock(CondBB, &CondBB->front(), GenDT, GenLI);
  MergeBB->setName("polly.merge");
  BasicBlock *ThenBB = BasicBlock::Create(Context, "polly.then", F);
  BasicBlock *ElseBB = BasicBlock::Create(Context, "polly.else", F);

  GenDT->addNewBlock(ThenBB, CondBB);
  GenDT->addNewBlock(ElseBB, CondBB);
  GenDT->changeImmediateDominator(MergeBB, CondBB);

  Loop *L = GenLI->getLoopFor(CondBB);
  if (L) {
    L->addBasicBlockToLoop(ThenBB, *GenLI);
    L->addBasicBlockToLoop(ElseBB, *GenLI);
  }

  CondBB->getTerminator()->eraseFromParent();

  Builder.SetInsertPoint(CondBB);
  Value *Predicate = ExprBuilder.create(Cond);
  Builder.CreateCondBr(Predicate, ThenBB, ElseBB);
  Builder.SetInsertPoint(ThenBB);
  Builder.CreateBr(MergeBB);
  Builder.SetInsertPoint(ElseBB);
  Builder.CreateBr(MergeBB);
  Builder.SetInsertPoint(&ThenBB->front());

  create(isl_ast_node_if_get_then(If));

  Builder.SetInsertPoint(&ElseBB->front());

  if (isl_ast_node_if_has_else(If))
    create(isl_ast_node_if_get_else(If));

  Builder.SetInsertPoint(&MergeBB->front());

  isl_ast_node_free(If);

  IfConditions++;
}

__isl_give isl_id_to_ast_expr *
IslNodeBuilder::createNewAccesses(ScopStmt *Stmt,
                                  __isl_keep isl_ast_node *Node) {
  isl::id_to_ast_expr NewAccesses =
      isl::id_to_ast_expr::alloc(Stmt->getParent()->getIslCtx(), 0);

  isl::ast_build Build = IslAstInfo::getBuild(isl::manage_copy(Node));
  assert(!Build.is_null() && "Could not obtain isl_ast_build from user node");
  Stmt->setAstBuild(Build);

  for (auto *MA : *Stmt) {
    if (!MA->hasNewAccessRelation()) {
      if (PollyGenerateExpressions) {
        if (!MA->isAffine())
          continue;
        if (MA->getLatestScopArrayInfo()->getBasePtrOriginSAI())
          continue;

        auto *BasePtr =
            dyn_cast<Instruction>(MA->getLatestScopArrayInfo()->getBasePtr());
        if (BasePtr && Stmt->getParent()->getRegion().contains(BasePtr))
          continue;
      } else {
        continue;
      }
    }
    assert(MA->isAffine() &&
           "Only affine memory accesses can be code generated");

    isl::union_map Schedule = Build.get_schedule();

#ifndef NDEBUG
    if (MA->isRead()) {
      auto Dom = Stmt->getDomain().release();
      auto SchedDom = isl_set_from_union_set(Schedule.domain().release());
      auto AccDom = isl_map_domain(MA->getAccessRelation().release());
      Dom = isl_set_intersect_params(Dom,
                                     Stmt->getParent()->getContext().release());
      SchedDom = isl_set_intersect_params(
          SchedDom, Stmt->getParent()->getContext().release());
      assert(isl_set_is_subset(SchedDom, AccDom) &&
             "Access relation not defined on full schedule domain");
      assert(isl_set_is_subset(Dom, AccDom) &&
             "Access relation not defined on full domain");
      isl_set_free(AccDom);
      isl_set_free(SchedDom);
      isl_set_free(Dom);
    }
#endif

    isl::pw_multi_aff PWAccRel = MA->applyScheduleToAccessRelation(Schedule);

    // isl cannot generate an index expression for access-nothing accesses.
    isl::set AccDomain = PWAccRel.domain();
    isl::set Context = S.getContext();
    AccDomain = AccDomain.intersect_params(Context);
    if (AccDomain.is_empty())
      continue;

    isl::ast_expr AccessExpr = Build.access_from(PWAccRel);
    NewAccesses = NewAccesses.set(MA->getId(), AccessExpr);
  }

  return NewAccesses.release();
}

void IslNodeBuilder::createSubstitutions(__isl_take isl_ast_expr *Expr,
                                         ScopStmt *Stmt, LoopToScevMapT &LTS) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "Expression of type 'op' expected");
  assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_call &&
         "Operation of type 'call' expected");
  for (int i = 0; i < isl_ast_expr_get_op_n_arg(Expr) - 1; ++i) {
    isl_ast_expr *SubExpr;
    Value *V;

    SubExpr = isl_ast_expr_get_op_arg(Expr, i + 1);
    V = ExprBuilder.create(SubExpr);
    ScalarEvolution *SE = Stmt->getParent()->getSE();
    LTS[Stmt->getLoopForDimension(i)] = SE->getUnknown(V);
  }

  isl_ast_expr_free(Expr);
}

void IslNodeBuilder::createSubstitutionsVector(
    __isl_take isl_ast_expr *Expr, ScopStmt *Stmt,
    std::vector<LoopToScevMapT> &VLTS, std::vector<Value *> &IVS,
    __isl_take isl_id *IteratorID) {
  int i = 0;

  Value *OldValue = IDToValue[IteratorID];
  for (Value *IV : IVS) {
    IDToValue[IteratorID] = IV;
    createSubstitutions(isl_ast_expr_copy(Expr), Stmt, VLTS[i]);
    i++;
  }

  IDToValue[IteratorID] = OldValue;
  isl_id_free(IteratorID);
  isl_ast_expr_free(Expr);
}

void IslNodeBuilder::generateCopyStmt(
    ScopStmt *Stmt, __isl_keep isl_id_to_ast_expr *NewAccesses) {
  assert(Stmt->size() == 2);
  auto ReadAccess = Stmt->begin();
  auto WriteAccess = ReadAccess++;
  assert((*ReadAccess)->isRead() && (*WriteAccess)->isMustWrite());
  assert((*ReadAccess)->getElementType() == (*WriteAccess)->getElementType() &&
         "Accesses use the same data type");
  assert((*ReadAccess)->isArrayKind() && (*WriteAccess)->isArrayKind());
  auto *AccessExpr =
      isl_id_to_ast_expr_get(NewAccesses, (*ReadAccess)->getId().release());
  auto *LoadValue = ExprBuilder.create(AccessExpr);
  AccessExpr =
      isl_id_to_ast_expr_get(NewAccesses, (*WriteAccess)->getId().release());
  auto *StoreAddr = ExprBuilder.createAccessAddress(AccessExpr).first;
  Builder.CreateStore(LoadValue, StoreAddr);
}

Value *IslNodeBuilder::materializeNonScopLoopInductionVariable(const Loop *L) {
  assert(!OutsideLoopIterations.contains(L) &&
         "trying to materialize loop induction variable twice");
  const SCEV *OuterLIV = SE.getAddRecExpr(SE.getUnknown(Builder.getInt64(0)),
                                          SE.getUnknown(Builder.getInt64(1)), L,
                                          SCEV::FlagAnyWrap);
  Value *V = generateSCEV(OuterLIV);
  OutsideLoopIterations[L] = SE.getUnknown(V);
  return V;
}

void IslNodeBuilder::createUser(__isl_take isl_ast_node *User) {
  LoopToScevMapT LTS;
  isl_id *Id;
  ScopStmt *Stmt;

  isl_ast_expr *Expr = isl_ast_node_user_get_expr(User);
  isl_ast_expr *StmtExpr = isl_ast_expr_get_op_arg(Expr, 0);
  Id = isl_ast_expr_get_id(StmtExpr);
  isl_ast_expr_free(StmtExpr);

  LTS.insert(OutsideLoopIterations.begin(), OutsideLoopIterations.end());

  Stmt = (ScopStmt *)isl_id_get_user(Id);
  auto *NewAccesses = createNewAccesses(Stmt, User);
  if (Stmt->isCopyStmt()) {
    generateCopyStmt(Stmt, NewAccesses);
    isl_ast_expr_free(Expr);
  } else {
    createSubstitutions(Expr, Stmt, LTS);

    if (Stmt->isBlockStmt())
      BlockGen.copyStmt(*Stmt, LTS, NewAccesses);
    else
      RegionGen.copyStmt(*Stmt, LTS, NewAccesses);
  }

  isl_id_to_ast_expr_free(NewAccesses);
  isl_ast_node_free(User);
  isl_id_free(Id);
}

void IslNodeBuilder::createBlock(__isl_take isl_ast_node *Block) {
  isl_ast_node_list *List = isl_ast_node_block_get_children(Block);

  for (int i = 0; i < isl_ast_node_list_n_ast_node(List); ++i)
    create(isl_ast_node_list_get_ast_node(List, i));

  isl_ast_node_free(Block);
  isl_ast_node_list_free(List);
}

void IslNodeBuilder::create(__isl_take isl_ast_node *Node) {
  switch (isl_ast_node_get_type(Node)) {
  case isl_ast_node_error:
    llvm_unreachable("code generation error");
  case isl_ast_node_mark:
    createMark(Node);
    return;
  case isl_ast_node_for:
    createFor(Node);
    return;
  case isl_ast_node_if:
    createIf(Node);
    return;
  case isl_ast_node_user:
    createUser(Node);
    return;
  case isl_ast_node_block:
    createBlock(Node);
    return;
  }

  llvm_unreachable("Unknown isl_ast_node type");
}

bool IslNodeBuilder::materializeValue(__isl_take isl_id *Id) {
  // If the Id is already mapped, skip it.
  if (!IDToValue.count(Id)) {
    auto *ParamSCEV = (const SCEV *)isl_id_get_user(Id);
    Value *V = nullptr;

    // Parameters could refer to invariant loads that need to be
    // preloaded before we can generate code for the parameter. Thus,
    // check if any value referred to in ParamSCEV is an invariant load
    // and if so make sure its equivalence class is preloaded.
    SetVector<Value *> Values;
    findValues(ParamSCEV, SE, Values);
    for (auto *Val : Values) {
      // Check if the value is an instruction in a dead block within the SCoP
      // and if so do not code generate it.
      if (auto *Inst = dyn_cast<Instruction>(Val)) {
        if (S.contains(Inst)) {
          bool IsDead = true;

          // Check for "undef" loads first, then if there is a statement for
          // the parent of Inst and lastly if the parent of Inst has an empty
          // domain. In the first and last case the instruction is dead but if
          // there is a statement or the domain is not empty Inst is not dead.
          auto MemInst = MemAccInst::dyn_cast(Inst);
          auto Address = MemInst ? MemInst.getPointerOperand() : nullptr;
          if (Address && SE.getUnknown(UndefValue::get(Address->getType())) ==
                             SE.getPointerBase(SE.getSCEV(Address))) {
          } else if (S.getStmtFor(Inst)) {
            IsDead = false;
          } else {
            auto *Domain = S.getDomainConditions(Inst->getParent()).release();
            IsDead = isl_set_is_empty(Domain);
            isl_set_free(Domain);
          }

          if (IsDead) {
            V = UndefValue::get(ParamSCEV->getType());
            break;
          }
        }
      }

      if (auto *IAClass = S.lookupInvariantEquivClass(Val)) {
        // Check if this invariant access class is empty, hence if we never
        // actually added a loads instruction to it. In that case it has no
        // (meaningful) users and we should not try to code generate it.
        if (IAClass->InvariantAccesses.empty())
          V = UndefValue::get(ParamSCEV->getType());

        if (!preloadInvariantEquivClass(*IAClass)) {
          isl_id_free(Id);
          return false;
        }
      }
    }

    V = V ? V : generateSCEV(ParamSCEV);
    IDToValue[Id] = V;
  }

  isl_id_free(Id);
  return true;
}

bool IslNodeBuilder::materializeParameters(__isl_take isl_set *Set) {
  for (unsigned i = 0, e = isl_set_dim(Set, isl_dim_param); i < e; ++i) {
    if (!isl_set_involves_dims(Set, isl_dim_param, i, 1))
      continue;
    isl_id *Id = isl_set_get_dim_id(Set, isl_dim_param, i);
    if (!materializeValue(Id))
      return false;
  }
  return true;
}

bool IslNodeBuilder::materializeParameters() {
  for (const SCEV *Param : S.parameters()) {
    isl_id *Id = S.getIdForParam(Param).release();
    if (!materializeValue(Id))
      return false;
  }
  return true;
}

Value *IslNodeBuilder::preloadUnconditionally(__isl_take isl_set *AccessRange,
                                              isl_ast_build *Build,
                                              Instruction *AccInst) {
  isl_pw_multi_aff *PWAccRel = isl_pw_multi_aff_from_set(AccessRange);
  isl_ast_expr *Access =
      isl_ast_build_access_from_pw_multi_aff(Build, PWAccRel);
  auto *Address = isl_ast_expr_address_of(Access);
  auto *AddressValue = ExprBuilder.create(Address);
  Value *PreloadVal;

  // Correct the type as the SAI might have a different type than the user
  // expects, especially if the base pointer is a struct.
  Type *Ty = AccInst->getType();

  auto *Ptr = AddressValue;
  auto Name = Ptr->getName();
  PreloadVal = Builder.CreateLoad(Ty, Ptr, Name + ".load");
  if (LoadInst *PreloadInst = dyn_cast<LoadInst>(PreloadVal))
    PreloadInst->setAlignment(cast<LoadInst>(AccInst)->getAlign());

  // TODO: This is only a hot fix for SCoP sequences that use the same load
  //       instruction contained and hoisted by one of the SCoPs.
  if (SE.isSCEVable(Ty))
    SE.forgetValue(AccInst);

  return PreloadVal;
}

Value *IslNodeBuilder::preloadInvariantLoad(const MemoryAccess &MA,
                                            __isl_take isl_set *Domain) {
  isl_set *AccessRange = isl_map_range(MA.getAddressFunction().release());
  AccessRange = isl_set_gist_params(AccessRange, S.getContext().release());

  if (!materializeParameters(AccessRange)) {
    isl_set_free(AccessRange);
    isl_set_free(Domain);
    return nullptr;
  }

  auto *Build =
      isl_ast_build_from_context(isl_set_universe(S.getParamSpace().release()));
  isl_set *Universe = isl_set_universe(isl_set_get_space(Domain));
  bool AlwaysExecuted = isl_set_is_equal(Domain, Universe);
  isl_set_free(Universe);

  Instruction *AccInst = MA.getAccessInstruction();
  Type *AccInstTy = AccInst->getType();

  Value *PreloadVal = nullptr;
  if (AlwaysExecuted) {
    PreloadVal = preloadUnconditionally(AccessRange, Build, AccInst);
    isl_ast_build_free(Build);
    isl_set_free(Domain);
    return PreloadVal;
  }

  if (!materializeParameters(Domain)) {
    isl_ast_build_free(Build);
    isl_set_free(AccessRange);
    isl_set_free(Domain);
    return nullptr;
  }

  isl_ast_expr *DomainCond = isl_ast_build_expr_from_set(Build, Domain);
  Domain = nullptr;

  ExprBuilder.setTrackOverflow(true);
  Value *Cond = ExprBuilder.createBool(DomainCond);
  Value *OverflowHappened = Builder.CreateNot(ExprBuilder.getOverflowState(),
                                              "polly.preload.cond.overflown");
  Cond = Builder.CreateAnd(Cond, OverflowHappened, "polly.preload.cond.result");
  ExprBuilder.setTrackOverflow(false);

  if (!Cond->getType()->isIntegerTy(1))
    Cond = Builder.CreateIsNotNull(Cond);

  BasicBlock *CondBB = SplitBlock(Builder.GetInsertBlock(),
                                  &*Builder.GetInsertPoint(), GenDT, GenLI);
  CondBB->setName("polly.preload.cond");

  BasicBlock *MergeBB = SplitBlock(CondBB, &CondBB->front(), GenDT, GenLI);
  MergeBB->setName("polly.preload.merge");

  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();
  BasicBlock *ExecBB = BasicBlock::Create(Context, "polly.preload.exec", F);

  GenDT->addNewBlock(ExecBB, CondBB);
  if (Loop *L = GenLI->getLoopFor(CondBB))
    L->addBasicBlockToLoop(ExecBB, *GenLI);

  auto *CondBBTerminator = CondBB->getTerminator();
  Builder.SetInsertPoint(CondBBTerminator);
  Builder.CreateCondBr(Cond, ExecBB, MergeBB);
  CondBBTerminator->eraseFromParent();

  Builder.SetInsertPoint(ExecBB);
  Builder.CreateBr(MergeBB);

  Builder.SetInsertPoint(ExecBB->getTerminator());
  Value *PreAccInst = preloadUnconditionally(AccessRange, Build, AccInst);
  Builder.SetInsertPoint(MergeBB->getTerminator());
  auto *MergePHI = Builder.CreatePHI(
      AccInstTy, 2, "polly.preload." + AccInst->getName() + ".merge");
  PreloadVal = MergePHI;

  if (!PreAccInst) {
    PreloadVal = nullptr;
    PreAccInst = UndefValue::get(AccInstTy);
  }

  MergePHI->addIncoming(PreAccInst, ExecBB);
  MergePHI->addIncoming(Constant::getNullValue(AccInstTy), CondBB);

  isl_ast_build_free(Build);
  return PreloadVal;
}

bool IslNodeBuilder::preloadInvariantEquivClass(
    InvariantEquivClassTy &IAClass) {
  // For an equivalence class of invariant loads we pre-load the representing
  // element with the unified execution context. However, we have to map all
  // elements of the class to the one preloaded load as they are referenced
  // during the code generation and therefore need to be mapped.
  const MemoryAccessList &MAs = IAClass.InvariantAccesses;
  if (MAs.empty())
    return true;

  MemoryAccess *MA = MAs.front();
  assert(MA->isArrayKind() && MA->isRead());

  // If the access function was already mapped, the preload of this equivalence
  // class was triggered earlier already and doesn't need to be done again.
  if (ValueMap.count(MA->getAccessInstruction()))
    return true;

  // Check for recursion which can be caused by additional constraints, e.g.,
  // non-finite loop constraints. In such a case we have to bail out and insert
  // a "false" runtime check that will cause the original code to be executed.
  auto PtrId = std::make_pair(IAClass.IdentifyingPointer, IAClass.AccessType);
  if (!PreloadedPtrs.insert(PtrId).second)
    return false;

  // The execution context of the IAClass.
  isl::set &ExecutionCtx = IAClass.ExecutionContext;

  // If the base pointer of this class is dependent on another one we have to
  // make sure it was preloaded already.
  auto *SAI = MA->getScopArrayInfo();
  if (auto *BaseIAClass = S.lookupInvariantEquivClass(SAI->getBasePtr())) {
    if (!preloadInvariantEquivClass(*BaseIAClass))
      return false;

    // After we preloaded the BaseIAClass we adjusted the BaseExecutionCtx and
    // we need to refine the ExecutionCtx.
    isl::set BaseExecutionCtx = BaseIAClass->ExecutionContext;
    ExecutionCtx = ExecutionCtx.intersect(BaseExecutionCtx);
  }

  // If the size of a dimension is dependent on another class, make sure it is
  // preloaded.
  for (unsigned i = 1, e = SAI->getNumberOfDimensions(); i < e; ++i) {
    const SCEV *Dim = SAI->getDimensionSize(i);
    SetVector<Value *> Values;
    findValues(Dim, SE, Values);
    for (auto *Val : Values) {
      if (auto *BaseIAClass = S.lookupInvariantEquivClass(Val)) {
        if (!preloadInvariantEquivClass(*BaseIAClass))
          return false;

        // After we preloaded the BaseIAClass we adjusted the BaseExecutionCtx
        // and we need to refine the ExecutionCtx.
        isl::set BaseExecutionCtx = BaseIAClass->ExecutionContext;
        ExecutionCtx = ExecutionCtx.intersect(BaseExecutionCtx);
      }
    }
  }

  Instruction *AccInst = MA->getAccessInstruction();
  Type *AccInstTy = AccInst->getType();

  Value *PreloadVal = preloadInvariantLoad(*MA, ExecutionCtx.copy());
  if (!PreloadVal)
    return false;

  for (const MemoryAccess *MA : MAs) {
    Instruction *MAAccInst = MA->getAccessInstruction();
    assert(PreloadVal->getType() == MAAccInst->getType());
    ValueMap[MAAccInst] = PreloadVal;
  }

  if (SE.isSCEVable(AccInstTy)) {
    isl_id *ParamId = S.getIdForParam(SE.getSCEV(AccInst)).release();
    if (ParamId)
      IDToValue[ParamId] = PreloadVal;
    isl_id_free(ParamId);
  }

  BasicBlock *EntryBB = &Builder.GetInsertBlock()->getParent()->getEntryBlock();
  auto *Alloca = new AllocaInst(AccInstTy, DL.getAllocaAddrSpace(),
                                AccInst->getName() + ".preload.s2a",
                                EntryBB->getFirstInsertionPt());
  Builder.CreateStore(PreloadVal, Alloca);
  ValueMapT PreloadedPointer;
  PreloadedPointer[PreloadVal] = AccInst;
  Annotator.addAlternativeAliasBases(PreloadedPointer);

  for (auto *DerivedSAI : SAI->getDerivedSAIs()) {
    Value *BasePtr = DerivedSAI->getBasePtr();

    for (const MemoryAccess *MA : MAs) {
      // As the derived SAI information is quite coarse, any load from the
      // current SAI could be the base pointer of the derived SAI, however we
      // should only change the base pointer of the derived SAI if we actually
      // preloaded it.
      if (BasePtr == MA->getOriginalBaseAddr()) {
        assert(BasePtr->getType() == PreloadVal->getType());
        DerivedSAI->setBasePtr(PreloadVal);
      }

      // For scalar derived SAIs we remap the alloca used for the derived value.
      if (BasePtr == MA->getAccessInstruction())
        ScalarMap[DerivedSAI] = Alloca;
    }
  }

  for (const MemoryAccess *MA : MAs) {
    Instruction *MAAccInst = MA->getAccessInstruction();
    // Use the escape system to get the correct value to users outside the SCoP.
    BlockGenerator::EscapeUserVectorTy EscapeUsers;
    for (auto *U : MAAccInst->users())
      if (Instruction *UI = dyn_cast<Instruction>(U))
        if (!S.contains(UI))
          EscapeUsers.push_back(UI);

    if (EscapeUsers.empty())
      continue;

    EscapeMap[MA->getAccessInstruction()] =
        std::make_pair(Alloca, std::move(EscapeUsers));
  }

  return true;
}

void IslNodeBuilder::allocateNewArrays(BBPair StartExitBlocks) {
  for (auto &SAI : S.arrays()) {
    if (SAI->getBasePtr())
      continue;

    assert(SAI->getNumberOfDimensions() > 0 && SAI->getDimensionSize(0) &&
           "The size of the outermost dimension is used to declare newly "
           "created arrays that require memory allocation.");

    Type *NewArrayType = nullptr;

    // Get the size of the array = size(dim_1)*...*size(dim_n)
    uint64_t ArraySizeInt = 1;
    for (int i = SAI->getNumberOfDimensions() - 1; i >= 0; i--) {
      auto *DimSize = SAI->getDimensionSize(i);
      unsigned UnsignedDimSize = static_cast<const SCEVConstant *>(DimSize)
                                     ->getAPInt()
                                     .getLimitedValue();

      if (!NewArrayType)
        NewArrayType = SAI->getElementType();

      NewArrayType = ArrayType::get(NewArrayType, UnsignedDimSize);
      ArraySizeInt *= UnsignedDimSize;
    }

    if (SAI->isOnHeap()) {
      LLVMContext &Ctx = NewArrayType->getContext();

      // Get the IntPtrTy from the Datalayout
      auto IntPtrTy = DL.getIntPtrType(Ctx);

      // Get the size of the element type in bits
      unsigned Size = SAI->getElemSizeInBytes();

      // Insert the malloc call at polly.start
      Builder.SetInsertPoint(std::get<0>(StartExitBlocks)->getTerminator());
      auto *CreatedArray = Builder.CreateMalloc(
          IntPtrTy, SAI->getElementType(),
          ConstantInt::get(Type::getInt64Ty(Ctx), Size),
          ConstantInt::get(Type::getInt64Ty(Ctx), ArraySizeInt), nullptr,
          SAI->getName());

      SAI->setBasePtr(CreatedArray);

      // Insert the free call at polly.exiting
      Builder.SetInsertPoint(std::get<1>(StartExitBlocks)->getTerminator());
      Builder.CreateFree(CreatedArray);
    } else {
      auto InstIt = Builder.GetInsertBlock()
                        ->getParent()
                        ->getEntryBlock()
                        .getTerminator()
                        ->getIterator();

      auto *CreatedArray = new AllocaInst(NewArrayType, DL.getAllocaAddrSpace(),
                                          SAI->getName(), InstIt);
      if (PollyTargetFirstLevelCacheLineSize)
        CreatedArray->setAlignment(Align(PollyTargetFirstLevelCacheLineSize));
      SAI->setBasePtr(CreatedArray);
    }
  }
}

bool IslNodeBuilder::preloadInvariantLoads() {
  auto &InvariantEquivClasses = S.getInvariantAccesses();
  if (InvariantEquivClasses.empty())
    return true;

  BasicBlock *PreLoadBB = SplitBlock(Builder.GetInsertBlock(),
                                     &*Builder.GetInsertPoint(), GenDT, GenLI);
  PreLoadBB->setName("polly.preload.begin");
  Builder.SetInsertPoint(&PreLoadBB->front());

  for (auto &IAClass : InvariantEquivClasses)
    if (!preloadInvariantEquivClass(IAClass))
      return false;

  return true;
}

void IslNodeBuilder::addParameters(__isl_take isl_set *Context) {
  // Materialize values for the parameters of the SCoP.
  materializeParameters();

  // Generate values for the current loop iteration for all surrounding loops.
  //
  // We may also reference loops outside of the scop which do not contain the
  // scop itself, but as the number of such scops may be arbitrarily large we do
  // not generate code for them here, but only at the point of code generation
  // where these values are needed.
  Loop *L = LI.getLoopFor(S.getEntry());

  while (L != nullptr && S.contains(L))
    L = L->getParentLoop();

  while (L != nullptr) {
    materializeNonScopLoopInductionVariable(L);
    L = L->getParentLoop();
  }

  isl_set_free(Context);
}

Value *IslNodeBuilder::generateSCEV(const SCEV *Expr) {
  /// We pass the insert location of our Builder, as Polly ensures during IR
  /// generation that there is always a valid CFG into which instructions are
  /// inserted. As a result, the insertpoint is known to be always followed by a
  /// terminator instruction. This means the insert point may be specified by a
  /// terminator instruction, but it can never point to an ->end() iterator
  /// which does not have a corresponding instruction. Hence, dereferencing
  /// the insertpoint to obtain an instruction is known to be save.
  ///
  /// We also do not need to update the Builder here, as new instructions are
  /// always inserted _before_ the given InsertLocation. As a result, the
  /// insert location remains valid.
  assert(Builder.GetInsertBlock()->end() != Builder.GetInsertPoint() &&
         "Insert location points after last valid instruction");
  Instruction *InsertLocation = &*Builder.GetInsertPoint();

  return expandCodeFor(S, SE, Builder.GetInsertBlock()->getParent(), *GenSE, DL,
                       "polly", Expr, Expr->getType(), InsertLocation,
                       &ValueMap, /*LoopToScevMap*/ nullptr,
                       StartBlock->getSinglePredecessor());
}

/// The AST expression we generate to perform the run-time check assumes
/// computations on integer types of infinite size. As we only use 64-bit
/// arithmetic we check for overflows, in case of which we set the result
/// of this run-time check to false to be conservatively correct,
Value *IslNodeBuilder::createRTC(isl_ast_expr *Condition) {
  auto ExprBuilder = getExprBuilder();

  // In case the AST expression has integers larger than 64 bit, bail out. The
  // resulting LLVM-IR will contain operations on types that use more than 64
  // bits. These are -- in case wrapping intrinsics are used -- translated to
  // runtime library calls that are not available on all systems (e.g., Android)
  // and consequently will result in linker errors.
  if (ExprBuilder.hasLargeInts(isl::manage_copy(Condition))) {
    isl_ast_expr_free(Condition);
    return Builder.getFalse();
  }

  ExprBuilder.setTrackOverflow(true);
  Value *RTC = ExprBuilder.create(Condition);
  if (!RTC->getType()->isIntegerTy(1))
    RTC = Builder.CreateIsNotNull(RTC);
  Value *OverflowHappened =
      Builder.CreateNot(ExprBuilder.getOverflowState(), "polly.rtc.overflown");

  if (PollyGenerateRTCPrint) {
    auto *F = Builder.GetInsertBlock()->getParent();
    RuntimeDebugBuilder::createCPUPrinter(
        Builder,
        "F: " + F->getName().str() + " R: " + S.getRegion().getNameStr() +
            "RTC: ",
        RTC, " Overflow: ", OverflowHappened,
        "\n"
        "  (0 failed, -1 succeeded)\n"
        "  (if one or both are 0 falling back to original code, if both are -1 "
        "executing Polly code)\n");
  }

  RTC = Builder.CreateAnd(RTC, OverflowHappened, "polly.rtc.result");
  ExprBuilder.setTrackOverflow(false);

  if (!isa<ConstantInt>(RTC))
    VersionedScops++;

  return RTC;
}
