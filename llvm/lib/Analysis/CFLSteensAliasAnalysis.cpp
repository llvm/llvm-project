//- CFLSteensAliasAnalysis.cpp - Unification-based Alias Analysis ---*- C++-*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a CFL-base, summary-based alias analysis algorithm. It
// does not depend on types. The algorithm is a mixture of the one described in
// "Demand-driven alias analysis for C" by Xin Zheng and Radu Rugina, and "Fast
// algorithms for Dyck-CFL-reachability with applications to Alias Analysis" by
// Zhang Q, Lyu M R, Yuan H, and Su Z. -- to summarize the papers, we build a
// graph of the uses of a variable, where each node is a memory location, and
// each edge is an action that happened on that memory location.  The "actions"
// can be one of Dereference, Reference, or Assign. The precision of this
// analysis is roughly the same as that of an one level context-sensitive
// Steensgaard's algorithm.
//
// Two variables are considered as aliasing iff you can reach one value's node
// from the other value's node and the language formed by concatenating all of
// the edge labels (actions) conforms to a context-free grammar.
//
// Because this algorithm requires a graph search on each query, we execute the
// algorithm outlined in "Fast algorithms..." (mentioned above)
// in order to transform the graph into sets of variables that may alias in
// ~nlogn time (n = number of variables), which makes queries take constant
// time.
//===----------------------------------------------------------------------===//

// N.B. AliasAnalysis as a whole is phrased as a FunctionPass at the moment, and
// CFLSteensAA is interprocedural. This is *technically* A Bad Thing, because
// FunctionPasses are only allowed to inspect the Function that they're being
// run on. Realistically, this likely isn't a problem until we allow
// FunctionPasses to run concurrently.

#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "CFLGraph.h"
#include "StratifiedSets.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <tuple>

using namespace llvm;
using namespace llvm::cflaa;

#define DEBUG_TYPE "cfl-steens-aa"

CFLSteensAAResult::CFLSteensAAResult(const TargetLibraryInfo &TLI)
    : AAResultBase(), TLI(TLI) {}
CFLSteensAAResult::CFLSteensAAResult(CFLSteensAAResult &&Arg)
    : AAResultBase(std::move(Arg)), TLI(Arg.TLI) {}
CFLSteensAAResult::~CFLSteensAAResult() {}

/// Information we have about a function and would like to keep around.
class CFLSteensAAResult::FunctionInfo {
  StratifiedSets<Value *> Sets;

  // RetParamRelations is a collection of ExternalRelations.
  SmallVector<ExternalRelation, 8> RetParamRelations;

  // RetParamAttributes is a collection of ExternalAttributes.
  SmallVector<ExternalAttribute, 8> RetParamAttributes;

public:
  FunctionInfo(Function &Fn, const SmallVectorImpl<Value *> &RetVals,
               StratifiedSets<Value *> S);

  const StratifiedSets<Value *> &getStratifiedSets() const { return Sets; }
  const SmallVectorImpl<ExternalRelation> &getRetParamRelations() const {
    return RetParamRelations;
  }
  const SmallVectorImpl<ExternalAttribute> &getRetParamAttributes() const {
    return RetParamAttributes;
  }
};

/// Try to go from a Value* to a Function*. Never returns nullptr.
static Optional<Function *> parentFunctionOfValue(Value *);

/// Returns possible functions called by the Inst* into the given
/// SmallVectorImpl. Returns true if targets found, false otherwise. This is
/// templated so we can use it with CallInsts and InvokeInsts.
static bool getPossibleTargets(CallSite, SmallVectorImpl<Function *> &);

const StratifiedIndex StratifiedLink::SetSentinel =
    std::numeric_limits<StratifiedIndex>::max();

namespace {

/// The maximum number of arguments we can put into a summary.
LLVM_CONSTEXPR unsigned MaxSupportedArgsInSummary = 50;

/// StratifiedSets call for knowledge of "direction", so this is how we
/// represent that locally.
enum class Level { Same, Above, Below };

/// Gets the edges our graph should have, based on an Instruction*
class GetEdgesVisitor : public InstVisitor<GetEdgesVisitor, void> {
  CFLSteensAAResult &AA;
  const TargetLibraryInfo &TLI;

  CFLGraph &Graph;
  SmallVectorImpl<Value *> &ReturnValues;
  SmallPtrSetImpl<Value *> &Externals;
  SmallPtrSetImpl<Value *> &Escapes;
  SmallVectorImpl<InstantiatedRelation> &InstantiatedRelations;
  SmallVectorImpl<InstantiatedAttr> &InstantiatedAttrs;

  static bool hasUsefulEdges(ConstantExpr *CE) {
    // ConstantExpr doesn't have terminators, invokes, or fences, so only needs
    // to check for compares.
    return CE->getOpcode() != Instruction::ICmp &&
           CE->getOpcode() != Instruction::FCmp;
  }

  void addNode(Value *Val) {
    if (!Graph.addNode(Val))
      return;

    if (isa<GlobalValue>(Val))
      Externals.insert(Val);
    else if (auto CExpr = dyn_cast<ConstantExpr>(Val))
      if (hasUsefulEdges(CExpr))
        visitConstantExpr(CExpr);
  }

  void addNodeWithAttr(Value *Val, AliasAttrs Attr) {
    addNode(Val);
    Graph.addAttr(Val, Attr);
  }

  void addEdge(Value *From, Value *To, EdgeType Type) {
    if (!From->getType()->isPointerTy() || !To->getType()->isPointerTy())
      return;
    addNode(From);
    if (To != From)
      addNode(To);
    Graph.addEdge(From, To, Type);
  }

public:
  GetEdgesVisitor(CFLSteensAAResult &AA, const TargetLibraryInfo &TLI,
                  CFLGraph &Graph, SmallVectorImpl<Value *> &ReturnValues,
                  SmallPtrSetImpl<Value *> &Externals,
                  SmallPtrSetImpl<Value *> &Escapes,
                  SmallVectorImpl<InstantiatedRelation> &InstantiatedRelations,
                  SmallVectorImpl<InstantiatedAttr> &InstantiatedAttrs)
      : AA(AA), TLI(TLI), Graph(Graph), ReturnValues(ReturnValues),
        Externals(Externals), Escapes(Escapes),
        InstantiatedRelations(InstantiatedRelations),
        InstantiatedAttrs(InstantiatedAttrs) {}

  void visitInstruction(Instruction &) {
    llvm_unreachable("Unsupported instruction encountered");
  }

  void visitReturnInst(ReturnInst &Inst) {
    if (auto RetVal = Inst.getReturnValue()) {
      if (RetVal->getType()->isPointerTy()) {
        addNode(RetVal);
        ReturnValues.push_back(RetVal);
      }
    }
  }

  void visitPtrToIntInst(PtrToIntInst &Inst) {
    auto *Ptr = Inst.getOperand(0);
    addNodeWithAttr(Ptr, getAttrEscaped());
  }

  void visitIntToPtrInst(IntToPtrInst &Inst) {
    auto *Ptr = &Inst;
    addNodeWithAttr(Ptr, getAttrUnknown());
  }

  void visitCastInst(CastInst &Inst) {
    auto *Src = Inst.getOperand(0);
    addEdge(Src, &Inst, EdgeType::Assign);
  }

  void visitBinaryOperator(BinaryOperator &Inst) {
    auto *Op1 = Inst.getOperand(0);
    auto *Op2 = Inst.getOperand(1);
    addEdge(Op1, &Inst, EdgeType::Assign);
    addEdge(Op2, &Inst, EdgeType::Assign);
  }

  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &Inst) {
    auto *Ptr = Inst.getPointerOperand();
    auto *Val = Inst.getNewValOperand();
    addEdge(Ptr, Val, EdgeType::Dereference);
  }

  void visitAtomicRMWInst(AtomicRMWInst &Inst) {
    auto *Ptr = Inst.getPointerOperand();
    auto *Val = Inst.getValOperand();
    addEdge(Ptr, Val, EdgeType::Dereference);
  }

  void visitPHINode(PHINode &Inst) {
    for (Value *Val : Inst.incoming_values())
      addEdge(Val, &Inst, EdgeType::Assign);
  }

  void visitGetElementPtrInst(GetElementPtrInst &Inst) {
    auto *Op = Inst.getPointerOperand();
    addEdge(Op, &Inst, EdgeType::Assign);
  }

  void visitSelectInst(SelectInst &Inst) {
    // Condition is not processed here (The actual statement producing
    // the condition result is processed elsewhere). For select, the
    // condition is evaluated, but not loaded, stored, or assigned
    // simply as a result of being the condition of a select.

    auto *TrueVal = Inst.getTrueValue();
    auto *FalseVal = Inst.getFalseValue();
    addEdge(TrueVal, &Inst, EdgeType::Assign);
    addEdge(FalseVal, &Inst, EdgeType::Assign);
  }

  void visitAllocaInst(AllocaInst &Inst) { Graph.addNode(&Inst); }

  void visitLoadInst(LoadInst &Inst) {
    auto *Ptr = Inst.getPointerOperand();
    auto *Val = &Inst;
    addEdge(Val, Ptr, EdgeType::Reference);
  }

  void visitStoreInst(StoreInst &Inst) {
    auto *Ptr = Inst.getPointerOperand();
    auto *Val = Inst.getValueOperand();
    addEdge(Ptr, Val, EdgeType::Dereference);
  }

  void visitVAArgInst(VAArgInst &Inst) {
    // We can't fully model va_arg here. For *Ptr = Inst.getOperand(0), it does
    // two things:
    //  1. Loads a value from *((T*)*Ptr).
    //  2. Increments (stores to) *Ptr by some target-specific amount.
    // For now, we'll handle this like a landingpad instruction (by placing the
    // result in its own group, and having that group alias externals).
    addNodeWithAttr(&Inst, getAttrUnknown());
  }

  static bool isFunctionExternal(Function *Fn) {
    return !Fn->hasExactDefinition();
  }

  bool tryInterproceduralAnalysis(CallSite CS,
                                  const SmallVectorImpl<Function *> &Fns) {
    assert(Fns.size() > 0);

    if (CS.arg_size() > MaxSupportedArgsInSummary)
      return false;

    // Exit early if we'll fail anyway
    for (auto *Fn : Fns) {
      if (isFunctionExternal(Fn) || Fn->isVarArg())
        return false;
      // Fail if the caller does not provide enough arguments
      assert(Fn->arg_size() <= CS.arg_size());
      auto &MaybeInfo = AA.ensureCached(Fn);
      if (!MaybeInfo.hasValue())
        return false;
    }

    for (auto *Fn : Fns) {
      auto &FnInfo = AA.ensureCached(Fn);
      assert(FnInfo.hasValue());

      auto &RetParamRelations = FnInfo->getRetParamRelations();
      for (auto &Relation : RetParamRelations) {
        auto IRelation = instantiateExternalRelation(Relation, CS);
        if (IRelation.hasValue())
          InstantiatedRelations.push_back(*IRelation);
      }

      auto &RetParamAttributes = FnInfo->getRetParamAttributes();
      for (auto &Attribute : RetParamAttributes) {
        auto IAttr = instantiateExternalAttribute(Attribute, CS);
        if (IAttr.hasValue())
          InstantiatedAttrs.push_back(*IAttr);
      }
    }

    return true;
  }

  void visitCallSite(CallSite CS) {
    auto Inst = CS.getInstruction();

    // Make sure all arguments and return value are added to the graph first
    for (Value *V : CS.args())
      addNode(V);
    if (Inst->getType()->isPointerTy())
      addNode(Inst);

    // Check if Inst is a call to a library function that allocates/deallocates
    // on the heap. Those kinds of functions do not introduce any aliases.
    // TODO: address other common library functions such as realloc(), strdup(),
    // etc.
    if (isMallocLikeFn(Inst, &TLI) || isCallocLikeFn(Inst, &TLI) ||
        isFreeCall(Inst, &TLI))
      return;

    // TODO: Add support for noalias args/all the other fun function attributes
    // that we can tack on.
    SmallVector<Function *, 4> Targets;
    if (getPossibleTargets(CS, Targets))
      if (tryInterproceduralAnalysis(CS, Targets))
        return;

    // Because the function is opaque, we need to note that anything
    // could have happened to the arguments (unless the function is marked
    // readonly or readnone), and that the result could alias just about
    // anything, too (unless the result is marked noalias).
    if (!CS.onlyReadsMemory())
      for (Value *V : CS.args()) {
        if (V->getType()->isPointerTy())
          Escapes.insert(V);
      }

    if (Inst->getType()->isPointerTy()) {
      auto *Fn = CS.getCalledFunction();
      if (Fn == nullptr || !Fn->doesNotAlias(0))
        Graph.addAttr(Inst, getAttrUnknown());
    }
  }

  /// Because vectors/aggregates are immutable and unaddressable, there's
  /// nothing we can do to coax a value out of them, other than calling
  /// Extract{Element,Value}. We can effectively treat them as pointers to
  /// arbitrary memory locations we can store in and load from.
  void visitExtractElementInst(ExtractElementInst &Inst) {
    auto *Ptr = Inst.getVectorOperand();
    auto *Val = &Inst;
    addEdge(Val, Ptr, EdgeType::Reference);
  }

  void visitInsertElementInst(InsertElementInst &Inst) {
    auto *Vec = Inst.getOperand(0);
    auto *Val = Inst.getOperand(1);
    addEdge(Vec, &Inst, EdgeType::Assign);
    addEdge(&Inst, Val, EdgeType::Dereference);
  }

  void visitLandingPadInst(LandingPadInst &Inst) {
    // Exceptions come from "nowhere", from our analysis' perspective.
    // So we place the instruction its own group, noting that said group may
    // alias externals
    addNodeWithAttr(&Inst, getAttrUnknown());
  }

  void visitInsertValueInst(InsertValueInst &Inst) {
    auto *Agg = Inst.getOperand(0);
    auto *Val = Inst.getOperand(1);
    addEdge(Agg, &Inst, EdgeType::Assign);
    addEdge(&Inst, Val, EdgeType::Dereference);
  }

  void visitExtractValueInst(ExtractValueInst &Inst) {
    auto *Ptr = Inst.getAggregateOperand();
    addEdge(&Inst, Ptr, EdgeType::Reference);
  }

  void visitShuffleVectorInst(ShuffleVectorInst &Inst) {
    auto *From1 = Inst.getOperand(0);
    auto *From2 = Inst.getOperand(1);
    addEdge(From1, &Inst, EdgeType::Assign);
    addEdge(From2, &Inst, EdgeType::Assign);
  }

  void visitConstantExpr(ConstantExpr *CE) {
    switch (CE->getOpcode()) {
    default:
      llvm_unreachable("Unknown instruction type encountered!");
// Build the switch statement using the Instruction.def file.
#define HANDLE_INST(NUM, OPCODE, CLASS)                                        \
  case Instruction::OPCODE:                                                    \
    visit##OPCODE(*(CLASS *)CE);                                               \
    break;
#include "llvm/IR/Instruction.def"
    }
  }
};

class CFLGraphBuilder {
  // Input of the builder
  CFLSteensAAResult &Analysis;
  const TargetLibraryInfo &TLI;

  // Output of the builder
  CFLGraph Graph;
  SmallVector<Value *, 4> ReturnedValues;

  // Auxiliary structures used by the builder
  SmallPtrSet<Value *, 8> ExternalValues;
  SmallPtrSet<Value *, 8> EscapedValues;
  SmallVector<InstantiatedRelation, 8> InstantiatedRelations;
  SmallVector<InstantiatedAttr, 8> InstantiatedAttrs;

  // Helper functions

  // Determines whether or not we an instruction is useless to us (e.g.
  // FenceInst)
  static bool hasUsefulEdges(Instruction *Inst) {
    bool IsNonInvokeRetTerminator = isa<TerminatorInst>(Inst) &&
                                    !isa<InvokeInst>(Inst) &&
                                    !isa<ReturnInst>(Inst);
    return !isa<CmpInst>(Inst) && !isa<FenceInst>(Inst) &&
           !IsNonInvokeRetTerminator;
  }

  void addArgumentToGraph(Argument &Arg) {
    if (Arg.getType()->isPointerTy()) {
      Graph.addNode(&Arg);
      ExternalValues.insert(&Arg);
    }
  }

  // Given an Instruction, this will add it to the graph, along with any
  // Instructions that are potentially only available from said Instruction
  // For example, given the following line:
  //   %0 = load i16* getelementptr ([1 x i16]* @a, 0, 0), align 2
  // addInstructionToGraph would add both the `load` and `getelementptr`
  // instructions to the graph appropriately.
  void addInstructionToGraph(Instruction &Inst) {
    if (!hasUsefulEdges(&Inst))
      return;

    GetEdgesVisitor(Analysis, TLI, Graph, ReturnedValues, ExternalValues,
                    EscapedValues, InstantiatedRelations, InstantiatedAttrs)
        .visit(Inst);
  }

  // Builds the graph needed for constructing the StratifiedSets for the given
  // function
  void buildGraphFrom(Function &Fn) {
    for (auto &Bb : Fn.getBasicBlockList())
      for (auto &Inst : Bb.getInstList())
        addInstructionToGraph(Inst);

    for (auto &Arg : Fn.args())
      addArgumentToGraph(Arg);
  }

public:
  CFLGraphBuilder(CFLSteensAAResult &Analysis, const TargetLibraryInfo &TLI,
                  Function &Fn)
      : Analysis(Analysis), TLI(TLI) {
    buildGraphFrom(Fn);
  }

  const CFLGraph &getCFLGraph() const { return Graph; }
  const SmallVector<Value *, 4> &getReturnValues() const {
    return ReturnedValues;
  }
  const SmallPtrSet<Value *, 8> &getExternalValues() const {
    return ExternalValues;
  }
  const SmallPtrSet<Value *, 8> &getEscapedValues() const {
    return EscapedValues;
  }
  const SmallVector<InstantiatedRelation, 8> &getInstantiatedRelations() const {
    return InstantiatedRelations;
  }
  const SmallVector<InstantiatedAttr, 8> &getInstantiatedAttrs() const {
    return InstantiatedAttrs;
  }
};
}

//===----------------------------------------------------------------------===//
// Function declarations that require types defined in the namespace above
//===----------------------------------------------------------------------===//

/// Gets the "Level" that one should travel in StratifiedSets
/// given an EdgeType.
static Level directionOfEdgeType(EdgeType);

/// Determines whether it would be pointless to add the given Value to our sets.
static bool canSkipAddingToSets(Value *Val);

static Optional<Function *> parentFunctionOfValue(Value *Val) {
  if (auto *Inst = dyn_cast<Instruction>(Val)) {
    auto *Bb = Inst->getParent();
    return Bb->getParent();
  }

  if (auto *Arg = dyn_cast<Argument>(Val))
    return Arg->getParent();
  return None;
}

static bool getPossibleTargets(CallSite CS,
                               SmallVectorImpl<Function *> &Output) {
  if (auto *Fn = CS.getCalledFunction()) {
    Output.push_back(Fn);
    return true;
  }

  // TODO: If the call is indirect, we might be able to enumerate all potential
  // targets of the call and return them, rather than just failing.
  return false;
}

static Level directionOfEdgeType(EdgeType Weight) {
  switch (Weight) {
  case EdgeType::Reference:
    return Level::Above;
  case EdgeType::Dereference:
    return Level::Below;
  case EdgeType::Assign:
    return Level::Same;
  }
  llvm_unreachable("Incomplete switch coverage");
}

static bool canSkipAddingToSets(Value *Val) {
  // Constants can share instances, which may falsely unify multiple
  // sets, e.g. in
  // store i32* null, i32** %ptr1
  // store i32* null, i32** %ptr2
  // clearly ptr1 and ptr2 should not be unified into the same set, so
  // we should filter out the (potentially shared) instance to
  // i32* null.
  if (isa<Constant>(Val)) {
    // TODO: Because all of these things are constant, we can determine whether
    // the data is *actually* mutable at graph building time. This will probably
    // come for free/cheap with offset awareness.
    bool CanStoreMutableData = isa<GlobalValue>(Val) ||
                               isa<ConstantExpr>(Val) ||
                               isa<ConstantAggregate>(Val);
    return !CanStoreMutableData;
  }

  return false;
}

CFLSteensAAResult::FunctionInfo::FunctionInfo(
    Function &Fn, const SmallVectorImpl<Value *> &RetVals,
    StratifiedSets<Value *> S)
    : Sets(std::move(S)) {
  // Historically, an arbitrary upper-bound of 50 args was selected. We may want
  // to remove this if it doesn't really matter in practice.
  if (Fn.arg_size() > MaxSupportedArgsInSummary)
    return;

  DenseMap<StratifiedIndex, InterfaceValue> InterfaceMap;

  // Our intention here is to record all InterfaceValues that share the same
  // StratifiedIndex in RetParamRelations. For each valid InterfaceValue, we
  // have its StratifiedIndex scanned here and check if the index is presented
  // in InterfaceMap: if it is not, we add the correspondence to the map;
  // otherwise, an aliasing relation is found and we add it to
  // RetParamRelations.

  auto AddToRetParamRelations = [&](unsigned InterfaceIndex,
                                    StratifiedIndex SetIndex) {
    unsigned Level = 0;
    while (true) {
      InterfaceValue CurrValue{InterfaceIndex, Level};

      auto Itr = InterfaceMap.find(SetIndex);
      if (Itr != InterfaceMap.end()) {
        if (CurrValue != Itr->second)
          RetParamRelations.push_back(ExternalRelation{CurrValue, Itr->second});
        break;
      }

      auto &Link = Sets.getLink(SetIndex);
      InterfaceMap.insert(std::make_pair(SetIndex, CurrValue));
      auto ExternalAttrs = getExternallyVisibleAttrs(Link.Attrs);
      if (ExternalAttrs.any())
        RetParamAttributes.push_back(
            ExternalAttribute{CurrValue, ExternalAttrs});

      if (!Link.hasBelow())
        break;

      ++Level;
      SetIndex = Link.Below;
    }
  };

  // Populate RetParamRelations for return values
  for (auto *RetVal : RetVals) {
    assert(RetVal != nullptr);
    assert(RetVal->getType()->isPointerTy());
    auto RetInfo = Sets.find(RetVal);
    if (RetInfo.hasValue())
      AddToRetParamRelations(0, RetInfo->Index);
  }

  // Populate RetParamRelations for parameters
  unsigned I = 0;
  for (auto &Param : Fn.args()) {
    if (Param.getType()->isPointerTy()) {
      auto ParamInfo = Sets.find(&Param);
      if (ParamInfo.hasValue())
        AddToRetParamRelations(I + 1, ParamInfo->Index);
    }
    ++I;
  }
}

// Builds the graph + StratifiedSets for a function.
CFLSteensAAResult::FunctionInfo CFLSteensAAResult::buildSetsFrom(Function *Fn) {
  CFLGraphBuilder GraphBuilder(*this, TLI, *Fn);
  StratifiedSetsBuilder<Value *> SetBuilder;

  auto &Graph = GraphBuilder.getCFLGraph();
  SmallVector<Value *, 16> Worklist;
  for (auto Node : Graph.nodes())
    Worklist.push_back(Node);

  while (!Worklist.empty()) {
    auto *CurValue = Worklist.pop_back_val();
    SetBuilder.add(CurValue);
    if (canSkipAddingToSets(CurValue))
      continue;

    auto Attr = Graph.attrFor(CurValue);
    SetBuilder.noteAttributes(CurValue, Attr);

    for (const auto &Edge : Graph.edgesFor(CurValue)) {
      auto Label = Edge.Type;
      auto *OtherValue = Edge.Other;

      if (canSkipAddingToSets(OtherValue))
        continue;

      bool Added;
      switch (directionOfEdgeType(Label)) {
      case Level::Above:
        Added = SetBuilder.addAbove(CurValue, OtherValue);
        break;
      case Level::Below:
        Added = SetBuilder.addBelow(CurValue, OtherValue);
        break;
      case Level::Same:
        Added = SetBuilder.addWith(CurValue, OtherValue);
        break;
      }

      if (Added)
        Worklist.push_back(OtherValue);
    }
  }

  // Special handling for globals and arguments
  for (auto *External : GraphBuilder.getExternalValues()) {
    SetBuilder.add(External);
    auto Attr = getGlobalOrArgAttrFromValue(*External);
    if (Attr.any()) {
      SetBuilder.noteAttributes(External, Attr);
      if (isa<GlobalValue>(External))
        SetBuilder.addAttributesBelow(External, 1, getAttrUnknown());
      else
        SetBuilder.addAttributesBelow(External, 1, getAttrCaller());
    }
  }

  // Special handling for interprocedural aliases
  for (auto &Edge : GraphBuilder.getInstantiatedRelations()) {
    auto FromVal = Edge.From.Val;
    auto ToVal = Edge.To.Val;
    SetBuilder.add(FromVal);
    SetBuilder.add(ToVal);
    SetBuilder.addBelowWith(FromVal, Edge.From.DerefLevel, ToVal,
                            Edge.To.DerefLevel);
  }

  // Special handling for interprocedural attributes
  for (auto &IPAttr : GraphBuilder.getInstantiatedAttrs()) {
    auto Val = IPAttr.IValue.Val;
    SetBuilder.add(Val);
    SetBuilder.addAttributesBelow(Val, IPAttr.IValue.DerefLevel, IPAttr.Attr);
  }

  // Special handling for opaque external functions
  for (auto *Escape : GraphBuilder.getEscapedValues()) {
    SetBuilder.add(Escape);
    SetBuilder.noteAttributes(Escape, getAttrEscaped());
    SetBuilder.addAttributesBelow(Escape, 1, getAttrUnknown());
  }

  return FunctionInfo(*Fn, GraphBuilder.getReturnValues(), SetBuilder.build());
}

void CFLSteensAAResult::scan(Function *Fn) {
  auto InsertPair = Cache.insert(std::make_pair(Fn, Optional<FunctionInfo>()));
  (void)InsertPair;
  assert(InsertPair.second &&
         "Trying to scan a function that has already been cached");

  // Note that we can't do Cache[Fn] = buildSetsFrom(Fn) here: the function call
  // may get evaluated after operator[], potentially triggering a DenseMap
  // resize and invalidating the reference returned by operator[]
  auto FunInfo = buildSetsFrom(Fn);
  Cache[Fn] = std::move(FunInfo);

  Handles.push_front(FunctionHandle(Fn, this));
}

void CFLSteensAAResult::evict(Function *Fn) { Cache.erase(Fn); }

/// Ensures that the given function is available in the cache, and returns the
/// entry.
const Optional<CFLSteensAAResult::FunctionInfo> &
CFLSteensAAResult::ensureCached(Function *Fn) {
  auto Iter = Cache.find(Fn);
  if (Iter == Cache.end()) {
    scan(Fn);
    Iter = Cache.find(Fn);
    assert(Iter != Cache.end());
    assert(Iter->second.hasValue());
  }
  return Iter->second;
}

AliasResult CFLSteensAAResult::query(const MemoryLocation &LocA,
                                     const MemoryLocation &LocB) {
  auto *ValA = const_cast<Value *>(LocA.Ptr);
  auto *ValB = const_cast<Value *>(LocB.Ptr);

  if (!ValA->getType()->isPointerTy() || !ValB->getType()->isPointerTy())
    return NoAlias;

  Function *Fn = nullptr;
  auto MaybeFnA = parentFunctionOfValue(ValA);
  auto MaybeFnB = parentFunctionOfValue(ValB);
  if (!MaybeFnA.hasValue() && !MaybeFnB.hasValue()) {
    // The only times this is known to happen are when globals + InlineAsm are
    // involved
    DEBUG(dbgs()
          << "CFLSteensAA: could not extract parent function information.\n");
    return MayAlias;
  }

  if (MaybeFnA.hasValue()) {
    Fn = *MaybeFnA;
    assert((!MaybeFnB.hasValue() || *MaybeFnB == *MaybeFnA) &&
           "Interprocedural queries not supported");
  } else {
    Fn = *MaybeFnB;
  }

  assert(Fn != nullptr);
  auto &MaybeInfo = ensureCached(Fn);
  assert(MaybeInfo.hasValue());

  auto &Sets = MaybeInfo->getStratifiedSets();
  auto MaybeA = Sets.find(ValA);
  if (!MaybeA.hasValue())
    return MayAlias;

  auto MaybeB = Sets.find(ValB);
  if (!MaybeB.hasValue())
    return MayAlias;

  auto SetA = *MaybeA;
  auto SetB = *MaybeB;
  auto AttrsA = Sets.getLink(SetA.Index).Attrs;
  auto AttrsB = Sets.getLink(SetB.Index).Attrs;

  // If both values are local (meaning the corresponding set has attribute
  // AttrNone or AttrEscaped), then we know that CFLSteensAA fully models them:
  // they may-alias each other if and only if they are in the same set.
  // If at least one value is non-local (meaning it either is global/argument or
  // it comes from unknown sources like integer cast), the situation becomes a
  // bit more interesting. We follow three general rules described below:
  // - Non-local values may alias each other
  // - AttrNone values do not alias any non-local values
  // - AttrEscaped do not alias globals/arguments, but they may alias
  // AttrUnknown values
  if (SetA.Index == SetB.Index)
    return MayAlias;
  if (AttrsA.none() || AttrsB.none())
    return NoAlias;
  if (hasUnknownOrCallerAttr(AttrsA) || hasUnknownOrCallerAttr(AttrsB))
    return MayAlias;
  if (isGlobalOrArgAttr(AttrsA) && isGlobalOrArgAttr(AttrsB))
    return MayAlias;
  return NoAlias;
}

ModRefInfo CFLSteensAAResult::getArgModRefInfo(ImmutableCallSite CS,
                                               unsigned ArgIdx) {
  if (auto CalledFunc = CS.getCalledFunction()) {
    auto &MaybeInfo = ensureCached(const_cast<Function *>(CalledFunc));
    if (!MaybeInfo.hasValue())
      return MRI_ModRef;
    auto &RetParamAttributes = MaybeInfo->getRetParamAttributes();
    auto &RetParamRelations = MaybeInfo->getRetParamRelations();

    bool ArgAttributeIsWritten =
        std::any_of(RetParamAttributes.begin(), RetParamAttributes.end(),
                    [ArgIdx](const ExternalAttribute &ExtAttr) {
                      return ExtAttr.IValue.Index == ArgIdx + 1;
                    });
    bool ArgIsAccessed =
        std::any_of(RetParamRelations.begin(), RetParamRelations.end(),
                    [ArgIdx](const ExternalRelation &ExtRelation) {
                      return ExtRelation.To.Index == ArgIdx + 1 ||
                             ExtRelation.From.Index == ArgIdx + 1;
                    });

    return (!ArgIsAccessed && !ArgAttributeIsWritten) ? MRI_NoModRef
                                                      : MRI_ModRef;
  }

  return MRI_ModRef;
}

FunctionModRefBehavior
CFLSteensAAResult::getModRefBehavior(ImmutableCallSite CS) {
  // If we know the callee, try analyzing it
  if (auto CalledFunc = CS.getCalledFunction())
    return getModRefBehavior(CalledFunc);

  // Otherwise, be conservative
  return FMRB_UnknownModRefBehavior;
}

FunctionModRefBehavior CFLSteensAAResult::getModRefBehavior(const Function *F) {
  assert(F != nullptr);

  // TODO: Remove the const_cast
  auto &MaybeInfo = ensureCached(const_cast<Function *>(F));
  if (!MaybeInfo.hasValue())
    return FMRB_UnknownModRefBehavior;
  auto &RetParamAttributes = MaybeInfo->getRetParamAttributes();
  auto &RetParamRelations = MaybeInfo->getRetParamRelations();

  // First, if any argument is marked Escpaed, Unknown or Global, anything may
  // happen to them and thus we can't draw any conclusion.
  if (!RetParamAttributes.empty())
    return FMRB_UnknownModRefBehavior;

  // Currently we don't (and can't) distinguish reads from writes in
  // RetParamRelations. All we can say is whether there may be memory access or
  // not.
  if (RetParamRelations.empty())
    return FMRB_DoesNotAccessMemory;

  // Check if something beyond argmem gets touched.
  bool AccessArgMemoryOnly =
      std::all_of(RetParamRelations.begin(), RetParamRelations.end(),
                  [](const ExternalRelation &ExtRelation) {
                    // Both DerefLevels has to be 0, since we don't know which
                    // one is a read and which is a write.
                    return ExtRelation.From.DerefLevel == 0 &&
                           ExtRelation.To.DerefLevel == 0;
                  });
  return AccessArgMemoryOnly ? FMRB_OnlyAccessesArgumentPointees
                             : FMRB_UnknownModRefBehavior;
}

char CFLSteensAA::PassID;

CFLSteensAAResult CFLSteensAA::run(Function &F, AnalysisManager<Function> &AM) {
  return CFLSteensAAResult(AM.getResult<TargetLibraryAnalysis>(F));
}

char CFLSteensAAWrapperPass::ID = 0;
INITIALIZE_PASS(CFLSteensAAWrapperPass, "cfl-steens-aa",
                "Unification-Based CFL Alias Analysis", false, true)

ImmutablePass *llvm::createCFLSteensAAWrapperPass() {
  return new CFLSteensAAWrapperPass();
}

CFLSteensAAWrapperPass::CFLSteensAAWrapperPass() : ImmutablePass(ID) {
  initializeCFLSteensAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void CFLSteensAAWrapperPass::initializePass() {
  auto &TLIWP = getAnalysis<TargetLibraryInfoWrapperPass>();
  Result.reset(new CFLSteensAAResult(TLIWP.getTLI()));
}

void CFLSteensAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}
