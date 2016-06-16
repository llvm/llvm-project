//===- CFLAliasAnalysis.cpp - CFL-Based Alias Analysis Implementation ------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a CFL-based context-insensitive alias analysis
// algorithm. It does not depend on types. The algorithm is a mixture of the one
// described in "Demand-driven alias analysis for C" by Xin Zheng and Radu
// Rugina, and "Fast algorithms for Dyck-CFL-reachability with applications to
// Alias Analysis" by Zhang Q, Lyu M R, Yuan H, and Su Z. -- to summarize the
// papers, we build a graph of the uses of a variable, where each node is a
// memory location, and each edge is an action that happened on that memory
// location.  The "actions" can be one of Dereference, Reference, or Assign.
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
// CFLAA is interprocedural. This is *technically* A Bad Thing, because
// FunctionPasses are only allowed to inspect the Function that they're being
// run on. Realistically, this likely isn't a problem until we allow
// FunctionPasses to run concurrently.

#include "llvm/Analysis/CFLAliasAnalysis.h"
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

#define DEBUG_TYPE "cfl-aa"

CFLAAResult::CFLAAResult(const TargetLibraryInfo &TLI)
    : AAResultBase(), TLI(TLI) {}
CFLAAResult::CFLAAResult(CFLAAResult &&Arg)
    : AAResultBase(std::move(Arg)), TLI(Arg.TLI) {}
CFLAAResult::~CFLAAResult() {}

/// Information we have about a function and would like to keep around.
struct CFLAAResult::FunctionInfo {
  StratifiedSets<Value *> Sets;
  // Lots of functions have < 4 returns. Adjust as necessary.
  SmallVector<Value *, 4> ReturnedValues;

  FunctionInfo(StratifiedSets<Value *> &&S, SmallVector<Value *, 4> &&RV)
      : Sets(std::move(S)), ReturnedValues(std::move(RV)) {}
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
/// StratifiedInfo Attribute things.
typedef unsigned StratifiedAttr;
LLVM_CONSTEXPR unsigned MaxStratifiedAttrIndex = NumStratifiedAttrs;
LLVM_CONSTEXPR unsigned AttrEscapedIndex = 0;
LLVM_CONSTEXPR unsigned AttrUnknownIndex = 1;
LLVM_CONSTEXPR unsigned AttrGlobalIndex = 2;
LLVM_CONSTEXPR unsigned AttrFirstArgIndex = 3;
LLVM_CONSTEXPR unsigned AttrLastArgIndex = MaxStratifiedAttrIndex;
LLVM_CONSTEXPR unsigned AttrMaxNumArgs = AttrLastArgIndex - AttrFirstArgIndex;

LLVM_CONSTEXPR StratifiedAttr AttrNone = 0;
LLVM_CONSTEXPR StratifiedAttr AttrEscaped = 1 << AttrEscapedIndex;
LLVM_CONSTEXPR StratifiedAttr AttrUnknown = 1 << AttrUnknownIndex;
LLVM_CONSTEXPR StratifiedAttr AttrGlobal = 1 << AttrGlobalIndex;

/// StratifiedSets call for knowledge of "direction", so this is how we
/// represent that locally.
enum class Level { Same, Above, Below };

/// Edges can be one of four "weights" -- each weight must have an inverse
/// weight (Assign has Assign; Reference has Dereference).
enum class EdgeType {
  /// The weight assigned when assigning from or to a value. For example, in:
  /// %b = getelementptr %a, 0
  /// ...The relationships are %b assign %a, and %a assign %b. This used to be
  /// two edges, but having a distinction bought us nothing.
  Assign,

  /// The edge used when we have an edge going from some handle to a Value.
  /// Examples of this include:
  /// %b = load %a              (%b Dereference %a)
  /// %b = extractelement %a, 0 (%a Dereference %b)
  Dereference,

  /// The edge used when our edge goes from a value to a handle that may have
  /// contained it at some point. Examples:
  /// %b = load %a              (%a Reference %b)
  /// %b = extractelement %a, 0 (%b Reference %a)
  Reference
};

/// The Program Expression Graph (PEG) of CFL analysis
class CFLGraph {
  typedef Value *Node;

  struct Edge {
    EdgeType Type;
    Node Other;
  };

  typedef std::vector<Edge> EdgeList;

  struct NodeInfo {
    EdgeList Edges;
    StratifiedAttrs Attr;
  };

  typedef DenseMap<Node, NodeInfo> NodeMap;
  NodeMap NodeImpls;

  // Gets the inverse of a given EdgeType.
  static EdgeType flipWeight(EdgeType Initial) {
    switch (Initial) {
    case EdgeType::Assign:
      return EdgeType::Assign;
    case EdgeType::Dereference:
      return EdgeType::Reference;
    case EdgeType::Reference:
      return EdgeType::Dereference;
    }
    llvm_unreachable("Incomplete coverage of EdgeType enum");
  }

  const NodeInfo *getNode(Node N) const {
    auto Itr = NodeImpls.find(N);
    if (Itr == NodeImpls.end())
      return nullptr;
    return &Itr->second;
  }
  NodeInfo *getNode(Node N) {
    auto Itr = NodeImpls.find(N);
    if (Itr == NodeImpls.end())
      return nullptr;
    return &Itr->second;
  }

  static Node nodeDeref(const NodeMap::value_type &P) { return P.first; }
  typedef std::pointer_to_unary_function<const NodeMap::value_type &, Node>
      NodeDerefFun;

public:
  typedef EdgeList::const_iterator const_edge_iterator;
  typedef mapped_iterator<NodeMap::const_iterator, NodeDerefFun>
      const_node_iterator;

  bool addNode(Node N) {
    return NodeImpls.insert(std::make_pair(N, NodeInfo{EdgeList(), AttrNone}))
        .second;
  }

  void addAttr(Node N, StratifiedAttrs Attr) {
    auto *Info = getNode(N);
    assert(Info != nullptr);
    Info->Attr |= Attr;
  }

  void addEdge(Node From, Node To, EdgeType Type) {
    auto *FromInfo = getNode(From);
    assert(FromInfo != nullptr);
    auto *ToInfo = getNode(To);
    assert(ToInfo != nullptr);

    FromInfo->Edges.push_back(Edge{Type, To});
    ToInfo->Edges.push_back(Edge{flipWeight(Type), From});
  }

  StratifiedAttrs attrFor(Node N) const {
    auto *Info = getNode(N);
    assert(Info != nullptr);
    return Info->Attr;
  }

  iterator_range<const_edge_iterator> edgesFor(Node N) const {
    auto *Info = getNode(N);
    assert(Info != nullptr);
    auto &Edges = Info->Edges;
    return make_range(Edges.begin(), Edges.end());
  }

  iterator_range<const_node_iterator> nodes() const {
    return make_range<const_node_iterator>(
        map_iterator(NodeImpls.begin(), NodeDerefFun(nodeDeref)),
        map_iterator(NodeImpls.end(), NodeDerefFun(nodeDeref)));
  }

  bool empty() const { return NodeImpls.empty(); }
  std::size_t size() const { return NodeImpls.size(); }
};

/// Gets the edges our graph should have, based on an Instruction*
class GetEdgesVisitor : public InstVisitor<GetEdgesVisitor, void> {
  CFLAAResult &AA;
  const TargetLibraryInfo &TLI;

  CFLGraph &Graph;
  SmallPtrSetImpl<Value *> &Externals;
  SmallPtrSetImpl<Value *> &Escapes;

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

  void addNodeWithAttr(Value *Val, StratifiedAttrs Attr) {
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
  GetEdgesVisitor(CFLAAResult &AA, const TargetLibraryInfo &TLI,
                  CFLGraph &Graph, SmallPtrSetImpl<Value *> &Externals,
                  SmallPtrSetImpl<Value *> &Escapes)
      : AA(AA), TLI(TLI), Graph(Graph), Externals(Externals), Escapes(Escapes) {
  }

  void visitInstruction(Instruction &) {
    llvm_unreachable("Unsupported instruction encountered");
  }

  void visitPtrToIntInst(PtrToIntInst &Inst) {
    auto *Ptr = Inst.getOperand(0);
    addNodeWithAttr(Ptr, AttrEscaped);
  }

  void visitIntToPtrInst(IntToPtrInst &Inst) {
    auto *Ptr = &Inst;
    addNodeWithAttr(Ptr, AttrUnknown);
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
    addNodeWithAttr(&Inst, AttrUnknown);
  }

  static bool isFunctionExternal(Function *Fn) {
    return Fn->isDeclaration() || !Fn->hasLocalLinkage();
  }

  /// Gets whether the sets at Index1 above, below, or equal to the sets at
  /// Index2. Returns None if they are not in the same set chain.
  static Optional<Level> getIndexRelation(const StratifiedSets<Value *> &Sets,
                                          StratifiedIndex Index1,
                                          StratifiedIndex Index2) {
    if (Index1 == Index2)
      return Level::Same;

    const auto *Current = &Sets.getLink(Index1);
    while (Current->hasBelow()) {
      if (Current->Below == Index2)
        return Level::Below;
      Current = &Sets.getLink(Current->Below);
    }

    Current = &Sets.getLink(Index1);
    while (Current->hasAbove()) {
      if (Current->Above == Index2)
        return Level::Above;
      Current = &Sets.getLink(Current->Above);
    }

    return None;
  }

  // Encodes the notion of a "use"
  struct Edge {
    // Which value the edge is coming from
    Value *From;

    // Which value the edge is pointing to
    Value *To;

    // Edge weight
    EdgeType Weight;

    // Whether we aliased any external values along the way that may be
    // invisible to the analysis
    StratifiedAttrs FromAttrs, ToAttrs;
  };

  bool
  tryInterproceduralAnalysis(const SmallVectorImpl<Function *> &Fns,
                             Value *FuncValue,
                             const iterator_range<User::op_iterator> &Args) {
    const unsigned ExpectedMaxArgs = 8;
    const unsigned MaxSupportedArgs = 50;
    assert(Fns.size() > 0);

    // This algorithm is n^2, so an arbitrary upper-bound of 50 args was
    // selected, so it doesn't take too long in insane cases.
    if (std::distance(Args.begin(), Args.end()) > (int)MaxSupportedArgs)
      return false;

    // Exit early if we'll fail anyway
    for (auto *Fn : Fns) {
      if (isFunctionExternal(Fn) || Fn->isVarArg())
        return false;
      auto &MaybeInfo = AA.ensureCached(Fn);
      if (!MaybeInfo.hasValue())
        return false;
    }

    SmallVector<Edge, 8> Output;
    SmallVector<Value *, ExpectedMaxArgs> Arguments(Args.begin(), Args.end());
    SmallVector<StratifiedInfo, ExpectedMaxArgs> Parameters;
    for (auto *Fn : Fns) {
      auto &Info = *AA.ensureCached(Fn);
      auto &Sets = Info.Sets;
      auto &RetVals = Info.ReturnedValues;

      Parameters.clear();
      for (auto &Param : Fn->args()) {
        auto MaybeInfo = Sets.find(&Param);
        // Did a new parameter somehow get added to the function/slip by?
        if (!MaybeInfo.hasValue())
          return false;
        Parameters.push_back(*MaybeInfo);
      }

      // Adding an edge from argument -> return value for each parameter that
      // may alias the return value
      for (unsigned I = 0, E = Parameters.size(); I != E; ++I) {
        auto &ParamInfo = Parameters[I];
        auto &ArgVal = Arguments[I];
        bool AddEdge = false;
        StratifiedAttrs RetAttrs, ParamAttrs;
        for (unsigned X = 0, XE = RetVals.size(); X != XE; ++X) {
          auto MaybeInfo = Sets.find(RetVals[X]);
          if (!MaybeInfo.hasValue())
            return false;

          auto &RetInfo = *MaybeInfo;
          RetAttrs |= Sets.getLink(RetInfo.Index).Attrs;
          ParamAttrs |= Sets.getLink(ParamInfo.Index).Attrs;
          auto MaybeRelation =
              getIndexRelation(Sets, ParamInfo.Index, RetInfo.Index);
          if (MaybeRelation.hasValue())
            AddEdge = true;
        }
        if (AddEdge)
          Output.push_back(
              Edge{FuncValue, ArgVal, EdgeType::Assign, RetAttrs, ParamAttrs});
      }

      if (Parameters.size() != Arguments.size())
        return false;

      /// Adding edges between arguments for arguments that may end up aliasing
      /// each other. This is necessary for functions such as
      /// void foo(int** a, int** b) { *a = *b; }
      /// (Technically, the proper sets for this would be those below
      /// Arguments[I] and Arguments[X], but our algorithm will produce
      /// extremely similar, and equally correct, results either way)
      for (unsigned I = 0, E = Arguments.size(); I != E; ++I) {
        auto &MainVal = Arguments[I];
        auto &MainInfo = Parameters[I];
        auto &MainAttrs = Sets.getLink(MainInfo.Index).Attrs;
        for (unsigned X = I + 1; X != E; ++X) {
          auto &SubInfo = Parameters[X];
          auto &SubVal = Arguments[X];
          auto &SubAttrs = Sets.getLink(SubInfo.Index).Attrs;
          auto MaybeRelation =
              getIndexRelation(Sets, MainInfo.Index, SubInfo.Index);

          if (!MaybeRelation.hasValue())
            continue;

          Output.push_back(
              Edge{MainVal, SubVal, EdgeType::Assign, MainAttrs, SubAttrs});
        }
      }
    }

    // Commit all edges in Output to CFLGraph
    for (const auto &Edge : Output) {
      addEdge(Edge.From, Edge.To, Edge.Weight);
      Graph.addAttr(Edge.From, Edge.FromAttrs);
      Graph.addAttr(Edge.To, Edge.ToAttrs);
    }

    return true;
  }

  void visitCallSite(CallSite CS) {
    auto Inst = CS.getInstruction();

    // Make sure all arguments and return value are added to the graph first
    for (Value *V : CS.args())
      addNode(V);
    if (!Inst->getType()->isVoidTy())
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
      if (tryInterproceduralAnalysis(Targets, Inst, CS.args()))
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

    if (!Inst->getType()->isVoidTy()) {
      auto *Fn = CS.getCalledFunction();
      if (Fn == nullptr || !Fn->doesNotAlias(0))
        Graph.addAttr(Inst, AttrUnknown);
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
    addNodeWithAttr(&Inst, AttrUnknown);
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
  CFLAAResult &Analysis;
  const TargetLibraryInfo &TLI;

  // Output of the builder
  CFLGraph Graph;
  SmallVector<Value *, 4> ReturnedValues;

  // Auxiliary structures used by the builder
  SmallPtrSet<Value *, 8> ExternalValues;
  SmallPtrSet<Value *, 8> EscapedValues;

  // Helper functions

  // Determines whether or not we an instruction is useless to us (e.g.
  // FenceInst)
  static bool hasUsefulEdges(Instruction *Inst) {
    bool IsNonInvokeTerminator =
        isa<TerminatorInst>(Inst) && !isa<InvokeInst>(Inst);
    return !isa<CmpInst>(Inst) && !isa<FenceInst>(Inst) &&
           !IsNonInvokeTerminator;
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
    // We don't want the edges of most "return" instructions, but we *do* want
    // to know what can be returned.
    if (isa<ReturnInst>(&Inst))
      ReturnedValues.push_back(&Inst);

    if (!hasUsefulEdges(&Inst))
      return;

    GetEdgesVisitor(Analysis, TLI, Graph, ExternalValues, EscapedValues)
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
  CFLGraphBuilder(CFLAAResult &Analysis, const TargetLibraryInfo &TLI,
                  Function &Fn)
      : Analysis(Analysis), TLI(TLI) {
    buildGraphFrom(Fn);
  }

  const CFLGraph &getCFLGraph() { return Graph; }
  SmallVector<Value *, 4> takeReturnValues() {
    return std::move(ReturnedValues);
  }
  const SmallPtrSet<Value *, 8> &getExternalValues() { return ExternalValues; }
  const SmallPtrSet<Value *, 8> &getEscapedValues() { return EscapedValues; }
};
}

//===----------------------------------------------------------------------===//
// Function declarations that require types defined in the namespace above
//===----------------------------------------------------------------------===//

/// Given a StratifiedAttrs, returns true if it marks the corresponding values
/// as globals or arguments
static bool isGlobalOrArgAttr(StratifiedAttrs Attr);

/// Given a StratifiedAttrs, returns true if the corresponding values come from
/// an unknown source (such as opaque memory or an integer cast)
static bool isUnknownAttr(StratifiedAttrs Attr);

/// Given an argument number, returns the appropriate StratifiedAttr to set.
static StratifiedAttr argNumberToAttr(unsigned ArgNum);

/// Given a Value, potentially return which StratifiedAttr it maps to.
static Optional<StratifiedAttr> valueToAttr(Value *Val);

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

static bool isGlobalOrArgAttr(StratifiedAttrs Attr) {
  return Attr.reset(AttrEscapedIndex).reset(AttrUnknownIndex).any();
}

static bool isUnknownAttr(StratifiedAttrs Attr) {
  return Attr.test(AttrUnknownIndex);
}

static Optional<StratifiedAttr> valueToAttr(Value *Val) {
  if (isa<GlobalValue>(Val))
    return AttrGlobal;

  if (auto *Arg = dyn_cast<Argument>(Val))
    // Only pointer arguments should have the argument attribute,
    // because things can't escape through scalars without us seeing a
    // cast, and thus, interaction with them doesn't matter.
    if (!Arg->hasNoAliasAttr() && Arg->getType()->isPointerTy())
      return argNumberToAttr(Arg->getArgNo());
  return None;
}

static StratifiedAttr argNumberToAttr(unsigned ArgNum) {
  if (ArgNum >= AttrMaxNumArgs)
    return AttrUnknown;
  return 1 << (ArgNum + AttrFirstArgIndex);
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

// Builds the graph + StratifiedSets for a function.
CFLAAResult::FunctionInfo CFLAAResult::buildSetsFrom(Function *Fn) {
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
    auto Attr = valueToAttr(External);
    if (Attr.hasValue()) {
      SetBuilder.noteAttributes(External, *Attr);
      SetBuilder.addAttributesBelow(External, AttrUnknown);
    }
  }

  for (auto *Escape : GraphBuilder.getEscapedValues()) {
    SetBuilder.add(Escape);
    SetBuilder.noteAttributes(Escape, AttrEscaped);
    SetBuilder.addAttributesBelow(Escape, AttrUnknown);
  }

  return FunctionInfo(SetBuilder.build(), GraphBuilder.takeReturnValues());
}

void CFLAAResult::scan(Function *Fn) {
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

void CFLAAResult::evict(Function *Fn) { Cache.erase(Fn); }

/// Ensures that the given function is available in the cache, and returns the
/// entry.
const Optional<CFLAAResult::FunctionInfo> &
CFLAAResult::ensureCached(Function *Fn) {
  auto Iter = Cache.find(Fn);
  if (Iter == Cache.end()) {
    scan(Fn);
    Iter = Cache.find(Fn);
    assert(Iter != Cache.end());
    assert(Iter->second.hasValue());
  }
  return Iter->second;
}

AliasResult CFLAAResult::query(const MemoryLocation &LocA,
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
    DEBUG(dbgs() << "CFLAA: could not extract parent function information.\n");
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

  auto &Sets = MaybeInfo->Sets;
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
  // AttrNone or AttrEscaped), then we know that CFLAA fully models them: they
  // may-alias each other if and only if they are in the same set
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
  if (isUnknownAttr(AttrsA) || isUnknownAttr(AttrsB))
    return MayAlias;
  if (isGlobalOrArgAttr(AttrsA) && isGlobalOrArgAttr(AttrsB))
    return MayAlias;
  return NoAlias;
}

char CFLAA::PassID;

CFLAAResult CFLAA::run(Function &F, AnalysisManager<Function> &AM) {
  return CFLAAResult(AM.getResult<TargetLibraryAnalysis>(F));
}

char CFLAAWrapperPass::ID = 0;
INITIALIZE_PASS(CFLAAWrapperPass, "cfl-aa", "CFL-Based Alias Analysis", false,
                true)

ImmutablePass *llvm::createCFLAAWrapperPass() { return new CFLAAWrapperPass(); }

CFLAAWrapperPass::CFLAAWrapperPass() : ImmutablePass(ID) {
  initializeCFLAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void CFLAAWrapperPass::initializePass() {
  auto &TLIWP = getAnalysis<TargetLibraryInfoWrapperPass>();
  Result.reset(new CFLAAResult(TLIWP.getTLI()));
}

void CFLAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}
