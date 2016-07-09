//======- CFLGraph.h - Abstract stratified sets implementation. --------======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines CFLGraph, an auxiliary data structure used by CFL-based
/// alias analysis.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFLGRAPH_H
#define LLVM_ANALYSIS_CFLGRAPH_H

#include "AliasAnalysisSummary.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
namespace cflaa {
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

/// \brief The Program Expression Graph (PEG) of CFL analysis
/// CFLGraph is auxiliary data structure used by CFL-based alias analysis to
/// describe flow-insensitive pointer-related behaviors. Given an LLVM function,
/// the main purpose of this graph is to abstract away unrelated facts and
/// translate the rest into a form that can be easily digested by CFL analyses.
class CFLGraph {
  typedef Value *Node;

  struct Edge {
    EdgeType Type;
    Node Other;
  };

  typedef std::vector<Edge> EdgeList;

  struct NodeInfo {
    EdgeList Edges;
    AliasAttrs Attr;
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
    return NodeImpls
        .insert(std::make_pair(N, NodeInfo{EdgeList(), getAttrNone()}))
        .second;
  }

  void addAttr(Node N, AliasAttrs Attr) {
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

  AliasAttrs attrFor(Node N) const {
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

///\brief A builder class used to create CFLGraph instance from a given function
/// The CFL-AA that uses this builder must provide its own type as a template
/// argument. This is necessary for interprocedural processing: CFLGraphBuilder
/// needs a way of obtaining the summary of other functions when callinsts are
/// encountered.
/// As a result, we expect the said CFL-AA to expose a getAliasSummary() public
/// member function that takes a Function& and returns the corresponding summary
/// as a const AliasSummary*.
template <typename CFLAA> class CFLGraphBuilder {
  // Input of the builder
  CFLAA &Analysis;
  const TargetLibraryInfo &TLI;

  // Output of the builder
  CFLGraph Graph;
  SmallVector<Value *, 4> ReturnedValues;

  // Auxiliary structures used by the builder
  SmallVector<InstantiatedRelation, 8> InstantiatedRelations;
  SmallVector<InstantiatedAttr, 8> InstantiatedAttrs;

  // Helper class
  /// Gets the edges our graph should have, based on an Instruction*
  class GetEdgesVisitor : public InstVisitor<GetEdgesVisitor, void> {
    CFLAA &AA;
    const TargetLibraryInfo &TLI;

    CFLGraph &Graph;
    SmallVectorImpl<Value *> &ReturnValues;
    SmallVectorImpl<InstantiatedRelation> &InstantiatedRelations;
    SmallVectorImpl<InstantiatedAttr> &InstantiatedAttrs;

    static bool hasUsefulEdges(ConstantExpr *CE) {
      // ConstantExpr doesn't have terminators, invokes, or fences, so only
      // needs
      // to check for compares.
      return CE->getOpcode() != Instruction::ICmp &&
             CE->getOpcode() != Instruction::FCmp;
    }

    // Returns possible functions called by CS into the given SmallVectorImpl.
    // Returns true if targets found, false otherwise.
    static bool getPossibleTargets(CallSite CS,
                                   SmallVectorImpl<Function *> &Output) {
      if (auto *Fn = CS.getCalledFunction()) {
        Output.push_back(Fn);
        return true;
      }

      // TODO: If the call is indirect, we might be able to enumerate all
      // potential
      // targets of the call and return them, rather than just failing.
      return false;
    }

    void addNode(Value *Val) {
      assert(Val != nullptr);
      if (!Graph.addNode(Val))
        return;

      if (isa<GlobalValue>(Val)) {
        Graph.addAttr(Val, getGlobalOrArgAttrFromValue(*Val));
        // Currently we do not attempt to be smart on globals
        InstantiatedAttrs.push_back(
            InstantiatedAttr{InstantiatedValue{Val, 1}, getAttrUnknown()});
      } else if (auto CExpr = dyn_cast<ConstantExpr>(Val))
        if (hasUsefulEdges(CExpr))
          visitConstantExpr(CExpr);
    }

    void addNodeWithAttr(Value *Val, AliasAttrs Attr) {
      addNode(Val);
      Graph.addAttr(Val, Attr);
    }

    void addEdge(Value *From, Value *To, EdgeType Type) {
      assert(From != nullptr && To != nullptr);
      if (!From->getType()->isPointerTy() || !To->getType()->isPointerTy())
        return;
      addNode(From);
      if (To != From)
        addNode(To);
      Graph.addEdge(From, To, Type);
    }

  public:
    GetEdgesVisitor(CFLGraphBuilder &Builder)
        : AA(Builder.Analysis), TLI(Builder.TLI), Graph(Builder.Graph),
          ReturnValues(Builder.ReturnedValues),
          InstantiatedRelations(Builder.InstantiatedRelations),
          InstantiatedAttrs(Builder.InstantiatedAttrs) {}

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
      // We can't fully model va_arg here. For *Ptr = Inst.getOperand(0), it
      // does
      // two things:
      //  1. Loads a value from *((T*)*Ptr).
      //  2. Increments (stores to) *Ptr by some target-specific amount.
      // For now, we'll handle this like a landingpad instruction (by placing
      // the
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
        if (!AA.getAliasSummary(*Fn))
          return false;
      }

      for (auto *Fn : Fns) {
        auto Summary = AA.getAliasSummary(*Fn);
        assert(Summary != nullptr);

        auto &RetParamRelations = Summary->RetParamRelations;
        for (auto &Relation : RetParamRelations) {
          auto IRelation = instantiateExternalRelation(Relation, CS);
          if (IRelation.hasValue())
            InstantiatedRelations.push_back(*IRelation);
        }

        auto &RetParamAttributes = Summary->RetParamAttributes;
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

      // Check if Inst is a call to a library function that
      // allocates/deallocates
      // on the heap. Those kinds of functions do not introduce any aliases.
      // TODO: address other common library functions such as realloc(),
      // strdup(),
      // etc.
      if (isMallocLikeFn(Inst, &TLI) || isCallocLikeFn(Inst, &TLI) ||
          isFreeCall(Inst, &TLI))
        return;

      // TODO: Add support for noalias args/all the other fun function
      // attributes
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
          if (V->getType()->isPointerTy()) {
            // The argument itself escapes.
            addNodeWithAttr(V, getAttrEscaped());
            // The fate of argument memory is unknown. Note that since
            // AliasAttrs
            // is transitive with respect to dereference, we only need to
            // specify
            // it for the first-level memory.
            InstantiatedAttrs.push_back(
                InstantiatedAttr{InstantiatedValue{V, 1}, getAttrUnknown()});
          }
        }

      if (Inst->getType()->isPointerTy()) {
        auto *Fn = CS.getCalledFunction();
        if (Fn == nullptr || !Fn->doesNotAlias(0))
          // No need to call addNodeWithAttr() since we've added Inst at the
          // beginning of this function and we know it is not a global.
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
    this->visit##OPCODE(*(CLASS *)CE);                                         \
    break;
#include "llvm/IR/Instruction.def"
      }
    }
  };

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
      Graph.addAttr(&Arg, getGlobalOrArgAttrFromValue(Arg));
      // Pointees of a formal parameter is known to the caller
      InstantiatedAttrs.push_back(
          InstantiatedAttr{InstantiatedValue{&Arg, 1}, getAttrCaller()});
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

    GetEdgesVisitor(*this).visit(Inst);
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
  CFLGraphBuilder(CFLAA &Analysis, const TargetLibraryInfo &TLI, Function &Fn)
      : Analysis(Analysis), TLI(TLI) {
    buildGraphFrom(Fn);
  }

  const CFLGraph &getCFLGraph() const { return Graph; }
  const SmallVector<Value *, 4> &getReturnValues() const {
    return ReturnedValues;
  }
  const SmallVector<InstantiatedRelation, 8> &getInstantiatedRelations() const {
    return InstantiatedRelations;
  }
  const SmallVector<InstantiatedAttr, 8> &getInstantiatedAttrs() const {
    return InstantiatedAttrs;
  }
};
}
}

#endif
