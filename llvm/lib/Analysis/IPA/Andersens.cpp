//===- Andersens.cpp - Andersen's Interprocedural Alias Analysis ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an implementation of Andersen's interprocedural alias
// analysis
//
// In pointer analysis terms, this is a subset-based, flow-insensitive,
// field-sensitive, and context-insensitive algorithm pointer algorithm.
//
// This algorithm is implemented as three stages:
//   1. Object identification.
//   2. Inclusion constraint identification.
//   3. Offline constraint graph optimization
//   4. Inclusion constraint solving.
//
// The object identification stage identifies all of the memory objects in the
// program, which includes globals, heap allocated objects, and stack allocated
// objects.
//
// The inclusion constraint identification stage finds all inclusion constraints
// in the program by scanning the program, looking for pointer assignments and
// other statements that effect the points-to graph.  For a statement like "A =
// B", this statement is processed to indicate that A can point to anything that
// B can point to.  Constraints can handle copies, loads, and stores, and
// address taking.
//
// The offline constraint graph optimization portion includes offline variable
// substitution algorithms intended to compute pointer and location
// equivalences.  Pointer equivalences are those pointers that will have the
// same points-to sets, and location equivalences are those variables that
// always appear together in points-to sets.  It also includes an offline
// cycle detection algorithm that allows cycles to be collapsed sooner 
// during solving.
//
// The inclusion constraint solving phase iteratively propagates the inclusion
// constraints until a fixed point is reached.  This is an O(N^3) algorithm.
//
// Function constraints are handled as if they were structs with X fields.
// Thus, an access to argument X of function Y is an access to node index
// getNode(Y) + X.  This representation allows handling of indirect calls
// without any issues.  To wit, an indirect call Y(a,b) is equivalent to
// *(Y + 1) = a, *(Y + 2) = b.
// The return node for a function is always located at getNode(F) +
// CallReturnPos. The arguments start at getNode(F) + CallArgPos.
//
// Future Improvements:
//   Use of BDD's.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "anders-aa"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <set>
#include <list>
#include <map>
#include <stack>
#include <vector>
#include <queue>

// Determining the actual set of nodes the universal set can consist of is very
// expensive because it means propagating around very large sets.  We rely on
// other analysis being able to determine which nodes can never be pointed to in
// order to disambiguate further than "points-to anything".
#define FULL_UNIVERSAL 0

using namespace llvm;
STATISTIC(NumIters      , "Number of iterations to reach convergence");
STATISTIC(NumConstraints, "Number of constraints");
STATISTIC(NumNodes      , "Number of nodes");
STATISTIC(NumUnified    , "Number of variables unified");
STATISTIC(NumErased     , "Number of redundant constraints erased");

static const unsigned SelfRep = (unsigned)-1;
static const unsigned Unvisited = (unsigned)-1;
// Position of the function return node relative to the function node.
static const unsigned CallReturnPos = 1;
// Position of the function call node relative to the function node.
static const unsigned CallFirstArgPos = 2;

namespace {
  struct BitmapKeyInfo {
    static inline SparseBitVector<> *getEmptyKey() {
      return reinterpret_cast<SparseBitVector<> *>(-1);
    }
    static inline SparseBitVector<> *getTombstoneKey() {
      return reinterpret_cast<SparseBitVector<> *>(-2);
    }
    static unsigned getHashValue(const SparseBitVector<> *bitmap) {
      return bitmap->getHashValue();
    }
    static bool isEqual(const SparseBitVector<> *LHS,
                        const SparseBitVector<> *RHS) {
      if (LHS == RHS)
        return true;
      else if (LHS == getEmptyKey() || RHS == getEmptyKey()
               || LHS == getTombstoneKey() || RHS == getTombstoneKey())
        return false;

      return *LHS == *RHS;
    }

    static bool isPod() { return true; }
  };

  class VISIBILITY_HIDDEN Andersens : public ModulePass, public AliasAnalysis,
                                      private InstVisitor<Andersens> {
    struct Node;

    /// Constraint - Objects of this structure are used to represent the various
    /// constraints identified by the algorithm.  The constraints are 'copy',
    /// for statements like "A = B", 'load' for statements like "A = *B",
    /// 'store' for statements like "*A = B", and AddressOf for statements like
    /// A = alloca;  The Offset is applied as *(A + K) = B for stores,
    /// A = *(B + K) for loads, and A = B + K for copies.  It is
    /// illegal on addressof constraints (because it is statically
    /// resolvable to A = &C where C = B + K)

    struct Constraint {
      enum ConstraintType { Copy, Load, Store, AddressOf } Type;
      unsigned Dest;
      unsigned Src;
      unsigned Offset;

      Constraint(ConstraintType Ty, unsigned D, unsigned S, unsigned O = 0)
        : Type(Ty), Dest(D), Src(S), Offset(O) {
        assert((Offset == 0 || Ty != AddressOf) &&
               "Offset is illegal on addressof constraints");
      }

      bool operator==(const Constraint &RHS) const {
        return RHS.Type == Type
          && RHS.Dest == Dest
          && RHS.Src == Src
          && RHS.Offset == Offset;
      }

      bool operator!=(const Constraint &RHS) const {
        return !(*this == RHS);
      }

      bool operator<(const Constraint &RHS) const {
        if (RHS.Type != Type)
          return RHS.Type < Type;
        else if (RHS.Dest != Dest)
          return RHS.Dest < Dest;
        else if (RHS.Src != Src)
          return RHS.Src < Src;
        return RHS.Offset < Offset;
      }
    };

    // Information DenseSet requires implemented in order to be able to do
    // it's thing
    struct PairKeyInfo {
      static inline std::pair<unsigned, unsigned> getEmptyKey() {
        return std::make_pair(~0U, ~0U);
      }
      static inline std::pair<unsigned, unsigned> getTombstoneKey() {
        return std::make_pair(~0U - 1, ~0U - 1);
      }
      static unsigned getHashValue(const std::pair<unsigned, unsigned> &P) {
        return P.first ^ P.second;
      }
      static unsigned isEqual(const std::pair<unsigned, unsigned> &LHS,
                              const std::pair<unsigned, unsigned> &RHS) {
        return LHS == RHS;
      }
    };
    
    struct ConstraintKeyInfo {
      static inline Constraint getEmptyKey() {
        return Constraint(Constraint::Copy, ~0U, ~0U, ~0U);
      }
      static inline Constraint getTombstoneKey() {
        return Constraint(Constraint::Copy, ~0U - 1, ~0U - 1, ~0U - 1);
      }
      static unsigned getHashValue(const Constraint &C) {
        return C.Src ^ C.Dest ^ C.Type ^ C.Offset;
      }
      static bool isEqual(const Constraint &LHS,
                          const Constraint &RHS) {
        return LHS.Type == RHS.Type && LHS.Dest == RHS.Dest
          && LHS.Src == RHS.Src && LHS.Offset == RHS.Offset;
      }
    };

    // Node class - This class is used to represent a node in the constraint
    // graph.  Due to various optimizations, it is not always the case that
    // there is a mapping from a Node to a Value.  In particular, we add
    // artificial Node's that represent the set of pointed-to variables shared
    // for each location equivalent Node.
    struct Node {
    private:
      static unsigned Counter;

    public:
      Value *Val;
      SparseBitVector<> *Edges;
      SparseBitVector<> *PointsTo;
      SparseBitVector<> *OldPointsTo;
      std::list<Constraint> Constraints;

      // Pointer and location equivalence labels
      unsigned PointerEquivLabel;
      unsigned LocationEquivLabel;
      // Predecessor edges, both real and implicit
      SparseBitVector<> *PredEdges;
      SparseBitVector<> *ImplicitPredEdges;
      // Set of nodes that point to us, only use for location equivalence.
      SparseBitVector<> *PointedToBy;
      // Number of incoming edges, used during variable substitution to early
      // free the points-to sets
      unsigned NumInEdges;
      // True if our points-to set is in the Set2PEClass map
      bool StoredInHash;
      // True if our node has no indirect constraints (complex or otherwise)
      bool Direct;
      // True if the node is address taken, *or* it is part of a group of nodes
      // that must be kept together.  This is set to true for functions and
      // their arg nodes, which must be kept at the same position relative to
      // their base function node.
      bool AddressTaken;

      // Nodes in cycles (or in equivalence classes) are united together using a
      // standard union-find representation with path compression.  NodeRep
      // gives the index into GraphNodes for the representative Node.
      unsigned NodeRep;

      // Modification timestamp.  Assigned from Counter.
      // Used for work list prioritization.
      unsigned Timestamp;

      explicit Node(bool direct = true) :
        Val(0), Edges(0), PointsTo(0), OldPointsTo(0), 
        PointerEquivLabel(0), LocationEquivLabel(0), PredEdges(0),
        ImplicitPredEdges(0), PointedToBy(0), NumInEdges(0),
        StoredInHash(false), Direct(direct), AddressTaken(false),
        NodeRep(SelfRep), Timestamp(0) { }

      Node *setValue(Value *V) {
        assert(Val == 0 && "Value already set for this node!");
        Val = V;
        return this;
      }

      /// getValue - Return the LLVM value corresponding to this node.
      ///
      Value *getValue() const { return Val; }

      /// addPointerTo - Add a pointer to the list of pointees of this node,
      /// returning true if this caused a new pointer to be added, or false if
      /// we already knew about the points-to relation.
      bool addPointerTo(unsigned Node) {
        return PointsTo->test_and_set(Node);
      }

      /// intersects - Return true if the points-to set of this node intersects
      /// with the points-to set of the specified node.
      bool intersects(Node *N) const;

      /// intersectsIgnoring - Return true if the points-to set of this node
      /// intersects with the points-to set of the specified node on any nodes
      /// except for the specified node to ignore.
      bool intersectsIgnoring(Node *N, unsigned) const;

      // Timestamp a node (used for work list prioritization)
      void Stamp() {
        Timestamp = Counter++;
      }

      bool isRep() const {
        return( (int) NodeRep < 0 );
      }
    };

    struct WorkListElement {
      Node* node;
      unsigned Timestamp;
      WorkListElement(Node* n, unsigned t) : node(n), Timestamp(t) {}

      // Note that we reverse the sense of the comparison because we
      // actually want to give low timestamps the priority over high,
      // whereas priority is typically interpreted as a greater value is
      // given high priority.
      bool operator<(const WorkListElement& that) const {
        return( this->Timestamp > that.Timestamp );
      }
    };

    // Priority-queue based work list specialized for Nodes.
    class WorkList {
      std::priority_queue<WorkListElement> Q;

    public:
      void insert(Node* n) {
        Q.push( WorkListElement(n, n->Timestamp) );
      }

      // We automatically discard non-representative nodes and nodes
      // that were in the work list twice (we keep a copy of the
      // timestamp in the work list so we can detect this situation by
      // comparing against the node's current timestamp).
      Node* pop() {
        while( !Q.empty() ) {
          WorkListElement x = Q.top(); Q.pop();
          Node* INode = x.node;

          if( INode->isRep() &&
              INode->Timestamp == x.Timestamp ) {
            return(x.node);
          }
        }
        return(0);
      }

      bool empty() {
        return Q.empty();
      }
    };

    /// GraphNodes - This vector is populated as part of the object
    /// identification stage of the analysis, which populates this vector with a
    /// node for each memory object and fills in the ValueNodes map.
    std::vector<Node> GraphNodes;

    /// ValueNodes - This map indicates the Node that a particular Value* is
    /// represented by.  This contains entries for all pointers.
    DenseMap<Value*, unsigned> ValueNodes;

    /// ObjectNodes - This map contains entries for each memory object in the
    /// program: globals, alloca's and mallocs.
    DenseMap<Value*, unsigned> ObjectNodes;

    /// ReturnNodes - This map contains an entry for each function in the
    /// program that returns a value.
    DenseMap<Function*, unsigned> ReturnNodes;

    /// VarargNodes - This map contains the entry used to represent all pointers
    /// passed through the varargs portion of a function call for a particular
    /// function.  An entry is not present in this map for functions that do not
    /// take variable arguments.
    DenseMap<Function*, unsigned> VarargNodes;


    /// Constraints - This vector contains a list of all of the constraints
    /// identified by the program.
    std::vector<Constraint> Constraints;

    // Map from graph node to maximum K value that is allowed (for functions,
    // this is equivalent to the number of arguments + CallFirstArgPos)
    std::map<unsigned, unsigned> MaxK;

    /// This enum defines the GraphNodes indices that correspond to important
    /// fixed sets.
    enum {
      UniversalSet = 0,
      NullPtr      = 1,
      NullObject   = 2,
      NumberSpecialNodes
    };
    // Stack for Tarjan's
    std::stack<unsigned> SCCStack;
    // Map from Graph Node to DFS number
    std::vector<unsigned> Node2DFS;
    // Map from Graph Node to Deleted from graph.
    std::vector<bool> Node2Deleted;
    // Same as Node Maps, but implemented as std::map because it is faster to
    // clear 
    std::map<unsigned, unsigned> Tarjan2DFS;
    std::map<unsigned, bool> Tarjan2Deleted;
    // Current DFS number
    unsigned DFSNumber;

    // Work lists.
    WorkList w1, w2;
    WorkList *CurrWL, *NextWL; // "current" and "next" work lists

    // Offline variable substitution related things

    // Temporary rep storage, used because we can't collapse SCC's in the
    // predecessor graph by uniting the variables permanently, we can only do so
    // for the successor graph.
    std::vector<unsigned> VSSCCRep;
    // Mapping from node to whether we have visited it during SCC finding yet.
    std::vector<bool> Node2Visited;
    // During variable substitution, we create unknowns to represent the unknown
    // value that is a dereference of a variable.  These nodes are known as
    // "ref" nodes (since they represent the value of dereferences).
    unsigned FirstRefNode;
    // During HVN, we create represent address taken nodes as if they were
    // unknown (since HVN, unlike HU, does not evaluate unions).
    unsigned FirstAdrNode;
    // Current pointer equivalence class number
    unsigned PEClass;
    // Mapping from points-to sets to equivalence classes
    typedef DenseMap<SparseBitVector<> *, unsigned, BitmapKeyInfo> BitVectorMap;
    BitVectorMap Set2PEClass;
    // Mapping from pointer equivalences to the representative node.  -1 if we
    // have no representative node for this pointer equivalence class yet.
    std::vector<int> PEClass2Node;
    // Mapping from pointer equivalences to representative node.  This includes
    // pointer equivalent but not location equivalent variables. -1 if we have
    // no representative node for this pointer equivalence class yet.
    std::vector<int> PENLEClass2Node;
    // Union/Find for HCD
    std::vector<unsigned> HCDSCCRep;
    // HCD's offline-detected cycles; "Statically DeTected"
    // -1 if not part of such a cycle, otherwise a representative node.
    std::vector<int> SDT;
    // Whether to use SDT (UniteNodes can use it during solving, but not before)
    bool SDTActive;

  public:
    static char ID;
    Andersens() : ModulePass(&ID) {}

    bool runOnModule(Module &M) {
      InitializeAliasAnalysis(this);
      IdentifyObjects(M);
      CollectConstraints(M);
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa-constraints"
      DEBUG(PrintConstraints());
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa"
      SolveConstraints();
      DEBUG(PrintPointsToGraph());

      // Free the constraints list, as we don't need it to respond to alias
      // requests.
      std::vector<Constraint>().swap(Constraints);
      //These are needed for Print() (-analyze in opt)
      //ObjectNodes.clear();
      //ReturnNodes.clear();
      //VarargNodes.clear();
      return false;
    }

    void releaseMemory() {
      // FIXME: Until we have transitively required passes working correctly,
      // this cannot be enabled!  Otherwise, using -count-aa with the pass
      // causes memory to be freed too early. :(
#if 0
      // The memory objects and ValueNodes data structures at the only ones that
      // are still live after construction.
      std::vector<Node>().swap(GraphNodes);
      ValueNodes.clear();
#endif
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.setPreservesAll();                         // Does not transform code
    }

    //------------------------------------------------
    // Implement the AliasAnalysis API
    //
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);
    virtual ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);
    virtual ModRefResult getModRefInfo(CallSite CS1, CallSite CS2);
    void getMustAliases(Value *P, std::vector<Value*> &RetVals);
    bool pointsToConstantMemory(const Value *P);

    virtual void deleteValue(Value *V) {
      ValueNodes.erase(V);
      getAnalysis<AliasAnalysis>().deleteValue(V);
    }

    virtual void copyValue(Value *From, Value *To) {
      ValueNodes[To] = ValueNodes[From];
      getAnalysis<AliasAnalysis>().copyValue(From, To);
    }

  private:
    /// getNode - Return the node corresponding to the specified pointer scalar.
    ///
    unsigned getNode(Value *V) {
      if (Constant *C = dyn_cast<Constant>(V))
        if (!isa<GlobalValue>(C))
          return getNodeForConstantPointer(C);

      DenseMap<Value*, unsigned>::iterator I = ValueNodes.find(V);
      if (I == ValueNodes.end()) {
#ifndef NDEBUG
        V->dump();
#endif
        assert(0 && "Value does not have a node in the points-to graph!");
      }
      return I->second;
    }

    /// getObject - Return the node corresponding to the memory object for the
    /// specified global or allocation instruction.
    unsigned getObject(Value *V) const {
      DenseMap<Value*, unsigned>::iterator I = ObjectNodes.find(V);
      assert(I != ObjectNodes.end() &&
             "Value does not have an object in the points-to graph!");
      return I->second;
    }

    /// getReturnNode - Return the node representing the return value for the
    /// specified function.
    unsigned getReturnNode(Function *F) const {
      DenseMap<Function*, unsigned>::iterator I = ReturnNodes.find(F);
      assert(I != ReturnNodes.end() && "Function does not return a value!");
      return I->second;
    }

    /// getVarargNode - Return the node representing the variable arguments
    /// formal for the specified function.
    unsigned getVarargNode(Function *F) const {
      DenseMap<Function*, unsigned>::iterator I = VarargNodes.find(F);
      assert(I != VarargNodes.end() && "Function does not take var args!");
      return I->second;
    }

    /// getNodeValue - Get the node for the specified LLVM value and set the
    /// value for it to be the specified value.
    unsigned getNodeValue(Value &V) {
      unsigned Index = getNode(&V);
      GraphNodes[Index].setValue(&V);
      return Index;
    }

    unsigned UniteNodes(unsigned First, unsigned Second,
                        bool UnionByRank = true);
    unsigned FindNode(unsigned Node);
    unsigned FindNode(unsigned Node) const;

    void IdentifyObjects(Module &M);
    void CollectConstraints(Module &M);
    bool AnalyzeUsesOfFunction(Value *);
    void CreateConstraintGraph();
    void OptimizeConstraints();
    unsigned FindEquivalentNode(unsigned, unsigned);
    void ClumpAddressTaken();
    void RewriteConstraints();
    void HU();
    void HVN();
    void HCD();
    void Search(unsigned Node);
    void UnitePointerEquivalences();
    void SolveConstraints();
    bool QueryNode(unsigned Node);
    void Condense(unsigned Node);
    void HUValNum(unsigned Node);
    void HVNValNum(unsigned Node);
    unsigned getNodeForConstantPointer(Constant *C);
    unsigned getNodeForConstantPointerTarget(Constant *C);
    void AddGlobalInitializerConstraints(unsigned, Constant *C);

    void AddConstraintsForNonInternalLinkage(Function *F);
    void AddConstraintsForCall(CallSite CS, Function *F);
    bool AddConstraintsForExternalCall(CallSite CS, Function *F);


    void PrintNode(const Node *N) const;
    void PrintConstraints() const ;
    void PrintConstraint(const Constraint &) const;
    void PrintLabels() const;
    void PrintPointsToGraph() const;

    //===------------------------------------------------------------------===//
    // Instruction visitation methods for adding constraints
    //
    friend class InstVisitor<Andersens>;
    void visitReturnInst(ReturnInst &RI);
    void visitInvokeInst(InvokeInst &II) { visitCallSite(CallSite(&II)); }
    void visitCallInst(CallInst &CI) { visitCallSite(CallSite(&CI)); }
    void visitCallSite(CallSite CS);
    void visitAllocationInst(AllocationInst &AI);
    void visitLoadInst(LoadInst &LI);
    void visitStoreInst(StoreInst &SI);
    void visitGetElementPtrInst(GetElementPtrInst &GEP);
    void visitPHINode(PHINode &PN);
    void visitCastInst(CastInst &CI);
    void visitICmpInst(ICmpInst &ICI) {} // NOOP!
    void visitFCmpInst(FCmpInst &ICI) {} // NOOP!
    void visitSelectInst(SelectInst &SI);
    void visitVAArg(VAArgInst &I);
    void visitInstruction(Instruction &I);

    //===------------------------------------------------------------------===//
    // Implement Analyize interface
    //
    void print(std::ostream &O, const Module* M) const {
      PrintPointsToGraph();
    }
  };
}

char Andersens::ID = 0;
static RegisterPass<Andersens>
X("anders-aa", "Andersen's Interprocedural Alias Analysis", false, true);
static RegisterAnalysisGroup<AliasAnalysis> Y(X);

// Initialize Timestamp Counter (static).
unsigned Andersens::Node::Counter = 0;

ModulePass *llvm::createAndersensPass() { return new Andersens(); }

//===----------------------------------------------------------------------===//
//                  AliasAnalysis Interface Implementation
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasResult Andersens::alias(const Value *V1, unsigned V1Size,
                                            const Value *V2, unsigned V2Size) {
  Node *N1 = &GraphNodes[FindNode(getNode(const_cast<Value*>(V1)))];
  Node *N2 = &GraphNodes[FindNode(getNode(const_cast<Value*>(V2)))];

  // Check to see if the two pointers are known to not alias.  They don't alias
  // if their points-to sets do not intersect.
  if (!N1->intersectsIgnoring(N2, NullObject))
    return NoAlias;

  return AliasAnalysis::alias(V1, V1Size, V2, V2Size);
}

AliasAnalysis::ModRefResult
Andersens::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  // The only thing useful that we can contribute for mod/ref information is
  // when calling external function calls: if we know that memory never escapes
  // from the program, it cannot be modified by an external call.
  //
  // NOTE: This is not really safe, at least not when the entire program is not
  // available.  The deal is that the external function could call back into the
  // program and modify stuff.  We ignore this technical niggle for now.  This
  // is, after all, a "research quality" implementation of Andersen's analysis.
  if (Function *F = CS.getCalledFunction())
    if (F->isDeclaration()) {
      Node *N1 = &GraphNodes[FindNode(getNode(P))];

      if (N1->PointsTo->empty())
        return NoModRef;
#if FULL_UNIVERSAL
      if (!UniversalSet->PointsTo->test(FindNode(getNode(P))))
        return NoModRef;  // Universal set does not contain P
#else
      if (!N1->PointsTo->test(UniversalSet))
        return NoModRef;  // P doesn't point to the universal set.
#endif
    }

  return AliasAnalysis::getModRefInfo(CS, P, Size);
}

AliasAnalysis::ModRefResult
Andersens::getModRefInfo(CallSite CS1, CallSite CS2) {
  return AliasAnalysis::getModRefInfo(CS1,CS2);
}

/// getMustAlias - We can provide must alias information if we know that a
/// pointer can only point to a specific function or the null pointer.
/// Unfortunately we cannot determine must-alias information for global
/// variables or any other memory memory objects because we do not track whether
/// a pointer points to the beginning of an object or a field of it.
void Andersens::getMustAliases(Value *P, std::vector<Value*> &RetVals) {
  Node *N = &GraphNodes[FindNode(getNode(P))];
  if (N->PointsTo->count() == 1) {
    Node *Pointee = &GraphNodes[N->PointsTo->find_first()];
    // If a function is the only object in the points-to set, then it must be
    // the destination.  Note that we can't handle global variables here,
    // because we don't know if the pointer is actually pointing to a field of
    // the global or to the beginning of it.
    if (Value *V = Pointee->getValue()) {
      if (Function *F = dyn_cast<Function>(V))
        RetVals.push_back(F);
    } else {
      // If the object in the points-to set is the null object, then the null
      // pointer is a must alias.
      if (Pointee == &GraphNodes[NullObject])
        RetVals.push_back(Constant::getNullValue(P->getType()));
    }
  }
  AliasAnalysis::getMustAliases(P, RetVals);
}

/// pointsToConstantMemory - If we can determine that this pointer only points
/// to constant memory, return true.  In practice, this means that if the
/// pointer can only point to constant globals, functions, or the null pointer,
/// return true.
///
bool Andersens::pointsToConstantMemory(const Value *P) {
  Node *N = &GraphNodes[FindNode(getNode(const_cast<Value*>(P)))];
  unsigned i;

  for (SparseBitVector<>::iterator bi = N->PointsTo->begin();
       bi != N->PointsTo->end();
       ++bi) {
    i = *bi;
    Node *Pointee = &GraphNodes[i];
    if (Value *V = Pointee->getValue()) {
      if (!isa<GlobalValue>(V) || (isa<GlobalVariable>(V) &&
                                   !cast<GlobalVariable>(V)->isConstant()))
        return AliasAnalysis::pointsToConstantMemory(P);
    } else {
      if (i != NullObject)
        return AliasAnalysis::pointsToConstantMemory(P);
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
//                       Object Identification Phase
//===----------------------------------------------------------------------===//

/// IdentifyObjects - This stage scans the program, adding an entry to the
/// GraphNodes list for each memory object in the program (global stack or
/// heap), and populates the ValueNodes and ObjectNodes maps for these objects.
///
void Andersens::IdentifyObjects(Module &M) {
  unsigned NumObjects = 0;

  // Object #0 is always the universal set: the object that we don't know
  // anything about.
  assert(NumObjects == UniversalSet && "Something changed!");
  ++NumObjects;

  // Object #1 always represents the null pointer.
  assert(NumObjects == NullPtr && "Something changed!");
  ++NumObjects;

  // Object #2 always represents the null object (the object pointed to by null)
  assert(NumObjects == NullObject && "Something changed!");
  ++NumObjects;

  // Add all the globals first.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    ObjectNodes[I] = NumObjects++;
    ValueNodes[I] = NumObjects++;
  }

  // Add nodes for all of the functions and the instructions inside of them.
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    // The function itself is a memory object.
    unsigned First = NumObjects;
    ValueNodes[F] = NumObjects++;
    if (isa<PointerType>(F->getFunctionType()->getReturnType()))
      ReturnNodes[F] = NumObjects++;
    if (F->getFunctionType()->isVarArg())
      VarargNodes[F] = NumObjects++;


    // Add nodes for all of the incoming pointer arguments.
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I)
      {
        if (isa<PointerType>(I->getType()))
          ValueNodes[I] = NumObjects++;
      }
    MaxK[First] = NumObjects - First;

    // Scan the function body, creating a memory object for each heap/stack
    // allocation in the body of the function and a node to represent all
    // pointer values defined by instructions and used as operands.
    for (inst_iterator II = inst_begin(F), E = inst_end(F); II != E; ++II) {
      // If this is an heap or stack allocation, create a node for the memory
      // object.
      if (isa<PointerType>(II->getType())) {
        ValueNodes[&*II] = NumObjects++;
        if (AllocationInst *AI = dyn_cast<AllocationInst>(&*II))
          ObjectNodes[AI] = NumObjects++;
      }

      // Calls to inline asm need to be added as well because the callee isn't
      // referenced anywhere else.
      if (CallInst *CI = dyn_cast<CallInst>(&*II)) {
        Value *Callee = CI->getCalledValue();
        if (isa<InlineAsm>(Callee))
          ValueNodes[Callee] = NumObjects++;
      }
    }
  }

  // Now that we know how many objects to create, make them all now!
  GraphNodes.resize(NumObjects);
  NumNodes += NumObjects;
}

//===----------------------------------------------------------------------===//
//                     Constraint Identification Phase
//===----------------------------------------------------------------------===//

/// getNodeForConstantPointer - Return the node corresponding to the constant
/// pointer itself.
unsigned Andersens::getNodeForConstantPointer(Constant *C) {
  assert(isa<PointerType>(C->getType()) && "Not a constant pointer!");

  if (isa<ConstantPointerNull>(C) || isa<UndefValue>(C))
    return NullPtr;
  else if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
    return getNode(GV);
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      return getNodeForConstantPointer(CE->getOperand(0));
    case Instruction::IntToPtr:
      return UniversalSet;
    case Instruction::BitCast:
      return getNodeForConstantPointer(CE->getOperand(0));
    default:
      cerr << "Constant Expr not yet handled: " << *CE << "\n";
      assert(0);
    }
  } else {
    assert(0 && "Unknown constant pointer!");
  }
  return 0;
}

/// getNodeForConstantPointerTarget - Return the node POINTED TO by the
/// specified constant pointer.
unsigned Andersens::getNodeForConstantPointerTarget(Constant *C) {
  assert(isa<PointerType>(C->getType()) && "Not a constant pointer!");

  if (isa<ConstantPointerNull>(C))
    return NullObject;
  else if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
    return getObject(GV);
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      return getNodeForConstantPointerTarget(CE->getOperand(0));
    case Instruction::IntToPtr:
      return UniversalSet;
    case Instruction::BitCast:
      return getNodeForConstantPointerTarget(CE->getOperand(0));
    default:
      cerr << "Constant Expr not yet handled: " << *CE << "\n";
      assert(0);
    }
  } else {
    assert(0 && "Unknown constant pointer!");
  }
  return 0;
}

/// AddGlobalInitializerConstraints - Add inclusion constraints for the memory
/// object N, which contains values indicated by C.
void Andersens::AddGlobalInitializerConstraints(unsigned NodeIndex,
                                                Constant *C) {
  if (C->getType()->isSingleValueType()) {
    if (isa<PointerType>(C->getType()))
      Constraints.push_back(Constraint(Constraint::Copy, NodeIndex,
                                       getNodeForConstantPointer(C)));
  } else if (C->isNullValue()) {
    Constraints.push_back(Constraint(Constraint::Copy, NodeIndex,
                                     NullObject));
    return;
  } else if (!isa<UndefValue>(C)) {
    // If this is an array or struct, include constraints for each element.
    assert(isa<ConstantArray>(C) || isa<ConstantStruct>(C));
    for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i)
      AddGlobalInitializerConstraints(NodeIndex,
                                      cast<Constant>(C->getOperand(i)));
  }
}

/// AddConstraintsForNonInternalLinkage - If this function does not have
/// internal linkage, realize that we can't trust anything passed into or
/// returned by this function.
void Andersens::AddConstraintsForNonInternalLinkage(Function *F) {
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I)
    if (isa<PointerType>(I->getType()))
      // If this is an argument of an externally accessible function, the
      // incoming pointer might point to anything.
      Constraints.push_back(Constraint(Constraint::Copy, getNode(I),
                                       UniversalSet));
}

/// AddConstraintsForCall - If this is a call to a "known" function, add the
/// constraints and return true.  If this is a call to an unknown function,
/// return false.
bool Andersens::AddConstraintsForExternalCall(CallSite CS, Function *F) {
  assert(F->isDeclaration() && "Not an external function!");

  // These functions don't induce any points-to constraints.
  if (F->getName() == "atoi" || F->getName() == "atof" ||
      F->getName() == "atol" || F->getName() == "atoll" ||
      F->getName() == "remove" || F->getName() == "unlink" ||
      F->getName() == "rename" || F->getName() == "memcmp" ||
      F->getName() == "llvm.memset" ||
      F->getName() == "strcmp" || F->getName() == "strncmp" ||
      F->getName() == "execl" || F->getName() == "execlp" ||
      F->getName() == "execle" || F->getName() == "execv" ||
      F->getName() == "execvp" || F->getName() == "chmod" ||
      F->getName() == "puts" || F->getName() == "write" ||
      F->getName() == "open" || F->getName() == "create" ||
      F->getName() == "truncate" || F->getName() == "chdir" ||
      F->getName() == "mkdir" || F->getName() == "rmdir" ||
      F->getName() == "read" || F->getName() == "pipe" ||
      F->getName() == "wait" || F->getName() == "time" ||
      F->getName() == "stat" || F->getName() == "fstat" ||
      F->getName() == "lstat" || F->getName() == "strtod" ||
      F->getName() == "strtof" || F->getName() == "strtold" ||
      F->getName() == "fopen" || F->getName() == "fdopen" ||
      F->getName() == "freopen" ||
      F->getName() == "fflush" || F->getName() == "feof" ||
      F->getName() == "fileno" || F->getName() == "clearerr" ||
      F->getName() == "rewind" || F->getName() == "ftell" ||
      F->getName() == "ferror" || F->getName() == "fgetc" ||
      F->getName() == "fgetc" || F->getName() == "_IO_getc" ||
      F->getName() == "fwrite" || F->getName() == "fread" ||
      F->getName() == "fgets" || F->getName() == "ungetc" ||
      F->getName() == "fputc" ||
      F->getName() == "fputs" || F->getName() == "putc" ||
      F->getName() == "ftell" || F->getName() == "rewind" ||
      F->getName() == "_IO_putc" || F->getName() == "fseek" ||
      F->getName() == "fgetpos" || F->getName() == "fsetpos" ||
      F->getName() == "printf" || F->getName() == "fprintf" ||
      F->getName() == "sprintf" || F->getName() == "vprintf" ||
      F->getName() == "vfprintf" || F->getName() == "vsprintf" ||
      F->getName() == "scanf" || F->getName() == "fscanf" ||
      F->getName() == "sscanf" || F->getName() == "__assert_fail" ||
      F->getName() == "modf")
    return true;


  // These functions do induce points-to edges.
  if (F->getName() == "llvm.memcpy" ||
      F->getName() == "llvm.memmove" ||
      F->getName() == "memmove") {

    const FunctionType *FTy = F->getFunctionType();
    if (FTy->getNumParams() > 1 && 
        isa<PointerType>(FTy->getParamType(0)) &&
        isa<PointerType>(FTy->getParamType(1))) {

      // *Dest = *Src, which requires an artificial graph node to represent the
      // constraint.  It is broken up into *Dest = temp, temp = *Src
      unsigned FirstArg = getNode(CS.getArgument(0));
      unsigned SecondArg = getNode(CS.getArgument(1));
      unsigned TempArg = GraphNodes.size();
      GraphNodes.push_back(Node());
      Constraints.push_back(Constraint(Constraint::Store,
                                       FirstArg, TempArg));
      Constraints.push_back(Constraint(Constraint::Load,
                                       TempArg, SecondArg));
      // In addition, Dest = Src
      Constraints.push_back(Constraint(Constraint::Copy,
                                       FirstArg, SecondArg));
      return true;
    }
  }

  // Result = Arg0
  if (F->getName() == "realloc" || F->getName() == "strchr" ||
      F->getName() == "strrchr" || F->getName() == "strstr" ||
      F->getName() == "strtok") {
    const FunctionType *FTy = F->getFunctionType();
    if (FTy->getNumParams() > 0 && 
        isa<PointerType>(FTy->getParamType(0))) {
      Constraints.push_back(Constraint(Constraint::Copy,
                                       getNode(CS.getInstruction()),
                                       getNode(CS.getArgument(0))));
      return true;
    }
  }

  return false;
}



/// AnalyzeUsesOfFunction - Look at all of the users of the specified function.
/// If this is used by anything complex (i.e., the address escapes), return
/// true.
bool Andersens::AnalyzeUsesOfFunction(Value *V) {

  if (!isa<PointerType>(V->getType())) return true;

  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (dyn_cast<LoadInst>(*UI)) {
      return false;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (V == SI->getOperand(1)) {
        return false;
      } else if (SI->getOperand(1)) {
        return true;  // Storing the pointer
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(*UI)) {
      if (AnalyzeUsesOfFunction(GEP)) return true;
    } else if (CallInst *CI = dyn_cast<CallInst>(*UI)) {
      // Make sure that this is just the function being called, not that it is
      // passing into the function.
      for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
        if (CI->getOperand(i) == V) return true;
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(*UI)) {
      // Make sure that this is just the function being called, not that it is
      // passing into the function.
      for (unsigned i = 3, e = II->getNumOperands(); i != e; ++i)
        if (II->getOperand(i) == V) return true;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(*UI)) {
      if (CE->getOpcode() == Instruction::GetElementPtr ||
          CE->getOpcode() == Instruction::BitCast) {
        if (AnalyzeUsesOfFunction(CE))
          return true;
      } else {
        return true;
      }
    } else if (ICmpInst *ICI = dyn_cast<ICmpInst>(*UI)) {
      if (!isa<ConstantPointerNull>(ICI->getOperand(1)))
        return true;  // Allow comparison against null.
    } else if (dyn_cast<FreeInst>(*UI)) {
      return false;
    } else {
      return true;
    }
  return false;
}

/// CollectConstraints - This stage scans the program, adding a constraint to
/// the Constraints list for each instruction in the program that induces a
/// constraint, and setting up the initial points-to graph.
///
void Andersens::CollectConstraints(Module &M) {
  // First, the universal set points to itself.
  Constraints.push_back(Constraint(Constraint::AddressOf, UniversalSet,
                                   UniversalSet));
  Constraints.push_back(Constraint(Constraint::Store, UniversalSet,
                                   UniversalSet));

  // Next, the null pointer points to the null object.
  Constraints.push_back(Constraint(Constraint::AddressOf, NullPtr, NullObject));

  // Next, add any constraints on global variables and their initializers.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    // Associate the address of the global object as pointing to the memory for
    // the global: &G = <G memory>
    unsigned ObjectIndex = getObject(I);
    Node *Object = &GraphNodes[ObjectIndex];
    Object->setValue(I);
    Constraints.push_back(Constraint(Constraint::AddressOf, getNodeValue(*I),
                                     ObjectIndex));

    if (I->hasInitializer()) {
      AddGlobalInitializerConstraints(ObjectIndex, I->getInitializer());
    } else {
      // If it doesn't have an initializer (i.e. it's defined in another
      // translation unit), it points to the universal set.
      Constraints.push_back(Constraint(Constraint::Copy, ObjectIndex,
                                       UniversalSet));
    }
  }

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    // Set up the return value node.
    if (isa<PointerType>(F->getFunctionType()->getReturnType()))
      GraphNodes[getReturnNode(F)].setValue(F);
    if (F->getFunctionType()->isVarArg())
      GraphNodes[getVarargNode(F)].setValue(F);

    // Set up incoming argument nodes.
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I)
      if (isa<PointerType>(I->getType()))
        getNodeValue(*I);

    // At some point we should just add constraints for the escaping functions
    // at solve time, but this slows down solving. For now, we simply mark
    // address taken functions as escaping and treat them as external.
    if (!F->hasLocalLinkage() || AnalyzeUsesOfFunction(F))
      AddConstraintsForNonInternalLinkage(F);

    if (!F->isDeclaration()) {
      // Scan the function body, creating a memory object for each heap/stack
      // allocation in the body of the function and a node to represent all
      // pointer values defined by instructions and used as operands.
      visit(F);
    } else {
      // External functions that return pointers return the universal set.
      if (isa<PointerType>(F->getFunctionType()->getReturnType()))
        Constraints.push_back(Constraint(Constraint::Copy,
                                         getReturnNode(F),
                                         UniversalSet));

      // Any pointers that are passed into the function have the universal set
      // stored into them.
      for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
           I != E; ++I)
        if (isa<PointerType>(I->getType())) {
          // Pointers passed into external functions could have anything stored
          // through them.
          Constraints.push_back(Constraint(Constraint::Store, getNode(I),
                                           UniversalSet));
          // Memory objects passed into external function calls can have the
          // universal set point to them.
#if FULL_UNIVERSAL
          Constraints.push_back(Constraint(Constraint::Copy,
                                           UniversalSet,
                                           getNode(I)));
#else
          Constraints.push_back(Constraint(Constraint::Copy,
                                           getNode(I),
                                           UniversalSet));
#endif
        }

      // If this is an external varargs function, it can also store pointers
      // into any pointers passed through the varargs section.
      if (F->getFunctionType()->isVarArg())
        Constraints.push_back(Constraint(Constraint::Store, getVarargNode(F),
                                         UniversalSet));
    }
  }
  NumConstraints += Constraints.size();
}


void Andersens::visitInstruction(Instruction &I) {
#ifdef NDEBUG
  return;          // This function is just a big assert.
#endif
  if (isa<BinaryOperator>(I))
    return;
  // Most instructions don't have any effect on pointer values.
  switch (I.getOpcode()) {
  case Instruction::Br:
  case Instruction::Switch:
  case Instruction::Unwind:
  case Instruction::Unreachable:
  case Instruction::Free:
  case Instruction::ICmp:
  case Instruction::FCmp:
    return;
  default:
    // Is this something we aren't handling yet?
    cerr << "Unknown instruction: " << I;
    abort();
  }
}

void Andersens::visitAllocationInst(AllocationInst &AI) {
  unsigned ObjectIndex = getObject(&AI);
  GraphNodes[ObjectIndex].setValue(&AI);
  Constraints.push_back(Constraint(Constraint::AddressOf, getNodeValue(AI),
                                   ObjectIndex));
}

void Andersens::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() && isa<PointerType>(RI.getOperand(0)->getType()))
    // return V   -->   <Copy/retval{F}/v>
    Constraints.push_back(Constraint(Constraint::Copy,
                                     getReturnNode(RI.getParent()->getParent()),
                                     getNode(RI.getOperand(0))));
}

void Andersens::visitLoadInst(LoadInst &LI) {
  if (isa<PointerType>(LI.getType()))
    // P1 = load P2  -->  <Load/P1/P2>
    Constraints.push_back(Constraint(Constraint::Load, getNodeValue(LI),
                                     getNode(LI.getOperand(0))));
}

void Andersens::visitStoreInst(StoreInst &SI) {
  if (isa<PointerType>(SI.getOperand(0)->getType()))
    // store P1, P2  -->  <Store/P2/P1>
    Constraints.push_back(Constraint(Constraint::Store,
                                     getNode(SI.getOperand(1)),
                                     getNode(SI.getOperand(0))));
}

void Andersens::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  // P1 = getelementptr P2, ... --> <Copy/P1/P2>
  Constraints.push_back(Constraint(Constraint::Copy, getNodeValue(GEP),
                                   getNode(GEP.getOperand(0))));
}

void Andersens::visitPHINode(PHINode &PN) {
  if (isa<PointerType>(PN.getType())) {
    unsigned PNN = getNodeValue(PN);
    for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
      // P1 = phi P2, P3  -->  <Copy/P1/P2>, <Copy/P1/P3>, ...
      Constraints.push_back(Constraint(Constraint::Copy, PNN,
                                       getNode(PN.getIncomingValue(i))));
  }
}

void Andersens::visitCastInst(CastInst &CI) {
  Value *Op = CI.getOperand(0);
  if (isa<PointerType>(CI.getType())) {
    if (isa<PointerType>(Op->getType())) {
      // P1 = cast P2  --> <Copy/P1/P2>
      Constraints.push_back(Constraint(Constraint::Copy, getNodeValue(CI),
                                       getNode(CI.getOperand(0))));
    } else {
      // P1 = cast int --> <Copy/P1/Univ>
#if 0
      Constraints.push_back(Constraint(Constraint::Copy, getNodeValue(CI),
                                       UniversalSet));
#else
      getNodeValue(CI);
#endif
    }
  } else if (isa<PointerType>(Op->getType())) {
    // int = cast P1 --> <Copy/Univ/P1>
#if 0
    Constraints.push_back(Constraint(Constraint::Copy,
                                     UniversalSet,
                                     getNode(CI.getOperand(0))));
#else
    getNode(CI.getOperand(0));
#endif
  }
}

void Andersens::visitSelectInst(SelectInst &SI) {
  if (isa<PointerType>(SI.getType())) {
    unsigned SIN = getNodeValue(SI);
    // P1 = select C, P2, P3   ---> <Copy/P1/P2>, <Copy/P1/P3>
    Constraints.push_back(Constraint(Constraint::Copy, SIN,
                                     getNode(SI.getOperand(1))));
    Constraints.push_back(Constraint(Constraint::Copy, SIN,
                                     getNode(SI.getOperand(2))));
  }
}

void Andersens::visitVAArg(VAArgInst &I) {
  assert(0 && "vaarg not handled yet!");
}

/// AddConstraintsForCall - Add constraints for a call with actual arguments
/// specified by CS to the function specified by F.  Note that the types of
/// arguments might not match up in the case where this is an indirect call and
/// the function pointer has been casted.  If this is the case, do something
/// reasonable.
void Andersens::AddConstraintsForCall(CallSite CS, Function *F) {
  Value *CallValue = CS.getCalledValue();
  bool IsDeref = F == NULL;

  // If this is a call to an external function, try to handle it directly to get
  // some taste of context sensitivity.
  if (F && F->isDeclaration() && AddConstraintsForExternalCall(CS, F))
    return;

  if (isa<PointerType>(CS.getType())) {
    unsigned CSN = getNode(CS.getInstruction());
    if (!F || isa<PointerType>(F->getFunctionType()->getReturnType())) {
      if (IsDeref)
        Constraints.push_back(Constraint(Constraint::Load, CSN,
                                         getNode(CallValue), CallReturnPos));
      else
        Constraints.push_back(Constraint(Constraint::Copy, CSN,
                                         getNode(CallValue) + CallReturnPos));
    } else {
      // If the function returns a non-pointer value, handle this just like we
      // treat a nonpointer cast to pointer.
      Constraints.push_back(Constraint(Constraint::Copy, CSN,
                                       UniversalSet));
    }
  } else if (F && isa<PointerType>(F->getFunctionType()->getReturnType())) {
#if FULL_UNIVERSAL
    Constraints.push_back(Constraint(Constraint::Copy,
                                     UniversalSet,
                                     getNode(CallValue) + CallReturnPos));
#else
    Constraints.push_back(Constraint(Constraint::Copy,
                                      getNode(CallValue) + CallReturnPos,
                                      UniversalSet));
#endif
                          
    
  }

  CallSite::arg_iterator ArgI = CS.arg_begin(), ArgE = CS.arg_end();
  bool external = !F ||  F->isDeclaration();
  if (F) {
    // Direct Call
    Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end();
    for (; AI != AE && ArgI != ArgE; ++AI, ++ArgI) 
      {
#if !FULL_UNIVERSAL
        if (external && isa<PointerType>((*ArgI)->getType())) 
          {
            // Add constraint that ArgI can now point to anything due to
            // escaping, as can everything it points to. The second portion of
            // this should be taken care of by universal = *universal
            Constraints.push_back(Constraint(Constraint::Copy,
                                             getNode(*ArgI),
                                             UniversalSet));
          }
#endif
        if (isa<PointerType>(AI->getType())) {
          if (isa<PointerType>((*ArgI)->getType())) {
            // Copy the actual argument into the formal argument.
            Constraints.push_back(Constraint(Constraint::Copy, getNode(AI),
                                             getNode(*ArgI)));
          } else {
            Constraints.push_back(Constraint(Constraint::Copy, getNode(AI),
                                             UniversalSet));
          }
        } else if (isa<PointerType>((*ArgI)->getType())) {
#if FULL_UNIVERSAL
          Constraints.push_back(Constraint(Constraint::Copy,
                                           UniversalSet,
                                           getNode(*ArgI)));
#else
          Constraints.push_back(Constraint(Constraint::Copy,
                                           getNode(*ArgI),
                                           UniversalSet));
#endif
        }
      }
  } else {
    //Indirect Call
    unsigned ArgPos = CallFirstArgPos;
    for (; ArgI != ArgE; ++ArgI) {
      if (isa<PointerType>((*ArgI)->getType())) {
        // Copy the actual argument into the formal argument.
        Constraints.push_back(Constraint(Constraint::Store,
                                         getNode(CallValue),
                                         getNode(*ArgI), ArgPos++));
      } else {
        Constraints.push_back(Constraint(Constraint::Store,
                                         getNode (CallValue),
                                         UniversalSet, ArgPos++));
      }
    }
  }
  // Copy all pointers passed through the varargs section to the varargs node.
  if (F && F->getFunctionType()->isVarArg())
    for (; ArgI != ArgE; ++ArgI)
      if (isa<PointerType>((*ArgI)->getType()))
        Constraints.push_back(Constraint(Constraint::Copy, getVarargNode(F),
                                         getNode(*ArgI)));
  // If more arguments are passed in than we track, just drop them on the floor.
}

void Andersens::visitCallSite(CallSite CS) {
  if (isa<PointerType>(CS.getType()))
    getNodeValue(*CS.getInstruction());

  if (Function *F = CS.getCalledFunction()) {
    AddConstraintsForCall(CS, F);
  } else {
    AddConstraintsForCall(CS, NULL);
  }
}

//===----------------------------------------------------------------------===//
//                         Constraint Solving Phase
//===----------------------------------------------------------------------===//

/// intersects - Return true if the points-to set of this node intersects
/// with the points-to set of the specified node.
bool Andersens::Node::intersects(Node *N) const {
  return PointsTo->intersects(N->PointsTo);
}

/// intersectsIgnoring - Return true if the points-to set of this node
/// intersects with the points-to set of the specified node on any nodes
/// except for the specified node to ignore.
bool Andersens::Node::intersectsIgnoring(Node *N, unsigned Ignoring) const {
  // TODO: If we are only going to call this with the same value for Ignoring,
  // we should move the special values out of the points-to bitmap.
  bool WeHadIt = PointsTo->test(Ignoring);
  bool NHadIt = N->PointsTo->test(Ignoring);
  bool Result = false;
  if (WeHadIt)
    PointsTo->reset(Ignoring);
  if (NHadIt)
    N->PointsTo->reset(Ignoring);
  Result = PointsTo->intersects(N->PointsTo);
  if (WeHadIt)
    PointsTo->set(Ignoring);
  if (NHadIt)
    N->PointsTo->set(Ignoring);
  return Result;
}

void dumpToDOUT(SparseBitVector<> *bitmap) {
#ifndef NDEBUG
  dump(*bitmap, DOUT);
#endif
}


/// Clump together address taken variables so that the points-to sets use up
/// less space and can be operated on faster.

void Andersens::ClumpAddressTaken() {
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa-renumber"
  std::vector<unsigned> Translate;
  std::vector<Node> NewGraphNodes;

  Translate.resize(GraphNodes.size());
  unsigned NewPos = 0;

  for (unsigned i = 0; i < Constraints.size(); ++i) {
    Constraint &C = Constraints[i];
    if (C.Type == Constraint::AddressOf) {
      GraphNodes[C.Src].AddressTaken = true;
    }
  }
  for (unsigned i = 0; i < NumberSpecialNodes; ++i) {
    unsigned Pos = NewPos++;
    Translate[i] = Pos;
    NewGraphNodes.push_back(GraphNodes[i]);
    DOUT << "Renumbering node " << i << " to node " << Pos << "\n";
  }

  // I believe this ends up being faster than making two vectors and splicing
  // them.
  for (unsigned i = NumberSpecialNodes; i < GraphNodes.size(); ++i) {
    if (GraphNodes[i].AddressTaken) {
      unsigned Pos = NewPos++;
      Translate[i] = Pos;
      NewGraphNodes.push_back(GraphNodes[i]);
      DOUT << "Renumbering node " << i << " to node " << Pos << "\n";
    }
  }

  for (unsigned i = NumberSpecialNodes; i < GraphNodes.size(); ++i) {
    if (!GraphNodes[i].AddressTaken) {
      unsigned Pos = NewPos++;
      Translate[i] = Pos;
      NewGraphNodes.push_back(GraphNodes[i]);
      DOUT << "Renumbering node " << i << " to node " << Pos << "\n";
    }
  }

  for (DenseMap<Value*, unsigned>::iterator Iter = ValueNodes.begin();
       Iter != ValueNodes.end();
       ++Iter)
    Iter->second = Translate[Iter->second];

  for (DenseMap<Value*, unsigned>::iterator Iter = ObjectNodes.begin();
       Iter != ObjectNodes.end();
       ++Iter)
    Iter->second = Translate[Iter->second];

  for (DenseMap<Function*, unsigned>::iterator Iter = ReturnNodes.begin();
       Iter != ReturnNodes.end();
       ++Iter)
    Iter->second = Translate[Iter->second];

  for (DenseMap<Function*, unsigned>::iterator Iter = VarargNodes.begin();
       Iter != VarargNodes.end();
       ++Iter)
    Iter->second = Translate[Iter->second];

  for (unsigned i = 0; i < Constraints.size(); ++i) {
    Constraint &C = Constraints[i];
    C.Src = Translate[C.Src];
    C.Dest = Translate[C.Dest];
  }

  GraphNodes.swap(NewGraphNodes);
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa"
}

/// The technique used here is described in "Exploiting Pointer and Location
/// Equivalence to Optimize Pointer Analysis. In the 14th International Static
/// Analysis Symposium (SAS), August 2007."  It is known as the "HVN" algorithm,
/// and is equivalent to value numbering the collapsed constraint graph without
/// evaluating unions.  This is used as a pre-pass to HU in order to resolve
/// first order pointer dereferences and speed up/reduce memory usage of HU.
/// Running both is equivalent to HRU without the iteration
/// HVN in more detail:
/// Imagine the set of constraints was simply straight line code with no loops
/// (we eliminate cycles, so there are no loops), such as:
/// E = &D
/// E = &C
/// E = F
/// F = G
/// G = F
/// Applying value numbering to this code tells us:
/// G == F == E
///
/// For HVN, this is as far as it goes.  We assign new value numbers to every
/// "address node", and every "reference node".
/// To get the optimal result for this, we use a DFS + SCC (since all nodes in a
/// cycle must have the same value number since the = operation is really
/// inclusion, not overwrite), and value number nodes we receive points-to sets
/// before we value our own node.
/// The advantage of HU over HVN is that HU considers the inclusion property, so
/// that if you have
/// E = &D
/// E = &C
/// E = F
/// F = G
/// F = &D
/// G = F
/// HU will determine that G == F == E.  HVN will not, because it cannot prove
/// that the points to information ends up being the same because they all
/// receive &D from E anyway.

void Andersens::HVN() {
  DOUT << "Beginning HVN\n";
  // Build a predecessor graph.  This is like our constraint graph with the
  // edges going in the opposite direction, and there are edges for all the
  // constraints, instead of just copy constraints.  We also build implicit
  // edges for constraints are implied but not explicit.  I.E for the constraint
  // a = &b, we add implicit edges *a = b.  This helps us capture more cycles
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    Constraint &C = Constraints[i];
    if (C.Type == Constraint::AddressOf) {
      GraphNodes[C.Src].AddressTaken = true;
      GraphNodes[C.Src].Direct = false;

      // Dest = &src edge
      unsigned AdrNode = C.Src + FirstAdrNode;
      if (!GraphNodes[C.Dest].PredEdges)
        GraphNodes[C.Dest].PredEdges = new SparseBitVector<>;
      GraphNodes[C.Dest].PredEdges->set(AdrNode);

      // *Dest = src edge
      unsigned RefNode = C.Dest + FirstRefNode;
      if (!GraphNodes[RefNode].ImplicitPredEdges)
        GraphNodes[RefNode].ImplicitPredEdges = new SparseBitVector<>;
      GraphNodes[RefNode].ImplicitPredEdges->set(C.Src);
    } else if (C.Type == Constraint::Load) {
      if (C.Offset == 0) {
        // dest = *src edge
        if (!GraphNodes[C.Dest].PredEdges)
          GraphNodes[C.Dest].PredEdges = new SparseBitVector<>;
        GraphNodes[C.Dest].PredEdges->set(C.Src + FirstRefNode);
      } else {
        GraphNodes[C.Dest].Direct = false;
      }
    } else if (C.Type == Constraint::Store) {
      if (C.Offset == 0) {
        // *dest = src edge
        unsigned RefNode = C.Dest + FirstRefNode;
        if (!GraphNodes[RefNode].PredEdges)
          GraphNodes[RefNode].PredEdges = new SparseBitVector<>;
        GraphNodes[RefNode].PredEdges->set(C.Src);
      }
    } else {
      // Dest = Src edge and *Dest = *Src edge
      if (!GraphNodes[C.Dest].PredEdges)
        GraphNodes[C.Dest].PredEdges = new SparseBitVector<>;
      GraphNodes[C.Dest].PredEdges->set(C.Src);
      unsigned RefNode = C.Dest + FirstRefNode;
      if (!GraphNodes[RefNode].ImplicitPredEdges)
        GraphNodes[RefNode].ImplicitPredEdges = new SparseBitVector<>;
      GraphNodes[RefNode].ImplicitPredEdges->set(C.Src + FirstRefNode);
    }
  }
  PEClass = 1;
  // Do SCC finding first to condense our predecessor graph
  DFSNumber = 0;
  Node2DFS.insert(Node2DFS.begin(), GraphNodes.size(), 0);
  Node2Deleted.insert(Node2Deleted.begin(), GraphNodes.size(), false);
  Node2Visited.insert(Node2Visited.begin(), GraphNodes.size(), false);

  for (unsigned i = 0; i < FirstRefNode; ++i) {
    unsigned Node = VSSCCRep[i];
    if (!Node2Visited[Node])
      HVNValNum(Node);
  }
  for (BitVectorMap::iterator Iter = Set2PEClass.begin();
       Iter != Set2PEClass.end();
       ++Iter)
    delete Iter->first;
  Set2PEClass.clear();
  Node2DFS.clear();
  Node2Deleted.clear();
  Node2Visited.clear();
  DOUT << "Finished HVN\n";

}

/// This is the workhorse of HVN value numbering. We combine SCC finding at the
/// same time because it's easy.
void Andersens::HVNValNum(unsigned NodeIndex) {
  unsigned MyDFS = DFSNumber++;
  Node *N = &GraphNodes[NodeIndex];
  Node2Visited[NodeIndex] = true;
  Node2DFS[NodeIndex] = MyDFS;

  // First process all our explicit edges
  if (N->PredEdges)
    for (SparseBitVector<>::iterator Iter = N->PredEdges->begin();
         Iter != N->PredEdges->end();
         ++Iter) {
      unsigned j = VSSCCRep[*Iter];
      if (!Node2Deleted[j]) {
        if (!Node2Visited[j])
          HVNValNum(j);
        if (Node2DFS[NodeIndex] > Node2DFS[j])
          Node2DFS[NodeIndex] = Node2DFS[j];
      }
    }

  // Now process all the implicit edges
  if (N->ImplicitPredEdges)
    for (SparseBitVector<>::iterator Iter = N->ImplicitPredEdges->begin();
         Iter != N->ImplicitPredEdges->end();
         ++Iter) {
      unsigned j = VSSCCRep[*Iter];
      if (!Node2Deleted[j]) {
        if (!Node2Visited[j])
          HVNValNum(j);
        if (Node2DFS[NodeIndex] > Node2DFS[j])
          Node2DFS[NodeIndex] = Node2DFS[j];
      }
    }

  // See if we found any cycles
  if (MyDFS == Node2DFS[NodeIndex]) {
    while (!SCCStack.empty() && Node2DFS[SCCStack.top()] >= MyDFS) {
      unsigned CycleNodeIndex = SCCStack.top();
      Node *CycleNode = &GraphNodes[CycleNodeIndex];
      VSSCCRep[CycleNodeIndex] = NodeIndex;
      // Unify the nodes
      N->Direct &= CycleNode->Direct;

      if (CycleNode->PredEdges) {
        if (!N->PredEdges)
          N->PredEdges = new SparseBitVector<>;
        *(N->PredEdges) |= CycleNode->PredEdges;
        delete CycleNode->PredEdges;
        CycleNode->PredEdges = NULL;
      }
      if (CycleNode->ImplicitPredEdges) {
        if (!N->ImplicitPredEdges)
          N->ImplicitPredEdges = new SparseBitVector<>;
        *(N->ImplicitPredEdges) |= CycleNode->ImplicitPredEdges;
        delete CycleNode->ImplicitPredEdges;
        CycleNode->ImplicitPredEdges = NULL;
      }

      SCCStack.pop();
    }

    Node2Deleted[NodeIndex] = true;

    if (!N->Direct) {
      GraphNodes[NodeIndex].PointerEquivLabel = PEClass++;
      return;
    }

    // Collect labels of successor nodes
    bool AllSame = true;
    unsigned First = ~0;
    SparseBitVector<> *Labels = new SparseBitVector<>;
    bool Used = false;

    if (N->PredEdges)
      for (SparseBitVector<>::iterator Iter = N->PredEdges->begin();
           Iter != N->PredEdges->end();
         ++Iter) {
        unsigned j = VSSCCRep[*Iter];
        unsigned Label = GraphNodes[j].PointerEquivLabel;
        // Ignore labels that are equal to us or non-pointers
        if (j == NodeIndex || Label == 0)
          continue;
        if (First == (unsigned)~0)
          First = Label;
        else if (First != Label)
          AllSame = false;
        Labels->set(Label);
    }

    // We either have a non-pointer, a copy of an existing node, or a new node.
    // Assign the appropriate pointer equivalence label.
    if (Labels->empty()) {
      GraphNodes[NodeIndex].PointerEquivLabel = 0;
    } else if (AllSame) {
      GraphNodes[NodeIndex].PointerEquivLabel = First;
    } else {
      GraphNodes[NodeIndex].PointerEquivLabel = Set2PEClass[Labels];
      if (GraphNodes[NodeIndex].PointerEquivLabel == 0) {
        unsigned EquivClass = PEClass++;
        Set2PEClass[Labels] = EquivClass;
        GraphNodes[NodeIndex].PointerEquivLabel = EquivClass;
        Used = true;
      }
    }
    if (!Used)
      delete Labels;
  } else {
    SCCStack.push(NodeIndex);
  }
}

/// The technique used here is described in "Exploiting Pointer and Location
/// Equivalence to Optimize Pointer Analysis. In the 14th International Static
/// Analysis Symposium (SAS), August 2007."  It is known as the "HU" algorithm,
/// and is equivalent to value numbering the collapsed constraint graph
/// including evaluating unions.
void Andersens::HU() {
  DOUT << "Beginning HU\n";
  // Build a predecessor graph.  This is like our constraint graph with the
  // edges going in the opposite direction, and there are edges for all the
  // constraints, instead of just copy constraints.  We also build implicit
  // edges for constraints are implied but not explicit.  I.E for the constraint
  // a = &b, we add implicit edges *a = b.  This helps us capture more cycles
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    Constraint &C = Constraints[i];
    if (C.Type == Constraint::AddressOf) {
      GraphNodes[C.Src].AddressTaken = true;
      GraphNodes[C.Src].Direct = false;

      GraphNodes[C.Dest].PointsTo->set(C.Src);
      // *Dest = src edge
      unsigned RefNode = C.Dest + FirstRefNode;
      if (!GraphNodes[RefNode].ImplicitPredEdges)
        GraphNodes[RefNode].ImplicitPredEdges = new SparseBitVector<>;
      GraphNodes[RefNode].ImplicitPredEdges->set(C.Src);
      GraphNodes[C.Src].PointedToBy->set(C.Dest);
    } else if (C.Type == Constraint::Load) {
      if (C.Offset == 0) {
        // dest = *src edge
        if (!GraphNodes[C.Dest].PredEdges)
          GraphNodes[C.Dest].PredEdges = new SparseBitVector<>;
        GraphNodes[C.Dest].PredEdges->set(C.Src + FirstRefNode);
      } else {
        GraphNodes[C.Dest].Direct = false;
      }
    } else if (C.Type == Constraint::Store) {
      if (C.Offset == 0) {
        // *dest = src edge
        unsigned RefNode = C.Dest + FirstRefNode;
        if (!GraphNodes[RefNode].PredEdges)
          GraphNodes[RefNode].PredEdges = new SparseBitVector<>;
        GraphNodes[RefNode].PredEdges->set(C.Src);
      }
    } else {
      // Dest = Src edge and *Dest = *Src edg
      if (!GraphNodes[C.Dest].PredEdges)
        GraphNodes[C.Dest].PredEdges = new SparseBitVector<>;
      GraphNodes[C.Dest].PredEdges->set(C.Src);
      unsigned RefNode = C.Dest + FirstRefNode;
      if (!GraphNodes[RefNode].ImplicitPredEdges)
        GraphNodes[RefNode].ImplicitPredEdges = new SparseBitVector<>;
      GraphNodes[RefNode].ImplicitPredEdges->set(C.Src + FirstRefNode);
    }
  }
  PEClass = 1;
  // Do SCC finding first to condense our predecessor graph
  DFSNumber = 0;
  Node2DFS.insert(Node2DFS.begin(), GraphNodes.size(), 0);
  Node2Deleted.insert(Node2Deleted.begin(), GraphNodes.size(), false);
  Node2Visited.insert(Node2Visited.begin(), GraphNodes.size(), false);

  for (unsigned i = 0; i < FirstRefNode; ++i) {
    if (FindNode(i) == i) {
      unsigned Node = VSSCCRep[i];
      if (!Node2Visited[Node])
        Condense(Node);
    }
  }

  // Reset tables for actual labeling
  Node2DFS.clear();
  Node2Visited.clear();
  Node2Deleted.clear();
  // Pre-grow our densemap so that we don't get really bad behavior
  Set2PEClass.resize(GraphNodes.size());

  // Visit the condensed graph and generate pointer equivalence labels.
  Node2Visited.insert(Node2Visited.begin(), GraphNodes.size(), false);
  for (unsigned i = 0; i < FirstRefNode; ++i) {
    if (FindNode(i) == i) {
      unsigned Node = VSSCCRep[i];
      if (!Node2Visited[Node])
        HUValNum(Node);
    }
  }
  // PEClass nodes will be deleted by the deleting of N->PointsTo in our caller.
  Set2PEClass.clear();
  DOUT << "Finished HU\n";
}


/// Implementation of standard Tarjan SCC algorithm as modified by Nuutilla.
void Andersens::Condense(unsigned NodeIndex) {
  unsigned MyDFS = DFSNumber++;
  Node *N = &GraphNodes[NodeIndex];
  Node2Visited[NodeIndex] = true;
  Node2DFS[NodeIndex] = MyDFS;

  // First process all our explicit edges
  if (N->PredEdges)
    for (SparseBitVector<>::iterator Iter = N->PredEdges->begin();
         Iter != N->PredEdges->end();
         ++Iter) {
      unsigned j = VSSCCRep[*Iter];
      if (!Node2Deleted[j]) {
        if (!Node2Visited[j])
          Condense(j);
        if (Node2DFS[NodeIndex] > Node2DFS[j])
          Node2DFS[NodeIndex] = Node2DFS[j];
      }
    }

  // Now process all the implicit edges
  if (N->ImplicitPredEdges)
    for (SparseBitVector<>::iterator Iter = N->ImplicitPredEdges->begin();
         Iter != N->ImplicitPredEdges->end();
         ++Iter) {
      unsigned j = VSSCCRep[*Iter];
      if (!Node2Deleted[j]) {
        if (!Node2Visited[j])
          Condense(j);
        if (Node2DFS[NodeIndex] > Node2DFS[j])
          Node2DFS[NodeIndex] = Node2DFS[j];
      }
    }

  // See if we found any cycles
  if (MyDFS == Node2DFS[NodeIndex]) {
    while (!SCCStack.empty() && Node2DFS[SCCStack.top()] >= MyDFS) {
      unsigned CycleNodeIndex = SCCStack.top();
      Node *CycleNode = &GraphNodes[CycleNodeIndex];
      VSSCCRep[CycleNodeIndex] = NodeIndex;
      // Unify the nodes
      N->Direct &= CycleNode->Direct;

      *(N->PointsTo) |= CycleNode->PointsTo;
      delete CycleNode->PointsTo;
      CycleNode->PointsTo = NULL;
      if (CycleNode->PredEdges) {
        if (!N->PredEdges)
          N->PredEdges = new SparseBitVector<>;
        *(N->PredEdges) |= CycleNode->PredEdges;
        delete CycleNode->PredEdges;
        CycleNode->PredEdges = NULL;
      }
      if (CycleNode->ImplicitPredEdges) {
        if (!N->ImplicitPredEdges)
          N->ImplicitPredEdges = new SparseBitVector<>;
        *(N->ImplicitPredEdges) |= CycleNode->ImplicitPredEdges;
        delete CycleNode->ImplicitPredEdges;
        CycleNode->ImplicitPredEdges = NULL;
      }
      SCCStack.pop();
    }

    Node2Deleted[NodeIndex] = true;

    // Set up number of incoming edges for other nodes
    if (N->PredEdges)
      for (SparseBitVector<>::iterator Iter = N->PredEdges->begin();
           Iter != N->PredEdges->end();
           ++Iter)
        ++GraphNodes[VSSCCRep[*Iter]].NumInEdges;
  } else {
    SCCStack.push(NodeIndex);
  }
}

void Andersens::HUValNum(unsigned NodeIndex) {
  Node *N = &GraphNodes[NodeIndex];
  Node2Visited[NodeIndex] = true;

  // Eliminate dereferences of non-pointers for those non-pointers we have
  // already identified.  These are ref nodes whose non-ref node:
  // 1. Has already been visited determined to point to nothing (and thus, a
  // dereference of it must point to nothing)
  // 2. Any direct node with no predecessor edges in our graph and with no
  // points-to set (since it can't point to anything either, being that it
  // receives no points-to sets and has none).
  if (NodeIndex >= FirstRefNode) {
    unsigned j = VSSCCRep[FindNode(NodeIndex - FirstRefNode)];
    if ((Node2Visited[j] && !GraphNodes[j].PointerEquivLabel)
        || (GraphNodes[j].Direct && !GraphNodes[j].PredEdges
            && GraphNodes[j].PointsTo->empty())){
      return;
    }
  }
    // Process all our explicit edges
  if (N->PredEdges)
    for (SparseBitVector<>::iterator Iter = N->PredEdges->begin();
         Iter != N->PredEdges->end();
         ++Iter) {
      unsigned j = VSSCCRep[*Iter];
      if (!Node2Visited[j])
        HUValNum(j);

      // If this edge turned out to be the same as us, or got no pointer
      // equivalence label (and thus points to nothing) , just decrement our
      // incoming edges and continue.
      if (j == NodeIndex || GraphNodes[j].PointerEquivLabel == 0) {
        --GraphNodes[j].NumInEdges;
        continue;
      }

      *(N->PointsTo) |= GraphNodes[j].PointsTo;

      // If we didn't end up storing this in the hash, and we're done with all
      // the edges, we don't need the points-to set anymore.
      --GraphNodes[j].NumInEdges;
      if (!GraphNodes[j].NumInEdges && !GraphNodes[j].StoredInHash) {
        delete GraphNodes[j].PointsTo;
        GraphNodes[j].PointsTo = NULL;
      }
    }
  // If this isn't a direct node, generate a fresh variable.
  if (!N->Direct) {
    N->PointsTo->set(FirstRefNode + NodeIndex);
  }

  // See If we have something equivalent to us, if not, generate a new
  // equivalence class.
  if (N->PointsTo->empty()) {
    delete N->PointsTo;
    N->PointsTo = NULL;
  } else {
    if (N->Direct) {
      N->PointerEquivLabel = Set2PEClass[N->PointsTo];
      if (N->PointerEquivLabel == 0) {
        unsigned EquivClass = PEClass++;
        N->StoredInHash = true;
        Set2PEClass[N->PointsTo] = EquivClass;
        N->PointerEquivLabel = EquivClass;
      }
    } else {
      N->PointerEquivLabel = PEClass++;
    }
  }
}

/// Rewrite our list of constraints so that pointer equivalent nodes are
/// replaced by their the pointer equivalence class representative.
void Andersens::RewriteConstraints() {
  std::vector<Constraint> NewConstraints;
  DenseSet<Constraint, ConstraintKeyInfo> Seen;

  PEClass2Node.clear();
  PENLEClass2Node.clear();

  // We may have from 1 to Graphnodes + 1 equivalence classes.
  PEClass2Node.insert(PEClass2Node.begin(), GraphNodes.size() + 1, -1);
  PENLEClass2Node.insert(PENLEClass2Node.begin(), GraphNodes.size() + 1, -1);

  // Rewrite constraints, ignoring non-pointer constraints, uniting equivalent
  // nodes, and rewriting constraints to use the representative nodes.
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    Constraint &C = Constraints[i];
    unsigned RHSNode = FindNode(C.Src);
    unsigned LHSNode = FindNode(C.Dest);
    unsigned RHSLabel = GraphNodes[VSSCCRep[RHSNode]].PointerEquivLabel;
    unsigned LHSLabel = GraphNodes[VSSCCRep[LHSNode]].PointerEquivLabel;

    // First we try to eliminate constraints for things we can prove don't point
    // to anything.
    if (LHSLabel == 0) {
      DEBUG(PrintNode(&GraphNodes[LHSNode]));
      DOUT << " is a non-pointer, ignoring constraint.\n";
      continue;
    }
    if (RHSLabel == 0) {
      DEBUG(PrintNode(&GraphNodes[RHSNode]));
      DOUT << " is a non-pointer, ignoring constraint.\n";
      continue;
    }
    // This constraint may be useless, and it may become useless as we translate
    // it.
    if (C.Src == C.Dest && C.Type == Constraint::Copy)
      continue;

    C.Src = FindEquivalentNode(RHSNode, RHSLabel);
    C.Dest = FindEquivalentNode(FindNode(LHSNode), LHSLabel);
    if ((C.Src == C.Dest && C.Type == Constraint::Copy)
        || Seen.count(C))
      continue;

    Seen.insert(C);
    NewConstraints.push_back(C);
  }
  Constraints.swap(NewConstraints);
  PEClass2Node.clear();
}

/// See if we have a node that is pointer equivalent to the one being asked
/// about, and if so, unite them and return the equivalent node.  Otherwise,
/// return the original node.
unsigned Andersens::FindEquivalentNode(unsigned NodeIndex,
                                       unsigned NodeLabel) {
  if (!GraphNodes[NodeIndex].AddressTaken) {
    if (PEClass2Node[NodeLabel] != -1) {
      // We found an existing node with the same pointer label, so unify them.
      // We specifically request that Union-By-Rank not be used so that
      // PEClass2Node[NodeLabel] U= NodeIndex and not the other way around.
      return UniteNodes(PEClass2Node[NodeLabel], NodeIndex, false);
    } else {
      PEClass2Node[NodeLabel] = NodeIndex;
      PENLEClass2Node[NodeLabel] = NodeIndex;
    }
  } else if (PENLEClass2Node[NodeLabel] == -1) {
    PENLEClass2Node[NodeLabel] = NodeIndex;
  }

  return NodeIndex;
}

void Andersens::PrintLabels() const {
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    if (i < FirstRefNode) {
      PrintNode(&GraphNodes[i]);
    } else if (i < FirstAdrNode) {
      DOUT << "REF(";
      PrintNode(&GraphNodes[i-FirstRefNode]);
      DOUT <<")";
    } else {
      DOUT << "ADR(";
      PrintNode(&GraphNodes[i-FirstAdrNode]);
      DOUT <<")";
    }

    DOUT << " has pointer label " << GraphNodes[i].PointerEquivLabel
         << " and SCC rep " << VSSCCRep[i]
         << " and is " << (GraphNodes[i].Direct ? "Direct" : "Not direct")
         << "\n";
  }
}

/// The technique used here is described in "The Ant and the
/// Grasshopper: Fast and Accurate Pointer Analysis for Millions of
/// Lines of Code. In Programming Language Design and Implementation
/// (PLDI), June 2007." It is known as the "HCD" (Hybrid Cycle
/// Detection) algorithm. It is called a hybrid because it performs an
/// offline analysis and uses its results during the solving (online)
/// phase. This is just the offline portion; the results of this
/// operation are stored in SDT and are later used in SolveContraints()
/// and UniteNodes().
void Andersens::HCD() {
  DOUT << "Starting HCD.\n";
  HCDSCCRep.resize(GraphNodes.size());

  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    GraphNodes[i].Edges = new SparseBitVector<>;
    HCDSCCRep[i] = i;
  }

  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    Constraint &C = Constraints[i];
    assert (C.Src < GraphNodes.size() && C.Dest < GraphNodes.size());
    if (C.Type == Constraint::AddressOf) {
      continue;
    } else if (C.Type == Constraint::Load) {
      if( C.Offset == 0 )
        GraphNodes[C.Dest].Edges->set(C.Src + FirstRefNode);
    } else if (C.Type == Constraint::Store) {
      if( C.Offset == 0 )
        GraphNodes[C.Dest + FirstRefNode].Edges->set(C.Src);
    } else {
      GraphNodes[C.Dest].Edges->set(C.Src);
    }
  }

  Node2DFS.insert(Node2DFS.begin(), GraphNodes.size(), 0);
  Node2Deleted.insert(Node2Deleted.begin(), GraphNodes.size(), false);
  Node2Visited.insert(Node2Visited.begin(), GraphNodes.size(), false);
  SDT.insert(SDT.begin(), GraphNodes.size() / 2, -1);

  DFSNumber = 0;
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    unsigned Node = HCDSCCRep[i];
    if (!Node2Deleted[Node])
      Search(Node);
  }

  for (unsigned i = 0; i < GraphNodes.size(); ++i)
    if (GraphNodes[i].Edges != NULL) {
      delete GraphNodes[i].Edges;
      GraphNodes[i].Edges = NULL;
    }

  while( !SCCStack.empty() )
    SCCStack.pop();

  Node2DFS.clear();
  Node2Visited.clear();
  Node2Deleted.clear();
  HCDSCCRep.clear();
  DOUT << "HCD complete.\n";
}

// Component of HCD: 
// Use Nuutila's variant of Tarjan's algorithm to detect
// Strongly-Connected Components (SCCs). For non-trivial SCCs
// containing ref nodes, insert the appropriate information in SDT.
void Andersens::Search(unsigned Node) {
  unsigned MyDFS = DFSNumber++;

  Node2Visited[Node] = true;
  Node2DFS[Node] = MyDFS;

  for (SparseBitVector<>::iterator Iter = GraphNodes[Node].Edges->begin(),
                                   End  = GraphNodes[Node].Edges->end();
       Iter != End;
       ++Iter) {
    unsigned J = HCDSCCRep[*Iter];
    assert(GraphNodes[J].isRep() && "Debug check; must be representative");
    if (!Node2Deleted[J]) {
      if (!Node2Visited[J])
        Search(J);
      if (Node2DFS[Node] > Node2DFS[J])
        Node2DFS[Node] = Node2DFS[J];
    }
  }

  if( MyDFS != Node2DFS[Node] ) {
    SCCStack.push(Node);
    return;
  }

  // This node is the root of a SCC, so process it.
  //
  // If the SCC is "non-trivial" (not a singleton) and contains a reference 
  // node, we place this SCC into SDT.  We unite the nodes in any case.
  if (!SCCStack.empty() && Node2DFS[SCCStack.top()] >= MyDFS) {
    SparseBitVector<> SCC;

    SCC.set(Node);

    bool Ref = (Node >= FirstRefNode);

    Node2Deleted[Node] = true;

    do {
      unsigned P = SCCStack.top(); SCCStack.pop();
      Ref |= (P >= FirstRefNode);
      SCC.set(P);
      HCDSCCRep[P] = Node;
    } while (!SCCStack.empty() && Node2DFS[SCCStack.top()] >= MyDFS);

    if (Ref) {
      unsigned Rep = SCC.find_first();
      assert(Rep < FirstRefNode && "The SCC didn't have a non-Ref node!");

      SparseBitVector<>::iterator i = SCC.begin();

      // Skip over the non-ref nodes
      while( *i < FirstRefNode )
        ++i;

      while( i != SCC.end() )
        SDT[ (*i++) - FirstRefNode ] = Rep;
    }
  }
}


/// Optimize the constraints by performing offline variable substitution and
/// other optimizations.
void Andersens::OptimizeConstraints() {
  DOUT << "Beginning constraint optimization\n";

  SDTActive = false;

  // Function related nodes need to stay in the same relative position and can't
  // be location equivalent.
  for (std::map<unsigned, unsigned>::iterator Iter = MaxK.begin();
       Iter != MaxK.end();
       ++Iter) {
    for (unsigned i = Iter->first;
         i != Iter->first + Iter->second;
         ++i) {
      GraphNodes[i].AddressTaken = true;
      GraphNodes[i].Direct = false;
    }
  }

  ClumpAddressTaken();
  FirstRefNode = GraphNodes.size();
  FirstAdrNode = FirstRefNode + GraphNodes.size();
  GraphNodes.insert(GraphNodes.end(), 2 * GraphNodes.size(),
                    Node(false));
  VSSCCRep.resize(GraphNodes.size());
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    VSSCCRep[i] = i;
  }
  HVN();
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    Node *N = &GraphNodes[i];
    delete N->PredEdges;
    N->PredEdges = NULL;
    delete N->ImplicitPredEdges;
    N->ImplicitPredEdges = NULL;
  }
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa-labels"
  DEBUG(PrintLabels());
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa"
  RewriteConstraints();
  // Delete the adr nodes.
  GraphNodes.resize(FirstRefNode * 2);

  // Now perform HU
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    Node *N = &GraphNodes[i];
    if (FindNode(i) == i) {
      N->PointsTo = new SparseBitVector<>;
      N->PointedToBy = new SparseBitVector<>;
      // Reset our labels
    }
    VSSCCRep[i] = i;
    N->PointerEquivLabel = 0;
  }
  HU();
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa-labels"
  DEBUG(PrintLabels());
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa"
  RewriteConstraints();
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    if (FindNode(i) == i) {
      Node *N = &GraphNodes[i];
      delete N->PointsTo;
      N->PointsTo = NULL;
      delete N->PredEdges;
      N->PredEdges = NULL;
      delete N->ImplicitPredEdges;
      N->ImplicitPredEdges = NULL;
      delete N->PointedToBy;
      N->PointedToBy = NULL;
    }
  }

  // perform Hybrid Cycle Detection (HCD)
  HCD();
  SDTActive = true;

  // No longer any need for the upper half of GraphNodes (for ref nodes).
  GraphNodes.erase(GraphNodes.begin() + FirstRefNode, GraphNodes.end());

  // HCD complete.

  DOUT << "Finished constraint optimization\n";
  FirstRefNode = 0;
  FirstAdrNode = 0;
}

/// Unite pointer but not location equivalent variables, now that the constraint
/// graph is built.
void Andersens::UnitePointerEquivalences() {
  DOUT << "Uniting remaining pointer equivalences\n";
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    if (GraphNodes[i].AddressTaken && GraphNodes[i].isRep()) {
      unsigned Label = GraphNodes[i].PointerEquivLabel;

      if (Label && PENLEClass2Node[Label] != -1)
        UniteNodes(i, PENLEClass2Node[Label]);
    }
  }
  DOUT << "Finished remaining pointer equivalences\n";
  PENLEClass2Node.clear();
}

/// Create the constraint graph used for solving points-to analysis.
///
void Andersens::CreateConstraintGraph() {
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    Constraint &C = Constraints[i];
    assert (C.Src < GraphNodes.size() && C.Dest < GraphNodes.size());
    if (C.Type == Constraint::AddressOf)
      GraphNodes[C.Dest].PointsTo->set(C.Src);
    else if (C.Type == Constraint::Load)
      GraphNodes[C.Src].Constraints.push_back(C);
    else if (C.Type == Constraint::Store)
      GraphNodes[C.Dest].Constraints.push_back(C);
    else if (C.Offset != 0)
      GraphNodes[C.Src].Constraints.push_back(C);
    else
      GraphNodes[C.Src].Edges->set(C.Dest);
  }
}

// Perform DFS and cycle detection.
bool Andersens::QueryNode(unsigned Node) {
  assert(GraphNodes[Node].isRep() && "Querying a non-rep node");
  unsigned OurDFS = ++DFSNumber;
  SparseBitVector<> ToErase;
  SparseBitVector<> NewEdges;
  Tarjan2DFS[Node] = OurDFS;

  // Changed denotes a change from a recursive call that we will bubble up.
  // Merged is set if we actually merge a node ourselves.
  bool Changed = false, Merged = false;

  for (SparseBitVector<>::iterator bi = GraphNodes[Node].Edges->begin();
       bi != GraphNodes[Node].Edges->end();
       ++bi) {
    unsigned RepNode = FindNode(*bi);
    // If this edge points to a non-representative node but we are
    // already planning to add an edge to its representative, we have no
    // need for this edge anymore.
    if (RepNode != *bi && NewEdges.test(RepNode)){
      ToErase.set(*bi);
      continue;
    }

    // Continue about our DFS.
    if (!Tarjan2Deleted[RepNode]){
      if (Tarjan2DFS[RepNode] == 0) {
        Changed |= QueryNode(RepNode);
        // May have been changed by QueryNode
        RepNode = FindNode(RepNode);
      }
      if (Tarjan2DFS[RepNode] < Tarjan2DFS[Node])
        Tarjan2DFS[Node] = Tarjan2DFS[RepNode];
    }

    // We may have just discovered that this node is part of a cycle, in
    // which case we can also erase it.
    if (RepNode != *bi) {
      ToErase.set(*bi);
      NewEdges.set(RepNode);
    }
  }

  GraphNodes[Node].Edges->intersectWithComplement(ToErase);
  GraphNodes[Node].Edges |= NewEdges;

  // If this node is a root of a non-trivial SCC, place it on our 
  // worklist to be processed.
  if (OurDFS == Tarjan2DFS[Node]) {
    while (!SCCStack.empty() && Tarjan2DFS[SCCStack.top()] >= OurDFS) {
      Node = UniteNodes(Node, SCCStack.top());

      SCCStack.pop();
      Merged = true;
    }
    Tarjan2Deleted[Node] = true;

    if (Merged)
      NextWL->insert(&GraphNodes[Node]);
  } else {
    SCCStack.push(Node);
  }

  return(Changed | Merged);
}

/// SolveConstraints - This stage iteratively processes the constraints list
/// propagating constraints (adding edges to the Nodes in the points-to graph)
/// until a fixed point is reached.
///
/// We use a variant of the technique called "Lazy Cycle Detection", which is
/// described in "The Ant and the Grasshopper: Fast and Accurate Pointer
/// Analysis for Millions of Lines of Code. In Programming Language Design and
/// Implementation (PLDI), June 2007."
/// The paper describes performing cycle detection one node at a time, which can
/// be expensive if there are no cycles, but there are long chains of nodes that
/// it heuristically believes are cycles (because it will DFS from each node
/// without state from previous nodes).
/// Instead, we use the heuristic to build a worklist of nodes to check, then
/// cycle detect them all at the same time to do this more cheaply.  This
/// catches cycles slightly later than the original technique did, but does it
/// make significantly cheaper.

void Andersens::SolveConstraints() {
  CurrWL = &w1;
  NextWL = &w2;

  OptimizeConstraints();
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa-constraints"
      DEBUG(PrintConstraints());
#undef DEBUG_TYPE
#define DEBUG_TYPE "anders-aa"

  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    Node *N = &GraphNodes[i];
    N->PointsTo = new SparseBitVector<>;
    N->OldPointsTo = new SparseBitVector<>;
    N->Edges = new SparseBitVector<>;
  }
  CreateConstraintGraph();
  UnitePointerEquivalences();
  assert(SCCStack.empty() && "SCC Stack should be empty by now!");
  Node2DFS.clear();
  Node2Deleted.clear();
  Node2DFS.insert(Node2DFS.begin(), GraphNodes.size(), 0);
  Node2Deleted.insert(Node2Deleted.begin(), GraphNodes.size(), false);
  DFSNumber = 0;
  DenseSet<Constraint, ConstraintKeyInfo> Seen;
  DenseSet<std::pair<unsigned,unsigned>, PairKeyInfo> EdgesChecked;

  // Order graph and add initial nodes to work list.
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    Node *INode = &GraphNodes[i];

    // Add to work list if it's a representative and can contribute to the
    // calculation right now.
    if (INode->isRep() && !INode->PointsTo->empty()
        && (!INode->Edges->empty() || !INode->Constraints.empty())) {
      INode->Stamp();
      CurrWL->insert(INode);
    }
  }
  std::queue<unsigned int> TarjanWL;
#if !FULL_UNIVERSAL
  // "Rep and special variables" - in order for HCD to maintain conservative
  // results when !FULL_UNIVERSAL, we need to treat the special variables in
  // the same way that the !FULL_UNIVERSAL tweak does throughout the rest of
  // the analysis - it's ok to add edges from the special nodes, but never
  // *to* the special nodes.
  std::vector<unsigned int> RSV;
#endif
  while( !CurrWL->empty() ) {
    DOUT << "Starting iteration #" << ++NumIters << "\n";

    Node* CurrNode;
    unsigned CurrNodeIndex;

    // Actual cycle checking code.  We cycle check all of the lazy cycle
    // candidates from the last iteration in one go.
    if (!TarjanWL.empty()) {
      DFSNumber = 0;
      
      Tarjan2DFS.clear();
      Tarjan2Deleted.clear();
      while (!TarjanWL.empty()) {
        unsigned int ToTarjan = TarjanWL.front();
        TarjanWL.pop();
        if (!Tarjan2Deleted[ToTarjan]
            && GraphNodes[ToTarjan].isRep()
            && Tarjan2DFS[ToTarjan] == 0)
          QueryNode(ToTarjan);
      }
    }
    
    // Add to work list if it's a representative and can contribute to the
    // calculation right now.
    while( (CurrNode = CurrWL->pop()) != NULL ) {
      CurrNodeIndex = CurrNode - &GraphNodes[0];
      CurrNode->Stamp();
      
          
      // Figure out the changed points to bits
      SparseBitVector<> CurrPointsTo;
      CurrPointsTo.intersectWithComplement(CurrNode->PointsTo,
                                           CurrNode->OldPointsTo);
      if (CurrPointsTo.empty())
        continue;

      *(CurrNode->OldPointsTo) |= CurrPointsTo;

      // Check the offline-computed equivalencies from HCD.
      bool SCC = false;
      unsigned Rep;

      if (SDT[CurrNodeIndex] >= 0) {
        SCC = true;
        Rep = FindNode(SDT[CurrNodeIndex]);

#if !FULL_UNIVERSAL
        RSV.clear();
#endif
        for (SparseBitVector<>::iterator bi = CurrPointsTo.begin();
             bi != CurrPointsTo.end(); ++bi) {
          unsigned Node = FindNode(*bi);
#if !FULL_UNIVERSAL
          if (Node < NumberSpecialNodes) {
            RSV.push_back(Node);
            continue;
          }
#endif
          Rep = UniteNodes(Rep,Node);
        }
#if !FULL_UNIVERSAL
        RSV.push_back(Rep);
#endif

        NextWL->insert(&GraphNodes[Rep]);

        if ( ! CurrNode->isRep() )
          continue;
      }

      Seen.clear();

      /* Now process the constraints for this node.  */
      for (std::list<Constraint>::iterator li = CurrNode->Constraints.begin();
           li != CurrNode->Constraints.end(); ) {
        li->Src = FindNode(li->Src);
        li->Dest = FindNode(li->Dest);

        // Delete redundant constraints
        if( Seen.count(*li) ) {
          std::list<Constraint>::iterator lk = li; li++;

          CurrNode->Constraints.erase(lk);
          ++NumErased;
          continue;
        }
        Seen.insert(*li);

        // Src and Dest will be the vars we are going to process.
        // This may look a bit ugly, but what it does is allow us to process
        // both store and load constraints with the same code.
        // Load constraints say that every member of our RHS solution has K
        // added to it, and that variable gets an edge to LHS. We also union
        // RHS+K's solution into the LHS solution.
        // Store constraints say that every member of our LHS solution has K
        // added to it, and that variable gets an edge from RHS. We also union
        // RHS's solution into the LHS+K solution.
        unsigned *Src;
        unsigned *Dest;
        unsigned K = li->Offset;
        unsigned CurrMember;
        if (li->Type == Constraint::Load) {
          Src = &CurrMember;
          Dest = &li->Dest;
        } else if (li->Type == Constraint::Store) {
          Src = &li->Src;
          Dest = &CurrMember;
        } else {
          // TODO Handle offseted copy constraint
          li++;
          continue;
        }

        // See if we can use Hybrid Cycle Detection (that is, check
        // if it was a statically detected offline equivalence that
        // involves pointers; if so, remove the redundant constraints).
        if( SCC && K == 0 ) {
#if FULL_UNIVERSAL
          CurrMember = Rep;

          if (GraphNodes[*Src].Edges->test_and_set(*Dest))
            if (GraphNodes[*Dest].PointsTo |= *(GraphNodes[*Src].PointsTo))
              NextWL->insert(&GraphNodes[*Dest]);
#else
          for (unsigned i=0; i < RSV.size(); ++i) {
            CurrMember = RSV[i];

            if (*Dest < NumberSpecialNodes)
              continue;
            if (GraphNodes[*Src].Edges->test_and_set(*Dest))
              if (GraphNodes[*Dest].PointsTo |= *(GraphNodes[*Src].PointsTo))
                NextWL->insert(&GraphNodes[*Dest]);
          }
#endif
          // since all future elements of the points-to set will be
          // equivalent to the current ones, the complex constraints
          // become redundant.
          //
          std::list<Constraint>::iterator lk = li; li++;
#if !FULL_UNIVERSAL
          // In this case, we can still erase the constraints when the
          // elements of the points-to sets are referenced by *Dest,
          // but not when they are referenced by *Src (i.e. for a Load
          // constraint). This is because if another special variable is
          // put into the points-to set later, we still need to add the
          // new edge from that special variable.
          if( lk->Type != Constraint::Load)
#endif
          GraphNodes[CurrNodeIndex].Constraints.erase(lk);
        } else {
          const SparseBitVector<> &Solution = CurrPointsTo;

          for (SparseBitVector<>::iterator bi = Solution.begin();
               bi != Solution.end();
               ++bi) {
            CurrMember = *bi;

            // Need to increment the member by K since that is where we are
            // supposed to copy to/from.  Note that in positive weight cycles,
            // which occur in address taking of fields, K can go past
            // MaxK[CurrMember] elements, even though that is all it could point
            // to.
            if (K > 0 && K > MaxK[CurrMember])
              continue;
            else
              CurrMember = FindNode(CurrMember + K);

            // Add an edge to the graph, so we can just do regular
            // bitmap ior next time.  It may also let us notice a cycle.
#if !FULL_UNIVERSAL
            if (*Dest < NumberSpecialNodes)
              continue;
#endif
            if (GraphNodes[*Src].Edges->test_and_set(*Dest))
              if (GraphNodes[*Dest].PointsTo |= *(GraphNodes[*Src].PointsTo))
                NextWL->insert(&GraphNodes[*Dest]);

          }
          li++;
        }
      }
      SparseBitVector<> NewEdges;
      SparseBitVector<> ToErase;

      // Now all we have left to do is propagate points-to info along the
      // edges, erasing the redundant edges.
      for (SparseBitVector<>::iterator bi = CurrNode->Edges->begin();
           bi != CurrNode->Edges->end();
           ++bi) {

        unsigned DestVar = *bi;
        unsigned Rep = FindNode(DestVar);

        // If we ended up with this node as our destination, or we've already
        // got an edge for the representative, delete the current edge.
        if (Rep == CurrNodeIndex ||
            (Rep != DestVar && NewEdges.test(Rep))) {
            ToErase.set(DestVar);
            continue;
        }
        
        std::pair<unsigned,unsigned> edge(CurrNodeIndex,Rep);
        
        // This is where we do lazy cycle detection.
        // If this is a cycle candidate (equal points-to sets and this
        // particular edge has not been cycle-checked previously), add to the
        // list to check for cycles on the next iteration.
        if (!EdgesChecked.count(edge) &&
            *(GraphNodes[Rep].PointsTo) == *(CurrNode->PointsTo)) {
          EdgesChecked.insert(edge);
          TarjanWL.push(Rep);
        }
        // Union the points-to sets into the dest
#if !FULL_UNIVERSAL
        if (Rep >= NumberSpecialNodes)
#endif
        if (GraphNodes[Rep].PointsTo |= CurrPointsTo) {
          NextWL->insert(&GraphNodes[Rep]);
        }
        // If this edge's destination was collapsed, rewrite the edge.
        if (Rep != DestVar) {
          ToErase.set(DestVar);
          NewEdges.set(Rep);
        }
      }
      CurrNode->Edges->intersectWithComplement(ToErase);
      CurrNode->Edges |= NewEdges;
    }

    // Switch to other work list.
    WorkList* t = CurrWL; CurrWL = NextWL; NextWL = t;
  }


  Node2DFS.clear();
  Node2Deleted.clear();
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    Node *N = &GraphNodes[i];
    delete N->OldPointsTo;
    delete N->Edges;
  }
  SDTActive = false;
  SDT.clear();
}

//===----------------------------------------------------------------------===//
//                               Union-Find
//===----------------------------------------------------------------------===//

// Unite nodes First and Second, returning the one which is now the
// representative node.  First and Second are indexes into GraphNodes
unsigned Andersens::UniteNodes(unsigned First, unsigned Second,
                               bool UnionByRank) {
  assert (First < GraphNodes.size() && Second < GraphNodes.size() &&
          "Attempting to merge nodes that don't exist");

  Node *FirstNode = &GraphNodes[First];
  Node *SecondNode = &GraphNodes[Second];

  assert (SecondNode->isRep() && FirstNode->isRep() &&
          "Trying to unite two non-representative nodes!");
  if (First == Second)
    return First;

  if (UnionByRank) {
    int RankFirst  = (int) FirstNode ->NodeRep;
    int RankSecond = (int) SecondNode->NodeRep;

    // Rank starts at -1 and gets decremented as it increases.
    // Translation: higher rank, lower NodeRep value, which is always negative.
    if (RankFirst > RankSecond) {
      unsigned t = First; First = Second; Second = t;
      Node* tp = FirstNode; FirstNode = SecondNode; SecondNode = tp;
    } else if (RankFirst == RankSecond) {
      FirstNode->NodeRep = (unsigned) (RankFirst - 1);
    }
  }

  SecondNode->NodeRep = First;
#if !FULL_UNIVERSAL
  if (First >= NumberSpecialNodes)
#endif
  if (FirstNode->PointsTo && SecondNode->PointsTo)
    FirstNode->PointsTo |= *(SecondNode->PointsTo);
  if (FirstNode->Edges && SecondNode->Edges)
    FirstNode->Edges |= *(SecondNode->Edges);
  if (!SecondNode->Constraints.empty())
    FirstNode->Constraints.splice(FirstNode->Constraints.begin(),
                                  SecondNode->Constraints);
  if (FirstNode->OldPointsTo) {
    delete FirstNode->OldPointsTo;
    FirstNode->OldPointsTo = new SparseBitVector<>;
  }

  // Destroy interesting parts of the merged-from node.
  delete SecondNode->OldPointsTo;
  delete SecondNode->Edges;
  delete SecondNode->PointsTo;
  SecondNode->Edges = NULL;
  SecondNode->PointsTo = NULL;
  SecondNode->OldPointsTo = NULL;

  NumUnified++;
  DOUT << "Unified Node ";
  DEBUG(PrintNode(FirstNode));
  DOUT << " and Node ";
  DEBUG(PrintNode(SecondNode));
  DOUT << "\n";

  if (SDTActive)
    if (SDT[Second] >= 0) {
      if (SDT[First] < 0)
        SDT[First] = SDT[Second];
      else {
        UniteNodes( FindNode(SDT[First]), FindNode(SDT[Second]) );
        First = FindNode(First);
      }
    }

  return First;
}

// Find the index into GraphNodes of the node representing Node, performing
// path compression along the way
unsigned Andersens::FindNode(unsigned NodeIndex) {
  assert (NodeIndex < GraphNodes.size()
          && "Attempting to find a node that can't exist");
  Node *N = &GraphNodes[NodeIndex];
  if (N->isRep())
    return NodeIndex;
  else
    return (N->NodeRep = FindNode(N->NodeRep));
}

// Find the index into GraphNodes of the node representing Node, 
// don't perform path compression along the way (for Print)
unsigned Andersens::FindNode(unsigned NodeIndex) const {
  assert (NodeIndex < GraphNodes.size()
          && "Attempting to find a node that can't exist");
  const Node *N = &GraphNodes[NodeIndex];
  if (N->isRep())
    return NodeIndex;
  else
    return FindNode(N->NodeRep);
}

//===----------------------------------------------------------------------===//
//                               Debugging Output
//===----------------------------------------------------------------------===//

void Andersens::PrintNode(const Node *N) const {
  if (N == &GraphNodes[UniversalSet]) {
    cerr << "<universal>";
    return;
  } else if (N == &GraphNodes[NullPtr]) {
    cerr << "<nullptr>";
    return;
  } else if (N == &GraphNodes[NullObject]) {
    cerr << "<null>";
    return;
  }
  if (!N->getValue()) {
    cerr << "artificial" << (intptr_t) N;
    return;
  }

  assert(N->getValue() != 0 && "Never set node label!");
  Value *V = N->getValue();
  if (Function *F = dyn_cast<Function>(V)) {
    if (isa<PointerType>(F->getFunctionType()->getReturnType()) &&
        N == &GraphNodes[getReturnNode(F)]) {
      cerr << F->getName() << ":retval";
      return;
    } else if (F->getFunctionType()->isVarArg() &&
               N == &GraphNodes[getVarargNode(F)]) {
      cerr << F->getName() << ":vararg";
      return;
    }
  }

  if (Instruction *I = dyn_cast<Instruction>(V))
    cerr << I->getParent()->getParent()->getName() << ":";
  else if (Argument *Arg = dyn_cast<Argument>(V))
    cerr << Arg->getParent()->getName() << ":";

  if (V->hasName())
    cerr << V->getName();
  else
    cerr << "(unnamed)";

  if (isa<GlobalValue>(V) || isa<AllocationInst>(V))
    if (N == &GraphNodes[getObject(V)])
      cerr << "<mem>";
}
void Andersens::PrintConstraint(const Constraint &C) const {
  if (C.Type == Constraint::Store) {
    cerr << "*";
    if (C.Offset != 0)
      cerr << "(";
  }
  PrintNode(&GraphNodes[C.Dest]);
  if (C.Type == Constraint::Store && C.Offset != 0)
    cerr << " + " << C.Offset << ")";
  cerr << " = ";
  if (C.Type == Constraint::Load) {
    cerr << "*";
    if (C.Offset != 0)
      cerr << "(";
  }
  else if (C.Type == Constraint::AddressOf)
    cerr << "&";
  PrintNode(&GraphNodes[C.Src]);
  if (C.Offset != 0 && C.Type != Constraint::Store)
    cerr << " + " << C.Offset;
  if (C.Type == Constraint::Load && C.Offset != 0)
    cerr << ")";
  cerr << "\n";
}

void Andersens::PrintConstraints() const {
  cerr << "Constraints:\n";

  for (unsigned i = 0, e = Constraints.size(); i != e; ++i)
    PrintConstraint(Constraints[i]);
}

void Andersens::PrintPointsToGraph() const {
  cerr << "Points-to graph:\n";
  for (unsigned i = 0, e = GraphNodes.size(); i != e; ++i) {
    const Node *N = &GraphNodes[i];
    if (FindNode(i) != i) {
      PrintNode(N);
      cerr << "\t--> same as ";
      PrintNode(&GraphNodes[FindNode(i)]);
      cerr << "\n";
    } else {
      cerr << "[" << (N->PointsTo->count()) << "] ";
      PrintNode(N);
      cerr << "\t--> ";

      bool first = true;
      for (SparseBitVector<>::iterator bi = N->PointsTo->begin();
           bi != N->PointsTo->end();
           ++bi) {
        if (!first)
          cerr << ", ";
        PrintNode(&GraphNodes[*bi]);
        first = false;
      }
      cerr << "\n";
    }
  }
}
