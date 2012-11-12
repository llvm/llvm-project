//===- CallGraph.h - Build a Module's call graph ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This interface is used to build and manipulate a call graph, which is a very
// useful tool for interprocedural optimization.
//
// Every function in a module is represented as a node in the call graph.  The
// callgraph node keeps track of which functions the are called by the function
// corresponding to the node.
//
// A call graph may contain nodes where the function that they correspond to is
// null.  These 'external' nodes are used to represent control flow that is not
// represented (or analyzable) in the module.  In particular, this analysis
// builds one external node such that:
//   1. All functions in the module without internal linkage will have edges
//      from this external node, indicating that they could be called by
//      functions outside of the module.
//   2. All functions whose address is used for something more than a direct
//      call, for example being stored into a memory location will also have an
//      edge from this external node.  Since they may be called by an unknown
//      caller later, they must be tracked as such.
//
// There is a second external node added for calls that leave this module.
// Functions have a call edge to the external node iff:
//   1. The function is external, reflecting the fact that they could call
//      anything without internal linkage or that has its address taken.
//   2. The function contains an indirect function call.
//
// As an extension in the future, there may be multiple nodes with a null
// function.  These will be used when we can prove (through pointer analysis)
// that an indirect call site can call only a specific set of functions.
//
// Because of these properties, the CallGraph captures a conservative superset
// of all of the caller-callee relationships, which is useful for
// transformations.
//
// The CallGraph class also attempts to figure out what the root of the
// CallGraph is, which it currently does by looking for a function named 'main'.
// If no function named 'main' is found, the external node is used as the entry
// node, reflecting the fact that any function without internal linkage could
// be called into (which is common for libraries).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLGRAPH_H
#define LLVM_ANALYSIS_CALLGRAPH_H

#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/IncludeFile.h"
#include <map>

namespace llvm {

class Function;
class Module;
class CallGraphNode;

//===----------------------------------------------------------------------===//
// CallGraph class definition
//
class CallGraph {
protected:
  Module *Mod;              // The module this call graph represents

  typedef std::map<const Function *, CallGraphNode *> FunctionMapTy;
  FunctionMapTy FunctionMap;    // Map from a function to its node

public:
  static char ID; // Class identification, replacement for typeinfo
  //===---------------------------------------------------------------------
  // Accessors.
  //
  typedef FunctionMapTy::iterator iterator;
  typedef FunctionMapTy::const_iterator const_iterator;

  /// getModule - Return the module the call graph corresponds to.
  ///
  Module &getModule() const { return *Mod; }

  inline       iterator begin()       { return FunctionMap.begin(); }
  inline       iterator end()         { return FunctionMap.end();   }
  inline const_iterator begin() const { return FunctionMap.begin(); }
  inline const_iterator end()   const { return FunctionMap.end();   }

  // Subscripting operators, return the call graph node for the provided
  // function
  inline const CallGraphNode *operator[](const Function *F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second;
  }
  inline CallGraphNode *operator[](const Function *F) {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second;
  }

  /// Returns the CallGraphNode which is used to represent undetermined calls
  /// into the callgraph.  Override this if you want behavioral inheritance.
  virtual CallGraphNode* getExternalCallingNode() const { return 0; }
  virtual CallGraphNode* getCallsExternalNode()   const { return 0; }

  /// Return the root/main method in the module, or some other root node, such
  /// as the externalcallingnode.  Overload these if you behavioral
  /// inheritance.
  virtual CallGraphNode* getRoot() { return 0; }
  virtual const CallGraphNode* getRoot() const { return 0; }

  //===---------------------------------------------------------------------
  // Functions to keep a call graph up to date with a function that has been
  // modified.
  //

  /// removeFunctionFromModule - Unlink the function from this module, returning
  /// it.  Because this removes the function from the module, the call graph
  /// node is destroyed.  This is only valid if the function does not call any
  /// other functions (ie, there are no edges in it's CGN).  The easiest way to
  /// do this is to dropAllReferences before calling this.
  ///
  Function *removeFunctionFromModule(CallGraphNode *CGN);
  Function *removeFunctionFromModule(Function *F) {
    return removeFunctionFromModule((*this)[F]);
  }

  /// getOrInsertFunction - This method is identical to calling operator[], but
  /// it will insert a new CallGraphNode for the specified function if one does
  /// not already exist.
  CallGraphNode *getOrInsertFunction(const Function *F);

  /// spliceFunction - Replace the function represented by this node by another.
  /// This does not rescan the body of the function, so it is suitable when
  /// splicing the body of one function to another while also updating all
  /// callers from the old function to the new.
  ///
  void spliceFunction(const Function *From, const Function *To);

  //===---------------------------------------------------------------------
  // Pass infrastructure interface glue code.
  //
protected:
  CallGraph() {}

public:
  virtual ~CallGraph() { destroy(); }

  /// initialize - Call this method before calling other methods,
  /// re/initializes the state of the CallGraph.
  ///
  void initialize(Module &M);

  void print(raw_ostream &o, Module *) const;
  void dump() const;
protected:
  // destroy - Release memory for the call graph
  virtual void destroy();
};

//===----------------------------------------------------------------------===//
// CallGraphNode class definition.
//
class CallGraphNode {
  friend class CallGraph;
  
  AssertingVH<Function> F;

  // CallRecord - This is a pair of the calling instruction (a call or invoke)
  // and the callgraph node being called.
public:
  typedef std::pair<WeakVH, CallGraphNode*> CallRecord;
private:
  std::vector<CallRecord> CalledFunctions;
  
  /// NumReferences - This is the number of times that this CallGraphNode occurs
  /// in the CalledFunctions array of this or other CallGraphNodes.
  unsigned NumReferences;

  CallGraphNode(const CallGraphNode &) LLVM_DELETED_FUNCTION;
  void operator=(const CallGraphNode &) LLVM_DELETED_FUNCTION;
 
  void DropRef() { --NumReferences; }
  void AddRef() { ++NumReferences; }
public:
  typedef std::vector<CallRecord> CalledFunctionsVector;

  
  // CallGraphNode ctor - Create a node for the specified function.
  inline CallGraphNode(Function *f) : F(f), NumReferences(0) {}
  ~CallGraphNode() {
    assert(NumReferences == 0 && "Node deleted while references remain");
  }
  
  //===---------------------------------------------------------------------
  // Accessor methods.
  //

  typedef std::vector<CallRecord>::iterator iterator;
  typedef std::vector<CallRecord>::const_iterator const_iterator;

  // getFunction - Return the function that this call graph node represents.
  Function *getFunction() const { return F; }

  inline iterator begin() { return CalledFunctions.begin(); }
  inline iterator end()   { return CalledFunctions.end();   }
  inline const_iterator begin() const { return CalledFunctions.begin(); }
  inline const_iterator end()   const { return CalledFunctions.end();   }
  inline bool empty() const { return CalledFunctions.empty(); }
  inline unsigned size() const { return (unsigned)CalledFunctions.size(); }

  /// getNumReferences - Return the number of other CallGraphNodes in this
  /// CallGraph that reference this node in their callee list.
  unsigned getNumReferences() const { return NumReferences; }
  
  // Subscripting operator - Return the i'th called function.
  //
  CallGraphNode *operator[](unsigned i) const {
    assert(i < CalledFunctions.size() && "Invalid index");
    return CalledFunctions[i].second;
  }

  /// dump - Print out this call graph node.
  ///
  void dump() const;
  void print(raw_ostream &OS) const;

  //===---------------------------------------------------------------------
  // Methods to keep a call graph up to date with a function that has been
  // modified
  //

  /// removeAllCalledFunctions - As the name implies, this removes all edges
  /// from this CallGraphNode to any functions it calls.
  void removeAllCalledFunctions() {
    while (!CalledFunctions.empty()) {
      CalledFunctions.back().second->DropRef();
      CalledFunctions.pop_back();
    }
  }
  
  /// stealCalledFunctionsFrom - Move all the callee information from N to this
  /// node.
  void stealCalledFunctionsFrom(CallGraphNode *N) {
    assert(CalledFunctions.empty() &&
           "Cannot steal callsite information if I already have some");
    std::swap(CalledFunctions, N->CalledFunctions);
  }
  

  /// addCalledFunction - Add a function to the list of functions called by this
  /// one.
  void addCalledFunction(CallSite CS, CallGraphNode *M) {
    assert(!CS.getInstruction() ||
           !CS.getCalledFunction() ||
           !CS.getCalledFunction()->isIntrinsic());
    CalledFunctions.push_back(std::make_pair(CS.getInstruction(), M));
    M->AddRef();
  }

  void removeCallEdge(iterator I) {
    I->second->DropRef();
    *I = CalledFunctions.back();
    CalledFunctions.pop_back();
  }
  
  
  /// removeCallEdgeFor - This method removes the edge in the node for the
  /// specified call site.  Note that this method takes linear time, so it
  /// should be used sparingly.
  void removeCallEdgeFor(CallSite CS);

  /// removeAnyCallEdgeTo - This method removes all call edges from this node
  /// to the specified callee function.  This takes more time to execute than
  /// removeCallEdgeTo, so it should not be used unless necessary.
  void removeAnyCallEdgeTo(CallGraphNode *Callee);

  /// removeOneAbstractEdgeTo - Remove one edge associated with a null callsite
  /// from this node to the specified callee function.
  void removeOneAbstractEdgeTo(CallGraphNode *Callee);
  
  /// replaceCallEdge - This method replaces the edge in the node for the
  /// specified call site with a new one.  Note that this method takes linear
  /// time, so it should be used sparingly.
  void replaceCallEdge(CallSite CS, CallSite NewCS, CallGraphNode *NewNode);
  
  /// allReferencesDropped - This is a special function that should only be
  /// used by the CallGraph class.
  void allReferencesDropped() {
    NumReferences = 0;
  }
};

//===----------------------------------------------------------------------===//
// GraphTraits specializations for call graphs so that they can be treated as
// graphs by the generic graph algorithms.
//

// Provide graph traits for tranversing call graphs using standard graph
// traversals.
template <> struct GraphTraits<CallGraphNode*> {
  typedef CallGraphNode NodeType;

  typedef CallGraphNode::CallRecord CGNPairTy;
  typedef std::pointer_to_unary_function<CGNPairTy, CallGraphNode*> CGNDerefFun;

  static NodeType *getEntryNode(CallGraphNode *CGN) { return CGN; }

  typedef mapped_iterator<NodeType::iterator, CGNDerefFun> ChildIteratorType;

  static inline ChildIteratorType child_begin(NodeType *N) {
    return map_iterator(N->begin(), CGNDerefFun(CGNDeref));
  }
  static inline ChildIteratorType child_end  (NodeType *N) {
    return map_iterator(N->end(), CGNDerefFun(CGNDeref));
  }

  static CallGraphNode *CGNDeref(CGNPairTy P) {
    return P.second;
  }

};

template <> struct GraphTraits<const CallGraphNode*> {
  typedef const CallGraphNode NodeType;
  typedef NodeType::const_iterator ChildIteratorType;

  static NodeType *getEntryNode(const CallGraphNode *CGN) { return CGN; }
  static inline ChildIteratorType child_begin(NodeType *N) { return N->begin();}
  static inline ChildIteratorType child_end  (NodeType *N) { return N->end(); }
};

template<> struct GraphTraits<CallGraph*> : public GraphTraits<CallGraphNode*> {
  static NodeType *getEntryNode(CallGraph *CGN) {
    return CGN->getExternalCallingNode();  // Start at the external node!
  }
  typedef std::pair<const Function*, CallGraphNode*> PairTy;
  typedef std::pointer_to_unary_function<PairTy, CallGraphNode&> DerefFun;

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<CallGraph::iterator, DerefFun> nodes_iterator;
  static nodes_iterator nodes_begin(CallGraph *CG) {
    return map_iterator(CG->begin(), DerefFun(CGdereference));
  }
  static nodes_iterator nodes_end  (CallGraph *CG) {
    return map_iterator(CG->end(), DerefFun(CGdereference));
  }

  static CallGraphNode &CGdereference(PairTy P) {
    return *P.second;
  }
};

template<> struct GraphTraits<const CallGraph*> :
  public GraphTraits<const CallGraphNode*> {
  static NodeType *getEntryNode(const CallGraph *CGN) {
    return CGN->getExternalCallingNode();
  }
  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef CallGraph::const_iterator nodes_iterator;
  static nodes_iterator nodes_begin(const CallGraph *CG) { return CG->begin(); }
  static nodes_iterator nodes_end  (const CallGraph *CG) { return CG->end(); }
};

} // End llvm namespace

// Make sure that any clients of this file link in CallGraph.cpp
FORCE_DEFINING_FILE_TO_BE_LINKED(CallGraph)

#endif
