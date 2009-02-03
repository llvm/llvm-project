//===- CallGraph.cpp - Build a Module's call graph ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CallGraph class and provides the BasicCallGraph
// default implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"
#include <ostream>
using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// BasicCallGraph class definition
//
class VISIBILITY_HIDDEN BasicCallGraph : public CallGraph, public ModulePass {
  // Root is root of the call graph, or the external node if a 'main' function
  // couldn't be found.
  //
  CallGraphNode *Root;

  // ExternalCallingNode - This node has edges to all external functions and
  // those internal functions that have their address taken.
  CallGraphNode *ExternalCallingNode;

  // CallsExternalNode - This node has edges to it from all functions making
  // indirect calls or calling an external function.
  CallGraphNode *CallsExternalNode;

public:
  static char ID; // Class identification, replacement for typeinfo
  BasicCallGraph() : ModulePass(&ID), Root(0), 
    ExternalCallingNode(0), CallsExternalNode(0) {}

  // runOnModule - Compute the call graph for the specified module.
  virtual bool runOnModule(Module &M) {
    CallGraph::initialize(M);
    
    ExternalCallingNode = getOrInsertFunction(0);
    CallsExternalNode = new CallGraphNode(0);
    Root = 0;
  
    // Add every function to the call graph...
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      addToCallGraph(I);
  
    // If we didn't find a main function, use the external call graph node
    if (Root == 0) Root = ExternalCallingNode;
    
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  void print(std::ostream *o, const Module *M) const {
    if (o) print(*o, M);
  }

  virtual void print(std::ostream &o, const Module *M) const {
    o << "CallGraph Root is: ";
    if (Function *F = getRoot()->getFunction())
      o << F->getName() << "\n";
    else
      o << "<<null function: 0x" << getRoot() << ">>\n";
    
    CallGraph::print(o, M);
  }

  virtual void releaseMemory() {
    destroy();
  }
  
  /// dump - Print out this call graph.
  ///
  inline void dump() const {
    print(cerr, Mod);
  }

  CallGraphNode* getExternalCallingNode() const { return ExternalCallingNode; }
  CallGraphNode* getCallsExternalNode()   const { return CallsExternalNode; }

  // getRoot - Return the root of the call graph, which is either main, or if
  // main cannot be found, the external node.
  //
  CallGraphNode *getRoot()             { return Root; }
  const CallGraphNode *getRoot() const { return Root; }

private:
  //===---------------------------------------------------------------------
  // Implementation of CallGraph construction
  //

  // addToCallGraph - Add a function to the call graph, and link the node to all
  // of the functions that it calls.
  //
  void addToCallGraph(Function *F) {
    CallGraphNode *Node = getOrInsertFunction(F);

    // If this function has external linkage, anything could call it.
    if (!F->hasLocalLinkage()) {
      ExternalCallingNode->addCalledFunction(CallSite(), Node);

      // Found the entry point?
      if (F->getName() == "main") {
        if (Root)    // Found multiple external mains?  Don't pick one.
          Root = ExternalCallingNode;
        else
          Root = Node;          // Found a main, keep track of it!
      }
    }

    // Loop over all of the users of the function, looking for non-call uses.
    for (Value::use_iterator I = F->use_begin(), E = F->use_end(); I != E; ++I)
      if ((!isa<CallInst>(I) && !isa<InvokeInst>(I))
          || !CallSite(cast<Instruction>(I)).isCallee(I)) {
        // Not a call, or being used as a parameter rather than as the callee.
        ExternalCallingNode->addCalledFunction(CallSite(), Node);
        break;
      }

    // If this function is not defined in this translation unit, it could call
    // anything.
    if (F->isDeclaration() && !F->isIntrinsic())
      Node->addCalledFunction(CallSite(), CallsExternalNode);

    // Look for calls by this function.
    for (Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB)
      for (BasicBlock::iterator II = BB->begin(), IE = BB->end();
           II != IE; ++II) {
        CallSite CS = CallSite::get(II);
        if (CS.getInstruction()) {
          const Function *Callee = CS.getCalledFunction();
          if (Callee)
            Node->addCalledFunction(CS, getOrInsertFunction(Callee));
          else
            Node->addCalledFunction(CS, CallsExternalNode);
        }
      }
  }

  //
  // destroy - Release memory for the call graph
  virtual void destroy() {
    /// CallsExternalNode is not in the function map, delete it explicitly.
    delete CallsExternalNode;
    CallsExternalNode = 0;
    CallGraph::destroy();
  }
};

} //End anonymous namespace

static RegisterAnalysisGroup<CallGraph> X("Call Graph");
static RegisterPass<BasicCallGraph>
Y("basiccg", "Basic CallGraph Construction", false, true);
static RegisterAnalysisGroup<CallGraph, true> Z(Y);

char CallGraph::ID = 0;
char BasicCallGraph::ID = 0;

void CallGraph::initialize(Module &M) {
  Mod = &M;
}

void CallGraph::destroy() {
  if (!FunctionMap.empty()) {
    for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
        I != E; ++I)
      delete I->second;
    FunctionMap.clear();
  }
}

void CallGraph::print(std::ostream &OS, const Module *M) const {
  for (CallGraph::const_iterator I = begin(), E = end(); I != E; ++I)
    I->second->print(OS);
}

void CallGraph::dump() const {
  print(cerr, 0);
}

//===----------------------------------------------------------------------===//
// Implementations of public modification methods
//

// removeFunctionFromModule - Unlink the function from this module, returning
// it.  Because this removes the function from the module, the call graph node
// is destroyed.  This is only valid if the function does not call any other
// functions (ie, there are no edges in it's CGN).  The easiest way to do this
// is to dropAllReferences before calling this.
//
Function *CallGraph::removeFunctionFromModule(CallGraphNode *CGN) {
  assert(CGN->CalledFunctions.empty() && "Cannot remove function from call "
         "graph if it references other functions!");
  Function *F = CGN->getFunction(); // Get the function for the call graph node
  delete CGN;                       // Delete the call graph node for this func
  FunctionMap.erase(F);             // Remove the call graph node from the map

  Mod->getFunctionList().remove(F);
  return F;
}

// changeFunction - This method changes the function associated with this
// CallGraphNode, for use by transformations that need to change the prototype
// of a Function (thus they must create a new Function and move the old code
// over).
void CallGraph::changeFunction(Function *OldF, Function *NewF) {
  iterator I = FunctionMap.find(OldF);
  CallGraphNode *&New = FunctionMap[NewF];
  assert(I != FunctionMap.end() && I->second && !New &&
         "OldF didn't exist in CG or NewF already does!");
  New = I->second;
  New->F = NewF;
  FunctionMap.erase(I);
}

// getOrInsertFunction - This method is identical to calling operator[], but
// it will insert a new CallGraphNode for the specified function if one does
// not already exist.
CallGraphNode *CallGraph::getOrInsertFunction(const Function *F) {
  CallGraphNode *&CGN = FunctionMap[F];
  if (CGN) return CGN;
  
  assert((!F || F->getParent() == Mod) && "Function not in current module!");
  return CGN = new CallGraphNode(const_cast<Function*>(F));
}

void CallGraphNode::print(std::ostream &OS) const {
  if (Function *F = getFunction())
    OS << "Call graph node for function: '" << F->getName() <<"'\n";
  else
    OS << "Call graph node <<null function: 0x" << this << ">>:\n";

  for (const_iterator I = begin(), E = end(); I != E; ++I)
    if (Function *FI = I->second->getFunction())
      OS << "  Calls function '" << FI->getName() <<"'\n";
  else
    OS << "  Calls external node\n";
  OS << "\n";
}

void CallGraphNode::dump() const { print(cerr); }

/// removeCallEdgeFor - This method removes the edge in the node for the
/// specified call site.  Note that this method takes linear time, so it
/// should be used sparingly.
void CallGraphNode::removeCallEdgeFor(CallSite CS) {
  for (CalledFunctionsVector::iterator I = CalledFunctions.begin(); ; ++I) {
    assert(I != CalledFunctions.end() && "Cannot find callsite to remove!");
    if (I->first == CS) {
      CalledFunctions.erase(I);
      return;
    }
  }
}


// removeAnyCallEdgeTo - This method removes any call edges from this node to
// the specified callee function.  This takes more time to execute than
// removeCallEdgeTo, so it should not be used unless necessary.
void CallGraphNode::removeAnyCallEdgeTo(CallGraphNode *Callee) {
  for (unsigned i = 0, e = CalledFunctions.size(); i != e; ++i)
    if (CalledFunctions[i].second == Callee) {
      CalledFunctions[i] = CalledFunctions.back();
      CalledFunctions.pop_back();
      --i; --e;
    }
}

/// removeOneAbstractEdgeTo - Remove one edge associated with a null callsite
/// from this node to the specified callee function.
void CallGraphNode::removeOneAbstractEdgeTo(CallGraphNode *Callee) {
  for (CalledFunctionsVector::iterator I = CalledFunctions.begin(); ; ++I) {
    assert(I != CalledFunctions.end() && "Cannot find callee to remove!");
    CallRecord &CR = *I;
    if (CR.second == Callee && !CR.first.getInstruction()) {
      CalledFunctions.erase(I);
      return;
    }
  }
}

/// replaceCallSite - Make the edge in the node for Old CallSite be for
/// New CallSite instead.  Note that this method takes linear time, so it
/// should be used sparingly.
void CallGraphNode::replaceCallSite(CallSite Old, CallSite New) {
  for (CalledFunctionsVector::iterator I = CalledFunctions.begin(); ; ++I) {
    assert(I != CalledFunctions.end() && "Cannot find callsite to replace!");
    if (I->first == Old) {
      I->first = New;
      return;
    }
  }
}

// Enuse that users of CallGraph.h also link with this file
DEFINING_FILE_FOR(CallGraph)
