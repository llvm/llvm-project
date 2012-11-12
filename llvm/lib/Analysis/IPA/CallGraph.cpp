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
#include "llvm/IntrinsicInst.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// BasicCallGraph class definition
//
class BasicCallGraph : public ModulePass, public CallGraph {
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
  BasicCallGraph() : ModulePass(ID), Root(0), 
    ExternalCallingNode(0), CallsExternalNode(0) {
      initializeBasicCallGraphPass(*PassRegistry::getPassRegistry());
    }

  // runOnModule - Compute the call graph for the specified module.
  virtual bool runOnModule(Module &M) {
    CallGraph::initialize(M);
    
    ExternalCallingNode = getOrInsertFunction(0);
    CallsExternalNode = new CallGraphNode(0);
    Root = 0;
  
    // Add every function to the call graph.
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      addToCallGraph(I);
  
    // If we didn't find a main function, use the external call graph node
    if (Root == 0) Root = ExternalCallingNode;
    
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  virtual void print(raw_ostream &OS, const Module *) const {
    OS << "CallGraph Root is: ";
    if (Function *F = getRoot()->getFunction())
      OS << F->getName() << "\n";
    else {
      OS << "<<null function: 0x" << getRoot() << ">>\n";
    }
    
    CallGraph::print(OS, 0);
  }

  virtual void releaseMemory() {
    destroy();
  }
  
  /// getAdjustedAnalysisPointer - This method is used when a pass implements
  /// an analysis interface through multiple inheritance.  If needed, it should
  /// override this to adjust the this pointer as needed for the specified pass
  /// info.
  virtual void *getAdjustedAnalysisPointer(AnalysisID PI) {
    if (PI == &CallGraph::ID)
      return (CallGraph*)this;
    return this;
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

    // If this function has its address taken, anything could call it.
    if (F->hasAddressTaken())
      ExternalCallingNode->addCalledFunction(CallSite(), Node);

    // If this function is not defined in this translation unit, it could call
    // anything.
    if (F->isDeclaration() && !F->isIntrinsic())
      Node->addCalledFunction(CallSite(), CallsExternalNode);

    // Look for calls by this function.
    for (Function::iterator BB = F->begin(), BBE = F->end(); BB != BBE; ++BB)
      for (BasicBlock::iterator II = BB->begin(), IE = BB->end();
           II != IE; ++II) {
        CallSite CS(cast<Value>(II));
        if (CS) {
          const Function *Callee = CS.getCalledFunction();
          if (!Callee)
            // Indirect calls of intrinsics are not allowed so no need to check.
            Node->addCalledFunction(CS, CallsExternalNode);
          else if (!Callee->isIntrinsic())
            Node->addCalledFunction(CS, getOrInsertFunction(Callee));
        }
      }
  }

  //
  // destroy - Release memory for the call graph
  virtual void destroy() {
    /// CallsExternalNode is not in the function map, delete it explicitly.
    if (CallsExternalNode) {
      CallsExternalNode->allReferencesDropped();
      delete CallsExternalNode;
      CallsExternalNode = 0;
    }
    CallGraph::destroy();
  }
};

} //End anonymous namespace

INITIALIZE_ANALYSIS_GROUP(CallGraph, "Call Graph", BasicCallGraph)
INITIALIZE_AG_PASS(BasicCallGraph, CallGraph, "basiccg",
                   "Basic CallGraph Construction", false, true, true)

char CallGraph::ID = 0;
char BasicCallGraph::ID = 0;

void CallGraph::initialize(Module &M) {
  Mod = &M;
}

void CallGraph::destroy() {
  if (FunctionMap.empty()) return;
  
  // Reset all node's use counts to zero before deleting them to prevent an
  // assertion from firing.
#ifndef NDEBUG
  for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
       I != E; ++I)
    I->second->allReferencesDropped();
#endif
  
  for (FunctionMapTy::iterator I = FunctionMap.begin(), E = FunctionMap.end();
      I != E; ++I)
    delete I->second;
  FunctionMap.clear();
}

void CallGraph::print(raw_ostream &OS, Module*) const {
  for (CallGraph::const_iterator I = begin(), E = end(); I != E; ++I)
    I->second->print(OS);
}
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void CallGraph::dump() const {
  print(dbgs(), 0);
}
#endif

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
  assert(CGN->empty() && "Cannot remove function from call "
         "graph if it references other functions!");
  Function *F = CGN->getFunction(); // Get the function for the call graph node
  delete CGN;                       // Delete the call graph node for this func
  FunctionMap.erase(F);             // Remove the call graph node from the map

  Mod->getFunctionList().remove(F);
  return F;
}

/// spliceFunction - Replace the function represented by this node by another.
/// This does not rescan the body of the function, so it is suitable when
/// splicing the body of the old function to the new while also updating all
/// callers from old to new.
///
void CallGraph::spliceFunction(const Function *From, const Function *To) {
  assert(FunctionMap.count(From) && "No CallGraphNode for function!");
  assert(!FunctionMap.count(To) &&
         "Pointing CallGraphNode at a function that already exists");
  FunctionMapTy::iterator I = FunctionMap.find(From);
  I->second->F = const_cast<Function*>(To);
  FunctionMap[To] = I->second;
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

void CallGraphNode::print(raw_ostream &OS) const {
  if (Function *F = getFunction())
    OS << "Call graph node for function: '" << F->getName() << "'";
  else
    OS << "Call graph node <<null function>>";
  
  OS << "<<" << this << ">>  #uses=" << getNumReferences() << '\n';

  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    OS << "  CS<" << I->first << "> calls ";
    if (Function *FI = I->second->getFunction())
      OS << "function '" << FI->getName() <<"'\n";
    else
      OS << "external node\n";
  }
  OS << '\n';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void CallGraphNode::dump() const { print(dbgs()); }
#endif

/// removeCallEdgeFor - This method removes the edge in the node for the
/// specified call site.  Note that this method takes linear time, so it
/// should be used sparingly.
void CallGraphNode::removeCallEdgeFor(CallSite CS) {
  for (CalledFunctionsVector::iterator I = CalledFunctions.begin(); ; ++I) {
    assert(I != CalledFunctions.end() && "Cannot find callsite to remove!");
    if (I->first == CS.getInstruction()) {
      I->second->DropRef();
      *I = CalledFunctions.back();
      CalledFunctions.pop_back();
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
      Callee->DropRef();
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
    if (CR.second == Callee && CR.first == 0) {
      Callee->DropRef();
      *I = CalledFunctions.back();
      CalledFunctions.pop_back();
      return;
    }
  }
}

/// replaceCallEdge - This method replaces the edge in the node for the
/// specified call site with a new one.  Note that this method takes linear
/// time, so it should be used sparingly.
void CallGraphNode::replaceCallEdge(CallSite CS,
                                    CallSite NewCS, CallGraphNode *NewNode){
  for (CalledFunctionsVector::iterator I = CalledFunctions.begin(); ; ++I) {
    assert(I != CalledFunctions.end() && "Cannot find callsite to remove!");
    if (I->first == CS.getInstruction()) {
      I->second->DropRef();
      I->first = NewCS.getInstruction();
      I->second = NewNode;
      NewNode->AddRef();
      return;
    }
  }
}

// Enuse that users of CallGraph.h also link with this file
DEFINING_FILE_FOR(CallGraph)
