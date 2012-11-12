//===- FunctionAttrs.cpp - Pass which marks functions readnone or readonly ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple interprocedural pass which walks the
// call-graph, looking for functions which do not access or only read
// non-local memory, and marking them readnone/readonly.  In addition,
// it marks function arguments (of pointer type) 'nocapture' if a call
// to the function does not create any copies of the pointer value that
// outlive the call.  This more or less means that the pointer is only
// dereferenced, and not returned from the function or stored in a global.
// This pass is implemented as a bottom-up traversal of the call-graph.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "functionattrs"
#include "llvm/Transforms/IPO.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/GlobalVariable.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/InstIterator.h"
using namespace llvm;

STATISTIC(NumReadNone, "Number of functions marked readnone");
STATISTIC(NumReadOnly, "Number of functions marked readonly");
STATISTIC(NumNoCapture, "Number of arguments marked nocapture");
STATISTIC(NumNoAlias, "Number of function returns marked noalias");

namespace {
  struct FunctionAttrs : public CallGraphSCCPass {
    static char ID; // Pass identification, replacement for typeid
    FunctionAttrs() : CallGraphSCCPass(ID), AA(0) {
      initializeFunctionAttrsPass(*PassRegistry::getPassRegistry());
    }

    // runOnSCC - Analyze the SCC, performing the transformation if possible.
    bool runOnSCC(CallGraphSCC &SCC);

    // AddReadAttrs - Deduce readonly/readnone attributes for the SCC.
    bool AddReadAttrs(const CallGraphSCC &SCC);

    // AddNoCaptureAttrs - Deduce nocapture attributes for the SCC.
    bool AddNoCaptureAttrs(const CallGraphSCC &SCC);

    // IsFunctionMallocLike - Does this function allocate new memory?
    bool IsFunctionMallocLike(Function *F,
                              SmallPtrSet<Function*, 8> &) const;

    // AddNoAliasAttrs - Deduce noalias attributes for the SCC.
    bool AddNoAliasAttrs(const CallGraphSCC &SCC);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<AliasAnalysis>();
      CallGraphSCCPass::getAnalysisUsage(AU);
    }

  private:
    AliasAnalysis *AA;
  };
}

char FunctionAttrs::ID = 0;
INITIALIZE_PASS_BEGIN(FunctionAttrs, "functionattrs",
                "Deduce function attributes", false, false)
INITIALIZE_AG_DEPENDENCY(CallGraph)
INITIALIZE_PASS_END(FunctionAttrs, "functionattrs",
                "Deduce function attributes", false, false)

Pass *llvm::createFunctionAttrsPass() { return new FunctionAttrs(); }


/// AddReadAttrs - Deduce readonly/readnone attributes for the SCC.
bool FunctionAttrs::AddReadAttrs(const CallGraphSCC &SCC) {
  SmallPtrSet<Function*, 8> SCCNodes;

  // Fill SCCNodes with the elements of the SCC.  Used for quickly
  // looking up whether a given CallGraphNode is in this SCC.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I)
    SCCNodes.insert((*I)->getFunction());

  // Check if any of the functions in the SCC read or write memory.  If they
  // write memory then they can't be marked readnone or readonly.
  bool ReadsMemory = false;
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (F == 0)
      // External node - may write memory.  Just give up.
      return false;

    AliasAnalysis::ModRefBehavior MRB = AA->getModRefBehavior(F);
    if (MRB == AliasAnalysis::DoesNotAccessMemory)
      // Already perfect!
      continue;

    // Definitions with weak linkage may be overridden at linktime with
    // something that writes memory, so treat them like declarations.
    if (F->isDeclaration() || F->mayBeOverridden()) {
      if (!AliasAnalysis::onlyReadsMemory(MRB))
        // May write memory.  Just give up.
        return false;

      ReadsMemory = true;
      continue;
    }

    // Scan the function body for instructions that may read or write memory.
    for (inst_iterator II = inst_begin(F), E = inst_end(F); II != E; ++II) {
      Instruction *I = &*II;

      // Some instructions can be ignored even if they read or write memory.
      // Detect these now, skipping to the next instruction if one is found.
      CallSite CS(cast<Value>(I));
      if (CS) {
        // Ignore calls to functions in the same SCC.
        if (CS.getCalledFunction() && SCCNodes.count(CS.getCalledFunction()))
          continue;
        AliasAnalysis::ModRefBehavior MRB = AA->getModRefBehavior(CS);
        // If the call doesn't access arbitrary memory, we may be able to
        // figure out something.
        if (AliasAnalysis::onlyAccessesArgPointees(MRB)) {
          // If the call does access argument pointees, check each argument.
          if (AliasAnalysis::doesAccessArgPointees(MRB))
            // Check whether all pointer arguments point to local memory, and
            // ignore calls that only access local memory.
            for (CallSite::arg_iterator CI = CS.arg_begin(), CE = CS.arg_end();
                 CI != CE; ++CI) {
              Value *Arg = *CI;
              if (Arg->getType()->isPointerTy()) {
                AliasAnalysis::Location Loc(Arg,
                                            AliasAnalysis::UnknownSize,
                                            I->getMetadata(LLVMContext::MD_tbaa));
                if (!AA->pointsToConstantMemory(Loc, /*OrLocal=*/true)) {
                  if (MRB & AliasAnalysis::Mod)
                    // Writes non-local memory.  Give up.
                    return false;
                  if (MRB & AliasAnalysis::Ref)
                    // Ok, it reads non-local memory.
                    ReadsMemory = true;
                }
              }
            }
          continue;
        }
        // The call could access any memory. If that includes writes, give up.
        if (MRB & AliasAnalysis::Mod)
          return false;
        // If it reads, note it.
        if (MRB & AliasAnalysis::Ref)
          ReadsMemory = true;
        continue;
      } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        // Ignore non-volatile loads from local memory. (Atomic is okay here.)
        if (!LI->isVolatile()) {
          AliasAnalysis::Location Loc = AA->getLocation(LI);
          if (AA->pointsToConstantMemory(Loc, /*OrLocal=*/true))
            continue;
        }
      } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        // Ignore non-volatile stores to local memory. (Atomic is okay here.)
        if (!SI->isVolatile()) {
          AliasAnalysis::Location Loc = AA->getLocation(SI);
          if (AA->pointsToConstantMemory(Loc, /*OrLocal=*/true))
            continue;
        }
      } else if (VAArgInst *VI = dyn_cast<VAArgInst>(I)) {
        // Ignore vaargs on local memory.
        AliasAnalysis::Location Loc = AA->getLocation(VI);
        if (AA->pointsToConstantMemory(Loc, /*OrLocal=*/true))
          continue;
      }

      // Any remaining instructions need to be taken seriously!  Check if they
      // read or write memory.
      if (I->mayWriteToMemory())
        // Writes memory.  Just give up.
        return false;

      // If this instruction may read memory, remember that.
      ReadsMemory |= I->mayReadFromMemory();
    }
  }

  // Success!  Functions in this SCC do not access memory, or only read memory.
  // Give them the appropriate attribute.
  bool MadeChange = false;
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (F->doesNotAccessMemory())
      // Already perfect!
      continue;

    if (F->onlyReadsMemory() && ReadsMemory)
      // No change.
      continue;

    MadeChange = true;

    // Clear out any existing attributes.
    AttrBuilder B;
    B.addAttribute(Attributes::ReadOnly)
      .addAttribute(Attributes::ReadNone);
    F->removeAttribute(AttrListPtr::FunctionIndex,
                       Attributes::get(F->getContext(), B));

    // Add in the new attribute.
    B.clear();
    B.addAttribute(ReadsMemory ? Attributes::ReadOnly : Attributes::ReadNone);
    F->addAttribute(AttrListPtr::FunctionIndex,
                    Attributes::get(F->getContext(), B));

    if (ReadsMemory)
      ++NumReadOnly;
    else
      ++NumReadNone;
  }

  return MadeChange;
}

namespace {
  // For a given pointer Argument, this retains a list of Arguments of functions
  // in the same SCC that the pointer data flows into. We use this to build an
  // SCC of the arguments.
  struct ArgumentGraphNode {
    Argument *Definition;
    SmallVector<ArgumentGraphNode*, 4> Uses;
  };

  class ArgumentGraph {
    // We store pointers to ArgumentGraphNode objects, so it's important that
    // that they not move around upon insert.
    typedef std::map<Argument*, ArgumentGraphNode> ArgumentMapTy;

    ArgumentMapTy ArgumentMap;

    // There is no root node for the argument graph, in fact:
    //   void f(int *x, int *y) { if (...) f(x, y); }
    // is an example where the graph is disconnected. The SCCIterator requires a
    // single entry point, so we maintain a fake ("synthetic") root node that
    // uses every node. Because the graph is directed and nothing points into
    // the root, it will not participate in any SCCs (except for its own).
    ArgumentGraphNode SyntheticRoot;

  public:
    ArgumentGraph() { SyntheticRoot.Definition = 0; }

    typedef SmallVectorImpl<ArgumentGraphNode*>::iterator iterator;

    iterator begin() { return SyntheticRoot.Uses.begin(); }
    iterator end() { return SyntheticRoot.Uses.end(); }
    ArgumentGraphNode *getEntryNode() { return &SyntheticRoot; }

    ArgumentGraphNode *operator[](Argument *A) {
      ArgumentGraphNode &Node = ArgumentMap[A];
      Node.Definition = A;
      SyntheticRoot.Uses.push_back(&Node);
      return &Node;
    }
  };

  // This tracker checks whether callees are in the SCC, and if so it does not
  // consider that a capture, instead adding it to the "Uses" list and
  // continuing with the analysis.
  struct ArgumentUsesTracker : public CaptureTracker {
    ArgumentUsesTracker(const SmallPtrSet<Function*, 8> &SCCNodes)
      : Captured(false), SCCNodes(SCCNodes) {}

    void tooManyUses() { Captured = true; }

    bool captured(Use *U) {
      CallSite CS(U->getUser());
      if (!CS.getInstruction()) { Captured = true; return true; }

      Function *F = CS.getCalledFunction();
      if (!F || !SCCNodes.count(F)) { Captured = true; return true; }

      Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end();
      for (CallSite::arg_iterator PI = CS.arg_begin(), PE = CS.arg_end();
           PI != PE; ++PI, ++AI) {
        if (AI == AE) {
          assert(F->isVarArg() && "More params than args in non-varargs call");
          Captured = true;
          return true;
        }
        if (PI == U) {
          Uses.push_back(AI);
          break;
        }
      }
      assert(!Uses.empty() && "Capturing call-site captured nothing?");
      return false;
    }

    bool Captured;  // True only if certainly captured (used outside our SCC).
    SmallVector<Argument*, 4> Uses;  // Uses within our SCC.

    const SmallPtrSet<Function*, 8> &SCCNodes;
  };
}

namespace llvm {
  template<> struct GraphTraits<ArgumentGraphNode*> {
    typedef ArgumentGraphNode NodeType;
    typedef SmallVectorImpl<ArgumentGraphNode*>::iterator ChildIteratorType;

    static inline NodeType *getEntryNode(NodeType *A) { return A; }
    static inline ChildIteratorType child_begin(NodeType *N) {
      return N->Uses.begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->Uses.end();
    }
  };
  template<> struct GraphTraits<ArgumentGraph*>
    : public GraphTraits<ArgumentGraphNode*> {
    static NodeType *getEntryNode(ArgumentGraph *AG) {
      return AG->getEntryNode();
    }
    static ChildIteratorType nodes_begin(ArgumentGraph *AG) {
      return AG->begin();
    }
    static ChildIteratorType nodes_end(ArgumentGraph *AG) {
      return AG->end();
    }
  };
}

/// AddNoCaptureAttrs - Deduce nocapture attributes for the SCC.
bool FunctionAttrs::AddNoCaptureAttrs(const CallGraphSCC &SCC) {
  bool Changed = false;

  SmallPtrSet<Function*, 8> SCCNodes;

  // Fill SCCNodes with the elements of the SCC.  Used for quickly
  // looking up whether a given CallGraphNode is in this SCC.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();
    if (F && !F->isDeclaration() && !F->mayBeOverridden())
      SCCNodes.insert(F);
  }

  ArgumentGraph AG;

  AttrBuilder B;
  B.addAttribute(Attributes::NoCapture);

  // Check each function in turn, determining which pointer arguments are not
  // captured.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (F == 0)
      // External node - only a problem for arguments that we pass to it.
      continue;

    // Definitions with weak linkage may be overridden at linktime with
    // something that captures pointers, so treat them like declarations.
    if (F->isDeclaration() || F->mayBeOverridden())
      continue;

    // Functions that are readonly (or readnone) and nounwind and don't return
    // a value can't capture arguments. Don't analyze them.
    if (F->onlyReadsMemory() && F->doesNotThrow() &&
        F->getReturnType()->isVoidTy()) {
      for (Function::arg_iterator A = F->arg_begin(), E = F->arg_end();
           A != E; ++A) {
        if (A->getType()->isPointerTy() && !A->hasNoCaptureAttr()) {
          A->addAttr(Attributes::get(F->getContext(), B));
          ++NumNoCapture;
          Changed = true;
        }
      }
      continue;
    }

    for (Function::arg_iterator A = F->arg_begin(), E = F->arg_end(); A!=E; ++A)
      if (A->getType()->isPointerTy() && !A->hasNoCaptureAttr()) {
        ArgumentUsesTracker Tracker(SCCNodes);
        PointerMayBeCaptured(A, &Tracker);
        if (!Tracker.Captured) {
          if (Tracker.Uses.empty()) {
            // If it's trivially not captured, mark it nocapture now.
            A->addAttr(Attributes::get(F->getContext(), B));
            ++NumNoCapture;
            Changed = true;
          } else {
            // If it's not trivially captured and not trivially not captured,
            // then it must be calling into another function in our SCC. Save
            // its particulars for Argument-SCC analysis later.
            ArgumentGraphNode *Node = AG[A];
            for (SmallVectorImpl<Argument*>::iterator UI = Tracker.Uses.begin(),
                   UE = Tracker.Uses.end(); UI != UE; ++UI)
              Node->Uses.push_back(AG[*UI]);
          }
        }
        // Otherwise, it's captured. Don't bother doing SCC analysis on it.
      }
  }

  // The graph we've collected is partial because we stopped scanning for
  // argument uses once we solved the argument trivially. These partial nodes
  // show up as ArgumentGraphNode objects with an empty Uses list, and for
  // these nodes the final decision about whether they capture has already been
  // made.  If the definition doesn't have a 'nocapture' attribute by now, it
  // captures.

  for (scc_iterator<ArgumentGraph*> I = scc_begin(&AG), E = scc_end(&AG);
       I != E; ++I) {
    std::vector<ArgumentGraphNode*> &ArgumentSCC = *I;
    if (ArgumentSCC.size() == 1) {
      if (!ArgumentSCC[0]->Definition) continue;  // synthetic root node

      // eg. "void f(int* x) { if (...) f(x); }"
      if (ArgumentSCC[0]->Uses.size() == 1 &&
          ArgumentSCC[0]->Uses[0] == ArgumentSCC[0]) {
        ArgumentSCC[0]->
          Definition->
          addAttr(Attributes::get(ArgumentSCC[0]->Definition->getContext(), B));
        ++NumNoCapture;
        Changed = true;
      }
      continue;
    }

    bool SCCCaptured = false;
    for (std::vector<ArgumentGraphNode*>::iterator I = ArgumentSCC.begin(),
           E = ArgumentSCC.end(); I != E && !SCCCaptured; ++I) {
      ArgumentGraphNode *Node = *I;
      if (Node->Uses.empty()) {
        if (!Node->Definition->hasNoCaptureAttr())
          SCCCaptured = true;
      }
    }
    if (SCCCaptured) continue;

    SmallPtrSet<Argument*, 8> ArgumentSCCNodes;
    // Fill ArgumentSCCNodes with the elements of the ArgumentSCC.  Used for
    // quickly looking up whether a given Argument is in this ArgumentSCC.
    for (std::vector<ArgumentGraphNode*>::iterator I = ArgumentSCC.begin(),
           E = ArgumentSCC.end(); I != E; ++I) {
      ArgumentSCCNodes.insert((*I)->Definition);
    }

    for (std::vector<ArgumentGraphNode*>::iterator I = ArgumentSCC.begin(),
           E = ArgumentSCC.end(); I != E && !SCCCaptured; ++I) {
      ArgumentGraphNode *N = *I;
      for (SmallVectorImpl<ArgumentGraphNode*>::iterator UI = N->Uses.begin(),
             UE = N->Uses.end(); UI != UE; ++UI) {
        Argument *A = (*UI)->Definition;
        if (A->hasNoCaptureAttr() || ArgumentSCCNodes.count(A))
          continue;
        SCCCaptured = true;
        break;
      }
    }
    if (SCCCaptured) continue;

    for (unsigned i = 0, e = ArgumentSCC.size(); i != e; ++i) {
      Argument *A = ArgumentSCC[i]->Definition;
      A->addAttr(Attributes::get(A->getContext(), B));
      ++NumNoCapture;
      Changed = true;
    }
  }

  return Changed;
}

/// IsFunctionMallocLike - A function is malloc-like if it returns either null
/// or a pointer that doesn't alias any other pointer visible to the caller.
bool FunctionAttrs::IsFunctionMallocLike(Function *F,
                              SmallPtrSet<Function*, 8> &SCCNodes) const {
  SmallSetVector<Value *, 8> FlowsToReturn;
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(I->getTerminator()))
      FlowsToReturn.insert(Ret->getReturnValue());

  for (unsigned i = 0; i != FlowsToReturn.size(); ++i) {
    Value *RetVal = FlowsToReturn[i];

    if (Constant *C = dyn_cast<Constant>(RetVal)) {
      if (!C->isNullValue() && !isa<UndefValue>(C))
        return false;

      continue;
    }

    if (isa<Argument>(RetVal))
      return false;

    if (Instruction *RVI = dyn_cast<Instruction>(RetVal))
      switch (RVI->getOpcode()) {
        // Extend the analysis by looking upwards.
        case Instruction::BitCast:
        case Instruction::GetElementPtr:
          FlowsToReturn.insert(RVI->getOperand(0));
          continue;
        case Instruction::Select: {
          SelectInst *SI = cast<SelectInst>(RVI);
          FlowsToReturn.insert(SI->getTrueValue());
          FlowsToReturn.insert(SI->getFalseValue());
          continue;
        }
        case Instruction::PHI: {
          PHINode *PN = cast<PHINode>(RVI);
          for (int i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
            FlowsToReturn.insert(PN->getIncomingValue(i));
          continue;
        }

        // Check whether the pointer came from an allocation.
        case Instruction::Alloca:
          break;
        case Instruction::Call:
        case Instruction::Invoke: {
          CallSite CS(RVI);
          if (CS.paramHasAttr(0, Attributes::NoAlias))
            break;
          if (CS.getCalledFunction() &&
              SCCNodes.count(CS.getCalledFunction()))
            break;
        } // fall-through
        default:
          return false;  // Did not come from an allocation.
      }

    if (PointerMayBeCaptured(RetVal, false, /*StoreCaptures=*/false))
      return false;
  }

  return true;
}

/// AddNoAliasAttrs - Deduce noalias attributes for the SCC.
bool FunctionAttrs::AddNoAliasAttrs(const CallGraphSCC &SCC) {
  SmallPtrSet<Function*, 8> SCCNodes;

  // Fill SCCNodes with the elements of the SCC.  Used for quickly
  // looking up whether a given CallGraphNode is in this SCC.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I)
    SCCNodes.insert((*I)->getFunction());

  // Check each function in turn, determining which functions return noalias
  // pointers.
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();

    if (F == 0)
      // External node - skip it;
      return false;

    // Already noalias.
    if (F->doesNotAlias(0))
      continue;

    // Definitions with weak linkage may be overridden at linktime, so
    // treat them like declarations.
    if (F->isDeclaration() || F->mayBeOverridden())
      return false;

    // We annotate noalias return values, which are only applicable to 
    // pointer types.
    if (!F->getReturnType()->isPointerTy())
      continue;

    if (!IsFunctionMallocLike(F, SCCNodes))
      return false;
  }

  bool MadeChange = false;
  for (CallGraphSCC::iterator I = SCC.begin(), E = SCC.end(); I != E; ++I) {
    Function *F = (*I)->getFunction();
    if (F->doesNotAlias(0) || !F->getReturnType()->isPointerTy())
      continue;

    F->setDoesNotAlias(0);
    ++NumNoAlias;
    MadeChange = true;
  }

  return MadeChange;
}

bool FunctionAttrs::runOnSCC(CallGraphSCC &SCC) {
  AA = &getAnalysis<AliasAnalysis>();

  bool Changed = AddReadAttrs(SCC);
  Changed |= AddNoCaptureAttrs(SCC);
  Changed |= AddNoAliasAttrs(SCC);
  return Changed;
}
