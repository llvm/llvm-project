//===- GlobalsModRef.cpp - Simple Mod/Ref Analysis for Globals ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This simple pass provides alias and mod/ref information for global values
// that do not have their address taken, and keeps track of whether functions
// read or write memory (are "pure").  For this simple (but very common) case,
// we can provide pretty accurate and useful information.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "globalsmodref-aa"
#include "llvm/Analysis/Passes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SCCIterator.h"
#include <set>
using namespace llvm;

STATISTIC(NumNonAddrTakenGlobalVars,
          "Number of global vars without address taken");
STATISTIC(NumNonAddrTakenFunctions,"Number of functions without address taken");
STATISTIC(NumNoMemFunctions, "Number of functions that do not access memory");
STATISTIC(NumReadMemFunctions, "Number of functions that only read memory");
STATISTIC(NumIndirectGlobalVars, "Number of indirect global objects");

namespace {
  /// FunctionRecord - One instance of this structure is stored for every
  /// function in the program.  Later, the entries for these functions are
  /// removed if the function is found to call an external function (in which
  /// case we know nothing about it.
  struct VISIBILITY_HIDDEN FunctionRecord {
    /// GlobalInfo - Maintain mod/ref info for all of the globals without
    /// addresses taken that are read or written (transitively) by this
    /// function.
    std::map<GlobalValue*, unsigned> GlobalInfo;

    /// MayReadAnyGlobal - May read global variables, but it is not known which.
    bool MayReadAnyGlobal;

    unsigned getInfoForGlobal(GlobalValue *GV) const {
      unsigned Effect = MayReadAnyGlobal ? AliasAnalysis::Ref : 0;
      std::map<GlobalValue*, unsigned>::const_iterator I = GlobalInfo.find(GV);
      if (I != GlobalInfo.end())
        Effect |= I->second;
      return Effect;
    }

    /// FunctionEffect - Capture whether or not this function reads or writes to
    /// ANY memory.  If not, we can do a lot of aggressive analysis on it.
    unsigned FunctionEffect;

    FunctionRecord() : MayReadAnyGlobal (false), FunctionEffect(0) {}
  };

  /// GlobalsModRef - The actual analysis pass.
  class VISIBILITY_HIDDEN GlobalsModRef
      : public ModulePass, public AliasAnalysis {
    /// NonAddressTakenGlobals - The globals that do not have their addresses
    /// taken.
    std::set<GlobalValue*> NonAddressTakenGlobals;

    /// IndirectGlobals - The memory pointed to by this global is known to be
    /// 'owned' by the global.
    std::set<GlobalValue*> IndirectGlobals;

    /// AllocsForIndirectGlobals - If an instruction allocates memory for an
    /// indirect global, this map indicates which one.
    std::map<Value*, GlobalValue*> AllocsForIndirectGlobals;

    /// FunctionInfo - For each function, keep track of what globals are
    /// modified or read.
    std::map<Function*, FunctionRecord> FunctionInfo;

  public:
    static char ID;
    GlobalsModRef() : ModulePass(&ID) {}

    bool runOnModule(Module &M) {
      InitializeAliasAnalysis(this);                 // set up super class
      AnalyzeGlobals(M);                          // find non-addr taken globals
      AnalyzeCallGraph(getAnalysis<CallGraph>(), M); // Propagate on CG
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.addRequired<CallGraph>();
      AU.setPreservesAll();                         // Does not transform code
    }

    //------------------------------------------------
    // Implement the AliasAnalysis API
    //
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);
    ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);
    ModRefResult getModRefInfo(CallSite CS1, CallSite CS2) {
      return AliasAnalysis::getModRefInfo(CS1,CS2);
    }
    bool hasNoModRefInfoForCalls() const { return false; }

    /// getModRefBehavior - Return the behavior of the specified function if
    /// called from the specified call site.  The call site may be null in which
    /// case the most generic behavior of this function should be returned.
    virtual ModRefBehavior getModRefBehavior(Function *F, CallSite CS,
                                         std::vector<PointerAccessInfo> *Info) {
      if (FunctionRecord *FR = getFunctionInfo(F)) {
        if (FR->FunctionEffect == 0)
          return DoesNotAccessMemory;
        else if ((FR->FunctionEffect & Mod) == 0)
          return OnlyReadsMemory;
      }
      return AliasAnalysis::getModRefBehavior(F, CS, Info);
    }

    virtual void deleteValue(Value *V);
    virtual void copyValue(Value *From, Value *To);

  private:
    /// getFunctionInfo - Return the function info for the function, or null if
    /// we don't have anything useful to say about it.
    FunctionRecord *getFunctionInfo(Function *F) {
      std::map<Function*, FunctionRecord>::iterator I = FunctionInfo.find(F);
      if (I != FunctionInfo.end())
        return &I->second;
      return 0;
    }

    void AnalyzeGlobals(Module &M);
    void AnalyzeCallGraph(CallGraph &CG, Module &M);
    bool AnalyzeUsesOfPointer(Value *V, std::vector<Function*> &Readers,
                              std::vector<Function*> &Writers,
                              GlobalValue *OkayStoreDest = 0);
    bool AnalyzeIndirectGlobalMemory(GlobalValue *GV);
  };
}

char GlobalsModRef::ID = 0;
static RegisterPass<GlobalsModRef>
X("globalsmodref-aa", "Simple mod/ref analysis for globals", false, true);
static RegisterAnalysisGroup<AliasAnalysis> Y(X);

Pass *llvm::createGlobalsModRefPass() { return new GlobalsModRef(); }

/// AnalyzeGlobals - Scan through the users of all of the internal
/// GlobalValue's in the program.  If none of them have their "address taken"
/// (really, their address passed to something nontrivial), record this fact,
/// and record the functions that they are used directly in.
void GlobalsModRef::AnalyzeGlobals(Module &M) {
  std::vector<Function*> Readers, Writers;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->hasLocalLinkage()) {
      if (!AnalyzeUsesOfPointer(I, Readers, Writers)) {
        // Remember that we are tracking this global.
        NonAddressTakenGlobals.insert(I);
        ++NumNonAddrTakenFunctions;
      }
      Readers.clear(); Writers.clear();
    }

  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    if (I->hasLocalLinkage()) {
      if (!AnalyzeUsesOfPointer(I, Readers, Writers)) {
        // Remember that we are tracking this global, and the mod/ref fns
        NonAddressTakenGlobals.insert(I);

        for (unsigned i = 0, e = Readers.size(); i != e; ++i)
          FunctionInfo[Readers[i]].GlobalInfo[I] |= Ref;

        if (!I->isConstant())  // No need to keep track of writers to constants
          for (unsigned i = 0, e = Writers.size(); i != e; ++i)
            FunctionInfo[Writers[i]].GlobalInfo[I] |= Mod;
        ++NumNonAddrTakenGlobalVars;

        // If this global holds a pointer type, see if it is an indirect global.
        if (isa<PointerType>(I->getType()->getElementType()) &&
            AnalyzeIndirectGlobalMemory(I))
          ++NumIndirectGlobalVars;
      }
      Readers.clear(); Writers.clear();
    }
}

/// AnalyzeUsesOfPointer - Look at all of the users of the specified pointer.
/// If this is used by anything complex (i.e., the address escapes), return
/// true.  Also, while we are at it, keep track of those functions that read and
/// write to the value.
///
/// If OkayStoreDest is non-null, stores into this global are allowed.
bool GlobalsModRef::AnalyzeUsesOfPointer(Value *V,
                                         std::vector<Function*> &Readers,
                                         std::vector<Function*> &Writers,
                                         GlobalValue *OkayStoreDest) {
  if (!isa<PointerType>(V->getType())) return true;

  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      Readers.push_back(LI->getParent()->getParent());
    } else if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (V == SI->getOperand(1)) {
        Writers.push_back(SI->getParent()->getParent());
      } else if (SI->getOperand(1) != OkayStoreDest) {
        return true;  // Storing the pointer
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(*UI)) {
      if (AnalyzeUsesOfPointer(GEP, Readers, Writers)) return true;
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
        if (AnalyzeUsesOfPointer(CE, Readers, Writers))
          return true;
      } else {
        return true;
      }
    } else if (ICmpInst *ICI = dyn_cast<ICmpInst>(*UI)) {
      if (!isa<ConstantPointerNull>(ICI->getOperand(1)))
        return true;  // Allow comparison against null.
    } else if (FreeInst *F = dyn_cast<FreeInst>(*UI)) {
      Writers.push_back(F->getParent()->getParent());
    } else {
      return true;
    }
  return false;
}

/// AnalyzeIndirectGlobalMemory - We found an non-address-taken global variable
/// which holds a pointer type.  See if the global always points to non-aliased
/// heap memory: that is, all initializers of the globals are allocations, and
/// those allocations have no use other than initialization of the global.
/// Further, all loads out of GV must directly use the memory, not store the
/// pointer somewhere.  If this is true, we consider the memory pointed to by
/// GV to be owned by GV and can disambiguate other pointers from it.
bool GlobalsModRef::AnalyzeIndirectGlobalMemory(GlobalValue *GV) {
  // Keep track of values related to the allocation of the memory, f.e. the
  // value produced by the malloc call and any casts.
  std::vector<Value*> AllocRelatedValues;

  // Walk the user list of the global.  If we find anything other than a direct
  // load or store, bail out.
  for (Value::use_iterator I = GV->use_begin(), E = GV->use_end(); I != E; ++I){
    if (LoadInst *LI = dyn_cast<LoadInst>(*I)) {
      // The pointer loaded from the global can only be used in simple ways:
      // we allow addressing of it and loading storing to it.  We do *not* allow
      // storing the loaded pointer somewhere else or passing to a function.
      std::vector<Function*> ReadersWriters;
      if (AnalyzeUsesOfPointer(LI, ReadersWriters, ReadersWriters))
        return false;  // Loaded pointer escapes.
      // TODO: Could try some IP mod/ref of the loaded pointer.
    } else if (StoreInst *SI = dyn_cast<StoreInst>(*I)) {
      // Storing the global itself.
      if (SI->getOperand(0) == GV) return false;

      // If storing the null pointer, ignore it.
      if (isa<ConstantPointerNull>(SI->getOperand(0)))
        continue;

      // Check the value being stored.
      Value *Ptr = SI->getOperand(0)->getUnderlyingObject();

      if (isa<MallocInst>(Ptr)) {
        // Okay, easy case.
      } else if (CallInst *CI = dyn_cast<CallInst>(Ptr)) {
        Function *F = CI->getCalledFunction();
        if (!F || !F->isDeclaration()) return false;     // Too hard to analyze.
        if (F->getName() != "calloc") return false;   // Not calloc.
      } else {
        return false;  // Too hard to analyze.
      }

      // Analyze all uses of the allocation.  If any of them are used in a
      // non-simple way (e.g. stored to another global) bail out.
      std::vector<Function*> ReadersWriters;
      if (AnalyzeUsesOfPointer(Ptr, ReadersWriters, ReadersWriters, GV))
        return false;  // Loaded pointer escapes.

      // Remember that this allocation is related to the indirect global.
      AllocRelatedValues.push_back(Ptr);
    } else {
      // Something complex, bail out.
      return false;
    }
  }

  // Okay, this is an indirect global.  Remember all of the allocations for
  // this global in AllocsForIndirectGlobals.
  while (!AllocRelatedValues.empty()) {
    AllocsForIndirectGlobals[AllocRelatedValues.back()] = GV;
    AllocRelatedValues.pop_back();
  }
  IndirectGlobals.insert(GV);
  return true;
}

/// AnalyzeCallGraph - At this point, we know the functions where globals are
/// immediately stored to and read from.  Propagate this information up the call
/// graph to all callers and compute the mod/ref info for all memory for each
/// function.
void GlobalsModRef::AnalyzeCallGraph(CallGraph &CG, Module &M) {
  // We do a bottom-up SCC traversal of the call graph.  In other words, we
  // visit all callees before callers (leaf-first).
  for (scc_iterator<CallGraph*> I = scc_begin(&CG), E = scc_end(&CG); I != E;
       ++I) {
    std::vector<CallGraphNode *> &SCC = *I;
    assert(!SCC.empty() && "SCC with no functions?");

    if (!SCC[0]->getFunction()) {
      // Calls externally - can't say anything useful.  Remove any existing
      // function records (may have been created when scanning globals).
      for (unsigned i = 0, e = SCC.size(); i != e; ++i)
        FunctionInfo.erase(SCC[i]->getFunction());
      continue;
    }

    FunctionRecord &FR = FunctionInfo[SCC[0]->getFunction()];

    bool KnowNothing = false;
    unsigned FunctionEffect = 0;

    // Collect the mod/ref properties due to called functions.  We only compute
    // one mod-ref set.
    for (unsigned i = 0, e = SCC.size(); i != e && !KnowNothing; ++i) {
      Function *F = SCC[i]->getFunction();
      if (!F) {
        KnowNothing = true;
        break;
      }

      if (F->isDeclaration()) {
        // Try to get mod/ref behaviour from function attributes.
        if (F->doesNotAccessMemory()) {
          // Can't do better than that!
        } else if (F->onlyReadsMemory()) {
          FunctionEffect |= Ref;
          if (!F->isIntrinsic())
            // This function might call back into the module and read a global -
            // consider every global as possibly being read by this function.
            FR.MayReadAnyGlobal = true;
        } else {
          FunctionEffect |= ModRef;
          // Can't say anything useful unless it's an intrinsic - they don't
          // read or write global variables of the kind considered here.
          KnowNothing = !F->isIntrinsic();
        }
        continue;
      }

      for (CallGraphNode::iterator CI = SCC[i]->begin(), E = SCC[i]->end();
           CI != E && !KnowNothing; ++CI)
        if (Function *Callee = CI->second->getFunction()) {
          if (FunctionRecord *CalleeFR = getFunctionInfo(Callee)) {
            // Propagate function effect up.
            FunctionEffect |= CalleeFR->FunctionEffect;

            // Incorporate callee's effects on globals into our info.
            for (std::map<GlobalValue*, unsigned>::iterator GI =
                   CalleeFR->GlobalInfo.begin(), E = CalleeFR->GlobalInfo.end();
                 GI != E; ++GI)
              FR.GlobalInfo[GI->first] |= GI->second;
            FR.MayReadAnyGlobal |= CalleeFR->MayReadAnyGlobal;
          } else {
            // Can't say anything about it.  However, if it is inside our SCC,
            // then nothing needs to be done.
            CallGraphNode *CalleeNode = CG[Callee];
            if (std::find(SCC.begin(), SCC.end(), CalleeNode) == SCC.end())
              KnowNothing = true;
          }
        } else {
          KnowNothing = true;
        }
    }

    // If we can't say anything useful about this SCC, remove all SCC functions
    // from the FunctionInfo map.
    if (KnowNothing) {
      for (unsigned i = 0, e = SCC.size(); i != e; ++i)
        FunctionInfo.erase(SCC[i]->getFunction());
      continue;
    }

    // Scan the function bodies for explicit loads or stores.
    for (unsigned i = 0, e = SCC.size(); i != e && FunctionEffect != ModRef;++i)
      for (inst_iterator II = inst_begin(SCC[i]->getFunction()),
             E = inst_end(SCC[i]->getFunction());
           II != E && FunctionEffect != ModRef; ++II)
        if (isa<LoadInst>(*II)) {
          FunctionEffect |= Ref;
          if (cast<LoadInst>(*II).isVolatile())
            // Volatile loads may have side-effects, so mark them as writing
            // memory (for example, a flag inside the processor).
            FunctionEffect |= Mod;
        } else if (isa<StoreInst>(*II)) {
          FunctionEffect |= Mod;
          if (cast<StoreInst>(*II).isVolatile())
            // Treat volatile stores as reading memory somewhere.
            FunctionEffect |= Ref;
        } else if (isa<MallocInst>(*II) || isa<FreeInst>(*II)) {
          FunctionEffect |= ModRef;
        }

    if ((FunctionEffect & Mod) == 0)
      ++NumReadMemFunctions;
    if (FunctionEffect == 0)
      ++NumNoMemFunctions;
    FR.FunctionEffect = FunctionEffect;

    // Finally, now that we know the full effect on this SCC, clone the
    // information to each function in the SCC.
    for (unsigned i = 1, e = SCC.size(); i != e; ++i)
      FunctionInfo[SCC[i]->getFunction()] = FR;
  }
}



/// alias - If one of the pointers is to a global that we are tracking, and the
/// other is some random pointer, we know there cannot be an alias, because the
/// address of the global isn't taken.
AliasAnalysis::AliasResult
GlobalsModRef::alias(const Value *V1, unsigned V1Size,
                     const Value *V2, unsigned V2Size) {
  // Get the base object these pointers point to.
  Value *UV1 = const_cast<Value*>(V1->getUnderlyingObject());
  Value *UV2 = const_cast<Value*>(V2->getUnderlyingObject());

  // If either of the underlying values is a global, they may be non-addr-taken
  // globals, which we can answer queries about.
  GlobalValue *GV1 = dyn_cast<GlobalValue>(UV1);
  GlobalValue *GV2 = dyn_cast<GlobalValue>(UV2);
  if (GV1 || GV2) {
    // If the global's address is taken, pretend we don't know it's a pointer to
    // the global.
    if (GV1 && !NonAddressTakenGlobals.count(GV1)) GV1 = 0;
    if (GV2 && !NonAddressTakenGlobals.count(GV2)) GV2 = 0;

    // If the the two pointers are derived from two different non-addr-taken
    // globals, or if one is and the other isn't, we know these can't alias.
    if ((GV1 || GV2) && GV1 != GV2)
      return NoAlias;

    // Otherwise if they are both derived from the same addr-taken global, we
    // can't know the two accesses don't overlap.
  }

  // These pointers may be based on the memory owned by an indirect global.  If
  // so, we may be able to handle this.  First check to see if the base pointer
  // is a direct load from an indirect global.
  GV1 = GV2 = 0;
  if (LoadInst *LI = dyn_cast<LoadInst>(UV1))
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(LI->getOperand(0)))
      if (IndirectGlobals.count(GV))
        GV1 = GV;
  if (LoadInst *LI = dyn_cast<LoadInst>(UV2))
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(LI->getOperand(0)))
      if (IndirectGlobals.count(GV))
        GV2 = GV;

  // These pointers may also be from an allocation for the indirect global.  If
  // so, also handle them.
  if (AllocsForIndirectGlobals.count(UV1))
    GV1 = AllocsForIndirectGlobals[UV1];
  if (AllocsForIndirectGlobals.count(UV2))
    GV2 = AllocsForIndirectGlobals[UV2];

  // Now that we know whether the two pointers are related to indirect globals,
  // use this to disambiguate the pointers.  If either pointer is based on an
  // indirect global and if they are not both based on the same indirect global,
  // they cannot alias.
  if ((GV1 || GV2) && GV1 != GV2)
    return NoAlias;

  return AliasAnalysis::alias(V1, V1Size, V2, V2Size);
}

AliasAnalysis::ModRefResult
GlobalsModRef::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  unsigned Known = ModRef;

  // If we are asking for mod/ref info of a direct call with a pointer to a
  // global we are tracking, return information if we have it.
  if (GlobalValue *GV = dyn_cast<GlobalValue>(P->getUnderlyingObject()))
    if (GV->hasLocalLinkage())
      if (Function *F = CS.getCalledFunction())
        if (NonAddressTakenGlobals.count(GV))
          if (FunctionRecord *FR = getFunctionInfo(F))
            Known = FR->getInfoForGlobal(GV);

  if (Known == NoModRef)
    return NoModRef; // No need to query other mod/ref analyses
  return ModRefResult(Known & AliasAnalysis::getModRefInfo(CS, P, Size));
}


//===----------------------------------------------------------------------===//
// Methods to update the analysis as a result of the client transformation.
//
void GlobalsModRef::deleteValue(Value *V) {
  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    if (NonAddressTakenGlobals.erase(GV)) {
      // This global might be an indirect global.  If so, remove it and remove
      // any AllocRelatedValues for it.
      if (IndirectGlobals.erase(GV)) {
        // Remove any entries in AllocsForIndirectGlobals for this global.
        for (std::map<Value*, GlobalValue*>::iterator
             I = AllocsForIndirectGlobals.begin(),
             E = AllocsForIndirectGlobals.end(); I != E; ) {
          if (I->second == GV) {
            AllocsForIndirectGlobals.erase(I++);
          } else {
            ++I;
          }
        }
      }
    }
  }

  // Otherwise, if this is an allocation related to an indirect global, remove
  // it.
  AllocsForIndirectGlobals.erase(V);

  AliasAnalysis::deleteValue(V);
}

void GlobalsModRef::copyValue(Value *From, Value *To) {
  AliasAnalysis::copyValue(From, To);
}
