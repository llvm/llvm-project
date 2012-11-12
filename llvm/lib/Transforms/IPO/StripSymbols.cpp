//===- StripSymbols.cpp - Strip symbols and debug info from a module ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The StripSymbols transformation implements code stripping. Specifically, it
// can delete:
// 
//   * names for virtual registers
//   * symbols for internal globals and functions
//   * debug information
//
// Note that this transformation makes code much less readable, so it should
// only be used in situations where the 'strip' utility would be used, such as
// reducing code size or making it harder to reverse engineer code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/DebugInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/TypeFinder.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace llvm;

namespace {
  class StripSymbols : public ModulePass {
    bool OnlyDebugInfo;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit StripSymbols(bool ODI = false) 
      : ModulePass(ID), OnlyDebugInfo(ODI) {
        initializeStripSymbolsPass(*PassRegistry::getPassRegistry());
      }

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  class StripNonDebugSymbols : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit StripNonDebugSymbols()
      : ModulePass(ID) {
        initializeStripNonDebugSymbolsPass(*PassRegistry::getPassRegistry());
      }

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  class StripDebugDeclare : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit StripDebugDeclare()
      : ModulePass(ID) {
        initializeStripDebugDeclarePass(*PassRegistry::getPassRegistry());
      }

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  class StripDeadDebugInfo : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit StripDeadDebugInfo()
      : ModulePass(ID) {
        initializeStripDeadDebugInfoPass(*PassRegistry::getPassRegistry());
      }

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char StripSymbols::ID = 0;
INITIALIZE_PASS(StripSymbols, "strip",
                "Strip all symbols from a module", false, false)

ModulePass *llvm::createStripSymbolsPass(bool OnlyDebugInfo) {
  return new StripSymbols(OnlyDebugInfo);
}

char StripNonDebugSymbols::ID = 0;
INITIALIZE_PASS(StripNonDebugSymbols, "strip-nondebug",
                "Strip all symbols, except dbg symbols, from a module",
                false, false)

ModulePass *llvm::createStripNonDebugSymbolsPass() {
  return new StripNonDebugSymbols();
}

char StripDebugDeclare::ID = 0;
INITIALIZE_PASS(StripDebugDeclare, "strip-debug-declare",
                "Strip all llvm.dbg.declare intrinsics", false, false)

ModulePass *llvm::createStripDebugDeclarePass() {
  return new StripDebugDeclare();
}

char StripDeadDebugInfo::ID = 0;
INITIALIZE_PASS(StripDeadDebugInfo, "strip-dead-debug-info",
                "Strip debug info for unused symbols", false, false)

ModulePass *llvm::createStripDeadDebugInfoPass() {
  return new StripDeadDebugInfo();
}

/// OnlyUsedBy - Return true if V is only used by Usr.
static bool OnlyUsedBy(Value *V, Value *Usr) {
  for(Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I) {
    User *U = *I;
    if (U != Usr)
      return false;
  }
  return true;
}

static void RemoveDeadConstant(Constant *C) {
  assert(C->use_empty() && "Constant is not dead!");
  SmallPtrSet<Constant*, 4> Operands;
  for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i)
    if (OnlyUsedBy(C->getOperand(i), C)) 
      Operands.insert(cast<Constant>(C->getOperand(i)));
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
    if (!GV->hasLocalLinkage()) return;   // Don't delete non static globals.
    GV->eraseFromParent();
  }
  else if (!isa<Function>(C))
    if (isa<CompositeType>(C->getType()))
      C->destroyConstant();

  // If the constant referenced anything, see if we can delete it as well.
  for (SmallPtrSet<Constant*, 4>::iterator OI = Operands.begin(),
         OE = Operands.end(); OI != OE; ++OI)
    RemoveDeadConstant(*OI);
}

// Strip the symbol table of its names.
//
static void StripSymtab(ValueSymbolTable &ST, bool PreserveDbgInfo) {
  for (ValueSymbolTable::iterator VI = ST.begin(), VE = ST.end(); VI != VE; ) {
    Value *V = VI->getValue();
    ++VI;
    if (!isa<GlobalValue>(V) || cast<GlobalValue>(V)->hasLocalLinkage()) {
      if (!PreserveDbgInfo || !V->getName().startswith("llvm.dbg"))
        // Set name to "", removing from symbol table!
        V->setName("");
    }
  }
}

// Strip any named types of their names.
static void StripTypeNames(Module &M, bool PreserveDbgInfo) {
  TypeFinder StructTypes;
  StructTypes.run(M, false);

  for (unsigned i = 0, e = StructTypes.size(); i != e; ++i) {
    StructType *STy = StructTypes[i];
    if (STy->isLiteral() || STy->getName().empty()) continue;
    
    if (PreserveDbgInfo && STy->getName().startswith("llvm.dbg"))
      continue;

    STy->setName("");
  }
}

/// Find values that are marked as llvm.used.
static void findUsedValues(GlobalVariable *LLVMUsed,
                           SmallPtrSet<const GlobalValue*, 8> &UsedValues) {
  if (LLVMUsed == 0) return;
  UsedValues.insert(LLVMUsed);
  
  ConstantArray *Inits = dyn_cast<ConstantArray>(LLVMUsed->getInitializer());
  if (Inits == 0) return;
  
  for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i)
    if (GlobalValue *GV = 
          dyn_cast<GlobalValue>(Inits->getOperand(i)->stripPointerCasts()))
      UsedValues.insert(GV);
}

/// StripSymbolNames - Strip symbol names.
static bool StripSymbolNames(Module &M, bool PreserveDbgInfo) {

  SmallPtrSet<const GlobalValue*, 8> llvmUsedValues;
  findUsedValues(M.getGlobalVariable("llvm.used"), llvmUsedValues);
  findUsedValues(M.getGlobalVariable("llvm.compiler.used"), llvmUsedValues);

  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->hasLocalLinkage() && llvmUsedValues.count(I) == 0)
      if (!PreserveDbgInfo || !I->getName().startswith("llvm.dbg"))
        I->setName("");     // Internal symbols can't participate in linkage
  }
  
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    if (I->hasLocalLinkage() && llvmUsedValues.count(I) == 0)
      if (!PreserveDbgInfo || !I->getName().startswith("llvm.dbg"))
        I->setName("");     // Internal symbols can't participate in linkage
    StripSymtab(I->getValueSymbolTable(), PreserveDbgInfo);
  }
  
  // Remove all names from types.
  StripTypeNames(M, PreserveDbgInfo);

  return true;
}

// StripDebugInfo - Strip debug info in the module if it exists.  
// To do this, we remove llvm.dbg.func.start, llvm.dbg.stoppoint, and 
// llvm.dbg.region.end calls, and any globals they point to if now dead.
static bool StripDebugInfo(Module &M) {

  bool Changed = false;

  // Remove all of the calls to the debugger intrinsics, and remove them from
  // the module.
  if (Function *Declare = M.getFunction("llvm.dbg.declare")) {
    while (!Declare->use_empty()) {
      CallInst *CI = cast<CallInst>(Declare->use_back());
      CI->eraseFromParent();
    }
    Declare->eraseFromParent();
    Changed = true;
  }

  if (Function *DbgVal = M.getFunction("llvm.dbg.value")) {
    while (!DbgVal->use_empty()) {
      CallInst *CI = cast<CallInst>(DbgVal->use_back());
      CI->eraseFromParent();
    }
    DbgVal->eraseFromParent();
    Changed = true;
  }

  for (Module::named_metadata_iterator NMI = M.named_metadata_begin(),
         NME = M.named_metadata_end(); NMI != NME;) {
    NamedMDNode *NMD = NMI;
    ++NMI;
    if (NMD->getName().startswith("llvm.dbg.")) {
      NMD->eraseFromParent();
      Changed = true;
    }
  }

  for (Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI)
    for (Function::iterator FI = MI->begin(), FE = MI->end(); FI != FE;
         ++FI)
      for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE;
           ++BI) {
        if (!BI->getDebugLoc().isUnknown()) {
          Changed = true;
          BI->setDebugLoc(DebugLoc());
        }
      }

  return Changed;
}

bool StripSymbols::runOnModule(Module &M) {
  bool Changed = false;
  Changed |= StripDebugInfo(M);
  if (!OnlyDebugInfo)
    Changed |= StripSymbolNames(M, false);
  return Changed;
}

bool StripNonDebugSymbols::runOnModule(Module &M) {
  return StripSymbolNames(M, true);
}

bool StripDebugDeclare::runOnModule(Module &M) {

  Function *Declare = M.getFunction("llvm.dbg.declare");
  std::vector<Constant*> DeadConstants;

  if (Declare) {
    while (!Declare->use_empty()) {
      CallInst *CI = cast<CallInst>(Declare->use_back());
      Value *Arg1 = CI->getArgOperand(0);
      Value *Arg2 = CI->getArgOperand(1);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg1->use_empty()) {
        if (Constant *C = dyn_cast<Constant>(Arg1)) 
          DeadConstants.push_back(C);
        else 
          RecursivelyDeleteTriviallyDeadInstructions(Arg1);
      }
      if (Arg2->use_empty())
        if (Constant *C = dyn_cast<Constant>(Arg2)) 
          DeadConstants.push_back(C);
    }
    Declare->eraseFromParent();
  }

  while (!DeadConstants.empty()) {
    Constant *C = DeadConstants.back();
    DeadConstants.pop_back();
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
      if (GV->hasLocalLinkage())
        RemoveDeadConstant(GV);
    } else
      RemoveDeadConstant(C);
  }

  return true;
}

/// getRealLinkageName - If special LLVM prefix that is used to inform the asm 
/// printer to not emit usual symbol prefix before the symbol name is used then
/// return linkage name after skipping this special LLVM prefix.
static StringRef getRealLinkageName(StringRef LinkageName) {
  char One = '\1';
  if (LinkageName.startswith(StringRef(&One, 1)))
    return LinkageName.substr(1);
  return LinkageName;
}

bool StripDeadDebugInfo::runOnModule(Module &M) {
  bool Changed = false;

  // Debugging infomration is encoded in llvm IR using metadata. This is designed
  // such a way that debug info for symbols preserved even if symbols are
  // optimized away by the optimizer. This special pass removes debug info for 
  // such symbols.

  // llvm.dbg.gv keeps track of debug info for global variables.
  if (NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.gv")) {
    SmallVector<MDNode *, 8> MDs;
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
      if (DIGlobalVariable(NMD->getOperand(i)).Verify())
        MDs.push_back(NMD->getOperand(i));
      else
        Changed = true;
    NMD->eraseFromParent();
    NMD = NULL;

    for (SmallVector<MDNode *, 8>::iterator I = MDs.begin(),
           E = MDs.end(); I != E; ++I) {
      GlobalVariable *GV = DIGlobalVariable(*I).getGlobal();
      if (GV && M.getGlobalVariable(GV->getName(), true)) {
        if (!NMD)
          NMD = M.getOrInsertNamedMetadata("llvm.dbg.gv");
        NMD->addOperand(*I);
      }
      else
        Changed = true;
    }
  }

  // llvm.dbg.sp keeps track of debug info for subprograms.
  if (NamedMDNode *NMD = M.getNamedMetadata("llvm.dbg.sp")) {
    SmallVector<MDNode *, 8> MDs;
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
      if (DISubprogram(NMD->getOperand(i)).Verify())
        MDs.push_back(NMD->getOperand(i));
      else
        Changed = true;
    NMD->eraseFromParent();
    NMD = NULL;

    for (SmallVector<MDNode *, 8>::iterator I = MDs.begin(),
           E = MDs.end(); I != E; ++I) {
      bool FnIsLive = false;
      if (Function *F = DISubprogram(*I).getFunction())
        if (M.getFunction(F->getName()))
          FnIsLive = true;
      if (FnIsLive) {
          if (!NMD)
            NMD = M.getOrInsertNamedMetadata("llvm.dbg.sp");
          NMD->addOperand(*I);
      } else {
        // Remove llvm.dbg.lv.fnname named mdnode which may have been used
        // to hold debug info for dead function's local variables.
        StringRef FName = DISubprogram(*I).getLinkageName();
        if (FName.empty())
          FName = DISubprogram(*I).getName();
        if (NamedMDNode *LVNMD = 
            M.getNamedMetadata(Twine("llvm.dbg.lv.", 
                                     getRealLinkageName(FName)))) 
          LVNMD->eraseFromParent();
      }
    }
  }

  return Changed;
}
