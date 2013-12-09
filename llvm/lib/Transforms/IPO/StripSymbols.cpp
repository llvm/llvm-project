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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"
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

  ConstantArray *Inits = cast<ConstantArray>(LLVMUsed->getInitializer());

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

/// Remove any debug info for global variables/functions in the given module for
/// which said global variable/function no longer exists (i.e. is null).
///
/// Debugging information is encoded in llvm IR using metadata. This is designed
/// such a way that debug info for symbols preserved even if symbols are
/// optimized away by the optimizer. This special pass removes debug info for
/// such symbols.
bool StripDeadDebugInfo::runOnModule(Module &M) {
  bool Changed = false;

  LLVMContext &C = M.getContext();

  // Find all debug info in F. This is actually overkill in terms of what we
  // want to do, but we want to try and be as resilient as possible in the face
  // of potential debug info changes by using the formal interfaces given to us
  // as much as possible.
  DebugInfoFinder F;
  F.processModule(M);

  // For each compile unit, find the live set of global variables/functions and
  // replace the current list of potentially dead global variables/functions
  // with the live list.
  SmallVector<Value *, 64> LiveGlobalVariables;
  SmallVector<Value *, 64> LiveSubprograms;
  DenseSet<const MDNode *> VisitedSet;

  for (DebugInfoFinder::iterator CI = F.compile_unit_begin(),
         CE = F.compile_unit_end(); CI != CE; ++CI) {
    // Create our compile unit.
    DICompileUnit DIC(*CI);
    assert(DIC.Verify() && "DIC must verify as a DICompileUnit.");

    // Create our live subprogram list.
    DIArray SPs = DIC.getSubprograms();
    bool SubprogramChange = false;
    for (unsigned i = 0, e = SPs.getNumElements(); i != e; ++i) {
      DISubprogram DISP(SPs.getElement(i));
      assert(DISP.Verify() && "DISP must verify as a DISubprogram.");

      // Make sure we visit each subprogram only once.
      if (!VisitedSet.insert(DISP).second)
        continue;

      // If the function referenced by DISP is not null, the function is live.
      if (DISP.getFunction())
        LiveSubprograms.push_back(DISP);
      else
        SubprogramChange = true;
    }

    // Create our live global variable list.
    DIArray GVs = DIC.getGlobalVariables();
    bool GlobalVariableChange = false;
    for (unsigned i = 0, e = GVs.getNumElements(); i != e; ++i) {
      DIGlobalVariable DIG(GVs.getElement(i));
      assert(DIG.Verify() && "DIG must verify as DIGlobalVariable.");

      // Make sure we only visit each global variable only once.
      if (!VisitedSet.insert(DIG).second)
        continue;

      // If the global variable referenced by DIG is not null, the global
      // variable is live.
      if (DIG.getGlobal())
        LiveGlobalVariables.push_back(DIG);
      else
        GlobalVariableChange = true;
    }

    // If we found dead subprograms or global variables, replace the current
    // subprogram list/global variable list with our new live subprogram/global
    // variable list.
    if (SubprogramChange) {
      // Make sure that 9 is still the index of the subprograms. This is to make
      // sure that an assert is hit if the location of the subprogram array
      // changes. This is just to make sure that this is updated if such an
      // event occurs.
      assert(DIC->getNumOperands() >= 10 &&
             SPs == DIC->getOperand(9) &&
             "DICompileUnits is expected to store Subprograms in operand "
             "9.");
      DIC->replaceOperandWith(9, MDNode::get(C, LiveSubprograms));
      Changed = true;
    }

    if (GlobalVariableChange) {
      // Make sure that 10 is still the index of global variables. This is to
      // make sure that an assert is hit if the location of the subprogram array
      // changes. This is just to make sure that this index is updated if such
      // an event occurs.
      assert(DIC->getNumOperands() >= 11 &&
             GVs == DIC->getOperand(10) &&
             "DICompileUnits is expected to store Global Variables in operand "
             "10.");
      DIC->replaceOperandWith(10, MDNode::get(C, LiveGlobalVariables));
      Changed = true;
    }

    // Reset lists for the next iteration.
    LiveSubprograms.clear();
    LiveGlobalVariables.clear();
  }

  return Changed;
}
