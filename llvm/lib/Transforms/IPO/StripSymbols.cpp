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
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN StripSymbols : public ModulePass {
    bool OnlyDebugInfo;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit StripSymbols(bool ODI = false) 
      : ModulePass(&ID), OnlyDebugInfo(ODI) {}

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };

  class VISIBILITY_HIDDEN StripNonDebugSymbols : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit StripNonDebugSymbols()
      : ModulePass(&ID) {}

    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }
  };
}

char StripSymbols::ID = 0;
static RegisterPass<StripSymbols>
X("strip", "Strip all symbols from a module");

ModulePass *llvm::createStripSymbolsPass(bool OnlyDebugInfo) {
  return new StripSymbols(OnlyDebugInfo);
}

char StripNonDebugSymbols::ID = 0;
static RegisterPass<StripNonDebugSymbols>
Y("strip-nondebug", "Strip all symbols, except dbg symbols, from a module");

ModulePass *llvm::createStripNonDebugSymbolsPass() {
  return new StripNonDebugSymbols();
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
  SmallPtrSet<Constant *, 4> Operands;
  for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i)
    if (isa<DerivedType>(C->getOperand(i)->getType()) &&
        OnlyUsedBy(C->getOperand(i), C)) 
      Operands.insert(C->getOperand(i));
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
    if (!GV->hasLocalLinkage()) return;   // Don't delete non static globals.
    GV->eraseFromParent();
  }
  else if (!isa<Function>(C))
    if (isa<CompositeType>(C->getType()))
      C->destroyConstant();

  // If the constant referenced anything, see if we can delete it as well.
  for (SmallPtrSet<Constant *, 4>::iterator OI = Operands.begin(),
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
      if (!PreserveDbgInfo || strncmp(V->getNameStart(), "llvm.dbg", 8))
        // Set name to "", removing from symbol table!
        V->setName("");
    }
  }
}

// Strip the symbol table of its names.
static void StripTypeSymtab(TypeSymbolTable &ST, bool PreserveDbgInfo) {
  for (TypeSymbolTable::iterator TI = ST.begin(), E = ST.end(); TI != E; ) {
    if (PreserveDbgInfo && strncmp(TI->first.c_str(), "llvm.dbg", 8) == 0)
      ++TI;
    else
      ST.remove(TI++);
  }
}

/// Find values that are marked as llvm.used.
void findUsedValues(Module &M,
                    SmallPtrSet<const GlobalValue*, 8>& llvmUsedValues) {
  if (GlobalVariable *LLVMUsed = M.getGlobalVariable("llvm.used")) {
    llvmUsedValues.insert(LLVMUsed);
    // Collect values that are preserved as per explicit request.
    // llvm.used is used to list these values.
    if (ConstantArray *Inits = 
        dyn_cast<ConstantArray>(LLVMUsed->getInitializer())) {
      for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i) {
        if (GlobalValue *GV = dyn_cast<GlobalValue>(Inits->getOperand(i)))
          llvmUsedValues.insert(GV);
        else if (ConstantExpr *CE =
                 dyn_cast<ConstantExpr>(Inits->getOperand(i)))
          if (CE->getOpcode() == Instruction::BitCast)
            if (GlobalValue *GV = dyn_cast<GlobalValue>(CE->getOperand(0)))
              llvmUsedValues.insert(GV);
      }
    }
  }
}

/// StripSymbolNames - Strip symbol names.
bool StripSymbolNames(Module &M, bool PreserveDbgInfo) {

  SmallPtrSet<const GlobalValue*, 8> llvmUsedValues;
  findUsedValues(M, llvmUsedValues);

  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->hasLocalLinkage() && llvmUsedValues.count(I) == 0)
      if (!PreserveDbgInfo || strncmp(I->getNameStart(), "llvm.dbg", 8))
        I->setName("");     // Internal symbols can't participate in linkage
  }
  
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    if (I->hasLocalLinkage() && llvmUsedValues.count(I) == 0)
      if (!PreserveDbgInfo || strncmp(I->getNameStart(), "llvm.dbg", 8))
        I->setName("");     // Internal symbols can't participate in linkage
    StripSymtab(I->getValueSymbolTable(), PreserveDbgInfo);
  }
  
  // Remove all names from types.
  StripTypeSymtab(M.getTypeSymbolTable(), PreserveDbgInfo);

  return true;
}

// StripDebugInfo - Strip debug info in the module if it exists.  
// To do this, we remove llvm.dbg.func.start, llvm.dbg.stoppoint, and 
// llvm.dbg.region.end calls, and any globals they point to if now dead.
bool StripDebugInfo(Module &M) {

  Function *FuncStart = M.getFunction("llvm.dbg.func.start");
  Function *StopPoint = M.getFunction("llvm.dbg.stoppoint");
  Function *RegionStart = M.getFunction("llvm.dbg.region.start");
  Function *RegionEnd = M.getFunction("llvm.dbg.region.end");
  Function *Declare = M.getFunction("llvm.dbg.declare");

  std::vector<Constant*> DeadConstants;

  // Remove all of the calls to the debugger intrinsics, and remove them from
  // the module.
  if (FuncStart) {
    while (!FuncStart->use_empty()) {
      CallInst *CI = cast<CallInst>(FuncStart->use_back());
      Value *Arg = CI->getOperand(1);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (Constant *C = dyn_cast<Constant>(Arg)) 
          DeadConstants.push_back(C);
    }
    FuncStart->eraseFromParent();
  }
  if (StopPoint) {
    while (!StopPoint->use_empty()) {
      CallInst *CI = cast<CallInst>(StopPoint->use_back());
      Value *Arg = CI->getOperand(3);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (Constant *C = dyn_cast<Constant>(Arg)) 
          DeadConstants.push_back(C);
    }
    StopPoint->eraseFromParent();
  }
  if (RegionStart) {
    while (!RegionStart->use_empty()) {
      CallInst *CI = cast<CallInst>(RegionStart->use_back());
      Value *Arg = CI->getOperand(1);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (Constant *C = dyn_cast<Constant>(Arg)) 
          DeadConstants.push_back(C);
    }
    RegionStart->eraseFromParent();
  }
  if (RegionEnd) {
    while (!RegionEnd->use_empty()) {
      CallInst *CI = cast<CallInst>(RegionEnd->use_back());
      Value *Arg = CI->getOperand(1);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg->use_empty())
        if (Constant *C = dyn_cast<Constant>(Arg)) 
          DeadConstants.push_back(C);
    }
    RegionEnd->eraseFromParent();
  }
  if (Declare) {
    while (!Declare->use_empty()) {
      CallInst *CI = cast<CallInst>(Declare->use_back());
      Value *Arg1 = CI->getOperand(1);
      Value *Arg2 = CI->getOperand(2);
      assert(CI->use_empty() && "llvm.dbg intrinsic should have void result");
      CI->eraseFromParent();
      if (Arg1->use_empty()) {
        if (Constant *C = dyn_cast<Constant>(Arg1)) 
          DeadConstants.push_back(C);
        if (Instruction *I = dyn_cast<Instruction>(Arg1))
          I->eraseFromParent();
      }
      if (Arg2->use_empty())
        if (Constant *C = dyn_cast<Constant>(Arg2)) 
          DeadConstants.push_back(C);
    }
    Declare->eraseFromParent();
  }

  SmallPtrSet<const GlobalValue*, 8> llvmUsedValues;
  findUsedValues(M, llvmUsedValues);

  // llvm.dbg.compile_units and llvm.dbg.subprograms are marked as linkonce
  // but since we are removing all debug information, make them internal now.
  // FIXME: Use private linkage maybe?
  if (Constant *C = M.getNamedGlobal("llvm.dbg.compile_units"))
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
      GV->setLinkage(GlobalValue::InternalLinkage);

  if (Constant *C = M.getNamedGlobal("llvm.dbg.subprograms"))
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
      GV->setLinkage(GlobalValue::InternalLinkage);
 
  if (Constant *C = M.getNamedGlobal("llvm.dbg.global_variables"))
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
      GV->setLinkage(GlobalValue::InternalLinkage);

  // Delete all dbg variables.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end(); 
       I != E; ++I) {
    GlobalVariable *GV = dyn_cast<GlobalVariable>(I);
    if (!GV) continue;
    if (GV->use_empty() && llvmUsedValues.count(I) == 0
        && (!GV->hasSection() 
            || strcmp(GV->getSection().c_str(), "llvm.metadata") == 0))
      DeadConstants.push_back(GV);
  }

  if (DeadConstants.empty())
    return false;

  // Delete any internal globals that were only used by the debugger intrinsics.
  while (!DeadConstants.empty()) {
    Constant *C = DeadConstants.back();
    DeadConstants.pop_back();
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
      if (GV->hasLocalLinkage())
        RemoveDeadConstant(GV);
    }
    else
      RemoveDeadConstant(C);
  }

  // Remove all llvm.dbg types.
  TypeSymbolTable &ST = M.getTypeSymbolTable();
  for (TypeSymbolTable::iterator TI = ST.begin(), TE = ST.end(); TI != TE; ) {
    if (!strncmp(TI->first.c_str(), "llvm.dbg.", 9))
      ST.remove(TI++);
    else 
      ++TI;
  }
  
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
