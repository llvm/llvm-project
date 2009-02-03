//===- ExtractFunction.cpp - Extract a function from Program --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements several methods that are used to extract functions,
// loops, or portions of a module from the rest of the module.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/System/Path.h"
#include "llvm/System/Signals.h"
#include <set>
#include <fstream>
#include <iostream>
using namespace llvm;

namespace llvm {
  bool DisableSimplifyCFG = false;
} // End llvm namespace

namespace {
  cl::opt<bool>
  NoDCE ("disable-dce",
         cl::desc("Do not use the -dce pass to reduce testcases"));
  cl::opt<bool, true>
  NoSCFG("disable-simplifycfg", cl::location(DisableSimplifyCFG),
         cl::desc("Do not use the -simplifycfg pass to reduce testcases"));
}

/// deleteInstructionFromProgram - This method clones the current Program and
/// deletes the specified instruction from the cloned module.  It then runs a
/// series of cleanup passes (ADCE and SimplifyCFG) to eliminate any code which
/// depends on the value.  The modified module is then returned.
///
Module *BugDriver::deleteInstructionFromProgram(const Instruction *I,
                                                unsigned Simplification) const {
  Module *Result = CloneModule(Program);

  const BasicBlock *PBB = I->getParent();
  const Function *PF = PBB->getParent();

  Module::iterator RFI = Result->begin(); // Get iterator to corresponding fn
  std::advance(RFI, std::distance(PF->getParent()->begin(),
                                  Module::const_iterator(PF)));

  Function::iterator RBI = RFI->begin();  // Get iterator to corresponding BB
  std::advance(RBI, std::distance(PF->begin(), Function::const_iterator(PBB)));

  BasicBlock::iterator RI = RBI->begin(); // Get iterator to corresponding inst
  std::advance(RI, std::distance(PBB->begin(), BasicBlock::const_iterator(I)));
  Instruction *TheInst = RI;              // Got the corresponding instruction!

  // If this instruction produces a value, replace any users with null values
  if (isa<StructType>(TheInst->getType()))
    TheInst->replaceAllUsesWith(UndefValue::get(TheInst->getType()));
  else if (TheInst->getType() != Type::VoidTy)
    TheInst->replaceAllUsesWith(Constant::getNullValue(TheInst->getType()));

  // Remove the instruction from the program.
  TheInst->getParent()->getInstList().erase(TheInst);

  
  //writeProgramToFile("current.bc", Result);
    
  // Spiff up the output a little bit.
  PassManager Passes;
  // Make sure that the appropriate target data is always used...
  Passes.add(new TargetData(Result));

  /// FIXME: If this used runPasses() like the methods below, we could get rid
  /// of the -disable-* options!
  if (Simplification > 1 && !NoDCE)
    Passes.add(createDeadCodeEliminationPass());
  if (Simplification && !DisableSimplifyCFG)
    Passes.add(createCFGSimplificationPass());      // Delete dead control flow

  Passes.add(createVerifierPass());
  Passes.run(*Result);
  return Result;
}

static const PassInfo *getPI(Pass *P) {
  const PassInfo *PI = P->getPassInfo();
  delete P;
  return PI;
}

/// performFinalCleanups - This method clones the current Program and performs
/// a series of cleanups intended to get rid of extra cruft on the module
/// before handing it to the user.
///
Module *BugDriver::performFinalCleanups(Module *M, bool MayModifySemantics) {
  // Make all functions external, so GlobalDCE doesn't delete them...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);

  std::vector<const PassInfo*> CleanupPasses;
  CleanupPasses.push_back(getPI(createGlobalDCEPass()));
  CleanupPasses.push_back(getPI(createDeadTypeEliminationPass()));

  if (MayModifySemantics)
    CleanupPasses.push_back(getPI(createDeadArgHackingPass()));
  else
    CleanupPasses.push_back(getPI(createDeadArgEliminationPass()));

  Module *New = runPassesOn(M, CleanupPasses);
  if (New == 0) {
    std::cerr << "Final cleanups failed.  Sorry. :(  Please report a bug!\n";
    return M;
  }
  delete M;
  return New;
}


/// ExtractLoop - Given a module, extract up to one loop from it into a new
/// function.  This returns null if there are no extractable loops in the
/// program or if the loop extractor crashes.
Module *BugDriver::ExtractLoop(Module *M) {
  std::vector<const PassInfo*> LoopExtractPasses;
  LoopExtractPasses.push_back(getPI(createSingleLoopExtractorPass()));

  Module *NewM = runPassesOn(M, LoopExtractPasses);
  if (NewM == 0) {
    Module *Old = swapProgramIn(M);
    std::cout << "*** Loop extraction failed: ";
    EmitProgressBitcode("loopextraction", true);
    std::cout << "*** Sorry. :(  Please report a bug!\n";
    swapProgramIn(Old);
    return 0;
  }

  // Check to see if we created any new functions.  If not, no loops were
  // extracted and we should return null.  Limit the number of loops we extract
  // to avoid taking forever.
  static unsigned NumExtracted = 32;
  if (M->size() == NewM->size() || --NumExtracted == 0) {
    delete NewM;
    return 0;
  } else {
    assert(M->size() < NewM->size() && "Loop extract removed functions?");
    Module::iterator MI = NewM->begin();
    for (unsigned i = 0, e = M->size(); i != e; ++i)
      ++MI;
  }

  return NewM;
}


// DeleteFunctionBody - "Remove" the function by deleting all of its basic
// blocks, making it external.
//
void llvm::DeleteFunctionBody(Function *F) {
  // delete the body of the function...
  F->deleteBody();
  assert(F->isDeclaration() && "This didn't make the function external!");
}

/// GetTorInit - Given a list of entries for static ctors/dtors, return them
/// as a constant array.
static Constant *GetTorInit(std::vector<std::pair<Function*, int> > &TorList) {
  assert(!TorList.empty() && "Don't create empty tor list!");
  std::vector<Constant*> ArrayElts;
  for (unsigned i = 0, e = TorList.size(); i != e; ++i) {
    std::vector<Constant*> Elts;
    Elts.push_back(ConstantInt::get(Type::Int32Ty, TorList[i].second));
    Elts.push_back(TorList[i].first);
    ArrayElts.push_back(ConstantStruct::get(Elts));
  }
  return ConstantArray::get(ArrayType::get(ArrayElts[0]->getType(), 
                                           ArrayElts.size()),
                            ArrayElts);
}

/// SplitStaticCtorDtor - A module was recently split into two parts, M1/M2, and
/// M1 has all of the global variables.  If M2 contains any functions that are
/// static ctors/dtors, we need to add an llvm.global_[cd]tors global to M2, and
/// prune appropriate entries out of M1s list.
static void SplitStaticCtorDtor(const char *GlobalName, Module *M1, Module *M2){
  GlobalVariable *GV = M1->getNamedGlobal(GlobalName);
  if (!GV || GV->isDeclaration() || GV->hasLocalLinkage() ||
      !GV->use_empty()) return;
  
  std::vector<std::pair<Function*, int> > M1Tors, M2Tors;
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!InitList) return;
  
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i))){
      if (CS->getNumOperands() != 2) return;  // Not array of 2-element structs.
      
      if (CS->getOperand(1)->isNullValue())
        break;  // Found a null terminator, stop here.
      
      ConstantInt *CI = dyn_cast<ConstantInt>(CS->getOperand(0));
      int Priority = CI ? CI->getSExtValue() : 0;
      
      Constant *FP = CS->getOperand(1);
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FP))
        if (CE->isCast())
          FP = CE->getOperand(0);
      if (Function *F = dyn_cast<Function>(FP)) {
        if (!F->isDeclaration())
          M1Tors.push_back(std::make_pair(F, Priority));
        else {
          // Map to M2's version of the function.
          F = M2->getFunction(F->getName());
          M2Tors.push_back(std::make_pair(F, Priority));
        }
      }
    }
  }
  
  GV->eraseFromParent();
  if (!M1Tors.empty()) {
    Constant *M1Init = GetTorInit(M1Tors);
    new GlobalVariable(M1Init->getType(), false, GlobalValue::AppendingLinkage,
                       M1Init, GlobalName, M1);
  }

  GV = M2->getNamedGlobal(GlobalName);
  assert(GV && "Not a clone of M1?");
  assert(GV->use_empty() && "llvm.ctors shouldn't have uses!");

  GV->eraseFromParent();
  if (!M2Tors.empty()) {
    Constant *M2Init = GetTorInit(M2Tors);
    new GlobalVariable(M2Init->getType(), false, GlobalValue::AppendingLinkage,
                       M2Init, GlobalName, M2);
  }
}


/// SplitFunctionsOutOfModule - Given a module and a list of functions in the
/// module, split the functions OUT of the specified module, and place them in
/// the new module.
Module *llvm::SplitFunctionsOutOfModule(Module *M,
                                        const std::vector<Function*> &F) {
  // Make sure functions & globals are all external so that linkage
  // between the two modules will work.
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    if (I->hasName() && *I->getNameStart() == '\01')
      I->setName(I->getNameStart()+1, I->getNameLen()-1);
    I->setLinkage(GlobalValue::ExternalLinkage);
  }

  Module *New = CloneModule(M);

  // Make sure global initializers exist only in the safe module (CBE->.so)
  for (Module::global_iterator I = New->global_begin(), E = New->global_end();
       I != E; ++I)
    I->setInitializer(0);  // Delete the initializer to make it external

  // Remove the Test functions from the Safe module
  std::set<std::pair<std::string, const PointerType*> > TestFunctions;
  for (unsigned i = 0, e = F.size(); i != e; ++i) {
    TestFunctions.insert(std::make_pair(F[i]->getName(), F[i]->getType()));  
    Function *TNOF = M->getFunction(F[i]->getName());
    assert(TNOF && "Function doesn't exist in module!");
    assert(TNOF->getFunctionType() == F[i]->getFunctionType() && "wrong type?");
    DEBUG(std::cerr << "Removing function " << F[i]->getName() << "\n");
    DeleteFunctionBody(TNOF);       // Function is now external in this module!
  }

  
  // Remove the Safe functions from the Test module
  for (Module::iterator I = New->begin(), E = New->end(); I != E; ++I)
    if (!TestFunctions.count(std::make_pair(I->getName(), I->getType())))
      DeleteFunctionBody(I);
  

  // Make sure that there is a global ctor/dtor array in both halves of the
  // module if they both have static ctor/dtor functions.
  SplitStaticCtorDtor("llvm.global_ctors", M, New);
  SplitStaticCtorDtor("llvm.global_dtors", M, New);
  
  return New;
}

//===----------------------------------------------------------------------===//
// Basic Block Extraction Code
//===----------------------------------------------------------------------===//

/// ExtractMappedBlocksFromModule - Extract all but the specified basic blocks
/// into their own functions.  The only detail is that M is actually a module
/// cloned from the one the BBs are in, so some mapping needs to be performed.
/// If this operation fails for some reason (ie the implementation is buggy),
/// this function should return null, otherwise it returns a new Module.
Module *BugDriver::ExtractMappedBlocksFromModule(const
                                                 std::vector<BasicBlock*> &BBs,
                                                 Module *M) {
  char *ExtraArg = NULL;

  sys::Path uniqueFilename("bugpoint-extractblocks");
  std::string ErrMsg;
  if (uniqueFilename.createTemporaryFileOnDisk(true, &ErrMsg)) {
    std::cout << "*** Basic Block extraction failed!\n";
    std::cerr << "Error creating temporary file: " << ErrMsg << "\n";
    M = swapProgramIn(M);
    EmitProgressBitcode("basicblockextractfail", true);
    swapProgramIn(M);
    return 0;
  }
  sys::RemoveFileOnSignal(uniqueFilename);

  std::ofstream BlocksToNotExtractFile(uniqueFilename.c_str());
  if (!BlocksToNotExtractFile) {
    std::cout << "*** Basic Block extraction failed!\n";
    std::cerr << "Error writing list of blocks to not extract: " << ErrMsg
              << "\n";
    M = swapProgramIn(M);
    EmitProgressBitcode("basicblockextractfail", true);
    swapProgramIn(M);
    return 0;
  }
  for (std::vector<BasicBlock*>::const_iterator I = BBs.begin(), E = BBs.end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    // If the BB doesn't have a name, give it one so we have something to key
    // off of.
    if (!BB->hasName()) BB->setName("tmpbb");
    BlocksToNotExtractFile << BB->getParent()->getName() << " "
                           << BB->getName() << "\n";
  }
  BlocksToNotExtractFile.close();

  const char *uniqueFN = uniqueFilename.c_str();
  ExtraArg = (char*)malloc(23 + strlen(uniqueFN));
  strcat(strcpy(ExtraArg, "--extract-blocks-file="), uniqueFN);

  std::vector<const PassInfo*> PI;
  std::vector<BasicBlock *> EmptyBBs; // This parameter is ignored.
  PI.push_back(getPI(createBlockExtractorPass(EmptyBBs)));
  Module *Ret = runPassesOn(M, PI, false, 1, &ExtraArg);

  if (uniqueFilename.exists())
    uniqueFilename.eraseFromDisk(); // Free disk space
  free(ExtraArg);

  if (Ret == 0) {
    std::cout << "*** Basic Block extraction failed, please report a bug!\n";
    M = swapProgramIn(M);
    EmitProgressBitcode("basicblockextractfail", true);
    swapProgramIn(M);
  }
  return Ret;
}
