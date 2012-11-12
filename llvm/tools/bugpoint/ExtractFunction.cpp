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
#include "llvm/DataLayout.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include <set>
using namespace llvm;

namespace llvm {
  bool DisableSimplifyCFG = false;
  extern cl::opt<std::string> OutputPrefix;
} // End llvm namespace

namespace {
  cl::opt<bool>
  NoDCE ("disable-dce",
         cl::desc("Do not use the -dce pass to reduce testcases"));
  cl::opt<bool, true>
  NoSCFG("disable-simplifycfg", cl::location(DisableSimplifyCFG),
         cl::desc("Do not use the -simplifycfg pass to reduce testcases"));

  Function* globalInitUsesExternalBA(GlobalVariable* GV) {
    if (!GV->hasInitializer())
      return 0;

    Constant *I = GV->getInitializer();

    // walk the values used by the initializer
    // (and recurse into things like ConstantExpr)
    std::vector<Constant*> Todo;
    std::set<Constant*> Done;
    Todo.push_back(I);

    while (!Todo.empty()) {
      Constant* V = Todo.back();
      Todo.pop_back();
      Done.insert(V);

      if (BlockAddress *BA = dyn_cast<BlockAddress>(V)) {
        Function *F = BA->getFunction();
        if (F->isDeclaration())
          return F;
      }

      for (User::op_iterator i = V->op_begin(), e = V->op_end(); i != e; ++i) {
        Constant *C = dyn_cast<Constant>(*i);
        if (C && !isa<GlobalValue>(C) && !Done.count(C))
          Todo.push_back(C);
      }
    }
    return 0;
  }
}  // end anonymous namespace

/// deleteInstructionFromProgram - This method clones the current Program and
/// deletes the specified instruction from the cloned module.  It then runs a
/// series of cleanup passes (ADCE and SimplifyCFG) to eliminate any code which
/// depends on the value.  The modified module is then returned.
///
Module *BugDriver::deleteInstructionFromProgram(const Instruction *I,
                                                unsigned Simplification) {
  // FIXME, use vmap?
  Module *Clone = CloneModule(Program);

  const BasicBlock *PBB = I->getParent();
  const Function *PF = PBB->getParent();

  Module::iterator RFI = Clone->begin(); // Get iterator to corresponding fn
  std::advance(RFI, std::distance(PF->getParent()->begin(),
                                  Module::const_iterator(PF)));

  Function::iterator RBI = RFI->begin();  // Get iterator to corresponding BB
  std::advance(RBI, std::distance(PF->begin(), Function::const_iterator(PBB)));

  BasicBlock::iterator RI = RBI->begin(); // Get iterator to corresponding inst
  std::advance(RI, std::distance(PBB->begin(), BasicBlock::const_iterator(I)));
  Instruction *TheInst = RI;              // Got the corresponding instruction!

  // If this instruction produces a value, replace any users with null values
  if (!TheInst->getType()->isVoidTy())
    TheInst->replaceAllUsesWith(Constant::getNullValue(TheInst->getType()));

  // Remove the instruction from the program.
  TheInst->getParent()->getInstList().erase(TheInst);

  // Spiff up the output a little bit.
  std::vector<std::string> Passes;

  /// Can we get rid of the -disable-* options?
  if (Simplification > 1 && !NoDCE)
    Passes.push_back("dce");
  if (Simplification && !DisableSimplifyCFG)
    Passes.push_back("simplifycfg");      // Delete dead control flow

  Passes.push_back("verify");
  Module *New = runPassesOn(Clone, Passes);
  delete Clone;
  if (!New) {
    errs() << "Instruction removal failed.  Sorry. :(  Please report a bug!\n";
    exit(1);
  }
  return New;
}

/// performFinalCleanups - This method clones the current Program and performs
/// a series of cleanups intended to get rid of extra cruft on the module
/// before handing it to the user.
///
Module *BugDriver::performFinalCleanups(Module *M, bool MayModifySemantics) {
  // Make all functions external, so GlobalDCE doesn't delete them...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);

  std::vector<std::string> CleanupPasses;
  CleanupPasses.push_back("globaldce");

  if (MayModifySemantics)
    CleanupPasses.push_back("deadarghaX0r");
  else
    CleanupPasses.push_back("deadargelim");

  Module *New = runPassesOn(M, CleanupPasses);
  if (New == 0) {
    errs() << "Final cleanups failed.  Sorry. :(  Please report a bug!\n";
    return M;
  }
  delete M;
  return New;
}


/// ExtractLoop - Given a module, extract up to one loop from it into a new
/// function.  This returns null if there are no extractable loops in the
/// program or if the loop extractor crashes.
Module *BugDriver::ExtractLoop(Module *M) {
  std::vector<std::string> LoopExtractPasses;
  LoopExtractPasses.push_back("loop-extract-single");

  Module *NewM = runPassesOn(M, LoopExtractPasses);
  if (NewM == 0) {
    outs() << "*** Loop extraction failed: ";
    EmitProgressBitcode(M, "loopextraction", true);
    outs() << "*** Sorry. :(  Please report a bug!\n";
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
  Type *Int32Ty = Type::getInt32Ty(TorList[0].first->getContext());
  
  StructType *STy =
    StructType::get(Int32Ty, TorList[0].first->getType(), NULL);
  for (unsigned i = 0, e = TorList.size(); i != e; ++i) {
    Constant *Elts[] = {
      ConstantInt::get(Int32Ty, TorList[i].second),
      TorList[i].first
    };
    ArrayElts.push_back(ConstantStruct::get(STy, Elts));
  }
  return ConstantArray::get(ArrayType::get(ArrayElts[0]->getType(), 
                                           ArrayElts.size()),
                            ArrayElts);
}

/// SplitStaticCtorDtor - A module was recently split into two parts, M1/M2, and
/// M1 has all of the global variables.  If M2 contains any functions that are
/// static ctors/dtors, we need to add an llvm.global_[cd]tors global to M2, and
/// prune appropriate entries out of M1s list.
static void SplitStaticCtorDtor(const char *GlobalName, Module *M1, Module *M2,
                                ValueToValueMapTy &VMap) {
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
          F = cast<Function>(VMap[F]);
          M2Tors.push_back(std::make_pair(F, Priority));
        }
      }
    }
  }
  
  GV->eraseFromParent();
  if (!M1Tors.empty()) {
    Constant *M1Init = GetTorInit(M1Tors);
    new GlobalVariable(*M1, M1Init->getType(), false,
                       GlobalValue::AppendingLinkage,
                       M1Init, GlobalName);
  }

  GV = M2->getNamedGlobal(GlobalName);
  assert(GV && "Not a clone of M1?");
  assert(GV->use_empty() && "llvm.ctors shouldn't have uses!");

  GV->eraseFromParent();
  if (!M2Tors.empty()) {
    Constant *M2Init = GetTorInit(M2Tors);
    new GlobalVariable(*M2, M2Init->getType(), false,
                       GlobalValue::AppendingLinkage,
                       M2Init, GlobalName);
  }
}


/// SplitFunctionsOutOfModule - Given a module and a list of functions in the
/// module, split the functions OUT of the specified module, and place them in
/// the new module.
Module *
llvm::SplitFunctionsOutOfModule(Module *M,
                                const std::vector<Function*> &F,
                                ValueToValueMapTy &VMap) {
  // Make sure functions & globals are all external so that linkage
  // between the two modules will work.
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    I->setLinkage(GlobalValue::ExternalLinkage);
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    if (I->hasName() && I->getName()[0] == '\01')
      I->setName(I->getName().substr(1));
    I->setLinkage(GlobalValue::ExternalLinkage);
  }

  ValueToValueMapTy NewVMap;
  Module *New = CloneModule(M, NewVMap);

  // Remove the Test functions from the Safe module
  std::set<Function *> TestFunctions;
  for (unsigned i = 0, e = F.size(); i != e; ++i) {
    Function *TNOF = cast<Function>(VMap[F[i]]);
    DEBUG(errs() << "Removing function ");
    DEBUG(WriteAsOperand(errs(), TNOF, false));
    DEBUG(errs() << "\n");
    TestFunctions.insert(cast<Function>(NewVMap[TNOF]));
    DeleteFunctionBody(TNOF);       // Function is now external in this module!
  }

  
  // Remove the Safe functions from the Test module
  for (Module::iterator I = New->begin(), E = New->end(); I != E; ++I)
    if (!TestFunctions.count(I))
      DeleteFunctionBody(I);
  

  // Try to split the global initializers evenly
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = cast<GlobalVariable>(NewVMap[I]);
    if (Function *TestFn = globalInitUsesExternalBA(I)) {
      if (Function *SafeFn = globalInitUsesExternalBA(GV)) {
        errs() << "*** Error: when reducing functions, encountered "
                  "the global '";
        WriteAsOperand(errs(), GV, false);
        errs() << "' with an initializer that references blockaddresses "
                  "from safe function '" << SafeFn->getName()
               << "' and from test function '" << TestFn->getName() << "'.\n";
        exit(1);
      }
      I->setInitializer(0);  // Delete the initializer to make it external
    } else {
      // If we keep it in the safe module, then delete it in the test module
      GV->setInitializer(0);
    }
  }

  // Make sure that there is a global ctor/dtor array in both halves of the
  // module if they both have static ctor/dtor functions.
  SplitStaticCtorDtor("llvm.global_ctors", M, New, NewVMap);
  SplitStaticCtorDtor("llvm.global_dtors", M, New, NewVMap);
  
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
  sys::Path uniqueFilename(OutputPrefix + "-extractblocks");
  std::string ErrMsg;
  if (uniqueFilename.createTemporaryFileOnDisk(true, &ErrMsg)) {
    outs() << "*** Basic Block extraction failed!\n";
    errs() << "Error creating temporary file: " << ErrMsg << "\n";
    EmitProgressBitcode(M, "basicblockextractfail", true);
    return 0;
  }
  sys::RemoveFileOnSignal(uniqueFilename);

  std::string ErrorInfo;
  tool_output_file BlocksToNotExtractFile(uniqueFilename.c_str(), ErrorInfo);
  if (!ErrorInfo.empty()) {
    outs() << "*** Basic Block extraction failed!\n";
    errs() << "Error writing list of blocks to not extract: " << ErrorInfo
           << "\n";
    EmitProgressBitcode(M, "basicblockextractfail", true);
    return 0;
  }
  for (std::vector<BasicBlock*>::const_iterator I = BBs.begin(), E = BBs.end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    // If the BB doesn't have a name, give it one so we have something to key
    // off of.
    if (!BB->hasName()) BB->setName("tmpbb");
    BlocksToNotExtractFile.os() << BB->getParent()->getName() << " "
                                << BB->getName() << "\n";
  }
  BlocksToNotExtractFile.os().close();
  if (BlocksToNotExtractFile.os().has_error()) {
    errs() << "Error writing list of blocks to not extract: " << ErrorInfo
           << "\n";
    EmitProgressBitcode(M, "basicblockextractfail", true);
    BlocksToNotExtractFile.os().clear_error();
    return 0;
  }
  BlocksToNotExtractFile.keep();

  std::string uniqueFN = "--extract-blocks-file=" + uniqueFilename.str();
  const char *ExtraArg = uniqueFN.c_str();

  std::vector<std::string> PI;
  PI.push_back("extract-blocks");
  Module *Ret = runPassesOn(M, PI, false, 1, &ExtraArg);

  uniqueFilename.eraseFromDisk(); // Free disk space

  if (Ret == 0) {
    outs() << "*** Basic Block extraction failed, please report a bug!\n";
    EmitProgressBitcode(M, "basicblockextractfail", true);
  }
  return Ret;
}
