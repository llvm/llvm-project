//===- CrashDebugger.cpp - Debug compilation crashes ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the bugpoint internals that narrow down compilation crashes
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ToolRunner.h"
#include "ListReducer.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include <fstream>
#include <set>
using namespace llvm;

namespace {
  cl::opt<bool>
  KeepMain("keep-main",
           cl::desc("Force function reduction to keep main"),
           cl::init(false));
}

namespace llvm {
  class ReducePassList : public ListReducer<const PassInfo*> {
    BugDriver &BD;
  public:
    ReducePassList(BugDriver &bd) : BD(bd) {}

    // doTest - Return true iff running the "removed" passes succeeds, and
    // running the "Kept" passes fail when run on the output of the "removed"
    // passes.  If we return true, we update the current module of bugpoint.
    //
    virtual TestResult doTest(std::vector<const PassInfo*> &Removed,
                              std::vector<const PassInfo*> &Kept);
  };
}

ReducePassList::TestResult
ReducePassList::doTest(std::vector<const PassInfo*> &Prefix,
                       std::vector<const PassInfo*> &Suffix) {
  sys::Path PrefixOutput;
  Module *OrigProgram = 0;
  if (!Prefix.empty()) {
    std::cout << "Checking to see if these passes crash: "
              << getPassesString(Prefix) << ": ";
    std::string PfxOutput;
    if (BD.runPasses(Prefix, PfxOutput))
      return KeepPrefix;

    PrefixOutput.set(PfxOutput);
    OrigProgram = BD.Program;

    BD.Program = ParseInputFile(PrefixOutput.toString());
    if (BD.Program == 0) {
      std::cerr << BD.getToolName() << ": Error reading bitcode file '"
                << PrefixOutput << "'!\n";
      exit(1);
    }
    PrefixOutput.eraseFromDisk();
  }

  std::cout << "Checking to see if these passes crash: "
            << getPassesString(Suffix) << ": ";

  if (BD.runPasses(Suffix)) {
    delete OrigProgram;            // The suffix crashes alone...
    return KeepSuffix;
  }

  // Nothing failed, restore state...
  if (OrigProgram) {
    delete BD.Program;
    BD.Program = OrigProgram;
  }
  return NoFailure;
}

namespace {
  /// ReduceCrashingGlobalVariables - This works by removing the global
  /// variable's initializer and seeing if the program still crashes. If it
  /// does, then we keep that program and try again.
  ///
  class ReduceCrashingGlobalVariables : public ListReducer<GlobalVariable*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *);
  public:
    ReduceCrashingGlobalVariables(BugDriver &bd,
                                  bool (*testFn)(BugDriver&, Module*))
      : BD(bd), TestFn(testFn) {}

    virtual TestResult doTest(std::vector<GlobalVariable*>& Prefix,
                              std::vector<GlobalVariable*>& Kept) {
      if (!Kept.empty() && TestGlobalVariables(Kept))
        return KeepSuffix;

      if (!Prefix.empty() && TestGlobalVariables(Prefix))
        return KeepPrefix;

      return NoFailure;
    }

    bool TestGlobalVariables(std::vector<GlobalVariable*>& GVs);
  };
}

bool
ReduceCrashingGlobalVariables::TestGlobalVariables(
                              std::vector<GlobalVariable*>& GVs) {
  // Clone the program to try hacking it apart...
  DenseMap<const Value*, Value*> ValueMap;
  Module *M = CloneModule(BD.getProgram(), ValueMap);

  // Convert list to set for fast lookup...
  std::set<GlobalVariable*> GVSet;

  for (unsigned i = 0, e = GVs.size(); i != e; ++i) {
    GlobalVariable* CMGV = cast<GlobalVariable>(ValueMap[GVs[i]]);
    assert(CMGV && "Global Variable not in module?!");
    GVSet.insert(CMGV);
  }

  std::cout << "Checking for crash with only these global variables: ";
  PrintGlobalVariableList(GVs);
  std::cout << ": ";

  // Loop over and delete any global variables which we aren't supposed to be
  // playing with...
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    if (I->hasInitializer()) {
      I->setInitializer(0);
      I->setLinkage(GlobalValue::ExternalLinkage);
    }

  // Try running the hacked up program...
  if (TestFn(BD, M)) {
    BD.setNewProgram(M);        // It crashed, keep the trimmed version...

    // Make sure to use global variable pointers that point into the now-current
    // module.
    GVs.assign(GVSet.begin(), GVSet.end());
    return true;
  }

  delete M;
  return false;
}

namespace llvm {
  /// ReduceCrashingFunctions reducer - This works by removing functions and
  /// seeing if the program still crashes. If it does, then keep the newer,
  /// smaller program.
  ///
  class ReduceCrashingFunctions : public ListReducer<Function*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *);
  public:
    ReduceCrashingFunctions(BugDriver &bd,
                            bool (*testFn)(BugDriver &, Module *))
      : BD(bd), TestFn(testFn) {}

    virtual TestResult doTest(std::vector<Function*> &Prefix,
                              std::vector<Function*> &Kept) {
      if (!Kept.empty() && TestFuncs(Kept))
        return KeepSuffix;
      if (!Prefix.empty() && TestFuncs(Prefix))
        return KeepPrefix;
      return NoFailure;
    }

    bool TestFuncs(std::vector<Function*> &Prefix);
  };
}

bool ReduceCrashingFunctions::TestFuncs(std::vector<Function*> &Funcs) {

  //if main isn't present, claim there is no problem
  if (KeepMain && find(Funcs.begin(), Funcs.end(),
                       BD.getProgram()->getFunction("main")) == Funcs.end())
    return false;

  // Clone the program to try hacking it apart...
  Module *M = CloneModule(BD.getProgram());

  // Convert list to set for fast lookup...
  std::set<Function*> Functions;
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i) {
    // FIXME: bugpoint should add names to all stripped symbols.
    assert(!Funcs[i]->getName().empty() &&
           "Bugpoint doesn't work on stripped modules yet PR718!");
    Function *CMF = M->getFunction(Funcs[i]->getName());
    assert(CMF && "Function not in module?!");
    assert(CMF->getFunctionType() == Funcs[i]->getFunctionType() && "wrong ty");
    Functions.insert(CMF);
  }

  std::cout << "Checking for crash with only these functions: ";
  PrintFunctionList(Funcs);
  std::cout << ": ";

  // Loop over and delete any functions which we aren't supposed to be playing
  // with...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isDeclaration() && !Functions.count(I))
      DeleteFunctionBody(I);

  // Try running the hacked up program...
  if (TestFn(BD, M)) {
    BD.setNewProgram(M);        // It crashed, keep the trimmed version...

    // Make sure to use function pointers that point into the now-current
    // module.
    Funcs.assign(Functions.begin(), Functions.end());
    return true;
  }
  delete M;
  return false;
}


namespace {
  /// ReduceCrashingBlocks reducer - This works by setting the terminators of
  /// all terminators except the specified basic blocks to a 'ret' instruction,
  /// then running the simplify-cfg pass.  This has the effect of chopping up
  /// the CFG really fast which can reduce large functions quickly.
  ///
  class ReduceCrashingBlocks : public ListReducer<const BasicBlock*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *);
  public:
    ReduceCrashingBlocks(BugDriver &bd, bool (*testFn)(BugDriver &, Module *))
      : BD(bd), TestFn(testFn) {}

    virtual TestResult doTest(std::vector<const BasicBlock*> &Prefix,
                              std::vector<const BasicBlock*> &Kept) {
      if (!Kept.empty() && TestBlocks(Kept))
        return KeepSuffix;
      if (!Prefix.empty() && TestBlocks(Prefix))
        return KeepPrefix;
      return NoFailure;
    }

    bool TestBlocks(std::vector<const BasicBlock*> &Prefix);
  };
}

bool ReduceCrashingBlocks::TestBlocks(std::vector<const BasicBlock*> &BBs) {
  // Clone the program to try hacking it apart...
  Module *M = CloneModule(BD.getProgram());

  // Convert list to set for fast lookup...
  std::set<BasicBlock*> Blocks;
  for (unsigned i = 0, e = BBs.size(); i != e; ++i) {
    // Convert the basic block from the original module to the new module...
    const Function *F = BBs[i]->getParent();
    Function *CMF = M->getFunction(F->getName());
    assert(CMF && "Function not in module?!");
    assert(CMF->getFunctionType() == F->getFunctionType() && "wrong type?");

    // Get the mapped basic block...
    Function::iterator CBI = CMF->begin();
    std::advance(CBI, std::distance(F->begin(),
                                    Function::const_iterator(BBs[i])));
    Blocks.insert(CBI);
  }

  std::cout << "Checking for crash with only these blocks:";
  unsigned NumPrint = Blocks.size();
  if (NumPrint > 10) NumPrint = 10;
  for (unsigned i = 0, e = NumPrint; i != e; ++i)
    std::cout << " " << BBs[i]->getName();
  if (NumPrint < Blocks.size())
    std::cout << "... <" << Blocks.size() << " total>";
  std::cout << ": ";

  // Loop over and delete any hack up any blocks that are not listed...
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    for (Function::iterator BB = I->begin(), E = I->end(); BB != E; ++BB)
      if (!Blocks.count(BB) && BB->getTerminator()->getNumSuccessors()) {
        // Loop over all of the successors of this block, deleting any PHI nodes
        // that might include it.
        for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
          (*SI)->removePredecessor(BB);

        TerminatorInst *BBTerm = BB->getTerminator();
        
        if (isa<StructType>(BBTerm->getType()))
           BBTerm->replaceAllUsesWith(UndefValue::get(BBTerm->getType()));
        else if (BB->getTerminator()->getType() != Type::VoidTy)
          BBTerm->replaceAllUsesWith(Constant::getNullValue(BBTerm->getType()));

        // Replace the old terminator instruction.
        BB->getInstList().pop_back();
        new UnreachableInst(BB);
      }

  // The CFG Simplifier pass may delete one of the basic blocks we are
  // interested in.  If it does we need to take the block out of the list.  Make
  // a "persistent mapping" by turning basic blocks into <function, name> pairs.
  // This won't work well if blocks are unnamed, but that is just the risk we
  // have to take.
  std::vector<std::pair<Function*, std::string> > BlockInfo;

  for (std::set<BasicBlock*>::iterator I = Blocks.begin(), E = Blocks.end();
       I != E; ++I)
    BlockInfo.push_back(std::make_pair((*I)->getParent(), (*I)->getName()));

  // Now run the CFG simplify pass on the function...
  PassManager Passes;
  Passes.add(createCFGSimplificationPass());
  Passes.add(createVerifierPass());
  Passes.run(*M);

  // Try running on the hacked up program...
  if (TestFn(BD, M)) {
    BD.setNewProgram(M);      // It crashed, keep the trimmed version...

    // Make sure to use basic block pointers that point into the now-current
    // module, and that they don't include any deleted blocks.
    BBs.clear();
    for (unsigned i = 0, e = BlockInfo.size(); i != e; ++i) {
      ValueSymbolTable &ST = BlockInfo[i].first->getValueSymbolTable();
      Value* V = ST.lookup(BlockInfo[i].second);
      if (V && V->getType() == Type::LabelTy)
        BBs.push_back(cast<BasicBlock>(V));
    }
    return true;
  }
  delete M;  // It didn't crash, try something else.
  return false;
}

/// DebugACrash - Given a predicate that determines whether a component crashes
/// on a program, try to destructively reduce the program while still keeping
/// the predicate true.
static bool DebugACrash(BugDriver &BD,  bool (*TestFn)(BugDriver &, Module *)) {
  // See if we can get away with nuking some of the global variable initializers
  // in the program...
  if (BD.getProgram()->global_begin() != BD.getProgram()->global_end()) {
    // Now try to reduce the number of global variable initializers in the
    // module to something small.
    Module *M = CloneModule(BD.getProgram());
    bool DeletedInit = false;

    for (Module::global_iterator I = M->global_begin(), E = M->global_end();
         I != E; ++I)
      if (I->hasInitializer()) {
        I->setInitializer(0);
        I->setLinkage(GlobalValue::ExternalLinkage);
        DeletedInit = true;
      }

    if (!DeletedInit) {
      delete M;  // No change made...
    } else {
      // See if the program still causes a crash...
      std::cout << "\nChecking to see if we can delete global inits: ";

      if (TestFn(BD, M)) {      // Still crashes?
        BD.setNewProgram(M);
        std::cout << "\n*** Able to remove all global initializers!\n";
      } else {                  // No longer crashes?
        std::cout << "  - Removing all global inits hides problem!\n";
        delete M;

        std::vector<GlobalVariable*> GVs;

        for (Module::global_iterator I = BD.getProgram()->global_begin(),
               E = BD.getProgram()->global_end(); I != E; ++I)
          if (I->hasInitializer())
            GVs.push_back(I);

        if (GVs.size() > 1 && !BugpointIsInterrupted) {
          std::cout << "\n*** Attempting to reduce the number of global "
                    << "variables in the testcase\n";

          unsigned OldSize = GVs.size();
          ReduceCrashingGlobalVariables(BD, TestFn).reduceList(GVs);

          if (GVs.size() < OldSize)
            BD.EmitProgressBitcode("reduced-global-variables");
        }
      }
    }
  }

  // Now try to reduce the number of functions in the module to something small.
  std::vector<Function*> Functions;
  for (Module::iterator I = BD.getProgram()->begin(),
         E = BD.getProgram()->end(); I != E; ++I)
    if (!I->isDeclaration())
      Functions.push_back(I);

  if (Functions.size() > 1 && !BugpointIsInterrupted) {
    std::cout << "\n*** Attempting to reduce the number of functions "
      "in the testcase\n";

    unsigned OldSize = Functions.size();
    ReduceCrashingFunctions(BD, TestFn).reduceList(Functions);

    if (Functions.size() < OldSize)
      BD.EmitProgressBitcode("reduced-function");
  }

  // Attempt to delete entire basic blocks at a time to speed up
  // convergence... this actually works by setting the terminator of the blocks
  // to a return instruction then running simplifycfg, which can potentially
  // shrinks the code dramatically quickly
  //
  if (!DisableSimplifyCFG && !BugpointIsInterrupted) {
    std::vector<const BasicBlock*> Blocks;
    for (Module::const_iterator I = BD.getProgram()->begin(),
           E = BD.getProgram()->end(); I != E; ++I)
      for (Function::const_iterator FI = I->begin(), E = I->end(); FI !=E; ++FI)
        Blocks.push_back(FI);
    ReduceCrashingBlocks(BD, TestFn).reduceList(Blocks);
  }

  // FIXME: This should use the list reducer to converge faster by deleting
  // larger chunks of instructions at a time!
  unsigned Simplification = 2;
  do {
    if (BugpointIsInterrupted) break;
    --Simplification;
    std::cout << "\n*** Attempting to reduce testcase by deleting instruc"
              << "tions: Simplification Level #" << Simplification << '\n';

    // Now that we have deleted the functions that are unnecessary for the
    // program, try to remove instructions that are not necessary to cause the
    // crash.  To do this, we loop through all of the instructions in the
    // remaining functions, deleting them (replacing any values produced with
    // nulls), and then running ADCE and SimplifyCFG.  If the transformed input
    // still triggers failure, keep deleting until we cannot trigger failure
    // anymore.
    //
    unsigned InstructionsToSkipBeforeDeleting = 0;
  TryAgain:

    // Loop over all of the (non-terminator) instructions remaining in the
    // function, attempting to delete them.
    unsigned CurInstructionNum = 0;
    for (Module::const_iterator FI = BD.getProgram()->begin(),
           E = BD.getProgram()->end(); FI != E; ++FI)
      if (!FI->isDeclaration())
        for (Function::const_iterator BI = FI->begin(), E = FI->end(); BI != E;
             ++BI)
          for (BasicBlock::const_iterator I = BI->begin(), E = --BI->end();
               I != E; ++I, ++CurInstructionNum)
            if (InstructionsToSkipBeforeDeleting) {
              --InstructionsToSkipBeforeDeleting;
            } else {
              if (BugpointIsInterrupted) goto ExitLoops;

              std::cout << "Checking instruction: " << *I;
              Module *M = BD.deleteInstructionFromProgram(I, Simplification);

              // Find out if the pass still crashes on this pass...
              if (TestFn(BD, M)) {
                // Yup, it does, we delete the old module, and continue trying
                // to reduce the testcase...
                BD.setNewProgram(M);
                InstructionsToSkipBeforeDeleting = CurInstructionNum;
                goto TryAgain;  // I wish I had a multi-level break here!
              }

              // This pass didn't crash without this instruction, try the next
              // one.
              delete M;
            }

    if (InstructionsToSkipBeforeDeleting) {
      InstructionsToSkipBeforeDeleting = 0;
      goto TryAgain;
    }

  } while (Simplification);
ExitLoops:

  // Try to clean up the testcase by running funcresolve and globaldce...
  if (!BugpointIsInterrupted) {
    std::cout << "\n*** Attempting to perform final cleanups: ";
    Module *M = CloneModule(BD.getProgram());
    M = BD.performFinalCleanups(M, true);

    // Find out if the pass still crashes on the cleaned up program...
    if (TestFn(BD, M)) {
      BD.setNewProgram(M);     // Yup, it does, keep the reduced version...
    } else {
      delete M;
    }
  }

  BD.EmitProgressBitcode("reduced-simplified");

  return false;
}

static bool TestForOptimizerCrash(BugDriver &BD, Module *M) {
  return BD.runPasses(M);
}

/// debugOptimizerCrash - This method is called when some pass crashes on input.
/// It attempts to prune down the testcase to something reasonable, and figure
/// out exactly which pass is crashing.
///
bool BugDriver::debugOptimizerCrash(const std::string &ID) {
  std::cout << "\n*** Debugging optimizer crash!\n";

  // Reduce the list of passes which causes the optimizer to crash...
  if (!BugpointIsInterrupted)
    ReducePassList(*this).reduceList(PassesToRun);

  std::cout << "\n*** Found crashing pass"
            << (PassesToRun.size() == 1 ? ": " : "es: ")
            << getPassesString(PassesToRun) << '\n';

  EmitProgressBitcode(ID);

  return DebugACrash(*this, TestForOptimizerCrash);
}

static bool TestForCodeGenCrash(BugDriver &BD, Module *M) {
  try {
    BD.compileProgram(M);
    std::cerr << '\n';
    return false;
  } catch (ToolExecutionError &) {
    std::cerr << "<crash>\n";
    return true;  // Tool is still crashing.
  }
}

/// debugCodeGeneratorCrash - This method is called when the code generator
/// crashes on an input.  It attempts to reduce the input as much as possible
/// while still causing the code generator to crash.
bool BugDriver::debugCodeGeneratorCrash() {
  std::cerr << "*** Debugging code generator crash!\n";

  return DebugACrash(*this, TestForCodeGenCrash);
}
