//===- Miscompilation.cpp - Debug program miscompilations -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements optimizer and code generation miscompilation debugging
// support.
//
//===----------------------------------------------------------------------===//

#include "BugDriver.h"
#include "ListReducer.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Config/config.h"   // for HAVE_LINK_R
using namespace llvm;

namespace llvm {
  extern cl::list<std::string> InputArgv;
}

namespace {
  static llvm::cl::opt<bool> 
    DisableLoopExtraction("disable-loop-extraction", 
        cl::desc("Don't extract loops when searching for miscompilations"),
        cl::init(false));

  class ReduceMiscompilingPasses : public ListReducer<const PassInfo*> {
    BugDriver &BD;
  public:
    ReduceMiscompilingPasses(BugDriver &bd) : BD(bd) {}

    virtual TestResult doTest(std::vector<const PassInfo*> &Prefix,
                              std::vector<const PassInfo*> &Suffix);
  };
}

/// TestResult - After passes have been split into a test group and a control
/// group, see if they still break the program.
///
ReduceMiscompilingPasses::TestResult
ReduceMiscompilingPasses::doTest(std::vector<const PassInfo*> &Prefix,
                                 std::vector<const PassInfo*> &Suffix) {
  // First, run the program with just the Suffix passes.  If it is still broken
  // with JUST the kept passes, discard the prefix passes.
  std::cout << "Checking to see if '" << getPassesString(Suffix)
            << "' compile correctly: ";

  std::string BitcodeResult;
  if (BD.runPasses(Suffix, BitcodeResult, false/*delete*/, true/*quiet*/)) {
    std::cerr << " Error running this sequence of passes"
              << " on the input program!\n";
    BD.setPassesToRun(Suffix);
    BD.EmitProgressBitcode("pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // Check to see if the finished program matches the reference output...
  if (BD.diffProgram(BitcodeResult, "", true /*delete bitcode*/)) {
    std::cout << " nope.\n";
    if (Suffix.empty()) {
      std::cerr << BD.getToolName() << ": I'm confused: the test fails when "
                << "no passes are run, nondeterministic program?\n";
      exit(1);
    }
    return KeepSuffix;         // Miscompilation detected!
  }
  std::cout << " yup.\n";      // No miscompilation!

  if (Prefix.empty()) return NoFailure;

  // Next, see if the program is broken if we run the "prefix" passes first,
  // then separately run the "kept" passes.
  std::cout << "Checking to see if '" << getPassesString(Prefix)
            << "' compile correctly: ";

  // If it is not broken with the kept passes, it's possible that the prefix
  // passes must be run before the kept passes to break it.  If the program
  // WORKS after the prefix passes, but then fails if running the prefix AND
  // kept passes, we can update our bitcode file to include the result of the
  // prefix passes, then discard the prefix passes.
  //
  if (BD.runPasses(Prefix, BitcodeResult, false/*delete*/, true/*quiet*/)) {
    std::cerr << " Error running this sequence of passes"
              << " on the input program!\n";
    BD.setPassesToRun(Prefix);
    BD.EmitProgressBitcode("pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // If the prefix maintains the predicate by itself, only keep the prefix!
  if (BD.diffProgram(BitcodeResult)) {
    std::cout << " nope.\n";
    sys::Path(BitcodeResult).eraseFromDisk();
    return KeepPrefix;
  }
  std::cout << " yup.\n";      // No miscompilation!

  // Ok, so now we know that the prefix passes work, try running the suffix
  // passes on the result of the prefix passes.
  //
  Module *PrefixOutput = ParseInputFile(BitcodeResult);
  if (PrefixOutput == 0) {
    std::cerr << BD.getToolName() << ": Error reading bitcode file '"
              << BitcodeResult << "'!\n";
    exit(1);
  }
  sys::Path(BitcodeResult).eraseFromDisk();  // No longer need the file on disk

  // Don't check if there are no passes in the suffix.
  if (Suffix.empty())
    return NoFailure;

  std::cout << "Checking to see if '" << getPassesString(Suffix)
            << "' passes compile correctly after the '"
            << getPassesString(Prefix) << "' passes: ";

  Module *OriginalInput = BD.swapProgramIn(PrefixOutput);
  if (BD.runPasses(Suffix, BitcodeResult, false/*delete*/, true/*quiet*/)) {
    std::cerr << " Error running this sequence of passes"
              << " on the input program!\n";
    BD.setPassesToRun(Suffix);
    BD.EmitProgressBitcode("pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // Run the result...
  if (BD.diffProgram(BitcodeResult, "", true/*delete bitcode*/)) {
    std::cout << " nope.\n";
    delete OriginalInput;     // We pruned down the original input...
    return KeepSuffix;
  }

  // Otherwise, we must not be running the bad pass anymore.
  std::cout << " yup.\n";      // No miscompilation!
  delete BD.swapProgramIn(OriginalInput); // Restore orig program & free test
  return NoFailure;
}

namespace {
  class ReduceMiscompilingFunctions : public ListReducer<Function*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *, Module *);
  public:
    ReduceMiscompilingFunctions(BugDriver &bd,
                                bool (*F)(BugDriver &, Module *, Module *))
      : BD(bd), TestFn(F) {}

    virtual TestResult doTest(std::vector<Function*> &Prefix,
                              std::vector<Function*> &Suffix) {
      if (!Suffix.empty() && TestFuncs(Suffix))
        return KeepSuffix;
      if (!Prefix.empty() && TestFuncs(Prefix))
        return KeepPrefix;
      return NoFailure;
    }

    bool TestFuncs(const std::vector<Function*> &Prefix);
  };
}

/// TestMergedProgram - Given two modules, link them together and run the
/// program, checking to see if the program matches the diff.  If the diff
/// matches, return false, otherwise return true.  If the DeleteInputs argument
/// is set to true then this function deletes both input modules before it
/// returns.
///
static bool TestMergedProgram(BugDriver &BD, Module *M1, Module *M2,
                              bool DeleteInputs) {
  // Link the two portions of the program back to together.
  std::string ErrorMsg;
  if (!DeleteInputs) {
    M1 = CloneModule(M1);
    M2 = CloneModule(M2);
  }
  if (Linker::LinkModules(M1, M2, &ErrorMsg)) {
    std::cerr << BD.getToolName() << ": Error linking modules together:"
              << ErrorMsg << '\n';
    exit(1);
  }
  delete M2;   // We are done with this module.

  Module *OldProgram = BD.swapProgramIn(M1);

  // Execute the program.  If it does not match the expected output, we must
  // return true.
  bool Broken = BD.diffProgram();

  // Delete the linked module & restore the original
  BD.swapProgramIn(OldProgram);
  delete M1;
  return Broken;
}

/// TestFuncs - split functions in a Module into two groups: those that are
/// under consideration for miscompilation vs. those that are not, and test
/// accordingly. Each group of functions becomes a separate Module.
///
bool ReduceMiscompilingFunctions::TestFuncs(const std::vector<Function*>&Funcs){
  // Test to see if the function is misoptimized if we ONLY run it on the
  // functions listed in Funcs.
  std::cout << "Checking to see if the program is misoptimized when "
            << (Funcs.size()==1 ? "this function is" : "these functions are")
            << " run through the pass"
            << (BD.getPassesToRun().size() == 1 ? "" : "es") << ":";
  PrintFunctionList(Funcs);
  std::cout << '\n';

  // Split the module into the two halves of the program we want.
  Module *ToNotOptimize = CloneModule(BD.getProgram());
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize, Funcs);

  // Run the predicate, note that the predicate will delete both input modules.
  return TestFn(BD, ToOptimize, ToNotOptimize);
}

/// DisambiguateGlobalSymbols - Mangle symbols to guarantee uniqueness by
/// modifying predominantly internal symbols rather than external ones.
///
static void DisambiguateGlobalSymbols(Module *M) {
  // Try not to cause collisions by minimizing chances of renaming an
  // already-external symbol, so take in external globals and functions as-is.
  // The code should work correctly without disambiguation (assuming the same
  // mangler is used by the two code generators), but having symbols with the
  // same name causes warnings to be emitted by the code generator.
  Mangler Mang(*M);
  // Agree with the CBE on symbol naming
  Mang.markCharUnacceptable('.');
  Mang.setPreserveAsmNames(true);
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    I->setName(Mang.getValueName(I));
  for (Module::iterator  I = M->begin(),  E = M->end();  I != E; ++I)
    I->setName(Mang.getValueName(I));
}

/// ExtractLoops - Given a reduced list of functions that still exposed the bug,
/// check to see if we can extract the loops in the region without obscuring the
/// bug.  If so, it reduces the amount of code identified.
///
static bool ExtractLoops(BugDriver &BD,
                         bool (*TestFn)(BugDriver &, Module *, Module *),
                         std::vector<Function*> &MiscompiledFunctions) {
  bool MadeChange = false;
  while (1) {
    if (BugpointIsInterrupted) return MadeChange;
    
    Module *ToNotOptimize = CloneModule(BD.getProgram());
    Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize,
                                                   MiscompiledFunctions);
    Module *ToOptimizeLoopExtracted = BD.ExtractLoop(ToOptimize);
    if (!ToOptimizeLoopExtracted) {
      // If the loop extractor crashed or if there were no extractible loops,
      // then this chapter of our odyssey is over with.
      delete ToNotOptimize;
      delete ToOptimize;
      return MadeChange;
    }

    std::cerr << "Extracted a loop from the breaking portion of the program.\n";

    // Bugpoint is intentionally not very trusting of LLVM transformations.  In
    // particular, we're not going to assume that the loop extractor works, so
    // we're going to test the newly loop extracted program to make sure nothing
    // has broken.  If something broke, then we'll inform the user and stop
    // extraction.
    AbstractInterpreter *AI = BD.switchToSafeInterpreter();
    if (TestMergedProgram(BD, ToOptimizeLoopExtracted, ToNotOptimize, false)) {
      BD.switchToInterpreter(AI);

      // Merged program doesn't work anymore!
      std::cerr << "  *** ERROR: Loop extraction broke the program. :("
                << " Please report a bug!\n";
      std::cerr << "      Continuing on with un-loop-extracted version.\n";

      BD.writeProgramToFile("bugpoint-loop-extract-fail-tno.bc", ToNotOptimize);
      BD.writeProgramToFile("bugpoint-loop-extract-fail-to.bc", ToOptimize);
      BD.writeProgramToFile("bugpoint-loop-extract-fail-to-le.bc",
                            ToOptimizeLoopExtracted);

      std::cerr << "Please submit the bugpoint-loop-extract-fail-*.bc files.\n";
      delete ToOptimize;
      delete ToNotOptimize;
      delete ToOptimizeLoopExtracted;
      return MadeChange;
    }
    delete ToOptimize;
    BD.switchToInterpreter(AI);

    std::cout << "  Testing after loop extraction:\n";
    // Clone modules, the tester function will free them.
    Module *TOLEBackup = CloneModule(ToOptimizeLoopExtracted);
    Module *TNOBackup  = CloneModule(ToNotOptimize);
    if (!TestFn(BD, ToOptimizeLoopExtracted, ToNotOptimize)) {
      std::cout << "*** Loop extraction masked the problem.  Undoing.\n";
      // If the program is not still broken, then loop extraction did something
      // that masked the error.  Stop loop extraction now.
      delete TOLEBackup;
      delete TNOBackup;
      return MadeChange;
    }
    ToOptimizeLoopExtracted = TOLEBackup;
    ToNotOptimize = TNOBackup;

    std::cout << "*** Loop extraction successful!\n";

    std::vector<std::pair<std::string, const FunctionType*> > MisCompFunctions;
    for (Module::iterator I = ToOptimizeLoopExtracted->begin(),
           E = ToOptimizeLoopExtracted->end(); I != E; ++I)
      if (!I->isDeclaration())
        MisCompFunctions.push_back(std::make_pair(I->getName(),
                                                  I->getFunctionType()));

    // Okay, great!  Now we know that we extracted a loop and that loop
    // extraction both didn't break the program, and didn't mask the problem.
    // Replace the current program with the loop extracted version, and try to
    // extract another loop.
    std::string ErrorMsg;
    if (Linker::LinkModules(ToNotOptimize, ToOptimizeLoopExtracted, &ErrorMsg)){
      std::cerr << BD.getToolName() << ": Error linking modules together:"
                << ErrorMsg << '\n';
      exit(1);
    }
    delete ToOptimizeLoopExtracted;

    // All of the Function*'s in the MiscompiledFunctions list are in the old
    // module.  Update this list to include all of the functions in the
    // optimized and loop extracted module.
    MiscompiledFunctions.clear();
    for (unsigned i = 0, e = MisCompFunctions.size(); i != e; ++i) {
      Function *NewF = ToNotOptimize->getFunction(MisCompFunctions[i].first);
                                                  
      assert(NewF && "Function not found??");
      assert(NewF->getFunctionType() == MisCompFunctions[i].second && 
             "found wrong function type?");
      MiscompiledFunctions.push_back(NewF);
    }

    BD.setNewProgram(ToNotOptimize);
    MadeChange = true;
  }
}

namespace {
  class ReduceMiscompiledBlocks : public ListReducer<BasicBlock*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *, Module *);
    std::vector<Function*> FunctionsBeingTested;
  public:
    ReduceMiscompiledBlocks(BugDriver &bd,
                            bool (*F)(BugDriver &, Module *, Module *),
                            const std::vector<Function*> &Fns)
      : BD(bd), TestFn(F), FunctionsBeingTested(Fns) {}

    virtual TestResult doTest(std::vector<BasicBlock*> &Prefix,
                              std::vector<BasicBlock*> &Suffix) {
      if (!Suffix.empty() && TestFuncs(Suffix))
        return KeepSuffix;
      if (TestFuncs(Prefix))
        return KeepPrefix;
      return NoFailure;
    }

    bool TestFuncs(const std::vector<BasicBlock*> &Prefix);
  };
}

/// TestFuncs - Extract all blocks for the miscompiled functions except for the
/// specified blocks.  If the problem still exists, return true.
///
bool ReduceMiscompiledBlocks::TestFuncs(const std::vector<BasicBlock*> &BBs) {
  // Test to see if the function is misoptimized if we ONLY run it on the
  // functions listed in Funcs.
  std::cout << "Checking to see if the program is misoptimized when all ";
  if (!BBs.empty()) {
    std::cout << "but these " << BBs.size() << " blocks are extracted: ";
    for (unsigned i = 0, e = BBs.size() < 10 ? BBs.size() : 10; i != e; ++i)
      std::cout << BBs[i]->getName() << " ";
    if (BBs.size() > 10) std::cout << "...";
  } else {
    std::cout << "blocks are extracted.";
  }
  std::cout << '\n';

  // Split the module into the two halves of the program we want.
  Module *ToNotOptimize = CloneModule(BD.getProgram());
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize,
                                                 FunctionsBeingTested);

  // Try the extraction.  If it doesn't work, then the block extractor crashed
  // or something, in which case bugpoint can't chase down this possibility.
  if (Module *New = BD.ExtractMappedBlocksFromModule(BBs, ToOptimize)) {
    delete ToOptimize;
    // Run the predicate, not that the predicate will delete both input modules.
    return TestFn(BD, New, ToNotOptimize);
  }
  delete ToOptimize;
  delete ToNotOptimize;
  return false;
}


/// ExtractBlocks - Given a reduced list of functions that still expose the bug,
/// extract as many basic blocks from the region as possible without obscuring
/// the bug.
///
static bool ExtractBlocks(BugDriver &BD,
                          bool (*TestFn)(BugDriver &, Module *, Module *),
                          std::vector<Function*> &MiscompiledFunctions) {
  if (BugpointIsInterrupted) return false;
  
  std::vector<BasicBlock*> Blocks;
  for (unsigned i = 0, e = MiscompiledFunctions.size(); i != e; ++i)
    for (Function::iterator I = MiscompiledFunctions[i]->begin(),
           E = MiscompiledFunctions[i]->end(); I != E; ++I)
      Blocks.push_back(I);

  // Use the list reducer to identify blocks that can be extracted without
  // obscuring the bug.  The Blocks list will end up containing blocks that must
  // be retained from the original program.
  unsigned OldSize = Blocks.size();

  // Check to see if all blocks are extractible first.
  if (ReduceMiscompiledBlocks(BD, TestFn,
                  MiscompiledFunctions).TestFuncs(std::vector<BasicBlock*>())) {
    Blocks.clear();
  } else {
    ReduceMiscompiledBlocks(BD, TestFn,MiscompiledFunctions).reduceList(Blocks);
    if (Blocks.size() == OldSize)
      return false;
  }

  Module *ProgClone = CloneModule(BD.getProgram());
  Module *ToExtract = SplitFunctionsOutOfModule(ProgClone,
                                                MiscompiledFunctions);
  Module *Extracted = BD.ExtractMappedBlocksFromModule(Blocks, ToExtract);
  if (Extracted == 0) {
    // Weird, extraction should have worked.
    std::cerr << "Nondeterministic problem extracting blocks??\n";
    delete ProgClone;
    delete ToExtract;
    return false;
  }

  // Otherwise, block extraction succeeded.  Link the two program fragments back
  // together.
  delete ToExtract;

  std::vector<std::pair<std::string, const FunctionType*> > MisCompFunctions;
  for (Module::iterator I = Extracted->begin(), E = Extracted->end();
       I != E; ++I)
    if (!I->isDeclaration())
      MisCompFunctions.push_back(std::make_pair(I->getName(),
                                                I->getFunctionType()));

  std::string ErrorMsg;
  if (Linker::LinkModules(ProgClone, Extracted, &ErrorMsg)) {
    std::cerr << BD.getToolName() << ": Error linking modules together:"
              << ErrorMsg << '\n';
    exit(1);
  }
  delete Extracted;

  // Set the new program and delete the old one.
  BD.setNewProgram(ProgClone);

  // Update the list of miscompiled functions.
  MiscompiledFunctions.clear();

  for (unsigned i = 0, e = MisCompFunctions.size(); i != e; ++i) {
    Function *NewF = ProgClone->getFunction(MisCompFunctions[i].first);
    assert(NewF && "Function not found??");
    assert(NewF->getFunctionType() == MisCompFunctions[i].second && 
           "Function has wrong type??");
    MiscompiledFunctions.push_back(NewF);
  }

  return true;
}


/// DebugAMiscompilation - This is a generic driver to narrow down
/// miscompilations, either in an optimization or a code generator.
///
static std::vector<Function*>
DebugAMiscompilation(BugDriver &BD,
                     bool (*TestFn)(BugDriver &, Module *, Module *)) {
  // Okay, now that we have reduced the list of passes which are causing the
  // failure, see if we can pin down which functions are being
  // miscompiled... first build a list of all of the non-external functions in
  // the program.
  std::vector<Function*> MiscompiledFunctions;
  Module *Prog = BD.getProgram();
  for (Module::iterator I = Prog->begin(), E = Prog->end(); I != E; ++I)
    if (!I->isDeclaration())
      MiscompiledFunctions.push_back(I);

  // Do the reduction...
  if (!BugpointIsInterrupted)
    ReduceMiscompilingFunctions(BD, TestFn).reduceList(MiscompiledFunctions);

  std::cout << "\n*** The following function"
            << (MiscompiledFunctions.size() == 1 ? " is" : "s are")
            << " being miscompiled: ";
  PrintFunctionList(MiscompiledFunctions);
  std::cout << '\n';

  // See if we can rip any loops out of the miscompiled functions and still
  // trigger the problem.

  if (!BugpointIsInterrupted && !DisableLoopExtraction &&
      ExtractLoops(BD, TestFn, MiscompiledFunctions)) {
    // Okay, we extracted some loops and the problem still appears.  See if we
    // can eliminate some of the created functions from being candidates.

    // Loop extraction can introduce functions with the same name (foo_code).
    // Make sure to disambiguate the symbols so that when the program is split
    // apart that we can link it back together again.
    DisambiguateGlobalSymbols(BD.getProgram());

    // Do the reduction...
    if (!BugpointIsInterrupted)
      ReduceMiscompilingFunctions(BD, TestFn).reduceList(MiscompiledFunctions);

    std::cout << "\n*** The following function"
              << (MiscompiledFunctions.size() == 1 ? " is" : "s are")
              << " being miscompiled: ";
    PrintFunctionList(MiscompiledFunctions);
    std::cout << '\n';
  }

  if (!BugpointIsInterrupted &&
      ExtractBlocks(BD, TestFn, MiscompiledFunctions)) {
    // Okay, we extracted some blocks and the problem still appears.  See if we
    // can eliminate some of the created functions from being candidates.

    // Block extraction can introduce functions with the same name (foo_code).
    // Make sure to disambiguate the symbols so that when the program is split
    // apart that we can link it back together again.
    DisambiguateGlobalSymbols(BD.getProgram());

    // Do the reduction...
    ReduceMiscompilingFunctions(BD, TestFn).reduceList(MiscompiledFunctions);

    std::cout << "\n*** The following function"
              << (MiscompiledFunctions.size() == 1 ? " is" : "s are")
              << " being miscompiled: ";
    PrintFunctionList(MiscompiledFunctions);
    std::cout << '\n';
  }

  return MiscompiledFunctions;
}

/// TestOptimizer - This is the predicate function used to check to see if the
/// "Test" portion of the program is misoptimized.  If so, return true.  In any
/// case, both module arguments are deleted.
///
static bool TestOptimizer(BugDriver &BD, Module *Test, Module *Safe) {
  // Run the optimization passes on ToOptimize, producing a transformed version
  // of the functions being tested.
  std::cout << "  Optimizing functions being tested: ";
  Module *Optimized = BD.runPassesOn(Test, BD.getPassesToRun(),
                                     /*AutoDebugCrashes*/true);
  std::cout << "done.\n";
  delete Test;

  std::cout << "  Checking to see if the merged program executes correctly: ";
  bool Broken = TestMergedProgram(BD, Optimized, Safe, true);
  std::cout << (Broken ? " nope.\n" : " yup.\n");
  return Broken;
}


/// debugMiscompilation - This method is used when the passes selected are not
/// crashing, but the generated output is semantically different from the
/// input.
///
bool BugDriver::debugMiscompilation() {
  // Make sure something was miscompiled...
  if (!BugpointIsInterrupted)
    if (!ReduceMiscompilingPasses(*this).reduceList(PassesToRun)) {
      std::cerr << "*** Optimized program matches reference output!  No problem"
                << " detected...\nbugpoint can't help you with your problem!\n";
      return false;
    }

  std::cout << "\n*** Found miscompiling pass"
            << (getPassesToRun().size() == 1 ? "" : "es") << ": "
            << getPassesString(getPassesToRun()) << '\n';
  EmitProgressBitcode("passinput");

  std::vector<Function*> MiscompiledFunctions =
    DebugAMiscompilation(*this, TestOptimizer);

  // Output a bunch of bitcode files for the user...
  std::cout << "Outputting reduced bitcode files which expose the problem:\n";
  Module *ToNotOptimize = CloneModule(getProgram());
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize,
                                                 MiscompiledFunctions);

  std::cout << "  Non-optimized portion: ";
  ToNotOptimize = swapProgramIn(ToNotOptimize);
  EmitProgressBitcode("tonotoptimize", true);
  setNewProgram(ToNotOptimize);   // Delete hacked module.

  std::cout << "  Portion that is input to optimizer: ";
  ToOptimize = swapProgramIn(ToOptimize);
  EmitProgressBitcode("tooptimize");
  setNewProgram(ToOptimize);      // Delete hacked module.

  return false;
}

/// CleanupAndPrepareModules - Get the specified modules ready for code
/// generator testing.
///
static void CleanupAndPrepareModules(BugDriver &BD, Module *&Test,
                                     Module *Safe) {
  // Clean up the modules, removing extra cruft that we don't need anymore...
  Test = BD.performFinalCleanups(Test);

  // If we are executing the JIT, we have several nasty issues to take care of.
  if (!BD.isExecutingJIT()) return;

  // First, if the main function is in the Safe module, we must add a stub to
  // the Test module to call into it.  Thus, we create a new function `main'
  // which just calls the old one.
  if (Function *oldMain = Safe->getFunction("main"))
    if (!oldMain->isDeclaration()) {
      // Rename it
      oldMain->setName("llvm_bugpoint_old_main");
      // Create a NEW `main' function with same type in the test module.
      Function *newMain = Function::Create(oldMain->getFunctionType(),
                                           GlobalValue::ExternalLinkage,
                                           "main", Test);
      // Create an `oldmain' prototype in the test module, which will
      // corresponds to the real main function in the same module.
      Function *oldMainProto = Function::Create(oldMain->getFunctionType(),
                                                GlobalValue::ExternalLinkage,
                                                oldMain->getName(), Test);
      // Set up and remember the argument list for the main function.
      std::vector<Value*> args;
      for (Function::arg_iterator
             I = newMain->arg_begin(), E = newMain->arg_end(),
             OI = oldMain->arg_begin(); I != E; ++I, ++OI) {
        I->setName(OI->getName());    // Copy argument names from oldMain
        args.push_back(I);
      }

      // Call the old main function and return its result
      BasicBlock *BB = BasicBlock::Create("entry", newMain);
      CallInst *call = CallInst::Create(oldMainProto, args.begin(), args.end(),
                                        "", BB);

      // If the type of old function wasn't void, return value of call
      ReturnInst::Create(call, BB);
    }

  // The second nasty issue we must deal with in the JIT is that the Safe
  // module cannot directly reference any functions defined in the test
  // module.  Instead, we use a JIT API call to dynamically resolve the
  // symbol.

  // Add the resolver to the Safe module.
  // Prototype: void *getPointerToNamedFunction(const char* Name)
  Constant *resolverFunc =
    Safe->getOrInsertFunction("getPointerToNamedFunction",
                              PointerType::getUnqual(Type::Int8Ty),
                              PointerType::getUnqual(Type::Int8Ty), (Type *)0);

  // Use the function we just added to get addresses of functions we need.
  for (Module::iterator F = Safe->begin(), E = Safe->end(); F != E; ++F) {
    if (F->isDeclaration() && !F->use_empty() && &*F != resolverFunc &&
        !F->isIntrinsic() /* ignore intrinsics */) {
      Function *TestFn = Test->getFunction(F->getName());

      // Don't forward functions which are external in the test module too.
      if (TestFn && !TestFn->isDeclaration()) {
        // 1. Add a string constant with its name to the global file
        Constant *InitArray = ConstantArray::get(F->getName());
        GlobalVariable *funcName =
          new GlobalVariable(InitArray->getType(), true /*isConstant*/,
                             GlobalValue::InternalLinkage, InitArray,
                             F->getName() + "_name", Safe);

        // 2. Use `GetElementPtr *funcName, 0, 0' to convert the string to an
        // sbyte* so it matches the signature of the resolver function.

        // GetElementPtr *funcName, ulong 0, ulong 0
        std::vector<Constant*> GEPargs(2,Constant::getNullValue(Type::Int32Ty));
        Value *GEP = ConstantExpr::getGetElementPtr(funcName, &GEPargs[0], 2);
        std::vector<Value*> ResolverArgs;
        ResolverArgs.push_back(GEP);

        // Rewrite uses of F in global initializers, etc. to uses of a wrapper
        // function that dynamically resolves the calls to F via our JIT API
        if (!F->use_empty()) {
          // Create a new global to hold the cached function pointer.
          Constant *NullPtr = ConstantPointerNull::get(F->getType());
          GlobalVariable *Cache =
            new GlobalVariable(F->getType(), false,GlobalValue::InternalLinkage,
                               NullPtr,F->getName()+".fpcache", F->getParent());

          // Construct a new stub function that will re-route calls to F
          const FunctionType *FuncTy = F->getFunctionType();
          Function *FuncWrapper = Function::Create(FuncTy,
                                                   GlobalValue::InternalLinkage,
                                                   F->getName() + "_wrapper",
                                                   F->getParent());
          BasicBlock *EntryBB  = BasicBlock::Create("entry", FuncWrapper);
          BasicBlock *DoCallBB = BasicBlock::Create("usecache", FuncWrapper);
          BasicBlock *LookupBB = BasicBlock::Create("lookupfp", FuncWrapper);

          // Check to see if we already looked up the value.
          Value *CachedVal = new LoadInst(Cache, "fpcache", EntryBB);
          Value *IsNull = new ICmpInst(ICmpInst::ICMP_EQ, CachedVal,
                                       NullPtr, "isNull", EntryBB);
          BranchInst::Create(LookupBB, DoCallBB, IsNull, EntryBB);

          // Resolve the call to function F via the JIT API:
          //
          // call resolver(GetElementPtr...)
          CallInst *Resolver =
            CallInst::Create(resolverFunc, ResolverArgs.begin(),
                             ResolverArgs.end(), "resolver", LookupBB);

          // Cast the result from the resolver to correctly-typed function.
          CastInst *CastedResolver =
            new BitCastInst(Resolver,
                            PointerType::getUnqual(F->getFunctionType()),
                            "resolverCast", LookupBB);

          // Save the value in our cache.
          new StoreInst(CastedResolver, Cache, LookupBB);
          BranchInst::Create(DoCallBB, LookupBB);

          PHINode *FuncPtr = PHINode::Create(NullPtr->getType(),
                                             "fp", DoCallBB);
          FuncPtr->addIncoming(CastedResolver, LookupBB);
          FuncPtr->addIncoming(CachedVal, EntryBB);

          // Save the argument list.
          std::vector<Value*> Args;
          for (Function::arg_iterator i = FuncWrapper->arg_begin(),
                 e = FuncWrapper->arg_end(); i != e; ++i)
            Args.push_back(i);

          // Pass on the arguments to the real function, return its result
          if (F->getReturnType() == Type::VoidTy) {
            CallInst::Create(FuncPtr, Args.begin(), Args.end(), "", DoCallBB);
            ReturnInst::Create(DoCallBB);
          } else {
            CallInst *Call = CallInst::Create(FuncPtr, Args.begin(), Args.end(),
                                              "retval", DoCallBB);
            ReturnInst::Create(Call, DoCallBB);
          }

          // Use the wrapper function instead of the old function
          F->replaceAllUsesWith(FuncWrapper);
        }
      }
    }
  }

  if (verifyModule(*Test) || verifyModule(*Safe)) {
    std::cerr << "Bugpoint has a bug, which corrupted a module!!\n";
    abort();
  }
}



/// TestCodeGenerator - This is the predicate function used to check to see if
/// the "Test" portion of the program is miscompiled by the code generator under
/// test.  If so, return true.  In any case, both module arguments are deleted.
///
static bool TestCodeGenerator(BugDriver &BD, Module *Test, Module *Safe) {
  CleanupAndPrepareModules(BD, Test, Safe);

  sys::Path TestModuleBC("bugpoint.test.bc");
  std::string ErrMsg;
  if (TestModuleBC.makeUnique(true, &ErrMsg)) {
    std::cerr << BD.getToolName() << "Error making unique filename: "
              << ErrMsg << "\n";
    exit(1);
  }
  if (BD.writeProgramToFile(TestModuleBC.toString(), Test)) {
    std::cerr << "Error writing bitcode to `" << TestModuleBC << "'\nExiting.";
    exit(1);
  }
  delete Test;

  // Make the shared library
  sys::Path SafeModuleBC("bugpoint.safe.bc");
  if (SafeModuleBC.makeUnique(true, &ErrMsg)) {
    std::cerr << BD.getToolName() << "Error making unique filename: "
              << ErrMsg << "\n";
    exit(1);
  }

  if (BD.writeProgramToFile(SafeModuleBC.toString(), Safe)) {
    std::cerr << "Error writing bitcode to `" << SafeModuleBC << "'\nExiting.";
    exit(1);
  }
  std::string SharedObject = BD.compileSharedObject(SafeModuleBC.toString());
  delete Safe;

  // Run the code generator on the `Test' code, loading the shared library.
  // The function returns whether or not the new output differs from reference.
  int Result = BD.diffProgram(TestModuleBC.toString(), SharedObject, false);

  if (Result)
    std::cerr << ": still failing!\n";
  else
    std::cerr << ": didn't fail.\n";
  TestModuleBC.eraseFromDisk();
  SafeModuleBC.eraseFromDisk();
  sys::Path(SharedObject).eraseFromDisk();

  return Result;
}


/// debugCodeGenerator - debug errors in LLC, LLI, or CBE.
///
bool BugDriver::debugCodeGenerator() {
  if ((void*)SafeInterpreter == (void*)Interpreter) {
    std::string Result = executeProgramSafely("bugpoint.safe.out");
    std::cout << "\n*** The \"safe\" i.e. 'known good' backend cannot match "
              << "the reference diff.  This may be due to a\n    front-end "
              << "bug or a bug in the original program, but this can also "
              << "happen if bugpoint isn't running the program with the "
              << "right flags or input.\n    I left the result of executing "
              << "the program with the \"safe\" backend in this file for "
              << "you: '"
              << Result << "'.\n";
    return true;
  }

  DisambiguateGlobalSymbols(Program);

  std::vector<Function*> Funcs = DebugAMiscompilation(*this, TestCodeGenerator);

  // Split the module into the two halves of the program we want.
  Module *ToNotCodeGen = CloneModule(getProgram());
  Module *ToCodeGen = SplitFunctionsOutOfModule(ToNotCodeGen, Funcs);

  // Condition the modules
  CleanupAndPrepareModules(*this, ToCodeGen, ToNotCodeGen);

  sys::Path TestModuleBC("bugpoint.test.bc");
  std::string ErrMsg;
  if (TestModuleBC.makeUnique(true, &ErrMsg)) {
    std::cerr << getToolName() << "Error making unique filename: "
              << ErrMsg << "\n";
    exit(1);
  }

  if (writeProgramToFile(TestModuleBC.toString(), ToCodeGen)) {
    std::cerr << "Error writing bitcode to `" << TestModuleBC << "'\nExiting.";
    exit(1);
  }
  delete ToCodeGen;

  // Make the shared library
  sys::Path SafeModuleBC("bugpoint.safe.bc");
  if (SafeModuleBC.makeUnique(true, &ErrMsg)) {
    std::cerr << getToolName() << "Error making unique filename: "
              << ErrMsg << "\n";
    exit(1);
  }

  if (writeProgramToFile(SafeModuleBC.toString(), ToNotCodeGen)) {
    std::cerr << "Error writing bitcode to `" << SafeModuleBC << "'\nExiting.";
    exit(1);
  }
  std::string SharedObject = compileSharedObject(SafeModuleBC.toString());
  delete ToNotCodeGen;

  std::cout << "You can reproduce the problem with the command line: \n";
  if (isExecutingJIT()) {
    std::cout << "  lli -load " << SharedObject << " " << TestModuleBC;
  } else {
    std::cout << "  llc -f " << TestModuleBC << " -o " << TestModuleBC<< ".s\n";
    std::cout << "  gcc " << SharedObject << " " << TestModuleBC
              << ".s -o " << TestModuleBC << ".exe";
#if defined (HAVE_LINK_R)
    std::cout << " -Wl,-R.";
#endif
    std::cout << "\n";
    std::cout << "  " << TestModuleBC << ".exe";
  }
  for (unsigned i=0, e = InputArgv.size(); i != e; ++i)
    std::cout << " " << InputArgv[i];
  std::cout << '\n';
  std::cout << "The shared object was created with:\n  llc -march=c "
            << SafeModuleBC << " -o temporary.c\n"
            << "  gcc -xc temporary.c -O2 -o " << SharedObject
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
            << " -G"            // Compile a shared library, `-G' for Sparc
#else
            << " -fPIC -shared"       // `-shared' for Linux/X86, maybe others
#endif
            << " -fno-strict-aliasing\n";

  return false;
}
