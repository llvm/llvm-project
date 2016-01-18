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
#include "ToolRunner.h"
#include "llvm/Config/config.h"   // for HAVE_LINK_R
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Transforms/Utils/Cloning.h"
using namespace llvm;

namespace llvm {
  extern cl::opt<std::string> OutputPrefix;
  extern cl::list<std::string> InputArgv;
}

namespace {
  static llvm::cl::opt<bool>
    DisableLoopExtraction("disable-loop-extraction",
        cl::desc("Don't extract loops when searching for miscompilations"),
        cl::init(false));
  static llvm::cl::opt<bool>
    DisableBlockExtraction("disable-block-extraction",
        cl::desc("Don't extract blocks when searching for miscompilations"),
        cl::init(false));

  class ReduceMiscompilingPasses : public ListReducer<std::string> {
    BugDriver &BD;
  public:
    ReduceMiscompilingPasses(BugDriver &bd) : BD(bd) {}

    TestResult doTest(std::vector<std::string> &Prefix,
                      std::vector<std::string> &Suffix,
                      std::string &Error) override;
  };
}

/// TestResult - After passes have been split into a test group and a control
/// group, see if they still break the program.
///
ReduceMiscompilingPasses::TestResult
ReduceMiscompilingPasses::doTest(std::vector<std::string> &Prefix,
                                 std::vector<std::string> &Suffix,
                                 std::string &Error) {
  // First, run the program with just the Suffix passes.  If it is still broken
  // with JUST the kept passes, discard the prefix passes.
  outs() << "Checking to see if '" << getPassesString(Suffix)
         << "' compiles correctly: ";

  std::string BitcodeResult;
  if (BD.runPasses(BD.getProgram(), Suffix, BitcodeResult, false/*delete*/,
                   true/*quiet*/)) {
    errs() << " Error running this sequence of passes"
           << " on the input program!\n";
    BD.setPassesToRun(Suffix);
    BD.EmitProgressBitcode(BD.getProgram(), "pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // Check to see if the finished program matches the reference output...
  bool Diff = BD.diffProgram(BD.getProgram(), BitcodeResult, "",
                             true /*delete bitcode*/, &Error);
  if (!Error.empty())
    return InternalError;
  if (Diff) {
    outs() << " nope.\n";
    if (Suffix.empty()) {
      errs() << BD.getToolName() << ": I'm confused: the test fails when "
             << "no passes are run, nondeterministic program?\n";
      exit(1);
    }
    return KeepSuffix;         // Miscompilation detected!
  }
  outs() << " yup.\n";      // No miscompilation!

  if (Prefix.empty()) return NoFailure;

  // Next, see if the program is broken if we run the "prefix" passes first,
  // then separately run the "kept" passes.
  outs() << "Checking to see if '" << getPassesString(Prefix)
         << "' compiles correctly: ";

  // If it is not broken with the kept passes, it's possible that the prefix
  // passes must be run before the kept passes to break it.  If the program
  // WORKS after the prefix passes, but then fails if running the prefix AND
  // kept passes, we can update our bitcode file to include the result of the
  // prefix passes, then discard the prefix passes.
  //
  if (BD.runPasses(BD.getProgram(), Prefix, BitcodeResult, false/*delete*/,
                   true/*quiet*/)) {
    errs() << " Error running this sequence of passes"
           << " on the input program!\n";
    BD.setPassesToRun(Prefix);
    BD.EmitProgressBitcode(BD.getProgram(), "pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // If the prefix maintains the predicate by itself, only keep the prefix!
  Diff = BD.diffProgram(BD.getProgram(), BitcodeResult, "", false, &Error);
  if (!Error.empty())
    return InternalError;
  if (Diff) {
    outs() << " nope.\n";
    sys::fs::remove(BitcodeResult);
    return KeepPrefix;
  }
  outs() << " yup.\n";      // No miscompilation!

  // Ok, so now we know that the prefix passes work, try running the suffix
  // passes on the result of the prefix passes.
  //
  std::unique_ptr<Module> PrefixOutput =
      parseInputFile(BitcodeResult, BD.getContext());
  if (!PrefixOutput) {
    errs() << BD.getToolName() << ": Error reading bitcode file '"
           << BitcodeResult << "'!\n";
    exit(1);
  }
  sys::fs::remove(BitcodeResult);

  // Don't check if there are no passes in the suffix.
  if (Suffix.empty())
    return NoFailure;

  outs() << "Checking to see if '" << getPassesString(Suffix)
            << "' passes compile correctly after the '"
            << getPassesString(Prefix) << "' passes: ";

  std::unique_ptr<Module> OriginalInput(
      BD.swapProgramIn(PrefixOutput.release()));
  if (BD.runPasses(BD.getProgram(), Suffix, BitcodeResult, false/*delete*/,
                   true/*quiet*/)) {
    errs() << " Error running this sequence of passes"
           << " on the input program!\n";
    BD.setPassesToRun(Suffix);
    BD.EmitProgressBitcode(BD.getProgram(), "pass-error",  false);
    exit(BD.debugOptimizerCrash());
  }

  // Run the result...
  Diff = BD.diffProgram(BD.getProgram(), BitcodeResult, "",
                        true /*delete bitcode*/, &Error);
  if (!Error.empty())
    return InternalError;
  if (Diff) {
    outs() << " nope.\n";
    return KeepSuffix;
  }

  // Otherwise, we must not be running the bad pass anymore.
  outs() << " yup.\n";      // No miscompilation!
  // Restore orig program & free test.
  delete BD.swapProgramIn(OriginalInput.release());
  return NoFailure;
}

namespace {
  class ReduceMiscompilingFunctions : public ListReducer<Function*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *, Module *, std::string &);
  public:
    ReduceMiscompilingFunctions(BugDriver &bd,
                                bool (*F)(BugDriver &, Module *, Module *,
                                          std::string &))
      : BD(bd), TestFn(F) {}

    TestResult doTest(std::vector<Function*> &Prefix,
                      std::vector<Function*> &Suffix,
                      std::string &Error) override {
      if (!Suffix.empty()) {
        bool Ret = TestFuncs(Suffix, Error);
        if (!Error.empty())
          return InternalError;
        if (Ret)
          return KeepSuffix;
      }
      if (!Prefix.empty()) {
        bool Ret = TestFuncs(Prefix, Error);
        if (!Error.empty())
          return InternalError;
        if (Ret)
          return KeepPrefix;
      }
      return NoFailure;
    }

    bool TestFuncs(const std::vector<Function*> &Prefix, std::string &Error);
  };
}

/// TestMergedProgram - Given two modules, link them together and run the
/// program, checking to see if the program matches the diff. If there is
/// an error, return NULL. If not, return the merged module. The Broken argument
/// will be set to true if the output is different. If the DeleteInputs
/// argument is set to true then this function deletes both input
/// modules before it returns.
///
static Module *TestMergedProgram(const BugDriver &BD, Module *M1, Module *M2,
                                 bool DeleteInputs, std::string &Error,
                                 bool &Broken) {
  // Link the two portions of the program back to together.
  if (!DeleteInputs) {
    M1 = CloneModule(M1);
    M2 = CloneModule(M2);
  }
  if (Linker::LinkModules(M1, M2))
    exit(1);
  delete M2;   // We are done with this module.

  // Execute the program.
  Broken = BD.diffProgram(M1, "", "", false, &Error);
  if (!Error.empty()) {
    // Delete the linked module
    delete M1;
    return nullptr;
  }
  return M1;
}

/// TestFuncs - split functions in a Module into two groups: those that are
/// under consideration for miscompilation vs. those that are not, and test
/// accordingly. Each group of functions becomes a separate Module.
///
bool ReduceMiscompilingFunctions::TestFuncs(const std::vector<Function*> &Funcs,
                                            std::string &Error) {
  // Test to see if the function is misoptimized if we ONLY run it on the
  // functions listed in Funcs.
  outs() << "Checking to see if the program is misoptimized when "
         << (Funcs.size()==1 ? "this function is" : "these functions are")
         << " run through the pass"
         << (BD.getPassesToRun().size() == 1 ? "" : "es") << ":";
  PrintFunctionList(Funcs);
  outs() << '\n';

  // Create a clone for two reasons:
  // * If the optimization passes delete any function, the deleted function
  //   will be in the clone and Funcs will still point to valid memory
  // * If the optimization passes use interprocedural information to break
  //   a function, we want to continue with the original function. Otherwise
  //   we can conclude that a function triggers the bug when in fact one
  //   needs a larger set of original functions to do so.
  ValueToValueMapTy VMap;
  Module *Clone = CloneModule(BD.getProgram(), VMap);
  Module *Orig = BD.swapProgramIn(Clone);

  std::vector<Function*> FuncsOnClone;
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i) {
    Function *F = cast<Function>(VMap[Funcs[i]]);
    FuncsOnClone.push_back(F);
  }

  // Split the module into the two halves of the program we want.
  VMap.clear();
  Module *ToNotOptimize = CloneModule(BD.getProgram(), VMap);
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize, FuncsOnClone,
                                                 VMap);

  // Run the predicate, note that the predicate will delete both input modules.
  bool Broken = TestFn(BD, ToOptimize, ToNotOptimize, Error);

  delete BD.swapProgramIn(Orig);

  return Broken;
}

/// DisambiguateGlobalSymbols - Give anonymous global values names.
///
static void DisambiguateGlobalSymbols(Module *M) {
  for (Module::global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    if (!I->hasName())
      I->setName("anon_global");
  for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->hasName())
      I->setName("anon_fn");
}

/// ExtractLoops - Given a reduced list of functions that still exposed the bug,
/// check to see if we can extract the loops in the region without obscuring the
/// bug.  If so, it reduces the amount of code identified.
///
static bool ExtractLoops(BugDriver &BD,
                         bool (*TestFn)(BugDriver &, Module *, Module *,
                                        std::string &),
                         std::vector<Function*> &MiscompiledFunctions,
                         std::string &Error) {
  bool MadeChange = false;
  while (1) {
    if (BugpointIsInterrupted) return MadeChange;

    ValueToValueMapTy VMap;
    Module *ToNotOptimize = CloneModule(BD.getProgram(), VMap);
    Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize,
                                                   MiscompiledFunctions,
                                                   VMap);
    Module *ToOptimizeLoopExtracted = BD.extractLoop(ToOptimize).release();
    if (!ToOptimizeLoopExtracted) {
      // If the loop extractor crashed or if there were no extractible loops,
      // then this chapter of our odyssey is over with.
      delete ToNotOptimize;
      delete ToOptimize;
      return MadeChange;
    }

    errs() << "Extracted a loop from the breaking portion of the program.\n";

    // Bugpoint is intentionally not very trusting of LLVM transformations.  In
    // particular, we're not going to assume that the loop extractor works, so
    // we're going to test the newly loop extracted program to make sure nothing
    // has broken.  If something broke, then we'll inform the user and stop
    // extraction.
    AbstractInterpreter *AI = BD.switchToSafeInterpreter();
    bool Failure;
    Module *New = TestMergedProgram(BD, ToOptimizeLoopExtracted,
                                    ToNotOptimize, false, Error, Failure);
    if (!New)
      return false;

    // Delete the original and set the new program.
    Module *Old = BD.swapProgramIn(New);
    for (unsigned i = 0, e = MiscompiledFunctions.size(); i != e; ++i)
      MiscompiledFunctions[i] = cast<Function>(VMap[MiscompiledFunctions[i]]);
    delete Old;

    if (Failure) {
      BD.switchToInterpreter(AI);

      // Merged program doesn't work anymore!
      errs() << "  *** ERROR: Loop extraction broke the program. :("
             << " Please report a bug!\n";
      errs() << "      Continuing on with un-loop-extracted version.\n";

      BD.writeProgramToFile(OutputPrefix + "-loop-extract-fail-tno.bc",
                            ToNotOptimize);
      BD.writeProgramToFile(OutputPrefix + "-loop-extract-fail-to.bc",
                            ToOptimize);
      BD.writeProgramToFile(OutputPrefix + "-loop-extract-fail-to-le.bc",
                            ToOptimizeLoopExtracted);

      errs() << "Please submit the "
             << OutputPrefix << "-loop-extract-fail-*.bc files.\n";
      delete ToOptimize;
      delete ToNotOptimize;
      return MadeChange;
    }
    delete ToOptimize;
    BD.switchToInterpreter(AI);

    outs() << "  Testing after loop extraction:\n";
    // Clone modules, the tester function will free them.
    Module *TOLEBackup = CloneModule(ToOptimizeLoopExtracted, VMap);
    Module *TNOBackup  = CloneModule(ToNotOptimize, VMap);

    for (unsigned i = 0, e = MiscompiledFunctions.size(); i != e; ++i)
      MiscompiledFunctions[i] = cast<Function>(VMap[MiscompiledFunctions[i]]);

    Failure = TestFn(BD, ToOptimizeLoopExtracted, ToNotOptimize, Error);
    if (!Error.empty())
      return false;

    ToOptimizeLoopExtracted = TOLEBackup;
    ToNotOptimize = TNOBackup;

    if (!Failure) {
      outs() << "*** Loop extraction masked the problem.  Undoing.\n";
      // If the program is not still broken, then loop extraction did something
      // that masked the error.  Stop loop extraction now.

      std::vector<std::pair<std::string, FunctionType*> > MisCompFunctions;
      for (Function *F : MiscompiledFunctions) {
        MisCompFunctions.emplace_back(F->getName(), F->getFunctionType());
      }

      if (Linker::LinkModules(ToNotOptimize, ToOptimizeLoopExtracted))
        exit(1);

      MiscompiledFunctions.clear();
      for (unsigned i = 0, e = MisCompFunctions.size(); i != e; ++i) {
        Function *NewF = ToNotOptimize->getFunction(MisCompFunctions[i].first);

        assert(NewF && "Function not found??");
        MiscompiledFunctions.push_back(NewF);
      }

      delete ToOptimizeLoopExtracted;
      BD.setNewProgram(ToNotOptimize);
      return MadeChange;
    }

    outs() << "*** Loop extraction successful!\n";

    std::vector<std::pair<std::string, FunctionType*> > MisCompFunctions;
    for (Module::iterator I = ToOptimizeLoopExtracted->begin(),
           E = ToOptimizeLoopExtracted->end(); I != E; ++I)
      if (!I->isDeclaration())
        MisCompFunctions.emplace_back(I->getName(), I->getFunctionType());

    // Okay, great!  Now we know that we extracted a loop and that loop
    // extraction both didn't break the program, and didn't mask the problem.
    // Replace the current program with the loop extracted version, and try to
    // extract another loop.
    if (Linker::LinkModules(ToNotOptimize, ToOptimizeLoopExtracted))
      exit(1);

    delete ToOptimizeLoopExtracted;

    // All of the Function*'s in the MiscompiledFunctions list are in the old
    // module.  Update this list to include all of the functions in the
    // optimized and loop extracted module.
    MiscompiledFunctions.clear();
    for (unsigned i = 0, e = MisCompFunctions.size(); i != e; ++i) {
      Function *NewF = ToNotOptimize->getFunction(MisCompFunctions[i].first);

      assert(NewF && "Function not found??");
      MiscompiledFunctions.push_back(NewF);
    }

    BD.setNewProgram(ToNotOptimize);
    MadeChange = true;
  }
}

namespace {
  class ReduceMiscompiledBlocks : public ListReducer<BasicBlock*> {
    BugDriver &BD;
    bool (*TestFn)(BugDriver &, Module *, Module *, std::string &);
    std::vector<Function*> FunctionsBeingTested;
  public:
    ReduceMiscompiledBlocks(BugDriver &bd,
                            bool (*F)(BugDriver &, Module *, Module *,
                                      std::string &),
                            const std::vector<Function*> &Fns)
      : BD(bd), TestFn(F), FunctionsBeingTested(Fns) {}

    TestResult doTest(std::vector<BasicBlock*> &Prefix,
                      std::vector<BasicBlock*> &Suffix,
                      std::string &Error) override {
      if (!Suffix.empty()) {
        bool Ret = TestFuncs(Suffix, Error);
        if (!Error.empty())
          return InternalError;
        if (Ret)
          return KeepSuffix;
      }
      if (!Prefix.empty()) {
        bool Ret = TestFuncs(Prefix, Error);
        if (!Error.empty())
          return InternalError;
        if (Ret)
          return KeepPrefix;
      }
      return NoFailure;
    }

    bool TestFuncs(const std::vector<BasicBlock*> &BBs, std::string &Error);
  };
}

/// TestFuncs - Extract all blocks for the miscompiled functions except for the
/// specified blocks.  If the problem still exists, return true.
///
bool ReduceMiscompiledBlocks::TestFuncs(const std::vector<BasicBlock*> &BBs,
                                        std::string &Error) {
  // Test to see if the function is misoptimized if we ONLY run it on the
  // functions listed in Funcs.
  outs() << "Checking to see if the program is misoptimized when all ";
  if (!BBs.empty()) {
    outs() << "but these " << BBs.size() << " blocks are extracted: ";
    for (unsigned i = 0, e = BBs.size() < 10 ? BBs.size() : 10; i != e; ++i)
      outs() << BBs[i]->getName() << " ";
    if (BBs.size() > 10) outs() << "...";
  } else {
    outs() << "blocks are extracted.";
  }
  outs() << '\n';

  // Split the module into the two halves of the program we want.
  ValueToValueMapTy VMap;
  Module *Clone = CloneModule(BD.getProgram(), VMap);
  Module *Orig = BD.swapProgramIn(Clone);
  std::vector<Function*> FuncsOnClone;
  std::vector<BasicBlock*> BBsOnClone;
  for (unsigned i = 0, e = FunctionsBeingTested.size(); i != e; ++i) {
    Function *F = cast<Function>(VMap[FunctionsBeingTested[i]]);
    FuncsOnClone.push_back(F);
  }
  for (unsigned i = 0, e = BBs.size(); i != e; ++i) {
    BasicBlock *BB = cast<BasicBlock>(VMap[BBs[i]]);
    BBsOnClone.push_back(BB);
  }
  VMap.clear();

  Module *ToNotOptimize = CloneModule(BD.getProgram(), VMap);
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize,
                                                 FuncsOnClone,
                                                 VMap);

  // Try the extraction.  If it doesn't work, then the block extractor crashed
  // or something, in which case bugpoint can't chase down this possibility.
  if (std::unique_ptr<Module> New =
          BD.extractMappedBlocksFromModule(BBsOnClone, ToOptimize)) {
    delete ToOptimize;
    // Run the predicate,
    // note that the predicate will delete both input modules.
    bool Ret = TestFn(BD, New.get(), ToNotOptimize, Error);
    delete BD.swapProgramIn(Orig);
    return Ret;
  }
  delete BD.swapProgramIn(Orig);
  delete ToOptimize;
  delete ToNotOptimize;
  return false;
}


/// ExtractBlocks - Given a reduced list of functions that still expose the bug,
/// extract as many basic blocks from the region as possible without obscuring
/// the bug.
///
static bool ExtractBlocks(BugDriver &BD,
                          bool (*TestFn)(BugDriver &, Module *, Module *,
                                         std::string &),
                          std::vector<Function*> &MiscompiledFunctions,
                          std::string &Error) {
  if (BugpointIsInterrupted) return false;

  std::vector<BasicBlock*> Blocks;
  for (unsigned i = 0, e = MiscompiledFunctions.size(); i != e; ++i)
    for (BasicBlock &BB : *MiscompiledFunctions[i])
      Blocks.push_back(&BB);

  // Use the list reducer to identify blocks that can be extracted without
  // obscuring the bug.  The Blocks list will end up containing blocks that must
  // be retained from the original program.
  unsigned OldSize = Blocks.size();

  // Check to see if all blocks are extractible first.
  bool Ret = ReduceMiscompiledBlocks(BD, TestFn, MiscompiledFunctions)
                                  .TestFuncs(std::vector<BasicBlock*>(), Error);
  if (!Error.empty())
    return false;
  if (Ret) {
    Blocks.clear();
  } else {
    ReduceMiscompiledBlocks(BD, TestFn,
                            MiscompiledFunctions).reduceList(Blocks, Error);
    if (!Error.empty())
      return false;
    if (Blocks.size() == OldSize)
      return false;
  }

  ValueToValueMapTy VMap;
  Module *ProgClone = CloneModule(BD.getProgram(), VMap);
  Module *ToExtract = SplitFunctionsOutOfModule(ProgClone,
                                                MiscompiledFunctions,
                                                VMap);
  std::unique_ptr<Module> Extracted =
      BD.extractMappedBlocksFromModule(Blocks, ToExtract);
  if (!Extracted) {
    // Weird, extraction should have worked.
    errs() << "Nondeterministic problem extracting blocks??\n";
    delete ProgClone;
    delete ToExtract;
    return false;
  }

  // Otherwise, block extraction succeeded.  Link the two program fragments back
  // together.
  delete ToExtract;

  std::vector<std::pair<std::string, FunctionType*> > MisCompFunctions;
  for (Module::iterator I = Extracted->begin(), E = Extracted->end();
       I != E; ++I)
    if (!I->isDeclaration())
      MisCompFunctions.emplace_back(I->getName(), I->getFunctionType());

  if (Linker::LinkModules(ProgClone, Extracted.get()))
    exit(1);

  // Set the new program and delete the old one.
  BD.setNewProgram(ProgClone);

  // Update the list of miscompiled functions.
  MiscompiledFunctions.clear();

  for (unsigned i = 0, e = MisCompFunctions.size(); i != e; ++i) {
    Function *NewF = ProgClone->getFunction(MisCompFunctions[i].first);
    assert(NewF && "Function not found??");
    MiscompiledFunctions.push_back(NewF);
  }

  return true;
}


/// DebugAMiscompilation - This is a generic driver to narrow down
/// miscompilations, either in an optimization or a code generator.
///
static std::vector<Function*>
DebugAMiscompilation(BugDriver &BD,
                     bool (*TestFn)(BugDriver &, Module *, Module *,
                                    std::string &),
                     std::string &Error) {
  // Okay, now that we have reduced the list of passes which are causing the
  // failure, see if we can pin down which functions are being
  // miscompiled... first build a list of all of the non-external functions in
  // the program.
  std::vector<Function*> MiscompiledFunctions;
  Module *Prog = BD.getProgram();
  for (Function &F : *Prog)
    if (!F.isDeclaration())
      MiscompiledFunctions.push_back(&F);

  // Do the reduction...
  if (!BugpointIsInterrupted)
    ReduceMiscompilingFunctions(BD, TestFn).reduceList(MiscompiledFunctions,
                                                       Error);
  if (!Error.empty()) {
    errs() << "\n***Cannot reduce functions: ";
    return MiscompiledFunctions;
  }
  outs() << "\n*** The following function"
         << (MiscompiledFunctions.size() == 1 ? " is" : "s are")
         << " being miscompiled: ";
  PrintFunctionList(MiscompiledFunctions);
  outs() << '\n';

  // See if we can rip any loops out of the miscompiled functions and still
  // trigger the problem.

  if (!BugpointIsInterrupted && !DisableLoopExtraction) {
    bool Ret = ExtractLoops(BD, TestFn, MiscompiledFunctions, Error);
    if (!Error.empty())
      return MiscompiledFunctions;
    if (Ret) {
      // Okay, we extracted some loops and the problem still appears.  See if
      // we can eliminate some of the created functions from being candidates.
      DisambiguateGlobalSymbols(BD.getProgram());

      // Do the reduction...
      if (!BugpointIsInterrupted)
        ReduceMiscompilingFunctions(BD, TestFn).reduceList(MiscompiledFunctions,
                                                           Error);
      if (!Error.empty())
        return MiscompiledFunctions;

      outs() << "\n*** The following function"
             << (MiscompiledFunctions.size() == 1 ? " is" : "s are")
             << " being miscompiled: ";
      PrintFunctionList(MiscompiledFunctions);
      outs() << '\n';
    }
  }

  if (!BugpointIsInterrupted && !DisableBlockExtraction) {
    bool Ret = ExtractBlocks(BD, TestFn, MiscompiledFunctions, Error);
    if (!Error.empty())
      return MiscompiledFunctions;
    if (Ret) {
      // Okay, we extracted some blocks and the problem still appears.  See if
      // we can eliminate some of the created functions from being candidates.
      DisambiguateGlobalSymbols(BD.getProgram());

      // Do the reduction...
      ReduceMiscompilingFunctions(BD, TestFn).reduceList(MiscompiledFunctions,
                                                         Error);
      if (!Error.empty())
        return MiscompiledFunctions;

      outs() << "\n*** The following function"
             << (MiscompiledFunctions.size() == 1 ? " is" : "s are")
             << " being miscompiled: ";
      PrintFunctionList(MiscompiledFunctions);
      outs() << '\n';
    }
  }

  return MiscompiledFunctions;
}

/// TestOptimizer - This is the predicate function used to check to see if the
/// "Test" portion of the program is misoptimized.  If so, return true.  In any
/// case, both module arguments are deleted.
///
static bool TestOptimizer(BugDriver &BD, Module *Test, Module *Safe,
                          std::string &Error) {
  // Run the optimization passes on ToOptimize, producing a transformed version
  // of the functions being tested.
  outs() << "  Optimizing functions being tested: ";
  std::unique_ptr<Module> Optimized = BD.runPassesOn(Test, BD.getPassesToRun(),
                                                     /*AutoDebugCrashes*/ true);
  outs() << "done.\n";
  delete Test;

  outs() << "  Checking to see if the merged program executes correctly: ";
  bool Broken;
  Module *New =
      TestMergedProgram(BD, Optimized.get(), Safe, true, Error, Broken);
  if (New) {
    outs() << (Broken ? " nope.\n" : " yup.\n");
    // Delete the original and set the new program.
    delete BD.swapProgramIn(New);
  }
  return Broken;
}


/// debugMiscompilation - This method is used when the passes selected are not
/// crashing, but the generated output is semantically different from the
/// input.
///
void BugDriver::debugMiscompilation(std::string *Error) {
  // Make sure something was miscompiled...
  if (!BugpointIsInterrupted)
    if (!ReduceMiscompilingPasses(*this).reduceList(PassesToRun, *Error)) {
      if (Error->empty())
        errs() << "*** Optimized program matches reference output!  No problem"
               << " detected...\nbugpoint can't help you with your problem!\n";
      return;
    }

  outs() << "\n*** Found miscompiling pass"
         << (getPassesToRun().size() == 1 ? "" : "es") << ": "
         << getPassesString(getPassesToRun()) << '\n';
  EmitProgressBitcode(Program, "passinput");

  std::vector<Function *> MiscompiledFunctions =
    DebugAMiscompilation(*this, TestOptimizer, *Error);
  if (!Error->empty())
    return;

  // Output a bunch of bitcode files for the user...
  outs() << "Outputting reduced bitcode files which expose the problem:\n";
  ValueToValueMapTy VMap;
  Module *ToNotOptimize = CloneModule(getProgram(), VMap);
  Module *ToOptimize = SplitFunctionsOutOfModule(ToNotOptimize,
                                                 MiscompiledFunctions,
                                                 VMap);

  outs() << "  Non-optimized portion: ";
  EmitProgressBitcode(ToNotOptimize, "tonotoptimize", true);
  delete ToNotOptimize;  // Delete hacked module.

  outs() << "  Portion that is input to optimizer: ";
  EmitProgressBitcode(ToOptimize, "tooptimize");
  delete ToOptimize;      // Delete hacked module.

  return;
}

/// CleanupAndPrepareModules - Get the specified modules ready for code
/// generator testing.
///
static void CleanupAndPrepareModules(BugDriver &BD, Module *&Test,
                                     Module *Safe) {
  // Clean up the modules, removing extra cruft that we don't need anymore...
  Test = BD.performFinalCleanups(Test).release();

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
        args.push_back(&*I);
      }

      // Call the old main function and return its result
      BasicBlock *BB = BasicBlock::Create(Safe->getContext(), "entry", newMain);
      CallInst *call = CallInst::Create(oldMainProto, args, "", BB);

      // If the type of old function wasn't void, return value of call
      ReturnInst::Create(Safe->getContext(), call, BB);
    }

  // The second nasty issue we must deal with in the JIT is that the Safe
  // module cannot directly reference any functions defined in the test
  // module.  Instead, we use a JIT API call to dynamically resolve the
  // symbol.

  // Add the resolver to the Safe module.
  // Prototype: void *getPointerToNamedFunction(const char* Name)
  Constant *resolverFunc =
    Safe->getOrInsertFunction("getPointerToNamedFunction",
                    Type::getInt8PtrTy(Safe->getContext()),
                    Type::getInt8PtrTy(Safe->getContext()),
                       (Type *)nullptr);

  // Use the function we just added to get addresses of functions we need.
  for (Module::iterator F = Safe->begin(), E = Safe->end(); F != E; ++F) {
    if (F->isDeclaration() && !F->use_empty() && &*F != resolverFunc &&
        !F->isIntrinsic() /* ignore intrinsics */) {
      Function *TestFn = Test->getFunction(F->getName());

      // Don't forward functions which are external in the test module too.
      if (TestFn && !TestFn->isDeclaration()) {
        // 1. Add a string constant with its name to the global file
        Constant *InitArray =
          ConstantDataArray::getString(F->getContext(), F->getName());
        GlobalVariable *funcName =
          new GlobalVariable(*Safe, InitArray->getType(), true /*isConstant*/,
                             GlobalValue::InternalLinkage, InitArray,
                             F->getName() + "_name");

        // 2. Use `GetElementPtr *funcName, 0, 0' to convert the string to an
        // sbyte* so it matches the signature of the resolver function.

        // GetElementPtr *funcName, ulong 0, ulong 0
        std::vector<Constant*> GEPargs(2,
                     Constant::getNullValue(Type::getInt32Ty(F->getContext())));
        Value *GEP = ConstantExpr::getGetElementPtr(InitArray->getType(),
                                                    funcName, GEPargs);
        std::vector<Value*> ResolverArgs;
        ResolverArgs.push_back(GEP);

        // Rewrite uses of F in global initializers, etc. to uses of a wrapper
        // function that dynamically resolves the calls to F via our JIT API
        if (!F->use_empty()) {
          // Create a new global to hold the cached function pointer.
          Constant *NullPtr = ConstantPointerNull::get(F->getType());
          GlobalVariable *Cache =
            new GlobalVariable(*F->getParent(), F->getType(),
                               false, GlobalValue::InternalLinkage,
                               NullPtr,F->getName()+".fpcache");

          // Construct a new stub function that will re-route calls to F
          FunctionType *FuncTy = F->getFunctionType();
          Function *FuncWrapper = Function::Create(FuncTy,
                                                   GlobalValue::InternalLinkage,
                                                   F->getName() + "_wrapper",
                                                   F->getParent());
          BasicBlock *EntryBB  = BasicBlock::Create(F->getContext(),
                                                    "entry", FuncWrapper);
          BasicBlock *DoCallBB = BasicBlock::Create(F->getContext(),
                                                    "usecache", FuncWrapper);
          BasicBlock *LookupBB = BasicBlock::Create(F->getContext(),
                                                    "lookupfp", FuncWrapper);

          // Check to see if we already looked up the value.
          Value *CachedVal = new LoadInst(Cache, "fpcache", EntryBB);
          Value *IsNull = new ICmpInst(*EntryBB, ICmpInst::ICMP_EQ, CachedVal,
                                       NullPtr, "isNull");
          BranchInst::Create(LookupBB, DoCallBB, IsNull, EntryBB);

          // Resolve the call to function F via the JIT API:
          //
          // call resolver(GetElementPtr...)
          CallInst *Resolver =
            CallInst::Create(resolverFunc, ResolverArgs, "resolver", LookupBB);

          // Cast the result from the resolver to correctly-typed function.
          CastInst *CastedResolver =
            new BitCastInst(Resolver,
                            PointerType::getUnqual(F->getFunctionType()),
                            "resolverCast", LookupBB);

          // Save the value in our cache.
          new StoreInst(CastedResolver, Cache, LookupBB);
          BranchInst::Create(DoCallBB, LookupBB);

          PHINode *FuncPtr = PHINode::Create(NullPtr->getType(), 2,
                                             "fp", DoCallBB);
          FuncPtr->addIncoming(CastedResolver, LookupBB);
          FuncPtr->addIncoming(CachedVal, EntryBB);

          // Save the argument list.
          std::vector<Value*> Args;
          for (Argument &A : FuncWrapper->args())
            Args.push_back(&A);

          // Pass on the arguments to the real function, return its result
          if (F->getReturnType()->isVoidTy()) {
            CallInst::Create(FuncPtr, Args, "", DoCallBB);
            ReturnInst::Create(F->getContext(), DoCallBB);
          } else {
            CallInst *Call = CallInst::Create(FuncPtr, Args,
                                              "retval", DoCallBB);
            ReturnInst::Create(F->getContext(),Call, DoCallBB);
          }

          // Use the wrapper function instead of the old function
          F->replaceAllUsesWith(FuncWrapper);
        }
      }
    }
  }

  if (verifyModule(*Test) || verifyModule(*Safe)) {
    errs() << "Bugpoint has a bug, which corrupted a module!!\n";
    abort();
  }
}



/// TestCodeGenerator - This is the predicate function used to check to see if
/// the "Test" portion of the program is miscompiled by the code generator under
/// test.  If so, return true.  In any case, both module arguments are deleted.
///
static bool TestCodeGenerator(BugDriver &BD, Module *Test, Module *Safe,
                              std::string &Error) {
  CleanupAndPrepareModules(BD, Test, Safe);

  SmallString<128> TestModuleBC;
  int TestModuleFD;
  std::error_code EC = sys::fs::createTemporaryFile("bugpoint.test", "bc",
                                                    TestModuleFD, TestModuleBC);
  if (EC) {
    errs() << BD.getToolName() << "Error making unique filename: "
           << EC.message() << "\n";
    exit(1);
  }
  if (BD.writeProgramToFile(TestModuleBC.str(), TestModuleFD, Test)) {
    errs() << "Error writing bitcode to `" << TestModuleBC.str()
           << "'\nExiting.";
    exit(1);
  }
  delete Test;

  FileRemover TestModuleBCRemover(TestModuleBC.str(), !SaveTemps);

  // Make the shared library
  SmallString<128> SafeModuleBC;
  int SafeModuleFD;
  EC = sys::fs::createTemporaryFile("bugpoint.safe", "bc", SafeModuleFD,
                                    SafeModuleBC);
  if (EC) {
    errs() << BD.getToolName() << "Error making unique filename: "
           << EC.message() << "\n";
    exit(1);
  }

  if (BD.writeProgramToFile(SafeModuleBC.str(), SafeModuleFD, Safe)) {
    errs() << "Error writing bitcode to `" << SafeModuleBC
           << "'\nExiting.";
    exit(1);
  }

  FileRemover SafeModuleBCRemover(SafeModuleBC.str(), !SaveTemps);

  std::string SharedObject = BD.compileSharedObject(SafeModuleBC.str(), Error);
  if (!Error.empty())
    return false;
  delete Safe;

  FileRemover SharedObjectRemover(SharedObject, !SaveTemps);

  // Run the code generator on the `Test' code, loading the shared library.
  // The function returns whether or not the new output differs from reference.
  bool Result = BD.diffProgram(BD.getProgram(), TestModuleBC.str(),
                               SharedObject, false, &Error);
  if (!Error.empty())
    return false;

  if (Result)
    errs() << ": still failing!\n";
  else
    errs() << ": didn't fail.\n";

  return Result;
}


/// debugCodeGenerator - debug errors in LLC, LLI, or CBE.
///
bool BugDriver::debugCodeGenerator(std::string *Error) {
  if ((void*)SafeInterpreter == (void*)Interpreter) {
    std::string Result = executeProgramSafely(Program, "bugpoint.safe.out",
                                              Error);
    if (Error->empty()) {
      outs() << "\n*** The \"safe\" i.e. 'known good' backend cannot match "
             << "the reference diff.  This may be due to a\n    front-end "
             << "bug or a bug in the original program, but this can also "
             << "happen if bugpoint isn't running the program with the "
             << "right flags or input.\n    I left the result of executing "
             << "the program with the \"safe\" backend in this file for "
             << "you: '"
             << Result << "'.\n";
    }
    return true;
  }

  DisambiguateGlobalSymbols(Program);

  std::vector<Function*> Funcs = DebugAMiscompilation(*this, TestCodeGenerator,
                                                      *Error);
  if (!Error->empty())
    return true;

  // Split the module into the two halves of the program we want.
  ValueToValueMapTy VMap;
  Module *ToNotCodeGen = CloneModule(getProgram(), VMap);
  Module *ToCodeGen = SplitFunctionsOutOfModule(ToNotCodeGen, Funcs, VMap);

  // Condition the modules
  CleanupAndPrepareModules(*this, ToCodeGen, ToNotCodeGen);

  SmallString<128> TestModuleBC;
  int TestModuleFD;
  std::error_code EC = sys::fs::createTemporaryFile("bugpoint.test", "bc",
                                                    TestModuleFD, TestModuleBC);
  if (EC) {
    errs() << getToolName() << "Error making unique filename: "
           << EC.message() << "\n";
    exit(1);
  }

  if (writeProgramToFile(TestModuleBC.str(), TestModuleFD, ToCodeGen)) {
    errs() << "Error writing bitcode to `" << TestModuleBC
           << "'\nExiting.";
    exit(1);
  }
  delete ToCodeGen;

  // Make the shared library
  SmallString<128> SafeModuleBC;
  int SafeModuleFD;
  EC = sys::fs::createTemporaryFile("bugpoint.safe", "bc", SafeModuleFD,
                                    SafeModuleBC);
  if (EC) {
    errs() << getToolName() << "Error making unique filename: "
           << EC.message() << "\n";
    exit(1);
  }

  if (writeProgramToFile(SafeModuleBC.str(), SafeModuleFD, ToNotCodeGen)) {
    errs() << "Error writing bitcode to `" << SafeModuleBC
           << "'\nExiting.";
    exit(1);
  }
  std::string SharedObject = compileSharedObject(SafeModuleBC.str(), *Error);
  if (!Error->empty())
    return true;
  delete ToNotCodeGen;

  outs() << "You can reproduce the problem with the command line: \n";
  if (isExecutingJIT()) {
    outs() << "  lli -load " << SharedObject << " " << TestModuleBC;
  } else {
    outs() << "  llc " << TestModuleBC << " -o " << TestModuleBC
           << ".s\n";
    outs() << "  cc " << SharedObject << " " << TestModuleBC.str()
              << ".s -o " << TestModuleBC << ".exe";
#if defined (HAVE_LINK_R)
    outs() << " -Wl,-R.";
#endif
    outs() << "\n";
    outs() << "  " << TestModuleBC << ".exe";
  }
  for (unsigned i = 0, e = InputArgv.size(); i != e; ++i)
    outs() << " " << InputArgv[i];
  outs() << '\n';
  outs() << "The shared object was created with:\n  llc -march=c "
         << SafeModuleBC.str() << " -o temporary.c\n"
         << "  cc -xc temporary.c -O2 -o " << SharedObject;
  if (TargetTriple.getArch() == Triple::sparc)
    outs() << " -G";              // Compile a shared library, `-G' for Sparc
  else
    outs() << " -fPIC -shared";   // `-shared' for Linux/X86, maybe others

  outs() << " -fno-strict-aliasing\n";

  return false;
}
