//===- BugDriver.h - Top-Level BugPoint class -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class contains all of the shared state and information that is used by
// the BugPoint tool to track down errors in optimizations.  This class is the
// main driver class that invokes all sub-functionality.
//
//===----------------------------------------------------------------------===//

#ifndef BUGDRIVER_H
#define BUGDRIVER_H

#include <vector>
#include <string>

namespace llvm {

class PassInfo;
class Module;
class GlobalVariable;
class Function;
class BasicBlock;
class AbstractInterpreter;
class Instruction;

class DebugCrashes;

class GCC;

extern bool DisableSimplifyCFG;

/// BugpointIsInterrupted - Set to true when the user presses ctrl-c.
///
extern bool BugpointIsInterrupted;

class BugDriver {
  const std::string ToolName;  // Name of bugpoint
  std::string ReferenceOutputFile; // Name of `good' output file
  Module *Program;             // The raw program, linked together
  std::vector<const PassInfo*> PassesToRun;
  AbstractInterpreter *Interpreter;   // How to run the program
  AbstractInterpreter *SafeInterpreter;  // To generate reference output, etc.
  GCC *gcc;
  bool run_as_child;
  bool run_find_bugs;
  unsigned Timeout;
  unsigned MemoryLimit;

  // FIXME: sort out public/private distinctions...
  friend class ReducePassList;
  friend class ReduceMisCodegenFunctions;

public:
  BugDriver(const char *toolname, bool as_child, bool find_bugs,
            unsigned timeout, unsigned memlimit);

  const std::string &getToolName() const { return ToolName; }

  // Set up methods... these methods are used to copy information about the
  // command line arguments into instance variables of BugDriver.
  //
  bool addSources(const std::vector<std::string> &FileNames);
  template<class It>
  void addPasses(It I, It E) { PassesToRun.insert(PassesToRun.end(), I, E); }
  void setPassesToRun(const std::vector<const PassInfo*> &PTR) {
    PassesToRun = PTR;
  }
  const std::vector<const PassInfo*> &getPassesToRun() const {
    return PassesToRun;
  }

  /// run - The top level method that is invoked after all of the instance
  /// variables are set up from command line arguments. The \p as_child argument
  /// indicates whether the driver is to run in parent mode or child mode.
  ///
  bool run();

  /// debugOptimizerCrash - This method is called when some optimizer pass
  /// crashes on input.  It attempts to prune down the testcase to something
  /// reasonable, and figure out exactly which pass is crashing.
  ///
  bool debugOptimizerCrash(const std::string &ID = "passes");

  /// debugCodeGeneratorCrash - This method is called when the code generator
  /// crashes on an input.  It attempts to reduce the input as much as possible
  /// while still causing the code generator to crash.
  bool debugCodeGeneratorCrash();

  /// debugMiscompilation - This method is used when the passes selected are not
  /// crashing, but the generated output is semantically different from the
  /// input.
  bool debugMiscompilation();

  /// debugPassMiscompilation - This method is called when the specified pass
  /// miscompiles Program as input.  It tries to reduce the testcase to
  /// something that smaller that still miscompiles the program.
  /// ReferenceOutput contains the filename of the file containing the output we
  /// are to match.
  ///
  bool debugPassMiscompilation(const PassInfo *ThePass,
                               const std::string &ReferenceOutput);

  /// compileSharedObject - This method creates a SharedObject from a given
  /// BitcodeFile for debugging a code generator.
  ///
  std::string compileSharedObject(const std::string &BitcodeFile);

  /// debugCodeGenerator - This method narrows down a module to a function or
  /// set of functions, using the CBE as a ``safe'' code generator for other
  /// functions that are not under consideration.
  bool debugCodeGenerator();

  /// isExecutingJIT - Returns true if bugpoint is currently testing the JIT
  ///
  bool isExecutingJIT();

  /// runPasses - Run all of the passes in the "PassesToRun" list, discard the
  /// output, and return true if any of the passes crashed.
  bool runPasses(Module *M = 0) {
    if (M == 0) M = Program;
    std::swap(M, Program);
    bool Result = runPasses(PassesToRun);
    std::swap(M, Program);
    return Result;
  }

  Module *getProgram() const { return Program; }

  /// swapProgramIn - Set the current module to the specified module, returning
  /// the old one.
  Module *swapProgramIn(Module *M) {
    Module *OldProgram = Program;
    Program = M;
    return OldProgram;
  }

  AbstractInterpreter *switchToSafeInterpreter() {
    AbstractInterpreter *Old = Interpreter;
    Interpreter = (AbstractInterpreter*)SafeInterpreter;
    return Old;
  }

  void switchToInterpreter(AbstractInterpreter *AI) {
    Interpreter = AI;
  }

  /// setNewProgram - If we reduce or update the program somehow, call this
  /// method to update bugdriver with it.  This deletes the old module and sets
  /// the specified one as the current program.
  void setNewProgram(Module *M);

  /// compileProgram - Try to compile the specified module, throwing an
  /// exception if an error occurs, or returning normally if not.  This is used
  /// for code generation crash testing.
  ///
  void compileProgram(Module *M);

  /// executeProgram - This method runs "Program", capturing the output of the
  /// program to a file, returning the filename of the file.  A recommended
  /// filename may be optionally specified.  If there is a problem with the code
  /// generator (e.g., llc crashes), this will throw an exception.
  ///
  std::string executeProgram(std::string RequestedOutputFilename = "",
                             std::string Bitcode = "",
                             const std::string &SharedObjects = "",
                             AbstractInterpreter *AI = 0,
                             bool *ProgramExitedNonzero = 0);

  /// executeProgramSafely - Used to create reference output with the "safe"
  /// backend, if reference output is not provided.  If there is a problem with
  /// the code generator (e.g., llc crashes), this will throw an exception.
  ///
  std::string executeProgramSafely(std::string OutputFile = "");

  /// createReferenceFile - calls compileProgram and then records the output
  /// into ReferenceOutputFile. Returns true if reference file created, false 
  /// otherwise. Note: initializeExecutionEnvironment should be called BEFORE
  /// this function.
  ///
  bool createReferenceFile(Module *M, const std::string &Filename
                                            = "bugpoint.reference.out");

  /// diffProgram - This method executes the specified module and diffs the
  /// output against the file specified by ReferenceOutputFile.  If the output
  /// is different, true is returned.  If there is a problem with the code
  /// generator (e.g., llc crashes), this will throw an exception.
  ///
  bool diffProgram(const std::string &BitcodeFile = "",
                   const std::string &SharedObj = "",
                   bool RemoveBitcode = false);
                   
  /// EmitProgressBitcode - This function is used to output the current Program
  /// to a file named "bugpoint-ID.bc".
  ///
  void EmitProgressBitcode(const std::string &ID, bool NoFlyer = false);

  /// deleteInstructionFromProgram - This method clones the current Program and
  /// deletes the specified instruction from the cloned module.  It then runs a
  /// series of cleanup passes (ADCE and SimplifyCFG) to eliminate any code
  /// which depends on the value.  The modified module is then returned.
  ///
  Module *deleteInstructionFromProgram(const Instruction *I, unsigned Simp)
    const;

  /// performFinalCleanups - This method clones the current Program and performs
  /// a series of cleanups intended to get rid of extra cruft on the module.  If
  /// the MayModifySemantics argument is true, then the cleanups is allowed to
  /// modify how the code behaves.
  ///
  Module *performFinalCleanups(Module *M, bool MayModifySemantics = false);

  /// ExtractLoop - Given a module, extract up to one loop from it into a new
  /// function.  This returns null if there are no extractable loops in the
  /// program or if the loop extractor crashes.
  Module *ExtractLoop(Module *M);

  /// ExtractMappedBlocksFromModule - Extract all but the specified basic blocks
  /// into their own functions.  The only detail is that M is actually a module
  /// cloned from the one the BBs are in, so some mapping needs to be performed.
  /// If this operation fails for some reason (ie the implementation is buggy),
  /// this function should return null, otherwise it returns a new Module.
  Module *ExtractMappedBlocksFromModule(const std::vector<BasicBlock*> &BBs,
                                        Module *M);

  /// runPassesOn - Carefully run the specified set of pass on the specified
  /// module, returning the transformed module on success, or a null pointer on
  /// failure.  If AutoDebugCrashes is set to true, then bugpoint will
  /// automatically attempt to track down a crashing pass if one exists, and
  /// this method will never return null.
  Module *runPassesOn(Module *M, const std::vector<const PassInfo*> &Passes,
                      bool AutoDebugCrashes = false, unsigned NumExtraArgs = 0,
                      const char * const *ExtraArgs = NULL);

  /// runPasses - Run the specified passes on Program, outputting a bitcode
  /// file and writting the filename into OutputFile if successful.  If the
  /// optimizations fail for some reason (optimizer crashes), return true,
  /// otherwise return false.  If DeleteOutput is set to true, the bitcode is
  /// deleted on success, and the filename string is undefined.  This prints to
  /// cout a single line message indicating whether compilation was successful
  /// or failed, unless Quiet is set.  ExtraArgs specifies additional arguments
  /// to pass to the child bugpoint instance.
  ///
  bool runPasses(const std::vector<const PassInfo*> &PassesToRun,
                 std::string &OutputFilename, bool DeleteOutput = false,
                 bool Quiet = false, unsigned NumExtraArgs = 0,
                 const char * const *ExtraArgs = NULL) const;
                 
  /// runManyPasses - Take the specified pass list and create different 
  /// combinations of passes to compile the program with. Compile the program with
  /// each set and mark test to see if it compiled correctly. If the passes 
  /// compiled correctly output nothing and rearrange the passes into a new order.
  /// If the passes did not compile correctly, output the command required to 
  /// recreate the failure. This returns true if a compiler error is found.
  ///
  bool runManyPasses(const std::vector<const PassInfo*> &AllPasses);

  /// writeProgramToFile - This writes the current "Program" to the named
  /// bitcode file.  If an error occurs, true is returned.
  ///
  bool writeProgramToFile(const std::string &Filename, Module *M = 0) const;

private:
  /// runPasses - Just like the method above, but this just returns true or
  /// false indicating whether or not the optimizer crashed on the specified
  /// input (true = crashed).
  ///
  bool runPasses(const std::vector<const PassInfo*> &PassesToRun,
                 bool DeleteOutput = true) const {
    std::string Filename;
    return runPasses(PassesToRun, Filename, DeleteOutput);
  }

  /// runAsChild - The actual "runPasses" guts that runs in a child process.
  int runPassesAsChild(const std::vector<const PassInfo*> &PassesToRun);

  /// initializeExecutionEnvironment - This method is used to set up the
  /// environment for executing LLVM programs.
  ///
  bool initializeExecutionEnvironment();
};

/// ParseInputFile - Given a bitcode or assembly input filename, parse and
/// return it, or return null if not possible.
///
Module *ParseInputFile(const std::string &InputFilename);


/// getPassesString - Turn a list of passes into a string which indicates the
/// command line options that must be passed to add the passes.
///
std::string getPassesString(const std::vector<const PassInfo*> &Passes);

/// PrintFunctionList - prints out list of problematic functions
///
void PrintFunctionList(const std::vector<Function*> &Funcs);

/// PrintGlobalVariableList - prints out list of problematic global variables
///
void PrintGlobalVariableList(const std::vector<GlobalVariable*> &GVs);

// DeleteFunctionBody - "Remove" the function by deleting all of it's basic
// blocks, making it external.
//
void DeleteFunctionBody(Function *F);

/// SplitFunctionsOutOfModule - Given a module and a list of functions in the
/// module, split the functions OUT of the specified module, and place them in
/// the new module.
Module *SplitFunctionsOutOfModule(Module *M, const std::vector<Function*> &F);

} // End llvm namespace

#endif
