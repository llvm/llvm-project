//===- opt.cpp - The LLVM Modular Optimizer -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Optimizations may be specified an arbitrary number of times on the command
// line, They are run in the order specified.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/CallGraphSCCPass.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/PassNameParser.h"
#include "llvm/System/Signals.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/LinkAllVMCore.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
using namespace llvm;

// The OptimizationList is automatically populated with registered Passes by the
// PassNameParser.
//
static cl::list<const PassInfo*, bool, PassNameParser>
PassList(cl::desc("Optimizations available:"));

// Other command line options...
//
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode file>"), 
    cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
PrintEachXForm("p", cl::desc("Print module after each transformation"));

static cl::opt<bool>
NoOutput("disable-output",
         cl::desc("Do not write result bitcode file"), cl::Hidden);

static cl::opt<bool>
NoVerify("disable-verify", cl::desc("Do not verify result module"), cl::Hidden);

static cl::opt<bool>
VerifyEach("verify-each", cl::desc("Verify after each transform"));

static cl::opt<bool>
StripDebug("strip-debug",
           cl::desc("Strip debugger symbol info from translation unit"));

static cl::opt<bool>
DisableInline("disable-inlining", cl::desc("Do not run the inliner pass"));

static cl::opt<bool> 
DisableOptimizations("disable-opt", 
                     cl::desc("Do not run any optimization passes"));

static cl::opt<bool>
StandardCompileOpts("std-compile-opts", 
                   cl::desc("Include the standard compile time optimizations"));

static cl::opt<bool>
Quiet("q", cl::desc("Obsolete option"), cl::Hidden);

static cl::alias
QuietA("quiet", cl::desc("Alias for -q"), cl::aliasopt(Quiet));

static cl::opt<bool>
AnalyzeOnly("analyze", cl::desc("Only perform analysis, no optimization"));

// ---------- Define Printers for module and function passes ------------
namespace {

struct CallGraphSCCPassPrinter : public CallGraphSCCPass {
  static char ID;
  const PassInfo *PassToPrint;
  CallGraphSCCPassPrinter(const PassInfo *PI) : 
    CallGraphSCCPass((intptr_t)&ID), PassToPrint(PI) {}

  virtual bool runOnSCC(const std::vector<CallGraphNode *>&SCC) {
    if (!Quiet) {
      cout << "Printing analysis '" << PassToPrint->getPassName() << "':\n";

      for (unsigned i = 0, e = SCC.size(); i != e; ++i) {
        Function *F = SCC[i]->getFunction();
        if (F) 
          getAnalysisID<Pass>(PassToPrint).print(cout, F->getParent());
      }
    }
    // Get and print pass...
    return false;
  }
  
  virtual const char *getPassName() const { return "'Pass' Printer"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint);
    AU.setPreservesAll();
  }
};

char CallGraphSCCPassPrinter::ID = 0;

struct ModulePassPrinter : public ModulePass {
  static char ID;
  const PassInfo *PassToPrint;
  ModulePassPrinter(const PassInfo *PI) : ModulePass((intptr_t)&ID),
                                          PassToPrint(PI) {}

  virtual bool runOnModule(Module &M) {
    if (!Quiet) {
      cout << "Printing analysis '" << PassToPrint->getPassName() << "':\n";
      getAnalysisID<Pass>(PassToPrint).print(cout, &M);
    }

    // Get and print pass...
    return false;
  }

  virtual const char *getPassName() const { return "'Pass' Printer"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint);
    AU.setPreservesAll();
  }
};

char ModulePassPrinter::ID = 0;
struct FunctionPassPrinter : public FunctionPass {
  const PassInfo *PassToPrint;
  static char ID;
  FunctionPassPrinter(const PassInfo *PI) : FunctionPass((intptr_t)&ID),
                                            PassToPrint(PI) {}

  virtual bool runOnFunction(Function &F) {
    if (!Quiet) { 
      cout << "Printing analysis '" << PassToPrint->getPassName()
           << "' for function '" << F.getName() << "':\n";
    }
    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint).print(cout, F.getParent());
    return false;
  }

  virtual const char *getPassName() const { return "FunctionPass Printer"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint);
    AU.setPreservesAll();
  }
};

char FunctionPassPrinter::ID = 0;

struct LoopPassPrinter : public LoopPass {
  static char ID;
  const PassInfo *PassToPrint;
  LoopPassPrinter(const PassInfo *PI) : 
    LoopPass((intptr_t)&ID), PassToPrint(PI) {}

  virtual bool runOnLoop(Loop *L, LPPassManager &LPM) {
    if (!Quiet) {
      cout << "Printing analysis '" << PassToPrint->getPassName() << "':\n";
      getAnalysisID<Pass>(PassToPrint).print(cout, 
                                  L->getHeader()->getParent()->getParent());
    }
    // Get and print pass...
    return false;
  }
  
  virtual const char *getPassName() const { return "'Pass' Printer"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint);
    AU.setPreservesAll();
  }
};

char LoopPassPrinter::ID = 0;

struct BasicBlockPassPrinter : public BasicBlockPass {
  const PassInfo *PassToPrint;
  static char ID;
  BasicBlockPassPrinter(const PassInfo *PI) 
    : BasicBlockPass((intptr_t)&ID), PassToPrint(PI) {}

  virtual bool runOnBasicBlock(BasicBlock &BB) {
    if (!Quiet) {
      cout << "Printing Analysis info for BasicBlock '" << BB.getName()
           << "': Pass " << PassToPrint->getPassName() << ":\n";
    }

    // Get and print pass...
    getAnalysisID<Pass>(PassToPrint).print(cout, BB.getParent()->getParent());
    return false;
  }

  virtual const char *getPassName() const { return "BasicBlockPass Printer"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredID(PassToPrint);
    AU.setPreservesAll();
  }
};

char BasicBlockPassPrinter::ID = 0;
inline void addPass(PassManager &PM, Pass *P) {
  // Add the pass to the pass manager...
  PM.add(P);

  // If we are verifying all of the intermediate steps, add the verifier...
  if (VerifyEach) PM.add(createVerifierPass());
}

void AddStandardCompilePasses(PassManager &PM) {
  PM.add(createVerifierPass());                  // Verify that input is correct

  addPass(PM, createLowerSetJmpPass());          // Lower llvm.setjmp/.longjmp

  // If the -strip-debug command line option was specified, do it.
  if (StripDebug)
    addPass(PM, createStripSymbolsPass(true));

  if (DisableOptimizations) return;

  addPass(PM, createRaiseAllocationsPass());     // call %malloc -> malloc inst
  addPass(PM, createCFGSimplificationPass());    // Clean up disgusting code
  addPass(PM, createPromoteMemoryToRegisterPass());// Kill useless allocas
  addPass(PM, createGlobalOptimizerPass());      // Optimize out global vars
  addPass(PM, createGlobalDCEPass());            // Remove unused fns and globs
  addPass(PM, createIPConstantPropagationPass());// IP Constant Propagation
  addPass(PM, createDeadArgEliminationPass());   // Dead argument elimination
  addPass(PM, createInstructionCombiningPass()); // Clean up after IPCP & DAE
  addPass(PM, createCFGSimplificationPass());    // Clean up after IPCP & DAE

  addPass(PM, createPruneEHPass());              // Remove dead EH info

  if (!DisableInline)
    addPass(PM, createFunctionInliningPass());   // Inline small functions
  addPass(PM, createArgumentPromotionPass());    // Scalarize uninlined fn args

  addPass(PM, createTailDuplicationPass());      // Simplify cfg by copying code
  addPass(PM, createInstructionCombiningPass()); // Cleanup for scalarrepl.
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
  addPass(PM, createScalarReplAggregatesPass()); // Break up aggregate allocas
  addPass(PM, createInstructionCombiningPass()); // Combine silly seq's
  addPass(PM, createCondPropagationPass());      // Propagate conditionals

  addPass(PM, createTailCallEliminationPass());  // Eliminate tail calls
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
  addPass(PM, createReassociatePass());          // Reassociate expressions
  addPass(PM, createLoopRotatePass());
  addPass(PM, createLICMPass());                 // Hoist loop invariants
  addPass(PM, createLoopUnswitchPass());         // Unswitch loops.
  // Disable for 2.1
  //addPass(PM, createLoopIndexSplitPass());       // Index split loops.
  addPass(PM, createInstructionCombiningPass()); // Clean up after LICM/reassoc
  addPass(PM, createIndVarSimplifyPass());       // Canonicalize indvars
  addPass(PM, createLoopUnrollPass());           // Unroll small loops
  addPass(PM, createInstructionCombiningPass()); // Clean up after the unroller
  addPass(PM, createGVNPass());                  // Remove redundancies
  addPass(PM, createSCCPPass());                 // Constant prop with SCCP

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  addPass(PM, createInstructionCombiningPass());
  addPass(PM, createCondPropagationPass());      // Propagate conditionals

  addPass(PM, createDeadStoreEliminationPass()); // Delete dead stores
  addPass(PM, createAggressiveDCEPass());        // SSA based 'Aggressive DCE'
  addPass(PM, createCFGSimplificationPass());    // Merge & remove BBs
  addPass(PM, createSimplifyLibCallsPass());     // Library Call Optimizations
  addPass(PM, createDeadTypeEliminationPass());  // Eliminate dead types
  addPass(PM, createConstantMergePass());        // Merge dup global constants
}

} // anonymous namespace


//===----------------------------------------------------------------------===//
// main for opt
//
int main(int argc, char **argv) {
  llvm_shutdown_obj X;  // Call llvm_shutdown() on exit.
  try {
    cl::ParseCommandLineOptions(argc, argv,
      " llvm .bc -> .bc modular optimizer and analysis printer \n");
    sys::PrintStackTraceOnErrorSignal();

    // Allocate a full target machine description only if necessary.
    // FIXME: The choice of target should be controllable on the command line.
    std::auto_ptr<TargetMachine> target;

    std::string ErrorMessage;

    // Load the input module...
    std::auto_ptr<Module> M;
    if (MemoryBuffer *Buffer
          = MemoryBuffer::getFileOrSTDIN(InputFilename, &ErrorMessage)) {
      M.reset(ParseBitcodeFile(Buffer, &ErrorMessage));
      delete Buffer;
    }
    
    if (M.get() == 0) {
      cerr << argv[0] << ": ";
      if (ErrorMessage.size())
        cerr << ErrorMessage << "\n";
      else
        cerr << "bitcode didn't read correctly.\n";
      return 1;
    }

    // Figure out what stream we are supposed to write to...
    // FIXME: cout is not binary!
    std::ostream *Out = &std::cout;  // Default to printing to stdout...
    if (OutputFilename != "-") {
      if (!Force && std::ifstream(OutputFilename.c_str())) {
        // If force is not specified, make sure not to overwrite a file!
        cerr << argv[0] << ": error opening '" << OutputFilename
             << "': file exists!\n"
             << "Use -f command line argument to force output\n";
        return 1;
      }
      std::ios::openmode io_mode = std::ios::out | std::ios::trunc |
                                   std::ios::binary;
      Out = new std::ofstream(OutputFilename.c_str(), io_mode);

      if (!Out->good()) {
        cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
        return 1;
      }

      // Make sure that the Output file gets unlinked from the disk if we get a
      // SIGINT
      sys::RemoveFileOnSignal(sys::Path(OutputFilename));
    }

    // If the output is set to be emitted to standard out, and standard out is a
    // console, print out a warning message and refuse to do it.  We don't
    // impress anyone by spewing tons of binary goo to a terminal.
    if (!Force && !NoOutput && CheckBitcodeOutputToConsole(Out,!Quiet)) {
      NoOutput = true;
    }

    // Create a PassManager to hold and optimize the collection of passes we are
    // about to build...
    //
    PassManager Passes;

    // Add an appropriate TargetData instance for this module...
    Passes.add(new TargetData(M.get()));

    // If -std-compile-opts is given, add in all the standard compilation 
    // optimizations first. This will handle -strip-debug, -disable-inline,
    // and -disable-opt as well.
    if (StandardCompileOpts)
      AddStandardCompilePasses(Passes);

    // otherwise if the -strip-debug command line option was specified, add it.
    else if (StripDebug)
      addPass(Passes, createStripSymbolsPass(true));

    // Create a new optimization pass for each one specified on the command line
    for (unsigned i = 0; i < PassList.size(); ++i) {
      const PassInfo *PassInf = PassList[i];
      Pass *P = 0;
      if (PassInf->getNormalCtor())
        P = PassInf->getNormalCtor()();
      else
        cerr << argv[0] << ": cannot create pass: "
             << PassInf->getPassName() << "\n";
      if (P) {
        addPass(Passes, P);
        
        if (AnalyzeOnly) {
          if (dynamic_cast<BasicBlockPass*>(P))
            Passes.add(new BasicBlockPassPrinter(PassInf));
          else if (dynamic_cast<LoopPass*>(P))
            Passes.add(new  LoopPassPrinter(PassInf));
          else if (dynamic_cast<FunctionPass*>(P))
            Passes.add(new FunctionPassPrinter(PassInf));
          else if (dynamic_cast<CallGraphSCCPass*>(P))
            Passes.add(new CallGraphSCCPassPrinter(PassInf));
          else
            Passes.add(new ModulePassPrinter(PassInf));
        }
      }
      
      if (PrintEachXForm)
        Passes.add(new PrintModulePass(&cerr));
    }

    // Check that the module is well formed on completion of optimization
    if (!NoVerify && !VerifyEach)
      Passes.add(createVerifierPass());

    // Write bitcode out to disk or cout as the last step...
    if (!NoOutput && !AnalyzeOnly)
      Passes.add(CreateBitcodeWriterPass(*Out));

    // Now that we have all of the passes ready, run them.
    Passes.run(*M.get());

    // Delete the ofstream.
    if (Out != &std::cout) 
      delete Out;
    return 0;

  } catch (const std::string& msg) {
    cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  llvm_shutdown();
  return 1;
}
