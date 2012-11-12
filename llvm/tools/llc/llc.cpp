//===-- llc.cpp - Implement the LLVM Native Code Generator ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the llc code generator driver. It provides a convenient
// command-line interface for generating native assembly-language code
// or C code, given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "llvm/LLVMContext.h"
#include "llvm/DataLayout.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Support/IRReader.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
using namespace llvm;

// General options for llc.  Other pass-specific options are specified
// within the corresponding llc passes, and target-specific options
// and back-end code generation options are specified with the target machine.
//
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));

// Determine optimization level.
static cl::opt<char>
OptLevel("O",
         cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                  "(default = '-O2')"),
         cl::Prefix,
         cl::ZeroOrMore,
         cl::init(' '));

static cl::opt<std::string>
TargetTriple("mtriple", cl::desc("Override target triple for module"));

cl::opt<bool> NoVerify("disable-verify", cl::Hidden,
                       cl::desc("Do not verify input module"));

cl::opt<bool>
DisableSimplifyLibCalls("disable-simplify-libcalls",
                        cl::desc("Disable simplify-libcalls"),
                        cl::init(false));

// GetFileNameRoot - Helper function to get the basename of a filename.
static inline std::string
GetFileNameRoot(const std::string &InputFilename) {
  std::string IFN = InputFilename;
  std::string outputFilename;
  int Len = IFN.length();
  if ((Len > 2) &&
      IFN[Len-3] == '.' &&
      ((IFN[Len-2] == 'b' && IFN[Len-1] == 'c') ||
       (IFN[Len-2] == 'l' && IFN[Len-1] == 'l'))) {
    outputFilename = std::string(IFN.begin(), IFN.end()-3); // s/.bc/.s/
  } else {
    outputFilename = IFN;
  }
  return outputFilename;
}

static tool_output_file *GetOutputStream(const char *TargetName,
                                         Triple::OSType OS,
                                         const char *ProgName) {
  // If we don't yet have an output filename, make one.
  if (OutputFilename.empty()) {
    if (InputFilename == "-")
      OutputFilename = "-";
    else {
      OutputFilename = GetFileNameRoot(InputFilename);

      switch (FileType) {
      case TargetMachine::CGFT_AssemblyFile:
        if (TargetName[0] == 'c') {
          if (TargetName[1] == 0)
            OutputFilename += ".cbe.c";
          else if (TargetName[1] == 'p' && TargetName[2] == 'p')
            OutputFilename += ".cpp";
          else
            OutputFilename += ".s";
        } else
          OutputFilename += ".s";
        break;
      case TargetMachine::CGFT_ObjectFile:
        if (OS == Triple::Win32)
          OutputFilename += ".obj";
        else
          OutputFilename += ".o";
        break;
      case TargetMachine::CGFT_Null:
        OutputFilename += ".null";
        break;
      }
    }
  }

  // Decide if we need "binary" output.
  bool Binary = false;
  switch (FileType) {
  case TargetMachine::CGFT_AssemblyFile:
    break;
  case TargetMachine::CGFT_ObjectFile:
  case TargetMachine::CGFT_Null:
    Binary = true;
    break;
  }

  // Open the file.
  std::string error;
  unsigned OpenFlags = 0;
  if (Binary) OpenFlags |= raw_fd_ostream::F_Binary;
  tool_output_file *FDOut = new tool_output_file(OutputFilename.c_str(), error,
                                                 OpenFlags);
  if (!error.empty()) {
    errs() << error << '\n';
    delete FDOut;
    return 0;
  }

  return FDOut;
}

// main - Entry point for the llc compiler.
//
int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets first, so that --version shows registered targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Initialize codegen and IR passes used by llc so that the -print-after,
  // -print-before, and -stop-after options work.
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeUnreachableBlockElimPass(*Registry);

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "llvm system compiler\n");

  // Load the module to be compiled...
  SMDiagnostic Err;
  std::auto_ptr<Module> M;
  Module *mod = 0;
  Triple TheTriple;

  bool SkipModule = MCPU == "help" ||
                    (!MAttrs.empty() && MAttrs.front() == "help");

  // If user just wants to list available options, skip module loading
  if (!SkipModule) {
    M.reset(ParseIRFile(InputFilename, Err, Context));
    mod = M.get();
    if (mod == 0) {
      Err.print(argv[0], errs());
      return 1;
    }

    // If we are supposed to override the target triple, do so now.
    if (!TargetTriple.empty())
      mod->setTargetTriple(Triple::normalize(TargetTriple));
    TheTriple = Triple(mod->getTargetTriple());
  } else {
    TheTriple = Triple(Triple::normalize(TargetTriple));
  }

  if (TheTriple.getTriple().empty())
    TheTriple.setTriple(sys::getDefaultTargetTriple());

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(MArch, TheTriple,
                                                         Error);
  if (!TheTarget) {
    errs() << argv[0] << ": " << Error;
    return 1;
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MAttrs.size()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  CodeGenOpt::Level OLvl = CodeGenOpt::Default;
  switch (OptLevel) {
  default:
    errs() << argv[0] << ": invalid optimization level.\n";
    return 1;
  case ' ': break;
  case '0': OLvl = CodeGenOpt::None; break;
  case '1': OLvl = CodeGenOpt::Less; break;
  case '2': OLvl = CodeGenOpt::Default; break;
  case '3': OLvl = CodeGenOpt::Aggressive; break;
  }

  TargetOptions Options;
  Options.LessPreciseFPMADOption = EnableFPMAD;
  Options.NoFramePointerElim = DisableFPElim;
  Options.NoFramePointerElimNonLeaf = DisableFPElimNonLeaf;
  Options.AllowFPOpFusion = FuseFPOps;
  Options.UnsafeFPMath = EnableUnsafeFPMath;
  Options.NoInfsFPMath = EnableNoInfsFPMath;
  Options.NoNaNsFPMath = EnableNoNaNsFPMath;
  Options.HonorSignDependentRoundingFPMathOption =
      EnableHonorSignDependentRoundingFPMath;
  Options.UseSoftFloat = GenerateSoftFloatCalls;
  if (FloatABIForCalls != FloatABI::Default)
    Options.FloatABIType = FloatABIForCalls;
  Options.NoZerosInBSS = DontPlaceZerosInBSS;
  Options.GuaranteedTailCallOpt = EnableGuaranteedTailCallOpt;
  Options.DisableTailCalls = DisableTailCalls;
  Options.StackAlignmentOverride = OverrideStackAlignment;
  Options.RealignStack = EnableRealignStack;
  Options.TrapFuncName = TrapFuncName;
  Options.PositionIndependentExecutable = EnablePIE;
  Options.EnableSegmentedStacks = SegmentedStacks;
  Options.UseInitArray = UseInitArray;
  Options.SSPBufferSize = SSPBufferSize;

  std::auto_ptr<TargetMachine>
    target(TheTarget->createTargetMachine(TheTriple.getTriple(),
                                          MCPU, FeaturesStr, Options,
                                          RelocModel, CMModel, OLvl));
  assert(target.get() && "Could not allocate target machine!");
  assert(mod && "Should have exited after outputting help!");
  TargetMachine &Target = *target.get();

  if (DisableDotLoc)
    Target.setMCUseLoc(false);

  if (DisableCFI)
    Target.setMCUseCFI(false);

  if (EnableDwarfDirectory)
    Target.setMCUseDwarfDirectory(true);

  if (GenerateSoftFloatCalls)
    FloatABIForCalls = FloatABI::Soft;

  // Disable .loc support for older OS X versions.
  if (TheTriple.isMacOSX() &&
      TheTriple.isMacOSXVersionLT(10, 6))
    Target.setMCUseLoc(false);

  // Figure out where we are going to send the output.
  OwningPtr<tool_output_file> Out
    (GetOutputStream(TheTarget->getName(), TheTriple.getOS(), argv[0]));
  if (!Out) return 1;

  // Build up all of the passes that we want to do to the module.
  PassManager PM;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfo *TLI = new TargetLibraryInfo(TheTriple);
  if (DisableSimplifyLibCalls)
    TLI->disableAllFunctions();
  PM.add(TLI);

  if (target.get()) {
    PM.add(new TargetTransformInfo(target->getScalarTargetTransformInfo(),
                                   target->getVectorTargetTransformInfo()));
  }

  // Add the target data from the target machine, if it exists, or the module.
  if (const DataLayout *TD = Target.getDataLayout())
    PM.add(new DataLayout(*TD));
  else
    PM.add(new DataLayout(mod));

  // Override default to generate verbose assembly.
  Target.setAsmVerbosityDefault(true);

  if (RelaxAll) {
    if (FileType != TargetMachine::CGFT_ObjectFile)
      errs() << argv[0]
             << ": warning: ignoring -mc-relax-all because filetype != obj";
    else
      Target.setMCRelaxAll(true);
  }

  {
    formatted_raw_ostream FOS(Out->os());

    AnalysisID StartAfterID = 0;
    AnalysisID StopAfterID = 0;
    const PassRegistry *PR = PassRegistry::getPassRegistry();
    if (!StartAfter.empty()) {
      const PassInfo *PI = PR->getPassInfo(StartAfter);
      if (!PI) {
        errs() << argv[0] << ": start-after pass is not registered.\n";
        return 1;
      }
      StartAfterID = PI->getTypeInfo();
    }
    if (!StopAfter.empty()) {
      const PassInfo *PI = PR->getPassInfo(StopAfter);
      if (!PI) {
        errs() << argv[0] << ": stop-after pass is not registered.\n";
        return 1;
      }
      StopAfterID = PI->getTypeInfo();
    }

    // Ask the target to add backend passes as necessary.
    if (Target.addPassesToEmitFile(PM, FOS, FileType, NoVerify,
                                   StartAfterID, StopAfterID)) {
      errs() << argv[0] << ": target does not support generation of this"
             << " file type!\n";
      return 1;
    }

    // Before executing passes, print the final values of the LLVM options.
    cl::PrintOptionValues();

    PM.run(*mod);
  }

  // Declare success.
  Out->keep();

  return 0;
}
