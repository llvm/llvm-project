//===-- llc.cpp - Implement the LLVM Native Code Generator ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the llc code generator driver. It provides a convenient
// command-line interface for generating native assembly-language code
// or C code, given LLVM bytecode.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bytecode/Reader.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/System/Signals.h"
#include "llvm/Config/config.h"
#include <fstream>
#include <iostream>
#include <memory>

using namespace llvm;

// General options for llc.  Other pass-specific options are specified
// within the corresponding llc passes, and target-specific options
// and back-end code generation options are specified with the target machine.
//
static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bytecode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));

static cl::opt<bool> Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool> Fast("fast", 
      cl::desc("Generate code quickly, potentially sacrificing code quality"));

static cl::opt<std::string>
TargetTriple("mtriple", cl::desc("Override target triple for module"));

static cl::opt<const TargetMachineRegistry::Entry*, false, TargetNameParser>
MArch("march", cl::desc("Architecture to generate code for:"));

static cl::opt<std::string>
MCPU("mcpu", 
  cl::desc("Target a specific cpu type (-mcpu=help for details)"),
  cl::value_desc("cpu-name"),
  cl::init(""));

static cl::list<std::string>
MAttrs("mattr", 
  cl::CommaSeparated,
  cl::desc("Target specific attributes (-mattr=help for details)"),
  cl::value_desc("a1,+a2,-a3,..."));

cl::opt<TargetMachine::CodeGenFileType>
FileType("filetype", cl::init(TargetMachine::AssemblyFile),
  cl::desc("Choose a file type (not all types are supported by all targets):"),
  cl::values(
       clEnumValN(TargetMachine::AssemblyFile,    "asm",
                  "  Emit an assembly ('.s') file"),
       clEnumValN(TargetMachine::ObjectFile,    "obj",
                  "  Emit a native object ('.o') file [experimental]"),
       clEnumValN(TargetMachine::DynamicLibrary, "dynlib",
                  "  Emit a native dynamic library ('.so') file"),
       clEnumValEnd));

cl::opt<bool> NoVerify("disable-verify", cl::Hidden,
                       cl::desc("Do not verify input module"));


// GetFileNameRoot - Helper function to get the basename of a filename.
static inline std::string
GetFileNameRoot(const std::string &InputFilename) {
  std::string IFN = InputFilename;
  std::string outputFilename;
  int Len = IFN.length();
  if ((Len > 2) &&
      IFN[Len-3] == '.' && IFN[Len-2] == 'b' && IFN[Len-1] == 'c') {
    outputFilename = std::string(IFN.begin(), IFN.end()-3); // s/.bc/.s/
  } else {
    outputFilename = IFN;
  }
  return outputFilename;
}


// main - Entry point for the llc compiler.
//
int main(int argc, char **argv) {
  try {
    cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
    sys::PrintStackTraceOnErrorSignal();

    // Load the module to be compiled...
    std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
    if (M.get() == 0) {
      std::cerr << argv[0] << ": bytecode didn't read correctly.\n";
      return 1;
    }
    Module &mod = *M.get();

    // If we are supposed to override the target triple, do so now.
    if (!TargetTriple.empty())
      mod.setTargetTriple(TargetTriple);
    
    // Allocate target machine.  First, check whether the user has
    // explicitly specified an architecture to compile for.
    if (MArch == 0) {
      std::string Err;
      MArch = TargetMachineRegistry::getClosestStaticTargetForModule(mod, Err);
      if (MArch == 0) {
        std::cerr << argv[0] << ": error auto-selecting target for module '"
                  << Err << "'.  Please use the -march option to explicitly "
                  << "pick a target.\n";
        return 1;
      }
    }

    // Package up features to be passed to target/subtarget
    std::string FeaturesStr;
    if (MCPU.size() || MAttrs.size()) {
      SubtargetFeatures Features;
      Features.setCPU(MCPU);
      for (unsigned i = 0; i != MAttrs.size(); ++i)
        Features.AddFeature(MAttrs[i]);
      FeaturesStr = Features.getString();
    }

    std::auto_ptr<TargetMachine> target(MArch->CtorFn(mod, FeaturesStr));
    assert(target.get() && "Could not allocate target machine!");
    TargetMachine &Target = *target.get();
    const TargetData &TD = Target.getTargetData();

    // Build up all of the passes that we want to do to the module...
    PassManager Passes;
    Passes.add(new TargetData(TD));

#ifndef NDEBUG
    if(!NoVerify)
      Passes.add(createVerifierPass());
#endif

    // Figure out where we are going to send the output...
    std::ostream *Out = 0;
    if (OutputFilename != "") {
      if (OutputFilename != "-") {
        // Specified an output filename?
        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          std::cerr << argv[0] << ": error opening '" << OutputFilename
                    << "': file exists!\n"
                    << "Use -f command line argument to force output\n";
          return 1;
        }
        Out = new std::ofstream(OutputFilename.c_str());

        // Make sure that the Out file gets unlinked from the disk if we get a
        // SIGINT
        sys::RemoveFileOnSignal(sys::Path(OutputFilename));
      } else {
        Out = &std::cout;
      }
    } else {
      if (InputFilename == "-") {
        OutputFilename = "-";
        Out = &std::cout;
      } else {
        OutputFilename = GetFileNameRoot(InputFilename);

        switch (FileType) {
        case TargetMachine::AssemblyFile:
          if (MArch->Name[0] != 'c' || MArch->Name[1] != 0)  // not CBE
            OutputFilename += ".s";
          else
            OutputFilename += ".cbe.c";
          break;
        case TargetMachine::ObjectFile:
          OutputFilename += ".o";
          break;
        case TargetMachine::DynamicLibrary:
          OutputFilename += LTDL_SHLIB_EXT;
          break;
        }

        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          std::cerr << argv[0] << ": error opening '" << OutputFilename
                    << "': file exists!\n"
                    << "Use -f command line argument to force output\n";
          return 1;
        }

        Out = new std::ofstream(OutputFilename.c_str());
        if (!Out->good()) {
          std::cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
          delete Out;
          return 1;
        }

        // Make sure that the Out file gets unlinked from the disk if we get a
        // SIGINT
        sys::RemoveFileOnSignal(sys::Path(OutputFilename));
      }
    }

    if (FileType != TargetMachine::AssemblyFile)
      std::cerr << "WARNING: only -filetype=asm is currently supported.\n";
    
    // Ask the target to add backend passes as necessary.
    if (Target.addPassesToEmitFile(Passes, *Out, FileType, Fast)) {
      std::cerr << argv[0] << ": target '" << Target.getName()
                << "' does not support generation of this file type!\n";
      if (Out != &std::cout) delete Out;
      // And the Out file is empty and useless, so remove it now.
      sys::Path(OutputFilename).eraseFromDisk();
      return 1;
    } else {
      // Run our queue of passes all at once now, efficiently.
      Passes.run(*M.get());
    }

    // Delete the ostream if it's not a stdout stream
    if (Out != &std::cout) delete Out;

    return 0;
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    std::cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  return 1;
}
