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
#include "llvm/Target/TargetMachineImpls.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "Support/CommandLine.h"
#include "Support/Signals.h"
#include <memory>
#include <fstream>

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

enum ArchName { noarch, X86, SparcV9, CBackend };

static cl::opt<ArchName>
Arch("march", cl::desc("Architecture to generate assembly for:"), cl::Prefix,
     cl::values(clEnumValN(X86,      "x86",     "  IA-32 (Pentium and above)"),
                clEnumValN(SparcV9,  "sparcv9", "  SPARC V9"),
                clEnumValN(CBackend, "c",       "  C backend"),
		0),
     cl::init(noarch));

// GetFileNameRoot - Helper function to get the basename of a filename...
static inline std::string
GetFileNameRoot(const std::string &InputFilename)
{
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
  cl::ParseCommandLineOptions(argc, argv, " llvm system compiler\n");
  PrintStackTraceOnErrorSignal();

  // Load the module to be compiled...
  std::auto_ptr<Module> M(ParseBytecodeFile(InputFilename));
  if (M.get() == 0) {
    std::cerr << argv[0] << ": bytecode didn't read correctly.\n";
    return 1;
  }
  Module &mod = *M.get();

  // Allocate target machine.  First, check whether the user has
  // explicitly specified an architecture to compile for.
  TargetMachine* (*TargetMachineAllocator)(const Module&,
                                           IntrinsicLowering *) = 0;
  switch (Arch) {
  case CBackend:
    TargetMachineAllocator = allocateCTargetMachine;
    break;
  case X86:
    TargetMachineAllocator = allocateX86TargetMachine;
    break;
  case SparcV9:
    TargetMachineAllocator = allocateSparcV9TargetMachine;
    break;
  default:
    // Decide what the default target machine should be, by looking at
    // the module. This heuristic (ILP32, LE -> IA32; LP64, BE ->
    // SPARCV9) is kind of gross, but it will work until we have more
    // sophisticated target information to work from.
    if (mod.getEndianness()  == Module::LittleEndian &&
        mod.getPointerSize() == Module::Pointer32) { 
      TargetMachineAllocator = allocateX86TargetMachine;
    } else if (mod.getEndianness()  == Module::BigEndian &&
               mod.getPointerSize() == Module::Pointer64) { 
      TargetMachineAllocator = allocateSparcV9TargetMachine;
    } else {
      // If the module is target independent, favor a target which matches the
      // current build system.
#if defined(i386) || defined(__i386__) || defined(__x86__)
      TargetMachineAllocator = allocateX86TargetMachine;
#elif defined(sparc) || defined(__sparc__) || defined(__sparcv9)
      TargetMachineAllocator = allocateSparcV9TargetMachine;
#else
      std::cerr << argv[0] << ": module does not specify a target to use.  "
                << "You must use the -march option.  If no native target is "
                << "available, use -march=c to emit C code.\n";
      return 1;
#endif
    } 
    break;
  }
  std::auto_ptr<TargetMachine> target(TargetMachineAllocator(mod, 0));
  assert(target.get() && "Could not allocate target machine!");
  TargetMachine &Target = *target.get();
  const TargetData &TD = Target.getTargetData();

  // Build up all of the passes that we want to do to the module...
  PassManager Passes;

  Passes.add(new TargetData("llc", TD.isLittleEndian(), TD.getPointerSize(),
                            TD.getPointerAlignment(), TD.getDoubleAlignment()));

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
      RemoveFileOnSignal(OutputFilename);
    } else {
      Out = &std::cout;
    }
  } else {
    if (InputFilename == "-") {
      OutputFilename = "-";
      Out = &std::cout;
    } else {
      OutputFilename = GetFileNameRoot(InputFilename); 

      if (Arch != CBackend)
        OutputFilename += ".s";
      else
        OutputFilename += ".cbe.c";
      
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
      RemoveFileOnSignal(OutputFilename);
    }
  }

  // Ask the target to add backend passes as necessary
  if (Target.addPassesToEmitAssembly(Passes, *Out)) {
    std::cerr << argv[0] << ": target '" << Target.getName()
              << "' does not support static compilation!\n";
    if (Out != &std::cout) delete Out;
    // And the Out file is empty and useless, so remove it now.
    std::remove(OutputFilename.c_str());
    return 1;
  } else {
    // Run our queue of passes all at once now, efficiently.
    Passes.run(*M.get());
  }

  // Delete the ostream if it's not a stdout stream
  if (Out != &std::cout) delete Out;

  return 0;
}
