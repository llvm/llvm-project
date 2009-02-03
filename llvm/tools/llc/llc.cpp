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

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/FileWriters.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/RegistryParser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/System/Signals.h"
#include "llvm/Config/config.h"
#include "llvm/LinkAllVMCore.h"
#include <fstream>
#include <iostream>
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

static cl::opt<bool> Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool> Fast("fast",
      cl::desc("Generate code quickly, potentially sacrificing code quality"));

static cl::opt<std::string>
TargetTriple("mtriple", cl::desc("Override target triple for module"));

static cl::opt<const TargetMachineRegistry::entry*, false,
               RegistryParser<TargetMachine> >
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
       clEnumValN(TargetMachine::AssemblyFile, "asm",
                  "Emit an assembly ('.s') file"),
       clEnumValN(TargetMachine::ObjectFile, "obj",
                  "Emit a native object ('.o') file [experimental]"),
       clEnumValN(TargetMachine::DynamicLibrary, "dynlib",
                  "Emit a native dynamic library ('.so') file"
                  " [experimental]"),
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

static raw_ostream *GetOutputStream(const char *ProgName) {
  if (OutputFilename != "") {
    if (OutputFilename == "-")
      return &outs();

    // Specified an output filename?
    if (!Force && std::ifstream(OutputFilename.c_str())) {
      // If force is not specified, make sure not to overwrite a file!
      std::cerr << ProgName << ": error opening '" << OutputFilename
                << "': file exists!\n"
                << "Use -f command line argument to force output\n";
      return 0;
    }
    // Make sure that the Out file gets unlinked from the disk if we get a
    // SIGINT
    sys::RemoveFileOnSignal(sys::Path(OutputFilename));

    std::string error;
    raw_ostream *Out = new raw_fd_ostream(OutputFilename.c_str(), true, error);
    if (!error.empty()) {
      std::cerr << error << '\n';
      delete Out;
      return 0;
    }

    return Out;
  }

  if (InputFilename == "-") {
    OutputFilename = "-";
    return &outs();
  }

  OutputFilename = GetFileNameRoot(InputFilename);

  bool Binary = false;
  switch (FileType) {
  case TargetMachine::AssemblyFile:
    if (MArch->Name[0] == 'c') {
      if (MArch->Name[1] == 0)
        OutputFilename += ".cbe.c";
      else if (MArch->Name[1] == 'p' && MArch->Name[2] == 'p')
        OutputFilename += ".cpp";
      else
        OutputFilename += ".s";
    } else
      OutputFilename += ".s";
    break;
  case TargetMachine::ObjectFile:
    OutputFilename += ".o";
    Binary = true;
    break;
  case TargetMachine::DynamicLibrary:
    OutputFilename += LTDL_SHLIB_EXT;
    Binary = true;
    break;
  }

  if (!Force && std::ifstream(OutputFilename.c_str())) {
    // If force is not specified, make sure not to overwrite a file!
    std::cerr << ProgName << ": error opening '" << OutputFilename
                          << "': file exists!\n"
                          << "Use -f command line argument to force output\n";
    return 0;
  }

  // Make sure that the Out file gets unlinked from the disk if we get a
  // SIGINT
  sys::RemoveFileOnSignal(sys::Path(OutputFilename));

  std::string error;
  raw_ostream *Out = new raw_fd_ostream(OutputFilename.c_str(), Binary, error);
  if (!error.empty()) {
    std::cerr << error << '\n';
    delete Out;
    return 0;
  }

  return Out;
}

// main - Entry point for the llc compiler.
//
int main(int argc, char **argv) {
  llvm_shutdown_obj X;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm system compiler\n");
  sys::PrintStackTraceOnErrorSignal();

  // Load the module to be compiled...
  std::string ErrorMessage;
  std::auto_ptr<Module> M;

  std::auto_ptr<MemoryBuffer> Buffer(
                   MemoryBuffer::getFileOrSTDIN(InputFilename, &ErrorMessage));
  if (Buffer.get())
    M.reset(ParseBitcodeFile(Buffer.get(), &ErrorMessage));
  if (M.get() == 0) {
    std::cerr << argv[0] << ": bitcode didn't read correctly.\n";
    std::cerr << "Reason: " << ErrorMessage << "\n";
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

  // Figure out where we are going to send the output...
  raw_ostream *Out = GetOutputStream(argv[0]);
  if (Out == 0) return 1;

  // If this target requires addPassesToEmitWholeFile, do it now.  This is
  // used by strange things like the C backend.
  if (Target.WantsWholeFile()) {
    PassManager PM;
    PM.add(new TargetData(*Target.getTargetData()));
    if (!NoVerify)
      PM.add(createVerifierPass());

    // Ask the target to add backend passes as necessary.
    if (Target.addPassesToEmitWholeFile(PM, *Out, FileType, Fast)) {
      std::cerr << argv[0] << ": target does not support generation of this"
                << " file type!\n";
      if (Out != &outs()) delete Out;
      // And the Out file is empty and useless, so remove it now.
      sys::Path(OutputFilename).eraseFromDisk();
      return 1;
    }
    PM.run(mod);
  } else {
    // Build up all of the passes that we want to do to the module.
    ExistingModuleProvider Provider(M.release());
    FunctionPassManager Passes(&Provider);
    Passes.add(new TargetData(*Target.getTargetData()));

#ifndef NDEBUG
    if (!NoVerify)
      Passes.add(createVerifierPass());
#endif

    // Ask the target to add backend passes as necessary.
    MachineCodeEmitter *MCE = 0;

    switch (Target.addPassesToEmitFile(Passes, *Out, FileType, Fast)) {
    default:
      assert(0 && "Invalid file model!");
      return 1;
    case FileModel::Error:
      std::cerr << argv[0] << ": target does not support generation of this"
                << " file type!\n";
      if (Out != &outs()) delete Out;
      // And the Out file is empty and useless, so remove it now.
      sys::Path(OutputFilename).eraseFromDisk();
      return 1;
    case FileModel::AsmFile:
      break;
    case FileModel::MachOFile:
      MCE = AddMachOWriter(Passes, *Out, Target);
      break;
    case FileModel::ElfFile:
      MCE = AddELFWriter(Passes, *Out, Target);
      break;
    }

    if (Target.addPassesToEmitFileFinish(Passes, MCE, Fast)) {
      std::cerr << argv[0] << ": target does not support generation of this"
                << " file type!\n";
      if (Out != &outs()) delete Out;
      // And the Out file is empty and useless, so remove it now.
      sys::Path(OutputFilename).eraseFromDisk();
      return 1;
    }

    Passes.doInitialization();

    // Run our queue of passes all at once now, efficiently.
    // TODO: this could lazily stream functions out of the module.
    for (Module::iterator I = mod.begin(), E = mod.end(); I != E; ++I)
      if (!I->isDeclaration())
        Passes.run(*I);

    Passes.doFinalization();
  }

  // Delete the ostream if it's not a stdout stream
  if (Out != &outs()) delete Out;

  return 0;
}
