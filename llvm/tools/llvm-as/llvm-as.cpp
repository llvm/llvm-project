//===--- llvm-as.cpp - The low-level LLVM assembler -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This utility may be invoked in the following manner:
//   llvm-as --help         - Output information about command line switches
//   llvm-as [options]      - Read LLVM asm from stdin, write bitcode to stdout
//   llvm-as [options] x.ll - Read LLVM asm from the x.ll file, write bitcode
//                            to the x.bc file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
#include <fstream>
#include <iostream>
#include <memory>
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input .llvm file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
Force("f", cl::desc("Overwrite output files"));

static cl::opt<bool>
DisableOutput("disable-output", cl::desc("Disable output"), cl::init(false));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as parsed"), cl::Hidden);

static cl::opt<bool>
DisableVerify("disable-verify", cl::Hidden,
              cl::desc("Do not run verifier on input LLVM (dangerous!)"));

int main(int argc, char **argv) {
  llvm_shutdown_obj X;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm .ll -> .bc assembler\n");
  sys::PrintStackTraceOnErrorSignal();

  int exitCode = 0;
  std::ostream *Out = 0;
  try {
    // Parse the file now...
    ParseError Err;
    std::auto_ptr<Module> M(ParseAssemblyFile(InputFilename, Err));
    if (M.get() == 0) {
      Err.PrintError(argv[0], errs());
      return 1;
    }

    if (!DisableVerify) {
      std::string Err;
      if (verifyModule(*M.get(), ReturnStatusAction, &Err)) {
        cerr << argv[0]
             << ": assembly parsed, but does not verify as correct!\n";
        cerr << Err;
        return 1;
      } 
    }

    if (DumpAsm) cerr << "Here's the assembly:\n" << *M.get();

    if (OutputFilename != "") {   // Specified an output filename?
      if (OutputFilename != "-") {  // Not stdout?
        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          cerr << argv[0] << ": error opening '" << OutputFilename
               << "': file exists!\n"
               << "Use -f command line argument to force output\n";
          return 1;
        }
        Out = new std::ofstream(OutputFilename.c_str(), std::ios::out |
                                std::ios::trunc | std::ios::binary);
      } else {                      // Specified stdout
        // FIXME: cout is not binary!
        Out = &std::cout;
      }
    } else {
      if (InputFilename == "-") {
        OutputFilename = "-";
        Out = &std::cout;
      } else {
        std::string IFN = InputFilename;
        int Len = IFN.length();
        if (IFN[Len-3] == '.' && IFN[Len-2] == 'l' && IFN[Len-1] == 'l') {
          // Source ends in .ll
          OutputFilename = std::string(IFN.begin(), IFN.end()-3);
        } else {
          OutputFilename = IFN;   // Append a .bc to it
        }
        OutputFilename += ".bc";

        if (!Force && std::ifstream(OutputFilename.c_str())) {
          // If force is not specified, make sure not to overwrite a file!
          cerr << argv[0] << ": error opening '" << OutputFilename
               << "': file exists!\n"
               << "Use -f command line argument to force output\n";
          return 1;
        }

        Out = new std::ofstream(OutputFilename.c_str(), std::ios::out |
                                std::ios::trunc | std::ios::binary);
        // Make sure that the Out file gets unlinked from the disk if we get a
        // SIGINT
        sys::RemoveFileOnSignal(sys::Path(OutputFilename));
      }
    }

    if (!Out->good()) {
      cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
      return 1;
    }

    if (!DisableOutput)
      if (Force || !CheckBitcodeOutputToConsole(Out,true))
        WriteBitcodeToFile(M.get(), *Out);
  } catch (const std::string& msg) {
    cerr << argv[0] << ": " << msg << "\n";
    exitCode = 1;
  } catch (...) {
    cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
    exitCode = 1;
  }

  if (Out != &std::cout) delete Out;
  return exitCode;
}

