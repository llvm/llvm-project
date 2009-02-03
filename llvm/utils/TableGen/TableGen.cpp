//===- TableGen.cpp - Top-Level TableGen implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TableGen is a tool which can be used to build up a description of something,
// then invoke one or more "tablegen backends" to emit information about the
// description in some predefined format.  In practice, this is used by the LLVM
// code generators to automate generation of a code generator through a
// high-level description of the target.
//
//===----------------------------------------------------------------------===//

#include "Record.h"
#include "TGParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Streams.h"
#include "llvm/System/Signals.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "CallingConvEmitter.h"
#include "CodeEmitterGen.h"
#include "RegisterInfoEmitter.h"
#include "InstrInfoEmitter.h"
#include "InstrEnumEmitter.h"
#include "AsmWriterEmitter.h"
#include "DAGISelEmitter.h"
#include "FastISelEmitter.h"
#include "SubtargetEmitter.h"
#include "IntrinsicEmitter.h"
#include "LLVMCConfigurationEmitter.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <ios>
using namespace llvm;

enum ActionType {
  PrintRecords,
  GenEmitter,
  GenRegisterEnums, GenRegister, GenRegisterHeader,
  GenInstrEnums, GenInstrs, GenAsmWriter,
  GenCallingConv,
  GenDAGISel,
  GenFastISel,
  GenSubtarget,
  GenIntrinsic,
  GenLLVMCConf,
  PrintEnums
};

namespace {
  cl::opt<ActionType>
  Action(cl::desc("Action to perform:"),
         cl::values(clEnumValN(PrintRecords, "print-records",
                               "Print all records to stdout (default)"),
                    clEnumValN(GenEmitter, "gen-emitter",
                               "Generate machine code emitter"),
                    clEnumValN(GenRegisterEnums, "gen-register-enums",
                               "Generate enum values for registers"),
                    clEnumValN(GenRegister, "gen-register-desc",
                               "Generate a register info description"),
                    clEnumValN(GenRegisterHeader, "gen-register-desc-header",
                               "Generate a register info description header"),
                    clEnumValN(GenInstrEnums, "gen-instr-enums",
                               "Generate enum values for instructions"),
                    clEnumValN(GenInstrs, "gen-instr-desc",
                               "Generate instruction descriptions"),
                    clEnumValN(GenCallingConv, "gen-callingconv",
                               "Generate calling convention descriptions"),
                    clEnumValN(GenAsmWriter, "gen-asm-writer",
                               "Generate assembly writer"),
                    clEnumValN(GenDAGISel, "gen-dag-isel",
                               "Generate a DAG instruction selector"),
                    clEnumValN(GenFastISel, "gen-fast-isel",
                               "Generate a \"fast\" instruction selector"),
                    clEnumValN(GenSubtarget, "gen-subtarget",
                               "Generate subtarget enumerations"),
                    clEnumValN(GenIntrinsic, "gen-intrinsic",
                               "Generate intrinsic information"),
                    clEnumValN(GenLLVMCConf, "gen-llvmc",
                               "Generate LLVMC configuration library"),
                    clEnumValN(PrintEnums, "print-enums",
                               "Print enum values for a class"),
                    clEnumValEnd));

  cl::opt<std::string>
  Class("class", cl::desc("Print Enum list for this class"),
        cl::value_desc("class name"));

  cl::opt<std::string>
  OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
                 cl::init("-"));

  cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

  cl::list<std::string>
  IncludeDirs("I", cl::desc("Directory of include files"),
              cl::value_desc("directory"), cl::Prefix);
}

RecordKeeper llvm::Records;

/// ParseFile - this function begins the parsing of the specified tablegen
/// file.
static bool ParseFile(const std::string &Filename,
                      const std::vector<std::string> &IncludeDirs) {
  std::string ErrorStr;
  MemoryBuffer *F = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), &ErrorStr);
  if (F == 0) {
    cerr << "Could not open input file '" + Filename + "': " << ErrorStr <<"\n";
    return true;
  }

  TGParser Parser(F);

  // Record the location of the include directory so that the lexer can find
  // it later.
  Parser.setIncludeDirs(IncludeDirs);

  return Parser.ParseFile();
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  // Parse the input file.
  if (ParseFile(InputFilename, IncludeDirs))
    return 1;

  std::ostream *Out = cout.stream();
  if (OutputFilename != "-") {
    Out = new std::ofstream(OutputFilename.c_str());

    if (!Out->good()) {
      cerr << argv[0] << ": error opening " << OutputFilename << "!\n";
      return 1;
    }

    // Make sure the file gets removed if *gasp* tablegen crashes...
    sys::RemoveFileOnSignal(sys::Path(OutputFilename));
  }

  try {
    switch (Action) {
    case PrintRecords:
      *Out << Records;           // No argument, dump all contents
      break;
    case GenEmitter:
      CodeEmitterGen(Records).run(*Out);
      break;

    case GenRegisterEnums:
      RegisterInfoEmitter(Records).runEnums(*Out);
      break;
    case GenRegister:
      RegisterInfoEmitter(Records).run(*Out);
      break;
    case GenRegisterHeader:
      RegisterInfoEmitter(Records).runHeader(*Out);
      break;

    case GenInstrEnums:
      InstrEnumEmitter(Records).run(*Out);
      break;
    case GenInstrs:
      InstrInfoEmitter(Records).run(*Out);
      break;
    case GenCallingConv:
      CallingConvEmitter(Records).run(*Out);
      break;
    case GenAsmWriter:
      AsmWriterEmitter(Records).run(*Out);
      break;

    case GenDAGISel:
      DAGISelEmitter(Records).run(*Out);
      break;
    case GenFastISel:
      FastISelEmitter(Records).run(*Out);
      break;
    case GenSubtarget:
      SubtargetEmitter(Records).run(*Out);
      break;
    case GenIntrinsic:
      IntrinsicEmitter(Records).run(*Out);
      break;
    case GenLLVMCConf:
      LLVMCConfigurationEmitter(Records).run(*Out);
      break;
    case PrintEnums:
    {
      std::vector<Record*> Recs = Records.getAllDerivedDefinitions(Class);
      for (unsigned i = 0, e = Recs.size(); i != e; ++i)
        *Out << Recs[i]->getName() << ", ";
      *Out << "\n";
      break;
    }
    default:
      assert(1 && "Invalid Action");
      return 1;
    }
  } catch (const std::string &Error) {
    cerr << argv[0] << ": " << Error << "\n";
    if (Out != cout.stream()) {
      delete Out;                             // Close the file
      std::remove(OutputFilename.c_str());    // Remove the file, it's broken
    }
    return 1;
  } catch (const char *Error) {
    cerr << argv[0] << ": " << Error << "\n";
    if (Out != cout.stream()) {
      delete Out;                             // Close the file
      std::remove(OutputFilename.c_str());    // Remove the file, it's broken
    }
    return 1;
  } catch (...) {
    cerr << argv[0] << ": Unknown unexpected exception occurred.\n";
    if (Out != cout.stream()) {
      delete Out;                             // Close the file
      std::remove(OutputFilename.c_str());    // Remove the file, it's broken
    }
    return 2;
  }

  if (Out != cout.stream()) {
    delete Out;                               // Close the file
  }
  return 0;
}
