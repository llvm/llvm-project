//===- Main.cpp - Top-Level TableGen implementation -----------------------===//
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

#include "TGParser.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/system_error.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <cstdio>
using namespace llvm;

namespace {
  cl::opt<std::string>
  OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
                 cl::init("-"));

  cl::opt<std::string>
  DependFilename("d",
                 cl::desc("Dependency filename"),
                 cl::value_desc("filename"),
                 cl::init(""));

  cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

  cl::list<std::string>
  IncludeDirs("I", cl::desc("Directory of include files"),
              cl::value_desc("directory"), cl::Prefix);
}

/// \brief Create a dependency file for `-d` option.
///
/// This functionality is really only for the benefit of the build system.
/// It is similar to GCC's `-M*` family of options.
static int createDependencyFile(const TGParser &Parser, const char *argv0) {
  if (OutputFilename == "-") {
    errs() << argv0 << ": the option -d must be used together with -o\n";
    return 1;
  }
  std::string Error;
  tool_output_file DepOut(DependFilename.c_str(), Error);
  if (!Error.empty()) {
    errs() << argv0 << ": error opening " << DependFilename
      << ":" << Error << "\n";
    return 1;
  }
  DepOut.os() << OutputFilename << ":";
  const std::vector<std::string> &Dependencies = Parser.getDependencies();
  for (std::vector<std::string>::const_iterator I = Dependencies.begin(),
                                                E = Dependencies.end();
       I != E; ++I) {
    DepOut.os() << " " << (*I);
  }
  DepOut.os() << "\n";
  DepOut.keep();
  return 0;
}

namespace llvm {

int TableGenMain(char *argv0, TableGenMainFn *MainFn) {
  RecordKeeper Records;

  // Parse the input file.
  OwningPtr<MemoryBuffer> File;
  if (error_code ec =
        MemoryBuffer::getFileOrSTDIN(InputFilename.c_str(), File)) {
    errs() << "Could not open input file '" << InputFilename << "': "
           << ec.message() <<"\n";
    return 1;
  }
  MemoryBuffer *F = File.take();

  // Tell SrcMgr about this buffer, which is what TGParser will pick up.
  SrcMgr.AddNewSourceBuffer(F, SMLoc());

  // Record the location of the include directory so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(IncludeDirs);

  TGParser Parser(SrcMgr, Records);

  if (Parser.ParseFile())
    return 1;

  std::string Error;
  tool_output_file Out(OutputFilename.c_str(), Error);
  if (!Error.empty()) {
    errs() << argv0 << ": error opening " << OutputFilename
      << ":" << Error << "\n";
    return 1;
  }
  if (!DependFilename.empty()) {
    if (int Ret = createDependencyFile(Parser, argv0))
      return Ret;
  }

  if (MainFn(Out.os(), Records))
    return 1;

  // Declare success.
  Out.keep();
  return 0;

  return 1;
}

}
