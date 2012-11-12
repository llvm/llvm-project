//===- FileUpdate.cpp - Conditionally update a file -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FileUpdate is a utility for conditionally updating a file from its input
// based on whether the input differs from the output. It is used to avoid
// unnecessary modifications in a build system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

static cl::opt<bool>
Quiet("quiet", cl::desc("Don't print unnecessary status information"),
      cl::init(false));

static cl::opt<std::string>
InputFilename("input-file", cl::desc("Input file (defaults to stdin)"),
              cl::init("-"), cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename(cl::Positional, cl::desc("<output-file>"), cl::Required);

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  if (OutputFilename == "-") {
    errs() << argv[0] << ": error: Can't update standard output\n";
    return 1;
  }

  // Get the input data.
  OwningPtr<MemoryBuffer> In;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFilename.c_str(), In)) {
    errs() << argv[0] << ": error: Unable to get input '"
           << InputFilename << "': " << ec.message() << '\n';
    return 1;
  }

  // Get the output data.
  OwningPtr<MemoryBuffer> Out;
  MemoryBuffer::getFile(OutputFilename.c_str(), Out);

  // If the output exists and the contents match, we are done.
  if (Out && In->getBufferSize() == Out->getBufferSize() &&
      memcmp(In->getBufferStart(), Out->getBufferStart(),
             Out->getBufferSize()) == 0) {
    if (!Quiet)
      errs() << argv[0] << ": Not updating '" << OutputFilename
             << "', contents match input.\n";
    return 0;
  }

  // Otherwise, overwrite the output.
  if (!Quiet)
    errs() << argv[0] << ": Updating '" << OutputFilename
           << "', contents changed.\n";
  std::string ErrorStr;
  tool_output_file OutStream(OutputFilename.c_str(), ErrorStr,
                             raw_fd_ostream::F_Binary);
  if (!ErrorStr.empty()) {
    errs() << argv[0] << ": Unable to write output '"
           << OutputFilename << "': " << ErrorStr << '\n';
    return 1;
  }

  OutStream.os().write(In->getBufferStart(), In->getBufferSize());

  // Declare success.
  OutStream.keep();

  return 0;
}
