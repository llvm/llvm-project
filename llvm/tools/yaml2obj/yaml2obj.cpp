//===- yaml2obj - Convert YAML to a binary object file --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program takes a YAML description of an object file and outputs the
// binary equivalent.
//
// This is used for writing tests that require binary files.
//
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

using namespace llvm;

static cl::opt<std::string>
  Input(cl::Positional, cl::desc("<input>"), cl::init("-"));

cl::opt<unsigned>
DocNum("docnum", cl::init(1),
       cl::desc("Read specified document from input (default = 1)"));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"));

LLVM_ATTRIBUTE_NORETURN static void error(Twine Message) {
  errs() << Message << "\n";
  exit(1);
}

static int convertYAML(yaml::Input &YIn, raw_ostream &Out) {
  unsigned CurDocNum = 0;
  do {
    if (++CurDocNum == DocNum) {
      yaml::YamlObjectFile Doc;
      YIn >> Doc;
      if (YIn.error())
        error("yaml2obj: Failed to parse YAML file!");
      if (Doc.Elf)
        return yaml2elf(*Doc.Elf, Out);
      if (Doc.Coff)
        return yaml2coff(*Doc.Coff, Out);
      if (Doc.MachO || Doc.FatMachO)
        return yaml2macho(Doc, Out);
      if (Doc.Wasm)
        return yaml2wasm(*Doc.Wasm, Out);
      error("yaml2obj: Unknown document type!");
    }
  } while (YIn.nextDocument());

  error("yaml2obj: Cannot find the " + utostr(DocNum) +
        llvm::getOrdinalSuffix(DocNum) + " document");
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::error_code EC;
  std::unique_ptr<tool_output_file> Out(
      new tool_output_file(OutputFilename, EC, sys::fs::F_None));
  if (EC)
    error("yaml2obj: Error opening '" + OutputFilename + "': " + EC.message());

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFileOrSTDIN(Input);
  if (!Buf)
    return 1;

  yaml::Input YIn(Buf.get()->getBuffer());

  int Res = convertYAML(YIn, Out->os());
  if (Res == 0)
    Out->keep();

  return Res;
}
