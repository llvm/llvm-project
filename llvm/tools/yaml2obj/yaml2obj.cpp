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

// TODO: The "right" way to tell what kind of object file a given YAML file
// corresponds to is to look at YAML "tags" (e.g. `!Foo`). Then, different
// tags (`!ELF`, `!COFF`, etc.) would be used to discriminate between them.
// Interpreting the tags is needed eventually for when writing test cases,
// so that we can e.g. have `!Archive` contain a sequence of `!ELF`, and
// just Do The Right Thing. However, interpreting these tags and acting on
// them appropriately requires some work in the YAML parser and the YAMLIO
// library.
enum YAMLObjectFormat {
  YOF_COFF,
  YOF_ELF,
  YOF_MACHO
};

cl::opt<YAMLObjectFormat> Format(
  "format",
  cl::desc("Interpret input as this type of object file"),
  cl::values(
    clEnumValN(YOF_COFF, "coff", "COFF object file format"),
    clEnumValN(YOF_ELF, "elf", "ELF object file format"),
    clEnumValN(YOF_MACHO, "macho", "Mach-O object file format"),
  clEnumValEnd));

cl::opt<unsigned>
DocNum("docnum", cl::init(1),
       cl::desc("Read specified document from input (default = 1)"));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"));

typedef int (*ConvertFuncPtr)(yaml::Input & YIn, raw_ostream &Out);

static int convertYAML(yaml::Input &YIn, raw_ostream &Out,
                       ConvertFuncPtr Convert) {
  unsigned CurDocNum = 0;
  do {
    if (++CurDocNum == DocNum)
      return Convert(YIn, Out);
  } while (YIn.nextDocument());

  errs() << "yaml2obj: Cannot find the " << DocNum
         << llvm::getOrdinalSuffix(DocNum) << " document\n";
  return 1;
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
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFileOrSTDIN(Input);
  if (!Buf)
    return 1;

  ConvertFuncPtr Convert = nullptr;
  if (Format == YOF_COFF)
    Convert = yaml2coff;
  else if (Format == YOF_ELF)
    Convert = yaml2elf;
  else if (Format == YOF_MACHO)
    Convert = yaml2macho;
  else {
    errs() << "Not yet implemented\n";
    return 1;
  }

  yaml::Input YIn(Buf.get()->getBuffer());

  int Res = convertYAML(YIn, Out->os(), Convert);
  if (Res == 0)
    Out->keep();

  return Res;
}
