//===- yaml2obj - Convert YAML to a binary object file --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
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
      if (Doc.Minidump)
        return yaml2minidump(*Doc.Minidump, Out);
      if (Doc.Wasm)
        return yaml2wasm(*Doc.Wasm, Out);
      error("yaml2obj: Unknown document type!");
    }
  } while (YIn.nextDocument());

  error("yaml2obj: Cannot find the " + Twine(DocNum) +
        llvm::getOrdinalSuffix(DocNum) + " document");
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> Out(
      new ToolOutputFile(OutputFilename, EC, sys::fs::F_None));
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

  Out->os().flush();
  return Res;
}
