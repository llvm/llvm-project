//===------ utils/obj2yaml.cpp - obj2yaml conversion tool -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/Minidump.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

using namespace llvm::opt;
#define OPTTABLE_CODE
#include "Opts.inc"

class Obj2YamlOptTable : public opt::OptTable {
public:
  Obj2YamlOptTable() : OptTable(optionTables()) { setDashDashParsing(true); }
};
} // namespace

static Error dumpObject(const ObjectFile &Obj, raw_ostream &OS) {
  if (Obj.isCOFF())
    return errorCodeToError(coff2yaml(OS, cast<COFFObjectFile>(Obj)));

  if (Obj.isXCOFF())
    return xcoff2yaml(OS, cast<XCOFFObjectFile>(Obj));

  if (Obj.isELF())
    return elf2yaml(OS, Obj);

  if (Obj.isGOFF())
    return goff2yaml(OS, cast<GOFFObjectFile>(Obj));

  if (Obj.isWasm())
    return errorCodeToError(wasm2yaml(OS, cast<WasmObjectFile>(Obj)));

  llvm_unreachable("unexpected object file format");
}

static Error dumpInput(StringRef File, unsigned RawSegment, raw_ostream &OS) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(File, /*IsText=*/false,
                                   /*RequiresNullTerminator=*/false);
  if (std::error_code EC = FileOrErr.getError())
    return errorCodeToError(EC);
  std::unique_ptr<MemoryBuffer> &Buffer = FileOrErr.get();
  MemoryBufferRef MemBuf = Buffer->getMemBufferRef();
  switch (identify_magic(MemBuf.getBuffer())) {
  case file_magic::archive:
    return archive2yaml(OS, MemBuf);
  case file_magic::dxcontainer_object:
    return dxcontainer2yaml(OS, MemBuf);
  case file_magic::offload_binary:
    return offload2yaml(OS, MemBuf);
  default:
    break;
  }

  Expected<std::unique_ptr<Binary>> BinOrErr =
      createBinary(MemBuf, /*Context=*/nullptr);
  if (!BinOrErr)
    return BinOrErr.takeError();

  Binary &Binary = *BinOrErr->get();
  // Universal MachO is not a subclass of ObjectFile, so it needs to be handled
  // here with the other binary types.
  if (Binary.isMachO() || Binary.isMachOUniversalBinary())
    return macho2yaml(OS, Binary, RawSegment);
  if (ObjectFile *Obj = dyn_cast<ObjectFile>(&Binary))
    return dumpObject(*Obj, OS);
  if (MinidumpFile *Minidump = dyn_cast<MinidumpFile>(&Binary))
    return minidump2yaml(OS, *Minidump);

  return Error::success();
}

static void reportError(StringRef Input, Error Err) {
  if (Input == "-")
    Input = "<stdin>";
  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  logAllUnhandledErrors(std::move(Err), OS);
  errs() << "Error reading file: " << Input << ": " << ErrMsg;
  errs().flush();
}

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);
  auto Fatal = [](const Twine &Msg) {
    WithColor::error(errs(), "obj2yaml") << Msg << '\n';
    exit(1);
  };
  BumpPtrAllocator A;
  StringSaver Saver(A);
  Obj2YamlOptTable Tbl;
  opt::InputArgList Args = Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver, Fatal);
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(outs(), "obj2yaml [options] <input file>",
                  "Dump a YAML description from an object file");
    return 0;
  }
  if (Args.hasArg(OPT_version)) {
    cl::PrintVersionMessage();
    return 0;
  }

  std::vector<std::string> Inputs = Args.getAllArgValues(OPT_INPUT);
  if (Inputs.size() > 1)
    Fatal("too many input files");
  StringRef InputFilename =
      Inputs.empty() ? StringRef("-") : StringRef(Inputs[0]);
  StringRef OutputFilename = Args.getLastArgValue(OPT_o, "-");
  unsigned RawSegment = RawSegments::none;
  for (StringRef S : Args.getAllArgValues(OPT_raw_segment_EQ)) {
    if (S == "data")
      RawSegment |= RawSegments::data;
    else if (S == "linkedit")
      RawSegment |= RawSegments::linkedit;
    else
      Fatal("unknown segment '" + S + "' for --raw-segment");
  }

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> Out(
      new ToolOutputFile(OutputFilename, EC, sys::fs::OF_Text));
  if (EC)
    Fatal("failed to open '" + OutputFilename + "': " + EC.message());
  if (Error Err = dumpInput(InputFilename, RawSegment, Out->os())) {
    reportError(InputFilename, std::move(Err));
    return 1;
  }
  Out->keep();

  return 0;
}
