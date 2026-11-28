//===- llvm-cvpack.cpp - Pack the CodeView debug info of COFF objects -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Moves the CodeView debug info that COFF objects share to a CodeView pack,
// restores the objects, and prints packs. See llvm/docs/CodeViewPacks.md.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CVPack/Materialize.h"
#include "llvm/DebugInfo/CVPack/PackFile.h"
#include "llvm/DebugInfo/CVPack/Producer.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/TypeDumpVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeTableCollection.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;

static cl::OptionCategory CVPackCategory("llvm-cvpack options");

static cl::SubCommand
    PackCommand("pack", "Pack objects against the base pack of the pack "
                        "directories, which is written when there is none");
static cl::SubCommand
    RebaseCommand("rebase", "Pack objects against a new base pack, written "
                            "from what the objects share");
static cl::SubCommand
    MaterializeCommand("materialize",
                       "Restore the original bytes of a packed object");
static cl::SubCommand DumpCommand("dump", "Print the records of packs");

static cl::list<std::string> PackPaths(
    "pack-path",
    cl::desc("Directory to find packs in, after the directory of the object "
             "that names a pack. A new pack is written to the first "
             "(default: the output directory)"),
    cl::value_desc("dir"), cl::sub(PackCommand), cl::sub(RebaseCommand),
    cl::sub(MaterializeCommand), cl::cat(CVPackCategory));

static cl::list<std::string> InputObjects(cl::Positional,
                                          cl::desc("<object>..."),
                                          cl::OneOrMore, cl::sub(PackCommand),
                                          cl::sub(RebaseCommand),
                                          cl::cat(CVPackCategory));
static cl::opt<std::string>
    OutputDirectory("out-dir",
                    cl::desc("Directory that receives the packed objects, "
                             "cvpacks.json, which lists the pack of each, and "
                             "a new pack unless --pack-path is given"),
                    cl::value_desc("dir"), cl::Required, cl::sub(PackCommand),
                    cl::sub(RebaseCommand), cl::cat(CVPackCategory));
static cl::opt<std::string>
    Base("base",
         cl::desc("Pack against this pack, given by its content id, and "
                  "never rebase"),
         cl::value_desc("id"), cl::sub(PackCommand), cl::cat(CVPackCategory));
static cl::opt<std::string> AutoRebase(
    "auto-rebase",
    cl::desc("Rebase when <missing>% of the base would be packed but is not "
             "in it, or less than <used>% of it is used (default 25,50); "
             "'no' never rebases"),
    cl::value_desc("missing,used"), cl::sub(PackCommand),
    cl::cat(CVPackCategory));
static cl::opt<bool>
    Prune("prune",
          cl::desc("Delete the packs that no packed object needs, in the "
                   "directory that receives new packs"),
          cl::sub(PackCommand), cl::sub(RebaseCommand),
          cl::cat(CVPackCategory));
static cl::opt<bool> Verbose("verbose",
                             cl::desc("Print the phases of the run, a "
                                      "summary, and why objects were left "
                                      "alone"),
                             cl::sub(PackCommand), cl::sub(RebaseCommand),
                             cl::cat(CVPackCategory));
static cl::opt<unsigned>
    Threads("threads",
            cl::desc("Number of threads (default: all hardware threads)"),
            cl::value_desc("n"), cl::init(0), cl::sub(PackCommand),
            cl::sub(RebaseCommand), cl::cat(CVPackCategory));

static cl::opt<std::string>
    PackedObject(cl::Positional, cl::desc("<packed object>"), cl::Required,
                 cl::sub(MaterializeCommand), cl::cat(CVPackCategory));
static cl::opt<std::string>
    OutputFile("o", cl::desc("The restored object ('-' for stdout)"),
               cl::value_desc("file"), cl::Required,
               cl::sub(MaterializeCommand), cl::cat(CVPackCategory));

static cl::list<std::string> InputPacks(cl::Positional, cl::desc("<pack>..."),
                                        cl::OneOrMore, cl::sub(DumpCommand),
                                        cl::cat(CVPackCategory));

static StringRef ToolName;

[[noreturn]] static void fatal(const Twine &Message) {
  WithColor::error(errs(), ToolName) << Message << '\n';
  exit(1);
}

static std::optional<PackId> parsePackId(StringRef Hex) {
  std::string Bytes;
  PackId Id;
  if (Hex.size() != 2 * Id.size() || !tryGetFromHex(Hex, Bytes))
    return std::nullopt;
  llvm::copy(Bytes, Id.begin());
  return Id;
}

static Error pack(bool Rebase) {
  PackOptions Options;
  Options.PackPaths.assign(PackPaths.begin(), PackPaths.end());
  Options.StagingDirectory = OutputDirectory;
  Options.Rebase = Rebase;
  Options.Prune = Prune;
  // By default the packs are next to the objects packed against them, where
  // a linker finds them without an option.
  if (Options.PackPaths.empty())
    Options.PackPaths.push_back(OutputDirectory);
  if (!Base.empty()) {
    Options.Base = parsePackId(Base);
    if (!Options.Base)
      fatal("--base: expected a 64-digit content id, got " + Base);
  }
  if (AutoRebase.getNumOccurrences()) {
    StringRef Value = AutoRebase;
    auto [Missing, Used] = Value.split(',');
    if (Value.equals_insensitive("no"))
      Options.AutoRebase = false;
    else if (Missing.getAsInteger(10, Options.MaxMissingPercent) ||
             Used.getAsInteger(10, Options.MinUsedPercent) ||
             Options.MinUsedPercent > 100)
      fatal("--auto-rebase: expected <missing>,<used> percentages or 'no', "
            "got " +
            Value);
  }
  if (Threads)
    parallel::strategy = hardware_concurrency(Threads);
  Options.Warn = [](const Twine &Message) {
    WithColor::warning(errs(), ToolName) << Message << '\n';
  };
  if (Verbose)
    Options.Log = [](const Twine &Message) { outs() << Message << '\n'; };

  SmallVector<StringRef, 0> Inputs(InputObjects.begin(), InputObjects.end());
  return packObjects(Options, Inputs);
}

static Error materialize() {
  auto Input = MemoryBuffer::getFile(PackedObject, /*IsText=*/false,
                                     /*RequiresNullTerminator=*/false);
  if (!Input)
    return createFileError(PackedObject, Input.getError());
  // The pack is next to the object, or in a pack directory.
  SmallVector<StringRef, 4> Directories(PackPaths.begin(), PackPaths.end());
  CVPackResolver Resolver(Directories);
  auto Original =
      materializeObject(arrayRefFromStringRef((*Input)->getBuffer()), Resolver,
                        sys::path::parent_path(PackedObject.getValue()));
  if (!Original)
    return createFileError(PackedObject, Original.takeError());
  // The input may be the output.
  Input->reset();
  return writeToOutput(OutputFile, [&](raw_ostream &OS) {
    OS << toStringRef(*Original);
    return Error::success();
  });
}

/// The CPU of a pack's symbols, from the machine of its objects.
static CPUType cpuTypeForMachine(uint16_t Machine) {
  switch (Machine) {
  case COFF::IMAGE_FILE_MACHINE_I386:
    return CPUType::Pentium3;
  case COFF::IMAGE_FILE_MACHINE_ARMNT:
    return CPUType::ARMNT;
  case COFF::IMAGE_FILE_MACHINE_ARM64:
  case COFF::IMAGE_FILE_MACHINE_ARM64EC:
  case COFF::IMAGE_FILE_MACHINE_ARM64X:
    return CPUType::ARM64;
  default:
    return CPUType::X64;
  }
}

static Error dump(StringRef Path) {
  auto Buffer = MemoryBuffer::getFile(Path, /*IsText=*/false,
                                      /*RequiresNullTerminator=*/false);
  if (!Buffer)
    return createFileError(Path, Buffer.getError());
  auto Pack = readPackFile(arrayRefFromStringRef((*Buffer)->getBuffer()));
  if (!Pack)
    return createFileError(Path, Pack.takeError());

  std::vector<ArrayRef<uint8_t>> TypeRecords;
  for (CVType Type : Pack->Types)
    TypeRecords.push_back(Type.RecordData);
  TypeTableCollection Types(TypeRecords);
  ScopedPrinter Printer(outs());
  Printer.startLine() << "CVPack: " << Path << '\n';
  Printer.printString("ContentId", toHex(Pack->ContentId, /*LowerCase=*/true));
  Printer.printHex("Machine", Pack->Machine);
  Printer.printNumber("Types", Pack->Types.size());
  Printer.printNumber("Symbols", Pack->Symbols.size());
  Printer.printNumber("Files", Pack->Files.size());
  Printer.printNumber("Functions", Pack->Functions.size());
  TypeDumpVisitor TypeDumper(Types, &Printer, false);
  if (Error E = visitTypeStream(Types, TypeDumper))
    return createFileError(Path, std::move(E));

  CVSymbolDumper SymbolDumper(Printer, Types, CodeViewContainer::ObjectFile,
                              std::unique_ptr<SymbolDumpDelegate>(),
                              cpuTypeForMachine(Pack->Machine), false);
  for (CVSymbol Symbol : Pack->Symbols)
    if (Error E = SymbolDumper.dump(Symbol))
      return createFileError(Path, std::move(E));
  Printer.flush();
  return Error::success();
}

static Error run() {
  if (PackCommand)
    return pack(/*Rebase=*/false);
  if (RebaseCommand)
    return pack(/*Rebase=*/true);
  if (MaterializeCommand)
    return materialize();
  for (const std::string &Path : InputPacks)
    if (Error E = dump(Path))
      return E;
  return Error::success();
}

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  ToolName = Argv[0];
  cl::HideUnrelatedOptions({&CVPackCategory, &getColorCategory()}, PackCommand);
  cl::HideUnrelatedOptions({&CVPackCategory, &getColorCategory()},
                           RebaseCommand);
  cl::HideUnrelatedOptions({&CVPackCategory, &getColorCategory()},
                           MaterializeCommand);
  cl::HideUnrelatedOptions({&CVPackCategory, &getColorCategory()}, DumpCommand);
  cl::ParseCommandLineOptions(
      Argc, Argv,
      "llvm-cvpack: share the CodeView debug info of COFF objects\n");

  if (!PackCommand && !RebaseCommand && !MaterializeCommand && !DumpCommand) {
    cl::PrintHelpMessage();
    return 1;
  }
  Error E = run();
  if (E) {
    WithColor::error(errs(), ToolName) << toString(std::move(E)) << '\n';
    return 1;
  }
  return 0;
}
