//===-- llvm-dwarfdump.cpp - Debug info dumping utility for llvm ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like "dwarfdump".
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/RelocVisitor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <system_error>

using namespace llvm;
using namespace object;

namespace {
using namespace llvm::cl;

OptionCategory DwarfDumpCategory("Specific Options");
static opt<bool> Help("h", desc("Alias for -help"), Hidden,
                      cat(DwarfDumpCategory));
static list<std::string>
    InputFilenames(Positional, desc("<input object files or .dSYM bundles>"),
                   ZeroOrMore, cat(DwarfDumpCategory));

cl::OptionCategory
    SectionCategory("Section-specific Dump Options",
                    "These control which sections are dumped.");
static opt<bool> DumpAll("all", desc("Dump all debug info sections"),
                         cat(SectionCategory));
static alias DumpAllAlias("a", desc("Alias for -all"), aliasopt(DumpAll));

static unsigned DumpType = DIDT_Null;
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME)                \
  static opt<bool> Dump##ENUM_NAME(CMDLINE_NAME,                               \
                                   desc("Dump the " ELF_NAME " section"),      \
                                   cat(SectionCategory));
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
static opt<bool> DumpUUID("uuid", desc("Show the UUID for each architecture"),
                          cat(DwarfDumpCategory));
static alias DumpUUIDAlias("u", desc("Alias for -uuid"), aliasopt(DumpUUID));

static opt<bool>
    SummarizeTypes("summarize-types",
                   desc("Abbreviate the description of type unit entries"));
static opt<bool> Verify("verify", desc("Verify the DWARF debug info"),
                        cat(DwarfDumpCategory));
static opt<bool> Quiet("quiet", desc("Use with -verify to not emit to STDOUT."),
                       cat(DwarfDumpCategory));
static opt<bool> Verbose("verbose",
                         desc("Print more low-level encoding details"),
                         cat(DwarfDumpCategory));
static alias VerboseAlias("v", desc("Alias for -verbose"), aliasopt(Verbose),
                          cat(DwarfDumpCategory));
} // namespace

static void error(StringRef Filename, std::error_code EC) {
  if (!EC)
    return;
  errs() << Filename << ": " << EC.message() << "\n";
  exit(1);
}

static DIDumpOptions getDumpOpts() {
  DIDumpOptions DumpOpts;
  DumpOpts.DumpType = DumpType;
  DumpOpts.SummarizeTypes = SummarizeTypes;
  DumpOpts.Verbose = Verbose;
  return DumpOpts;
}

static bool dumpObjectFile(ObjectFile &Obj, Twine Filename) {
  std::unique_ptr<DWARFContext> DICtx = DWARFContext::create(Obj);
  logAllUnhandledErrors(DICtx->loadRegisterInfo(Obj), errs(),
                        Filename.str() + ": ");
  // The UUID dump already contains all the same information.
  if (!(DumpType & DIDT_UUID) || DumpType == DIDT_All)
    outs() << Filename << ":\tfile format " << Obj.getFileFormatName() << '\n';

  // Dump the complete DWARF structure.
  DICtx->dump(outs(), getDumpOpts());
  return true;
}

static bool verifyObjectFile(ObjectFile &Obj, Twine Filename) {
  std::unique_ptr<DIContext> DICtx = DWARFContext::create(Obj);

  // Verify the DWARF and exit with non-zero exit status if verification
  // fails.
  raw_ostream &stream = Quiet ? nulls() : outs();
  stream << "Verifying " << Filename.str() << ":\tfile format "
  << Obj.getFileFormatName() << "\n";
  bool Result = DICtx->verify(stream, DumpType, getDumpOpts());
  if (Result)
    stream << "No errors.\n";
  else
    stream << "Errors detected.\n";
  return Result;
}

static bool handleBuffer(StringRef Filename, MemoryBufferRef Buffer,
                         std::function<bool(ObjectFile &, Twine)> HandleObj) {
  Expected<std::unique_ptr<Binary>> BinOrErr = object::createBinary(Buffer);
  if (!BinOrErr)
    error(Filename, errorToErrorCode(BinOrErr.takeError()));

  bool Result = true;
  if (auto *Obj = dyn_cast<ObjectFile>(BinOrErr->get()))
    Result = HandleObj(*Obj, Filename);
  else if (auto *Fat = dyn_cast<MachOUniversalBinary>(BinOrErr->get()))
    for (auto &ObjForArch : Fat->objects()) {
      auto MachOOrErr = ObjForArch.getAsObjectFile();
      error(Filename, errorToErrorCode(MachOOrErr.takeError()));
      if (!HandleObj(**MachOOrErr,
                     Filename + " (" + ObjForArch.getArchFlagName() + ")"))
        Result = false;
    }
  return Result;
}

static bool handleFile(StringRef Filename,
                       std::function<bool(ObjectFile &, Twine)> HandleObj) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
  MemoryBuffer::getFileOrSTDIN(Filename);
  error(Filename, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BuffOrErr.get());
  return handleBuffer(Filename, *Buffer, HandleObj);
}

/// If the input path is a .dSYM bundle (as created by the dsymutil tool),
/// replace it with individual entries for each of the object files inside the
/// bundle otherwise return the input path.
static std::vector<std::string> expandBundle(const std::string &InputPath) {
  std::vector<std::string> BundlePaths;
  SmallString<256> BundlePath(InputPath);
  // Manually open up the bundle to avoid introducing additional dependencies.
  if (sys::fs::is_directory(BundlePath) &&
      sys::path::extension(BundlePath) == ".dSYM") {
    std::error_code EC;
    sys::path::append(BundlePath, "Contents", "Resources", "DWARF");
    for (sys::fs::directory_iterator Dir(BundlePath, EC), DirEnd;
         Dir != DirEnd && !EC; Dir.increment(EC)) {
      const std::string &Path = Dir->path();
      sys::fs::file_status Status;
      EC = sys::fs::status(Path, Status);
      error(Path, EC);
      switch (Status.type()) {
      case sys::fs::file_type::regular_file:
      case sys::fs::file_type::symlink_file:
      case sys::fs::file_type::type_unknown:
        BundlePaths.push_back(Path);
        break;
      default: /*ignore*/;
      }
    }
    error(BundlePath, EC);
  }
  if (!BundlePaths.size())
    BundlePaths.push_back(InputPath);
  return BundlePaths;
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();

  HideUnrelatedOptions({&DwarfDumpCategory, &SectionCategory});
  cl::ParseCommandLineOptions(
      argc, argv,
      "pretty-print DWARF debug information in object files"
      " and debug info archives.\n");

  if (Help) {
    PrintHelpMessage(/*Hidden =*/false, /*Categorized =*/true);
    return 0;
  }

  // Defaults to dumping all sections, unless brief mode is specified in which
  // case only the .debug_info section in dumped.
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME)                \
  if (Dump##ENUM_NAME)                                                         \
    DumpType |= DIDT_##ENUM_NAME;
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
  if (DumpUUID)
    DumpType |= DIDT_UUID;
  if (DumpAll)
    DumpType = DIDT_All;
  if (DumpType == DIDT_Null) {
    if (Verbose)
      DumpType = DIDT_All;
    else
      DumpType = DIDT_DebugInfo;
  }

  // Defaults to a.out if no filenames specified.
  if (InputFilenames.size() == 0)
    InputFilenames.push_back("a.out");

  // Expand any .dSYM bundles to the individual object files contained therein.
  std::vector<std::string> Objects;
  for (const auto &F : InputFilenames) {
    auto Objs = expandBundle(F);
    Objects.insert(Objects.end(), Objs.begin(), Objs.end());
  }

  if (Verify) {
    // If we encountered errors during verify, exit with a non-zero exit status.
    if (!std::all_of(Objects.begin(), Objects.end(), [](std::string Object) {
          return handleFile(Object, verifyObjectFile);
        }))
      exit(1);
  } else {
    std::for_each(Objects.begin(), Objects.end(), [](std::string Object) {
      handleFile(Object, dumpObjectFile);
    });
  }

  return EXIT_SUCCESS;
}
