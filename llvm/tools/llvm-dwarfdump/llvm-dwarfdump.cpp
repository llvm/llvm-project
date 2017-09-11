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

static cl::list<std::string>
InputFilenames(cl::Positional, cl::desc("<input object files or .dSYM bundles>"),
               cl::ZeroOrMore);

static cl::opt<bool> DumpAll("all", cl::desc("Dump all debug info sections"));
static cl::alias DumpAllAlias("a", cl::desc("Alias for --all"),
                              cl::aliasopt(DumpAll));

static uint64_t DumpType = DIDT_Null;
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME)                \
  static cl::opt<bool> Dump##ENUM_NAME(                                        \
      CMDLINE_NAME, cl::desc("Dump the " ELF_NAME " section"));
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION

static cl::opt<bool>
    SummarizeTypes("summarize-types",
                   cl::desc("Abbreviate the description of type unit entries"));
static cl::opt<bool> Verify("verify", cl::desc("Verify the DWARF debug info"));
static cl::opt<bool> Quiet("quiet",
                           cl::desc("Use with -verify to not emit to STDOUT."));
static cl::opt<bool> Verbose("verbose",
                             cl::desc("Print more low-level encoding details"));
static cl::alias VerboseAlias("v", cl::desc("Alias for -verbose"),
                              cl::aliasopt(Verbose));

static void error(StringRef Filename, std::error_code EC) {
  if (!EC)
    return;
  errs() << Filename << ": " << EC.message() << "\n";
  exit(1);
}

static void DumpObjectFile(ObjectFile &Obj, Twine Filename) {
  std::unique_ptr<DWARFContext> DICtx = DWARFContext::create(Obj);
  logAllUnhandledErrors(DICtx->loadRegisterInfo(Obj), errs(),
                        Filename.str() + ": ");

  outs() << Filename.str() << ":\tfile format " << Obj.getFileFormatName()
         << "\n\n";


  // Dump the complete DWARF structure.
  DIDumpOptions DumpOpts;
  DumpOpts.DumpType = DumpType;
  DumpOpts.SummarizeTypes = SummarizeTypes;
  DumpOpts.Brief = !Verbose;
  DICtx->dump(outs(), DumpOpts);
}

static void DumpInput(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  error(Filename, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buff = std::move(BuffOrErr.get());

  Expected<std::unique_ptr<Binary>> BinOrErr =
      object::createBinary(Buff->getMemBufferRef());
  if (!BinOrErr)
    error(Filename, errorToErrorCode(BinOrErr.takeError()));

  if (auto *Obj = dyn_cast<ObjectFile>(BinOrErr->get()))
    DumpObjectFile(*Obj, Filename);
  else if (auto *Fat = dyn_cast<MachOUniversalBinary>(BinOrErr->get()))
    for (auto &ObjForArch : Fat->objects()) {
      auto MachOOrErr = ObjForArch.getAsObjectFile();
      error(Filename, errorToErrorCode(MachOOrErr.takeError()));
      DumpObjectFile(**MachOOrErr,
                     Filename + " (" + ObjForArch.getArchFlagName() + ")");
    }
}

static bool VerifyObjectFile(ObjectFile &Obj, Twine Filename) {
  std::unique_ptr<DIContext> DICtx = DWARFContext::create(Obj);

  // Verify the DWARF and exit with non-zero exit status if verification
  // fails.
  raw_ostream &stream = Quiet ? nulls() : outs();
  stream << "Verifying " << Filename.str() << ":\tfile format "
  << Obj.getFileFormatName() << "\n";
  bool Result = DICtx->verify(stream, DumpType);
  if (Result)
    stream << "No errors.\n";
  else
    stream << "Errors detected.\n";
  return Result;
}

static bool VerifyInput(StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
  MemoryBuffer::getFileOrSTDIN(Filename);
  error(Filename, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buff = std::move(BuffOrErr.get());

  Expected<std::unique_ptr<Binary>> BinOrErr =
  object::createBinary(Buff->getMemBufferRef());
  if (!BinOrErr)
    error(Filename, errorToErrorCode(BinOrErr.takeError()));

  bool Result = true;
  if (auto *Obj = dyn_cast<ObjectFile>(BinOrErr->get()))
    Result = VerifyObjectFile(*Obj, Filename);
  else if (auto *Fat = dyn_cast<MachOUniversalBinary>(BinOrErr->get()))
    for (auto &ObjForArch : Fat->objects()) {
      auto MachOOrErr = ObjForArch.getAsObjectFile();
      error(Filename, errorToErrorCode(MachOOrErr.takeError()));
      if (!VerifyObjectFile(**MachOOrErr, Filename + " (" + ObjForArch.getArchFlagName() + ")"))
        Result = false;
    }
  return Result;
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

  cl::ParseCommandLineOptions(argc, argv, "llvm dwarf dumper\n");

  // Defaults to dumping all sections, unless brief mode is specified in which
  // case only the .debug_info section in dumped.
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME)                \
  if (Dump##ENUM_NAME)                                                         \
    DumpType |= DIDT_##ENUM_NAME;
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
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
    if (!std::all_of(Objects.begin(), Objects.end(), VerifyInput))
      exit(1);
  } else {
    std::for_each(Objects.begin(), Objects.end(), DumpInput);
  }

  return EXIT_SUCCESS;
}
