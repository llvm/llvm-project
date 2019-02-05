//===- dsymutil.cpp - Debug info dumping utility for llvm -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that aims to be a dropin replacement for Darwin's
// dsymutil.
//===----------------------------------------------------------------------===//

#include "dsymutil.h"
#include "BinaryHolder.h"
#include "CFBundle.h"
#include "DebugMap.h"
#include "LinkUtils.h"
#include "MachOUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFVerifier.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/thread.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>

using namespace llvm;
using namespace llvm::cl;
using namespace llvm::dsymutil;
using namespace object;

static OptionCategory DsymCategory("Specific Options");
static opt<bool> Help("h", desc("Alias for -help"), Hidden);
static opt<bool> Version("v", desc("Alias for -version"), Hidden);

static list<std::string> InputFiles(Positional, OneOrMore,
                                    desc("<input files>"), cat(DsymCategory));

static opt<std::string>
    OutputFileOpt("o",
                  desc("Specify the output file. default: <input file>.dwarf"),
                  value_desc("filename"), cat(DsymCategory));
static alias OutputFileOptA("out", desc("Alias for -o"),
                            aliasopt(OutputFileOpt));

static opt<std::string> OsoPrependPath(
    "oso-prepend-path",
    desc("Specify a directory to prepend to the paths of object files."),
    value_desc("path"), cat(DsymCategory));

static opt<bool> Assembly(
    "S",
    desc("Output textual assembly instead of a binary dSYM companion file."),
    init(false), cat(DsymCategory), cl::Hidden);

static opt<bool> DumpStab(
    "symtab",
    desc("Dumps the symbol table found in executable or object file(s) and\n"
         "exits."),
    init(false), cat(DsymCategory));
static alias DumpStabA("s", desc("Alias for --symtab"), aliasopt(DumpStab));

static opt<bool> FlatOut("flat",
                         desc("Produce a flat dSYM file (not a bundle)."),
                         init(false), cat(DsymCategory));
static alias FlatOutA("f", desc("Alias for --flat"), aliasopt(FlatOut));

static opt<bool> Minimize(
    "minimize",
    desc("When used when creating a dSYM file with Apple accelerator tables,\n"
         "this option will suppress the emission of the .debug_inlines, \n"
         ".debug_pubnames, and .debug_pubtypes sections since dsymutil \n"
         "has better equivalents: .apple_names and .apple_types. When used in\n"
         "conjunction with --update option, this option will cause redundant\n"
         "accelerator tables to be removed."),
    init(false), cat(DsymCategory));
static alias MinimizeA("z", desc("Alias for --minimize"), aliasopt(Minimize));

static opt<bool> Update(
    "update",
    desc("Updates existing dSYM files to contain the latest accelerator\n"
         "tables and other DWARF optimizations."),
    init(false), cat(DsymCategory));
static alias UpdateA("u", desc("Alias for --update"), aliasopt(Update));

static opt<std::string> SymbolMap(
    "symbol-map",
    desc("Updates the existing dSYMs inplace using symbol map specified."),
    value_desc("bcsymbolmap"), cat(DsymCategory));

static cl::opt<AccelTableKind> AcceleratorTable(
    "accelerator", cl::desc("Output accelerator tables."),
    cl::values(clEnumValN(AccelTableKind::Default, "Default",
                          "Default for input."),
               clEnumValN(AccelTableKind::Apple, "Apple", "Apple"),
               clEnumValN(AccelTableKind::Dwarf, "Dwarf", "DWARF")),
    cl::init(AccelTableKind::Default), cat(DsymCategory));

static opt<unsigned> NumThreads(
    "num-threads",
    desc("Specifies the maximum number (n) of simultaneous threads to use\n"
         "when linking multiple architectures."),
    value_desc("n"), init(0), cat(DsymCategory));
static alias NumThreadsA("j", desc("Alias for --num-threads"),
                         aliasopt(NumThreads));

static opt<bool> Verbose("verbose", desc("Verbosity level"), init(false),
                         cat(DsymCategory));

static opt<bool>
    NoOutput("no-output",
             desc("Do the link in memory, but do not emit the result file."),
             init(false), cat(DsymCategory));

static opt<bool>
    NoTimestamp("no-swiftmodule-timestamp",
                desc("Don't check timestamp for swiftmodule files."),
                init(false), cat(DsymCategory));

static list<std::string> ArchFlags(
    "arch",
    desc("Link DWARF debug information only for specified CPU architecture\n"
         "types. This option can be specified multiple times, once for each\n"
         "desired architecture. All CPU architectures will be linked by\n"
         "default."),
    value_desc("arch"), ZeroOrMore, cat(DsymCategory));

static opt<bool>
    NoODR("no-odr",
          desc("Do not use ODR (One Definition Rule) for type uniquing."),
          init(false), cat(DsymCategory));

static opt<bool> DumpDebugMap(
    "dump-debug-map",
    desc("Parse and dump the debug map to standard output. Not DWARF link "
         "will take place."),
    init(false), cat(DsymCategory));

static opt<bool> InputIsYAMLDebugMap(
    "y", desc("Treat the input file is a YAML debug map rather than a binary."),
    init(false), cat(DsymCategory));

static opt<bool> Verify("verify", desc("Verify the linked DWARF debug info."),
                        cat(DsymCategory));

static opt<std::string>
    Toolchain("toolchain", desc("Embed toolchain information in dSYM bundle."),
              cat(DsymCategory));

static opt<bool>
    PaperTrailWarnings("papertrail",
                       desc("Embed warnings in the linked DWARF debug info."),
                       cat(DsymCategory));

static Error createPlistFile(llvm::StringRef Bin, llvm::StringRef BundleRoot) {
  if (NoOutput)
    return Error::success();

  // Create plist file to write to.
  llvm::SmallString<128> InfoPlist(BundleRoot);
  llvm::sys::path::append(InfoPlist, "Contents/Info.plist");
  std::error_code EC;
  llvm::raw_fd_ostream PL(InfoPlist, EC, llvm::sys::fs::F_Text);
  if (EC)
    return make_error<StringError>(
        "cannot create Plist: " + toString(errorCodeToError(EC)), EC);

  CFBundleInfo BI = getBundleInfo(Bin);

  if (BI.IDStr.empty()) {
    llvm::StringRef BundleID = *llvm::sys::path::rbegin(BundleRoot);
    if (llvm::sys::path::extension(BundleRoot) == ".dSYM")
      BI.IDStr = llvm::sys::path::stem(BundleID);
    else
      BI.IDStr = BundleID;
  }

  // Print out information to the plist file.
  PL << "<?xml version=\"1.0\" encoding=\"UTF-8\"\?>\n"
     << "<!DOCTYPE plist PUBLIC \"-//Apple Computer//DTD PLIST 1.0//EN\" "
     << "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
     << "<plist version=\"1.0\">\n"
     << "\t<dict>\n"
     << "\t\t<key>CFBundleDevelopmentRegion</key>\n"
     << "\t\t<string>English</string>\n"
     << "\t\t<key>CFBundleIdentifier</key>\n"
     << "\t\t<string>com.apple.xcode.dsym." << BI.IDStr << "</string>\n"
     << "\t\t<key>CFBundleInfoDictionaryVersion</key>\n"
     << "\t\t<string>6.0</string>\n"
     << "\t\t<key>CFBundlePackageType</key>\n"
     << "\t\t<string>dSYM</string>\n"
     << "\t\t<key>CFBundleSignature</key>\n"
     << "\t\t<string>\?\?\?\?</string>\n";

  if (!BI.OmitShortVersion()) {
    PL << "\t\t<key>CFBundleShortVersionString</key>\n";
    PL << "\t\t<string>";
    printHTMLEscaped(BI.ShortVersionStr, PL);
    PL << "</string>\n";
  }

  PL << "\t\t<key>CFBundleVersion</key>\n";
  PL << "\t\t<string>";
  printHTMLEscaped(BI.VersionStr, PL);
  PL << "</string>\n";

  if (!Toolchain.empty()) {
    PL << "\t\t<key>Toolchain</key>\n";
    PL << "\t\t<string>";
    printHTMLEscaped(Toolchain, PL);
    PL << "</string>\n";
  }

  PL << "\t</dict>\n"
     << "</plist>\n";

  PL.close();
  return Error::success();
}

static Error createBundleDir(llvm::StringRef BundleBase) {
  if (NoOutput)
    return Error::success();

  llvm::SmallString<128> Bundle(BundleBase);
  llvm::sys::path::append(Bundle, "Contents", "Resources", "DWARF");
  if (std::error_code EC =
          create_directories(Bundle.str(), true, llvm::sys::fs::perms::all_all))
    return make_error<StringError>(
        "cannot create bundle: " + toString(errorCodeToError(EC)), EC);

  return Error::success();
}

static bool verify(llvm::StringRef OutputFile, llvm::StringRef Arch) {
  if (OutputFile == "-") {
    WithColor::warning() << "verification skipped for " << Arch
                         << "because writing to stdout.\n";
    return true;
  }

  Expected<OwningBinary<Binary>> BinOrErr = createBinary(OutputFile);
  if (!BinOrErr) {
    WithColor::error() << OutputFile << ": " << toString(BinOrErr.takeError());
    return false;
  }

  Binary &Binary = *BinOrErr.get().getBinary();
  if (auto *Obj = dyn_cast<MachOObjectFile>(&Binary)) {
    raw_ostream &os = Verbose ? errs() : nulls();
    os << "Verifying DWARF for architecture: " << Arch << "\n";
    std::unique_ptr<DWARFContext> DICtx = DWARFContext::create(*Obj);
    DIDumpOptions DumpOpts;
    bool success = DICtx->verify(os, DumpOpts.noImplicitRecursion());
    if (!success)
      WithColor::error() << "verification failed for " << Arch << '\n';
    return success;
  }

  return false;
}

static Expected<std::string> getOutputFileName(llvm::StringRef InputFile) {
  if (OutputFileOpt == "-")
    return OutputFileOpt;

  // When updating, do in place replacement.
  if (OutputFileOpt.empty() && (Update || !SymbolMap.empty()))
    return InputFile;

  // If a flat dSYM has been requested, things are pretty simple.
  if (FlatOut) {
    if (OutputFileOpt.empty()) {
      if (InputFile == "-")
        return "a.out.dwarf";
      return (InputFile + ".dwarf").str();
    }

    return OutputFileOpt;
  }

  // We need to create/update a dSYM bundle.
  // A bundle hierarchy looks like this:
  //   <bundle name>.dSYM/
  //       Contents/
  //          Info.plist
  //          Resources/
  //             DWARF/
  //                <DWARF file(s)>
  std::string DwarfFile =
      InputFile == "-" ? llvm::StringRef("a.out") : InputFile;
  llvm::SmallString<128> BundleDir(OutputFileOpt);
  if (BundleDir.empty())
    BundleDir = DwarfFile + ".dSYM";
  if (auto E = createBundleDir(BundleDir))
    return std::move(E);
  if (auto E = createPlistFile(DwarfFile, BundleDir))
    return std::move(E);

  llvm::sys::path::append(BundleDir, "Contents", "Resources", "DWARF",
                          llvm::sys::path::filename(DwarfFile));
  return BundleDir.str();
}

/// Parses the command line options into the LinkOptions struct and performs
/// some sanity checking. Returns an error in case the latter fails.
static Expected<LinkOptions> getOptions() {
  LinkOptions Options;

  Options.Verbose = Verbose;
  Options.NoOutput = NoOutput;
  Options.NoODR = NoODR;
  Options.Minimize = Minimize;
  Options.Update = Update;
  Options.NoTimestamp = NoTimestamp;
  Options.PrependPath = OsoPrependPath;
  Options.TheAccelTableKind = AcceleratorTable;

  if (!SymbolMap.empty())
    Options.Update = true;

  if (Assembly)
    Options.FileType = OutputFileType::Assembly;

  if (Options.Update && std::find(InputFiles.begin(), InputFiles.end(), "-") !=
                            InputFiles.end()) {
    // FIXME: We cannot use stdin for an update because stdin will be
    // consumed by the BinaryHolder during the debugmap parsing, and
    // then we will want to consume it again in DwarfLinker. If we
    // used a unique BinaryHolder object that could cache multiple
    // binaries this restriction would go away.
    return make_error<StringError>(
        "standard input cannot be used as input for a dSYM update.",
        inconvertibleErrorCode());
  }

  if (NumThreads == 0)
    Options.Threads = llvm::thread::hardware_concurrency();
  if (DumpDebugMap || Verbose)
    Options.Threads = 1;

  return Options;
}

/// Return a list of input files. This function has logic for dealing with the
/// special case where we might have dSYM bundles as input. The function
/// returns an error when the directory structure doesn't match that of a dSYM
/// bundle.
static Expected<std::vector<std::string>> getInputs(bool DsymAsInput) {
  if (!DsymAsInput)
    return InputFiles;

  // If we are updating, we might get dSYM bundles as input.
  std::vector<std::string> Inputs;
  for (const auto &Input : InputFiles) {
    if (!llvm::sys::fs::is_directory(Input)) {
      Inputs.push_back(Input);
      continue;
    }

    // Make sure that we're dealing with a dSYM bundle.
    SmallString<256> BundlePath(Input);
    sys::path::append(BundlePath, "Contents", "Resources", "DWARF");
    if (!llvm::sys::fs::is_directory(BundlePath))
      return make_error<StringError>(
          Input + " is a directory, but doesn't look like a dSYM bundle.",
          inconvertibleErrorCode());

    // Create a directory iterator to iterate over all the entries in the
    // bundle.
    std::error_code EC;
    llvm::sys::fs::directory_iterator DirIt(BundlePath, EC);
    llvm::sys::fs::directory_iterator DirEnd;
    if (EC)
      return errorCodeToError(EC);

    // Add each entry to the list of inputs.
    while (DirIt != DirEnd) {
      Inputs.push_back(DirIt->path());
      DirIt.increment(EC);
      if (EC)
        return errorCodeToError(EC);
    }
  }
  return Inputs;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  void *P = (void *)(intptr_t)getOutputFileName;
  std::string SDKPath = llvm::sys::fs::getMainExecutable(argv[0], P);
  SDKPath = llvm::sys::path::parent_path(SDKPath);

  HideUnrelatedOptions({&DsymCategory, &ColorCategory});
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "manipulate archived DWARF debug symbol files.\n\n"
      "dsymutil links the DWARF debug information found in the object files\n"
      "for the executable <input file> by using debug symbols information\n"
      "contained in its symbol table.\n");

  if (Help) {
    PrintHelpMessage();
    return 0;
  }

  if (Version) {
    llvm::cl::PrintVersionMessage();
    return 0;
  }

  auto OptionsOrErr = getOptions();
  if (!OptionsOrErr) {
    WithColor::error() << toString(OptionsOrErr.takeError());
    return 1;
  }

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  auto InputsOrErr = getInputs(OptionsOrErr->Update);
  if (!InputsOrErr) {
    WithColor::error() << toString(InputsOrErr.takeError()) << '\n';
    return 1;
  }

  if (!FlatOut && OutputFileOpt == "-") {
    WithColor::error() << "cannot emit to standard output without --flat\n";
    return 1;
  }

  if (InputsOrErr->size() > 1 && FlatOut && !OutputFileOpt.empty()) {
    WithColor::error() << "cannot use -o with multiple inputs in flat mode\n";
    return 1;
  }

  if (InputFiles.size() > 1 && !SymbolMap.empty() &&
      !llvm::sys::fs::is_directory(SymbolMap)) {
    WithColor::error() << "when unobfuscating multiple files, --symbol-map "
                       << "needs to point to a directory.\n";
    return 1;
  }

  if (getenv("RC_DEBUG_OPTIONS"))
    PaperTrailWarnings = true;

  if (PaperTrailWarnings && InputIsYAMLDebugMap)
    WithColor::warning()
        << "Paper trail warnings are not supported for YAML input";

  for (const auto &Arch : ArchFlags)
    if (Arch != "*" && Arch != "all" &&
        !llvm::object::MachOObjectFile::isValidArch(Arch)) {
      WithColor::error() << "unsupported cpu architecture: '" << Arch << "'\n";
      return 1;
    }

  SymbolMapLoader SymMapLoader(SymbolMap);

  for (auto &InputFile : *InputsOrErr) {
    // Dump the symbol table for each input file and requested arch
    if (DumpStab) {
      if (!dumpStab(InputFile, ArchFlags, OsoPrependPath))
        return 1;
      continue;
    }

    auto DebugMapPtrsOrErr =
        parseDebugMap(InputFile, ArchFlags, OsoPrependPath, PaperTrailWarnings,
                      Verbose, InputIsYAMLDebugMap);

    if (auto EC = DebugMapPtrsOrErr.getError()) {
      WithColor::error() << "cannot parse the debug map for '" << InputFile
                         << "': " << EC.message() << '\n';
      return 1;
    }

    if (OptionsOrErr->Update) {
      // The debug map should be empty. Add one object file corresponding to
      // the input file.
      for (auto &Map : *DebugMapPtrsOrErr)
        Map->addDebugMapObject(InputFile,
                               llvm::sys::TimePoint<std::chrono::seconds>());
    }

    // Ensure that the debug map is not empty (anymore).
    if (DebugMapPtrsOrErr->empty()) {
      WithColor::error() << "no architecture to link\n";
      return 1;
    }

    // Shared a single binary holder for all the link steps.
    BinaryHolder BinHolder;

    NumThreads =
        std::min<unsigned>(OptionsOrErr->Threads, DebugMapPtrsOrErr->size());
    llvm::ThreadPool Threads(NumThreads);

    // If there is more than one link to execute, we need to generate
    // temporary files.
    bool NeedsTempFiles =
        !DumpDebugMap && (OutputFileOpt != "-") &&
        (DebugMapPtrsOrErr->size() != 1 || OptionsOrErr->Update);

    llvm::SmallVector<MachOUtils::ArchAndFile, 4> TempFiles;
    std::atomic_char AllOK(1);
    for (auto &Map : *DebugMapPtrsOrErr) {
      if (Verbose || DumpDebugMap)
        Map->print(llvm::outs());

      if (DumpDebugMap)
        continue;

      if (!SymbolMap.empty())
        OptionsOrErr->Translator = SymMapLoader.Load(InputFile, *Map);

      if (Map->begin() == Map->end())
        WithColor::warning()
            << "no debug symbols in executable (-arch "
            << MachOUtils::getArchName(Map->getTriple().getArchName()) << ")\n";

      // Using a std::shared_ptr rather than std::unique_ptr because move-only
      // types don't work with std::bind in the ThreadPool implementation.
      std::shared_ptr<raw_fd_ostream> OS;

      Expected<std::string> OutputFileOrErr = getOutputFileName(InputFile);
      if (!OutputFileOrErr) {
        WithColor::error() << toString(OutputFileOrErr.takeError());
        return 1;
      }

      std::string OutputFile = *OutputFileOrErr;
      if (NeedsTempFiles) {
        TempFiles.emplace_back(Map->getTriple().getArchName().str());

        auto E = TempFiles.back().createTempFile();
        if (E) {
          WithColor::error() << toString(std::move(E));
          return 1;
        }

        auto &TempFile = *(TempFiles.back().File);
        OS = std::make_shared<raw_fd_ostream>(TempFile.FD,
                                              /*shouldClose*/ false);
        OutputFile = TempFile.TmpName;
      } else {
        std::error_code EC;
        OS = std::make_shared<raw_fd_ostream>(NoOutput ? "-" : OutputFile, EC,
                                              sys::fs::F_None);
        if (EC) {
          WithColor::error() << OutputFile << ": " << EC.message();
          return 1;
        }
      }

      auto LinkLambda = [&,
                         OutputFile](std::shared_ptr<raw_fd_ostream> Stream) {
        AllOK.fetch_and(linkDwarf(*Stream, BinHolder, *Map, *OptionsOrErr));
        Stream->flush();
        if (Verify && !NoOutput)
          AllOK.fetch_and(verify(OutputFile, Map->getTriple().getArchName()));
      };

      // FIXME: The DwarfLinker can have some very deep recursion that can max
      // out the (significantly smaller) stack when using threads. We don't
      // want this limitation when we only have a single thread.
      if (NumThreads == 1)
        LinkLambda(OS);
      else
        Threads.async(LinkLambda, OS);
    }

    Threads.wait();

    if (!AllOK)
      return 1;

    if (NeedsTempFiles) {
      Expected<std::string> OutputFileOrErr = getOutputFileName(InputFile);
      if (!OutputFileOrErr) {
        WithColor::error() << toString(OutputFileOrErr.takeError());
        return 1;
      }
      if (!MachOUtils::generateUniversalBinary(TempFiles, *OutputFileOrErr,
                                               *OptionsOrErr, SDKPath))
        return 1;
    }
  }

  return 0;
}
