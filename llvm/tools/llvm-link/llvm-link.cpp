//===- llvm-link.cpp - Low-level LLVM linker ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-link a.bc b.bc c.bc -o x.bc
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include <cstdlib>
#include <string>
#include <system_error>
#include <vector>

#include <memory>
#include <utility>
using namespace llvm;

using namespace llvm::opt;

namespace {
enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "LinkOpts.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "LinkOpts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "LinkOpts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "LinkOpts.inc"
#undef OPTION
};

class LinkOptTable : public opt::GenericOptTable {
public:
  LinkOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {
    setGroupedShortOptions(true);
  }
};

struct LinkerConfig {
  std::vector<std::string> InputFilenames;
  std::vector<std::string> OverridingInputs;
  std::vector<std::string> Imports;
  std::string SummaryIndex;
  std::string OutputFilename = "-";

  bool Internalize = false;
  bool DisableDITypeMap = false;
  bool OnlyNeeded = false;
  bool Force = false;
  bool DisableLazyLoad = false;
  bool OutputAssembly = false;
  bool Verbose = false;
  bool DumpAsm = false;
  bool SuppressWarnings = false;
  bool NoVerify = false;
  bool IgnoreNonBitcode = false;
};

LinkerConfig Config;
} // namespace

static ExitOnError ExitOnErr;

static Expected<LinkerConfig> parseArgs(int Argc, char **Argv) {
  LinkOptTable Tbl;
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  SmallVector<const char *, 16> ArgvStorage(Argv, Argv + Argc);
  cl::ExpandResponseFiles(Saver,
                          Triple(sys::getProcessTriple()).isOSWindows()
                              ? cl::TokenizeWindowsCommandLine
                              : cl::TokenizeGNUCommandLine,
                          ArgvStorage);

  unsigned MissingIndex, MissingCount;
  opt::InputArgList Args =
      Tbl.ParseArgs(ArgvStorage, MissingIndex, MissingCount);
  StringRef ToolName = ArgvStorage[0];

  if (MissingCount)
    return createStringError(errc::invalid_argument,
                             "argument to '%s' is missing",
                             Args.getArgString(MissingIndex));

  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(
        outs(),
        (Twine(ToolName) + " [options] <input bitcode files>").str().c_str(),
        "Low-level LLVM linker");
    std::exit(0);
  }

  if (Args.hasArg(OPT_version)) {
    outs() << ToolName << '\n';
    cl::PrintVersionMessage();
    std::exit(0);
  }

  for (auto *A : Args.filtered(OPT_UNKNOWN))
    return createStringError(errc::invalid_argument, "unknown argument '%s'",
                             A->getAsString(Args).c_str());

  LinkerConfig ParsedConfig;
  for (auto *A : Args.filtered(OPT_INPUT))
    ParsedConfig.InputFilenames.push_back(A->getValue());
  if (ParsedConfig.InputFilenames.empty())
    return createStringError(errc::invalid_argument,
                             "no input files were specified");

  for (auto *A : Args.filtered(OPT_override))
    ParsedConfig.OverridingInputs.push_back(A->getValue());
  for (auto *A : Args.filtered(OPT_import))
    ParsedConfig.Imports.push_back(A->getValue());

  if (auto *A = Args.getLastArg(OPT_summary_index))
    ParsedConfig.SummaryIndex = A->getValue();
  if (auto *A = Args.getLastArg(OPT_output, OPT_o))
    ParsedConfig.OutputFilename = A->getValue();

  ParsedConfig.Internalize = Args.hasArg(OPT_internalize);
  ParsedConfig.DisableDITypeMap = Args.hasArg(OPT_disable_debug_info_type_map);
  ParsedConfig.OnlyNeeded = Args.hasArg(OPT_only_needed);
  ParsedConfig.Force = Args.hasArg(OPT_f);
  ParsedConfig.DisableLazyLoad = Args.hasArg(OPT_disable_lazy_loading);
  ParsedConfig.OutputAssembly = Args.hasArg(OPT_S);
  ParsedConfig.Verbose = Args.hasArg(OPT_v);
  ParsedConfig.DumpAsm = Args.hasArg(OPT_d);
  ParsedConfig.SuppressWarnings = Args.hasArg(OPT_suppress_warnings);
  ParsedConfig.NoVerify = Args.hasArg(OPT_disable_verify);
  ParsedConfig.IgnoreNonBitcode = Args.hasArg(OPT_ignore_non_bitcode);

  return ParsedConfig;
}

// Read the specified bitcode file in and return it. This routine searches the
// link path for the specified file to try to find it...
//
static std::unique_ptr<Module> loadFile(const char *argv0,
                                        std::unique_ptr<MemoryBuffer> Buffer,
                                        LLVMContext &Context,
                                        bool MaterializeMetadata = true) {
  SMDiagnostic Err;
  if (Config.Verbose)
    errs() << "Loading '" << Buffer->getBufferIdentifier() << "'\n";
  std::unique_ptr<Module> Result;
  if (Config.DisableLazyLoad)
    Result = parseIR(*Buffer, Err, Context);
  else
    Result =
        getLazyIRModule(std::move(Buffer), Err, Context, !MaterializeMetadata);

  if (!Result) {
    Err.print(argv0, errs());
    return nullptr;
  }

  if (MaterializeMetadata) {
    ExitOnErr(Result->materializeMetadata());
    UpgradeDebugInfo(*Result);
  }

  return Result;
}

static std::unique_ptr<Module> loadArFile(const char *Argv0,
                                          std::unique_ptr<MemoryBuffer> Buffer,
                                          LLVMContext &Context) {
  std::unique_ptr<Module> Result(new Module("ArchiveModule", Context));
  StringRef ArchiveName = Buffer->getBufferIdentifier();
  if (Config.Verbose)
    errs() << "Reading library archive file '" << ArchiveName
           << "' to memory\n";
  Expected<std::unique_ptr<object::Archive>> ArchiveOrError =
      object::Archive::create(Buffer->getMemBufferRef());
  if (!ArchiveOrError)
    ExitOnErr(ArchiveOrError.takeError());

  std::unique_ptr<object::Archive> Archive = std::move(ArchiveOrError.get());

  Linker L(*Result);
  Error Err = Error::success();
  for (const object::Archive::Child &C : Archive->children(Err)) {
    Expected<StringRef> Ename = C.getName();
    if (Error E = Ename.takeError()) {
      errs() << Argv0 << ": ";
      WithColor::error() << " failed to read name of archive member"
                         << ArchiveName << "'\n";
      return nullptr;
    }
    std::string ChildName = Ename.get().str();
    if (Config.Verbose)
      errs() << "Parsing member '" << ChildName
             << "' of archive library to module.\n";
    SMDiagnostic ParseErr;
    Expected<MemoryBufferRef> MemBuf = C.getMemoryBufferRef();
    if (Error E = MemBuf.takeError()) {
      errs() << Argv0 << ": ";
      WithColor::error() << " loading memory for member '" << ChildName
                         << "' of archive library failed'" << ArchiveName
                         << "'\n";
      return nullptr;
    };

    if (!isBitcode(reinterpret_cast<const unsigned char *>(
                       MemBuf.get().getBufferStart()),
                   reinterpret_cast<const unsigned char *>(
                       MemBuf.get().getBufferEnd()))) {
      if (Config.IgnoreNonBitcode)
        continue;
      errs() << Argv0 << ": ";
      WithColor::error() << "  member of archive is not a bitcode file: '"
                         << ChildName << "'\n";
      return nullptr;
    }

    std::unique_ptr<Module> M;
    if (Config.DisableLazyLoad)
      M = parseIR(MemBuf.get(), ParseErr, Context);
    else
      M = getLazyIRModule(MemoryBuffer::getMemBuffer(MemBuf.get(), false),
                          ParseErr, Context);

    if (!M) {
      errs() << Argv0 << ": ";
      WithColor::error() << " parsing member '" << ChildName
                         << "' of archive library failed'" << ArchiveName
                         << "'\n";
      return nullptr;
    }
    if (Config.Verbose)
      errs() << "Linking member '" << ChildName << "' of archive library.\n";
    if (L.linkInModule(std::move(M)))
      return nullptr;
  } // end for each child
  ExitOnErr(std::move(Err));
  return Result;
}

namespace {

/// Helper to load on demand a Module from file and cache it for subsequent
/// queries during function importing.
class ModuleLazyLoaderCache {
  /// Cache of lazily loaded module for import.
  StringMap<std::unique_ptr<Module>> ModuleMap;

  /// Retrieve a Module from the cache or lazily load it on demand.
  std::function<std::unique_ptr<Module>(const char *argv0,
                                        const std::string &FileName)>
      createLazyModule;

public:
  /// Create the loader, Module will be initialized in \p Context.
  ModuleLazyLoaderCache(std::function<std::unique_ptr<Module>(
                            const char *argv0, const std::string &FileName)>
                            createLazyModule)
      : createLazyModule(std::move(createLazyModule)) {}

  /// Retrieve a Module from the cache or lazily load it on demand.
  Module &operator()(const char *argv0, const std::string &FileName);

  std::unique_ptr<Module> takeModule(const std::string &FileName) {
    auto I = ModuleMap.find(FileName);
    assert(I != ModuleMap.end());
    std::unique_ptr<Module> Ret = std::move(I->second);
    ModuleMap.erase(I);
    return Ret;
  }
};

// Get a Module for \p FileName from the cache, or load it lazily.
Module &ModuleLazyLoaderCache::operator()(const char *argv0,
                                          const std::string &Identifier) {
  auto &Module = ModuleMap[Identifier];
  if (!Module) {
    Module = createLazyModule(argv0, Identifier);
    assert(Module && "Failed to create lazy module!");
  }
  return *Module;
}
} // anonymous namespace

namespace {
struct LLVMLinkDiagnosticHandler : public DiagnosticHandler {
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    unsigned Severity = DI.getSeverity();
    switch (Severity) {
    case DS_Error:
      WithColor::error();
      break;
    case DS_Warning:
      if (Config.SuppressWarnings)
        return true;
      WithColor::warning();
      break;
    case DS_Remark:
    case DS_Note:
      llvm_unreachable("Only expecting warnings and errors");
    }

    DiagnosticPrinterRawOStream DP(errs());
    DI.print(DP);
    errs() << '\n';
    return true;
  }
};
} // namespace

/// Import any functions requested via the -import option.
static bool importFunctions(const char *argv0, Module &DestModule) {
  if (Config.SummaryIndex.empty())
    return true;
  std::unique_ptr<ModuleSummaryIndex> Index =
      ExitOnErr(llvm::getModuleSummaryIndexForFile(Config.SummaryIndex));

  // Map of Module -> List of globals to import from the Module
  FunctionImporter::ImportIDTable ImportIDs;
  FunctionImporter::ImportMapTy ImportList(ImportIDs);

  auto ModuleLoader = [&DestModule](const char *argv0,
                                    const std::string &Identifier) {
    std::unique_ptr<MemoryBuffer> Buffer = ExitOnErr(errorOrToExpected(
        MemoryBuffer::getFileOrSTDIN(Identifier, /*IsText=*/true)));
    return loadFile(argv0, std::move(Buffer), DestModule.getContext(), false);
  };

  ModuleLazyLoaderCache ModuleLoaderCache(ModuleLoader);
  // Owns the filename strings used to key into the ImportList. Normally this is
  // constructed from the index and the strings are owned by the index, however,
  // since we are synthesizing this data structure from options we need a cache
  // to own those strings.
  StringSet<> FileNameStringCache;
  for (const auto &Import : Config.Imports) {
    // Identify the requested function and its bitcode source file.
    size_t Idx = Import.find(':');
    if (Idx == std::string::npos) {
      errs() << "Import parameter bad format: " << Import << "\n";
      return false;
    }
    std::string FunctionName = Import.substr(0, Idx);
    std::string FileName = Import.substr(Idx + 1, std::string::npos);

    // Load the specified source module.
    auto &SrcModule = ModuleLoaderCache(argv0, FileName);

    if (!Config.NoVerify && verifyModule(SrcModule, &errs())) {
      errs() << argv0 << ": " << FileName;
      WithColor::error() << "input module is broken!\n";
      return false;
    }

    Function *F = SrcModule.getFunction(FunctionName);
    if (!F) {
      errs() << "Ignoring import request for non-existent function "
             << FunctionName << " from " << FileName << "\n";
      continue;
    }
    // We cannot import weak_any functions without possibly affecting the
    // order they are seen and selected by the linker, changing program
    // semantics.
    if (F->hasWeakAnyLinkage()) {
      errs() << "Ignoring import request for weak-any function " << FunctionName
             << " from " << FileName << "\n";
      continue;
    }

    if (Config.Verbose)
      errs() << "Importing " << FunctionName << " from " << FileName << "\n";

    // `-import` specifies the `<filename,function-name>` pairs to import as
    // definition, so make the import type definition directly.
    // FIXME: A follow-up patch should add test coverage for import declaration
    // in `llvm-link` CLI (e.g., by introducing a new command line option).
    ImportList.addDefinition(
        FileNameStringCache.insert(FileName).first->getKey(), F->getGUID());
  }
  auto CachedModuleLoader = [&](StringRef Identifier) {
    return ModuleLoaderCache.takeModule(std::string(Identifier));
  };
  FunctionImporter Importer(*Index, CachedModuleLoader,
                            /*ClearDSOLocalOnDeclarations=*/false);
  ExitOnErr(Importer.importFunctions(DestModule, ImportList));

  return true;
}

static bool linkFiles(const char *argv0, LLVMContext &Context, Linker &L,
                      const std::vector<std::string> &Files, unsigned Flags) {
  // Filter out flags that don't apply to the first file we load.
  unsigned ApplicableFlags = Flags & Linker::Flags::OverrideFromSrc;
  // Similar to some flags, internalization doesn't apply to the first file.
  bool InternalizeLinkedSymbols = false;
  for (const auto &File : Files) {
    auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(File, /*IsText=*/true);

    // When we encounter a missing file, make sure we expose its name.
    if (auto EC = BufferOrErr.getError())
      if (EC == std::errc::no_such_file_or_directory)
        ExitOnErr(createStringError(EC, "No such file or directory: '%s'",
                                    File.c_str()));

    std::unique_ptr<MemoryBuffer> Buffer =
        ExitOnErr(errorOrToExpected(std::move(BufferOrErr)));

    std::unique_ptr<Module> M =
        identify_magic(Buffer->getBuffer()) == file_magic::archive
            ? loadArFile(argv0, std::move(Buffer), Context)
            : loadFile(argv0, std::move(Buffer), Context);
    if (!M) {
      errs() << argv0 << ": ";
      WithColor::error() << " loading file '" << File << "'\n";
      return false;
    }

    // Note that when ODR merging types cannot verify input files in here When
    // doing that debug metadata in the src module might already be pointing to
    // the destination.
    if (Config.DisableDITypeMap && !Config.NoVerify &&
        verifyModule(*M, &errs())) {
      errs() << argv0 << ": " << File << ": ";
      WithColor::error() << "input module is broken!\n";
      return false;
    }

    // If a module summary index is supplied, load it so linkInModule can treat
    // local functions/variables as exported and promote if necessary.
    if (!Config.SummaryIndex.empty()) {
      std::unique_ptr<ModuleSummaryIndex> Index =
          ExitOnErr(llvm::getModuleSummaryIndexForFile(Config.SummaryIndex));

      // Conservatively mark all internal values as promoted, since this tool
      // does not do the ThinLink that would normally determine what values to
      // promote.
      for (auto &I : *Index) {
        for (auto &S : I.second.getSummaryList()) {
          if (GlobalValue::isLocalLinkage(S->linkage()))
            S->setExternalLinkageForTest();
        }
      }

      // Promotion
      renameModuleForThinLTO(*M, *Index,
                             /*ClearDSOLocalOnDeclarations=*/false);
    }

    if (Config.Verbose)
      errs() << "Linking in '" << File << "'\n";

    bool Err = false;
    if (InternalizeLinkedSymbols) {
      Err = L.linkInModule(
          std::move(M), ApplicableFlags, [](Module &M, const StringSet<> &GVS) {
            internalizeModule(M, [&GVS](const GlobalValue &GV) {
              return !GV.hasName() || (GVS.count(GV.getName()) == 0);
            });
          });
    } else {
      Err = L.linkInModule(std::move(M), ApplicableFlags);
    }

    if (Err)
      return false;

    // Internalization applies to linking of subsequent files.
    InternalizeLinkedSymbols = Config.Internalize;

    // All linker flags apply to linking of subsequent files.
    ApplicableFlags = Flags;
  }

  return true;
}

int llvm_link_main(int argc, char **argv, const llvm::ToolContext &) {
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  Expected<LinkerConfig> ParsedConfig = parseArgs(argc, argv);
  if (!ParsedConfig) {
    logAllUnhandledErrors(ParsedConfig.takeError(),
                          WithColor::error(errs(), argv[0]));
    return 1;
  }
  Config = std::move(*ParsedConfig);

  LLVMContext Context;
  Context.setDiagnosticHandler(std::make_unique<LLVMLinkDiagnosticHandler>(),
                               true);

  if (!Config.DisableDITypeMap)
    Context.enableDebugTypeODRUniquing();

  auto Composite = std::make_unique<Module>("llvm-link", Context);
  Linker L(*Composite);

  unsigned Flags = Linker::Flags::None;
  if (Config.OnlyNeeded)
    Flags |= Linker::Flags::LinkOnlyNeeded;

  // First add all the regular input files
  if (!linkFiles(argv[0], Context, L, Config.InputFilenames, Flags))
    return 1;

  // Next the -override ones.
  if (!linkFiles(argv[0], Context, L, Config.OverridingInputs,
                 Flags | Linker::Flags::OverrideFromSrc))
    return 1;

  // Import any functions requested via -import
  if (!importFunctions(argv[0], *Composite))
    return 1;

  if (Config.DumpAsm)
    errs() << "Here's the assembly:\n" << *Composite;

  std::error_code EC;
  ToolOutputFile Out(Config.OutputFilename, EC,
                     Config.OutputAssembly ? sys::fs::OF_TextWithCRLF
                                           : sys::fs::OF_None);
  if (EC) {
    WithColor::error() << EC.message() << '\n';
    return 1;
  }

  if (!Config.NoVerify && verifyModule(*Composite, &errs())) {
    errs() << argv[0] << ": ";
    WithColor::error() << "linked module is broken!\n";
    return 1;
  }

  if (Config.Verbose)
    errs() << "Writing bitcode...\n";
  Composite->removeDebugIntrinsicDeclarations();
  if (Config.OutputAssembly) {
    Composite->print(Out.os(), nullptr, /* ShouldPreserveUseListOrder */ false);
  } else if (Config.Force || !CheckBitcodeOutputToConsole(Out.os())) {
    WriteBitcodeToFile(*Composite, Out.os(),
                       /* ShouldPreserveUseListOrder */ true);
  }

  // Declare success.
  Out.keep();

  return 0;
}
