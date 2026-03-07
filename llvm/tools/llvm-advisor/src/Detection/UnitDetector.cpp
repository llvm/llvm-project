//===--------------------- UnitDetector.cpp - LLVM Advisor ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the UnitDetector code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "UnitDetector.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Types.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include <memory>

namespace llvm {
namespace advisor {

namespace {

struct DriverInvocation {
  std::string Executable;
  std::unique_ptr<clang::DiagnosticOptions> DiagnosticOptions;
  std::unique_ptr<clang::DiagnosticConsumer> Client;
  std::unique_ptr<clang::DiagnosticsEngine> Diagnostics;
  std::unique_ptr<clang::driver::Driver> Driver;
  std::unique_ptr<clang::driver::Compilation> Compilation;
};

bool isClangExecutable(llvm::StringRef Compiler) {
  return llvm::sys::path::filename(Compiler).starts_with("clang");
}

std::unique_ptr<DriverInvocation>
buildDriverInvocation(llvm::StringRef Compiler,
                      llvm::ArrayRef<std::string> Args) {
  if (!isClangExecutable(Compiler))
    return nullptr;

  auto Path = llvm::sys::findProgramByName(Compiler);
  if (!Path)
    return nullptr;

  auto Invocation = std::make_unique<DriverInvocation>();
  Invocation->Executable = *Path;
  Invocation->DiagnosticOptions = std::make_unique<clang::DiagnosticOptions>();
  auto DiagBuffer = std::make_unique<clang::TextDiagnosticBuffer>();
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(
      new clang::DiagnosticIDs());
  Invocation->Diagnostics = std::make_unique<clang::DiagnosticsEngine>(
      DiagIDs, *Invocation->DiagnosticOptions, DiagBuffer.get(),
      /*ShouldOwnClient=*/false);

  auto Driver = std::make_unique<clang::driver::Driver>(
      Invocation->Executable, llvm::sys::getDefaultTargetTriple(),
      *Invocation->Diagnostics);
  Driver->setCheckInputsExist(false);

  llvm::SmallVector<const char *, 32> Argv;
  Argv.push_back(Invocation->Executable.c_str());
  for (const auto &Arg : Args)
    Argv.push_back(Arg.c_str());

  auto *Compilation = Driver->BuildCompilation(Argv);
  if (!Compilation)
    return nullptr;

  Invocation->Client = std::move(DiagBuffer);
  Invocation->Driver = std::move(Driver);
  Invocation->Compilation.reset(Compilation);
  return Invocation;
}

} // namespace

UnitDetector::UnitDetector(const AdvisorConfig &Config) : config(Config) {}

static bool isSourceExtension(llvm::StringRef Extension) {
  return Extension == ".c" || Extension == ".C" || Extension == ".cc" ||
         Extension == ".cpp" || Extension == ".cxx";
}

/// Capture compiler flags from the original command-line arguments, excluding:
///   - source files (already tracked via CompilationUnitInfo::sources)
///   - object files, archives and other linker inputs that must not be
///     replayed when re-invoking the compiler for data extraction
static void captureCompileFlags(const llvm::SmallVectorImpl<std::string> &Args,
                                CompilationUnitInfo &Unit) {
  // Extensions that identify linker/archiver inputs rather than compiler flags.
  auto IsLinkerInput = [](llvm::StringRef Ext) -> bool {
    return Ext == ".o" || Ext == ".O" || Ext == ".obj" || Ext == ".OBJ" ||
           Ext == ".a" || Ext == ".so" || Ext == ".dylib" || Ext == ".lib" ||
           Ext == ".dll";
  };

  for (size_t I = 0; I < Args.size(); ++I) {
    llvm::StringRef Arg(Args[I]);

    if (Arg.empty())
      continue;

    // Flag argument (starts with '-').
    if (Arg[0] == '-') {
      // Skip paired flags whose value is the next argument and whose value is
      // not itself a compilation flag (e.g. -o <output>, -MF <depfile>).
      // We keep -I/-D/-U/-std=/-m*/-f*/-W*/-O* and similar because they
      // affect re-compilation.
      if ((Arg == "-o" || Arg == "-MF" || Arg == "-MT" || Arg == "-MQ" ||
           Arg == "-isystem" || Arg == "-isysroot" || Arg == "-iprefix" ||
           Arg == "-iwithprefix" || Arg == "-iwithprefixbefore" ||
           Arg == "-idirafter") &&
          I + 1 < Args.size()) {
        // Consume flag + value but do NOT push them; output paths and dep-file
        // paths are irrelevant when replaying compilation for extraction.
        ++I;
        continue;
      }
      Unit.compileFlags.push_back(Args[I]);
      continue;
    }

    // Positional argument: keep only if it is a source file that is not
    // already tracked, and drop linker inputs / output file names.
    llvm::StringRef Ext = llvm::sys::path::extension(Arg);
    if (isSourceExtension(Ext) || IsLinkerInput(Ext))
      continue; // source files live in 'sources'; linker inputs are noise

    // Bare positional values that are not recognisable file extensions
    // (e.g. library short names passed after -l... in non-joined form) are
    // also excluded because they are not valid compiler flags.
  }
}

static bool isClangCompileCommand(const clang::driver::Command &Cmd) {
  llvm::StringRef ToolName = Cmd.getCreator().getName();
  return ToolName.starts_with("clang");
}

llvm::Expected<llvm::SmallVector<CompilationUnitInfo, 4>>
UnitDetector::detectUnits(llvm::StringRef Compiler,
                          const llvm::SmallVectorImpl<std::string> &Args,
                          const clang::driver::Compilation *DriverCompilation) {
  std::string ResolvedCompiler = Compiler.str();
  if (auto CompilerPath = llvm::sys::findProgramByName(Compiler))
    ResolvedCompiler = *CompilerPath;

  std::unique_ptr<DriverInvocation> LocalInvocation;
  if (!DriverCompilation) {
    LocalInvocation = buildDriverInvocation(Compiler, Args);
    if (LocalInvocation)
      DriverCompilation = LocalInvocation->Compilation.get();
  }

  if (DriverCompilation) {
    llvm::SmallVector<CompilationUnitInfo, 4> Units;
    const auto &Jobs = DriverCompilation->getJobs();
    for (const auto &Cmd : Jobs) {
      if (!isClangCompileCommand(Cmd))
        continue;

      CompilationUnitInfo Unit;
      Unit.sources.clear();
      for (const auto &Input : Cmd.getInputInfos()) {
        if (!Input.isFilename())
          continue;
        if (!clang::driver::types::isSrcFile(Input.getType()))
          continue;
        SourceFile Source;
        Source.path = Input.getFilename();
        Source.language = classifier.getLanguage(Source.path);
        Source.isHeader = false;
        Unit.sources.push_back(Source);
      }

      if (Unit.sources.empty())
        continue;

      Unit.name = generateUnitName(Unit.sources);
      Unit.compilerPath = ResolvedCompiler;
      Unit.cc1Args.reserve(Cmd.getArguments().size());
      for (const char *Arg : Cmd.getArguments()) {
        if (Arg)
          Unit.cc1Args.emplace_back(Arg);
      }

      const auto &Outputs = Cmd.getOutputFilenames();
      if (!Outputs.empty())
        Unit.outputObject = Outputs.front();

      Unit.targetArch =
          DriverCompilation->getDefaultToolChain().getTripleString();
      captureCompileFlags(Args, Unit);
      Units.push_back(std::move(Unit));
    }

    if (!Units.empty()) {
      for (auto &Unit : Units)
        extractBuildInfo(Args, Unit);
      return Units;
    }
  }

  auto Sources = findSourceFiles(Args);
  if (Sources.empty()) {
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "No source files found");
  }

  CompilationUnitInfo Unit;
  Unit.name = generateUnitName(Sources);
  Unit.compilerPath = ResolvedCompiler;
  Unit.sources = Sources;

  captureCompileFlags(Args, Unit);

  extractBuildInfo(Args, Unit);

  return llvm::SmallVector<CompilationUnitInfo, 4>{Unit};
}

llvm::SmallVector<SourceFile, 4> UnitDetector::findSourceFiles(
    const llvm::SmallVectorImpl<std::string> &Args) const {
  llvm::SmallVector<SourceFile, 4> Sources;

  for (const auto &Arg : Args) {
    if (Arg.empty() || Arg[0] == '-')
      continue;

    llvm::StringRef Extension = llvm::sys::path::extension(Arg);
    if (Extension == ".c" || Extension == ".cpp" || Extension == ".cc" ||
        Extension == ".cxx" || Extension == ".C") {

      SourceFile Source;
      Source.path = Arg;
      Source.language = classifier.getLanguage(Arg);
      Source.isHeader = false;
      Sources.push_back(Source);
    }
  }

  return Sources;
}

void UnitDetector::extractBuildInfo(
    const llvm::SmallVectorImpl<std::string> &Args, CompilationUnitInfo &Unit) {
  for (size_t I = 0; I < Args.size(); ++I) {
    const auto &Arg = Args[I];

    if (Arg == "-o" && I + 1 < Args.size()) {
      llvm::StringRef Output = Args[I + 1];
      llvm::StringRef Ext = llvm::sys::path::extension(Output);
      if (Ext == ".o")
        Unit.outputObject = Args[I + 1];
      else
        Unit.outputExecutable = Args[I + 1];
    }

    llvm::StringRef ArgRef(Arg);
    if (ArgRef.contains("openmp") || ArgRef.contains("offload") ||
        ArgRef.contains("cuda")) {
      Unit.hasOffloading = true;
    }

    if (llvm::StringRef(Arg).starts_with("-march="))
      Unit.targetArch = Arg.substr(7);
  }
}

std::string UnitDetector::generateUnitName(
    const llvm::SmallVectorImpl<SourceFile> &Sources) const {
  if (Sources.empty())
    return "unknown";

  // Use first source file name as base
  std::string BaseName = llvm::sys::path::stem(Sources[0].path).str();

  // Add hash for uniqueness when multiple sources
  if (Sources.size() > 1) {
    std::string Combined;
    for (const auto &Source : Sources)
      Combined += Source.path;
    auto Hash = llvm::hash_value(Combined);
    BaseName += "_" + std::to_string(static_cast<size_t>(Hash) % 10000);
  }

  return BaseName;
}

} // namespace advisor
} // namespace llvm
