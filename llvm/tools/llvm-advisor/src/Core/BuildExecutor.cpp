//===---------------- BuildExecutor.cpp - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the BuildExecutor code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "BuildExecutor.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Options/Options.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace advisor {

namespace {

bool isClangCompiler(llvm::StringRef CompilerPath) {
  llvm::StringRef Name = llvm::sys::path::filename(CompilerPath);
  return Name.starts_with("clang");
}

/// Resolve a compiler name/path to an absolute path on disk via getToolPath
/// and PATH lookup.  Returns an error if the compiler cannot be found.
llvm::Expected<std::string>
resolveCompilerPath(const AdvisorConfig &Config, llvm::StringRef Compiler) {
  std::string Requested = Config.getToolPath(Compiler);
  if (auto Resolved = llvm::sys::findProgramByName(Requested))
    return *Resolved;
  if (llvm::sys::fs::exists(Requested))
    return Requested;
  return llvm::createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Compiler not found: " + Requested);
}

/// Build a clang::driver::Driver + Compilation from an argument list.
/// For non-Clang compilers the Driver and Compilation fields are left null and
/// InstrumentedArgs is set to a verbatim copy of Args.
///
/// \param SetCheckInputsExist  Pass true when the compilation will be executed
///                             (so the driver validates that input files exist),
///                             false when the result is used for analysis only.
llvm::Expected<BuildExecutor::PreparedBuild>
buildDriverCompilation(llvm::StringRef CompilerPath,
                       llvm::ArrayRef<std::string> Args,
                       bool SetCheckInputsExist) {
  BuildExecutor::PreparedBuild Build;
  Build.CompilerPath = CompilerPath.str();
  Build.InstrumentedArgs.assign(Args.begin(), Args.end());
  Build.UsesDriver = isClangCompiler(CompilerPath);
  if (!Build.UsesDriver)
    return Build;

  auto DiagOpts = std::make_shared<clang::DiagnosticOptions>();
  auto DiagPrinter = std::make_unique<clang::TextDiagnosticPrinter>(
      llvm::errs(), DiagOpts.get());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(
      new clang::DiagnosticIDs());
  auto Diagnostics = std::make_shared<clang::DiagnosticsEngine>(
      DiagIDs, DiagOpts.get(), DiagPrinter.get());

  auto Driver = std::make_unique<clang::driver::Driver>(
      Build.CompilerPath, llvm::sys::getDefaultTargetTriple(), *Diagnostics);
  Driver->setTitle("llvm-advisor");
  Driver->setCheckInputsExist(SetCheckInputsExist);

  llvm::SmallVector<const char *, 32> Argv;
  Argv.push_back(Build.CompilerPath.c_str());
  for (const auto &Arg : Args)
    Argv.push_back(Arg.c_str());

  auto Compilation = Driver->BuildCompilation(Argv);
  if (!Compilation)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Failed to build clang driver invocation for: " + CompilerPath.str());

  Build.Driver = std::move(Driver);
  Build.Compilation = std::move(Compilation);
  Build.Diagnostics = std::move(Diagnostics);
  Build.DiagnosticClient = std::move(DiagPrinter);
  return Build;
}

} // namespace

BuildExecutor::BuildExecutor(const AdvisorConfig &config) : config(config) {}

llvm::Expected<int> BuildExecutor::execute(
    llvm::StringRef Compiler, const llvm::SmallVectorImpl<std::string> &Args,
    BuildContext &BuildCtx, llvm::StringRef TempDir,
    llvm::StringRef ArtifactRoot) {
  auto Prepared =
      prepareBuild(Compiler, Args, BuildCtx, TempDir, ArtifactRoot);
  if (!Prepared)
    return Prepared.takeError();
  return executePreparedBuild(*Prepared);
}

llvm::Expected<BuildExecutor::PreparedBuild>
BuildExecutor::buildOriginalCompilation(
    llvm::StringRef Compiler,
    const llvm::SmallVectorImpl<std::string> &Args) {
  auto CompilerPathOrErr = resolveCompilerPath(config, Compiler);
  if (!CompilerPathOrErr)
    return CompilerPathOrErr.takeError();

  // setCheckInputsExist=false: we only need the Driver model for analysis,
  // not a real compilation that validates input files.
  return buildDriverCompilation(*CompilerPathOrErr, Args,
                                /*SetCheckInputsExist=*/false);
}

llvm::Expected<BuildExecutor::PreparedBuild>
BuildExecutor::prepareBuild(llvm::StringRef Compiler,
                            const llvm::SmallVectorImpl<std::string> &Args,
                            BuildContext &BuildCtx, llvm::StringRef TempDir,
                            llvm::StringRef ArtifactRoot) {
  auto CompilerPathOrErr = resolveCompilerPath(config, Compiler);
  if (!CompilerPathOrErr)
    return CompilerPathOrErr.takeError();
  const std::string &ResolvedCompiler = *CompilerPathOrErr;

  // Instrument the user args (add debug info, remarks, profiling flags, etc.).
  auto InstrumentedArgsOrErr = instrumentCompilerArgs(
      ResolvedCompiler, Args, BuildCtx, TempDir, ArtifactRoot);
  if (!InstrumentedArgsOrErr)
    return InstrumentedArgsOrErr.takeError();

  // Build the execution compilation from the instrumented args.
  // setCheckInputsExist=true so the driver validates inputs before execution.
  auto BuildOrErr =
      buildDriverCompilation(ResolvedCompiler, *InstrumentedArgsOrErr,
                             /*SetCheckInputsExist=*/true);
  if (!BuildOrErr)
    return BuildOrErr.takeError();

  // buildDriverCompilation stores the args it receives into InstrumentedArgs.
  // Since we passed the already-instrumented list that is correct — no
  // additional fixup needed here.
  return BuildOrErr;
}

llvm::Expected<int>
BuildExecutor::executePreparedBuild(PreparedBuild &Build) {
  if (config.getVerbose()) {
    llvm::outs() << "Executing: " << Build.CompilerPath;
    for (const auto &Arg : Build.InstrumentedArgs)
      llvm::outs() << " " << Arg;
    llvm::outs() << "\n";
  }

  if (!Build.UsesDriver) {
    llvm::SmallVector<llvm::StringRef, 16> ExecArgs;
    ExecArgs.push_back(Build.CompilerPath); // argv[0]
    for (const auto &Arg : Build.InstrumentedArgs)
      ExecArgs.push_back(Arg);
    return llvm::sys::ExecuteAndWait(Build.CompilerPath, ExecArgs);
  }

  llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>
      FailingCommands;
  int Result =
      Build.Driver->ExecuteCompilation(*Build.Compilation, FailingCommands);

  if (!FailingCommands.empty()) {
    for (const auto &Failure : FailingCommands) {
      const auto *Cmd = Failure.second;
      llvm::errs() << "clang job failed";
      if (Cmd)
        llvm::errs() << ": " << Cmd->getExecutable();
      llvm::errs() << " (exit code: " << Failure.first << ")\n";
    }
  }

  return Result;
}

// NOTE: We parse and rewrite compiler flags using the clang Driver's
// opt::ArgList / DerivedArgList infrastructure so that we manipulate arguments
// at a structured level rather than via raw string operations.  The Driver is
// instantiated with setCheckInputsExist(false) because we only need the parsed
// arg model here, not a real compilation.
llvm::Expected<llvm::SmallVector<std::string, 16>>
BuildExecutor::instrumentCompilerArgs(
    llvm::StringRef CompilerPath, const llvm::SmallVectorImpl<std::string> &Args,
    BuildContext &BuildCtx, llvm::StringRef TempDir,
    llvm::StringRef ArtifactRoot) {
  std::string OutputRoot =
      ArtifactRoot.empty() ? std::string(TempDir) : std::string(ArtifactRoot);
  auto registerExpectedFile = [&](llvm::StringRef Path) {
    if (!Path.empty())
      BuildCtx.expectedGeneratedFiles.push_back(Path.str());
  };

  const bool IsClangTool = isClangCompiler(CompilerPath);

  // For non-Clang compilers we only guarantee debug info is present.
  if (!IsClangTool) {
    llvm::SmallVector<std::string, 16> Result(Args.begin(), Args.end());
    bool HasDebug = false;
    for (const auto &Arg : Result) {
      if (llvm::StringRef(Arg).starts_with("-g")) {
        HasDebug = true;
        break;
      }
    }
    if (!HasDebug) {
      Result.push_back("-g");
      BuildCtx.hasDebugInfo = true;
    }
    return Result;
  }

  // ── Clang path: parse with Driver opt infrastructure ───────────────────────
  auto DiagOpts = std::make_shared<clang::DiagnosticOptions>();
  auto DiagPrinter = std::make_unique<clang::TextDiagnosticPrinter>(
      llvm::errs(), DiagOpts.get());
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(
      new clang::DiagnosticIDs());
  auto Diagnostics = std::make_shared<clang::DiagnosticsEngine>(
      DiagIDs, DiagOpts.get(), DiagPrinter.get());

  std::string CompilerStorage = CompilerPath.str();
  auto Driver = std::make_unique<clang::driver::Driver>(
      CompilerStorage, llvm::sys::getDefaultTargetTriple(), *Diagnostics);
  Driver->setTitle("llvm-advisor");
  Driver->setCheckInputsExist(false);

  llvm::SmallVector<const char *, 32> Argv;
  Argv.push_back(CompilerStorage.c_str());
  for (const auto &Arg : Args)
    Argv.push_back(Arg.c_str());

  bool ContainsError = false;
  auto InputArgs = std::make_unique<llvm::opt::InputArgList>(
      Driver->ParseArgStrings(llvm::makeArrayRef(Argv).slice(1),
                              /*UseDriverMode=*/true, ContainsError));
  if (ContainsError)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "failed to parse compiler arguments for instrumentation");

  std::unique_ptr<llvm::opt::DerivedArgList> Derived(
      Driver->TranslateInputArgs(*InputArgs));
  if (!Derived)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "failed to translate compiler arguments for instrumentation");

  const auto &Opts = Driver->getOpts();

  // Ensure debug info is always present so DWARF extraction works.
  if (!Derived->hasArgNoClaim(clang::options::OPT_g_Group)) {
    Derived->AddFlagArg(nullptr, Opts.getOption(clang::options::OPT_g_Flag));
    BuildCtx.hasDebugInfo = true;
  }

  // ── Optimization remarks ───────────────────────────────────────────────────
  // Rewrite any existing -foptimization-record-file= paths to land inside
  // OutputRoot so they are captured alongside other artifacts.
  auto rewriteRemarksFile = [&](llvm::StringRef Requested) -> std::string {
    llvm::StringRef FileName = llvm::sys::path::filename(Requested);
    std::string NewPath = (llvm::Twine(OutputRoot) + "/" + FileName).str();
    registerExpectedFile(NewPath);
    return NewPath;
  };

  bool FoundExplicitRemarksFile = false;
  for (llvm::opt::Arg *A :
       Derived->filtered(clang::options::OPT_foptimization_record_file_EQ)) {
    FoundExplicitRemarksFile = true;
    std::string NewPath = rewriteRemarksFile(A->getValue());
    A->getValues()[0] = Derived->MakeArgString(NewPath);
  }

  const bool HasRemarks =
      Derived->hasArgNoClaim(clang::options::OPT_fsave_optimization_record) ||
      Derived->hasArgNoClaim(clang::options::OPT_fsave_optimization_record_EQ) ||
      Derived->hasArgNoClaim(clang::options::OPT_foptimization_record_file_EQ);

  // Helper: attach both the enable flag and an explicit output path.
  auto attachRemarks = [&](llvm::StringRef RemarksPath) {
    Derived->AddFlagArg(
        nullptr, Opts.getOption(clang::options::OPT_fsave_optimization_record));
    Derived->AddJoinedArg(
        nullptr,
        Opts.getOption(clang::options::OPT_foptimization_record_file_EQ),
        Derived->MakeArgString(RemarksPath));
    registerExpectedFile(RemarksPath);
  };

  if (!HasRemarks) {
    // No remarks requested at all — enable them with a controlled output path.
    attachRemarks((llvm::Twine(OutputRoot) + "/remarks.opt.yaml").str());
  } else if (!FoundExplicitRemarksFile) {
    // -fsave-optimization-record (or -EQ variant) present but no explicit
    // output file.  Pin the output to OutputRoot so we know where to find it.
    std::string RemarksPath =
        (llvm::Twine(OutputRoot) + "/remarks.opt.yaml").str();
    Derived->AddJoinedArg(
        nullptr,
        Opts.getOption(clang::options::OPT_foptimization_record_file_EQ),
        Derived->MakeArgString(RemarksPath));
    registerExpectedFile(RemarksPath);
  }
  // else: user provided an explicit path and we already rewrote it above.

  // ── Coverage / PGO profiling ───────────────────────────────────────────────
  const bool HasProfile =
      Derived->hasArgNoClaim(clang::options::OPT_fprofile_instr_generate) ||
      Derived->hasArgNoClaim(clang::options::OPT_fprofile_instr_generate_EQ);

  if (config.getRunProfiler() && !HasProfile) {
    std::string Profraw = (llvm::Twine(OutputRoot) + "/profile.profraw").str();
    Derived->AddJoinedArg(
        nullptr,
        Opts.getOption(clang::options::OPT_fprofile_instr_generate_EQ),
        Derived->MakeArgString(Profraw));
    Derived->AddFlagArg(
        nullptr, Opts.getOption(clang::options::OPT_fcoverage_mapping));
    registerExpectedFile(Profraw);

    std::string Profdata =
        (llvm::Twine(OutputRoot) + "/profile.profdata").str();
    std::string Report = (llvm::Twine(OutputRoot) + "/coverage.json").str();
    registerExpectedFile(Profdata);
    registerExpectedFile(Report);

    CoverageProfileSite Site;
    Site.rawProfile = Profraw;
    Site.indexedProfile = Profdata;
    Site.reportPath = Report;
    BuildCtx.coverageSites.push_back(std::move(Site));
  }

  // ── Rpass remarks ──────────────────────────────────────────────────────────
  if (!Derived->hasArgNoClaim(clang::options::OPT_Rpass_EQ)) {
    Derived->AddJoinedArg(nullptr, Opts.getOption(clang::options::OPT_Rpass_EQ),
                          Derived->MakeArgString("kernel-info"));
    Derived->AddJoinedArg(nullptr, Opts.getOption(clang::options::OPT_Rpass_EQ),
                          Derived->MakeArgString("analysis"));
  }

  // ── Diagnostic format ──────────────────────────────────────────────────────
  const bool HasDiagFormat =
      Derived->hasArgNoClaim(clang::options::OPT_fdiagnostics_format_EQ) ||
      Derived->hasArgNoClaim(clang::options::OPT_fdiagnostics_format);

  if (!HasDiagFormat) {
    Derived->AddFlagArg(
        nullptr,
        Opts.getOption(clang::options::OPT_fdiagnostics_parseable_fixits));
    Derived->AddFlagArg(
        nullptr,
        Opts.getOption(clang::options::OPT_fdiagnostics_absolute_paths));
  }

  // ── Render back to a flat string list ─────────────────────────────────────
  llvm::opt::ArgStringList Rendered;
  for (const llvm::opt::Arg *A : *Derived)
    A->render(*Derived, Rendered);

  llvm::SmallVector<std::string, 16> FinalArgs;
  FinalArgs.reserve(Rendered.size());
  for (const char *S : Rendered)
    FinalArgs.emplace_back(S);
  return FinalArgs;
}

} // namespace advisor
} // namespace llvm