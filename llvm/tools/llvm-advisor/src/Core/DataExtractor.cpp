//===------------------ DataExtractor.cpp - LLVM Advisor ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the DataExtractor code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "DataExtractor.h"
#include "../Utils/FileManager.h"
#include "../Utils/ProcessRunner.h"
#include "CoverageProcessor.h"
#include "DriverContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <system_error>

namespace llvm {
namespace advisor {

namespace {

struct CoveragePaths {
  std::string InstrumentedExe;
  std::string Profraw;
  std::string Profdata;
  std::string Report;
  std::string Instructions;
};

CoveragePaths computeCoverageArtifacts(const CompilationUnit &Unit,
                                       const SourceFile &Source) {
  CoveragePaths Paths;
  Paths.InstrumentedExe =
      Unit.makeArtifactPath("coverage", Source.path, ".instrumented");
  Paths.Profraw = Unit.makeArtifactPath("coverage", Source.path, ".profraw");
  Paths.Profdata = Unit.makeArtifactPath("coverage", Source.path, ".profdata");
  Paths.Report =
      Unit.makeArtifactPath("coverage", Source.path, ".coverage.json");
  Paths.Instructions =
      Unit.makeArtifactPath("coverage", Source.path, ".instructions.txt");
  return Paths;
}

Error emitCoverageReport(const AdvisorConfig &Config,
                         const CoveragePaths &Paths) {
  if (!sys::fs::exists(Paths.Profraw))
    return Error::success();

  if (auto Err =
          CoverageProcessor::mergeRawProfile(Paths.Profraw, Paths.Profdata))
    return Err;
  if (auto Err = CoverageProcessor::exportCoverageReport(
          Paths.InstrumentedExe, Paths.Profdata, Paths.Report))
    return Err;
  return Error::success();
}

std::optional<std::string>
queryCompilerResourceDir(const AdvisorConfig &Config,
                         llvm::StringRef CompilerPath) {
  llvm::SmallVector<std::string, 8> Args = {"-print-resource-dir"};
  auto RunOrErr = ProcessRunner::run(CompilerPath, Args, Config.getTimeout());
  if (!RunOrErr) {
    llvm::consumeError(RunOrErr.takeError());
    return std::nullopt;
  }
  if (RunOrErr->exitCode != 0)
    return std::nullopt;

  llvm::StringRef ResourceDir = llvm::StringRef(RunOrErr->stdout).trim();
  if (ResourceDir.empty())
    return std::nullopt;
  return ResourceDir.str();
}

Error emitCFGGraph(StringRef BitcodePath, StringRef OutputFile) {
  LLVMContext Context;
  SMDiagnostic Err;
  auto Module = parseIRFile(BitcodePath, Err, Context);
  if (!Module)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Failed to parse bitcode: " +
                                 Err.getMessage().str());

  for (auto &Func : *Module) {
    if (Func.isDeclaration())
      continue;

    DOTFuncInfo Info(&Func);
    std::error_code EC;
    raw_fd_ostream OS(OutputFile, EC, sys::fs::OF_Text);
    if (EC)
      return createStringError(EC, "Failed to open CFG output");
    llvm::WriteGraph(OS, &Info, /*ShortNames=*/false,
                     Func.getName().empty() ? "cfg" : Func.getName());
    return Error::success();
  }

  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "Module contains no functions for CFG export");
}

class IncludeTreeCallbacks : public clang::PPCallbacks {
public:
  explicit IncludeTreeCallbacks(llvm::raw_ostream &OS) : OS(OS) {}

  void FileChanged(clang::SourceLocation,
                   clang::PPCallbacks::FileChangeReason Reason,
                   clang::SrcMgr::CharacteristicKind, clang::FileID) override {
    if (Reason == clang::PPCallbacks::EnterFile)
      ++Depth;
    else if (Reason == clang::PPCallbacks::ExitFile && Depth)
      --Depth;
  }

  void InclusionDirective(clang::SourceLocation HashLoc, const clang::Token &,
                          llvm::StringRef FileName, bool,
                          clang::CharSourceRange,
                          clang::OptionalFileEntryRef File,
                          llvm::StringRef, llvm::StringRef,
                          const clang::Module *, bool,
                          clang::SrcMgr::CharacteristicKind) override {
    OS.indent(Depth * 2);
    if (File)
      OS << File->getName() << "\n";
    else
      OS << FileName << " (missing)\n";
  }

private:
  llvm::raw_ostream &OS;
  unsigned Depth = 0;
};

class IncludeTreeAction : public clang::PreprocessOnlyAction {
public:
  explicit IncludeTreeAction(llvm::raw_ostream &OS) : OS(OS) {}

protected:
  bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
    CI.getPreprocessor().addPPCallbacks(
        std::make_unique<IncludeTreeCallbacks>(OS));
    return clang::PreprocessOnlyAction::BeginSourceFileAction(CI);
  }

private:
  llvm::raw_ostream &OS;
};

class DependencyRecorder : public clang::DependencyCollector {
public:
  explicit DependencyRecorder(llvm::StringSet<> &Files) : Files(Files) {}

  bool sawDependency(llvm::StringRef Filename, bool, bool, bool,
                     bool IsMissing) override {
    if (!IsMissing)
      Files.insert(Filename);
    return true;
  }

private:
  llvm::StringSet<> &Files;
};

} // namespace

/// Open \p Path as an LLVM object file. This single-step helper replaces the
/// repeated MemoryBuffer::getFile + createObjectFile(MemBufferRef) pattern.
static Expected<object::OwningBinary<object::ObjectFile>>
openObjectFile(StringRef Path) {
  return object::ObjectFile::createObjectFile(Path);
}

static bool hasRegisteredCodegenTarget() {
  std::string Error;
  return llvm::TargetRegistry::lookupTarget(llvm::sys::getDefaultTargetTriple(),
                                            Error) != nullptr;
}

DataExtractor::DataExtractor(const AdvisorConfig &Config) : Config(Config) {}

Error DataExtractor::extractAllData(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  if (Config.getVerbose()) {
    outs() << "Extracting data for unit: " << Unit.getName() << "\n";
  }

  const bool HasCodegenTarget = hasRegisteredCodegenTarget();
  if (!HasCodegenTarget && Config.getVerbose()) {
    outs() << "Codegen target is not registered; skipping codegen-dependent "
              "extractors\n";
  }

  // Create extraction subdirectories
  sys::fs::create_directories(TempDir + "/ir");
  sys::fs::create_directories(TempDir + "/assembly");
  sys::fs::create_directories(TempDir + "/ast");
  sys::fs::create_directories(TempDir + "/preprocessed");
  sys::fs::create_directories(TempDir + "/include-tree");
  sys::fs::create_directories(TempDir + "/dependencies");
  sys::fs::create_directories(TempDir + "/debug");
  sys::fs::create_directories(TempDir + "/static-analyzer");
  sys::fs::create_directories(TempDir + "/diagnostics");
  sys::fs::create_directories(TempDir + "/coverage");
  sys::fs::create_directories(TempDir + "/time-trace");
  sys::fs::create_directories(TempDir + "/runtime-trace");
  sys::fs::create_directories(TempDir + "/binary-analysis");
  sys::fs::create_directories(TempDir + "/pgo");
  sys::fs::create_directories(TempDir + "/ftime-report");
  sys::fs::create_directories(TempDir + "/version-info");
  sys::fs::create_directories(TempDir + "/sources");

  if (HasCodegenTarget) {
    if (auto Err = extractIR(Unit, TempDir))
      return Err;
    if (auto Err = extractAssembly(Unit, TempDir))
      return Err;
  }
  if (auto Err = extractAST(Unit, TempDir))
    return Err;
  if (auto Err = extractPreprocessed(Unit, TempDir))
    return Err;
  if (auto Err = extractIncludeTree(Unit, TempDir))
    return Err;
  if (auto Err = extractDependencies(Unit, TempDir))
    return Err;
  if (HasCodegenTarget)
    if (auto Err = extractDebugInfo(Unit, TempDir))
      return Err;
  if (auto Err = extractStaticAnalysis(Unit, TempDir))
    return Err;
  if (auto Err = extractMacroExpansion(Unit, TempDir))
    return Err;
  if (auto Err = extractCompilationPhases(Unit, TempDir))
    return Err;
  if (auto Err = extractFTimeReport(Unit, TempDir))
    return Err;
  if (auto Err = extractVersionInfo(Unit, TempDir))
    return Err;
  if (auto Err = extractSources(Unit, TempDir))
    return Err;

  // Run additional extractors
  auto requiresCodegenTarget = [&](ExtractorMethod Method) {
    return Method == &DataExtractor::extractCoverage ||
           Method == &DataExtractor::extractTimeTrace ||
           Method == &DataExtractor::extractBinarySize ||
           Method == &DataExtractor::extractSymbols ||
           Method == &DataExtractor::extractObjdump ||
           Method == &DataExtractor::extractXRay ||
           Method == &DataExtractor::extractOptDot;
  };

  for (size_t i = 0; i < numExtractors; ++i) {
    const auto &extractor = extractors[i];
    if (!HasCodegenTarget && requiresCodegenTarget(extractor.method))
      continue;
    if (auto Err = (this->*extractor.method)(Unit, TempDir)) {
      if (Config.getVerbose()) {
        errs() << extractor.name
               << " extraction failed: " << toString(std::move(Err)) << "\n";
      }
    }
  }

  return Error::success();
}

llvm::SmallVector<std::string, 8>
DataExtractor::getBaseCompilerArgs(const CompilationUnitInfo &UnitInfo) const {
  llvm::SmallVector<std::string, 8> BaseArgs;

  // Preserve relevant compile flags and handle paired flags that forward
  // arguments to specific toolchains (e.g. OpenMP target flags).
  for (size_t I = 0; I < UnitInfo.compileFlags.size(); ++I) {
    const std::string &Flag = UnitInfo.compileFlags[I];

    // Handle paired forwarding flags that must precede their next argument.
    // Example: -Xopenmp-target -march=sm_70
    if (StringRef(Flag) == "-Xopenmp-target" ||
        StringRef(Flag).starts_with("-Xopenmp-target=")) {
      BaseArgs.push_back(Flag);
      // If the flag is the two-argument form, also copy the next arg if
      // present.
      if (StringRef(Flag) == "-Xopenmp-target" &&
          I + 1 < UnitInfo.compileFlags.size()) {
        BaseArgs.push_back(UnitInfo.compileFlags[I + 1]);
        ++I; // consume the next argument
      }
      continue;
    }

    // Commonly needed flags for reproducing preprocessing/IR/ASM
    if (StringRef(Flag).starts_with("-I") ||
        StringRef(Flag).starts_with("-D") ||
        StringRef(Flag).starts_with("-U") ||
        StringRef(Flag).starts_with("-std=") ||
        StringRef(Flag).starts_with("-m") ||
        StringRef(Flag).starts_with("-f") ||
        StringRef(Flag).starts_with("-W") ||
        StringRef(Flag).starts_with("-O")) {
      // Skip instrumentation/file-emission flags added by the executor
      if (StringRef(Flag).starts_with("-fsave-optimization-record") ||
          StringRef(Flag).starts_with("-fprofile-instr-generate") ||
          StringRef(Flag).starts_with("-fcoverage-mapping") ||
          StringRef(Flag).starts_with("-foptimization-record-file")) {
        continue;
      }
      BaseArgs.push_back(Flag);
      continue;
    }

    // Preserve explicit target specification when present
    if (StringRef(Flag).starts_with("--target=") ||
        StringRef(Flag) == "-target") {
      BaseArgs.push_back(Flag);
      if (StringRef(Flag) == "-target" &&
          I + 1 < UnitInfo.compileFlags.size()) {
        BaseArgs.push_back(UnitInfo.compileFlags[I + 1]);
        ++I;
      }
      continue;
    }
  }

  return BaseArgs;
}

namespace {
class FileASTDumpAction : public clang::ASTFrontendAction {
public:
  FileASTDumpAction(std::unique_ptr<llvm::raw_ostream> OS,
                    clang::ASTDumpOutputFormat Format)
      : OutputStream(std::move(OS)), DumpFormat(Format) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override {
    return clang::CreateASTDumper(std::move(OutputStream), /*Filter=*/"",
                                  /*DumpDecls=*/true, /*Deserialize=*/false,
                                  /*DumpLookups=*/false,
                                  /*DumpDeclTypes=*/false, DumpFormat);
  }

private:
  std::unique_ptr<llvm::raw_ostream> OutputStream;
  clang::ASTDumpOutputFormat DumpFormat;
};
} // namespace

Error DataExtractor::runFrontendAction(
    const CompilationUnitInfo &UnitInfo, llvm::StringRef SourcePath,
    llvm::StringRef OutputFile, llvm::ArrayRef<std::string> ExtraArgs,
    llvm::function_ref<std::unique_ptr<clang::FrontendAction>()> ActionFactory,
    llvm::raw_ostream *DiagOS,
    llvm::function_ref<void(clang::CompilerInstance &)> SetupCompiler) {
  std::string CompilerPath = Config.getToolPath("clang");
  if (!UnitInfo.compilerPath.empty())
    CompilerPath = UnitInfo.compilerPath;
  if (CompilerPath.empty())
    CompilerPath = "clang";

  llvm::SmallVector<std::string, 32> DriverArgs;
  auto BaseArgs = getBaseCompilerArgs(UnitInfo);
  DriverArgs.append(BaseArgs.begin(), BaseArgs.end());
  if (!SourcePath.empty())
    DriverArgs.push_back(SourcePath.str());
  if (auto ResourceDir = queryCompilerResourceDir(Config, CompilerPath)) {
    DriverArgs.push_back("-resource-dir");
    DriverArgs.push_back(*ResourceDir);
  }
  DriverArgs.append(ExtraArgs.begin(), ExtraArgs.end());

  auto DiagOpts = std::make_unique<clang::DiagnosticOptions>();
  llvm::raw_ostream &DiagStream = DiagOS ? *DiagOS : llvm::errs();
  auto DiagPrinter = std::make_unique<clang::TextDiagnosticPrinter>(
      DiagStream, *DiagOpts);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(
      new clang::DiagnosticIDs());
  clang::DiagnosticsEngine Diags(DiagIDs, *DiagOpts, DiagPrinter.get(),
                                 /*ShouldOwnClient=*/false);

  auto Driver = std::make_unique<clang::driver::Driver>(
      CompilerPath, llvm::sys::getDefaultTargetTriple(), Diags);
  Driver->setCheckInputsExist(false);
  Driver->setTitle("llvm-advisor");

  llvm::SmallVector<const char *, 32> DriverArgv;
  DriverArgv.push_back(CompilerPath.c_str());
  for (const auto &Arg : DriverArgs)
    DriverArgv.push_back(Arg.c_str());

  std::unique_ptr<clang::driver::Compilation> Compilation(
      Driver->BuildCompilation(DriverArgv));
  if (!Compilation)
    return createStringError(
        std::errc::invalid_argument,
        "Failed to build driver compilation for frontend action");

  const clang::driver::Command *FrontendCmd = nullptr;
  for (const auto &Cmd : Compilation->getJobs()) {
    if (llvm::StringRef(Cmd.getCreator().getName()).starts_with("clang")) {
      FrontendCmd = &Cmd;
      break;
    }
  }

  if (!FrontendCmd)
    return createStringError(std::errc::invalid_argument,
                             "No clang frontend command available");

  llvm::SmallVector<const char *, 64> CC1Args;
  CC1Args.reserve(FrontendCmd->getArguments().size());
  for (const char *Arg : FrontendCmd->getArguments())
    CC1Args.push_back(Arg);

  auto Invocation = std::make_shared<clang::CompilerInvocation>();
  llvm::ArrayRef<const char *> ArgRef(CC1Args);
  if (!clang::CompilerInvocation::CreateFromArgs(*Invocation, ArgRef, Diags))
    return createStringError(std::errc::invalid_argument,
                             "Failed to build compiler invocation");

  clang::CompilerInstance Clang(std::move(Invocation));
  Clang.createDiagnostics(DiagPrinter.release(), true);
  if (!Clang.hasDiagnostics())
    return createStringError(std::errc::invalid_argument,
                             "Failed to create diagnostics engine");

  Clang.createFileManager();
  Clang.createSourceManager();

  if (SetupCompiler)
    SetupCompiler(Clang);

  Clang.getFrontendOpts().OutputFile = OutputFile.str();

  auto Action = ActionFactory();
  if (!Action)
    return createStringError(std::errc::invalid_argument,
                             "Failed to construct frontend action");

  if (!Clang.ExecuteAction(*Action))
    return createStringError(std::errc::io_error,
                             "Frontend action execution failed");

  return Error::success();
}

Error DataExtractor::extractIR(CompilationUnit &Unit, llvm::StringRef TempDir) {
  (void)TempDir;
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string OutputFile = Unit.makeArtifactPath("ir", Source.path, ".ll");

    llvm::SmallVector<std::string, 4> ExtraArgs;
    ExtraArgs.push_back("-c");

    auto Err = runFrontendAction(
        Unit.getInfo(), Source.path, OutputFile, ExtraArgs,
        []() -> std::unique_ptr<clang::FrontendAction> {
          return std::make_unique<clang::EmitLLVMOnlyAction>();
        });
    if (Err) {
      if (Config.getVerbose())
        errs() << "Failed to extract IR for " << Source.path << ": "
               << toString(std::move(Err)) << "\n";
      continue;
    }

    if (sys::fs::exists(OutputFile))
      Unit.addGeneratedFile("ir", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractAssembly(CompilationUnit &Unit,
                                     llvm::StringRef TempDir) {
  (void)TempDir;
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string OutputFile =
        Unit.makeArtifactPath("assembly", Source.path, ".s");

    llvm::SmallVector<std::string, 4> ExtraArgs;
    ExtraArgs.push_back("-c");

    auto Err = runFrontendAction(
        Unit.getInfo(), Source.path, OutputFile, ExtraArgs,
        []() -> std::unique_ptr<clang::FrontendAction> {
          return std::make_unique<clang::EmitAssemblyAction>();
        });
    if (Err) {
      if (Config.getVerbose())
        errs() << "Failed to extract assembly for " << Source.path << ": "
               << toString(std::move(Err)) << "\n";
      continue;
    }

    if (sys::fs::exists(OutputFile))
      Unit.addGeneratedFile("assembly", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractAST(CompilationUnit &Unit,
                                llvm::StringRef TempDir) {
  (void)TempDir;
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string OutputFile = Unit.makeArtifactPath("ast", Source.path, ".ast");
    std::error_code EC;
    auto OS = std::make_unique<llvm::raw_fd_ostream>(OutputFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open AST file " << OutputFile << ": "
               << EC.message() << "\n";
      continue;
    }

    auto ActionHolder =
        std::make_unique<FileASTDumpAction>(std::move(OS), clang::ADOF_Default);
    auto Err = runFrontendAction(Unit.getInfo(), Source.path, /*OutputFile*/ "",
                                 {"-fsyntax-only"},
                                 [Action = std::move(ActionHolder)]() mutable {
                                   return std::move(Action);
                                 });
    if (Err) {
      if (Config.getVerbose())
        errs() << "Failed to dump AST for " << Source.path << ": "
               << toString(std::move(Err)) << "\n";
      continue;
    }

    Unit.addGeneratedFile("ast", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractPreprocessed(CompilationUnit &Unit,
                                         llvm::StringRef TempDir) {
  (void)TempDir;
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    llvm::StringRef Ext = (Source.language == "C++") ? ".ii" : ".i";
    std::string OutputFile =
        Unit.makeArtifactPath("preprocessed", Source.path, Ext);

    llvm::SmallVector<std::string, 2> ExtraArgs = {"-E"};
    auto Err = runFrontendAction(
        Unit.getInfo(), Source.path, OutputFile, ExtraArgs,
        []() -> std::unique_ptr<clang::FrontendAction> {
          return std::make_unique<clang::PrintPreprocessedAction>();
        });
    if (Err) {
      if (Config.getVerbose())
        errs() << "Failed to extract preprocessed output for " << Source.path
               << ": " << toString(std::move(Err)) << "\n";
      continue;
    }

    Unit.addGeneratedFile("preprocessed", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractIncludeTree(CompilationUnit &Unit,
                                        llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile = std::string(TempDir) + "/include-tree/" +
                             sys::path::stem(source.path).str() +
                             ".include.txt";

    std::error_code EC;
    raw_fd_ostream OS(OutputFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open include tree file: " << EC.message() << "\n";
      continue;
    }

    if (auto Err = runFrontendAction(
            Unit.getInfo(), source.path, "", {"-fsyntax-only"},
            [&]() -> std::unique_ptr<clang::FrontendAction> {
              return std::make_unique<IncludeTreeAction>(OS);
            })) {
      if (Config.getVerbose())
        errs() << "Include tree extraction failed: " << toString(std::move(Err))
               << "\n";
      continue;
    }

    Unit.addGeneratedFile("include-tree", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractDependencies(CompilationUnit &Unit,
                                         llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile = std::string(TempDir) + "/dependencies/" +
                             sys::path::stem(source.path).str() + ".deps.txt";

    llvm::StringSet<> Files;
    auto Collector = std::make_shared<DependencyRecorder>(Files);
    auto Setup = [Collector](clang::CompilerInstance &CI) {
      CI.addDependencyCollector(Collector);
    };

    auto Err = runFrontendAction(
        Unit.getInfo(), source.path, "", {"-E"},
        []() -> std::unique_ptr<clang::FrontendAction> {
          return std::make_unique<clang::PreprocessOnlyAction>();
        },
        nullptr, Setup);
    if (Err) {
      if (Config.getVerbose())
        errs() << "Dependency extraction failed: " << toString(std::move(Err))
               << "\n";
      continue;
    }

    std::error_code EC;
    raw_fd_ostream OS(OutputFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to write dependencies: " << EC.message() << "\n";
      continue;
    }
    for (const auto &Entry : Files)
      OS << Entry.getKey() << "\n";
    Unit.addGeneratedFile("dependencies", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractDebugInfo(CompilationUnit &Unit,
                                      llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile = std::string(TempDir) + "/debug/" +
                             sys::path::stem(source.path).str() + ".debug.txt";
    std::string ObjectFile = std::string(TempDir) + "/debug/" +
                             sys::path::stem(source.path).str() + ".o";

    llvm::SmallVector<std::string, 8> ExtraArgs = {"-c", "-g"};
    auto Err =
        runFrontendAction(Unit.getInfo(), source.path, ObjectFile, ExtraArgs,
                          []() -> std::unique_ptr<clang::FrontendAction> {
                            return std::make_unique<clang::EmitObjAction>();
                          });
    if (Err) {
      if (Config.getVerbose())
        errs() << "Failed to build object for debug extraction: "
               << toString(std::move(Err)) << "\n";
      continue;
    }

    auto ObjectOrErr = openObjectFile(ObjectFile);
    if (!ObjectOrErr) {
      if (Config.getVerbose())
        errs() << "Failed to parse object for DWARF: "
               << toString(ObjectOrErr.takeError()) << "\n";
      continue;
    }

    auto Context = llvm::DWARFContext::create(*ObjectOrErr->getBinary());
    std::error_code EC;
    raw_fd_ostream OS(OutputFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open debug output file " << OutputFile << "\n";
      continue;
    }
    Context->dump(OS, llvm::DIDumpOptions());
    Unit.addGeneratedFile("debug", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractStaticAnalysis(CompilationUnit &Unit,
                                           llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string OutputFile = std::string(TempDir) + "/static-analyzer/" +
                             sys::path::stem(source.path).str() +
                             ".analysis.txt";

    std::error_code EC;
    raw_fd_ostream OS(OutputFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open analyzer output: " << EC.message() << "\n";
      continue;
    }

    if (auto Err = runFrontendAction(
            Unit.getInfo(), source.path, "",
            {"-Xclang", "-analyzer-output=text"},
            []() -> std::unique_ptr<clang::FrontendAction> {
              return std::make_unique<clang::ento::AnalysisAction>();
            },
            &OS)) {
      if (Config.getVerbose())
        errs() << "Static analyzer failed: " << toString(std::move(Err))
               << "\n";
      continue;
    }

    Unit.addGeneratedFile("static-analyzer", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractMacroExpansion(CompilationUnit &Unit,
                                           llvm::StringRef TempDir) {
  (void)TempDir;
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string OutputFile = Unit.makeArtifactPath(
        "macro-expansion", Source.path,
        (Source.language == "C++") ? ".macro.ii" : ".macro.i");

    llvm::SmallVector<std::string, 4> ExtraArgs = {"-E", "-dM"};
    auto Err = runFrontendAction(
        Unit.getInfo(), Source.path, OutputFile, ExtraArgs,
        []() -> std::unique_ptr<clang::FrontendAction> {
          return std::make_unique<clang::PrintPreprocessedAction>();
        });
    if (Err) {
      if (Config.getVerbose())
        errs() << "Failed to extract macro expansion for " << Source.path
               << ": " << toString(std::move(Err)) << "\n";
      continue;
    }

    Unit.addGeneratedFile("macro-expansion", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractCompilationPhases(CompilationUnit &Unit,
                                              llvm::StringRef TempDir) {
  auto DriverCtx = createDriverContext(Config, Unit.getInfo());
  if (!DriverCtx)
    return Error::success();

  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile =
        (TempDir + "/debug/" + sys::path::stem(source.path).str() +
         ".phases.txt")
            .str();
    std::string bindingsFile =
        (TempDir + "/debug/" + sys::path::stem(source.path).str() +
         ".bindings.txt")
            .str();

    std::error_code BindEC;
    raw_fd_ostream BindOS(bindingsFile, BindEC);
    if (!BindEC) {
      BindOS << "Driver job bindings for: " << source.path << "\n\n";
      for (const auto &Cmd : DriverCtx->Compilation->getJobs()) {
        const auto *JA =
            llvm::dyn_cast<clang::driver::JobAction>(&Cmd.getSource());
        if (!JA)
          continue;
        BindOS << Cmd.getCreator().getName() << " -> " << Cmd.getExecutable()
               << "\n";
        BindOS << "  Inputs:";
        for (const auto &Info : Cmd.getInputInfos()) {
          if (Info.isFilename())
            BindOS << " " << Info.getFilename();
        }
        BindOS << "\n  Outputs:";
        for (const auto &Out : Cmd.getOutputFilenames())
          BindOS << " " << Out;
        BindOS << "\n\n";
      }
      Unit.addGeneratedFile("compilation-phases", bindingsFile);
    }

    std::error_code VerbEC;
    raw_fd_ostream VerbOS(outputFile, VerbEC);
    if (!VerbEC) {
      VerbOS << "Verbose compilation breakdown for: " << source.path << "\n";
      VerbOS << "Target triple: "
             << DriverCtx->Compilation->getDefaultToolChain().getTripleString()
             << "\n\n";
      for (const auto &Cmd : DriverCtx->Compilation->getJobs()) {
        VerbOS << Cmd.getExecutable();
        for (const char *Arg : Cmd.getArguments())
          VerbOS << ' ' << Arg;
        VerbOS << "\n";
      }
      Unit.addGeneratedFile("compilation-phases", outputFile);
    }
  }
  return Error::success();
}

Error DataExtractor::extractFTimeReport(CompilationUnit &Unit,
                                        llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string outputFile = std::string(TempDir) + "/ftime-report/" +
                             sys::path::stem(source.path).str() + ".ftime.txt";

    std::string ReportBuffer;
    llvm::raw_string_ostream ReportStream(ReportBuffer);
    llvm::SmallVector<std::string, 8> ExtraArgs = {"-fsyntax-only",
                                                   "-ftime-report"};
    auto Err = runFrontendAction(
        Unit.getInfo(), source.path, "", ExtraArgs,
        []() -> std::unique_ptr<clang::FrontendAction> {
          return std::make_unique<clang::SyntaxOnlyAction>();
        },
        &ReportStream);
    ReportStream.flush();
    if (Err) {
      if (Config.getVerbose())
        errs() << "ftime-report failed: " << toString(std::move(Err)) << "\n";
      continue;
    }

    std::error_code EC;
    raw_fd_ostream OS(outputFile, EC);
    if (!EC) {
      OS << "FTIME REPORT:\n" << ReportBuffer;
      Unit.addGeneratedFile("ftime-report", outputFile);
      if (Config.getVerbose())
        outs() << "FTime Report: " << outputFile << "\n";
    }
  }
  return Error::success();
}

Error DataExtractor::extractVersionInfo(CompilationUnit &Unit,
                                        llvm::StringRef TempDir) {
  std::string OutputFile =
      std::string(TempDir) + "/version-info/clang-version.txt";

  std::error_code EC;
  raw_fd_ostream OS(OutputFile, EC);
  if (!EC) {
    OS << clang::getClangFullVersion() << "\n";
    Unit.addGeneratedFile("version-info", OutputFile);
    if (Config.getVerbose())
      outs() << "Version Info: " << OutputFile << "\n";
  }
  return Error::success();
}

Error DataExtractor::extractASTJSON(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  (void)TempDir;
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string OutputFile =
        Unit.makeArtifactPath("ast", Source.path, ".ast.json");
    std::error_code EC;
    auto OS = std::make_unique<llvm::raw_fd_ostream>(OutputFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open AST JSON file " << OutputFile << ": "
               << EC.message() << "\n";
      continue;
    }

    auto ActionHolder =
        std::make_unique<FileASTDumpAction>(std::move(OS), clang::ADOF_JSON);
    auto Err =
        runFrontendAction(Unit.getInfo(), Source.path, "", {"-fsyntax-only"},
                          [Action = std::move(ActionHolder)]() mutable {
                            return std::move(Action);
                          });
    if (Err) {
      if (Config.getVerbose())
        errs() << "Failed to dump AST JSON for " << Source.path << ": "
               << toString(std::move(Err)) << "\n";
      continue;
    }

    Unit.addGeneratedFile("ast-json", OutputFile);
    if (Config.getVerbose())
      outs() << "AST JSON: " << OutputFile << "\n";
  }
  return Error::success();
}

Error DataExtractor::extractDiagnostics(CompilationUnit &Unit,
                                        llvm::StringRef TempDir) {
  (void)TempDir;
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string OutputFile =
        Unit.makeArtifactPath("diagnostics", Source.path, ".diagnostics.txt");

    std::error_code EC;
    raw_fd_ostream OS(OutputFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open diagnostics file " << OutputFile << ": "
               << EC.message() << "\n";
      continue;
    }

    OS << "Diagnostics for: " << Source.path << "\n";
    auto runDiag = [&](llvm::ArrayRef<std::string> ExtraArgs,
                       llvm::StringRef Title) -> Error {
      llvm::SmallString<4096> Buffer;
      llvm::raw_svector_ostream Stream(Buffer);
      if (auto Err = runFrontendAction(
              Unit.getInfo(), Source.path, "", ExtraArgs,
              []() -> std::unique_ptr<clang::FrontendAction> {
                return std::make_unique<clang::SyntaxOnlyAction>();
              },
              &Stream))
        return Err;
      OS << Title << "\n" << Buffer << "\n";
      return Error::success();
    };

    llvm::SmallVector<std::string, 8> PrimaryArgs = {
        "-fdiagnostics-parseable-fixits", "-fdiagnostics-absolute-paths",
        "-Wall", "-Wextra", "-fsyntax-only"};
    if (auto Err = runDiag(PrimaryArgs, "Primary diagnostics"))
      if (Config.getVerbose())
        errs() << "Primary diagnostics failed for " << Source.path << ": "
               << toString(std::move(Err)) << "\n";

    llvm::SmallVector<std::string, 8> Extra = {
        "-Weverything", "-Wno-c++98-compat", "-Wno-c++98-compat-pedantic",
        "-fsyntax-only"};
    if (auto Err = runDiag(Extra, "Extended diagnostics"))
      if (Config.getVerbose())
        errs() << "Extended diagnostics failed for " << Source.path << ": "
               << toString(std::move(Err)) << "\n";

    Unit.addGeneratedFile("diagnostics", OutputFile);
  }
  return Error::success();
}

Error DataExtractor::extractCoverage(CompilationUnit &Unit,
                                     llvm::StringRef TempDir) {
  auto processExistingProfiles = [&](const SourceFile &Source) {
    CoveragePaths Paths = computeCoverageArtifacts(Unit, Source);
    if (auto Err = emitCoverageReport(Config, Paths)) {
      if (Config.getVerbose())
        errs() << "Coverage report generation failed: "
               << toString(std::move(Err)) << "\n";
      return;
    }
    if (sys::fs::exists(Paths.Profdata))
      Unit.addGeneratedFile("coverage", Paths.Profdata);
    if (sys::fs::exists(Paths.Report))
      Unit.addGeneratedFile("coverage", Paths.Report);
  };

  for (const auto &source : Unit.getInfo().sources)
    processExistingProfiles(source);

  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    CoveragePaths Paths = computeCoverageArtifacts(Unit, source);
    std::string SourceStem = sys::path::stem(source.path).str();
    // Produce an instrumented *object* file; linking a single TU into an
    // executable would fail for the vast majority of real translation units
    // (no main function). The object file is sufficient for coverage-map
    // analysis and the instructions file tells the user how to link and run.
    std::string InstrumentedObj =
        std::string(TempDir) + "/" + SourceStem + "_cov.o";

    if (auto CompileErr = runFrontendAction(
            Unit.getInfo(), source.path, InstrumentedObj,
            {"-fprofile-instr-generate", "-fcoverage-mapping", "-c"},
            []() -> std::unique_ptr<clang::FrontendAction> {
              return std::make_unique<clang::EmitObjAction>();
            })) {
      if (Config.getVerbose())
        errs() << "Failed to compile with coverage instrumentation: "
               << toString(std::move(CompileErr)) << "\n";
      continue;
    }

    if (!sys::fs::exists(InstrumentedObj))
      continue;

    // Persist the instrumented object for manual linking/execution.
    if (auto CopyErr =
            FileManager::copyFile(InstrumentedObj, Paths.InstrumentedExe)) {
      if (Config.getVerbose())
        errs() << "Failed to preserve instrumented object: "
               << toString(std::move(CopyErr)) << "\n";
      sys::fs::remove(InstrumentedObj);
      continue;
    }
    sys::fs::remove(InstrumentedObj);

    std::error_code EC;
    raw_fd_ostream OS(Paths.Instructions, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to write coverage instructions: " << EC.message()
               << "\n";
      continue;
    }

    OS << "Instrumented object: " << Paths.InstrumentedExe << "\n";
    OS << "To collect coverage data manually:\n";
    OS << "  1. Link the object into an executable (add any required "
          "libraries):\n"
       << "     clang " << Paths.InstrumentedExe
       << " -fprofile-instr-generate -o <binary>\n";
    OS << "  2. Run the executable with LLVM_PROFILE_FILE=" << Paths.Profraw
       << "\n";
    OS << "  3. Merge profiles:\n     " << Config.getToolPath("llvm-profdata")
       << " merge -sparse " << Paths.Profraw << " -o " << Paths.Profdata
       << "\n";
    OS << "  4. Export coverage:\n     " << Config.getToolPath("llvm-cov")
       << " export <binary>"
       << " -instr-profile=" << Paths.Profdata << " -format=json > "
       << Paths.Report << "\n";

    Unit.addGeneratedFile("coverage", Paths.InstrumentedExe);
    Unit.addGeneratedFile("coverage", Paths.Instructions);

    if (auto Err = emitCoverageReport(Config, Paths)) {
      if (Config.getVerbose())
        errs() << "Coverage export failed: " << toString(std::move(Err))
               << "\n";
    } else {
      if (sys::fs::exists(Paths.Profdata))
        Unit.addGeneratedFile("coverage", Paths.Profdata);
      if (sys::fs::exists(Paths.Report)) {
        Unit.addGeneratedFile("coverage", Paths.Report);
        if (Config.getVerbose())
          outs() << "Coverage report generated: " << Paths.Report << "\n";
      }
    }
  }
  return Error::success();
}

Error DataExtractor::extractTimeTrace(CompilationUnit &Unit,
                                      llvm::StringRef TempDir) {
  for (const auto &source : Unit.getInfo().sources) {
    if (source.isHeader)
      continue;

    std::string SourceStem = sys::path::stem(source.path).str();
    std::string TraceFile =
        std::string(TempDir) + "/time-trace/" + SourceStem + ".trace.json";
    std::string TempObject =
        std::string(TempDir) + "/" + SourceStem + "_trace.o";

    auto Setup = [&TraceFile](clang::CompilerInstance &CI) {
      CI.getFrontendOpts().TimeTracePath = TraceFile;
      CI.getFrontendOpts().TimeTraceGranularity = 0;
    };

    auto TraceErr = runFrontendAction(
        Unit.getInfo(), source.path, TempObject, {"-c", "-ftime-trace"},
        []() -> std::unique_ptr<clang::FrontendAction> {
          return std::make_unique<clang::EmitObjAction>();
        },
        nullptr, Setup);

    sys::fs::remove(TempObject);

    if (TraceErr) {
      if (Config.getVerbose())
        errs() << "Time trace failed for " << source.path << ": "
               << toString(std::move(TraceErr)) << "\n";
      continue;
    }

    if (sys::fs::exists(TraceFile)) {
      Unit.addGeneratedFile("time-trace", TraceFile);
      if (Config.getVerbose())
        outs() << "Time trace: " << TraceFile << "\n";
    } else if (Config.getVerbose()) {
      errs() << "Time trace file not generated for " << source.path << "\n";
    }
  }
  return Error::success();
}

Error DataExtractor::extractRuntimeTrace(CompilationUnit &Unit,
                                         llvm::StringRef TempDir) {

  // Check for OpenMP offloading flags
  bool HasOffloading = false;
  for (const auto &Flag : Unit.getInfo().compileFlags) {
    StringRef FlagStr(Flag);
    // Check for various OpenMP offloading indicators
    if (FlagStr.contains("offload") ||          // Generic offload flags
        FlagStr.contains("-fopenmp-targets") || // OpenMP target specification
        FlagStr.contains("-Xopenmp-target") ||  // OpenMP target-specific flags
        FlagStr.contains("nvptx") ||            // NVIDIA GPU targets
        FlagStr.contains("amdgcn") ||           // AMD GPU targets
        FlagStr.contains("-fopenmp")) { // Basic OpenMP (may have runtime)
      HasOffloading = true;
      break;
    }
  }

  if (!HasOffloading) {
    if (Config.getVerbose())
      outs() << "Runtime trace skipped - no OpenMP offloading flags detected\n";
    return Error::success();
  }

  if (Config.getVerbose())
    outs()
        << "OpenMP offloading detected, attempting runtime trace extraction\n";

  // Find executable name from compile flags
  std::string ExecutableStr = "a.out"; // Default executable name
  for (size_t I = 0; I < Unit.getInfo().compileFlags.size(); ++I) {
    if (Unit.getInfo().compileFlags[I] == "-o" &&
        I + 1 < Unit.getInfo().compileFlags.size()) {
      ExecutableStr = Unit.getInfo().compileFlags[I + 1];
      break;
    }
  }

  llvm::StringRef Executable = ExecutableStr;

  if (Config.getVerbose())
    outs() << "Looking for executable: " << Executable << "\n";

  if (!sys::fs::exists(Executable)) {
    if (Config.getVerbose()) {
      outs() << "Runtime trace skipped - executable not found: " << Executable
             << "\n";
      outs() << "Note: Executable is needed to generate runtime profile data\n";
      outs() << "Checked current directory for: " << Executable << "\n";

      // List current directory contents for debugging
      outs() << "Current directory contents:\n";
      std::error_code EC;
      for (sys::fs::directory_iterator I(".", EC), E; I != E && !EC;
           I.increment(EC)) {
        outs() << "  " << I->path() << "\n";
      }
    }

    return Error::success();
  }

  // Create a trace directory
  SmallString<256> TraceDir;
  sys::path::append(TraceDir, TempDir, "runtime-trace");
  if (auto ec = sys::fs::create_directories(TraceDir, true)) {
    if (Config.getVerbose())
      outs() << "Warning: Failed to create runtime trace directory: "
             << ec.message() << "\n";
  }

  // Prepare trace file
  SmallString<256> TraceFile;
  sys::path::append(TraceFile, TraceDir, "profile.json");

  if (Config.getVerbose()) {
    outs() << "Running runtime trace: " << Executable << "\n";
    outs() << "Trace file: " << TraceFile << "\n";
  }

  // Set environment variable for OpenMP target profiling
  SmallVector<std::string, 8> Env;
  Env.push_back((Twine("LIBOMPTARGET_PROFILE=") + TraceFile).str());

  if (Config.getVerbose()) {
    outs() << "Setting environment: LIBOMPTARGET_PROFILE=" << TraceFile << "\n";
    outs() << "Executing: " << Executable << "\n";
  }

  // Run executable with profiling environment
  auto Result =
      ProcessRunner::runWithEnv(Executable, {}, Env, Config.getTimeout());

  if (Config.getVerbose()) {
    if (Result) {
      outs() << "Runtime trace completed with exit code: " << Result->exitCode
             << "\n";

      if (!Result->stdout.empty())
        outs() << "STDOUT: " << Result->stdout << "\n";
      if (!Result->stderr.empty())
        outs() << "STDERR: " << Result->stderr << "\n";
    } else {
      outs() << "Runtime trace failed: " << toString(Result.takeError())
             << "\n";
    }
  }

  // Register trace file if generated
  if (sys::fs::exists(TraceFile)) {
    Unit.addGeneratedFile("runtime-trace", TraceFile.str());
    if (Config.getVerbose())
      outs() << "Runtime trace saved: " << TraceFile << "\n";
  } else {
    if (Config.getVerbose()) {
      outs() << "Runtime trace failed - no trace file generated at: "
             << TraceFile << "\n";
      outs() << "This may happen if:\n";
      outs() << "  1. The program didn't use OpenMP target offloading\n";
      outs() << "  2. The runtime doesn't support LIBOMPTARGET_PROFILE\n";
      outs() << "  3. The program crashed before generating profile data\n";
    }
  }

  return Error::success();
}

Error DataExtractor::extractSARIF(CompilationUnit &Unit,
                                  llvm::StringRef TempDir) {
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string SourceStem = sys::path::stem(Source.path).str();
    std::string SarifFile =
        (TempDir + "/static-analyzer/" + SourceStem + ".sarif").str();

    if (auto Err =
            runFrontendAction(Unit.getInfo(), Source.path, SarifFile,
                              {"-Xclang", "-analyzer-output=sarif"},
                              []() -> std::unique_ptr<clang::FrontendAction> {
                                return std::make_unique<
                                    clang::ento::AnalysisAction>();
                              })) {
      if (Config.getVerbose())
        errs() << "Failed to extract SARIF static analysis for " << Source.path
               << ": " << toString(std::move(Err)) << "\n";
      continue;
    }

    uint64_t FileSize = 0;
    if (!sys::fs::file_size(SarifFile, FileSize) && FileSize > 0) {
      Unit.addGeneratedFile("static-analysis-sarif", SarifFile);
      if (Config.getVerbose())
        outs() << "SARIF static analysis extracted: " << SarifFile << "\n";
    } else if (Config.getVerbose()) {
      outs() << "SARIF file empty for " << Source.path << "\n";
    }
  }
  return Error::success();
}

Error DataExtractor::extractBinarySize(CompilationUnit &Unit,
                                       llvm::StringRef TempDir) {
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string SourceStem = sys::path::stem(Source.path).str();
    std::string ObjectFile =
        std::string(TempDir) + "/" + SourceStem + "_size.o";
    std::string SizeFile =
        std::string(TempDir) + "/binary-analysis/" + SourceStem + ".size.txt";

    if (auto CompileErr =
            runFrontendAction(Unit.getInfo(), Source.path, ObjectFile, {"-c"},
                              []() -> std::unique_ptr<clang::FrontendAction> {
                                return std::make_unique<clang::EmitObjAction>();
                              })) {
      if (Config.getVerbose())
        errs() << "Failed to compile for size analysis: "
               << toString(std::move(CompileErr)) << "\n";
      continue;
    }

    auto ObjOrErr = openObjectFile(ObjectFile);
    if (!ObjOrErr) {
      if (Config.getVerbose())
        errs() << "Failed to open object for size analysis: "
               << toString(ObjOrErr.takeError()) << "\n";
      sys::fs::remove(ObjectFile);
      continue;
    }

    std::error_code EC;
    raw_fd_ostream OS(SizeFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to write size report: " << EC.message() << "\n";
      sys::fs::remove(ObjectFile);
      continue;
    }

    OS << "Binary size analysis for: " << Source.path << "\n";
    OS << "Generated from object file: " << ObjectFile << "\n\n";
    OS << llvm::formatv("{0,-32} {1,12}\n", "Section", "Bytes");

    uint64_t TotalSize = 0;
    for (const auto &Section : ObjOrErr->getBinary()->sections()) {
      if (Section.isVirtual())
        continue;
      auto NameOrErr = Section.getName();
      llvm::StringRef Name = NameOrErr ? *NameOrErr : "<unnamed>";
      uint64_t Size = Section.getSize();
      TotalSize += Size;
      OS << llvm::formatv("{0,-32} {1,12}\n", Name, Size);
    }
    OS << llvm::formatv("{0,-32} {1,12}\n", "TOTAL", TotalSize);

    Unit.addGeneratedFile("binary-size", SizeFile);
    if (Config.getVerbose())
      outs() << "Binary size: " << SizeFile << "\n";

    sys::fs::remove(ObjectFile);
  }
  return Error::success();
}

Error DataExtractor::extractPGO(CompilationUnit &Unit,
                                llvm::StringRef TempDir) {
  // Look for existing profile raw data file from compilation
  std::string profrawFile = std::string(TempDir) + "/profile.profraw";

  if (sys::fs::exists(profrawFile)) {
    std::string profileFile = (TempDir + "/pgo/merged.profdata").str();
    std::string profileText = (TempDir + "/pgo/profile.txt").str();
    std::string profileJson = (TempDir + "/pgo/profile.json").str();

    sys::fs::create_directories((llvm::Twine(TempDir) + "/pgo").str());

    if (auto Err =
            CoverageProcessor::mergeRawProfile(profrawFile, profileFile)) {
      if (Config.getVerbose())
        errs() << "Failed to merge PGO profile data: "
               << toString(std::move(Err)) << "\n";
      return Error::success();
    }

    if (auto Err = CoverageProcessor::summarizeProfile(profileFile, profileText,
                                                       profileJson)) {
      if (Config.getVerbose())
        errs() << "Failed to summarize PGO profile: "
               << toString(std::move(Err)) << "\n";
    } else {
      Unit.addGeneratedFile("pgo-profile", profileText);
      Unit.addGeneratedFile("pgo-profile-json", profileJson);
      if (Config.getVerbose()) {
        outs() << "PGO profile data extracted: " << profileText << "\n";
        outs() << "PGO profile JSON extracted: " << profileJson << "\n";
      }
    }
    sys::fs::remove(profileFile);
  } else if (Config.getVerbose()) {
    outs() << "No PGO profile data found to extract\n";
  }

  return Error::success();
}

Error DataExtractor::extractSymbols(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string SourceStem = sys::path::stem(Source.path).str();
    std::string ObjectFile =
        std::string(TempDir) + "/" + SourceStem + "_symbols.o";
    std::string SymbolsFile = std::string(TempDir) + "/binary-analysis/" +
                              SourceStem + ".symbols.txt";

    if (auto CompileErr =
            runFrontendAction(Unit.getInfo(), Source.path, ObjectFile, {"-c"},
                              []() -> std::unique_ptr<clang::FrontendAction> {
                                return std::make_unique<clang::EmitObjAction>();
                              })) {
      if (Config.getVerbose())
        errs() << "Failed to compile for symbol extraction: "
               << toString(std::move(CompileErr)) << "\n";
      continue;
    }

    std::error_code EC;
    raw_fd_ostream OS(SymbolsFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open symbol output: " << EC.message() << "\n";
      sys::fs::remove(ObjectFile);
      continue;
    }

    OS << "Symbol table for: " << Source.path << "\n";
    OS << "Generated from object file: " << ObjectFile << "\n\n";

    auto ObjOrErr = openObjectFile(ObjectFile);
    if (!ObjOrErr) {
      if (Config.getVerbose())
        errs() << "Failed to open object for symbols: "
               << toString(ObjOrErr.takeError()) << "\n";
      sys::fs::remove(ObjectFile);
      continue;
    }

    OS << llvm::formatv("{0:>16}  {1:-8}  {2}  {3}\n", "Address", "Type",
                        "Scope", "Name");
    for (const auto &Sym : ObjOrErr->getBinary()->symbols()) {
      auto NameOrErr = Sym.getName();
      if (!NameOrErr || NameOrErr->empty())
        continue;
      auto TypeOrErr = Sym.getType();
      if (!TypeOrErr)
        continue;
      uint64_t Address = 0;
      if (auto AddrOrErr = Sym.getValue())
        Address = *AddrOrErr;
      auto FlagsOrErr = Sym.getFlags();
      if (!FlagsOrErr)
        continue;
      unsigned Flags = *FlagsOrErr;

      llvm::StringRef TypeLabel;
      switch (*TypeOrErr) {
      case llvm::object::SymbolRef::ST_Function:
        TypeLabel = "FUNC";
        break;
      case llvm::object::SymbolRef::ST_Data:
        TypeLabel = "DATA";
        break;
      case llvm::object::SymbolRef::ST_Debug:
        TypeLabel = "DEBUG";
        break;
      case llvm::object::SymbolRef::ST_File:
        TypeLabel = "FILE";
        break;
      default:
        TypeLabel = "OTHER";
        break;
      }

      char Scope = ' ';
      if (Flags & llvm::object::SymbolRef::SF_Global)
        Scope = 'G';
      else if (Flags & llvm::object::SymbolRef::SF_Weak)
        Scope = 'W';
      else if (Flags & llvm::object::SymbolRef::SF_Undefined)
        Scope = 'U';

      OS << llvm::formatv("{0:016x}  {1,-8}  {2}  {3}\n", Address, TypeLabel,
                          Scope, *NameOrErr);
    }

    Unit.addGeneratedFile("symbols", SymbolsFile);
    if (Config.getVerbose())
      outs() << "Symbols: " << SymbolsFile << "\n";

    sys::fs::remove(ObjectFile);
  }
  return Error::success();
}

Error DataExtractor::extractObjdump(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string SourceStem = sys::path::stem(Source.path).str();
    std::string ObjectFile =
        std::string(TempDir) + "/" + SourceStem + "_objdump.o";
    std::string ObjdumpFile = std::string(TempDir) + "/binary-analysis/" +
                              SourceStem + ".objdump.txt";

    if (auto CompileErr =
            runFrontendAction(Unit.getInfo(), Source.path, ObjectFile, {"-c"},
                              []() -> std::unique_ptr<clang::FrontendAction> {
                                return std::make_unique<clang::EmitObjAction>();
                              })) {
      if (Config.getVerbose())
        errs() << "Failed to compile for objdump: "
               << toString(std::move(CompileErr)) << "\n";
      continue;
    }

    std::error_code EC;
    raw_fd_ostream OS(ObjdumpFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open objdump output: " << EC.message() << "\n";
      sys::fs::remove(ObjectFile);
      continue;
    }

    OS << "Object dump for: " << Source.path << "\n";
    OS << "Generated from object file: " << ObjectFile << "\n\n";

    auto ObjOrErr = openObjectFile(ObjectFile);
    if (!ObjOrErr) {
      if (Config.getVerbose())
        errs() << "Failed to open object for objdump: "
               << toString(ObjOrErr.takeError()) << "\n";
      sys::fs::remove(ObjectFile);
      continue;
    }

    for (const auto &Sec : ObjOrErr->getBinary()->sections()) {
      if (!Sec.isText())
        continue;
      auto NameOrErr = Sec.getName();
      llvm::StringRef Name = NameOrErr ? *NameOrErr : "<text>";
      OS << "Section " << Name << "\n";
      if (auto Contents = Sec.getContents()) {
        const unsigned char *Data =
            reinterpret_cast<const unsigned char *>(Contents->bytes_begin());
        for (uint64_t I = 0; I < Contents->size();) {
          OS << llvm::formatv("  {0:08x}:", Sec.getAddress() + I);
          unsigned LineBytes = std::min<uint64_t>(16, Contents->size() - I);
          for (unsigned B = 0; B < LineBytes; ++B)
            OS << ' ' << llvm::format_hex_no_prefix(Data[I + B], 2, true);
          OS << "\n";
          I += LineBytes;
        }
      }
    }

    Unit.addGeneratedFile("objdump", ObjdumpFile);
    if (Config.getVerbose())
      outs() << "Objdump: " << ObjdumpFile << "\n";

    sys::fs::remove(ObjectFile);
  }
  return Error::success();
}

Error DataExtractor::extractXRay(CompilationUnit &Unit,
                                 llvm::StringRef TempDir) {
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string SourceStem = sys::path::stem(Source.path).str();
    std::string ObjectFile =
        std::string(TempDir) + "/" + SourceStem + "_xray.o";
    std::string XrayFile =
        std::string(TempDir) + "/binary-analysis/" + SourceStem + ".xray.txt";

    // Compile with XRay instrumentation to an object file (no linking required
    // to inspect the embedded XRay map sections).
    if (auto CompileErr = runFrontendAction(
            Unit.getInfo(), Source.path, ObjectFile,
            {"-fxray-instrument", "-fxray-instruction-threshold=1", "-c"},
            []() -> std::unique_ptr<clang::FrontendAction> {
              return std::make_unique<clang::EmitObjAction>();
            })) {
      if (Config.getVerbose())
        errs() << "Failed to compile with XRay for " << Source.path << ": "
               << toString(std::move(CompileErr)) << "\n";
      continue;
    }

    std::error_code EC;
    raw_fd_ostream OS(XrayFile, EC);
    if (EC) {
      if (Config.getVerbose())
        errs() << "Failed to open XRay output: " << EC.message() << "\n";
      sys::fs::remove(ObjectFile);
      continue;
    }

    OS << "XRay analysis for: " << Source.path << "\n";
    OS << "Generated from object file: " << ObjectFile << "\n\n";
    OS << "XRay instrumentation:\n";

    auto ObjOrErr = openObjectFile(ObjectFile);
    if (!ObjOrErr) {
      if (Config.getVerbose())
        errs() << "Failed to open XRay object: "
               << toString(ObjOrErr.takeError()) << "\n";
      sys::fs::remove(ObjectFile);
      continue;
    }

    for (const auto &Section : ObjOrErr->getBinary()->sections()) {
      auto NameOrErr = Section.getName();
      if (!NameOrErr || !NameOrErr->starts_with("xray"))
        continue;
      OS << "Section " << *NameOrErr << " size " << Section.getSize()
         << " bytes\n";
    }

    Unit.addGeneratedFile("xray", XrayFile);
    if (Config.getVerbose())
      outs() << "XRay: " << XrayFile << "\n";

    sys::fs::remove(ObjectFile);
  }
  return Error::success();
}

Error DataExtractor::extractOptDot(CompilationUnit &Unit,
                                   llvm::StringRef TempDir) {
  for (const auto &Source : Unit.getInfo().sources) {
    if (Source.isHeader)
      continue;

    std::string SourceStem = sys::path::stem(Source.path).str();
    std::string BitcodeFile = std::string(TempDir) + "/" + SourceStem + ".bc";
    std::string DotFile =
        std::string(TempDir) + "/ir/" + SourceStem + ".cfg.dot";

    // Emit bitcode in-process using the LLVM/Clang library.
    if (auto CompileErr =
            runFrontendAction(Unit.getInfo(), Source.path, BitcodeFile, {"-c"},
                              []() -> std::unique_ptr<clang::FrontendAction> {
                                return std::make_unique<clang::EmitBCAction>();
                              })) {
      if (Config.getVerbose())
        errs() << "Failed to emit bitcode for CFG DOT: "
               << toString(std::move(CompileErr)) << "\n";
      continue;
    }

    if (!sys::fs::exists(BitcodeFile))
      continue;

    if (auto Err = emitCFGGraph(BitcodeFile, DotFile)) {
      if (Config.getVerbose())
        errs() << "Failed to generate CFG DOT: " << toString(std::move(Err))
               << "\n";
    } else {
      Unit.addGeneratedFile("cfg-dot", DotFile);
      if (Config.getVerbose())
        outs() << "CFG DOT: " << DotFile << "\n";
    }
    sys::fs::remove(BitcodeFile);
  }
  return Error::success();
}

Error DataExtractor::extractSources(CompilationUnit &Unit,
                                    llvm::StringRef TempDir) {
  if (Config.getVerbose()) {
    outs() << "Extracting source files based on dependencies...\n";
  }

  // Create sources directory
  SmallString<256> SourcesDir;
  sys::path::append(SourcesDir, TempDir, "sources");
  if (auto EC = sys::fs::create_directories(SourcesDir)) {
    if (Config.getVerbose()) {
      outs() << "Warning: Failed to create sources directory: " << EC.message()
             << "\n";
    }
    return Error::success(); // Continue even if we can't create directory
  }

  // Find and parse dependencies files
  SmallString<256> depsDir;
  sys::path::append(depsDir, TempDir, "dependencies");

  if (!sys::fs::exists(depsDir)) {
    if (Config.getVerbose()) {
      outs() << "No dependencies directory found, skipping source extraction\n";
    }
    return Error::success();
  }

  std::error_code EC;
  for (sys::fs::directory_iterator I(depsDir, EC), E; I != E && !EC;
       I.increment(EC)) {
    StringRef filePath = I->path();
    if (!filePath.ends_with(".deps.txt")) {
      continue;
    }

    if (Config.getVerbose()) {
      outs() << "Processing dependencies file: " << filePath << "\n";
    }

    // Read and parse dependencies file
    auto bufferOrErr = MemoryBuffer::getFile(filePath);
    if (!bufferOrErr) {
      if (Config.getVerbose()) {
        outs() << "Warning: Failed to read dependencies file: " << filePath
               << "\n";
      }
      continue;
    }

    StringRef content = bufferOrErr.get()->getBuffer();
    SmallVector<StringRef, 16> lines;
    content.split(lines, '\n');

    for (StringRef line : lines) {
      line = line.trim();
      if (line.empty())
        continue;

      // Each line written by extractDependencies is a single file path.
      StringRef sourceFile = line;

      // Convert relative paths to absolute paths.
      SmallString<256> absoluteSourcePath;
      if (sys::path::is_absolute(sourceFile)) {
        absoluteSourcePath = sourceFile;
      } else {
        SmallString<256> currentDir;
        sys::fs::current_path(currentDir);
        absoluteSourcePath = currentDir;
        sys::path::append(absoluteSourcePath, sourceFile);
      }

      if (!sys::fs::exists(absoluteSourcePath)) {
        if (Config.getVerbose())
          outs() << "Warning: Source file not found: " << absoluteSourcePath
                 << "\n";
        continue;
      }

      SmallString<256> destPath;
      sys::path::append(destPath, SourcesDir, sys::path::filename(sourceFile));

      if (auto copyErr =
              FileManager::copyFile(absoluteSourcePath.str(), destPath.str())) {
        if (Config.getVerbose())
          outs() << "Warning: Failed to copy source file " << absoluteSourcePath
                 << " to " << destPath.str() << "\n";
      } else {
        Unit.addGeneratedFile("sources", destPath.str());
        if (Config.getVerbose())
          outs() << "Copied source: " << sourceFile << " -> " << destPath.str()
                 << "\n";
      }
    }
  }

  return Error::success();
}

const DataExtractor::ExtractorInfo DataExtractor::extractors[] = {
    {&DataExtractor::extractASTJSON, "AST JSON"},
    {&DataExtractor::extractDiagnostics, "Diagnostics"},
    {&DataExtractor::extractCoverage, "Coverage"},
    {&DataExtractor::extractTimeTrace, "Time trace"},
    {&DataExtractor::extractRuntimeTrace, "Runtime trace"},
    {&DataExtractor::extractSARIF, "SARIF"},
    {&DataExtractor::extractBinarySize, "Binary size"},
    {&DataExtractor::extractPGO, "PGO"},
    {&DataExtractor::extractSymbols, "Symbols"},
    {&DataExtractor::extractObjdump, "Objdump"},
    {&DataExtractor::extractXRay, "XRay"},
    {&DataExtractor::extractOptDot, "Opt DOT"}};

const size_t DataExtractor::numExtractors =
    sizeof(DataExtractor::extractors) / sizeof(DataExtractor::ExtractorInfo);

} // namespace advisor
} // namespace llvm
