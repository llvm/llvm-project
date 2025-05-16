//===-- llvm-split: command line tool for testing module splitting --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program can be used to test the llvm::SplitModule and
// TargetMachine::splitModule functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Utils/SYCLUtils.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/SplitModuleByCategory.h"

using namespace llvm;

static cl::OptionCategory SplitCategory("Split Options");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input bitcode file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"),
                                          cl::cat(SplitCategory));

static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(SplitCategory));

static cl::opt<unsigned> NumOutputs("j", cl::Prefix, cl::init(2),
                                    cl::desc("Number of output files"),
                                    cl::cat(SplitCategory));

static cl::opt<bool>
    PreserveLocals("preserve-locals", cl::Prefix, cl::init(false),
                   cl::desc("Split without externalizing locals"),
                   cl::cat(SplitCategory));

static cl::opt<bool>
    RoundRobin("round-robin", cl::Prefix, cl::init(false),
               cl::desc("Use round-robin distribution of functions to "
                        "modules instead of the default name-hash-based one"),
               cl::cat(SplitCategory));

static cl::opt<std::string>
    MTriple("mtriple",
            cl::desc("Target triple. When present, a TargetMachine is created "
                     "and TargetMachine::splitModule is used instead of the "
                     "common SplitModule logic."),
            cl::value_desc("triple"), cl::cat(SplitCategory));

static cl::opt<std::string>
    MCPU("mcpu", cl::desc("Target CPU, ignored if --mtriple is not used"),
         cl::value_desc("cpu"), cl::cat(SplitCategory));

static cl::opt<sycl::IRSplitMode> SYCLSplitMode(
    "sycl-split",
    cl::desc("SYCL Split Mode. If present, SYCL splitting algorithm is used "
             "with the specified mode."),
    cl::Optional, cl::init(sycl::IRSplitMode::IRSM_NONE),
    cl::values(clEnumValN(sycl::IRSplitMode::IRSM_PER_TU, "source",
                          "1 ouptput module per translation unit"),
               clEnumValN(sycl::IRSplitMode::IRSM_PER_KERNEL, "kernel",
                          "1 output module per kernel")),
    cl::cat(SplitCategory));

static cl::opt<bool> OutputAssembly{
    "S", cl::desc("Write output as LLVM assembly"), cl::cat(SplitCategory)};

void writeStringToFile(StringRef Content, StringRef Path) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC);
  if (EC) {
    errs() << formatv("error opening file: {0}, error: {1}\n", Path,
                      EC.message());
    exit(1);
  }

  OS << Content << "\n";
}

void writeModuleToFile(const Module &M, StringRef Path, bool OutputAssembly) {
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(Path, FD)) {
    errs() << formatv("error opening file: {0}, error: {1}", Path, EC.message())
           << '\n';
    exit(1);
  }

  raw_fd_ostream OS(FD, /*ShouldClose*/ true);
  if (OutputAssembly)
    M.print(OS, /*AssemblyAnnotationWriter*/ nullptr);
  else
    WriteBitcodeToFile(M, OS);
}

void writeSplitModulesAsTable(ArrayRef<sycl::ModuleAndSYCLMetadata> Modules,
                              StringRef Path) {
  SmallVector<SmallString<64>> Columns;
  Columns.emplace_back("Code");
  Columns.emplace_back("Symbols");

  sycl::StringTable Table;
  Table.emplace_back(std::move(Columns));
  for (const auto &[I, SM] : enumerate(Modules)) {
    SmallString<128> SymbolsFile;
    (Twine(Path) + "_" + Twine(I) + ".sym").toVector(SymbolsFile);
    writeStringToFile(SM.Symbols, SymbolsFile);
    SmallVector<SmallString<64>> Row;
    Row.emplace_back(SM.ModuleFilePath);
    Row.emplace_back(SymbolsFile);
    Table.emplace_back(std::move(Row));
  }

  std::error_code EC;
  raw_fd_ostream OS((Path + ".table").str(), EC);
  if (EC) {
    errs() << formatv("error opening file: {0}\n", Path);
    exit(1);
  }

  sycl::writeStringTable(Table, OS);
}

void cleanupModule(Module &M) {
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  ModulePassManager MPM;
  MPM.addPass(GlobalDCEPass()); // Delete unreachable globals.
  MPM.run(M, MAM);
}

Error runSYCLSplitModule(std::unique_ptr<Module> M) {
  SmallVector<sycl::ModuleAndSYCLMetadata> SplitModules;
  auto PostSplitCallback = [&](std::unique_ptr<Module> MPart) {
    if (verifyModule(*MPart)) {
      errs() << "Broken Module!\n";
      exit(1);
    }

    // TODO: DCE is a crucial pass in a SYCL post-link pipeline.
    //       At the moment, LIT checking can't be perfomed without DCE.
    cleanupModule(*MPart);
    size_t ID = SplitModules.size();
    StringRef ModuleSuffix = OutputAssembly ? ".ll" : ".bc";
    std::string ModulePath =
        (Twine(OutputFilename) + "_" + Twine(ID) + ModuleSuffix).str();
    writeModuleToFile(*MPart, ModulePath, OutputAssembly);
    auto Symbols = sycl::makeSymbolTable(*MPart);
    SplitModules.emplace_back(std::move(ModulePath), std::move(Symbols));
  };

  auto Categorizer = sycl::FunctionCategorizer(SYCLSplitMode);
  sycl::SplitModuleByCategory(std::move(M), std::move(Categorizer),
                              PostSplitCallback);
  writeSplitModulesAsTable(SplitModules, OutputFilename);
  return Error::success();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  LLVMContext Context;
  SMDiagnostic Err;
  cl::HideUnrelatedOptions({&SplitCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "LLVM module splitter\n");

  std::unique_ptr<TargetMachine> TM;
  if (!MTriple.empty()) {
    InitializeAllTargets();
    InitializeAllTargetMCs();

    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(MTriple, Error);
    if (!T) {
      errs() << "unknown target '" << MTriple << "': " << Error << "\n";
      return 1;
    }

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        Triple(MTriple), MCPU, /*FS*/ "", Options, std::nullopt, std::nullopt));
  }

  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);

  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  unsigned I = 0;
  const auto HandleModulePart = [&](std::unique_ptr<Module> MPart) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> Out(
        new ToolOutputFile(OutputFilename + utostr(I++), EC, sys::fs::OF_None));
    if (EC) {
      errs() << EC.message() << '\n';
      exit(1);
    }

    if (verifyModule(*MPart, &errs())) {
      errs() << "Broken module!\n";
      exit(1);
    }

    WriteBitcodeToFile(*MPart, Out->os());

    // Declare success.
    Out->keep();
  };

  if (SYCLSplitMode != sycl::IRSplitMode::IRSM_NONE) {
    auto E = runSYCLSplitModule(std::move(M));
    if (E) {
      errs() << E << "\n";
      Err.print(argv[0], errs());
      return 1;
    }

    return 0;
  }

  if (TM) {
    if (PreserveLocals) {
      errs() << "warning: --preserve-locals has no effect when using "
                "TargetMachine::splitModule\n";
    }
    if (RoundRobin)
      errs() << "warning: --round-robin has no effect when using "
                "TargetMachine::splitModule\n";

    if (TM->splitModule(*M, NumOutputs, HandleModulePart))
      return 0;

    errs() << "warning: "
              "TargetMachine::splitModule failed, falling back to default "
              "splitModule implementation\n";
  }

  SplitModule(*M, NumOutputs, HandleModulePart, PreserveLocals, RoundRobin);
  return 0;
}
