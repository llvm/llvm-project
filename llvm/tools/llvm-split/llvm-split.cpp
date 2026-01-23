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

enum class SplitByCategoryType {
  SBCT_ByModuleId,
  SBCT_ByKernel,
  SBCT_None,
};

static cl::opt<SplitByCategoryType> SplitByCategory(
    "split-by-category",
    cl::desc("Split by category. If present, splitting by category is used "
             "with the specified categorization type."),
    cl::Optional, cl::init(SplitByCategoryType::SBCT_None),
    cl::values(clEnumValN(SplitByCategoryType::SBCT_ByModuleId, "module-id",
                          "one output module per translation unit marked with "
                          "\"module-id\" attribute"),
               clEnumValN(SplitByCategoryType::SBCT_ByKernel, "kernel",
                          "one output module per kernel")),
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

/// EntryPointCategorizer is used for splitting by category either by module-id
/// or by kernels. It doesn't provide categories for functions other than
/// kernels. Categorizer computes a string key for the given Function and
/// records the association between the string key and an integer category. If a
/// string key is already belongs to some category than the corresponding
/// integer category is returned.
class EntryPointCategorizer {
public:
  EntryPointCategorizer(SplitByCategoryType Type) : Type(Type) {}

  EntryPointCategorizer() = delete;
  EntryPointCategorizer(EntryPointCategorizer &) = delete;
  EntryPointCategorizer &operator=(const EntryPointCategorizer &) = delete;
  EntryPointCategorizer(EntryPointCategorizer &&) = default;
  EntryPointCategorizer &operator=(EntryPointCategorizer &&) = default;

  /// Returns integer specifying the category for the given \p F.
  /// If the given function isn't a kernel then returns std::nullopt.
  std::optional<int> operator()(const Function &F) {
    if (!isEntryPoint(F))
      return std::nullopt; // skip the function.

    auto StringKey = computeFunctionCategory(Type, F);
    if (auto it = StrKeyToID.find(StringRef(StringKey)); it != StrKeyToID.end())
      return it->second;

    int ID = static_cast<int>(StrKeyToID.size());
    return StrKeyToID.try_emplace(std::move(StringKey), ID).first->second;
  }

private:
  static bool isEntryPoint(const Function &F) {
    if (F.isDeclaration())
      return false;

    return F.getCallingConv() == CallingConv::SPIR_KERNEL ||
           F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
           F.getCallingConv() == CallingConv::PTX_Kernel;
  }

  static SmallString<0> computeFunctionCategory(SplitByCategoryType Type,
                                                const Function &F) {
    static constexpr char ATTR_MODULE_ID[] = "module-id";
    SmallString<0> Key;
    switch (Type) {
    case SplitByCategoryType::SBCT_ByKernel:
      Key = F.getName().str();
      break;
    case SplitByCategoryType::SBCT_ByModuleId:
      Key = F.getFnAttribute(ATTR_MODULE_ID).getValueAsString().str();
      break;
    default:
      llvm_unreachable("unexpected mode.");
    }

    return Key;
  }

private:
  struct KeyInfo {
    static SmallString<0> getEmptyKey() { return SmallString<0>(""); }

    static SmallString<0> getTombstoneKey() { return SmallString<0>("-"); }

    static bool isEqual(const SmallString<0> &LHS, const SmallString<0> &RHS) {
      return LHS == RHS;
    }

    static unsigned getHashValue(const SmallString<0> &S) {
      return llvm::hash_value(StringRef(S));
    }
  };

  SplitByCategoryType Type;
  DenseMap<SmallString<0>, int, KeyInfo> StrKeyToID;
};

void cleanupModule(Module &M) {
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  ModulePassManager MPM;
  MPM.addPass(GlobalDCEPass()); // Delete unreachable globals.
  MPM.run(M, MAM);
}

Error runSplitModuleByCategory(std::unique_ptr<Module> M) {
  size_t OutputID = 0;
  auto PostSplitCallback = [&](std::unique_ptr<Module> MPart) {
    if (verifyModule(*MPart)) {
      errs() << "Broken Module!\n";
      exit(1);
    }

    // TODO: DCE is a crucial pass since it removes unused declarations.
    //       At the moment, LIT checking can't be perfomed without DCE.
    cleanupModule(*MPart);
    size_t ID = OutputID;
    ++OutputID;
    StringRef ModuleSuffix = OutputAssembly ? ".ll" : ".bc";
    std::string ModulePath =
        (Twine(OutputFilename) + "_" + Twine(ID) + ModuleSuffix).str();
    writeModuleToFile(*MPart, ModulePath, OutputAssembly);
  };

  auto Categorizer = EntryPointCategorizer(SplitByCategory);
  splitModuleTransitiveFromEntryPoints(std::move(M), Categorizer,
                                       PostSplitCallback);
  return Error::success();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  LLVMContext Context;
  SMDiagnostic Err;
  cl::HideUnrelatedOptions({&SplitCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "LLVM module splitter\n");

  Triple TT(MTriple);

  std::unique_ptr<TargetMachine> TM;
  if (!MTriple.empty()) {
    InitializeAllTargets();
    InitializeAllTargetMCs();

    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T) {
      errs() << "unknown target '" << MTriple << "': " << Error << "\n";
      return 1;
    }

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TT, MCPU, /*FS*/ "", Options, std::nullopt, std::nullopt));
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

  if (SplitByCategory != SplitByCategoryType::SBCT_None) {
    auto E = runSplitModuleByCategory(std::move(M));
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
