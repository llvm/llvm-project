//===- jit-runner.cpp - MLIR CPU Execution Driver Library -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a library that provides a shared implementation for command line
// utilities that execute an MLIR file on the CPU by translating MLIR to LLVM
// IR before JIT-compiling and executing the latter.
//
// The translation can be customized by providing an MLIR to MLIR
// transformation.
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/JitRunner.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/ParseUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>

#define DEBUG_TYPE "jit-runner"

using namespace mlir;
using llvm::Error;

namespace {
/// This options struct prevents the need for global static initializers, and
/// is only initialized if the JITRunner is invoked.
struct Options {
  llvm::cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-")};
  llvm::cl::opt<std::string> mainFuncName{
      "e", llvm::cl::desc("The function to be called"),
      llvm::cl::value_desc("<function name>"), llvm::cl::init("main")};
  llvm::cl::opt<std::string> mainFuncType{
      "entry-point-result",
      llvm::cl::desc("Textual description of the function type to be called"),
      llvm::cl::value_desc("f32 | i32 | i64 | void"), llvm::cl::init("f32")};

  llvm::cl::OptionCategory optFlags{"opt-like flags"};

  // CLI variables for -On options.
  llvm::cl::opt<bool> optO0{"O0",
                            llvm::cl::desc("Run opt passes and codegen at O0"),
                            llvm::cl::cat(optFlags)};
  llvm::cl::opt<bool> optO1{"O1",
                            llvm::cl::desc("Run opt passes and codegen at O1"),
                            llvm::cl::cat(optFlags)};
  llvm::cl::opt<bool> optO2{"O2",
                            llvm::cl::desc("Run opt passes and codegen at O2"),
                            llvm::cl::cat(optFlags)};
  llvm::cl::opt<bool> optO3{"O3",
                            llvm::cl::desc("Run opt passes and codegen at O3"),
                            llvm::cl::cat(optFlags)};

  llvm::cl::list<std::string> mAttrs{
      "mattr", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Target specific attributes (-mattr=help for details)"),
      llvm::cl::value_desc("a1,+a2,-a3,..."), llvm::cl::cat(optFlags)};

  llvm::cl::opt<std::string> mArch{
      "march",
      llvm::cl::desc("Architecture to generate code for (see --version)")};

  llvm::cl::OptionCategory clOptionsCategory{"linking options"};
  llvm::cl::list<std::string> clSharedLibs{
      "shared-libs", llvm::cl::desc("Libraries to link dynamically"),
      llvm::cl::MiscFlags::CommaSeparated, llvm::cl::cat(clOptionsCategory)};

  /// CLI variables for debugging.
  llvm::cl::opt<bool> dumpObjectFile{
      "dump-object-file",
      llvm::cl::desc("Dump JITted-compiled object to file specified with "
                     "-object-filename (<input file>.o by default).")};

  llvm::cl::opt<std::string> objectFilename{
      "object-filename",
      llvm::cl::desc("Dump JITted-compiled object to file <input file>.o")};

  llvm::cl::opt<bool> hostSupportsJit{"host-supports-jit",
                                      llvm::cl::desc("Report host JIT support"),
                                      llvm::cl::Hidden};

  llvm::cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      llvm::cl::desc(
          "Disable implicit addition of a top-level module op during parsing"),
      llvm::cl::init(false)};
};

struct CompileAndExecuteConfig {
  /// LLVM module transformer that is passed to ExecutionEngine.
  std::function<llvm::Error(llvm::Module *)> transformer;

  /// A custom function that is passed to ExecutionEngine. It processes MLIR
  /// module and creates LLVM IR module.
  llvm::function_ref<std::unique_ptr<llvm::Module>(Operation *,
                                                   llvm::LLVMContext &)>
      llvmModuleBuilder;

  /// A custom function that is passed to ExecutinEngine to register symbols at
  /// runtime.
  llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
      runtimeSymbolMap;
};

} // namespace

static OwningOpRef<Operation *> parseMLIRInput(StringRef inputFilename,
                                               bool insertImplicitModule,
                                               MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());
  OwningOpRef<Operation *> module =
      parseSourceFileForTool(sourceMgr, context, insertImplicitModule);
  if (!module)
    return nullptr;
  if (!module.get()->hasTrait<OpTrait::SymbolTable>()) {
    llvm::errs() << "Error: top-level op must be a symbol table.\n";
    return nullptr;
  }
  return module;
}

static inline Error makeStringError(const Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

static std::optional<unsigned> getCommandLineOptLevel(Options &options) {
  std::optional<unsigned> optLevel;
  SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      options.optO0, options.optO1, options.optO2, options.optO3};

  // Determine if there is an optimization flag present.
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optLevel = j;
      break;
    }
  }
  return optLevel;
}

// JIT-compile the given module and run "entryPoint" with "args" as arguments.
static Error
compileAndExecute(Options &options, Operation *module, StringRef entryPoint,
                  CompileAndExecuteConfig config, void **args,
                  std::unique_ptr<llvm::TargetMachine> tm = nullptr) {
  std::optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel;
  if (auto clOptLevel = getCommandLineOptLevel(options))
    jitCodeGenOptLevel = static_cast<llvm::CodeGenOpt::Level>(*clOptLevel);

  SmallVector<StringRef, 4> sharedLibs(options.clSharedLibs.begin(),
                                       options.clSharedLibs.end());

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.llvmModuleBuilder = config.llvmModuleBuilder;
  if (config.transformer)
    engineOptions.transformer = config.transformer;
  engineOptions.jitCodeGenOptLevel = jitCodeGenOptLevel;
  engineOptions.sharedLibPaths = sharedLibs;
  engineOptions.enableObjectDump = true;
  auto expectedEngine =
      mlir::ExecutionEngine::create(module, engineOptions, std::move(tm));
  if (!expectedEngine)
    return expectedEngine.takeError();

  auto engine = std::move(*expectedEngine);

  auto expectedFPtr = engine->lookupPacked(entryPoint);
  if (!expectedFPtr)
    return expectedFPtr.takeError();

  if (options.dumpObjectFile)
    engine->dumpToObjectFile(options.objectFilename.empty()
                                 ? options.inputFilename + ".o"
                                 : options.objectFilename);

  void (*fptr)(void **) = *expectedFPtr;
  (*fptr)(args);

  return Error::success();
}

static Error compileAndExecuteVoidFunction(
    Options &options, Operation *module, StringRef entryPoint,
    CompileAndExecuteConfig config, std::unique_ptr<llvm::TargetMachine> tm) {
  auto mainFunction = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(module, entryPoint));
  if (!mainFunction || mainFunction.empty())
    return makeStringError("entry point not found");

  auto resultType = dyn_cast<LLVM::LLVMVoidType>(
      mainFunction.getFunctionType().getReturnType());
  if (!resultType)
    return makeStringError("expected void function");

  void *empty = nullptr;
  return compileAndExecute(options, module, entryPoint, std::move(config),
                           &empty, std::move(tm));
}

template <typename Type>
Error checkCompatibleReturnType(LLVM::LLVMFuncOp mainFunction);
template <>
Error checkCompatibleReturnType<int32_t>(LLVM::LLVMFuncOp mainFunction) {
  auto resultType = dyn_cast<IntegerType>(
      cast<LLVM::LLVMFunctionType>(mainFunction.getFunctionType())
          .getReturnType());
  if (!resultType || resultType.getWidth() != 32)
    return makeStringError("only single i32 function result supported");
  return Error::success();
}
template <>
Error checkCompatibleReturnType<int64_t>(LLVM::LLVMFuncOp mainFunction) {
  auto resultType = dyn_cast<IntegerType>(
      cast<LLVM::LLVMFunctionType>(mainFunction.getFunctionType())
          .getReturnType());
  if (!resultType || resultType.getWidth() != 64)
    return makeStringError("only single i64 function result supported");
  return Error::success();
}
template <>
Error checkCompatibleReturnType<float>(LLVM::LLVMFuncOp mainFunction) {
  if (!isa<Float32Type>(
          cast<LLVM::LLVMFunctionType>(mainFunction.getFunctionType())
              .getReturnType()))
    return makeStringError("only single f32 function result supported");
  return Error::success();
}
template <typename Type>
Error compileAndExecuteSingleReturnFunction(
    Options &options, Operation *module, StringRef entryPoint,
    CompileAndExecuteConfig config, std::unique_ptr<llvm::TargetMachine> tm) {
  auto mainFunction = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(module, entryPoint));
  if (!mainFunction || mainFunction.isExternal())
    return makeStringError("entry point not found");

  if (cast<LLVM::LLVMFunctionType>(mainFunction.getFunctionType())
          .getNumParams() != 0)
    return makeStringError("function inputs not supported");

  if (Error error = checkCompatibleReturnType<Type>(mainFunction))
    return error;

  Type res;
  struct {
    void *data;
  } data;
  data.data = &res;
  if (auto error =
          compileAndExecute(options, module, entryPoint, std::move(config),
                            (void **)&data, std::move(tm)))
    return error;

  // Intentional printing of the output so we can test.
  llvm::outs() << res << '\n';

  return Error::success();
}

/// Entry point for all CPU runners. Expects the common argc/argv arguments for
/// standard C++ main functions.
int mlir::JitRunnerMain(int argc, char **argv, const DialectRegistry &registry,
                        JitRunnerConfig config) {
  llvm::ExitOnError exitOnErr;

  // Create the options struct containing the command line options for the
  // runner. This must come before the command line options are parsed.
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  if (options.hostSupportsJit) {
    auto j = llvm::orc::LLJITBuilder().create();
    if (j)
      llvm::outs() << "true\n";
    else {
      llvm::outs() << "false\n";
      exitOnErr(j.takeError());
    }
    return 0;
  }

  std::optional<unsigned> optLevel = getCommandLineOptLevel(options);
  SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      options.optO0, options.optO1, options.optO2, options.optO3};

  MLIRContext context(registry);

  auto m = parseMLIRInput(options.inputFilename, !options.noImplicitModule,
                          &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  JitRunnerOptions runnerOptions{options.mainFuncName, options.mainFuncType};
  if (config.mlirTransformer)
    if (failed(config.mlirTransformer(m.get(), runnerOptions)))
      return EXIT_FAILURE;

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return EXIT_FAILURE;
  }

  // Configure TargetMachine builder based on the command line options
  llvm::SubtargetFeatures features;
  if (!options.mAttrs.empty()) {
    for (StringRef attr : options.mAttrs)
      features.AddFeature(attr);
    tmBuilderOrError->addFeatures(features.getFeatures());
  }

  if (!options.mArch.empty()) {
    tmBuilderOrError->getTargetTriple().setArchName(options.mArch);
  }

  // Build TargetMachine
  auto tmOrError = tmBuilderOrError->createTargetMachine();

  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    exitOnErr(tmOrError.takeError());
  }

  LLVM_DEBUG({
    llvm::dbgs() << "  JITTargetMachineBuilder is "
                 << llvm::orc::JITTargetMachineBuilderPrinter(*tmBuilderOrError,
                                                              "\n");
  });

  CompileAndExecuteConfig compileAndExecuteConfig;
  if (optLevel) {
    compileAndExecuteConfig.transformer = mlir::makeOptimizingTransformer(
        *optLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());
  }
  compileAndExecuteConfig.llvmModuleBuilder = config.llvmModuleBuilder;
  compileAndExecuteConfig.runtimeSymbolMap = config.runtimesymbolMap;

  // Get the function used to compile and execute the module.
  using CompileAndExecuteFnT =
      Error (*)(Options &, Operation *, StringRef, CompileAndExecuteConfig,
                std::unique_ptr<llvm::TargetMachine> tm);
  auto compileAndExecuteFn =
      StringSwitch<CompileAndExecuteFnT>(options.mainFuncType.getValue())
          .Case("i32", compileAndExecuteSingleReturnFunction<int32_t>)
          .Case("i64", compileAndExecuteSingleReturnFunction<int64_t>)
          .Case("f32", compileAndExecuteSingleReturnFunction<float>)
          .Case("void", compileAndExecuteVoidFunction)
          .Default(nullptr);

  Error error = compileAndExecuteFn
                    ? compileAndExecuteFn(
                          options, m.get(), options.mainFuncName.getValue(),
                          compileAndExecuteConfig, std::move(tmOrError.get()))
                    : makeStringError("unsupported function type");

  int exitCode = EXIT_SUCCESS;
  llvm::handleAllErrors(std::move(error),
                        [&exitCode](const llvm::ErrorInfoBase &info) {
                          llvm::errs() << "Error: ";
                          info.log(llvm::errs());
                          llvm::errs() << '\n';
                          exitCode = EXIT_FAILURE;
                        });

  return exitCode;
}
