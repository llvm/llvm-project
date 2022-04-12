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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cstdint>
#include <numeric>

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

  llvm::cl::OptionCategory clOptionsCategory{"linking options"};
  llvm::cl::list<std::string> clSharedLibs{
      "shared-libs", llvm::cl::desc("Libraries to link dynamically"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::cat(clOptionsCategory)};

  /// CLI variables for debugging.
  llvm::cl::opt<bool> dumpObjectFile{
      "dump-object-file",
      llvm::cl::desc("Dump JITted-compiled object to file specified with "
                     "-object-filename (<input file>.o by default).")};

  llvm::cl::opt<std::string> objectFilename{
      "object-filename",
      llvm::cl::desc("Dump JITted-compiled object to file <input file>.o")};
};

struct CompileAndExecuteConfig {
  /// LLVM module transformer that is passed to ExecutionEngine.
  std::function<llvm::Error(llvm::Module *)> transformer;

  /// A custom function that is passed to ExecutionEngine. It processes MLIR
  /// module and creates LLVM IR module.
  llvm::function_ref<std::unique_ptr<llvm::Module>(ModuleOp,
                                                   llvm::LLVMContext &)>
      llvmModuleBuilder;

  /// A custom function that is passed to ExecutinEngine to register symbols at
  /// runtime.
  llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
      runtimeSymbolMap;
};

} // namespace

static OwningOpRef<ModuleOp> parseMLIRInput(StringRef inputFilename,
                                            MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, context);
}

static inline Error makeStringError(const Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

static Optional<unsigned> getCommandLineOptLevel(Options &options) {
  Optional<unsigned> optLevel;
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
static Error compileAndExecute(Options &options, ModuleOp module,
                               StringRef entryPoint,
                               CompileAndExecuteConfig config, void **args) {
  Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel;
  if (auto clOptLevel = getCommandLineOptLevel(options))
    jitCodeGenOptLevel =
        static_cast<llvm::CodeGenOpt::Level>(clOptLevel.getValue());

  // If shared library implements custom mlir-runner library init and destroy
  // functions, we'll use them to register the library with the execution
  // engine. Otherwise we'll pass library directly to the execution engine.
  SmallVector<SmallString<256>, 4> libPaths;

  // Use absolute library path so that gdb can find the symbol table.
  transform(
      options.clSharedLibs, std::back_inserter(libPaths),
      [](std::string libPath) {
        SmallString<256> absPath(libPath.begin(), libPath.end());
        cantFail(llvm::errorCodeToError(llvm::sys::fs::make_absolute(absPath)));
        return absPath;
      });

  // Libraries that we'll pass to the ExecutionEngine for loading.
  SmallVector<StringRef, 4> executionEngineLibs;

  using MlirRunnerInitFn = void (*)(llvm::StringMap<void *> &);
  using MlirRunnerDestroyFn = void (*)();

  llvm::StringMap<void *> exportSymbols;
  SmallVector<MlirRunnerDestroyFn> destroyFns;

  // Handle libraries that do support mlir-runner init/destroy callbacks.
  for (auto &libPath : libPaths) {
    auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(libPath.c_str());
    void *initSym = lib.getAddressOfSymbol("__mlir_runner_init");
    void *destroySim = lib.getAddressOfSymbol("__mlir_runner_destroy");

    // Library does not support mlir runner, load it with ExecutionEngine.
    if (!initSym || !destroySim) {
      executionEngineLibs.push_back(libPath);
      continue;
    }

    auto initFn = reinterpret_cast<MlirRunnerInitFn>(initSym);
    initFn(exportSymbols);

    auto destroyFn = reinterpret_cast<MlirRunnerDestroyFn>(destroySim);
    destroyFns.push_back(destroyFn);
  }

  // Build a runtime symbol map from the config and exported symbols.
  auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
    auto symbolMap = config.runtimeSymbolMap ? config.runtimeSymbolMap(interner)
                                             : llvm::orc::SymbolMap();
    for (auto &exportSymbol : exportSymbols)
      symbolMap[interner(exportSymbol.getKey())] =
          llvm::JITEvaluatedSymbol::fromPointer(exportSymbol.getValue());
    return symbolMap;
  };

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.llvmModuleBuilder = config.llvmModuleBuilder;
  engineOptions.transformer = config.transformer;
  engineOptions.jitCodeGenOptLevel = jitCodeGenOptLevel;
  engineOptions.sharedLibPaths = executionEngineLibs;
  engineOptions.enableObjectCache = true;
  auto expectedEngine = mlir::ExecutionEngine::create(module, engineOptions);
  if (!expectedEngine)
    return expectedEngine.takeError();

  auto engine = std::move(*expectedEngine);
  engine->registerSymbols(runtimeSymbolMap);

  auto expectedFPtr = engine->lookupPacked(entryPoint);
  if (!expectedFPtr)
    return expectedFPtr.takeError();

  if (options.dumpObjectFile)
    engine->dumpToObjectFile(options.objectFilename.empty()
                                 ? options.inputFilename + ".o"
                                 : options.objectFilename);

  void (*fptr)(void **) = *expectedFPtr;
  (*fptr)(args);

  // Run all dynamic library destroy callbacks to prepare for the shutdown.
  llvm::for_each(destroyFns, [](MlirRunnerDestroyFn destroy) { destroy(); });

  return Error::success();
}

static Error compileAndExecuteVoidFunction(Options &options, ModuleOp module,
                                           StringRef entryPoint,
                                           CompileAndExecuteConfig config) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.empty())
    return makeStringError("entry point not found");
  void *empty = nullptr;
  return compileAndExecute(options, module, entryPoint, config, &empty);
}

template <typename Type>
Error checkCompatibleReturnType(LLVM::LLVMFuncOp mainFunction);
template <>
Error checkCompatibleReturnType<int32_t>(LLVM::LLVMFuncOp mainFunction) {
  auto resultType = mainFunction.getFunctionType()
                        .cast<LLVM::LLVMFunctionType>()
                        .getReturnType()
                        .dyn_cast<IntegerType>();
  if (!resultType || resultType.getWidth() != 32)
    return makeStringError("only single i32 function result supported");
  return Error::success();
}
template <>
Error checkCompatibleReturnType<int64_t>(LLVM::LLVMFuncOp mainFunction) {
  auto resultType = mainFunction.getFunctionType()
                        .cast<LLVM::LLVMFunctionType>()
                        .getReturnType()
                        .dyn_cast<IntegerType>();
  if (!resultType || resultType.getWidth() != 64)
    return makeStringError("only single i64 function result supported");
  return Error::success();
}
template <>
Error checkCompatibleReturnType<float>(LLVM::LLVMFuncOp mainFunction) {
  if (!mainFunction.getFunctionType()
           .cast<LLVM::LLVMFunctionType>()
           .getReturnType()
           .isa<Float32Type>())
    return makeStringError("only single f32 function result supported");
  return Error::success();
}
template <typename Type>
Error compileAndExecuteSingleReturnFunction(Options &options, ModuleOp module,
                                            StringRef entryPoint,
                                            CompileAndExecuteConfig config) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.isExternal())
    return makeStringError("entry point not found");

  if (mainFunction.getFunctionType()
          .cast<LLVM::LLVMFunctionType>()
          .getNumParams() != 0)
    return makeStringError("function inputs not supported");

  if (Error error = checkCompatibleReturnType<Type>(mainFunction))
    return error;

  Type res;
  struct {
    void *data;
  } data;
  data.data = &res;
  if (auto error = compileAndExecute(options, module, entryPoint, config,
                                     (void **)&data))
    return error;

  // Intentional printing of the output so we can test.
  llvm::outs() << res << '\n';

  return Error::success();
}

/// Entry point for all CPU runners. Expects the common argc/argv arguments for
/// standard C++ main functions.
int mlir::JitRunnerMain(int argc, char **argv, const DialectRegistry &registry,
                        JitRunnerConfig config) {
  // Create the options struct containing the command line options for the
  // runner. This must come before the command line options are parsed.
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  Optional<unsigned> optLevel = getCommandLineOptLevel(options);
  SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      options.optO0, options.optO1, options.optO2, options.optO3};

  MLIRContext context(registry);

  auto m = parseMLIRInput(options.inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  if (config.mlirTransformer)
    if (failed(config.mlirTransformer(m.get())))
      return EXIT_FAILURE;

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return EXIT_FAILURE;
  }
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    return EXIT_FAILURE;
  }

  CompileAndExecuteConfig compileAndExecuteConfig;
  if (optLevel) {
    compileAndExecuteConfig.transformer = mlir::makeOptimizingTransformer(
        *optLevel, /*sizeLevel=*/0, /*targetMachine=*/tmOrError->get());
  }
  compileAndExecuteConfig.llvmModuleBuilder = config.llvmModuleBuilder;
  compileAndExecuteConfig.runtimeSymbolMap = config.runtimesymbolMap;

  // Get the function used to compile and execute the module.
  using CompileAndExecuteFnT =
      Error (*)(Options &, ModuleOp, StringRef, CompileAndExecuteConfig);
  auto compileAndExecuteFn =
      StringSwitch<CompileAndExecuteFnT>(options.mainFuncType.getValue())
          .Case("i32", compileAndExecuteSingleReturnFunction<int32_t>)
          .Case("i64", compileAndExecuteSingleReturnFunction<int64_t>)
          .Case("f32", compileAndExecuteSingleReturnFunction<float>)
          .Case("void", compileAndExecuteVoidFunction)
          .Default(nullptr);

  Error error = compileAndExecuteFn
                    ? compileAndExecuteFn(options, m.get(),
                                          options.mainFuncName.getValue(),
                                          compileAndExecuteConfig)
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
