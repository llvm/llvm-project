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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
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

  // CLI list of pass information
  llvm::cl::list<const llvm::PassInfo *, bool, llvm::PassNameParser> llvmPasses{
      llvm::cl::desc("LLVM optimizing passes to run"), llvm::cl::cat(optFlags)};

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
} // end anonymous namespace

static OwningModuleRef parseMLIRInput(StringRef inputFilename,
                                      MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return OwningModuleRef(parseSourceFile(sourceMgr, context));
}

static inline Error make_string_error(const Twine &message) {
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
static Error
compileAndExecute(Options &options, ModuleOp module, StringRef entryPoint,
                  std::function<llvm::Error(llvm::Module *)> transformer,
                  void **args) {
  Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel;
  if (auto clOptLevel = getCommandLineOptLevel(options))
    jitCodeGenOptLevel =
        static_cast<llvm::CodeGenOpt::Level>(clOptLevel.getValue());
  SmallVector<StringRef, 4> libs(options.clSharedLibs.begin(),
                                 options.clSharedLibs.end());
  auto expectedEngine = mlir::ExecutionEngine::create(module, transformer,
                                                      jitCodeGenOptLevel, libs);
  if (!expectedEngine)
    return expectedEngine.takeError();

  auto engine = std::move(*expectedEngine);
  auto expectedFPtr = engine->lookup(entryPoint);
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
    Options &options, ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.empty())
    return make_string_error("entry point not found");
  void *empty = nullptr;
  return compileAndExecute(options, module, entryPoint, transformer, &empty);
}

template <typename Type>
Error checkCompatibleReturnType(LLVM::LLVMFuncOp mainFunction);
template <>
Error checkCompatibleReturnType<int32_t>(LLVM::LLVMFuncOp mainFunction) {
  if (!mainFunction.getType().getFunctionResultType().isIntegerTy(32))
    return make_string_error("only single llvm.i32 function result supported");
  return Error::success();
}
template <>
Error checkCompatibleReturnType<int64_t>(LLVM::LLVMFuncOp mainFunction) {
  if (!mainFunction.getType().getFunctionResultType().isIntegerTy(64))
    return make_string_error("only single llvm.i64 function result supported");
  return Error::success();
}
template <>
Error checkCompatibleReturnType<float>(LLVM::LLVMFuncOp mainFunction) {
  if (!mainFunction.getType().getFunctionResultType().isFloatTy())
    return make_string_error("only single llvm.f32 function result supported");
  return Error::success();
}
template <typename Type>
Error compileAndExecuteSingleReturnFunction(
    Options &options, ModuleOp module, StringRef entryPoint,
    std::function<llvm::Error(llvm::Module *)> transformer) {
  auto mainFunction = module.lookupSymbol<LLVM::LLVMFuncOp>(entryPoint);
  if (!mainFunction || mainFunction.isExternal())
    return make_string_error("entry point not found");

  if (mainFunction.getType().getFunctionNumParams() != 0)
    return make_string_error("function inputs not supported");

  if (Error error = checkCompatibleReturnType<Type>(mainFunction))
    return error;

  Type res;
  struct {
    void *data;
  } data;
  data.data = &res;
  if (auto error = compileAndExecute(options, module, entryPoint, transformer,
                                     (void **)&data))
    return error;

  // Intentional printing of the output so we can test.
  llvm::outs() << res << '\n';

  return Error::success();
}

/// Entry point for all CPU runners. Expects the common argc/argv
/// arguments for standard C++ main functions and an mlirTransformer.
/// The latter is applied after parsing the input into MLIR IR and
/// before passing the MLIR module to the ExecutionEngine.
int mlir::JitRunnerMain(
    int argc, char **argv,
    function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer) {
  // Create the options struct containing the command line options for the
  // runner. This must come before the command line options are parsed.
  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CPU execution driver\n");

  Optional<unsigned> optLevel = getCommandLineOptLevel(options);
  SmallVector<std::reference_wrapper<llvm::cl::opt<bool>>, 4> optFlags{
      options.optO0, options.optO1, options.optO2, options.optO3};
  unsigned optCLIPosition = 0;
  // Determine if there is an optimization flag present, and its CLI position
  // (optCLIPosition).
  for (unsigned j = 0; j < 4; ++j) {
    auto &flag = optFlags[j].get();
    if (flag) {
      optCLIPosition = flag.getPosition();
      break;
    }
  }
  // Generate vector of pass information, plus the index at which we should
  // insert any optimization passes in that vector (optPosition).
  SmallVector<const llvm::PassInfo *, 4> passes;
  unsigned optPosition = 0;
  for (unsigned i = 0, e = options.llvmPasses.size(); i < e; ++i) {
    passes.push_back(options.llvmPasses[i]);
    if (optCLIPosition < options.llvmPasses.getPosition(i)) {
      optPosition = i;
      optCLIPosition = UINT_MAX; // To ensure we never insert again
    }
  }

  MLIRContext context(/*loadAllDialects=*/false);
  registerAllDialects(&context);

  auto m = parseMLIRInput(options.inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  if (mlirTransformer)
    if (failed(mlirTransformer(m.get())))
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

  auto transformer = mlir::makeLLVMPassesTransformer(
      passes, optLevel, /*targetMachine=*/tmOrError->get(), optPosition);

  // Get the function used to compile and execute the module.
  using CompileAndExecuteFnT =
      Error (*)(Options &, ModuleOp, StringRef,
                std::function<llvm::Error(llvm::Module *)>);
  auto compileAndExecuteFn =
      llvm::StringSwitch<CompileAndExecuteFnT>(options.mainFuncType.getValue())
          .Case("i32", compileAndExecuteSingleReturnFunction<int32_t>)
          .Case("i64", compileAndExecuteSingleReturnFunction<int64_t>)
          .Case("f32", compileAndExecuteSingleReturnFunction<float>)
          .Case("void", compileAndExecuteVoidFunction)
          .Default(nullptr);

  Error error =
      compileAndExecuteFn
          ? compileAndExecuteFn(options, m.get(),
                                options.mainFuncName.getValue(), transformer)
          : make_string_error("unsupported function type");

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
