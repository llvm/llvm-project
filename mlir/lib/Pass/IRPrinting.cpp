//===- IRPrinting.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
//===----------------------------------------------------------------------===//
// IRPrinter
//===----------------------------------------------------------------------===//

class IRPrinterInstrumentation : public PassInstrumentation {
public:
  IRPrinterInstrumentation(std::unique_ptr<PassManager::IRPrinterConfig> config)
      : config(std::move(config)) {}

private:
  /// Instrumentation hooks.
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;

  /// Configuration to use.
  std::unique_ptr<PassManager::IRPrinterConfig> config;

  /// The following is a set of fingerprints for operations that are currently
  /// being operated on in a pass. This field is only used when the
  /// configuration asked for change detection.
  DenseMap<Pass *, OperationFingerPrint> beforePassFingerPrints;
};
} // namespace

static void printIR(Operation *op, bool printModuleScope, raw_ostream &out,
                    OpPrintingFlags flags) {
  // Otherwise, check to see if we are not printing at module scope.
  if (!printModuleScope)
    return op->print(out << " //----- //\n",
                     op->getBlock() ? flags.useLocalScope() : flags);

  // Otherwise, we are printing at module scope.
  out << " ('" << op->getName() << "' operation";
  if (auto symbolName =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    out << ": @" << symbolName.getValue();
  out << ") //----- //\n";

  // Find the top-level operation.
  auto *topLevelOp = op;
  while (auto *parentOp = topLevelOp->getParentOp())
    topLevelOp = parentOp;
  topLevelOp->print(out, flags);
}

/// Instrumentation hooks.
void IRPrinterInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  if (isa<OpToOpPassAdaptor>(pass))
    return;
  // If the config asked to detect changes, record the current fingerprint.
  if (config->shouldPrintAfterOnlyOnChange())
    beforePassFingerPrints.try_emplace(pass, op);

  config->printBeforeIfEnabled(pass, op, [&](raw_ostream &out) {
    out << "// -----// IR Dump Before " << pass->getName() << " ("
        << pass->getArgument() << ")";
    printIR(op, config->shouldPrintAtModuleScope(), out,
            config->getOpPrintingFlags());
    out << "\n\n";
  });
}

void IRPrinterInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  if (isa<OpToOpPassAdaptor>(pass))
    return;

  // Check to see if we are only printing on failure.
  if (config->shouldPrintAfterOnlyOnFailure())
    return;

  // If the config asked to detect changes, compare the current fingerprint with
  // the previous.
  if (config->shouldPrintAfterOnlyOnChange()) {
    auto fingerPrintIt = beforePassFingerPrints.find(pass);
    assert(fingerPrintIt != beforePassFingerPrints.end() &&
           "expected valid fingerprint");
    // If the fingerprints are the same, we don't print the IR.
    if (fingerPrintIt->second == OperationFingerPrint(op)) {
      beforePassFingerPrints.erase(fingerPrintIt);
      return;
    }
    beforePassFingerPrints.erase(fingerPrintIt);
  }

  config->printAfterIfEnabled(pass, op, [&](raw_ostream &out) {
    out << "// -----// IR Dump After " << pass->getName() << " ("
        << pass->getArgument() << ")";
    printIR(op, config->shouldPrintAtModuleScope(), out,
            config->getOpPrintingFlags());
    out << "\n\n";
  });
}

void IRPrinterInstrumentation::runAfterPassFailed(Pass *pass, Operation *op) {
  if (isa<OpToOpPassAdaptor>(pass))
    return;
  if (config->shouldPrintAfterOnlyOnChange())
    beforePassFingerPrints.erase(pass);

  config->printAfterIfEnabled(pass, op, [&](raw_ostream &out) {
    out << formatv("// -----// IR Dump After {0} Failed ({1})", pass->getName(),
                   pass->getArgument());
    printIR(op, config->shouldPrintAtModuleScope(), out,
            config->getOpPrintingFlags());
    out << "\n\n";
  });
}

//===----------------------------------------------------------------------===//
// IRPrinterConfig
//===----------------------------------------------------------------------===//

/// Initialize the configuration.
PassManager::IRPrinterConfig::IRPrinterConfig(bool printModuleScope,
                                              bool printAfterOnlyOnChange,
                                              bool printAfterOnlyOnFailure,
                                              OpPrintingFlags opPrintingFlags)
    : printModuleScope(printModuleScope),
      printAfterOnlyOnChange(printAfterOnlyOnChange),
      printAfterOnlyOnFailure(printAfterOnlyOnFailure),
      opPrintingFlags(opPrintingFlags) {}
PassManager::IRPrinterConfig::~IRPrinterConfig() = default;

/// A hook that may be overridden by a derived config that checks if the IR
/// of 'operation' should be dumped *before* the pass 'pass' has been
/// executed. If the IR should be dumped, 'printCallback' should be invoked
/// with the stream to dump into.
void PassManager::IRPrinterConfig::printBeforeIfEnabled(
    Pass *pass, Operation *operation, PrintCallbackFn printCallback) {
  // By default, never print.
}

/// A hook that may be overridden by a derived config that checks if the IR
/// of 'operation' should be dumped *after* the pass 'pass' has been
/// executed. If the IR should be dumped, 'printCallback' should be invoked
/// with the stream to dump into.
void PassManager::IRPrinterConfig::printAfterIfEnabled(
    Pass *pass, Operation *operation, PrintCallbackFn printCallback) {
  // By default, never print.
}

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

namespace {
/// Simple wrapper config that allows for the simpler interface defined above.
struct BasicIRPrinterConfig : public PassManager::IRPrinterConfig {
  BasicIRPrinterConfig(
      std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
      std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
      bool printModuleScope, bool printAfterOnlyOnChange,
      bool printAfterOnlyOnFailure, OpPrintingFlags opPrintingFlags,
      raw_ostream &out)
      : IRPrinterConfig(printModuleScope, printAfterOnlyOnChange,
                        printAfterOnlyOnFailure, opPrintingFlags),
        shouldPrintBeforePass(std::move(shouldPrintBeforePass)),
        shouldPrintAfterPass(std::move(shouldPrintAfterPass)), out(out) {
    assert((this->shouldPrintBeforePass || this->shouldPrintAfterPass) &&
           "expected at least one valid filter function");
  }

  void printBeforeIfEnabled(Pass *pass, Operation *operation,
                            PrintCallbackFn printCallback) final {
    if (shouldPrintBeforePass && shouldPrintBeforePass(pass, operation))
      printCallback(out);
  }

  void printAfterIfEnabled(Pass *pass, Operation *operation,
                           PrintCallbackFn printCallback) final {
    if (shouldPrintAfterPass && shouldPrintAfterPass(pass, operation))
      printCallback(out);
  }

  /// Filter functions for before and after pass execution.
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass;

  /// The stream to output to.
  raw_ostream &out;
};
} // namespace

/// Return pairs of (sanitized op name, symbol name) for `op` and all parent
/// operations. Op names are sanitized by replacing periods with underscores.
/// The pairs are returned in order of outer-most to inner-most (ancestors of
/// `op` first, `op` last). This information is used to construct the directory
/// tree for the `FileTreeIRPrinterConfig` below.
/// The counter for `op` will be incremented by this call.
static std::pair<SmallVector<std::pair<std::string, StringRef>>, std::string>
getOpAndSymbolNames(Operation *op, StringRef passName,
                    llvm::DenseMap<Operation *, unsigned> &counters) {
  SmallVector<std::pair<std::string, StringRef>> pathElements;
  SmallVector<unsigned> countPrefix;

  if (!counters.contains(op))
    counters[op] = -1;

  Operation *iter = op;
  ++counters[op];
  while (iter) {
    countPrefix.push_back(counters[iter]);
    StringAttr symbolName =
        iter->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    std::string opName =
        llvm::join(llvm::split(iter->getName().getStringRef().str(), '.'), "_");
    pathElements.emplace_back(opName, symbolName ? symbolName.strref()
                                                 : "no-symbol-name");
    iter = iter->getParentOp();
  }
  // Return in the order of top level (module) down to `op`.
  std::reverse(countPrefix.begin(), countPrefix.end());
  std::reverse(pathElements.begin(), pathElements.end());

  std::string passFileName = llvm::formatv(
      "{0:$[_]}_{1}.mlir",
      llvm::make_range(countPrefix.begin(), countPrefix.end()), passName);

  return {pathElements, passFileName};
}

static LogicalResult createDirectoryOrPrintErr(llvm::StringRef dirPath) {
  if (std::error_code ec =
          llvm::sys::fs::create_directory(dirPath, /*IgnoreExisting=*/true)) {
    llvm::errs() << "Error while creating directory " << dirPath << ": "
                 << ec.message() << "\n";
    return failure();
  }
  return success();
}

/// Creates  directories (if required) and opens an output file for the
/// FileTreeIRPrinterConfig.
static std::unique_ptr<llvm::ToolOutputFile>
createTreePrinterOutputPath(Operation *op, llvm::StringRef passArgument,
                            llvm::StringRef rootDir,
                            llvm::DenseMap<Operation *, unsigned> &counters) {
  // Create the path. We will create a tree rooted at the given 'rootDir'
  // directory. The root directory will contain folders with the names of
  // modules. Sub-directories within those folders mirror the nesting
  // structure of the pass manager, using symbol names for directory names.
  auto [opAndSymbolNames, fileName] =
      getOpAndSymbolNames(op, passArgument, counters);

  // Create all the directories, starting at the root. Abort early if we fail to
  // create any directory.
  llvm::SmallString<128> path(rootDir);
  if (failed(createDirectoryOrPrintErr(path)))
    return nullptr;

  for (const auto &[opName, symbolName] : opAndSymbolNames) {
    llvm::sys::path::append(path, opName + "_" + symbolName);
    if (failed(createDirectoryOrPrintErr(path)))
      return nullptr;
  }

  // Open output file.
  llvm::sys::path::append(path, fileName);
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> file = openOutputFile(path, &error);
  if (!file) {
    llvm::errs() << "Error opening output file " << path << ": " << error
                 << "\n";
    return nullptr;
  }
  return file;
}

namespace {
/// A configuration that prints the IR before/after each pass to a set of files
/// in the specified directory. The files are organized into subdirectories that
/// mirror the nesting structure of the IR.
struct FileTreeIRPrinterConfig : public PassManager::IRPrinterConfig {
  FileTreeIRPrinterConfig(
      std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
      std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
      bool printModuleScope, bool printAfterOnlyOnChange,
      bool printAfterOnlyOnFailure, OpPrintingFlags opPrintingFlags,
      llvm::StringRef treeDir)
      : IRPrinterConfig(printModuleScope, printAfterOnlyOnChange,
                        printAfterOnlyOnFailure, opPrintingFlags),
        shouldPrintBeforePass(std::move(shouldPrintBeforePass)),
        shouldPrintAfterPass(std::move(shouldPrintAfterPass)),
        treeDir(treeDir) {
    assert((this->shouldPrintBeforePass || this->shouldPrintAfterPass) &&
           "expected at least one valid filter function");
  }

  void printBeforeIfEnabled(Pass *pass, Operation *operation,
                            PrintCallbackFn printCallback) final {
    if (!shouldPrintBeforePass || !shouldPrintBeforePass(pass, operation))
      return;
    std::unique_ptr<llvm::ToolOutputFile> file = createTreePrinterOutputPath(
        operation, pass->getArgument(), treeDir, counters);
    if (!file)
      return;
    printCallback(file->os());
    file->keep();
  }

  void printAfterIfEnabled(Pass *pass, Operation *operation,
                           PrintCallbackFn printCallback) final {
    if (!shouldPrintAfterPass || !shouldPrintAfterPass(pass, operation))
      return;
    std::unique_ptr<llvm::ToolOutputFile> file = createTreePrinterOutputPath(
        operation, pass->getArgument(), treeDir, counters);
    if (!file)
      return;
    printCallback(file->os());
    file->keep();
  }

  /// Filter functions for before and after pass execution.
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass;

  /// Directory that should be used as the root of the file tree.
  std::string treeDir;

  /// Counters used for labeling the prefix. Every op which could be targeted by
  /// a pass gets its own counter.
  llvm::DenseMap<Operation *, unsigned> counters;
};

} // namespace

/// Add an instrumentation to print the IR before and after pass execution,
/// using the provided configuration.
void PassManager::enableIRPrinting(std::unique_ptr<IRPrinterConfig> config) {
  if (config->shouldPrintAtModuleScope() &&
      getContext()->isMultithreadingEnabled())
    llvm::report_fatal_error("IR printing can't be setup on a pass-manager "
                             "without disabling multi-threading first.");
  addInstrumentation(
      std::make_unique<IRPrinterInstrumentation>(std::move(config)));
}

/// Add an instrumentation to print the IR before and after pass execution.
void PassManager::enableIRPrinting(
    std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
    std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
    bool printModuleScope, bool printAfterOnlyOnChange,
    bool printAfterOnlyOnFailure, raw_ostream &out,
    OpPrintingFlags opPrintingFlags) {
  enableIRPrinting(std::make_unique<BasicIRPrinterConfig>(
      std::move(shouldPrintBeforePass), std::move(shouldPrintAfterPass),
      printModuleScope, printAfterOnlyOnChange, printAfterOnlyOnFailure,
      opPrintingFlags, out));
}

/// Add an instrumentation to print the IR before and after pass execution.
void PassManager::enableIRPrintingToFileTree(
    std::function<bool(Pass *, Operation *)> shouldPrintBeforePass,
    std::function<bool(Pass *, Operation *)> shouldPrintAfterPass,
    bool printModuleScope, bool printAfterOnlyOnChange,
    bool printAfterOnlyOnFailure, StringRef printTreeDir,
    OpPrintingFlags opPrintingFlags) {
  enableIRPrinting(std::make_unique<FileTreeIRPrinterConfig>(
      std::move(shouldPrintBeforePass), std::move(shouldPrintAfterPass),
      printModuleScope, printAfterOnlyOnChange, printAfterOnlyOnFailure,
      opPrintingFlags, printTreeDir));
}
