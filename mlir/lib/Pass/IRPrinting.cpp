//===- IRPrinting.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SHA1.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
//===----------------------------------------------------------------------===//
// OperationFingerPrint
//===----------------------------------------------------------------------===//

/// A unique fingerprint for a specific operation, and all of it's internal
/// operations.
class OperationFingerPrint {
public:
  OperationFingerPrint(Operation *topOp) {
    llvm::SHA1 hasher;

    // Hash each of the operations based upon their mutable bits:
    topOp->walk([&](Operation *op) {
      //   - Operation pointer
      addDataToHash(hasher, op);
      //   - Attributes
      addDataToHash(hasher, op->getAttrDictionary());
      //   - Blocks in Regions
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          addDataToHash(hasher, &block);
          for (BlockArgument arg : block.getArguments())
            addDataToHash(hasher, arg);
        }
      }
      //   - Location
      addDataToHash(hasher, op->getLoc().getAsOpaquePointer());
      //   - Operands
      for (Value operand : op->getOperands())
        addDataToHash(hasher, operand);
      //   - Successors
      for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i)
        addDataToHash(hasher, op->getSuccessor(i));
    });
    hash = hasher.result();
  }

  bool operator==(const OperationFingerPrint &other) const {
    return hash == other.hash;
  }
  bool operator!=(const OperationFingerPrint &other) const {
    return !(*this == other);
  }

private:
  template <typename T> void addDataToHash(llvm::SHA1 &hasher, const T &data) {
    hasher.update(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&data), sizeof(T)));
  }

  SmallString<20> hash;
};

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
} // end anonymous namespace

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
    out << "// -----// IR Dump Before " << pass->getName();
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
    out << "// -----// IR Dump After " << pass->getName();
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
    out << formatv("// -----// IR Dump After {0} Failed", pass->getName());
    printIR(op, config->shouldPrintAtModuleScope(), out,
            OpPrintingFlags().printGenericOpForm());
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
PassManager::IRPrinterConfig::~IRPrinterConfig() {}

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
        shouldPrintBeforePass(shouldPrintBeforePass),
        shouldPrintAfterPass(shouldPrintAfterPass), out(out) {
    assert((shouldPrintBeforePass || shouldPrintAfterPass) &&
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
} // end anonymous namespace

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
