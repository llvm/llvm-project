//===- Translation.cpp - Translation registry -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of the translation registry.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Translation CommandLine Options
//===----------------------------------------------------------------------===//

struct TranslationOptions {
  llvm::cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      llvm::cl::desc("Disable the parsing of an implicit top-level module op"),
      llvm::cl::init(false)};
};

static llvm::ManagedStatic<TranslationOptions> clOptions;

void mlir::registerTranslationCLOptions() { *clOptions; }

//===----------------------------------------------------------------------===//
// Translation Registry
//===----------------------------------------------------------------------===//

/// Get the mutable static map between registered file-to-file MLIR
/// translations.
static llvm::StringMap<Translation> &getTranslationRegistry() {
  static llvm::StringMap<Translation> translationBundle;
  return translationBundle;
}

/// Register the given translation.
static void registerTranslation(StringRef name, StringRef description,
                                Optional<llvm::Align> inputAlignment,
                                const TranslateFunction &function) {
  auto &registry = getTranslationRegistry();
  if (registry.count(name))
    llvm::report_fatal_error(
        "Attempting to overwrite an existing <file-to-file> function");
  assert(function &&
         "Attempting to register an empty translate <file-to-file> function");
  registry[name] = Translation(function, description, inputAlignment);
}

TranslateRegistration::TranslateRegistration(
    StringRef name, StringRef description, const TranslateFunction &function) {
  registerTranslation(name, description, /*inputAlignment=*/llvm::None,
                      function);
}

//===----------------------------------------------------------------------===//
// Translation to MLIR
//===----------------------------------------------------------------------===//

// Puts `function` into the to-MLIR translation registry unless there is already
// a function registered for the same name.
static void registerTranslateToMLIRFunction(
    StringRef name, StringRef description, Optional<llvm::Align> inputAlignment,
    const TranslateSourceMgrToMLIRFunction &function) {
  auto wrappedFn = [function](llvm::SourceMgr &sourceMgr, raw_ostream &output,
                              MLIRContext *context) {
    OwningOpRef<Operation *> op = function(sourceMgr, context);
    if (!op || failed(verify(*op)))
      return failure();
    op.get()->print(output);
    return success();
  };
  registerTranslation(name, description, inputAlignment, wrappedFn);
}

TranslateToMLIRRegistration::TranslateToMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateSourceMgrToMLIRFunction &function,
    Optional<llvm::Align> inputAlignment) {
  registerTranslateToMLIRFunction(name, description, inputAlignment, function);
}
/// Wraps `function` with a lambda that extracts a StringRef from a source
/// manager and registers the wrapper lambda as a to-MLIR conversion.
TranslateToMLIRRegistration::TranslateToMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateStringRefToMLIRFunction &function,
    Optional<llvm::Align> inputAlignment) {
  registerTranslateToMLIRFunction(
      name, description, inputAlignment,
      [function](llvm::SourceMgr &sourceMgr, MLIRContext *ctx) {
        const llvm::MemoryBuffer *buffer =
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        return function(buffer->getBuffer(), ctx);
      });
}

//===----------------------------------------------------------------------===//
// Translation from MLIR
//===----------------------------------------------------------------------===//

TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, StringRef description,
    const TranslateFromMLIRFunction &function,
    const std::function<void(DialectRegistry &)> &dialectRegistration) {
  registerTranslation(
      name, description, /*inputAlignment=*/llvm::None,
      [function, dialectRegistration](llvm::SourceMgr &sourceMgr,
                                      raw_ostream &output,
                                      MLIRContext *context) {
        DialectRegistry registry;
        dialectRegistration(registry);
        context->appendDialectRegistry(registry);
        bool implicitModule =
            (!clOptions.isConstructed() || !clOptions->noImplicitModule);
        OwningOpRef<Operation *> op =
            parseSourceFileForTool(sourceMgr, context, implicitModule);
        if (!op || failed(verify(*op)))
          return failure();
        return function(op.get(), output);
      });
}

//===----------------------------------------------------------------------===//
// Translation Parser
//===----------------------------------------------------------------------===//

TranslationParser::TranslationParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const Translation *>(opt) {
  for (const auto &kv : getTranslationRegistry())
    addLiteralOption(kv.first(), &kv.second, kv.second.getDescription());
}

void TranslationParser::printOptionInfo(const llvm::cl::Option &o,
                                        size_t globalWidth) const {
  TranslationParser *tp = const_cast<TranslationParser *>(this);
  llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                       [](const TranslationParser::OptionInfo *lhs,
                          const TranslationParser::OptionInfo *rhs) {
                         return lhs->Name.compare(rhs->Name);
                       });
  llvm::cl::parser<const Translation *>::printOptionInfo(o, globalWidth);
}
