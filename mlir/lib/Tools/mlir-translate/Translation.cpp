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
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Translation Registry
//===----------------------------------------------------------------------===//

/// Get the mutable static map between registered file-to-file MLIR translations
/// and the TranslateFunctions that perform those translations.
static llvm::StringMap<TranslateFunction> &getTranslationRegistry() {
  static llvm::StringMap<TranslateFunction> translationRegistry;
  return translationRegistry;
}

/// Register the given translation.
static void registerTranslation(StringRef name,
                                const TranslateFunction &function) {
  auto &translationRegistry = getTranslationRegistry();
  if (translationRegistry.find(name) != translationRegistry.end())
    llvm::report_fatal_error(
        "Attempting to overwrite an existing <file-to-file> function");
  assert(function &&
         "Attempting to register an empty translate <file-to-file> function");
  translationRegistry[name] = function;
}

TranslateRegistration::TranslateRegistration(
    StringRef name, const TranslateFunction &function) {
  registerTranslation(name, function);
}

//===----------------------------------------------------------------------===//
// Translation to MLIR
//===----------------------------------------------------------------------===//

// Puts `function` into the to-MLIR translation registry unless there is already
// a function registered for the same name.
static void registerTranslateToMLIRFunction(
    StringRef name, const TranslateSourceMgrToMLIRFunction &function) {
  auto wrappedFn = [function](llvm::SourceMgr &sourceMgr, raw_ostream &output,
                              MLIRContext *context) {
    OwningOpRef<ModuleOp> module = function(sourceMgr, context);
    if (!module || failed(verify(*module)))
      return failure();
    module->print(output);
    return success();
  };
  registerTranslation(name, wrappedFn);
}

TranslateToMLIRRegistration::TranslateToMLIRRegistration(
    StringRef name, const TranslateSourceMgrToMLIRFunction &function) {
  registerTranslateToMLIRFunction(name, function);
}

/// Wraps `function` with a lambda that extracts a StringRef from a source
/// manager and registers the wrapper lambda as a to-MLIR conversion.
TranslateToMLIRRegistration::TranslateToMLIRRegistration(
    StringRef name, const TranslateStringRefToMLIRFunction &function) {
  registerTranslateToMLIRFunction(
      name, [function](llvm::SourceMgr &sourceMgr, MLIRContext *ctx) {
        const llvm::MemoryBuffer *buffer =
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        return function(buffer->getBuffer(), ctx);
      });
}

//===----------------------------------------------------------------------===//
// Translation from MLIR
//===----------------------------------------------------------------------===//

TranslateFromMLIRRegistration::TranslateFromMLIRRegistration(
    StringRef name, const TranslateFromMLIRFunction &function,
    const std::function<void(DialectRegistry &)> &dialectRegistration) {
  registerTranslation(name, [function, dialectRegistration](
                                llvm::SourceMgr &sourceMgr, raw_ostream &output,
                                MLIRContext *context) {
    DialectRegistry registry;
    dialectRegistration(registry);
    context->appendDialectRegistry(registry);
    auto module = parseSourceFile<ModuleOp>(sourceMgr, context);
    if (!module || failed(verify(*module)))
      return failure();
    return function(module.get(), output);
  });
}

//===----------------------------------------------------------------------===//
// Translation Parser
//===----------------------------------------------------------------------===//

TranslationParser::TranslationParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const TranslateFunction *>(opt) {
  for (const auto &kv : getTranslationRegistry())
    addLiteralOption(kv.first(), &kv.second, kv.first());
}

void TranslationParser::printOptionInfo(const llvm::cl::Option &o,
                                        size_t globalWidth) const {
  TranslationParser *tp = const_cast<TranslationParser *>(this);
  llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                       [](const TranslationParser::OptionInfo *lhs,
                          const TranslationParser::OptionInfo *rhs) {
                         return lhs->Name.compare(rhs->Name);
                       });
  llvm::cl::parser<const TranslateFunction *>::printOptionInfo(o, globalWidth);
}
