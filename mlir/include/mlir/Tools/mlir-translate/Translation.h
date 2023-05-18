//===- Translation.h - Translation registry ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry for user-provided translations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTRANSLATE_TRANSLATION_H
#define MLIR_TOOLS_MLIRTRANSLATE_TRANSLATION_H

#include "mlir/IR/Operation.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace mlir {
template <typename OpTy>
class OwningOpRef;

/// Interface of the function that translates the sources managed by `sourceMgr`
/// to MLIR. The source manager has at least one buffer. The implementation
/// should create a new MLIR Operation in the given context and return a
/// pointer to it, or a nullptr in case of any error.
using TranslateSourceMgrToMLIRFunction = std::function<OwningOpRef<Operation *>(
    const std::shared_ptr<llvm::SourceMgr> &sourceMgr, MLIRContext *)>;
using TranslateRawSourceMgrToMLIRFunction =
    std::function<OwningOpRef<Operation *>(llvm::SourceMgr &sourceMgr,
                                           MLIRContext *)>;

/// Interface of the function that translates the given string to MLIR. The
/// implementation should create a new MLIR Operation in the given context. If
/// source-related error reporting is required from within the function, use
/// TranslateSourceMgrToMLIRFunction instead.
using TranslateStringRefToMLIRFunction =
    std::function<OwningOpRef<Operation *>(llvm::StringRef, MLIRContext *)>;

/// Interface of the function that translates MLIR to a different format and
/// outputs the result to a stream. It is allowed to modify the operation.
using TranslateFromMLIRFunction =
    std::function<LogicalResult(Operation *, llvm::raw_ostream &output)>;

/// Interface of the function that performs file-to-file translation involving
/// MLIR. The input file is held in the given MemoryBuffer; the output file
/// should be written to the given raw_ostream. The implementation should create
/// all MLIR constructs needed during the process inside the given context. This
/// can be used for round-tripping external formats through the MLIR system.
using TranslateFunction = std::function<LogicalResult(
    const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
    llvm::raw_ostream &output, MLIRContext *)>;

/// Interface of the function that adds all dialects and dialect extensions used
/// for the translation to the given DialectRegistry.
using DialectRegistrationFunction = std::function<void(DialectRegistry &)>;

/// This class contains all of the components necessary for performing a
/// translation.
class Translation {
public:
  Translation() = default;
  Translation(TranslateFunction function, StringRef description,
              std::optional<llvm::Align> inputAlignment)
      : function(std::move(function)), description(description),
        inputAlignment(inputAlignment) {}

  /// Return the description of this translation.
  StringRef getDescription() const { return description; }

  /// Return the optional alignment desired for the input of the translation.
  std::optional<llvm::Align> getInputAlignment() const {
    return inputAlignment;
  }

  /// Invoke the translation function with the given input and output streams.
  LogicalResult operator()(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                           llvm::raw_ostream &output,
                           MLIRContext *context) const {
    return function(sourceMgr, output, context);
  }

private:
  /// The underlying translation function.
  TranslateFunction function;

  /// The description of the translation.
  StringRef description;

  /// An optional alignment desired for the input of the translation.
  std::optional<llvm::Align> inputAlignment;
};

/// Use Translate[ToMLIR|FromMLIR]Registration as an initializer that
/// registers a function and associates it with name. This requires that a
/// translation has not been registered to a given name. `inputAlign` is an
/// optional expected alignment for the input data.
///
/// Usage:
///
///   // At file scope.
///   namespace mlir {
///   void registerTRexToMLIRRegistration() {
///     TranslateToMLIRRegistration Unused(&MySubCommand, [] { ... });
///   }
///   } // namespace mlir
///
/// \{
struct TranslateToMLIRRegistration {
  TranslateToMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const TranslateSourceMgrToMLIRFunction &function,
      const DialectRegistrationFunction &dialectRegistration =
          [](DialectRegistry &) {},
      std::optional<llvm::Align> inputAlignment = std::nullopt);
  TranslateToMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const TranslateRawSourceMgrToMLIRFunction &function,
      const DialectRegistrationFunction &dialectRegistration =
          [](DialectRegistry &) {},
      std::optional<llvm::Align> inputAlignment = std::nullopt);
  TranslateToMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const TranslateStringRefToMLIRFunction &function,
      const DialectRegistrationFunction &dialectRegistration =
          [](DialectRegistry &) {},
      std::optional<llvm::Align> inputAlignment = std::nullopt);
};

struct TranslateFromMLIRRegistration {
  TranslateFromMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const TranslateFromMLIRFunction &function,
      const DialectRegistrationFunction &dialectRegistration =
          [](DialectRegistry &) {});

  template <typename FuncTy, typename OpTy = detail::first_argument<FuncTy>,
            typename = std::enable_if_t<!std::is_same_v<OpTy, Operation *>>>
  TranslateFromMLIRRegistration(
      llvm::StringRef name, llvm::StringRef description, FuncTy function,
      const DialectRegistrationFunction &dialectRegistration =
          [](DialectRegistry &) {})
      : TranslateFromMLIRRegistration(
            name, description,
            [function](Operation *op, raw_ostream &os) -> LogicalResult {
              if (auto casted = dyn_cast<OpTy>(op))
                return function(casted, os);
              return emitError(op->getLoc())
                     << "expected a '" << OpTy::getOperationName()
                     << "' op, got '" << op->getName().getStringRef() << "'";
            },
            dialectRegistration) {}
};
struct TranslateRegistration {
  TranslateRegistration(llvm::StringRef name, llvm::StringRef description,
                        const TranslateFunction &function);
};
/// \}

/// A command line parser for translation functions.
struct TranslationParser : public llvm::cl::parser<const Translation *> {
  TranslationParser(llvm::cl::Option &opt);

  void printOptionInfo(const llvm::cl::Option &o,
                       size_t globalWidth) const override;
};

/// Register command-line options used by the translation registry.
void registerTranslationCLOptions();

} // namespace mlir

#endif // MLIR_TOOLS_MLIRTRANSLATE_TRANSLATION_H
