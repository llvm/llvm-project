//===- MLIRContext.h - MLIR Global Context Class ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MLIRCONTEXT_H
#define MLIR_IR_MLIRCONTEXT_H

#include "mlir/Support/LLVM.h"
#include <functional>
#include <memory>
#include <vector>

namespace mlir {
class AbstractOperation;
class DiagnosticEngine;
class Dialect;
class InFlightDiagnostic;
class Location;
class MLIRContextImpl;
class StorageUniquer;

/// MLIRContext is the top-level object for a collection of MLIR modules.  It
/// holds immortal uniqued objects like types, and the tables used to unique
/// them.
///
/// MLIRContext gets a redundant "MLIR" prefix because otherwise it ends up with
/// a very generic name ("Context") and because it is uncommon for clients to
/// interact with it.
///
class MLIRContext {
public:
  explicit MLIRContext();
  ~MLIRContext();

  /// Return information about all registered IR dialects.
  std::vector<Dialect *> getRegisteredDialects();

  /// Get a registered IR dialect with the given namespace. If an exact match is
  /// not found, then return nullptr.
  Dialect *getRegisteredDialect(StringRef name);

  /// Get a registered IR dialect for the given derived dialect type. The
  /// derived type must provide a static 'getDialectNamespace' method.
  template <typename T> T *getRegisteredDialect() {
    return static_cast<T *>(getRegisteredDialect(T::getDialectNamespace()));
  }

  /// Return true if we allow to create operation for unregistered dialects.
  bool allowsUnregisteredDialects();

  /// Enables creating operations in unregistered dialects.
  void allowUnregisteredDialects(bool allow = true);

  /// Return true if multi-threading is enabled by the context.
  bool isMultithreadingEnabled();

  /// Set the flag specifying if multi-threading is disabled by the context.
  void disableMultithreading(bool disable = true);
  void enableMultithreading(bool enable = true) {
    disableMultithreading(!enable);
  }

  /// Return true if we should attach the operation to diagnostics emitted via
  /// Operation::emit.
  bool shouldPrintOpOnDiagnostic();

  /// Set the flag specifying if we should attach the operation to diagnostics
  /// emitted via Operation::emit.
  void printOpOnDiagnostic(bool enable);

  /// Return true if we should attach the current stacktrace to diagnostics when
  /// emitted.
  bool shouldPrintStackTraceOnDiagnostic();

  /// Set the flag specifying if we should attach the current stacktrace when
  /// emitting diagnostics.
  void printStackTraceOnDiagnostic(bool enable);

  /// Return information about all registered operations.  This isn't very
  /// efficient: typically you should ask the operations about their properties
  /// directly.
  std::vector<AbstractOperation *> getRegisteredOperations();

  /// Return true if this operation name is registered in this context.
  bool isOperationRegistered(StringRef name);

  // This is effectively private given that only MLIRContext.cpp can see the
  // MLIRContextImpl type.
  MLIRContextImpl &getImpl() { return *impl; }

  /// Returns the diagnostic engine for this context.
  DiagnosticEngine &getDiagEngine();

  /// Returns the storage uniquer used for creating affine constructs.
  StorageUniquer &getAffineUniquer();

  /// Returns the storage uniquer used for constructing type storage instances.
  /// This should not be used directly.
  StorageUniquer &getTypeUniquer();

  /// Returns the storage uniquer used for constructing attribute storage
  /// instances. This should not be used directly.
  StorageUniquer &getAttributeUniquer();

private:
  const std::unique_ptr<MLIRContextImpl> impl;

  MLIRContext(const MLIRContext &) = delete;
  void operator=(const MLIRContext &) = delete;
};

//===----------------------------------------------------------------------===//
// MLIRContext CommandLine Options
//===----------------------------------------------------------------------===//

/// Register a set of useful command-line options that can be used to configure
/// various flags within the MLIRContext. These flags are used when constructing
/// an MLIR context for initialization.
void registerMLIRContextCLOptions();

} // end namespace mlir

#endif // MLIR_IR_MLIRCONTEXT_H
