//===--- CppTranslationInterface.h - Translation to Cpp iface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines dialect interfaces for translation to Cpp.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_CPPTRANSLATIONINTERFACE_H
#define MLIR_TARGET_CPP_CPPTRANSLATIONINTERFACE_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
struct CppEmitter;

/// Base class for dialect interfaces providing translation to Cpp. Dialects
/// should implement this interface with supported operation translations to
/// be registered and used with translate-to-cpp.
class CppTranslationDialectInterface
    : public DialectInterface::Base<CppTranslationDialectInterface> {
public:
  CppTranslationDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// Hook for derived dialect interface to provide op translation to Cpp.
  virtual LogicalResult emitOperation(Operation *op, CppEmitter &cppEmitter,
                                      bool trailingSemicolon) const {
    return failure();
  }
};

/// Interface collection for translation to Cpp, dispatches to a concrete
/// interface implementation based on the dialect to which the given op belongs.
class CppTranslationInterface
    : public DialectInterfaceCollection<CppTranslationDialectInterface> {
public:
  using Base::Base;

  /// Translates the given operation to Cpp using the derived dialect interface.
  virtual LogicalResult emitOperation(Operation *op, CppEmitter &cppEmitter,
                                      bool trailingSemicolon) const {
    if (const CppTranslationDialectInterface *iface = getInterfaceFor(op))
      return iface->emitOperation(op, cppEmitter, trailingSemicolon);
    return failure();
  }
};

} // namespace mlir

#endif // MLIR_TARGET_CPP_CPPTRANSLATIONINTERFACE_H
