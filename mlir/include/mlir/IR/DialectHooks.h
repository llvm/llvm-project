//===- DialectHooks.h - MLIR DialectHooks mechanism -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines abstraction and registration mechanism for dialect hooks.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECT_HOOKS_H
#define MLIR_IR_DIALECT_HOOKS_H

#include "mlir/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
using DialectHooksSetter = std::function<void(MLIRContext *)>;

/// Dialect hooks allow external components to register their functions to
/// be called for specific tasks specialized per dialect, such as decoding
/// of opaque constants. To register concrete dialect hooks, one should
/// define a DialectHooks subclass and use it as a template
/// argument to DialectHooksRegistration. For example,
///     class MyHooks : public DialectHooks {...};
///     static DialectHooksRegistration<MyHooks, MyDialect> hooksReg;
/// The subclass should override DialectHook methods for supported hooks.
class DialectHooks {
public:
  // Returns hook to constant fold an operation.
  DialectConstantFoldHook getConstantFoldHook() { return nullptr; }
  // Returns hook to decode opaque constant tensor.
  DialectConstantDecodeHook getDecodeHook() { return nullptr; }
  // Returns hook to extract an element of an opaque constant tensor.
  DialectExtractElementHook getExtractElementHook() { return nullptr; }

private:
  /// Registers a function that will set hooks in the registered dialects.
  /// Registrations are deduplicated by dialect TypeID and only the first
  /// registration will be used.
  static void registerDialectHooksSetter(TypeID typeID,
                                         const DialectHooksSetter &function);
  template <typename ConcreteHooks>
  friend void registerDialectHooks(StringRef dialectName);
};

void registerDialectHooksSetter(TypeID typeID,
                                const DialectHooksSetter &function);

/// Utility to register dialect hooks. Client can register their dialect hooks
/// with the global registry by calling
/// registerDialectHooks<MyHooks>("dialect_namespace");
template <typename ConcreteHooks>
void registerDialectHooks(StringRef dialectName) {
  DialectHooks::registerDialectHooksSetter(
      TypeID::get<ConcreteHooks>(), [dialectName](MLIRContext *ctx) {
        Dialect *dialect = ctx->getRegisteredDialect(dialectName);
        if (!dialect) {
          llvm::errs() << "error: cannot register hooks for unknown dialect '"
                       << dialectName << "'\n";
          abort();
        }
        // Set hooks.
        ConcreteHooks hooks;
        if (auto h = hooks.getConstantFoldHook())
          dialect->constantFoldHook = h;
        if (auto h = hooks.getDecodeHook())
          dialect->decodeHook = h;
        if (auto h = hooks.getExtractElementHook())
          dialect->extractElementHook = h;
      });
}

/// DialectHooksRegistration provides a global initializer that registers
/// a dialect hooks setter routine.
/// Usage:
///
///   // At namespace scope.
///   static DialectHooksRegistration<MyHooks> Unused("dialect_namespace");
template <typename ConcreteHooks> struct DialectHooksRegistration {
  DialectHooksRegistration(StringRef dialectName) {
    registerDialectHooks<ConcreteHooks>(dialectName);
  }
};

} // namespace mlir

#endif
