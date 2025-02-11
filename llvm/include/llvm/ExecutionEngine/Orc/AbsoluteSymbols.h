//===------ AbsoluteSymbols.h - Absolute symbols utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// absoluteSymbols function and related utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ABSOLUTESYMBOLS_H
#define LLVM_EXECUTIONENGINE_ORC_ABSOLUTESYMBOLS_H

#include "llvm/ExecutionEngine/Orc/MaterializationUnit.h"

namespace llvm::orc {

/// A MaterializationUnit implementation for pre-existing absolute symbols.
///
/// All symbols will be resolved and marked ready as soon as the unit is
/// materialized.
class AbsoluteSymbolsMaterializationUnit : public MaterializationUnit {
public:
  AbsoluteSymbolsMaterializationUnit(SymbolMap Symbols);

  StringRef getName() const override;

private:
  void materialize(std::unique_ptr<MaterializationResponsibility> R) override;
  void discard(const JITDylib &JD, const SymbolStringPtr &Name) override;
  static MaterializationUnit::Interface extractFlags(const SymbolMap &Symbols);

  SymbolMap Symbols;
};

/// Create an AbsoluteSymbolsMaterializationUnit with the given symbols.
/// Useful for inserting absolute symbols into a JITDylib. E.g.:
/// \code{.cpp}
///   JITDylib &JD = ...;
///   SymbolStringPtr Foo = ...;
///   ExecutorSymbolDef FooSym = ...;
///   if (auto Err = JD.define(absoluteSymbols({
///         { Foo, FooSym },
///         { Bar, BarSym }
///       })))
///     return Err;
/// \endcode
///
inline std::unique_ptr<AbsoluteSymbolsMaterializationUnit>
absoluteSymbols(SymbolMap Symbols) {
  return std::make_unique<AbsoluteSymbolsMaterializationUnit>(
      std::move(Symbols));
}

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_ABSOLUTESYMBOLS_H
