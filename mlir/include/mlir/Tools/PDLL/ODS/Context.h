//===- Context.h - MLIR PDLL ODS Context ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_ODS_CONTEXT_H_
#define MLIR_TOOLS_PDLL_ODS_CONTEXT_H_

#include <string>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
class SMLoc;
} // namespace llvm

namespace mlir {
namespace pdll {
namespace ods {
class AttributeConstraint;
class Dialect;
class Operation;
class TypeConstraint;

/// This class contains all of the registered ODS operation classes.
class Context {
public:
  Context();
  ~Context();

  /// Insert a new attribute constraint with the context. Returns the inserted
  /// constraint, or a previously inserted constraint with the same name.
  const AttributeConstraint &insertAttributeConstraint(StringRef name,
                                                       StringRef summary,
                                                       StringRef cppClass);

  /// Insert a new type constraint with the context. Returns the inserted
  /// constraint, or a previously inserted constraint with the same name.
  const TypeConstraint &insertTypeConstraint(StringRef name, StringRef summary,
                                             StringRef cppClass);

  /// Insert a new dialect with the context. Returns the inserted dialect, or a
  /// previously inserted dialect with the same name.
  Dialect &insertDialect(StringRef name);

  /// Lookup a dialect registered with the given name, or null if no dialect
  /// with that name was inserted.
  const Dialect *lookupDialect(StringRef name) const;

  /// Return a range of all of the registered dialects.
  auto getDialects() const {
    return llvm::make_pointee_range(llvm::make_second_range(dialects));
  }

  /// Insert a new operation with the context. Returns the inserted operation,
  /// and a boolean indicating if the operation newly inserted (false if the
  /// operation already existed).
  std::pair<Operation *, bool>
  insertOperation(StringRef name, StringRef summary, StringRef desc, SMLoc loc);

  /// Lookup an operation registered with the given name, or null if no
  /// operation with that name is registered.
  const Operation *lookupOperation(StringRef name) const;

  /// Print the contents of this context to the provided stream.
  void print(raw_ostream &os) const;

private:
  llvm::StringMap<std::unique_ptr<AttributeConstraint>> attributeConstraints;
  llvm::StringMap<std::unique_ptr<Dialect>> dialects;
  llvm::StringMap<std::unique_ptr<TypeConstraint>> typeConstraints;
};
} // namespace ods
} // namespace pdll
} // namespace mlir

#endif // MLIR_PDL_pdll_ODS_CONTEXT_H_
