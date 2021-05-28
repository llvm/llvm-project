//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect wrapper to simplify using TableGen Record defining a MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_DIALECT_H_
#define MLIR_TABLEGEN_DIALECT_H_

#include "mlir/Support/LLVM.h"
#include <string>
#include <vector>

namespace llvm {
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {
// Wrapper class that contains a MLIR dialect's information defined in TableGen
// and provides helper methods for accessing them.
class Dialect {
public:
  explicit Dialect(const llvm::Record *def);

  // Returns the name of this dialect.
  StringRef getName() const;

  // Returns the C++ namespaces that ops of this dialect should be placed into.
  StringRef getCppNamespace() const;

  // Returns this dialect's C++ class name.
  std::string getCppClassName() const;

  // Returns the summary description of the dialect. Returns empty string if
  // none.
  StringRef getSummary() const;

  // Returns the description of the dialect. Returns empty string if none.
  StringRef getDescription() const;

  // Returns the list of dialect (class names) that this dialect depends on.
  // These are dialects that will be loaded on construction of this dialect.
  ArrayRef<StringRef> getDependentDialects() const;

  // Returns the dialects extra class declaration code.
  llvm::Optional<StringRef> getExtraClassDeclaration() const;

  /// Returns true if this dialect has a canonicalizer.
  bool hasCanonicalizer() const;

  // Returns true if this dialect has a constant materializer.
  bool hasConstantMaterializer() const;

  /// Returns true if this dialect has an operation attribute verifier.
  bool hasOperationAttrVerify() const;

  /// Returns true if this dialect has a region argument attribute verifier.
  bool hasRegionArgAttrVerify() const;

  /// Returns true if this dialect has a region result attribute verifier.
  bool hasRegionResultAttrVerify() const;

  /// Returns true if this dialect has fallback interfaces for its operations.
  bool hasOperationInterfaceFallback() const;

  // Returns whether two dialects are equal by checking the equality of the
  // underlying record.
  bool operator==(const Dialect &other) const;

  bool operator!=(const Dialect &other) const { return !(*this == other); }

  // Compares two dialects by comparing the names of the dialects.
  bool operator<(const Dialect &other) const;

  // Returns whether the dialect is defined.
  explicit operator bool() const { return def != nullptr; }

private:
  const llvm::Record *def;
  std::vector<StringRef> dependentDialects;
};
} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_DIALECT_H_
