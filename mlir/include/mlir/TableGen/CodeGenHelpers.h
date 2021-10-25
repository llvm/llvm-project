//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating C++ from tablegen
// structures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CODEGENHELPERS_H
#define MLIR_TABLEGEN_CODEGENHELPERS_H

#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

class Constraint;

// Simple RAII helper for defining ifdef-undef-endif scopes.
class IfDefScope {
public:
  IfDefScope(llvm::StringRef name, llvm::raw_ostream &os)
      : name(name.str()), os(os) {
    os << "#ifdef " << name << "\n"
       << "#undef " << name << "\n\n";
  }
  ~IfDefScope() { os << "\n#endif  // " << name << "\n\n"; }

private:
  std::string name;
  llvm::raw_ostream &os;
};

// A helper RAII class to emit nested namespaces for this op.
class NamespaceEmitter {
public:
  NamespaceEmitter(raw_ostream &os, const Dialect &dialect) : os(os) {
    if (!dialect)
      return;
    emitNamespaceStarts(os, dialect.getCppNamespace());
  }
  NamespaceEmitter(raw_ostream &os, StringRef cppNamespace) : os(os) {
    emitNamespaceStarts(os, cppNamespace);
  }

  ~NamespaceEmitter() {
    for (StringRef ns : llvm::reverse(namespaces))
      os << "} // namespace " << ns << "\n";
  }

private:
  void emitNamespaceStarts(raw_ostream &os, StringRef cppNamespace) {
    llvm::SplitString(cppNamespace, namespaces, "::");
    for (StringRef ns : namespaces)
      os << "namespace " << ns << " {\n";
  }
  raw_ostream &os;
  SmallVector<StringRef, 2> namespaces;
};

/// This class deduplicates shared operation verification code by emitting
/// static functions alongside the op definitions. These methods are local to
/// the definition file, and are invoked within the operation verify methods.
/// An example is shown below:
///
/// static LogicalResult localVerify(...)
///
/// LogicalResult OpA::verify(...) {
///  if (failed(localVerify(...)))
///    return failure();
///  ...
/// }
///
/// LogicalResult OpB::verify(...) {
///  if (failed(localVerify(...)))
///    return failure();
///  ...
/// }
///
class StaticVerifierFunctionEmitter {
public:
  StaticVerifierFunctionEmitter(const llvm::RecordKeeper &records,
                                raw_ostream &os);

  /// Emit the static verifier functions for `llvm::Record`s. The
  /// `signatureFormat` describes the required arguments and it must have a
  /// placeholder for function name.
  /// Example,
  ///   const char *typeVerifierSignature =
  ///     "static ::mlir::LogicalResult {0}(::mlir::Operation *op, ::mlir::Type"
  ///     " type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex)";
  ///
  /// `errorHandlerFormat` describes the error message to return. It may have a
  /// placeholder for the summary of Constraint and bring more information for
  /// the error message.
  /// Example,
  ///   const char *typeVerifierErrorHandler =
  ///       " op->emitOpError(valueKind) << \" #\" << valueGroupStartIndex << "
  ///       "\" must be {0}, but got \" << type";
  ///
  /// `typeArgName` is used to identify the argument that needs to check its
  /// type. The constraint template will replace `$_self` with it.
  void emitFunctionsFor(StringRef signatureFormat, StringRef errorHandlerFormat,
                        StringRef typeArgName, ArrayRef<llvm::Record *> opDefs,
                        bool emitDecl);

  /// Get the name of the local function used for the given type constraint.
  /// These functions are used for operand and result constraints and have the
  /// form:
  ///   LogicalResult(Operation *op, Type type, StringRef valueKind,
  ///                 unsigned valueGroupStartIndex);
  StringRef getTypeConstraintFn(const Constraint &constraint) const;

private:
  /// Returns a unique name to use when generating local methods.
  static std::string getUniqueName(const llvm::RecordKeeper &records);

  /// Emit local methods for the type constraints used within the provided op
  /// definitions.
  void emitTypeConstraintMethods(StringRef signatureFormat,
                                 StringRef errorHandlerFormat,
                                 StringRef typeArgName,
                                 ArrayRef<llvm::Record *> opDefs,
                                 bool emitDecl);

  raw_indented_ostream os;

  /// A unique label for the file currently being generated. This is used to
  /// ensure that the local functions have a unique name.
  std::string uniqueOutputLabel;

  /// A set of functions implementing type constraints, used for operand and
  /// result verification.
  llvm::DenseMap<const void *, std::string> localTypeConstraints;
};

// Escape a string using C++ encoding. E.g. foo"bar -> foo\x22bar.
std::string escapeString(StringRef value);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CODEGENHELPERS_H
