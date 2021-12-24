//===- OpClass.h - Implementation of an Op Class --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_OPCLASS_H_
#define MLIR_TOOLS_MLIRTBLGEN_OPCLASS_H_

#include "mlir/TableGen/Class.h"

namespace mlir {
namespace tblgen {

/// Class for holding an op for C++ code emission. The class is specialized to
/// add Op-specific declarations to the class.
class OpClass : public Class {
public:
  /// Create an operation class with extra class declarations, whose default
  /// visibility is public. Also declares at the top of the class:
  ///
  /// - inheritance of constructors from `Op`
  /// - inheritance of `print`
  /// - a type alias for the associated adaptor class
  ///
  OpClass(StringRef name, StringRef extraClassDeclaration);

  /// Add an op trait.
  void addTrait(Twine trait) { parent.addTemplateParam(trait.str()); }

  /// The operation class is finalized by calling `Class::finalize` to delcare
  /// all pending private and public methods (ops don't have custom constructors
  /// or fields). Then, the extra class declarations are appended to the end of
  /// the class declaration.
  void finalize() override;

private:
  /// Hand-written extra class declarations.
  StringRef extraClassDeclaration;
  /// The parent class, which also contains the traits to be inherited.
  ParentClass &parent;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_OPCLASS_H_
