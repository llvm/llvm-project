//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Toy language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_DIALECT_H_
#define MLIR_TUTORIAL_TOY_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "toy/ShapeInferenceInterface.h"

namespace mlir {
namespace toy {
namespace detail {
struct StructTypeStorage;
} // end namespace detail

/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// A hook used to materialize constant values with the given type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  /// Parse an instance of a type registered to the toy dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to the toy dialect.
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "toy"; }
};

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

//===----------------------------------------------------------------------===//
// Toy Types
//===----------------------------------------------------------------------===//

/// Create a local enumeration with all of the types that are defined by Toy.
namespace ToyTypes {
enum Types {
  Struct = mlir::Type::FIRST_TOY_TYPE,
};
} // end namespace ToyTypes

/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
} // end namespace toy
} // end namespace mlir

#endif // MLIR_TUTORIAL_TOY_DIALECT_H_
