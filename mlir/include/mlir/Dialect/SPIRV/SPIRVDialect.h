//===- SPIRVDialect.h - MLIR SPIR-V dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SPIR-V dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_
#define MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace spirv {

enum class Decoration : uint32_t;

class SPIRVDialect : public Dialect {
public:
  explicit SPIRVDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "spv"; }

  //===--------------------------------------------------------------------===//
  // Type
  //===--------------------------------------------------------------------===//

  /// Checks if the given `type` is valid in SPIR-V dialect.
  static bool isValidType(Type type);

  /// Checks if the given `scalar type` is valid in SPIR-V dialect.
  static bool isValidScalarType(Type type);

  /// Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  /// Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  //===--------------------------------------------------------------------===//
  // Attribute
  //===--------------------------------------------------------------------===//

  /// Returns the attribute name to use when specifying decorations on results
  /// of operations.
  static std::string getAttributeName(Decoration decoration);

  /// Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  /// Prints an attribute registered to this dialect.
  void printAttribute(Attribute, DialectAsmPrinter &printer) const override;

  /// Provides a hook for verifying SPIR-V dialect attributes attached to the
  /// given op.
  LogicalResult verifyOperationAttribute(Operation *op,
                                         NamedAttribute attribute) override;

  /// Provides a hook for verifying SPIR-V dialect attributes attached to the
  /// given op's region argument.
  LogicalResult verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                         unsigned argIndex,
                                         NamedAttribute attribute) override;

  /// Provides a hook for verifying SPIR-V dialect attributes attached to the
  /// given op's region result.
  LogicalResult verifyRegionResultAttribute(Operation *op, unsigned regionIndex,
                                            unsigned resultIndex,
                                            NamedAttribute attribute) override;

  //===--------------------------------------------------------------------===//
  // Constant
  //===--------------------------------------------------------------------===//

  /// Provides a hook for materializing a constant to this dialect.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_
