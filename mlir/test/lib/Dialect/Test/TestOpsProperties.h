//===- TestOpsProperties.h - Test dialect ops in `Properties` ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTOPSPROPERTIES_H
#define MLIR_TEST_DIALECT_TEST_TESTOPSPROPERTIES_H

#include "TestDialect.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace test {
// Keep this type distinct from SmallVector<int32_t>, which another test parser
// uses with the generic FieldParser specialization in a separate translation
// unit.
struct KeyValueSpecializedList : llvm::SmallVector<int32_t> {
  using llvm::SmallVector<int32_t>::SmallVector;
};
} // namespace test

namespace mlir {
// Self-delimiting full specializations remain usable in a keyed prop-dict even
// when their storage type is optional or container-like.
template <>
struct FieldParser<test::KeyValueSpecializedList> {
  static FailureOr<test::KeyValueSpecializedList> parse(AsmParser &parser) {
    test::KeyValueSpecializedList values;
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
          int32_t value;
          if (parser.parseInteger(value))
            return failure();
          values.push_back(value);
          return success();
        }))
      return failure();
    return values;
  }
};

template <>
struct FieldParser<std::optional<int16_t>> {
  static FailureOr<std::optional<int16_t>> parse(AsmParser &parser) {
    if (succeeded(parser.parseOptionalKeyword("none")))
      return std::optional<int16_t>{};
    int16_t value;
    if (parser.parseKeyword("some") || parser.parseLess() ||
        parser.parseInteger(value) || parser.parseGreater())
      return failure();
    return std::optional<int16_t>{value};
  }
};
} // namespace mlir

namespace test {
//===----------------------------------------------------------------------===//
// PropertiesWithCustomPrint
//===----------------------------------------------------------------------===//

struct PropertiesWithCustomPrint {
  /// A shared_ptr to a const object is safe: it is equivalent to a value-based
  /// member. Here the label will be deallocated when the last operation
  /// refering to it is destroyed. However there is no pool-allocation: this is
  /// offloaded to the client.
  std::shared_ptr<const std::string> label;
  int value;
  bool operator==(const PropertiesWithCustomPrint &rhs) const {
    return value == rhs.value && *label == *rhs.label;
  }
};

llvm::LogicalResult setPropertiesFromAttribute(
    PropertiesWithCustomPrint &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);
mlir::DictionaryAttr
getPropertiesAsAttribute(mlir::MLIRContext *ctx,
                         const PropertiesWithCustomPrint &prop);
llvm::hash_code computeHash(const PropertiesWithCustomPrint &prop);
void customPrintProperties(mlir::OpAsmPrinter &p,
                           const PropertiesWithCustomPrint &prop);
mlir::ParseResult customParseProperties(mlir::OpAsmParser &parser,
                                        PropertiesWithCustomPrint &prop);

//===----------------------------------------------------------------------===//
// MyPropStruct
//===----------------------------------------------------------------------===//
namespace test_properties {
class MyPropStruct {
public:
  std::string content;
  // These three methods are invoked through the  `MyStructProperty` wrapper
  // defined in TestOps.td
  mlir::Attribute asAttribute(mlir::MLIRContext *ctx) const;
  static llvm::LogicalResult
  setFromAttr(MyPropStruct &prop, mlir::Attribute attr,
              llvm::function_ref<mlir::InFlightDiagnostic()> emitError);
  llvm::hash_code hash() const;
  bool operator==(const MyPropStruct &rhs) const {
    return content == rhs.content;
  }
};
inline llvm::hash_code hash_value(const MyPropStruct &S) { return S.hash(); }
} // namespace test_properties
using test_properties::MyPropStruct;

llvm::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         MyPropStruct &prop);
void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         MyPropStruct &prop);

//===----------------------------------------------------------------------===//
// VersionedProperties
//===----------------------------------------------------------------------===//

struct VersionedProperties {
  // For the sake of testing, assume that this object was associated to version
  // 1.2 of the test dialect when having only one int value. In the current
  // version 2.0, the property has two values. We also assume that the class is
  // upgrade-able if value2 = 0.
  int value1;
  int value2;
  bool operator==(const VersionedProperties &rhs) const {
    return value1 == rhs.value1 && value2 == rhs.value2;
  }
};

llvm::LogicalResult setPropertiesFromAttribute(
    VersionedProperties &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);
mlir::DictionaryAttr getPropertiesAsAttribute(mlir::MLIRContext *ctx,
                                              const VersionedProperties &prop);
llvm::hash_code computeHash(const VersionedProperties &prop);
void customPrintProperties(mlir::OpAsmPrinter &p,
                           const VersionedProperties &prop);
mlir::ParseResult customParseProperties(mlir::OpAsmParser &parser,
                                        VersionedProperties &prop);

//===----------------------------------------------------------------------===//
// Bytecode Support
//===----------------------------------------------------------------------===//

llvm::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         llvm::MutableArrayRef<int64_t> prop);
void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         llvm::ArrayRef<int64_t> prop);
} // namespace test

#define GET_OP_CLASSES
#include "TestOpsProperties.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTOPSPROPERTIES_H
