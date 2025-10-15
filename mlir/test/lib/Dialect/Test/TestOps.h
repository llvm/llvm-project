//===- TestOps.h - MLIR Test Dialect Operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTOPS_H
#define MLIR_TESTOPS_H

#include "TestAttributes.h"
#include "TestInterfaces.h"
#include "TestTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"

namespace test {
class TestDialect;

//===----------------------------------------------------------------------===//
// TestResource
//===----------------------------------------------------------------------===//

/// A test resource for side effects.
struct TestResource : public mlir::SideEffects::Resource::Base<TestResource> {
  llvm::StringRef getName() final { return "<Test>"; }
};

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
#include "TestOps.h.inc"

#endif // MLIR_TESTOPS_H
