//===- TestDialect.h - MLIR Dialect for testing -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a fake 'test' dialect that can be used for testing things
// that do not have a respective counterpart in the main source directories.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTDIALECT_H
#define MLIR_TESTDIALECT_H

#include "TestTypes.h"
#include "TestAttributes.h"
#include "TestInterfaces.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include <memory>

namespace mlir {
class DLTIDialect;
class RewritePatternSet;
} // namespace mlir

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "TestOpInterfaces.h.inc"
#include "TestOpsDialect.h.inc"

namespace test {
// Define some classes to exercises the Properties feature.

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
class MyPropStruct {
public:
  std::string content;
  // These three methods are invoked through the  `MyStructProperty` wrapper
  // defined in TestOps.td
  mlir::Attribute asAttribute(mlir::MLIRContext *ctx) const;
  static mlir::LogicalResult setFromAttr(MyPropStruct &prop,
                                         mlir::Attribute attr,
                                         mlir::InFlightDiagnostic *diag);
  llvm::hash_code hash() const;
  bool operator==(const MyPropStruct &rhs) const {
    return content == rhs.content;
  }
};
} // namespace test

#define GET_OP_CLASSES
#include "TestOps.h.inc"

namespace test {

// Op deliberately defined in C++ code rather than ODS to test that C++
// Ops can still use the old `fold` method.
class ManualCppOpWithFold
    : public mlir::Op<ManualCppOpWithFold, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() {
    return "test.manual_cpp_op_with_fold";
  }

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

  mlir::OpFoldResult fold(llvm::ArrayRef<mlir::Attribute> attributes);
};

void registerTestDialect(::mlir::DialectRegistry &registry);
void populateTestReductionPatterns(::mlir::RewritePatternSet &patterns);
} // namespace test

#endif // MLIR_TESTDIALECT_H
