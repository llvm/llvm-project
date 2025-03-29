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

#include "TestAttributes.h"
#include "TestInterfaces.h"
#include "TestTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
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
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"

#include <memory>

namespace mlir {
class RewritePatternSet;
} // end namespace mlir

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "TestOpsDialect.h.inc"

namespace test {

//===----------------------------------------------------------------------===//
// TestDialect version utilities
//===----------------------------------------------------------------------===//

struct TestDialectVersion : public mlir::DialectVersion {
  TestDialectVersion() = default;
  TestDialectVersion(uint32_t majorVersion, uint32_t minorVersion)
      : major_(majorVersion), minor_(minorVersion){};
  // We cannot use 'major' and 'minor' here because these identifiers may
  // already be used by <sys/types.h> on many POSIX systems including Linux and
  // FreeBSD.
  uint32_t major_ = 2;
  uint32_t minor_ = 0;
};

} // namespace test

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
void testSideEffectOpGetEffect(
    mlir::Operation *op,
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::TestEffects::Effect>> &effects);
} // namespace test

#endif // MLIR_TESTDIALECT_H
