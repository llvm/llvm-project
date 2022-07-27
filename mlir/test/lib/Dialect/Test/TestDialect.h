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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
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

namespace mlir {
class DLTIDialect;
class RewritePatternSet;
} // namespace mlir

namespace test {
class TestDialect;

//===----------------------------------------------------------------------===//
// External Elements Data
//===----------------------------------------------------------------------===//

/// This class represents a single external elements instance. It keeps track of
/// the data, and deallocates when destructed.
class TestExternalElementsData : public mlir::AsmResourceBlob {
public:
  using mlir::AsmResourceBlob::AsmResourceBlob;
  TestExternalElementsData(mlir::AsmResourceBlob &&blob)
      : mlir::AsmResourceBlob(std::move(blob)) {}

  /// Return the data of this external elements instance.
  llvm::ArrayRef<uint64_t> getData() const;

  /// Allocate a new external elements instance with the given number of
  /// elements.
  static TestExternalElementsData allocate(size_t numElements);
};

/// A handle used to reference external elements instances.
struct TestExternalElementsDataHandle
    : public mlir::AsmDialectResourceHandleBase<
          TestExternalElementsDataHandle,
          llvm::StringMapEntry<std::unique_ptr<TestExternalElementsData>>,
          TestDialect> {
  using AsmDialectResourceHandleBase::AsmDialectResourceHandleBase;

  /// Return a key to use for this handle.
  llvm::StringRef getKey() const { return getResource()->getKey(); }

  /// Return the data referenced by this handle.
  TestExternalElementsData *getData() const {
    return getResource()->getValue().get();
  }
};

/// This class acts as a manager for external elements data. It provides API
/// for creating and accessing registered elements data.
class TestExternalElementsDataManager {
  using DataMap = llvm::StringMap<std::unique_ptr<TestExternalElementsData>>;

public:
  /// Return the data registered for the given name, or nullptr if no data is
  /// registered.
  const TestExternalElementsData *getData(llvm::StringRef name) const;

  /// Register an entry with the provided name, which may be modified if another
  /// entry was already inserted with that name. Returns the inserted entry.
  std::pair<DataMap::iterator, bool> insert(llvm::StringRef name);

  /// Set the data for the given entry, which is expected to exist.
  void setData(llvm::StringRef name, TestExternalElementsData &&data);

private:
  llvm::StringMap<std::unique_ptr<TestExternalElementsData>> dataMap;
};
} // namespace test

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "TestOpInterfaces.h.inc"
#include "TestOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TestOps.h.inc"

namespace test {
void registerTestDialect(::mlir::DialectRegistry &registry);
void populateTestReductionPatterns(::mlir::RewritePatternSet &patterns);
} // namespace test

#endif // MLIR_TESTDIALECT_H
