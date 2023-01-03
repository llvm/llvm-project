//===- LLVMImportInterface.h - Import from LLVM interface -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines dialect interfaces for the LLVM IR import.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_LLVMIMPORTINTERFACE_H
#define MLIR_TARGET_LLVMIR_LLVMIMPORTINTERFACE_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/FormatVariadic.h"

namespace llvm {
class IRBuilderBase;
} // namespace llvm

namespace mlir {
namespace LLVM {
class ModuleImport;
} // namespace LLVM

/// Base class for dialect interfaces used to import LLVM IR. Dialects that can
/// be imported should provide an implementation of this interface for the
/// supported intrinsics. The interface may be implemented in a separate library
/// to avoid the "main" dialect library depending on LLVM IR. The interface can
/// be attached using the delayed registration mechanism available in
/// DialectRegistry.
class LLVMImportDialectInterface
    : public DialectInterface::Base<LLVMImportDialectInterface> {
public:
  LLVMImportDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// Hook for derived dialect interfaces to implement the import of
  /// intrinsics into MLIR.
  virtual LogicalResult
  convertIntrinsic(OpBuilder &builder, llvm::CallInst *inst,
                   LLVM::ModuleImport &moduleImport) const {
    return failure();
  }

  /// Hook for derived dialect interfaces to publish the supported intrinsics.
  /// As every LLVM IR intrinsic has a unique integer identifier, the function
  /// returns the list of supported intrinsic identifiers.
  virtual ArrayRef<unsigned> getSupportedIntrinsics() const { return {}; }
};

/// Interface collection for the import of LLVM IR that dispatches to a concrete
/// dialect interface implementation. Queries the dialect interfaces to obtain a
/// list of the supported LLVM IR constructs and then builds a mapping for the
/// efficient dispatch.
class LLVMImportInterface
    : public DialectInterfaceCollection<LLVMImportDialectInterface> {
public:
  using Base::Base;

  /// Queries all dialect interfaces to build a map from intrinsic identifiers
  /// to the dialect interface that supports importing the intrinsic. Returns
  /// failure if multiple dialect interfaces translate the same LLVM IR
  /// intrinsic.
  LogicalResult initializeImport() {
    for (const LLVMImportDialectInterface &iface : *this) {
      // Verify the supported intrinsics have not been mapped before.
      const auto *it =
          llvm::find_if(iface.getSupportedIntrinsics(), [&](unsigned id) {
            return intrinsicToDialect.count(id);
          });
      if (it != iface.getSupportedIntrinsics().end()) {
        return emitError(
            UnknownLoc::get(iface.getContext()),
            llvm::formatv("expected unique conversion for intrinsic ({0}), but "
                          "got conflicting {1} and {2} conversions",
                          *it, iface.getDialect()->getNamespace(),
                          intrinsicToDialect.lookup(*it)->getNamespace()));
      }
      // Add a mapping for all supported intrinsic identifiers.
      for (unsigned id : iface.getSupportedIntrinsics())
        intrinsicToDialect[id] = iface.getDialect();
    }

    return success();
  }

  /// Converts the LLVM intrinsic to an MLIR operation if a conversion exists.
  /// Returns failure otherwise.
  LogicalResult convertIntrinsic(OpBuilder &builder, llvm::CallInst *inst,
                                 LLVM::ModuleImport &moduleImport) const {
    // Lookup the dialect interface for the given intrinsic.
    Dialect *dialect = intrinsicToDialect.lookup(inst->getIntrinsicID());
    if (!dialect)
      return failure();

    // Dispatch the conversion to the dialect interface.
    const LLVMImportDialectInterface *iface = getInterfaceFor(dialect);
    assert(iface && "expected to find a dialect interface");
    return iface->convertIntrinsic(builder, inst, moduleImport);
  }

  /// Returns true if the given LLVM IR intrinsic is convertible to an MLIR
  /// operation.
  bool isConvertibleIntrinsic(llvm::Intrinsic::ID id) {
    return intrinsicToDialect.count(id);
  }

private:
  DenseMap<unsigned, Dialect *> intrinsicToDialect;
};

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_LLVMIMPORTINTERFACE_H
