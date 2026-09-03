//===- OpenMPCommon.h - Utils for translating MLIR dialect to LLVM IR------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines general utilities for MLIR Dialect translations to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_OPENMPCOMMON_H
#define MLIR_TARGET_LLVMIR_DIALECT_OPENMPCOMMON_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/IRBuilder.h"

namespace mlir {
namespace LLVM {

/// Create a constant string location from the MLIR Location information.
/// When \p ForOffloadMap is true, the layout matches Clang's
/// \c emitMappingInformation (first \c ';'-delimited field is the map name for
/// \c libomptarget). When false, the layout matches \c ident_t (file then
/// function name in the encoded string).
llvm::Constant *createSourceLocStrFromLocation(Location loc,
                                               llvm::OpenMPIRBuilder &builder,
                                               StringRef name, uint32_t &strLen,
                                               bool ForOffloadMap = false);

/// Create a constant string representing the mapping information extracted from
/// the MLIR location information.
llvm::Constant *createMappingInformation(Location loc,
                                         llvm::OpenMPIRBuilder &builder);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_OPENMPCOMMON_H
