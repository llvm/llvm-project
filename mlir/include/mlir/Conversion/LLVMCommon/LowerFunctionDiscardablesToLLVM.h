//===- LowerFunctionDiscardablesToLLVM.h - Func discardables to llvm - C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for lowering discardable attributes on any FunctionOpInterface
// (e.g. func.func, gpu.func) into llvm.func properties and discardables.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMCOMMON_LOWERFUNCTIONDISCARDABLESTOLLVM_H
#define MLIR_CONVERSION_LLVMCOMMON_LOWERFUNCTIONDISCARDABLESTOLLVM_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {

/// Result of lowering discardable attributes from a `FunctionOpInterface` to
/// what `llvm.func` expects: typed inherent properties plus remaining
/// discardable attributes.
struct LoweredLLVMFuncAttrs {
  LLVM::LLVMFuncOp::Properties properties;
  NamedAttrList discardableAttrs;
};

/// Partition `funcOp`'s discardables for `llvm.func`: `sym_name`,
/// `function_type`, and typed `properties` from `llvm.*` ODS attrs; other
/// discardables unchanged. Fails if that property set is invalid; drops
/// ODS-named attrs without `llvm.`.
FailureOr<LoweredLLVMFuncAttrs>
lowerDiscardableAttrsForLLVMFunc(FunctionOpInterface funcOp, Type llvmFuncType);

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMCOMMON_LOWERFUNCTIONDISCARDABLESTOLLVM_H
