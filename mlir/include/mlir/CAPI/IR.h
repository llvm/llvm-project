//===- IR.h - C API Utils for Core MLIR classes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// core MLIR classes. This file should not be included from C++ code other than
// C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_IR_H
#define MLIR_CAPI_IR_H

#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

DEFINE_C_API_PTR_METHODS(MlirContext, mlir::MLIRContext)
DEFINE_C_API_PTR_METHODS(MlirDialect, mlir::Dialect)
DEFINE_C_API_PTR_METHODS(MlirOperation, mlir::Operation)
DEFINE_C_API_PTR_METHODS(MlirBlock, mlir::Block)
DEFINE_C_API_PTR_METHODS(MlirOpPrintingFlags, mlir::OpPrintingFlags)
DEFINE_C_API_PTR_METHODS(MlirRegion, mlir::Region)

DEFINE_C_API_METHODS(MlirAttribute, mlir::Attribute)
DEFINE_C_API_METHODS(MlirIdentifier, mlir::Identifier)
DEFINE_C_API_METHODS(MlirLocation, mlir::Location)
DEFINE_C_API_METHODS(MlirModule, mlir::ModuleOp)
DEFINE_C_API_METHODS(MlirType, mlir::Type)
DEFINE_C_API_METHODS(MlirValue, mlir::Value)

#endif // MLIR_CAPI_IR_H
