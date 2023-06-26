//===- GENXDialect.h - MLIR GENX IR dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GENX dialect in MLIR, containing Intel GenX operations
// and GenX specific extensions to the LLVM type system.
//
// The following links contain more information about GenXIntrinsics functions
//
// https://github.com/intel/vc-intrinsics
// https://github.com/intel/vc-intrinsics/blob/master/GenXIntrinsics/docs/GenXLangRef.rst
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_GENXDIALECT_H_
#define MLIR_DIALECT_LLVMIR_GENXDIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOps.h.inc"

#include "mlir/Dialect/LLVMIR/GENXOpsDialect.h.inc"

#endif /* MLIR_DIALECT_LLVMIR_GENXDIALECT_H_ */
