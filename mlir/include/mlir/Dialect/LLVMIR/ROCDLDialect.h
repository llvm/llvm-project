//===- ROCDLDialect.h - MLIR ROCDL IR dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ROCDL dialect in MLIR, containing ROCDL operations
// and ROCDL specific extensions to the LLVM type system.
//
// Unfortunately there does not exists a formal definition of ROCDL IR that be
// pointed to here. However the following links contain more information about
// ROCDL (ROCm-Device-Library)
//
// https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/amd-stg-open/doc/OCML.md
// https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/amd-stg-open/doc/OCKL.md
// https://llvm.org/docs/AMDGPUUsage.html
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_ROCDLDIALECT_H_
#define MLIR_DIALECT_LLVMIR_ROCDLDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/LLVMIR/ROCDLOpsEnums.h.inc"

///// Ops /////
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOps.h.inc"

#include "mlir/Dialect/LLVMIR/ROCDLOpsDialect.h.inc"

#endif /* MLIR_DIALECT_LLVMIR_ROCDLDIALECT_H_ */
