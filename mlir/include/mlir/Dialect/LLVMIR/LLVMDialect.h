//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the LLVM IR dialect in MLIR with all its ops, attributes
// and types.
//
// Prefer the narrower headers: this umbrella header only exists for
// compatibility. Including all ops of the LLVM dialect is expensive to compile,
// so include this header only if most of the headers below are needed anyway.
// Otherwise include:
//  - LLVMDialectDecl.h for the dialect, its attributes and its types,
//  - LLVMInterfaces.h for its interfaces,
//  - LLVM<Partition>Ops.h for the ops of an op partition, e.g. LLVMArithOps.h
//    for `llvm.add` and LLVMMemoryOps.h for `llvm.load`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "mlir/Dialect/LLVMIR/LLVMArithOps.h"
#include "mlir/Dialect/LLVMIR/LLVMCastOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstrainedIntrinsicOps.h"
#include "mlir/Dialect/LLVMIR/LLVMControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMCoroIntrinsicOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDebugIntrinsicOps.h"
#include "mlir/Dialect/LLVMIR/LLVMFloatIntrinsicOps.h"
#include "mlir/Dialect/LLVMIR/LLVMIntegerIntrinsicOps.h"
#include "mlir/Dialect/LLVMIR/LLVMMemoryIntrinsicOps.h"
#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.h"
#include "mlir/Dialect/LLVMIR/LLVMMiscIntrinsicOps.h"
#include "mlir/Dialect/LLVMIR/LLVMModuleOps.h"
#include "mlir/Dialect/LLVMIR/LLVMVectorIntrinsicOps.h"

#endif // MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
