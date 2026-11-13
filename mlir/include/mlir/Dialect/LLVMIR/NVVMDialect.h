//===- NVVMDialect.h - MLIR NVVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVVM IR dialect in MLIR with all its ops, attributes
// and enums. The NVVM dialect only contains GPU specific additions on top of
// the general LLVM dialect.
//
// Prefer the narrower headers: this umbrella header only exists for
// compatibility. Including all ops of the NVVM dialect is expensive to compile,
// so include this header only if most of the headers below are needed anyway.
// Otherwise include:
//  - NVVMDialectDecl.h for the dialect, NVVMAttributes.h for its attributes
//    and enums, and NVVMRequiresSMTraits.h for its traits,
//  - NVVM<Partition>Ops.h for the ops of an op partition, e.g.
//    NVVMIndexSRegOps.h for `nvvm.read.ptx.sreg.tid.x` and NVVMMmaOps.h for
//    `nvvm.mma.sync`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_NVVMDIALECT_H_

#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Dialect/LLVMIR/NVVMIntrinsicLowering.h"
#include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

///// Ops /////
#include "mlir/Dialect/LLVMIR/NVVMArithOps.h"
#include "mlir/Dialect/LLVMIR/NVVMConvertOps.h"
#include "mlir/Dialect/LLVMIR/NVVMIndexSRegOps.h"
#include "mlir/Dialect/LLVMIR/NVVMMemoryOps.h"
#include "mlir/Dialect/LLVMIR/NVVMMiscOps.h"
#include "mlir/Dialect/LLVMIR/NVVMMiscSRegOps.h"
#include "mlir/Dialect/LLVMIR/NVVMMmaOps.h"
#include "mlir/Dialect/LLVMIR/NVVMSyncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMTcgenOps.h"
#include "mlir/Dialect/LLVMIR/NVVMWarpOps.h"

#endif /* MLIR_DIALECT_LLVMIR_NVVMDIALECT_H_ */
