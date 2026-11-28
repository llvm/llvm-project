//===-- LLVMAllOps.td - Combined LLVM dialect operations ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_ALL_OPS
#define LLVMIR_ALL_OPS

// Generate the core and intrinsic operations by op partition from one file, so
// that they share a single dialect registration hook.
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "mlir/Dialect/LLVMIR/LLVMIntrinsicOps.td"

#endif // LLVMIR_ALL_OPS
