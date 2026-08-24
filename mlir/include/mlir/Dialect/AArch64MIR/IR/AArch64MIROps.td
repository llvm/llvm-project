//===- AArch64MIROps.td - AArch64 MIR operations ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIROPS_TD
#define MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIROPS_TD

include "mlir/Dialect/AArch64MIR/IR/AArch64MIRBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// The full set of AArch64 selected ops is generated from AArch64's instruction
// TableGen by the `-gen-mir-target-ops` backend and checked in as
// AArch64MIRGenOps.td.
include "mlir/Dialect/AArch64MIR/IR/AArch64MIRGenOps.td"

#endif // MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIROPS_TD
