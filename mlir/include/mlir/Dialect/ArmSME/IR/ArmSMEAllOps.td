//===-- ArmSMEAllOps.td - Combined ArmSME dialect operations -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARMSME_ALL_OPS
#define ARMSME_ALL_OPS

// Generate the core and intrinsic operations by op partition from one file, so
// that they share a single dialect registration hook.
include "mlir/Dialect/ArmSME/IR/ArmSMEOps.td"
include "mlir/Dialect/ArmSME/IR/ArmSMEIntrinsicOps.td"

#endif // ARMSME_ALL_OPS
