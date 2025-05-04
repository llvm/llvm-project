//===- AggregatedOpInterface.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an interface for decomposing operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_AGGREGATEDOPINTERFACE_H_
#define MLIR_INTERFACES_AGGREGATEDOPINTERFACE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

/// Include the generated interface declarations.
#include "mlir/Interfaces/AggregatedOpInterface.h.inc"

#endif // MLIR_INTERFACES_AGGREGATEDOPINTERFACE_H_
