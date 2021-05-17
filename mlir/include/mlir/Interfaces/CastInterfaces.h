//===- CastInterfaces.h - Cast Interfaces for MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the cast interfaces defined in
// `CastInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CASTINTERFACES_H
#define MLIR_INTERFACES_CASTINTERFACES_H

#include "mlir/IR/OpDefinition.h"

/// Include the generated interface declarations.
#include "mlir/Interfaces/CastInterfaces.h.inc"

#endif // MLIR_INTERFACES_CASTINTERFACES_H
