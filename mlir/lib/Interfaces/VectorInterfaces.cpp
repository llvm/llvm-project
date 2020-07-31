//===- VectorInterfaces.cpp - Unrollable vector operations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/VectorInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// VectorUnroll Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the VectorUntoll interfaces.
#include "mlir/Interfaces/VectorInterfaces.cpp.inc"
