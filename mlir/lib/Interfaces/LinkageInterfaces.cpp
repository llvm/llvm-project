//===- LinkageInterfaces.cpp - Interfaces for Linkage -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/LinkageInterfaces.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::link;

/// Include the definitions of the interface.
#include "mlir/Interfaces/LinkageInterfaces.cpp.inc"
