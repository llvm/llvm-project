//===- TestDialect.h - MLIR Dialect for testing -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a fake 'test' dialect that can be used for testing things
// that do not have a respective counterpart in the main source directories.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTDIALECT_H
#define MLIR_TESTDIALECT_H

#include "TestParametricAttributes.h"
#include "TestParametricInterfaces.h"
#include "TestParametricTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ParametricSpecializationOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <memory>

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "TestParametricOpInterfaces.h.inc"
#include "TestParametricOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TestParametricOps.h.inc"

namespace testparametric {
void registerTestParametricDialect(::mlir::DialectRegistry &registry);
} // namespace testparametric

#endif // MLIR_TESTDIALECT_H
