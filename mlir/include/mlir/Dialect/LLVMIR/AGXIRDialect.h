//===- AGXIRDialect.h - MLIR AGXIR IR dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AGXIR dialect wraps the intrinsics of the experimental AGX (Apple GPU)
// LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_AGXIRDIALECT_H_
#define MLIR_DIALECT_LLVMIR_AGXIRDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/LLVMIR/AGXIROpsDialect.h.inc"

#include "mlir/Dialect/LLVMIR/AGXIROpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/AGXIROpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/AGXIROps.h.inc"

#endif // MLIR_DIALECT_LLVMIR_AGXIRDIALECT_H_
