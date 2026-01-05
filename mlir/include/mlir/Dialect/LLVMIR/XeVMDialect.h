//===-- XeVMDialect.h - MLIR XeVM target definitions ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "mlir/Dialect/LLVMIR/XeVMOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/XeVMOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/XeVMOps.h.inc"

#include "mlir/Dialect/LLVMIR/XeVMOpsDialect.h.inc"

#endif /* MLIR_DIALECT_LLVMIR_XEVMDIALECT_H_ */
