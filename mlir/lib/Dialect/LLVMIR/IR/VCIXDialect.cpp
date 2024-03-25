//===- VCIXDialect.cpp - MLIR VCIX ops implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VCIX dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/VCIXDialect.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace vcix;

#include "mlir/Dialect/LLVMIR/VCIXOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// VCIXDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

void VCIXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/VCIXOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/VCIXOpsAttributes.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/VCIXOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/VCIXOpsAttributes.cpp.inc"
