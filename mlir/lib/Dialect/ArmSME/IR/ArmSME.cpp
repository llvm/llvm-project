//===- ArmSMEDialect.cpp - MLIR ArmSME dialect implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmSME dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::arm_sme;

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSMEDialect.cpp.inc"

#include "mlir/Dialect/ArmSME/IR/ArmSMEEnums.cpp.inc"

#include "mlir/Dialect/ArmSME/IR/ArmSMEOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSMEOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSMEIntrinsicOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSMETypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSMEAttrDefs.cpp.inc"

void ArmSMEDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/ArmSME/IR/ArmSMEAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/ArmSME/IR/ArmSMEOps.cpp.inc"
      ,
#define GET_OP_LIST
#include "mlir/Dialect/ArmSME/IR/ArmSMEIntrinsicOps.cpp.inc"
      >();
}
