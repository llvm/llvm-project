//===- ArmSMEDialect.h - MLIR Dialect for Arm SME ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for ArmSME in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSME_IR_ARMSME_H
#define MLIR_DIALECT_ARMSME_IR_ARMSME_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/ArmSME/IR/ArmSMEEnums.h"
#include "mlir/Dialect/ArmSME/IR/ArmSMEOpInterfaces.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSMEAttrDefs.h.inc"

#include "mlir/Dialect/ArmSME/IR/ArmSMEDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSMEOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSME/IR/ArmSMEIntrinsicOps.h.inc"

#endif // MLIR_DIALECT_ARMSME_IR_ARMSME_H
