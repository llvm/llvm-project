//===-- ArmSME.td - ArmSME dialect definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the ArmSME dialect as well as some
// shared definitions.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSME
#define ARMSME

include "mlir/IR/DialectBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// ArmSME Dialect
//===----------------------------------------------------------------------===//

def ArmSME_Dialect : Dialect {
  let name = "arm_sme";
  let cppNamespace = "::mlir::arm_sme";
  let summary = "Basic dialect to target Arm SME architectures";
  let description = [{
    This dialect contains the definitions necessary to target Arm SME
    scalable matrix operations.

    Sources:
    https://developer.arm.com/documentation/ddi0616
    https://developer.arm.com/documentation/ddi0602/2023-03/SME-Instructions
  }];
  let dependentDialects = ["scf::SCFDialect", "vector::VectorDialect",
                           "memref::MemRefDialect"];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// ArmSME type definitions
//===----------------------------------------------------------------------===//

def SVEVector : ScalableVectorOfRankAndLengthAndType<
  [1], [16, 8, 4, 2, 1], [I8, I16, I32, I64, I128, F16, BF16, F32, F64]>;

def SVEPredicate : ScalableVectorOfRankAndLengthAndType<
  [1], [16, 8, 4, 2, 1], [I1]>;


#endif // ARMSME
