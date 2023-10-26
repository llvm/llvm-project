//===- GENXDialect.h - MLIR GENX IR dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GENX dialect in MLIR, containing Intel GenX operations
// and GenX specific extensions to the LLVM type system.
//
// The following links contain more information about GenXIntrinsics functions
//
// https://github.com/intel/vc-intrinsics
// https://github.com/intel/vc-intrinsics/blob/master/GenXIntrinsics/docs/GenXLangRef.rst
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_GENXDIALECT_H_
#define MLIR_DIALECT_LLVMIR_GENXDIALECT_H_

#include "mlir/Dialect/LLVMIR/GENXTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


namespace mlir {
namespace GENX {

/// GENX memory space identifiers following SPIRV storage class convention
/// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/docs/SPIRVRepresentationInLLVM.rst#address-spaces
///
enum GENXMemorySpace {
  kFunction = 0,        // OpenCL workitem address space 
  kCrossWorkgroup = 1,  // OpenCL Global memory
  kUniformConstant = 2, // OpenCL Constant memory
  kWorkgroup = 3,       // OpenCL Local memory
  kGeneric = 4          // OpenCL Generic memory
};

} // namespace GENX
} // namespace mlir

///// Ops /////
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOps.h.inc"

#include "mlir/Dialect/LLVMIR/GENXOpsDialect.h.inc"

#endif /* MLIR_DIALECT_LLVMIR_GENXDIALECT_H_ */
