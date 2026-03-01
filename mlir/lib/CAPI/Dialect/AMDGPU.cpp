//===- AMDGPU.cpp - C Interface for AMDGPU dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/AMDGPU.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AMDGPU, amdgpu,
                                      mlir::amdgpu::AMDGPUDialect)

using namespace mlir;
using namespace mlir::amdgpu;

//===---------------------------------------------------------------------===//
// TDMBaseType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAAMDGPUTDMBaseType(MlirType type) {
  return isa<amdgpu::TDMBaseType>(unwrap(type));
}

MlirTypeID mlirAMDGPUTDMBaseTypeGetTypeID() {
  return wrap(amdgpu::TDMBaseType::getTypeID());
}

MlirType mlirAMDGPUTDMBaseTypeGet(MlirContext ctx, MlirType elementType) {
  return wrap(amdgpu::TDMBaseType::get(unwrap(ctx), unwrap(elementType)));
}

MlirStringRef mlirAMDGPUTDMBaseTypeGetName(void) {
  return wrap(amdgpu::TDMBaseType::name);
}

//===---------------------------------------------------------------------===//
// TDMDescriptorType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAAMDGPUTDMDescriptorType(MlirType type) {
  return isa<amdgpu::TDMDescriptorType>(unwrap(type));
}

MlirTypeID mlirAMDGPUTDMDescriptorTypeGetTypeID() {
  return wrap(amdgpu::TDMDescriptorType::getTypeID());
}

MlirType mlirAMDGPUTDMDescriptorTypeGet(MlirContext ctx) {
  return wrap(amdgpu::TDMDescriptorType::get(unwrap(ctx)));
}

MlirStringRef mlirAMDGPUTDMDescriptorTypeGetName(void) {
  return wrap(amdgpu::TDMDescriptorType::name);
}

//===---------------------------------------------------------------------===//
// TDMGatherBaseType
//===---------------------------------------------------------------------===//

bool mlirTypeIsAAMDGPUTDMGatherBaseType(MlirType type) {
  return isa<amdgpu::TDMGatherBaseType>(unwrap(type));
}

MlirTypeID mlirAMDGPUTDMGatherBaseTypeGetTypeID() {
  return wrap(amdgpu::TDMGatherBaseType::getTypeID());
}

MlirType mlirAMDGPUTDMGatherBaseTypeGet(MlirContext ctx, MlirType elementType,
                                        MlirType indexType) {
  return wrap(amdgpu::TDMGatherBaseType::get(unwrap(ctx), unwrap(elementType),
                                             unwrap(indexType)));
}

MlirStringRef mlirAMDGPUTDMGatherBaseTypeGetName(void) {
  return wrap(amdgpu::TDMGatherBaseType::name);
}
