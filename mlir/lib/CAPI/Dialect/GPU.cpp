//===- GPU.cpp - C Interface for GPU dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/GPU.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(GPU, gpu, gpu::GPUDialect)

//===-------------------------------------------------------------------===//
// AsyncTokenType
//===-------------------------------------------------------------------===//

bool mlirTypeIsAGPUAsyncTokenType(MlirType type) {
  return isa<gpu::AsyncTokenType>(unwrap(type));
}

MlirType mlirGPUAsyncTokenTypeGet(MlirContext ctx) {
  return wrap(gpu::AsyncTokenType::get(unwrap(ctx)));
}

//===---------------------------------------------------------------------===//
// ObjectAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAGPUObjectAttr(MlirAttribute attr) {
  return llvm::isa<gpu::ObjectAttr>(unwrap(attr));
}

MlirAttribute mlirGPUObjectAttrGet(MlirContext mlirCtx, MlirAttribute target,
                                   uint32_t format, MlirStringRef objectStrRef,
                                   MlirAttribute mlirObjectProps) {
  MLIRContext *ctx = unwrap(mlirCtx);
  llvm::StringRef object = unwrap(objectStrRef);
  DictionaryAttr objectProps;
  if (mlirObjectProps.ptr != nullptr)
    objectProps = llvm::cast<DictionaryAttr>(unwrap(mlirObjectProps));
  return wrap(gpu::ObjectAttr::get(
      ctx, unwrap(target), static_cast<gpu::CompilationTarget>(format),
      StringAttr::get(ctx, object), objectProps, nullptr));
}

MlirAttribute mlirGPUObjectAttrGetWithKernels(MlirContext mlirCtx,
                                              MlirAttribute target,
                                              uint32_t format,
                                              MlirStringRef objectStrRef,
                                              MlirAttribute mlirObjectProps,
                                              MlirAttribute mlirKernelsAttr) {
  MLIRContext *ctx = unwrap(mlirCtx);
  llvm::StringRef object = unwrap(objectStrRef);
  DictionaryAttr objectProps;
  if (mlirObjectProps.ptr != nullptr)
    objectProps = llvm::cast<DictionaryAttr>(unwrap(mlirObjectProps));
  gpu::KernelTableAttr kernels;
  if (mlirKernelsAttr.ptr != nullptr)
    kernels = llvm::cast<gpu::KernelTableAttr>(unwrap(mlirKernelsAttr));
  return wrap(gpu::ObjectAttr::get(
      ctx, unwrap(target), static_cast<gpu::CompilationTarget>(format),
      StringAttr::get(ctx, object), objectProps, kernels));
}

MlirAttribute mlirGPUObjectAttrGetTarget(MlirAttribute mlirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return wrap(objectAttr.getTarget());
}

uint32_t mlirGPUObjectAttrGetFormat(MlirAttribute mlirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return static_cast<uint32_t>(objectAttr.getFormat());
}

MlirStringRef mlirGPUObjectAttrGetObject(MlirAttribute mlirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  llvm::StringRef object = objectAttr.getObject();
  return mlirStringRefCreate(object.data(), object.size());
}

bool mlirGPUObjectAttrHasProperties(MlirAttribute mlirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return objectAttr.getProperties() != nullptr;
}

MlirAttribute mlirGPUObjectAttrGetProperties(MlirAttribute mlirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return wrap(objectAttr.getProperties());
}

bool mlirGPUObjectAttrHasKernels(MlirAttribute mlirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return objectAttr.getKernels() != nullptr;
}

MlirAttribute mlirGPUObjectAttrGetKernels(MlirAttribute mlirObjectAttr) {
  gpu::ObjectAttr objectAttr =
      llvm::cast<gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return wrap(objectAttr.getKernels());
}
