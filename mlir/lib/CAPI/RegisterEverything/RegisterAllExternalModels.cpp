//===- RegisterAllExtensions.cpp - Register all MLIR extensions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/RegisterAllExternalModels.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVM/XeVM/Target.h"
#include "mlir/Target/SPIRV/Target.h"

using namespace mlir;

void mlirRegisterAllExternalModels(MlirDialectRegistry mlirRegistry) {
  mlir::DialectRegistry *registry = unwrap(mlirRegistry);
  // Register all external models.
  affine::registerValueBoundsOpInterfaceExternalModels(*registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(*registry);
  arith::registerBufferizableOpInterfaceExternalModels(*registry);
  arith::registerBufferViewFlowOpInterfaceExternalModels(*registry);
  arith::registerShardingInterfaceExternalModels(*registry);
  arith::registerValueBoundsOpInterfaceExternalModels(*registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      *registry);
  builtin::registerCastOpInterfaceExternalModels(*registry);
  cf::registerBufferizableOpInterfaceExternalModels(*registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(*registry);
  gpu::registerBufferDeallocationOpInterfaceExternalModels(*registry);
  gpu::registerValueBoundsOpInterfaceExternalModels(*registry);
  LLVM::registerInlinerInterface(*registry);
  NVVM::registerInlinerInterface(*registry);
  linalg::registerAllDialectInterfaceImplementations(*registry);
  linalg::registerRuntimeVerifiableOpInterfaceExternalModels(*registry);
  memref::registerAllocationOpInterfaceExternalModels(*registry);
  memref::registerBufferViewFlowOpInterfaceExternalModels(*registry);
  memref::registerRuntimeVerifiableOpInterfaceExternalModels(*registry);
  memref::registerValueBoundsOpInterfaceExternalModels(*registry);
  memref::registerMemorySlotExternalModels(*registry);
  ml_program::registerBufferizableOpInterfaceExternalModels(*registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(*registry);
  scf::registerBufferizableOpInterfaceExternalModels(*registry);
  scf::registerValueBoundsOpInterfaceExternalModels(*registry);
  shape::registerBufferizableOpInterfaceExternalModels(*registry);
  sparse_tensor::registerBufferizableOpInterfaceExternalModels(*registry);
  tensor::registerBufferizableOpInterfaceExternalModels(*registry);
  tensor::registerFindPayloadReplacementOpInterfaceExternalModels(*registry);
  tensor::registerInferTypeOpInterfaceExternalModels(*registry);
  tensor::registerRuntimeVerifiableOpInterfaceExternalModels(*registry);
  tensor::registerSubsetOpInterfaceExternalModels(*registry);
  tensor::registerTilingInterfaceExternalModels(*registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(*registry);
  tosa::registerShardingInterfaceExternalModels(*registry);
  vector::registerBufferizableOpInterfaceExternalModels(*registry);
  vector::registerSubsetOpInterfaceExternalModels(*registry);
  vector::registerValueBoundsOpInterfaceExternalModels(*registry);
  NVVM::registerNVVMTargetInterfaceExternalModels(*registry);
  ROCDL::registerROCDLTargetInterfaceExternalModels(*registry);
  spirv::registerSPIRVTargetInterfaceExternalModels(*registry);
  xevm::registerXeVMTargetInterfaceExternalModels(*registry);
}
