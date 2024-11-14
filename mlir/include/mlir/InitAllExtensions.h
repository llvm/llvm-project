//===- InitAllExtensions.h - MLIR Extension Registration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialect
// extensions to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITALLEXTENSIONS_H_
#define MLIR_INITALLEXTENSIONS_H_

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.h"
#include "mlir/Dialect/Tensor/Extensions/AllExtensions.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/DebugExtension/DebugExtension.h"
#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtension.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtension.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"

#include <cstdlib>

namespace mlir {

/// This function may be called to register all MLIR dialect extensions with the
/// provided registry.
/// If you're building a compiler, you generally shouldn't use this: you would
/// individually register the specific extensions that are useful for the
/// pipelines and transformations you are using.
inline void registerAllExtensions(DialectRegistry &registry) {
  // Register all conversions to LLVM extensions.
  arith::registerConvertArithToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  func::registerAllExtensions(registry);
  tensor::registerAllExtensions(registry);
  registerConvertFuncToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  registerConvertNVVMToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);

  // Register all transform dialect extensions.
  affine::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  dlti::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  gpu::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  nvgpu::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  sparse_tensor::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  transform::registerDebugExtension(registry);
  transform::registerIRDLExtension(registry);
  transform::registerLoopExtension(registry);
  transform::registerPDLExtension(registry);
  vector::registerTransformDialectExtension(registry);

  // Translation extensions need to be registered by calling
  // `registerAllToLLVMIRTranslations` (see All.h).
}

} // namespace mlir

#endif // MLIR_INITALLEXTENSIONS_H_
