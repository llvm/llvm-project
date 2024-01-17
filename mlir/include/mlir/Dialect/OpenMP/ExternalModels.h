//===- ExternalModels.h - External models owned by the OMP dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the OpenMP external models for other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_EXTERNALMODELS_H
#define MLIR_DIALECT_OPENMP_EXTERNALMODELS_H

namespace mlir {
class DialectRegistry;
namespace omp {
/// Register OMP external models for the Builtin dialect.
void registerBuiltinExternalModels(DialectRegistry &registry);
/// Register OMP external models for the Func dialect.
void registerFuncExternalModels(DialectRegistry &registry);
/// Register OMP external models for the GPU dialect.
void registerGPUExternalModels(DialectRegistry &registry);
/// Register OMP external models for the LLVM dialect.
void registerLLVMExternalModels(DialectRegistry &registry);
/// Register all OMP external models.
inline void registerAllExternalModels(DialectRegistry &registry) {
  registerBuiltinExternalModels(registry);
  registerFuncExternalModels(registry);
  registerGPUExternalModels(registry);
  registerLLVMExternalModels(registry);
}
} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_EXTERNALMODELS_H
