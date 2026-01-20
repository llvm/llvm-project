//===- MemoryAccessOpInterfacesImpl.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NVGPU_TRANSFORMS_MEMORYACCESSOPINTERFACESIMPL_H
#define MLIR_DIALECT_NVGPU_TRANSFORMS_MEMORYACCESSOPINTERFACESIMPL_H

namespace mlir {

class DialectRegistry;

namespace nvgpu {
void registerMemoryAccessOpInterfacesExternalModels(DialectRegistry &registry);
} // namespace nvgpu
} // namespace mlir

#endif // MLIR_DIALECT_NVGPU_TRANSFORMS_MEMORYACCESSOPINTERFACESIMPL_H
