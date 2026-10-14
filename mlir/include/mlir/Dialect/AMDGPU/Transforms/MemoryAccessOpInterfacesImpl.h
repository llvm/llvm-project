//===- MemoryAccessOpInterfacesImpl.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_TRANSFORMS_MEMORYACCESSOPINTERFACESIMPL_H
#define MLIR_DIALECT_AMDGPU_TRANSFORMS_MEMORYACCESSOPINTERFACESIMPL_H

namespace mlir {

class DialectRegistry;

namespace amdgpu {
void registerMemoryAccessOpInterfacesExternalModels(DialectRegistry &registry);
} // namespace amdgpu
} // namespace mlir

#endif // MLIR_DIALECT_AMDGPU_TRANSFORMS_MEMORYACCESSOPINTERFACESIMPL_H
