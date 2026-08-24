//===-- Types.h - Kernel language API types -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_TYPES_H
#define LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_TYPES_H

#include <cstddef>
#include <cstdint>

struct uint3 {
  // CUDA/HIP uint3 is plain vector storage; default init does not zero it.
  unsigned int x, y, z;
};

struct dim3 {
  // CUDA/HIP dim3 model launch dimensions; omitted axes default to one.
  dim3(unsigned int X = 1, unsigned int Y = 1, unsigned int Z = 1)
      : x(X), y(Y), z(Z) {}
  dim3(uint3 V) : x(V.x), y(V.y), z(V.z) {}
  operator uint3() const { return {x, y, z}; }

  unsigned int x, y, z;
};

struct CallConfigurationTy {
  dim3 GridSize;
  dim3 BlockSize;
  size_t SharedMemory;
  void *Stream;
};

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_TYPES_H
