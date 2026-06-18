//===------- Types.h - Kernel Language (CUDA/HIP) api types ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Types.h"
#include <cstddef>
#include <cstdint>

struct uint3 {
  unsigned x = 0, y = 0, z = 0;
};

using dim3 = uint3;

struct CallConfigurationTy {
  dim3 GridSize;
  dim3 BlockSize;
  size_t SharedMemory;
  void *Stream;
};
