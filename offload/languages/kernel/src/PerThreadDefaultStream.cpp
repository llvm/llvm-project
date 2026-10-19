//===-- PerThreadDefaultStream.cpp - Default stream mode override ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>

// Linked exactly once by the driver for -fgpu-default-stream=per-thread.
extern "C" uint32_t __LLVMOffloadingPerThreadDefaultStream = 1;
