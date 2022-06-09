//===--- Float16bits.h - supports 2-byte floats ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements f16 and bf16 to support the compilation and execution
// of programs using these types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_FLOAT16BITS_H_
#define MLIR_EXECUTIONENGINE_FLOAT16BITS_H_

#include <cstdint>
#include <iostream>

// Implements half precision and bfloat with f16 and bf16, using the MLIR type
// names. These data types are also used for c-interface runtime routines.
extern "C" {
struct f16 {
  f16(float f = 0);
  uint16_t bits;
};

struct bf16 {
  bf16(float f = 0);
  uint16_t bits;
};
}

// Outputs a half precision value.
std::ostream &operator<<(std::ostream &os, const f16 &f);
// Outputs a bfloat value.
std::ostream &operator<<(std::ostream &os, const bf16 &d);

#endif // MLIR_EXECUTIONENGINE_FLOAT16BITS_H_
