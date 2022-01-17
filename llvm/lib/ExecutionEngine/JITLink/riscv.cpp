//===------ riscv.cpp - Generic JITLink riscv edge kinds, utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing riscv objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/riscv.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace riscv {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case R_RISCV_32:
    return "R_RISCV_32";
  case R_RISCV_64:
    return "R_RISCV_64";
  case R_RISCV_BRANCH:
    return "R_RISCV_BRANCH";
  case R_RISCV_HI20:
    return "R_RISCV_HI20";
  case R_RISCV_LO12_I:
    return "R_RISCV_LO12_I";
  case R_RISCV_PCREL_HI20:
    return "R_RISCV_PCREL_HI20";
  case R_RISCV_PCREL_LO12_I:
    return "R_RISCV_PCREL_LO12_I";
  case R_RISCV_PCREL_LO12_S:
    return "R_RISCV_PCREL_LO12_S";
  case R_RISCV_CALL:
    return "R_RISCV_CALL";
  }
  return getGenericEdgeKindName(K);
}
} // namespace riscv
} // namespace jitlink
} // namespace llvm
