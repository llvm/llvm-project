//===---- aarch64.cpp - Generic JITLink aarch64 edge kinds, utilities -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic utilities for graphs representing aarch64 objects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/aarch64.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {
namespace aarch64 {

const char *getEdgeKindName(Edge::Kind K) {
  switch (K) {
  case R_AARCH64_CALL26:
    return "R_AARCH64_CALL26";
  }
  return getGenericEdgeKindName(K);
}
} // namespace aarch64
} // namespace jitlink
} // namespace llvm
