//===- Chipset.h - AMDGPU Chipset version struct ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_AMDGPUTOROCDL_CHIPSET_H_
#define MLIR_CONVERSION_AMDGPUTOROCDL_CHIPSET_H_

#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace amdgpu {
struct Chipset {
  Chipset() = default;
  Chipset(unsigned majorVersion, unsigned minorVersion)
      : majorVersion(majorVersion), minorVersion(minorVersion){};
  static FailureOr<Chipset> parse(StringRef name);

  unsigned majorVersion = 0;
  unsigned minorVersion = 0;
};
} // end namespace amdgpu
} // end namespace mlir

#endif
