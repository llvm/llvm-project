//===- Chipset.cpp - AMDGPU Chipset version struct parsing ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::amdgpu {

FailureOr<Chipset> Chipset::parse(StringRef name) {
  if (!name.consume_front("gfx"))
    return failure();
  if (name.size() < 3)
    return failure();

  unsigned major = 0;
  unsigned minor = 0;
  unsigned stepping = 0;

  StringRef majorRef = name.drop_back(2);
  StringRef minorRef = name.take_back(2).drop_back(1);
  StringRef steppingRef = name.take_back(1);
  if (majorRef.getAsInteger(10, major))
    return failure();
  if (minorRef.getAsInteger(16, minor))
    return failure();
  if (steppingRef.getAsInteger(16, stepping))
    return failure();
  return Chipset(major, minor, stepping);
}

} // namespace mlir::amdgpu
