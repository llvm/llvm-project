//===- Deserialization.cpp - MLIR SPIR-V Deserialization ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SPIRV/Deserialization.h"

#include "Deserializer.h"

namespace mlir {
spirv::OwningSPIRVModuleRef spirv::deserialize(ArrayRef<uint32_t> binary,
                                               MLIRContext *context) {
  Deserializer deserializer(binary, context);

  if (failed(deserializer.deserialize()))
    return nullptr;

  return deserializer.collect();
}
} // namespace mlir
