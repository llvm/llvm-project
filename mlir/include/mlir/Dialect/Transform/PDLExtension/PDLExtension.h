//===- PDLExtension.h - PDL extension for Transform dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSION_H
#define MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSION_H

namespace mlir {
class DialectRegistry;

namespace transform {
/// Registers the PDL extension of the Transform dialect in the given registry.
void registerPDLExtension(DialectRegistry &dialectRegistry);
} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_PDLEXTENSION_PDLEXTENSION_H
