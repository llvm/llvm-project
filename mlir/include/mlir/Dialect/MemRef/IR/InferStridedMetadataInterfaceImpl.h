//===- InferStridedMetadataOpInterfaceImpl.h - Impl. of infer strided md --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_IR_INFERSTRIDEDMETADATAOPINTERFACEIMPL_H
#define MLIR_DIALECT_MEMREF_IR_INFERSTRIDEDMETADATAOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace memref {
/// Register the external models for the infer strided metadata op interface,
/// for the `memref` dialect. This implementation assumes that the strided
/// metadata of a ranked memref consists of one offset, and zero or more sizes
/// and strides.
void registerInferStridedMetadataOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_IR_INFERSTRIDEDMETADATAOPINTERFACEIMPL_H
