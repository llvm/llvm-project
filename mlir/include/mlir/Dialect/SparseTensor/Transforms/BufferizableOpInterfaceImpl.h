//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace mlir {
class DialectRegistry;

namespace sparse_tensor {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
