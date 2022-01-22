//===- BufferizationInterfaceImpl.h - Bufferization Impl. of Op Interface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONINTERFACEIMPL_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONINTERFACEIMPL_H_

namespace mlir {

class DialectRegistry;

namespace bufferization {
namespace bufferization_ext {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace bufferization_ext
} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONINTERFACEIMPL_H_
