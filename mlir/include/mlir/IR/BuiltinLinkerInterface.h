//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines registration of builtin dialect linker interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINLINKERINTERFACE_H
#define MLIR_IR_BUILTINLINKERINTERFACE_H

namespace mlir {
class DialectRegistry;

namespace builtin {
void registerLinkerInterface(DialectRegistry &registry);
} // namespace builtin

} // namespace mlir
#endif // MLIR_IR_BUILTINLINKERINTERFACE_H
