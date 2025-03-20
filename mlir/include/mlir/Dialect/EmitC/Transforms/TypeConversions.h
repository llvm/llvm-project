//===- TypeConversions.h - Convert signless types into C/C++ types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_TRANSFORMS_TYPECONVERSIONS_H
#define MLIR_DIALECT_EMITC_TRANSFORMS_TYPECONVERSIONS_H

#include <optional>

namespace mlir {
class TypeConverter;
class Type;
void populateEmitCSizeTTypeConversions(TypeConverter &converter);

namespace emitc {
std::optional<Type> getUnsignedTypeFor(Type ty);
std::optional<Type> getSignedTypeFor(Type ty);
} // namespace emitc

} // namespace mlir

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_TYPECONVERSIONS_H
