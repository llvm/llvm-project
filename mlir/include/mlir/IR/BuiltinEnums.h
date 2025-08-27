//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINENUMS_H
#define MLIR_IR_BUILTINENUMS_H

//===----------------------------------------------------------------------===//
// Tablegen Enums Declarations
//===----------------------------------------------------------------------===//

namespace mlir {

#define GET_ENUM_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinAttributesEnums.h.inc"

} // namespace mlir


#endif // MLIR_IR_BUILTIENUMS_H
