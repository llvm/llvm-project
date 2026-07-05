//===- AttrOrTypeFormatGen.h - MLIR attribute and type format generator ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_ATTRORTYPEFORMATGEN_H_
#define MLIR_TOOLS_MLIRTBLGEN_ATTRORTYPEFORMATGEN_H_

#include "mlir/TableGen/Class.h"

namespace mlir {
namespace tblgen {
class AttrOrTypeDef;

/// Generate a parser and printer based on a custom assembly format for an
/// attribute or type.
void generateAttrOrTypeFormat(const AttrOrTypeDef &def, MethodBody &parser,
                              MethodBody &printer);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_ATTRORTYPEFORMATGEN_H_
