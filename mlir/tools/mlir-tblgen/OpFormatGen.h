//===- OpFormatGen.h - MLIR operation format generator ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for generating parsers and printers from the
// declarative format.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_OPFORMATGEN_H_
#define MLIR_TOOLS_MLIRTBLGEN_OPFORMATGEN_H_

namespace mlir {
namespace tblgen {
class OpClass;
class Operator;

// Generate the assembly format for the given operator.
void generateOpFormat(const Operator &constOp, OpClass &opClass,
                      bool hasProperties);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_OPFORMATGEN_H_
