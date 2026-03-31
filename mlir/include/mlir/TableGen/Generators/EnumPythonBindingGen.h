//===- EnumPythonBindingGen.h - Python enum bindings generator --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_ENUMPYTHONBINDINGGEN_H
#define MLIR_TABLEGEN_GENERATORS_ENUMPYTHONBINDINGGEN_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Emit Python bindings for enum attributes. If dialectName is non-empty,
/// only emit builders for EnumAttr records belonging to that dialect.
bool emitPythonEnums(const llvm::RecordKeeper &records,
                     llvm::StringRef dialectName, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_ENUMPYTHONBINDINGGEN_H
