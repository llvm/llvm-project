//===- PassCAPIGen.h - MLIR pass C API generation utilities ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for generating C API bindings for passes from
// TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_PASSCAPIGEN_H
#define MLIR_TABLEGEN_GENERATORS_PASSCAPIGEN_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Emit a C header declaring the C API for all passes in records.
/// prefix is used to namespace the generated function names.
void emitPassCAPIHeader(const llvm::RecordKeeper &records,
                        llvm::StringRef prefix, llvm::raw_ostream &os);

/// Emit the C implementation for the C API of all passes in records.
/// prefix is used to namespace the generated function names.
void emitPassCAPIImpl(const llvm::RecordKeeper &records, llvm::StringRef prefix,
                      llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_PASSCAPIGEN_H
