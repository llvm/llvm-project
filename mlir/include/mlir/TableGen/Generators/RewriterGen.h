//===- RewriterGen.h - MLIR pattern rewriter generator --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for generating C++ pattern rewriters from
// TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_REWRITERGEN_H
#define MLIR_TABLEGEN_GENERATORS_REWRITERGEN_H

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Emit pattern rewriters for all Pattern definitions in records.
void emitRewriters(const llvm::RecordKeeper &records, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_REWRITERGEN_H
