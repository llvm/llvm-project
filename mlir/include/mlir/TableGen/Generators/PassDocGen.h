//===- PassDocGen.h - MLIR pass documentation generation utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for generating documentation for passes from
// TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_PASSDOCGEN_H
#define MLIR_TABLEGEN_GENERATORS_PASSDOCGEN_H

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Emit documentation for all passes derived from PassBase in records.
void emitPassDocs(const llvm::RecordKeeper &records, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_PASSDOCGEN_H
