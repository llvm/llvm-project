// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_TABLEGEN_REWRITER_H_
#define MLIR_TABLEGEN_REWRITER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/TableGen/Record.h"

namespace mlir {
namespace tblgen {
void emitRewriters(const llvm::RecordKeeper &records, raw_ostream &os);
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_REWRITER_H_
