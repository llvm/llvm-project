// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_TABLEGEN_OPENMP_H_
#define MLIR_TABLEGEN_OPENMP_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
namespace tblgen {

/// Verify that all properties of `OpenMP_Clause`s of records deriving from
/// `OpenMP_Op`s have been inherited by the latter.
bool verifyOpenmpDecls(const llvm::RecordKeeper &records, raw_ostream &);

/// Generate structures to represent clause-related operands, based on existing
/// `OpenMP_Clause` definitions and aggregate them into operation-specific
/// structures according to the `clauses` argument of each definition deriving
/// from `OpenMP_Op`.
bool genOpenmpClauseOps(const llvm::RecordKeeper &records, raw_ostream &os);

/// Emit op declarations for all op records.
bool emitOpDecls(const llvm::RecordKeeper &records, raw_ostream &os);

/// Emit op definitions for all op records.
bool emitOpDefs(const llvm::RecordKeeper &records, raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_OPENMP_H_
