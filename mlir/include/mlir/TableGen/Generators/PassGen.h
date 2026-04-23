//===- PassGen.h - MLIR pass C++ generation utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for generating C++ code for pass declarations
// and definitions from TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_PASSGEN_H
#define MLIR_TABLEGEN_GENERATORS_PASSGEN_H

#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Extract the list of passes from the TableGen records.
std::vector<Pass> getPasses(const llvm::RecordKeeper &records);

/// Emit the struct definition used to set pass options programmatically.
/// Emits nothing if the pass has no options.
void emitPassOptionsStruct(const Pass &pass, llvm::raw_ostream &os);

/// Emit the public declarations for a single pass (guarded by
/// GEN_PASS_DECL_<PASSNAME>).
void emitPassDecls(const Pass &pass, llvm::raw_ostream &os);

/// Emit the base class definition for a single pass (guarded by
/// GEN_PASS_DEF_<PASSNAME>).
void emitPassDefs(const Pass &pass, llvm::raw_ostream &os);

/// Emit the option member declarations for a single pass.
void emitPassOptionDecls(const Pass &pass, llvm::raw_ostream &os);

/// Emit the statistic member declarations for a single pass.
void emitPassStatisticDecls(const Pass &pass, llvm::raw_ostream &os);

/// Emit registration code for all passes. groupName is the name of the pass
/// group used in the generated `register<GroupName>Passes()` function.
void emitRegistrations(llvm::ArrayRef<Pass> passes, llvm::StringRef groupName,
                       llvm::raw_ostream &os);

/// Emit the complete header content (declarations + definitions) for a single
/// pass.
void emitPass(const Pass &pass, llvm::raw_ostream &os);

/// Emit the complete header content for all passes in records, including
/// registration code. groupName names the generated registration function.
void emitPasses(const llvm::RecordKeeper &records, llvm::StringRef groupName,
                llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_PASSGEN_H
