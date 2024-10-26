//===- Directive.h - Directive class --------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_TABLEGEN_DIRECTIVE_H_
#define MLIR_TABLEGEN_DIRECTIVE_H_

#include "mlir/Support/LLVM.h"
#include "llvm/TableGen/Record.h"

#include <string>
#include <vector>

namespace mlir {
namespace tblgen {
bool emitDirectiveDecls(const llvm::RecordKeeper &records,
                        llvm::StringRef dialect, raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_DIRECTIVE_H_
