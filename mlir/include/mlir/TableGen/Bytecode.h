//===-- Bytecode.h - Bytecode definitions -*- C++ -----------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_TABLEGEN_BYTECODE_H
#define MLIR_TABLEGEN_BYTECODE_H

namespace llvm {
class RecordKeeper;
} // namespace llvm

namespace mlir::tblgen {
bool emitBCRW(const llvm::RecordKeeper &records, raw_ostream &os,
              const std::string &selectedBcDialect);
} // namespace mlir::tblgen

#endif // MLIR_TABLEGEN_BYTECODE_H
