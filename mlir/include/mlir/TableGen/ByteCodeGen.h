//===- ByteCodeGen.h - Generator info ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_BYTECODEGEN_H_
#define MLIR_TABLEGEN_BYTECODEGEN_H_

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

namespace mlir::tblgen {

bool emitBCRW(const llvm::RecordKeeper &records, raw_ostream &os,
              const std::string &selectedBcDialect);

} // namespace mlir::tblgen

#endif // MLIR_TABLEGEN_BYTECODEGEN_H_
