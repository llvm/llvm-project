//===- LLVMGen.h - Generator info -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_LLVMGEN_H_
#define MLIR_TABLEGEN_LLVMGEN_H_

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

namespace mlir::tblgen {

bool emitLLVMBuilders(const llvm::RecordKeeper &recordKeeper, raw_ostream &os);
bool emitLLVMOpMLIRBuilders(const llvm::RecordKeeper &recordKeeper,
                            raw_ostream &os);
bool emitLLVMIntrMLIRBuilders(const llvm::RecordKeeper &recordKeeper,
                              raw_ostream &os);
template <bool ConvertTo>
bool emitLLVMEnumConversionDefs(const llvm::RecordKeeper &recordKeeper,
                                raw_ostream &os);
bool emitLLVMConvertibleIntrinsics(const llvm::RecordKeeper &recordKeeper,
                                   raw_ostream &os);
bool emitLLVMIntrinsics(const llvm::RecordKeeper &records,
                        llvm::raw_ostream &os, const std::string &nameFilter,
                        const std::string &accessGroupRegexp,
                        const std::string &aliasAnalysisRegexp,
                        const std::string &opBaseClass);

} // namespace mlir::tblgen

#endif // MLIR_TABLEGEN_LLVMGEN_H_
