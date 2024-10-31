// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_TABLEGEN_LLVMIR_H_
#define MLIR_TABLEGEN_LLVMIR_H_

#include "mlir/Support/LLVM.h"

#include "llvm/Support/CommandLine.h"

namespace llvm {
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Emit all builders.  Returns false on success because of the generator
// registration requirements.
bool emitLLVMIRConversionBuilders(const llvm::RecordKeeper &records,
                                  raw_ostream &os);

// Emit all op builders. Returns false on success because of the
// generator registration requirements.
bool emitLLVMIROpMLIRBuilders(const llvm::RecordKeeper &records,
                              raw_ostream &os);

// Emit all intrinsic MLIR builders. Returns false on success because of the
// generator registration requirements.
bool emitLLVMIRIntrMLIRBuilders(const llvm::RecordKeeper &records,
                                raw_ostream &os);

// Emits conversion function "LLVMClass convertEnumToLLVM(Enum)" and containing
// switch-based logic to convert from the MLIR LLVM dialect enum attribute case
// (Enum) to the corresponding LLVM API enumerant
void emitOneEnumToConversion(const llvm::Record *record, raw_ostream &os);

// Emits conversion function "Enum convertEnumFromLLVM(LLVMClass)" and
// containing switch-based logic to convert from the LLVM API enumerant to MLIR
// LLVM dialect enum attribute (Enum).
void emitOneEnumFromConversion(const llvm::Record *record, raw_ostream &os);

// Emits conversion function "LLVMClass convertEnumToLLVM(Enum)" and containing
// switch-based logic to convert from the MLIR LLVM dialect enum attribute case
// (Enum) to the corresponding LLVM API C-style enumerant
void emitOneCEnumToConversion(const llvm::Record *record, raw_ostream &os);

// Emits conversion function "Enum convertEnumFromLLVM(LLVMEnum)" and
// containing switch-based logic to convert from the LLVM API C-style enumerant
// to MLIR LLVM dialect enum attribute (Enum).
void emitOneCEnumFromConversion(const llvm::Record *record, raw_ostream &os);

// Emit the list of LLVM IR intrinsics identifiers that are convertible to a
// matching MLIR LLVM dialect intrinsic operation.
bool emitConvertibleLLVMIRIntrinsics(const llvm::RecordKeeper &records,
                                     raw_ostream &os);

/// Traverses the list of TableGen definitions derived from the "Intrinsic"
/// class and generates MLIR ODS definitions for those intrinsics that have
/// the name matching the filter.
bool emitLLVMIRIntrinsics(const llvm::RecordKeeper &records,
                          llvm::raw_ostream &os,
                          const llvm::cl::opt<std::string> &nameFilter,
                          const std::string &accessGroupRegexp,
                          const std::string &aliasAnalysisRegexp,
                          const std::string &opBaseClass);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_LLVMIR_H_
