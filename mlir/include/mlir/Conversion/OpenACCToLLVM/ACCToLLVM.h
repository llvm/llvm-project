//===- ACCToLLVM.h - Convert OpenACC to LLVM dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVM_H
#define MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVM_H

#include "mlir/Dialect/OpenACC/OpenACCRuntimeUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include <functional>
#include <memory>

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class Operation;
class Pass;
class RewritePatternSet;
class Region;
class SymbolTable;

namespace acc {
class OpenACCSupport;
} // namespace acc

#define GEN_PASS_DECL_CONVERTACCTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

/// Configure conversion legality for OpenACC executable directives lowered to
/// runtime calls.
void configureACCExecutableDirectiveConversionLegality(
    ConversionTarget &target);

/// Populate patterns that lower OpenACC executable directives (init, shutdown,
/// wait, set) to LLVM runtime calls. The runtime declarations and globals the
/// patterns add are created in \p globalSymbolRegion and registered in
/// \p symbolTable.
void populateACCExecutableDirectivePatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    Region &globalSymbolRegion, SymbolTable &symbolTable,
    const acc::ACCRuntimeCallConfig &config = {});

/// Returns the address operand of a dialect-specific load operation. Used when
/// moving capture/update dependencies into a cmpxchg loop. Patterns already
/// recognize `llvm.load` and `memref.load`; this callback covers other dialect
/// specific loads.
using ACCAtomicLoadAddressCallback = std::function<Value(Operation *)>;

/// Configure conversion legality for OpenACC atomic operations.
void configureACCAtomicConversionLegality(ConversionTarget &target);

/// Populate patterns that lower OpenACC atomic operations to LLVM dialect.
void populateACCAtomicPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    acc::OpenACCSupport &accSupport,
    ACCAtomicLoadAddressCallback getLoadAddress = {});

/// Configure conversion legality for OpenACC data directives.
void configureACCDataDirectiveConversionLegality(ConversionTarget &target);

/// Populate patterns that lower OpenACC data directives (`acc.data`,
/// `enter_data`, `exit_data`, `update`) to `__tgt_acc_data_*` runtime calls.
/// Clauses that can be repeated per device type are taken from the ones that
/// apply to \p clauseDeviceType. The runtime declarations and globals the
/// patterns add are created in \p globalSymbolRegion and registered in
/// \p symbolTable.
void populateACCDataDirectivePatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    acc::OpenACCSupport &accSupport, Region &globalSymbolRegion,
    SymbolTable &symbolTable, const acc::ACCRuntimeCallConfig &config = {},
    acc::DeviceType clauseDeviceType = acc::DeviceType::None);

/// Populate the patterns that remove OpenACC data clause operations once the
/// constructs holding them have turned their mappings into runtime calls. A
/// data entry operation is replaced by the address of the object it named, and
/// a data exit or bounds operation is erased. Clause operations whose result
/// is not that address stay with the construct that holds them.
///
/// A conversion has to populate these only once every construct holding such a
/// clause operation is lowered in the same conversion, as the mappings would
/// otherwise be lost.
void populateACCDataClauseOpPatterns(LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVM_H
