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

#include <functional>
#include <memory>

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class Operation;
class Pass;
class RewritePatternSet;
class Value;

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
/// wait, set) to LLVM runtime calls.
void populateACCExecutableDirectivePatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
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

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVM_H
