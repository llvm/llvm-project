//===------ EmitCToCppTranslation.h - EmitC Dialect to Cpp-------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for EmitC dialect to Cpp translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_DIALECT_EMITC_EMITCTOCPPTRANSLATION_H
#define MLIR_TARGET_CPP_DIALECT_EMITC_EMITCTOCPPTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the EmitC dialect and the translation from it to the Cpp in the
/// given registry;
void registerEmitCDialectCppTranslation(DialectRegistry &registry);

/// Register the EmitC dialect and the translation from it in the registry
/// associated with the given context.
void registerEmitCDialectCppTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_CPP_DIALECT_EMITC_EMITCTOCPPTRANSLATION_H
