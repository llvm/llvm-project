//===------- FuncToCppTranslation.h - Func Dialect to Cpp -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for Func dialect to Cpp translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_DIALECT_FUNC_FUNCTOCPPTRANSLATION_H
#define MLIR_TARGET_CPP_DIALECT_FUNC_FUNCTOCPPTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the Func dialect and the translation from it to the Cpp in the
/// given registry;
void registerFuncDialectCppTranslation(DialectRegistry &registry);

/// Register the Func dialect and the translation from it in the registry
/// associated with the given context.
void registerFuncDialectCppTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_CPP_DIALECT_FUNC_FUNCTOCPPTRANSLATION_H
