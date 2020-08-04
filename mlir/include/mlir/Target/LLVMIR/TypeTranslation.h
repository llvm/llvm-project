//===- TypeTranslation.h - Translate types between MLIR & LLVM --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the type translation function going from MLIR LLVM dialect
// to LLVM IR and back.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_TYPETRANSLATION_H
#define MLIR_TARGET_LLVMIR_TYPETRANSLATION_H

namespace llvm {
class LLVMContext;
class Type;
} // namespace llvm

namespace mlir {

class MLIRContext;

namespace LLVM {

class LLVMTypeNew;

llvm::Type *translateTypeToLLVMIR(LLVMTypeNew type, llvm::LLVMContext &context);
LLVMTypeNew translateTypeFromLLVMIR(llvm::Type *type, MLIRContext &context);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_TYPETRANSLATION_H
