//===- LLVMOps.h - LLVM operation implementation helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_LLVMIR_IR_LLVMOPS_H
#define MLIR_LIB_DIALECT_LLVMIR_IR_LLVMOPS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <optional>

namespace mlir::LLVM {

mlir::Type getI1SameShape(mlir::Type type);
mlir::LLVM::LLVMStructType getValAndBoolStructType(mlir::Type valType);

void printLLVMLinkage(mlir::OpAsmPrinter &p, mlir::Operation *,
                      mlir::LLVM::LinkageAttr val);
mlir::ParseResult parseLLVMLinkage(mlir::OpAsmParser &p,
                                   mlir::LLVM::LinkageAttr &val);

void printOpBundles(mlir::OpAsmPrinter &p, mlir::Operation *op,
                    mlir::OperandRangeRange opBundleOperands,
                    mlir::TypeRangeRange opBundleOperandTypes,
                    std::optional<mlir::ArrayAttr> opBundleTags);
std::optional<mlir::ParseResult> parseOpBundles(
    mlir::OpAsmParser &p,
    mlir::SmallVector<mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>>
        &opBundleOperands,
    mlir::SmallVector<mlir::SmallVector<mlir::Type>> &opBundleOperandTypes,
    mlir::ArrayAttr &opBundleTags);

mlir::ParseResult parseSwitchOpCases(
    mlir::OpAsmParser &parser, mlir::Type flagType,
    mlir::DenseIntElementsAttr &caseValues,
    mlir::SmallVectorImpl<mlir::Block *> &caseDestinations,
    mlir::SmallVectorImpl<
        mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &caseOperands,
    mlir::SmallVectorImpl<mlir::SmallVector<mlir::Type>> &caseOperandTypes);
void printSwitchOpCases(mlir::OpAsmPrinter &p, mlir::LLVM::SwitchOp op,
                        mlir::Type flagType,
                        mlir::DenseIntElementsAttr caseValues,
                        mlir::SuccessorRange caseDestinations,
                        mlir::OperandRangeRange caseOperands,
                        const mlir::TypeRangeRange &caseOperandTypes);

mlir::ParseResult parseGEPIndices(
    mlir::OpAsmParser &parser,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &indices,
    mlir::DenseI32ArrayAttr &rawConstantIndices);
void printGEPIndices(mlir::OpAsmPrinter &printer, mlir::LLVM::GEPOp gepOp,
                     mlir::OperandRange indices,
                     mlir::DenseI32ArrayAttr rawConstantIndices);

mlir::ParseResult parseInsertExtractValueElementType(
    mlir::AsmParser &parser, mlir::Type &valueType, mlir::Type containerType,
    mlir::DenseI64ArrayAttr position);
void printInsertExtractValueElementType(mlir::AsmPrinter &printer,
                                        mlir::Operation *op,
                                        mlir::Type valueType,
                                        mlir::Type containerType,
                                        mlir::DenseI64ArrayAttr position);

mlir::ParseResult parseShuffleType(mlir::AsmParser &parser, mlir::Type v1Type,
                                   mlir::Type &resType,
                                   mlir::DenseI32ArrayAttr mask);
void printShuffleType(mlir::AsmPrinter &printer, mlir::Operation *op,
                      mlir::Type v1Type, mlir::Type resType,
                      mlir::DenseI32ArrayAttr mask);

mlir::ParseResult parseIndirectBrOpSucessors(
    mlir::OpAsmParser &parser, mlir::Type &flagType,
    mlir::SmallVectorImpl<mlir::Block *> &succOperandBlocks,
    mlir::SmallVectorImpl<
        mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &succOperands,
    mlir::SmallVectorImpl<mlir::SmallVector<mlir::Type>> &succOperandsTypes);
void printIndirectBrOpSucessors(mlir::OpAsmPrinter &p,
                                mlir::LLVM::IndirectBrOp op,
                                mlir::Type flagType, mlir::SuccessorRange succs,
                                mlir::OperandRangeRange succOperands,
                                const mlir::TypeRangeRange &succOperandsTypes);

} // namespace mlir::LLVM

#include "mlir/Dialect/LLVMIR/LLVMAllOps.h.inc"

#endif // MLIR_LIB_DIALECT_LLVMIR_IR_LLVMOPS_H
