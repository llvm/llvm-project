//===- SPIRVOpDefinition.h - SPIR-V op implementation helpers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_SPIRV_IR_SPIRVOPDEFINITION_H
#define MLIR_LIB_DIALECT_SPIRV_IR_SPIRVOPDEFINITION_H

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

namespace mlir::spirv {

bool isNestedInFunctionOpInterface(Operation *op);
bool isNestedInGraphARMOpInterface(Operation *op);
bool isDirectInModuleLikeOp(Operation *op);
Type getMatchingBoolType(Type operandType);

ParseResult parseImageOperands(OpAsmParser &parser,
                               spirv::ImageOperandsAttr &attr);
void printImageOperands(OpAsmPrinter &printer, Operation *imageOp,
                        spirv::ImageOperandsAttr attr);

ParseResult parseSwitchOpCases(
    OpAsmParser &parser, Type &selectorType, Block *&defaultTarget,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &defaultOperands,
    SmallVectorImpl<Type> &defaultOperandTypes, DenseIntElementsAttr &literals,
    SmallVectorImpl<Block *> &targets,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>>
        &targetOperands,
    SmallVectorImpl<SmallVector<Type>> &targetOperandTypes);
void printSwitchOpCases(OpAsmPrinter &p, SwitchOp op, Type selectorType,
                        Block *defaultTarget, OperandRange defaultOperands,
                        TypeRange defaultOperandTypes,
                        DenseIntElementsAttr literals, SuccessorRange targets,
                        OperandRangeRange targetOperands,
                        const TypeRangeRange &targetOperandTypes);

} // namespace mlir::spirv

#endif // MLIR_LIB_DIALECT_SPIRV_IR_SPIRVOPDEFINITION_H
