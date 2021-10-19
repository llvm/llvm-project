//===- StructBuilder.cpp - Helper for building LLVM structs  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// StructBuilder implementation
//===----------------------------------------------------------------------===//

StructBuilder::StructBuilder(Value v) : value(v), structType(v.getType()) {
  assert(value != nullptr && "value cannot be null");
  assert(LLVM::isCompatibleType(structType) && "expected llvm type");
}

Value StructBuilder::extractPtr(OpBuilder &builder, Location loc,
                                unsigned pos) {
  Type type = structType.cast<LLVM::LLVMStructType>().getBody()[pos];
  return builder.create<LLVM::ExtractValueOp>(loc, type, value,
                                              builder.getI64ArrayAttr(pos));
}

void StructBuilder::setPtr(OpBuilder &builder, Location loc, unsigned pos,
                           Value ptr) {
  value = builder.create<LLVM::InsertValueOp>(loc, structType, value, ptr,
                                              builder.getI64ArrayAttr(pos));
}
