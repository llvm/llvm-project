//===- NVVMWarpOps.cpp - NVVM dialect warp-level ops ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Warp` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMWarpOps.h"

#include "IR/NVVMOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Casting.h"

#include <optional>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

LogicalResult ShflOp::verify() {
  auto returnStructType = llvm::dyn_cast<LLVM::LLVMStructType>(getType());

  auto verifyTypeError = [&](Twine desc, Type expectedType,
                             Type actualType) -> LogicalResult {
    return emitOpError("expected " + desc + " to be of type ")
           << expectedType << " but got " << actualType << " instead";
  };

  if (returnStructType) {
    if (!getReturnValueAndIsValid())
      return emitOpError("\"return_value_and_is_valid\" attribute must be "
                         "specified when the return type is a struct type");

    if (returnStructType.getBody().size() != 2)
      return emitOpError("expected return type to be a two-element struct");

    llvm::ArrayRef<Type> returnStruct = returnStructType.getBody();
    auto resultType = returnStruct[0];
    if (resultType != getVal().getType())
      return verifyTypeError("first element in the returned struct",
                             getVal().getType(), resultType);

    auto predicateType = returnStruct[1];
    if (!predicateType.isInteger(1))
      return verifyTypeError("second element in the returned struct",
                             mlir::IntegerType::get(getContext(), 1),
                             predicateType);
  } else {
    if (getReturnValueAndIsValid())
      return emitOpError("expected return type to be a two-element struct");

    if (getType() != getVal().getType())
      return verifyTypeError("return type", getVal().getType(), getType());
  }
  return success();
}

LogicalResult
ShflOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ShflOp::Adaptor adaptor,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  Type valType = adaptor.getVal().getType();
  if (adaptor.getReturnValueAndIsValid())
    inferredReturnTypes.push_back(LLVM::LLVMStructType::getLiteral(
        context, {valType, IntegerType::get(context, 1)}));
  else
    inferredReturnTypes.push_back(valType);
  return success();
}

LogicalResult NVVM::MatchSyncOp::verify() {
  if (getKind() == NVVM::MatchSyncKind::all) {
    auto type = llvm::dyn_cast<LLVM::LLVMStructType>(getType());
    if (!type || type.getBody().size() != 2 ||
        !type.getBody()[0].isInteger(32) || !type.getBody()[1].isInteger(1)) {
      return emitOpError("match.sync 'all' returns a two element struct with "
                         "first element as i32 and second element as i1");
    }
  } else {
    if (!getType().isInteger(32)) {
      return emitOpError("match.sync 'any' returns an i32");
    }
  }
  return success();
}

LogicalResult MatchSyncOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    MatchSyncOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  if (adaptor.getKind() == NVVM::MatchSyncKind::all)
    inferredReturnTypes.push_back(LLVM::LLVMStructType::getLiteral(
        context,
        {IntegerType::get(context, 32), IntegerType::get(context, 1)}));
  else
    inferredReturnTypes.push_back(IntegerType::get(context, 32));
  return success();
}

LogicalResult NVVM::VoteSyncOp::verify() {
  if (getKind() == NVVM::VoteSyncKind::ballot) {
    if (!getType().isInteger(32)) {
      return emitOpError("vote.sync 'ballot' returns an i32");
    }
  } else {
    if (!getType().isInteger(1)) {
      return emitOpError("vote.sync 'any', 'all' and 'uni' returns an i1");
    }
  }
  return success();
}

LogicalResult VoteSyncOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    VoteSyncOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  unsigned width = adaptor.getKind() == NVVM::VoteSyncKind::ballot ? 32 : 1;
  inferredReturnTypes.push_back(IntegerType::get(context, width));
  return success();
}

LogicalResult NVVM::ReduxOp::verify() {
  mlir::Type reduxType = getType();

  if (!reduxType.isF32()) {
    if (getAbs())
      return emitOpError("abs attribute is supported only for f32 type");
    if (getNan())
      return emitOpError("nan attribute is supported only for f32 type");
  }

  NVVM::ReductionKind kind = getKind();
  switch (kind) {
  case NVVM::ReductionKind::ADD:
  case NVVM::ReductionKind::AND:
  case NVVM::ReductionKind::OR:
  case NVVM::ReductionKind::XOR:
  case NVVM::ReductionKind::MAX:
  case NVVM::ReductionKind::MIN:
  case NVVM::ReductionKind::UMAX:
  case NVVM::ReductionKind::UMIN:
    if (!reduxType.isInteger(32))
      return emitOpError("'")
             << kind << "' reduction kind unsupported with " << reduxType
             << " type. Only supported type is 'i32'.";
    break;
  case NVVM::ReductionKind::FMIN:
  case NVVM::ReductionKind::FMAX:
    if (!reduxType.isF32())
      return emitOpError("'")
             << kind << "' reduction kind unsupported with " << reduxType
             << " type. Only supported type is 'f32'.";
    break;
  }

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMWarpOps.cpp.inc"
