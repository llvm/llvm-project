//===- NVVMMiscOps.cpp - NVVM dialect miscellaneous ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Misc` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMMiscOps.h"

#include "IR/NVVMOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/Casting.h"

#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

LogicalResult PMEventOp::verify() {
  auto eventId = getEventId();
  auto maskedEventId = getMaskedEventId();
  if (!maskedEventId && !eventId) {
    return emitOpError() << "either `id` or `mask` must be set";
  }

  if (maskedEventId && eventId) {
    return emitOpError() << "`id` and `mask` cannot be set at the same time";
  }

  if (eventId) {
    if (eventId < 0 || eventId > 15) {
      return emitOpError() << "`id` must be between 0 and 15";
    }
  }

  return llvm::success();
}

LogicalResult NVVM::SetMaxRegisterOp::verify() {
  if (getRegCount() % 8)
    return emitOpError("new register size must be multiple of 8");
  if (getRegCount() < 24 || getRegCount() > 256)
    return emitOpError("new register size must be in between 24 to 256");
  return success();
}

//===----------------------------------------------------------------------===//
// getIntrinsicID/getIntrinsicIDAndArgs methods
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair
PMEventOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                                 llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::PMEventOp>(op);
  llvm::Type *i16Ty = llvm::Type::getInt16Ty(mt.getLLVMContext());

  // With event-id, mask is generated as (1 << event-id)
  llvm::Value *maskVal;
  if (auto eventAttr = thisOp.getEventIdAttr()) {
    uint16_t mask = static_cast<uint16_t>(1u << eventAttr.getInt());
    maskVal = llvm::ConstantInt::get(i16Ty, mask);
  } else {
    maskVal =
        llvm::ConstantInt::get(i16Ty, thisOp.getMaskedEventIdAttr().getValue());
  }

  return {llvm::Intrinsic::nvvm_pm_event_mask, {maskVal}};
}

bool NVVM::InlinePtxOp::getAsmValues(
    RewriterBase &rewriter,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>>
        &asmValues) {
  for (auto arg : getReadWriteArgs())
    asmValues.push_back({arg, mlir::NVVM::PTXRegisterMod::ReadWrite});
  for (auto arg : getResults())
    asmValues.push_back({arg, mlir::NVVM::PTXRegisterMod::Write});
  for (auto arg : getReadOnlyArgs())
    asmValues.push_back({arg, mlir::NVVM::PTXRegisterMod::Read});
  if (getPredicate())
    asmValues.push_back({getPredicate(), mlir::NVVM::PTXRegisterMod::Read});
  return false; // No manual mapping needed
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMiscOps.cpp.inc"
