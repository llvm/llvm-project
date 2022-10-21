//===- HLSLResource.cpp - HLSL Resource helper objects --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with HLSL Resources.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace llvm::hlsl;

GlobalVariable *FrontendResource::getGlobalVariable() {
  return cast<GlobalVariable>(
      cast<ConstantAsMetadata>(Entry->getOperand(0))->getValue());
}

StringRef FrontendResource::getSourceType() {
  return cast<MDString>(Entry->getOperand(1))->getString();
}

uint32_t FrontendResource::FrontendResource::getResourceKind() {
  return cast<ConstantInt>(
             cast<ConstantAsMetadata>(Entry->getOperand(2))->getValue())
      ->getLimitedValue();
}
uint32_t FrontendResource::getResourceIndex() {
  return cast<ConstantInt>(
             cast<ConstantAsMetadata>(Entry->getOperand(3))->getValue())
      ->getLimitedValue();
}
uint32_t FrontendResource::getSpace() {
  return cast<ConstantInt>(
             cast<ConstantAsMetadata>(Entry->getOperand(4))->getValue())
      ->getLimitedValue();
}

FrontendResource::FrontendResource(GlobalVariable *GV, StringRef TypeStr,
                                   ResourceKind RK, uint32_t ResIndex,
                                   uint32_t Space) {
  auto &Ctx = GV->getContext();
  IRBuilder<> B(Ctx);
  Entry = MDNode::get(
      Ctx, {ValueAsMetadata::get(GV), MDString::get(Ctx, TypeStr),
            ConstantAsMetadata::get(B.getInt32(static_cast<int>(RK))),
            ConstantAsMetadata::get(B.getInt32(ResIndex)),
            ConstantAsMetadata::get(B.getInt32(Space))});
}
