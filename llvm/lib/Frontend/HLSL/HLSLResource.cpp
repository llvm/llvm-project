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

Constant *FrontendResource::getID() {
  return cast<ConstantAsMetadata>(Entry->getOperand(2))->getValue();
}

FrontendResource::FrontendResource(GlobalVariable *GV, StringRef TypeStr,
                                   uint32_t Counter) {
  auto &Ctx = GV->getContext();
  IRBuilder<> B(Ctx);
  Entry =
      MDNode::get(Ctx, {ValueAsMetadata::get(GV), MDString::get(Ctx, TypeStr),
                        ConstantAsMetadata::get(B.getInt32(Counter))});
}
