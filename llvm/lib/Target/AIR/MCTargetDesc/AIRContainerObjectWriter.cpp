//===-- AIRContainerObjectWriter.cpp - AIR target object writer -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin target-side AIRLib object writer; pairs the AIR backend with the
// core AIRLibObjectWriter (which actually emits the .metallib bytes).
//
//===----------------------------------------------------------------------===//

#include "AIRContainerObjectWriter.h"
#include "llvm/MC/MCAIRLibObjectWriter.h"

using namespace llvm;

namespace {
class AIRContainerObjectWriter : public MCAIRLibTargetWriter {
public:
  AIRContainerObjectWriter() : MCAIRLibTargetWriter() {}
};
} // namespace

std::unique_ptr<MCObjectTargetWriter>
llvm::createAIRContainerTargetObjectWriter() {
  return std::make_unique<AIRContainerObjectWriter>();
}
