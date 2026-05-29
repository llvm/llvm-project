//===-- MetalContainerObjectWriter.cpp - Metal target object writer -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin target-side MetalLib object writer; pairs the Metal backend with the
// core MetalLibObjectWriter (which actually emits the .metallib bytes).
//
//===----------------------------------------------------------------------===//

#include "MetalContainerObjectWriter.h"
#include "llvm/MC/MCMetalLibObjectWriter.h"

using namespace llvm;

namespace {
class MetalContainerObjectWriter : public MCMetalLibTargetWriter {
public:
  MetalContainerObjectWriter() : MCMetalLibTargetWriter() {}
};
} // namespace

std::unique_ptr<MCObjectTargetWriter>
llvm::createMetalContainerTargetObjectWriter() {
  return std::make_unique<MetalContainerObjectWriter>();
}
