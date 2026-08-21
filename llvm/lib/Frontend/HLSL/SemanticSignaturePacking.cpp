//===- SemanticSignaturePacking.cpp - HLSL signature packing helpers -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements helpers for packing HLSL semantic signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/SemanticSignaturePacking.h"

using namespace llvm;
using namespace llvm::hlsl;

Error llvm::hlsl::packSignatureStacked(
    MutableArrayRef<SemanticSignatureElement>, Triple::EnvironmentType,
    IOType) {
  return Error::success();
}
