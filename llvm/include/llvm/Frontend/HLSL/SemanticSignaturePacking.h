//===- SemanticSignaturePacking.h - HLSL signature packing helpers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file declares helpers for packing HLSL semantic signatures.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_SEMANTICSIGNATUREPACKING_H
#define LLVM_FRONTEND_HLSL_SEMANTICSIGNATUREPACKING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm::hlsl {

/// Identifies which signature associated with an entry point is being packed.
enum class SemanticSignatureKind {
  Input,
  Output,
  PatchConstOrPrim,
};

LLVM_ABI Error
packSignaturePrefixStable(MutableArrayRef<SemanticSignatureElement> Elements,
                          Triple::EnvironmentType ShaderStage,
                          SemanticSignatureKind SignatureKind,
                          bool UseNative16BitTypes);

} // namespace llvm::hlsl

#endif // LLVM_FRONTEND_HLSL_SEMANTICSIGNATUREPACKING_H
