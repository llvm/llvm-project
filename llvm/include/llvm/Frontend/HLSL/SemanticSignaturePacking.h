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

static constexpr unsigned MaxSignatureRows = 32;
static constexpr unsigned MaxSignatureCols = 4;

/// Denotes the element that could not be packed and why.
class SignaturePackingError : public ErrorInfo<SignaturePackingError> {
public:
  enum ErrorKind {
    SignatureOverflow,
  };

  LLVM_ABI static char ID;

  SignaturePackingError(ErrorKind Kind, unsigned ElementIndex)
      : Kind(Kind), ElementIndex(ElementIndex) {}

  ErrorKind getErrorKind() const { return Kind; }
  unsigned getElementIndex() const { return ElementIndex; }

  LLVM_ABI void log(raw_ostream &OS) const override;

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

private:
  ErrorKind Kind;
  unsigned ElementIndex;
};

/// Packs eligible signature elements into consecutive rows.
///
/// See llvm/docs/DirectX/SemanticSignatures.md#stacked-packing for details.
LLVM_ABI Error
packSignatureStacked(MutableArrayRef<SemanticSignatureElement> Elements,
                     Triple::EnvironmentType ShaderStage, IOType IOTy);

/// Packs each eligible element at the row denoted by its semantic index and at
/// column zero. Declaration order does not affect placement, and gaps between
/// semantic indices remain unused. An element is left unallocated if it is not
/// part of the signature.
///
/// Returns a SignaturePackingError that denotes the first element that cannot
/// be placed, or success if all eligible elements were placed.
LLVM_ABI Error
packSignatureIndexed(MutableArrayRef<SemanticSignatureElement> Elements,
                     Triple::EnvironmentType ShaderStage, IOType IOTy);

} // namespace llvm::hlsl

#endif // LLVM_FRONTEND_HLSL_SEMANTICSIGNATUREPACKING_H
