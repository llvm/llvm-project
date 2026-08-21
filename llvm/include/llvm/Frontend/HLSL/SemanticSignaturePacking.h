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

/// Iterates through Elements that belong to the signature described by
/// ShaderStage and IOTy and packs each element into 32 registers with 4
/// components by updating its StartRow and StartCol in place. An element is
/// left unallocated if it is not part of the signature.
///
/// Elements are visited in declaration order. Each element starts at column
/// zero of the first row after the preceding element, and a multi-row element
/// occupies consecutive rows. Elements are never co-packed into the same row;
/// interpolation mode, component type, and semantic kind do not otherwise
/// affect placement.
///
/// Returns a SignaturePackingError that denotes the first element that cannot
/// be placed, or success if all eligible elements were placed.
LLVM_ABI Error
packSignatureStacked(MutableArrayRef<SemanticSignatureElement> Elements,
                     Triple::EnvironmentType ShaderStage, IOType IOTy);

} // namespace llvm::hlsl

#endif // LLVM_FRONTEND_HLSL_SEMANTICSIGNATUREPACKING_H
