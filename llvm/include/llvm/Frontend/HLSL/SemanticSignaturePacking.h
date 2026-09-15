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
class LLVM_ABI SignaturePackingError : public ErrorInfo<SignaturePackingError> {
public:
  enum ErrorKind {
    SignatureOverflow,
  };

  static char ID;

  SignaturePackingError(ErrorKind Kind, unsigned ElementIndex)
      : Kind(Kind), ElementIndex(ElementIndex) {}

  ErrorKind getErrorKind() const { return Kind; }
  unsigned getElementIndex() const { return ElementIndex; }

  void log(raw_ostream &OS) const override;

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
///
/// On failure, Elements is left partially packed: the elements preceding the
/// one reported by the returned SignaturePackingError keep the locations
/// they were assigned, while that element and the ones following it retain the
/// unallocated row and column sentinels.
LLVM_ABI Expected<unsigned>
packSignatureStacked(MutableArrayRef<SemanticSignatureElement> Elements,
                     Triple::EnvironmentType ShaderStage, IOType IOTy);

} // namespace llvm::hlsl

#endif // LLVM_FRONTEND_HLSL_SEMANTICSIGNATUREPACKING_H
