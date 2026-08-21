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
#include "llvm/ADT/STLExtras.h"
#include <cassert>

using namespace llvm;
using namespace llvm::hlsl;

char SignaturePackingError::ID;

void SignaturePackingError::log(raw_ostream &OS) const {
  switch (Kind) {
  case SignatureOverflow:
    OS << "signature elements do not fit in 32 rows";
    break;
  }
  OS << " (element " << ElementIndex << ")";
}

Error llvm::hlsl::packSignatureStacked(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy) {
  unsigned NextRow = 0;
  for (const auto &[Index, Element] : enumerate(Elements)) {
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && "signature element must have at least one row");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");

    SemanticInterpretation Interpretation =
        getInterpretationKind(Element.SemanticKind, ShaderStage, IOTy);
    if (Interpretation == SemanticInterpretation::NotAllocated)
      continue;

    assert((Interpretation == SemanticInterpretation::Arbitrary ||
            Interpretation == SemanticInterpretation::SV ||
            Interpretation == SemanticInterpretation::SGV) &&
           "unexpected semantic interpretation for stacked packing");

    if (Element.Rows > MaxSignatureRows - NextRow)
      return make_error<SignaturePackingError>(
          SignaturePackingError::SignatureOverflow,
          static_cast<unsigned>(Index));

    Element.StartRow = NextRow;
    Element.StartCol = 0;
    NextRow += Element.Rows;
  }

  return Error::success();
}

Error llvm::hlsl::packSignatureIndexed(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy) {
  for (const auto &[Index, Element] : enumerate(Elements)) {
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && "signature element must have at least one row");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");

    SemanticInterpretation Interpretation =
        getInterpretationKind(Element.SemanticKind, ShaderStage, IOTy);
    if (Interpretation == SemanticInterpretation::NotAllocated)
      continue;

    assert(Interpretation == SemanticInterpretation::Target &&
           "unexpected semantic interpretation for indexed packing");
    assert(Element.Rows == 1 && Element.SemanticIndices.size() == 1 &&
           "target elements must occupy one semantic row");

    const uint32_t Row = Element.SemanticIndices.front();
    if (Row >= MaxSignatureRows)
      return make_error<SignaturePackingError>(
          SignaturePackingError::SignatureOverflow,
          static_cast<unsigned>(Index));

    Element.StartRow = Row;
    Element.StartCol = 0;
  }

  return Error::success();
}
