//===- SemanticSignatures.h - HLSL Semantic Signature helper objects ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains structure definitions of HLSL Semantic Signature
/// objects.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_SEMANTICSIGNATURES_H
#define LLVM_FRONTEND_HLSL_SEMANTICSIGNATURES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DXILABI.h"
#include <cstdint>
#include <string>

namespace llvm {
namespace hlsl {

// Definitions of the in-memory data layout structures

// Sentinel values denoting that an element is unallocated
static constexpr uint32_t UnallocatedRow = ~0U;
static constexpr uint8_t UnallocatedCol = 0xFF;

// Models a single packed range of signature rows with its semantic name and
// indices, register placement, component masks, and stage-specific attributes.
struct SemanticSignatureElement {
  uint32_t SigId;
  StringRef SemanticName;
  dxil::ElementType CompType = dxil::ElementType::Invalid;
  dxbc::PSV::SemanticKind SemanticKind = dxbc::PSV::SemanticKind::Arbitrary;
  SmallVector<uint32_t> SemanticIndices;
  dxbc::PSV::InterpolationMode InterpMode =
      dxbc::PSV::InterpolationMode::Undefined;
  uint32_t Rows = 1;
  uint8_t Cols = 1;
  uint32_t StartRow = UnallocatedRow;
  uint8_t StartCol = UnallocatedCol;
  uint8_t UsageMask = 0;
  uint8_t DynIndexMask = 0;
  uint32_t GSStream = 0;

  bool isAllocated() const {
    return StartRow != UnallocatedRow && StartCol != UnallocatedCol;
  }

  uint8_t getDeclaredMask() const {
    if (!isAllocated())
      return 0;
    return static_cast<uint8_t>(((1U << Cols) - 1U) << StartCol);
  }

  uint8_t getAlwaysReadsMask() const { return UsageMask; }

  uint8_t getNeverWritesMask() const {
    return static_cast<uint8_t>(~UsageMask & getDeclaredMask());
  }

  dxbc::SigMinPrecision getMinPrecision(bool UseMinPrecision) const {
    if (!UseMinPrecision)
      return dxbc::SigMinPrecision::Default;
    switch (CompType) {
    case dxil::ElementType::F16:
      return dxbc::SigMinPrecision::Float16;
    case dxil::ElementType::I16:
    case dxil::ElementType::SNormF16:
    case dxil::ElementType::UNormF16:
      return dxbc::SigMinPrecision::SInt16;
    case dxil::ElementType::U16:
      return dxbc::SigMinPrecision::UInt16;
    default:
      return dxbc::SigMinPrecision::Default;
    }
  }
};

} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_SEMANTICSIGNATURES_H
