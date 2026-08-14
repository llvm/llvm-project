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

static constexpr unsigned MaxSignatureRows = 32;
static constexpr unsigned MaxSignatureCols = 4;
static constexpr unsigned MaxClipCullRows = 2;
static constexpr unsigned MaxGeometryStreams = 4;

/// Denotes the element that could not be packed and why, so that the caller
/// can report a diagnostic that points at the offending element.
class SignaturePackingError : public ErrorInfo<SignaturePackingError> {
public:
  /// Denotes why an element could not be packed into its signature.
  enum ErrorKind {
    /// The element does not fit in the 32 rows of the signature.
    SignatureOverflow,
    /// The element does not fit in the two rows that are dedicated to clip and
    /// cull distances.
    ClipCullOverflow,
  };

  LLVM_ABI static char ID;

  SignaturePackingError(ErrorKind Kind, unsigned ElementIndex)
      : Kind(Kind), ElementIndex(ElementIndex) {}

  ErrorKind getErrorKind() const { return Kind; }

  /// Index into the Elements that were being packed.
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
/// ShaderStage and SignatureKind and 'packs' each element into 32 registers
/// with 4 components by updating their StartRow and StartCol in place. An
/// element is left unallocated if it is not part of the signature for the
/// signature type.
///
/// Returns a SignaturePackingError that denotes the first element that cannot
/// be placed, or success if all eligible elements were placed.
///
/// With the exception of some special cases listed below, the packing
/// algorithm can be visualised as placing rectangles onto a grid of 32 rows and
/// 4 cols, where each element is placed at the first compatible row and can't
/// be moved after. So each SemanticSignatureElement will cover the rectangle
/// defined by its position of (StartCol, StartRow) and size of (Cols, Rows).
///
/// For example:
///
/// struct Foo {
///   float3   A[3] : A;
///   float1x2 B    : B; // column-major
///   float2   C    : C;
///   float    D    : D;
/// };
///
/// Packs into:
/// reg0: A[0].xyz | B[0][0].w
/// reg1: A[1].xyz | B[0][1].w
/// reg2: A[2].xyz | D.w
/// reg3: C.xy     | unused.zw
///
/// As we can see, elements can be co-packed into the same register (row), this
/// is restricted by the following:
///
/// Interpolation Mode: An element may only be placed in a row if its
/// InterpMode is equivalent to all other non-Undefined InterpModes. This
/// implies that if all InterpModes are undefined then it can be placed.
///
/// Component Width: An element may only be placed in a row if its CompType's
/// bitwidth is equivalent. This is only applicable when UseNative16BitTypes is
/// set and so half/float16_t/int16_t have a different bitwidth.
///
/// Component Order: An element may only be placed in a row if all existing
/// elements have a lesser or equal semantic category, defined as follows:
/// an arbitrary value (Foo) < a system value (SV_RenderTargetArrayIndex)
/// < a system generated value (SV_PrimitiveID). These are internally
/// categorized and are dependent on the ShaderStage and SignatureKind.
///
/// Dynamic Indexing: An element that covers more than one row may be indexed
/// dynamically, which makes every row it covers dynamically indexable. System
/// values and system generated values may not be placed in such a row.
///
/// Special Cases:
///
/// Clip and Cull Distances: Elements categorized as clip or cull distances are
/// packed only with each other into dedicated rows. Together they may use at
/// most eight components spread over at most two rows; if any element
/// covers multiple rows, the two dedicated rows must be adjacent. These rules
/// do not apply when a ClipDistance or CullDistance SemanticKind is categorized
/// as an arbitrary value.
///
/// Tessellation Factor: An element denoting a tessellation factor that covers
/// multiple rows is searched for only in the last column.
///
/// Geometry Streams: For a geometry shader output signature, elements that
/// carry different GSStream values are packed into independent grids.
LLVM_ABI Error packSignaturePrefixStable(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, SemanticSignatureKind SignatureKind,
    bool UseNative16BitTypes);

} // namespace llvm::hlsl

#endif // LLVM_FRONTEND_HLSL_SEMANTICSIGNATUREPACKING_H
