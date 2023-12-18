//===- HLSLResource.h - HLSL Resource helper objects ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with HLSL Resources.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLRESOURCE_H
#define LLVM_FRONTEND_HLSL_HLSLRESOURCE_H

#include "llvm/IR/Metadata.h"

namespace llvm {
class GlobalVariable;

namespace hlsl {

enum class ResourceClass : uint8_t {
  SRV = 0,
  UAV,
  CBuffer,
  Sampler,
  Invalid,
  NumClasses = Invalid,
};

// The value ordering of this enumeration is part of the DXIL ABI. Elements
// can only be added to the end, and not removed.
enum class ResourceKind : uint32_t {
  Invalid = 0,
  Texture1D,
  Texture2D,
  Texture2DMS,
  Texture3D,
  TextureCube,
  Texture1DArray,
  Texture2DArray,
  Texture2DMSArray,
  TextureCubeArray,
  TypedBuffer,
  RawBuffer,
  StructuredBuffer,
  CBuffer,
  Sampler,
  TBuffer,
  RTAccelerationStructure,
  FeedbackTexture2D,
  FeedbackTexture2DArray,
  NumEntries,
};

// The value ordering of this enumeration is part of the DXIL ABI. Elements
// can only be added to the end, and not removed.
enum class ElementType : uint32_t {
  Invalid = 0,
  I1,
  I16,
  U16,
  I32,
  U32,
  I64,
  U64,
  F16,
  F32,
  F64,
  SNormF16,
  UNormF16,
  SNormF32,
  UNormF32,
  SNormF64,
  UNormF64,
  PackedS8x32,
  PackedU8x32,
};

class FrontendResource {
  MDNode *Entry;

public:
  FrontendResource(MDNode *E) : Entry(E) {
    assert(Entry->getNumOperands() == 6 && "Unexpected metadata shape");
  }

  FrontendResource(GlobalVariable *GV, ResourceKind RK, ElementType ElTy,
                   bool IsROV, uint32_t ResIndex, uint32_t Space);

  GlobalVariable *getGlobalVariable();
  StringRef getSourceType();
  ResourceKind getResourceKind();
  ElementType getElementType();
  bool getIsROV();
  uint32_t getResourceIndex();
  uint32_t getSpace();
  MDNode *getMetadata() { return Entry; }
};
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLRESOURCE_H
