//===- DXILResource.h - DXIL Resource helper objects ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with DXIL Resources.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILRESOURCE_H
#define LLVM_TARGET_DIRECTX_DXILRESOURCE_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Metadata.h"
#include <cstdint>

namespace llvm {
class Module;
class GlobalVariable;

namespace dxil {

// FIXME: Ultimately this class and some of these utilities should be moved into
// a new LLVMFrontendHLSL library so that they can be reused in Clang.
// See issue https://github.com/llvm/llvm-project/issues/58000.
class FrontendResource {
  MDNode *Entry;

public:
  FrontendResource(MDNode *E) : Entry(E) {
    assert(Entry->getNumOperands() == 3 && "Unexpected metadata shape");
  }

  GlobalVariable *getGlobalVariable();
  StringRef getSourceType();
  Constant *getID();
};

class ResourceBase {
protected:
  uint32_t ID;
  GlobalVariable *GV;
  StringRef Name;
  uint32_t Space;
  uint32_t LowerBound;
  uint32_t RangeSize;
  ResourceBase(uint32_t I, FrontendResource R);

  void write(LLVMContext &Ctx, MutableArrayRef<Metadata *> Entries);

  // The value ordering of this enumeration is part of the DXIL ABI. Elements
  // can only be added to the end, and not removed.
  enum class Kinds : uint32_t {
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
  enum class ComponentType : uint32_t {
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
    LastEntry
  };

public:
  struct ExtendedProperties {
    llvm::Optional<ComponentType> ElementType;

    // The value ordering of this enumeration is part of the DXIL ABI. Elements
    // can only be added to the end, and not removed.
    enum Tags : uint32_t {
      TypedBufferElementType = 0,
      StructuredBufferElementStride,
      SamplerFeedbackKind,
      Atomic64Use
    };

    MDNode *write(LLVMContext &Ctx);
  };
};

class UAVResource : public ResourceBase {
  ResourceBase::Kinds Shape;
  bool GloballyCoherent;
  bool HasCounter;
  bool IsROV;
  ResourceBase::ExtendedProperties ExtProps;

  void parseSourceType(StringRef S);

public:
  UAVResource(uint32_t I, FrontendResource R);

  MDNode *write();
};

// FIXME: Fully computing the resource structures requires analyzing the IR
// because some flags are set based on what operations are performed on the
// resource. This partial patch handles some of the leg work, but not all of it.
// See issue https://github.com/llvm/llvm-project/issues/57936.
class Resources {
  Module &Mod;
  llvm::SmallVector<UAVResource> UAVs;

  void collectUAVs();

public:
  Resources(Module &M) : Mod(M) { collectUAVs(); }

  void write();
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILRESOURCE_H
