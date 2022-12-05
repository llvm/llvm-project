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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>

namespace llvm {
class Module;
class GlobalVariable;

namespace dxil {

class ResourceBase {
protected:
  uint32_t ID;
  GlobalVariable *GV;
  StringRef Name;
  uint32_t Space;
  uint32_t LowerBound;
  uint32_t RangeSize;
  ResourceBase(uint32_t I, hlsl::FrontendResource R);

  void write(LLVMContext &Ctx, MutableArrayRef<Metadata *> Entries) const;

  void print(raw_ostream &O, StringRef IDPrefix, StringRef BindingPrefix) const;
  using Kinds = hlsl::ResourceKind;
  static StringRef getKindName(Kinds Kind);
  static void printKind(Kinds Kind, unsigned Alignment, raw_ostream &OS,
                        bool SRV = false, bool HasCounter = false,
                        uint32_t SampleCount = 0);

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

  static StringRef getComponentTypeName(ComponentType CompType);
  static void printComponentType(Kinds Kind, ComponentType CompType,
                                 unsigned Alignment, raw_ostream &OS);

public:
  struct ExtendedProperties {
    std::optional<ComponentType> ElementType;

    // The value ordering of this enumeration is part of the DXIL ABI. Elements
    // can only be added to the end, and not removed.
    enum Tags : uint32_t {
      TypedBufferElementType = 0,
      StructuredBufferElementStride,
      SamplerFeedbackKind,
      Atomic64Use
    };

    MDNode *write(LLVMContext &Ctx) const;
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
  UAVResource(uint32_t I, hlsl::FrontendResource R);

  MDNode *write() const;
  void print(raw_ostream &O) const;
};

// FIXME: Fully computing the resource structures requires analyzing the IR
// because some flags are set based on what operations are performed on the
// resource. This partial patch handles some of the leg work, but not all of it.
// See issue https://github.com/llvm/llvm-project/issues/57936.
class Resources {
  llvm::SmallVector<UAVResource> UAVs;

  void collectUAVs(Module &M);

public:
  void collect(Module &M);
  void write(Module &M) const;
  void print(raw_ostream &O) const;
  LLVM_DUMP_METHOD void dump() const;
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILRESOURCE_H
