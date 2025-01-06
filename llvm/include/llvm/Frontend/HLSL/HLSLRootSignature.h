//===- HLSLRootSignature.h - HLSL Root Signature helper objects -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with HLSL Root
/// Signatures.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
#define LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H

#include <stdint.h>

#include "llvm/Support/Endian.h"

namespace llvm {
namespace hlsl {
namespace root_signature {

// This is a copy from DebugInfo/CodeView/CodeView.h
#define RS_DEFINE_ENUM_CLASS_FLAGS_OPERATORS(Class)                            \
  inline Class operator|(Class a, Class b) {                                   \
    return static_cast<Class>(llvm::to_underlying(a) |                         \
                              llvm::to_underlying(b));                         \
  }                                                                            \
  inline Class operator&(Class a, Class b) {                                   \
    return static_cast<Class>(llvm::to_underlying(a) &                         \
                              llvm::to_underlying(b));                         \
  }                                                                            \
  inline Class operator~(Class a) {                                            \
    return static_cast<Class>(~llvm::to_underlying(a));                        \
  }                                                                            \
  inline Class &operator|=(Class &a, Class b) {                                \
    a = a | b;                                                                 \
    return a;                                                                  \
  }                                                                            \
  inline Class &operator&=(Class &a, Class b) {                                \
    a = a & b;                                                                 \
    return a;                                                                  \
  }

// Various enumerations and flags

enum class RootFlags : uint32_t {
  None = 0,
  AllowInputAssemblerInputLayout = 0x1,
  DenyVertexShaderRootAccess = 0x2,
  DenyHullShaderRootAccess = 0x4,
  DenyDomainShaderRootAccess = 0x8,
  DenyGeometryShaderRootAccess = 0x10,
  DenyPixelShaderRootAccess = 0x20,
  AllowStreamOutput = 0x40,
  LocalRootSignature = 0x80,
  DenyAmplificationShaderRootAccess = 0x100,
  DenyMeshShaderRootAccess = 0x200,
  CBVSRVUAVHeapDirectlyIndexed = 0x400,
  SamplerHeapDirectlyIndexed = 0x800,
  AllowLowTierReservedHwCbLimit = 0x80000000,
  ValidFlags = 0x80000fff
};
RS_DEFINE_ENUM_CLASS_FLAGS_OPERATORS(RootFlags)

// Define the in-memory layout structures

struct RootElement {
  enum class ElementType {
    RootFlags,
  };

  ElementType Tag;
  union {
    RootFlags Flags;
  };

  // Constructors
  RootElement(RootFlags Flags) : Tag(ElementType::RootFlags), Flags(Flags) {}
};

} // namespace root_signature
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
