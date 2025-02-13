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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/DXILABI.h"
#include <variant>

namespace llvm {
class LLVMContext;
class MDNode;
class Metadata;

namespace hlsl {
namespace rootsig {

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

// Definition of the various enumerations and flags

enum class DescriptorRangeOffset : uint32_t;

enum class DescriptorRangeFlags : unsigned {
  None = 0,
  DescriptorsVolatile = 0x1,
  DataVolatile = 0x2,
  DataStaticWhileSetAtExecute = 0x4,
  DataStatic = 0x8,
  DescriptorsStaticKeepingBufferBoundsChecks = 0x10000,
  ValidFlags = 0x1000f,
  ValidSamplerFlags = DescriptorsVolatile,
};
RS_DEFINE_ENUM_CLASS_FLAGS_OPERATORS(DescriptorRangeFlags)

enum class ShaderVisibility {
  All = 0,
  Vertex = 1,
  Hull = 2,
  Domain = 3,
  Geometry = 4,
  Pixel = 5,
  Amplification = 6,
  Mesh = 7,
};

// Definitions of the in-memory data layout structures

// Models the different registers: bReg | tReg | uReg | sReg
enum class RegisterType { BReg, TReg, UReg, SReg };
struct Register {
  RegisterType ViewType;
  uint32_t Number;
};

// Models the end of a descriptor table and stores its visibility
struct DescriptorTable {
  ShaderVisibility Visibility = ShaderVisibility::All;
  uint32_t NumClauses = 0; // The number of clauses in the table
};

static const DescriptorRangeOffset DescriptorTableOffsetAppend =
    DescriptorRangeOffset(0xffffffff);
// Models DTClause : CBV | SRV | UAV | Sampler, by collecting like parameters
using ClauseType = llvm::dxil::ResourceClass;
struct DescriptorTableClause {
  ClauseType Type;
  Register Register;
  uint32_t NumDescriptors = 1;
  uint32_t Space = 0;
  DescriptorRangeOffset Offset = DescriptorTableOffsetAppend;
  DescriptorRangeFlags Flags;

  void SetDefaultFlags() {
    switch (Type) {
    case ClauseType::CBuffer:
      Flags = DescriptorRangeFlags::DataStaticWhileSetAtExecute;
      break;
    case ClauseType::SRV:
      Flags = DescriptorRangeFlags::DataStaticWhileSetAtExecute;
      break;
    case ClauseType::UAV:
      Flags = DescriptorRangeFlags::DataVolatile;
      break;
    case ClauseType::Sampler:
      Flags = DescriptorRangeFlags::None;
      break;
    }
  }
};

// Models RootElement : DescriptorTable | DescriptorTableClause
using RootElement = std::variant<DescriptorTable, DescriptorTableClause>;

// Models a reference to all assignment parameter types that any RootElement
// may have. Things of the form: Keyword = Param
using ParamType = std::variant<uint32_t *, DescriptorRangeOffset *,
                               DescriptorRangeFlags *, ShaderVisibility *>;

class MetadataBuilder {
public:
  MetadataBuilder(llvm::LLVMContext &Ctx, ArrayRef<RootElement> Elements)
      : Ctx(Ctx), Elements(Elements) {}

  /// Iterates through the elements and dispatches onto the correct Build method
  ///
  /// Accumulates the root signature and returns the Metadata node that is just
  /// a list of all the elements
  MDNode *BuildRootSignature();

private:
  /// Define the various builders for the different metadata types
  MDNode *BuildDescriptorTable(const DescriptorTable &Table);
  MDNode *BuildDescriptorTableClause(const DescriptorTableClause &Clause);

  llvm::LLVMContext &Ctx;
  ArrayRef<RootElement> Elements;
  SmallVector<Metadata *> GeneratedMetadata;
};

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
