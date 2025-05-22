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
#include "llvm/Support/DXILABI.h"
#include "llvm/Support/raw_ostream.h"
#include <variant>

namespace llvm {
class LLVMContext;
class MDNode;
class Metadata;

namespace hlsl {
namespace rootsig {

// Definition of the various enumerations and flags

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
  ValidFlags = 0x00000fff
};

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

// Models the parameter values of root constants
struct RootConstants {
  uint32_t Num32BitConstants;
  Register Reg;
  uint32_t Space = 0;
  ShaderVisibility Visibility = ShaderVisibility::All;
};

using DescriptorType = llvm::dxil::ResourceClass;
// Models RootDescriptor : CBV | SRV | UAV, by collecting like parameters
struct RootDescriptor {
  DescriptorType Type;
  Register Reg;
};

// Models the end of a descriptor table and stores its visibility
struct DescriptorTable {
  ShaderVisibility Visibility = ShaderVisibility::All;
  // Denotes that the previous NumClauses in the RootElement array
  // are the clauses in the table.
  uint32_t NumClauses = 0;

  void dump(raw_ostream &OS) const;
};

static const uint32_t NumDescriptorsUnbounded = 0xffffffff;
static const uint32_t DescriptorTableOffsetAppend = 0xffffffff;
// Models DTClause : CBV | SRV | UAV | Sampler, by collecting like parameters
using ClauseType = llvm::dxil::ResourceClass;
struct DescriptorTableClause {
  ClauseType Type;
  Register Reg;
  uint32_t NumDescriptors = 1;
  uint32_t Space = 0;
  uint32_t Offset = DescriptorTableOffsetAppend;
  DescriptorRangeFlags Flags;

  void setDefaultFlags() {
    switch (Type) {
    case ClauseType::CBuffer:
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

  void dump(raw_ostream &OS) const;
};

/// Models RootElement : RootFlags | RootConstants | RootDescriptor
///  | DescriptorTable | DescriptorTableClause
///
/// A Root Signature is modeled in-memory by an array of RootElements. These
/// aim to map closely to their DSL grammar reprsentation defined in the spec.
///
/// Each optional parameter has its default value defined in the struct, and,
/// each mandatory parameter does not have a default initialization.
///
/// For the variants RootFlags, RootConstants and DescriptorTableClause: each
/// data member maps directly to a parameter in the grammar.
///
/// The DescriptorTable is modelled by having its Clauses as the previous
/// RootElements in the array, and it holds a data member for the Visibility
/// parameter.
using RootElement = std::variant<RootFlags, RootConstants, RootDescriptor,
                                 DescriptorTable, DescriptorTableClause>;

void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements);

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
