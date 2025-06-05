//===-- DXILABI.h - ABI Sensitive Values for DXIL ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of various constants and enums that are
// required to remain stable as per the DXIL format's requirements.
//
// Documentation for DXIL can be found in
// https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DXILABI_H
#define LLVM_SUPPORT_DXILABI_H

#include <cstdint>

namespace llvm {
namespace dxil {

enum class ResourceClass : uint8_t {
  SRV = 0,
  UAV,
  CBuffer,
  Sampler,
};

/// The kind of resource for an SRV or UAV resource. Sometimes referred to as
/// "Shape" in the DXIL docs.
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

/// The element type of an SRV or UAV resource.
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

/// Metadata tags for extra resource properties.
enum class ExtPropTags : uint32_t {
  ElementType = 0,
  StructuredBufferStride = 1,
  SamplerFeedbackKind = 2,
  Atomic64Use = 3,
};

enum class SamplerType : uint32_t {
  Default = 0,
  Comparison = 1,
  Mono = 2, // Note: Seems to be unused.
};

enum class SamplerFeedbackType : uint32_t {
  MinMip = 0,
  MipRegionUsed = 1,
};

const unsigned MinWaveSize = 4;
const unsigned MaxWaveSize = 128;

// Definition of the various d3d12.h enumerations and flags. The definitions of
// all values here correspond to their description in the d3d12.h header and
// are carried over from their values in DXC. For reference:
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/

// D3D12_ROOT_SIGNATURE_FLAGS
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

// D3D12_ROOT_DESCRIPTOR_FLAGS
enum class RootDescriptorFlags : unsigned {
  None = 0,
  DataVolatile = 0x2,
  DataStaticWhileSetAtExecute = 0x4,
  DataStatic = 0x8,
  ValidFlags = 0xe,
};

// D3D12_DESCRIPTOR_RANGE_FLAGS
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

// D3D12_SHADER_VISIBILITY
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

// D3D12_FILTER
enum class SamplerFilter {
  MinMagMipPoint = 0,
  MinMagPointMipLinear = 0x1,
  MinPointMagLinearMipPoint = 0x4,
  MinPointMagMipLinear = 0x5,
  MinLinearMagMipPoint = 0x10,
  MinLinearMagPointMipLinear = 0x11,
  MinMagLinearMipPoint = 0x14,
  MinMagMipLinear = 0x15,
  Anisotropic = 0x55,
  ComparisonMinMagMipPoint = 0x80,
  ComparisonMinMagPointMipLinear = 0x81,
  ComparisonMinPointMagLinearMipPoint = 0x84,
  ComparisonMinPointMagMipLinear = 0x85,
  ComparisonMinLinearMagMipPoint = 0x90,
  ComparisonMinLinearMagPointMipLinear = 0x91,
  ComparisonMinMagLinearMipPoint = 0x94,
  ComparisonMinMagMipLinear = 0x95,
  ComparisonAnisotropic = 0xd5,
  MinimumMinMagMipPoint = 0x100,
  MinimumMinMagPointMipLinear = 0x101,
  MinimumMinPointMagLinearMipPoint = 0x104,
  MinimumMinPointMagMipLinear = 0x105,
  MinimumMinLinearMagMipPoint = 0x110,
  MinimumMinLinearMagPointMipLinear = 0x111,
  MinimumMinMagLinearMipPoint = 0x114,
  MinimumMinMagMipLinear = 0x115,
  MinimumAnisotropic = 0x155,
  MaximumMinMagMipPoint = 0x180,
  MaximumMinMagPointMipLinear = 0x181,
  MaximumMinPointMagLinearMipPoint = 0x184,
  MaximumMinPointMagMipLinear = 0x185,
  MaximumMinLinearMagMipPoint = 0x190,
  MaximumMinLinearMagPointMipLinear = 0x191,
  MaximumMinMagLinearMipPoint = 0x194,
  MaximumMinMagMipLinear = 0x195,
  MaximumAnisotropic = 0x1d5
};

// D3D12_TEXTURE_ADDRESS_MODE
enum class TextureAddressMode {
  Wrap = 1,
  Mirror = 2,
  Clamp = 3,
  Border = 4,
  MirrorOnce = 5
};

// D3D12_COMPARISON_FUNC
enum class ComparisonFunc : unsigned {
  Never = 1,
  Less = 2,
  Equal = 3,
  LessEqual = 4,
  Greater = 5,
  NotEqual = 6,
  GreaterEqual = 7,
  Always = 8
};

// D3D12_STATIC_BORDER_COLOR
enum class StaticBorderColor {
  TransparentBlack = 0,
  OpaqueBlack = 1,
  OpaqueWhite = 2,
  OpaqueBlackUint = 3,
  OpaqueWhiteUint = 4
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_SUPPORT_DXILABI_H
