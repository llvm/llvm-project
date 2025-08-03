//===-- llvm/BinaryFormat/DXContainer.h - The DXBC file format --*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines manifest constants for the DXContainer object file format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_DXCONTAINER_H
#define LLVM_BINARYFORMAT_DXCONTAINER_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/TargetParser/Triple.h"

#include <stdint.h>

namespace llvm {
template <typename T> struct EnumEntry;

// The DXContainer file format is arranged as a header and "parts". Semantically
// parts are similar to sections in other object file formats. The File format
// structure is roughly:

// ┌────────────────────────────────┐
// │             Header             │
// ├────────────────────────────────┤
// │              Part              │
// ├────────────────────────────────┤
// │              Part              │
// ├────────────────────────────────┤
// │              ...               │
// └────────────────────────────────┘

namespace dxbc {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

inline Triple::EnvironmentType getShaderStage(uint32_t Kind) {
  assert(Kind <= Triple::Amplification - Triple::Pixel &&
         "Shader kind out of expected range.");
  return static_cast<Triple::EnvironmentType>(Triple::Pixel + Kind);
}

struct Hash {
  uint8_t Digest[16];
};

enum class HashFlags : uint32_t {
  None = 0,           // No flags defined.
  IncludesSource = 1, // This flag indicates that the shader hash was computed
                      // taking into account source information (-Zss)
};

struct ShaderHash {
  uint32_t Flags; // dxbc::HashFlags
  uint8_t Digest[16];

  LLVM_ABI bool isPopulated();

  void swapBytes() { sys::swapByteOrder(Flags); }
};

struct ContainerVersion {
  uint16_t Major;
  uint16_t Minor;

  void swapBytes() {
    sys::swapByteOrder(Major);
    sys::swapByteOrder(Minor);
  }
};

struct Header {
  uint8_t Magic[4]; // "DXBC"
  Hash FileHash;
  ContainerVersion Version;
  uint32_t FileSize;
  uint32_t PartCount;

  void swapBytes() {
    Version.swapBytes();
    sys::swapByteOrder(FileSize);
    sys::swapByteOrder(PartCount);
  }
  // Structure is followed by part offsets: uint32_t PartOffset[PartCount];
  // The offset is to a PartHeader, which is followed by the Part Data.
};

/// Use this type to describe the size and type of a DXIL container part.
struct PartHeader {
  uint8_t Name[4];
  uint32_t Size;

  void swapBytes() { sys::swapByteOrder(Size); }
  StringRef getName() const {
    return StringRef(reinterpret_cast<const char *>(&Name[0]), 4);
  }
  // Structure is followed directly by part data: uint8_t PartData[PartSize].
};

struct BitcodeHeader {
  uint8_t Magic[4];     // ACSII "DXIL".
  uint8_t MinorVersion; // DXIL version.
  uint8_t MajorVersion; // DXIL version.
  uint16_t Unused;
  uint32_t Offset; // Offset to LLVM bitcode (from start of header).
  uint32_t Size;   // Size of LLVM bitcode (in bytes).
  // Followed by uint8_t[BitcodeHeader.Size] at &BitcodeHeader + Header.Offset

  void swapBytes() {
    sys::swapByteOrder(MinorVersion);
    sys::swapByteOrder(MajorVersion);
    sys::swapByteOrder(Offset);
    sys::swapByteOrder(Size);
  }
};

struct ProgramHeader {
  uint8_t Version;
  uint8_t Unused;
  uint16_t ShaderKind;
  uint32_t Size; // Size in uint32_t words including this header.
  BitcodeHeader Bitcode;

  void swapBytes() {
    sys::swapByteOrder(ShaderKind);
    sys::swapByteOrder(Size);
    Bitcode.swapBytes();
  }
  uint8_t getMajorVersion() { return Version >> 4; }
  uint8_t getMinorVersion() { return Version & 0xF; }
  static uint8_t getVersion(uint8_t Major, uint8_t Minor) {
    return (Major << 4) | Minor;
  }
};

static_assert(sizeof(ProgramHeader) == 24, "ProgramHeader Size incorrect!");

#define CONTAINER_PART(Part) Part,
enum class PartType {
  Unknown = 0,
#include "DXContainerConstants.def"
};

#define SHADER_FEATURE_FLAG(Num, DxilModuleNum, Val, Str) Val = 1ull << Num,
enum class FeatureFlags : uint64_t {
#include "DXContainerConstants.def"
};
static_assert((uint64_t)FeatureFlags::NextUnusedBit <= 1ull << 63,
              "Shader flag bits exceed enum size.");

#define ROOT_SIGNATURE_FLAG(Num, Val) Val = Num,
enum class RootFlags : uint32_t {
#include "DXContainerConstants.def"

  LLVM_MARK_AS_BITMASK_ENUM(SamplerHeapDirectlyIndexed)
};

LLVM_ABI ArrayRef<EnumEntry<RootFlags>> getRootFlags();

#define ROOT_DESCRIPTOR_FLAG(Num, Enum, Flag) Enum = Num,
enum class RootDescriptorFlags : uint32_t {
#include "DXContainerConstants.def"

  LLVM_MARK_AS_BITMASK_ENUM(DataStatic)
};

LLVM_ABI ArrayRef<EnumEntry<RootDescriptorFlags>> getRootDescriptorFlags();

#define DESCRIPTOR_RANGE_FLAG(Num, Enum, Flag) Enum = Num,
enum class DescriptorRangeFlags : uint32_t {
#include "DXContainerConstants.def"

  LLVM_MARK_AS_BITMASK_ENUM(DescriptorsStaticKeepingBufferBoundsChecks)
};

LLVM_ABI ArrayRef<EnumEntry<DescriptorRangeFlags>> getDescriptorRangeFlags();

#define ROOT_PARAMETER(Val, Enum) Enum = Val,
enum class RootParameterType : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<RootParameterType>> getRootParameterTypes();

#define DESCRIPTOR_RANGE(Val, Enum) Enum = Val,
enum class DescriptorRangeType : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<DescriptorRangeType>> getDescriptorRangeTypes();

#define ROOT_PARAMETER(Val, Enum)                                              \
  case Val:                                                                    \
    return true;
inline bool isValidParameterType(uint32_t V) {
  switch (V) {
#include "DXContainerConstants.def"
  }
  return false;
}

#define SHADER_VISIBILITY(Val, Enum) Enum = Val,
enum class ShaderVisibility : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<ShaderVisibility>> getShaderVisibility();

#define SHADER_VISIBILITY(Val, Enum)                                           \
  case Val:                                                                    \
    return true;
inline bool isValidShaderVisibility(uint32_t V) {
  switch (V) {
#include "DXContainerConstants.def"
  }
  return false;
}

#define FILTER(Val, Enum) Enum = Val,
enum class SamplerFilter : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<SamplerFilter>> getSamplerFilters();

#define TEXTURE_ADDRESS_MODE(Val, Enum) Enum = Val,
enum class TextureAddressMode : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<TextureAddressMode>> getTextureAddressModes();

#define COMPARISON_FUNC(Val, Enum) Enum = Val,
enum class ComparisonFunc : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<ComparisonFunc>> getComparisonFuncs();

#define STATIC_BORDER_COLOR(Val, Enum) Enum = Val,
enum class StaticBorderColor : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<StaticBorderColor>> getStaticBorderColors();

LLVM_ABI PartType parsePartType(StringRef S);

struct VertexPSVInfo {
  uint8_t OutputPositionPresent;
  uint8_t Unused[3];

  void swapBytes() {
    // nothing to swap
  }
};

struct HullPSVInfo {
  uint32_t InputControlPointCount;
  uint32_t OutputControlPointCount;
  uint32_t TessellatorDomain;
  uint32_t TessellatorOutputPrimitive;

  void swapBytes() {
    sys::swapByteOrder(InputControlPointCount);
    sys::swapByteOrder(OutputControlPointCount);
    sys::swapByteOrder(TessellatorDomain);
    sys::swapByteOrder(TessellatorOutputPrimitive);
  }
};

struct DomainPSVInfo {
  uint32_t InputControlPointCount;
  uint8_t OutputPositionPresent;
  uint8_t Unused[3];
  uint32_t TessellatorDomain;

  void swapBytes() {
    sys::swapByteOrder(InputControlPointCount);
    sys::swapByteOrder(TessellatorDomain);
  }
};

struct GeometryPSVInfo {
  uint32_t InputPrimitive;
  uint32_t OutputTopology;
  uint32_t OutputStreamMask;
  uint8_t OutputPositionPresent;
  uint8_t Unused[3];

  void swapBytes() {
    sys::swapByteOrder(InputPrimitive);
    sys::swapByteOrder(OutputTopology);
    sys::swapByteOrder(OutputStreamMask);
  }
};

struct PixelPSVInfo {
  uint8_t DepthOutput;
  uint8_t SampleFrequency;
  uint8_t Unused[2];

  void swapBytes() {
    // nothing to swap
  }
};

struct MeshPSVInfo {
  uint32_t GroupSharedBytesUsed;
  uint32_t GroupSharedBytesDependentOnViewID;
  uint32_t PayloadSizeInBytes;
  uint16_t MaxOutputVertices;
  uint16_t MaxOutputPrimitives;

  void swapBytes() {
    sys::swapByteOrder(GroupSharedBytesUsed);
    sys::swapByteOrder(GroupSharedBytesDependentOnViewID);
    sys::swapByteOrder(PayloadSizeInBytes);
    sys::swapByteOrder(MaxOutputVertices);
    sys::swapByteOrder(MaxOutputPrimitives);
  }
};

struct AmplificationPSVInfo {
  uint32_t PayloadSizeInBytes;

  void swapBytes() { sys::swapByteOrder(PayloadSizeInBytes); }
};

union PipelinePSVInfo {
  VertexPSVInfo VS;
  HullPSVInfo HS;
  DomainPSVInfo DS;
  GeometryPSVInfo GS;
  PixelPSVInfo PS;
  MeshPSVInfo MS;
  AmplificationPSVInfo AS;

  void swapBytes(Triple::EnvironmentType Stage) {
    switch (Stage) {
    case Triple::EnvironmentType::Pixel:
      PS.swapBytes();
      break;
    case Triple::EnvironmentType::Vertex:
      VS.swapBytes();
      break;
    case Triple::EnvironmentType::Geometry:
      GS.swapBytes();
      break;
    case Triple::EnvironmentType::Hull:
      HS.swapBytes();
      break;
    case Triple::EnvironmentType::Domain:
      DS.swapBytes();
      break;
    case Triple::EnvironmentType::Mesh:
      MS.swapBytes();
      break;
    case Triple::EnvironmentType::Amplification:
      AS.swapBytes();
      break;
    default:
      break;
    }
  }
};

static_assert(sizeof(PipelinePSVInfo) == 4 * sizeof(uint32_t),
              "Pipeline-specific PSV info must fit in 16 bytes.");

namespace PSV {

#define SEMANTIC_KIND(Val, Enum) Enum = Val,
enum class SemanticKind : uint8_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<SemanticKind>> getSemanticKinds();

#define COMPONENT_TYPE(Val, Enum) Enum = Val,
enum class ComponentType : uint8_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<ComponentType>> getComponentTypes();

#define INTERPOLATION_MODE(Val, Enum) Enum = Val,
enum class InterpolationMode : uint8_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<InterpolationMode>> getInterpolationModes();

#define RESOURCE_TYPE(Val, Enum) Enum = Val,
enum class ResourceType : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<ResourceType>> getResourceTypes();

#define RESOURCE_KIND(Val, Enum) Enum = Val,
enum class ResourceKind : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<ResourceKind>> getResourceKinds();

#define RESOURCE_FLAG(Index, Enum) bool Enum = false;
struct ResourceFlags {
  ResourceFlags() : Flags(0U) {};
  struct FlagsBits {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  };
  union {
    uint32_t Flags;
    FlagsBits Bits;
  };
  bool operator==(const uint32_t RFlags) const { return Flags == RFlags; }
};

namespace v0 {
struct RuntimeInfo {
  PipelinePSVInfo StageInfo;
  uint32_t MinimumWaveLaneCount; // minimum lane count required, 0 if unused
  uint32_t MaximumWaveLaneCount; // maximum lane count required,
                                 // 0xffffffff if unused
  void swapBytes() {
    // Skip the union because we don't know which field it has
    sys::swapByteOrder(MinimumWaveLaneCount);
    sys::swapByteOrder(MaximumWaveLaneCount);
  }

  void swapBytes(Triple::EnvironmentType Stage) { StageInfo.swapBytes(Stage); }
};

struct ResourceBindInfo {
  ResourceType Type;
  uint32_t Space;
  uint32_t LowerBound;
  uint32_t UpperBound;

  void swapBytes() {
    sys::swapByteOrder(Type);
    sys::swapByteOrder(Space);
    sys::swapByteOrder(LowerBound);
    sys::swapByteOrder(UpperBound);
  }
};

struct SignatureElement {
  uint32_t NameOffset;
  uint32_t IndicesOffset;

  uint8_t Rows;
  uint8_t StartRow;
  uint8_t Cols : 4;
  uint8_t StartCol : 2;
  uint8_t Allocated : 1;
  uint8_t Unused : 1;
  SemanticKind Kind;

  ComponentType Type;
  InterpolationMode Mode;
  uint8_t DynamicMask : 4;
  uint8_t Stream : 2;
  uint8_t Unused2 : 2;
  uint8_t Reserved;

  void swapBytes() {
    sys::swapByteOrder(NameOffset);
    sys::swapByteOrder(IndicesOffset);
  }
};

static_assert(sizeof(SignatureElement) == 4 * sizeof(uint32_t),
              "PSV Signature elements must fit in 16 bytes.");

} // namespace v0

namespace v1 {

struct MeshRuntimeInfo {
  uint8_t SigPrimVectors; // Primitive output for MS
  uint8_t MeshOutputTopology;
};

union GeometryExtraInfo {
  uint16_t MaxVertexCount;            // MaxVertexCount for GS only (max 1024)
  uint8_t SigPatchConstOrPrimVectors; // Output for HS; Input for DS;
                                      // Primitive output for MS (overlaps
                                      // MeshInfo::SigPrimVectors)
  MeshRuntimeInfo MeshInfo;
};
struct RuntimeInfo : public v0::RuntimeInfo {
  uint8_t ShaderStage; // PSVShaderKind
  uint8_t UsesViewID;
  GeometryExtraInfo GeomData;

  // PSVSignatureElement counts
  uint8_t SigInputElements;
  uint8_t SigOutputElements;
  uint8_t SigPatchOrPrimElements;

  // Number of packed vectors per signature
  uint8_t SigInputVectors;
  uint8_t SigOutputVectors[4];

  void swapBytes() {
    // nothing to swap since everything is single-byte or a union field
  }

  void swapBytes(Triple::EnvironmentType Stage) {
    v0::RuntimeInfo::swapBytes(Stage);
    if (Stage == Triple::EnvironmentType::Geometry)
      sys::swapByteOrder(GeomData.MaxVertexCount);
  }
};

} // namespace v1

namespace v2 {
struct RuntimeInfo : public v1::RuntimeInfo {
  uint32_t NumThreadsX;
  uint32_t NumThreadsY;
  uint32_t NumThreadsZ;

  void swapBytes() {
    sys::swapByteOrder(NumThreadsX);
    sys::swapByteOrder(NumThreadsY);
    sys::swapByteOrder(NumThreadsZ);
  }

  void swapBytes(Triple::EnvironmentType Stage) {
    v1::RuntimeInfo::swapBytes(Stage);
  }
};

struct ResourceBindInfo : public v0::ResourceBindInfo {
  ResourceKind Kind;
  ResourceFlags Flags;

  void swapBytes() {
    v0::ResourceBindInfo::swapBytes();
    sys::swapByteOrder(Kind);
    sys::swapByteOrder(Flags.Flags);
  }
};

} // namespace v2

namespace v3 {
struct RuntimeInfo : public v2::RuntimeInfo {
  uint32_t EntryNameOffset;

  void swapBytes() {
    v2::RuntimeInfo::swapBytes();
    sys::swapByteOrder(EntryNameOffset);
  }

  void swapBytes(Triple::EnvironmentType Stage) {
    v2::RuntimeInfo::swapBytes(Stage);
  }
};

} // namespace v3
} // namespace PSV

#define COMPONENT_PRECISION(Val, Enum) Enum = Val,
enum class SigMinPrecision : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<SigMinPrecision>> getSigMinPrecisions();

#define D3D_SYSTEM_VALUE(Val, Enum) Enum = Val,
enum class D3DSystemValue : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<D3DSystemValue>> getD3DSystemValues();

#define COMPONENT_TYPE(Val, Enum) Enum = Val,
enum class SigComponentType : uint32_t {
#include "DXContainerConstants.def"
};

LLVM_ABI ArrayRef<EnumEntry<SigComponentType>> getSigComponentTypes();

struct ProgramSignatureHeader {
  uint32_t ParamCount;
  uint32_t FirstParamOffset;

  void swapBytes() {
    sys::swapByteOrder(ParamCount);
    sys::swapByteOrder(FirstParamOffset);
  }
};

struct ProgramSignatureElement {
  uint32_t Stream;     // Stream index (parameters must appear in non-decreasing
                       // stream order)
  uint32_t NameOffset; // Offset from the start of the ProgramSignatureHeader to
                       // the start of the null terminated string for the name.
  uint32_t Index;      // Semantic Index
  D3DSystemValue SystemValue; // Semantic type. Similar to PSV::SemanticKind.
  SigComponentType CompType;  // Type of bits.
  uint32_t Register;          // Register Index (row index)
  uint8_t Mask;               // Mask (column allocation)

  // The ExclusiveMask has a different meaning for input and output signatures.
  // For an output signature, masked components of the output register are never
  // written to.
  // For an input signature, masked components of the input register are always
  // read.
  uint8_t ExclusiveMask;

  uint16_t Unused;
  SigMinPrecision MinPrecision; // Minimum precision of input/output data

  void swapBytes() {
    sys::swapByteOrder(Stream);
    sys::swapByteOrder(NameOffset);
    sys::swapByteOrder(Index);
    sys::swapByteOrder(SystemValue);
    sys::swapByteOrder(CompType);
    sys::swapByteOrder(Register);
    sys::swapByteOrder(Mask);
    sys::swapByteOrder(ExclusiveMask);
    sys::swapByteOrder(MinPrecision);
  }
};

static_assert(sizeof(ProgramSignatureElement) == 32,
              "ProgramSignatureElement is misaligned");

namespace RTS0 {
namespace v1 {
struct StaticSampler {
  uint32_t Filter;
  uint32_t AddressU;
  uint32_t AddressV;
  uint32_t AddressW;
  float MipLODBias;
  uint32_t MaxAnisotropy;
  uint32_t ComparisonFunc;
  uint32_t BorderColor;
  float MinLOD;
  float MaxLOD;
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  uint32_t ShaderVisibility;
  void swapBytes() {
    sys::swapByteOrder(Filter);
    sys::swapByteOrder(AddressU);
    sys::swapByteOrder(AddressV);
    sys::swapByteOrder(AddressW);
    sys::swapByteOrder(MipLODBias);
    sys::swapByteOrder(MaxAnisotropy);
    sys::swapByteOrder(ComparisonFunc);
    sys::swapByteOrder(BorderColor);
    sys::swapByteOrder(MinLOD);
    sys::swapByteOrder(MaxLOD);
    sys::swapByteOrder(ShaderRegister);
    sys::swapByteOrder(RegisterSpace);
    sys::swapByteOrder(ShaderVisibility);
  };
};

struct DescriptorRange {
  uint32_t RangeType;
  uint32_t NumDescriptors;
  uint32_t BaseShaderRegister;
  uint32_t RegisterSpace;
  uint32_t OffsetInDescriptorsFromTableStart;
  void swapBytes() {
    sys::swapByteOrder(RangeType);
    sys::swapByteOrder(NumDescriptors);
    sys::swapByteOrder(BaseShaderRegister);
    sys::swapByteOrder(RegisterSpace);
    sys::swapByteOrder(OffsetInDescriptorsFromTableStart);
  }
};

struct RootDescriptor {
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  void swapBytes() {
    sys::swapByteOrder(ShaderRegister);
    sys::swapByteOrder(RegisterSpace);
  }
};

// following dx12 naming
// https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_root_constants
struct RootConstants {
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  uint32_t Num32BitValues;

  void swapBytes() {
    sys::swapByteOrder(ShaderRegister);
    sys::swapByteOrder(RegisterSpace);
    sys::swapByteOrder(Num32BitValues);
  }
};

struct RootParameterHeader {
  uint32_t ParameterType;
  uint32_t ShaderVisibility;
  uint32_t ParameterOffset;

  void swapBytes() {
    sys::swapByteOrder(ParameterType);
    sys::swapByteOrder(ShaderVisibility);
    sys::swapByteOrder(ParameterOffset);
  }
};

struct RootSignatureHeader {
  uint32_t Version;
  uint32_t NumParameters;
  uint32_t ParametersOffset;
  uint32_t NumStaticSamplers;
  uint32_t StaticSamplerOffset;
  uint32_t Flags;

  void swapBytes() {
    sys::swapByteOrder(Version);
    sys::swapByteOrder(NumParameters);
    sys::swapByteOrder(ParametersOffset);
    sys::swapByteOrder(NumStaticSamplers);
    sys::swapByteOrder(StaticSamplerOffset);
    sys::swapByteOrder(Flags);
  }
};
} // namespace v1

namespace v2 {
struct RootDescriptor : public v1::RootDescriptor {
  uint32_t Flags;

  RootDescriptor() = default;
  explicit RootDescriptor(v1::RootDescriptor &Base)
      : v1::RootDescriptor(Base), Flags(0u) {}

  void swapBytes() {
    v1::RootDescriptor::swapBytes();
    sys::swapByteOrder(Flags);
  }
};

struct DescriptorRange {
  uint32_t RangeType;
  uint32_t NumDescriptors;
  uint32_t BaseShaderRegister;
  uint32_t RegisterSpace;
  uint32_t Flags;
  uint32_t OffsetInDescriptorsFromTableStart;
  void swapBytes() {
    sys::swapByteOrder(RangeType);
    sys::swapByteOrder(NumDescriptors);
    sys::swapByteOrder(BaseShaderRegister);
    sys::swapByteOrder(RegisterSpace);
    sys::swapByteOrder(OffsetInDescriptorsFromTableStart);
    sys::swapByteOrder(Flags);
  }
};
} // namespace v2
} // namespace RTS0

// D3D_ROOT_SIGNATURE_VERSION
enum class RootSignatureVersion {
  V1_0 = 0x1,
  V1_1 = 0x2,
};

} // namespace dxbc
} // namespace llvm

#endif // LLVM_BINARYFORMAT_DXCONTAINER_H
