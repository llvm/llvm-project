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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/TargetParser/Triple.h"

#include <stdint.h>

namespace llvm {

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

  bool isPopulated();

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
  uint8_t MajorVersion; // DXIL version.
  uint8_t MinorVersion; // DXIL version.
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
  uint8_t MinorVersion : 4;
  uint8_t MajorVersion : 4;
  uint8_t Unused;
  uint16_t ShaderKind;
  uint32_t Size; // Size in uint32_t words including this header.
  BitcodeHeader Bitcode;

  void swapBytes() {
    sys::swapByteOrder(ShaderKind);
    sys::swapByteOrder(Size);
    Bitcode.swapBytes();
  }
};

static_assert(sizeof(ProgramHeader) == 24, "ProgramHeader Size incorrect!");

#define CONTAINER_PART(Part) Part,
enum class PartType {
  Unknown = 0,
#include "DXContainerConstants.def"
};

#define SHADER_FLAG(Num, Val, Str) Val = 1ull << Num,
enum class FeatureFlags : uint64_t {
#include "DXContainerConstants.def"
};
static_assert((uint64_t)FeatureFlags::NextUnusedBit <= 1ull << 63,
              "Shader flag bits exceed enum size.");

PartType parsePartType(StringRef S);

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
  uint8_t SigPatchConstOrPrimElements;

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

} // namespace v2
} // namespace PSV

} // namespace dxbc
} // namespace llvm

#endif // LLVM_BINARYFORMAT_DXCONTAINER_H
