//===- llvm/MC/DXContainerPSVInfo.h - DXContainer PSVInfo -*- C++ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_DXCONTAINERPSVINFO_H
#define LLVM_MC_DXCONTAINERPSVINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/TargetParser/Triple.h"

#include <array>
#include <numeric>
#include <stdint.h>

namespace llvm {

class raw_ostream;

namespace mcdxbc {

struct PSVSignatureElement {
  StringRef Name;
  SmallVector<uint32_t> Indices;
  uint8_t StartRow;
  uint8_t Cols;
  uint8_t StartCol;
  bool Allocated;
  dxbc::PSV::SemanticKind Kind;
  dxbc::PSV::ComponentType Type;
  dxbc::PSV::InterpolationMode Mode;
  uint8_t DynamicMask;
  uint8_t Stream;
};

// This data structure is a helper for reading and writing PSV RuntimeInfo data.
// It is implemented in the BinaryFormat library so that it can be used by both
// the MC layer and Object tools.
// This structure is used to represent the extracted data in an inspectable and
// modifiable format, and can be used to serialize the data back into valid PSV
// RuntimeInfo.
struct PSVRuntimeInfo {
  PSVRuntimeInfo() : DXConStrTabBuilder(StringTableBuilder::DXContainer) {}
  bool IsFinalized = false;
  dxbc::PSV::v3::RuntimeInfo BaseData;
  SmallVector<dxbc::PSV::v2::ResourceBindInfo> Resources;
  SmallVector<PSVSignatureElement> InputElements;
  SmallVector<PSVSignatureElement> OutputElements;
  SmallVector<PSVSignatureElement> PatchOrPrimElements;

  // TODO: Make this interface user-friendly.
  // The interface here is bad, and we'll want to change this in the future. We
  // probably will want to build out these mask vectors as vectors of bools and
  // have this utility object convert them to the bit masks. I don't want to
  // over-engineer this API now since we don't know what the data coming in to
  // feed it will look like, so I kept it extremely simple for the immediate use
  // case.
  std::array<SmallVector<uint32_t>, 4> OutputVectorMasks;
  SmallVector<uint32_t> PatchOrPrimMasks;
  std::array<SmallVector<uint32_t>, 4> InputOutputMap;
  SmallVector<uint32_t> InputPatchMap;
  SmallVector<uint32_t> PatchOutputMap;

  StringTableBuilder DXConStrTabBuilder;
  SmallVector<uint32_t, 64> IndexBuffer;
  SmallVector<llvm::dxbc::PSV::v0::SignatureElement, 32> SignatureElements;
  SmallVector<StringRef, 32> SemanticNames;

  llvm::StringRef EntryFunctionName;

  // Serialize PSVInfo into the provided raw_ostream. The version field
  // specifies the data version to encode, the default value specifies encoding
  // the highest supported version.
  void write(raw_ostream &OS,
             uint32_t Version = std::numeric_limits<uint32_t>::max()) const;

  static constexpr size_t npos = StringRef::npos;
  static size_t FindSequence(ArrayRef<uint32_t> Buffer,
                             ArrayRef<uint32_t> Sequence) {
    if (Buffer.size() < Sequence.size())
      return npos;
    for (size_t Idx = 0; Idx <= Buffer.size() - Sequence.size(); ++Idx) {
      if (0 == memcmp(static_cast<const void *>(&Buffer[Idx]),
                      static_cast<const void *>(Sequence.begin()),
                      Sequence.size() * sizeof(uint32_t)))
        return Idx;
    }
    return npos;
  }

  static void ProcessElementList(
      StringTableBuilder &StrTabBuilder, SmallVectorImpl<uint32_t> &IndexBuffer,
      SmallVectorImpl<llvm::dxbc::PSV::v0::SignatureElement> &FinalElements,
      SmallVectorImpl<StringRef> &SemanticNames,
      ArrayRef<PSVSignatureElement> Elements) {
    for (const auto &El : Elements) {
      // Put the name in the string table and the name list.
      StrTabBuilder.add(El.Name);
      SemanticNames.push_back(El.Name);

      llvm::dxbc::PSV::v0::SignatureElement FinalElement;
      memset(&FinalElement, 0, sizeof(llvm::dxbc::PSV::v0::SignatureElement));
      FinalElement.Rows = static_cast<uint8_t>(El.Indices.size());
      FinalElement.StartRow = El.StartRow;
      FinalElement.Cols = El.Cols;
      FinalElement.StartCol = El.StartCol;
      FinalElement.Allocated = El.Allocated;
      FinalElement.Kind = El.Kind;
      FinalElement.Type = El.Type;
      FinalElement.Mode = El.Mode;
      FinalElement.DynamicMask = El.DynamicMask;
      FinalElement.Stream = El.Stream;

      size_t Idx = FindSequence(IndexBuffer, El.Indices);
      if (Idx == npos) {
        FinalElement.IndicesOffset = static_cast<uint32_t>(IndexBuffer.size());
        IndexBuffer.insert(IndexBuffer.end(), El.Indices.begin(),
                           El.Indices.end());
      } else
        FinalElement.IndicesOffset = static_cast<uint32_t>(Idx);
      FinalElements.push_back(FinalElement);
    }
  }

  void finalize(Triple::EnvironmentType Stage) {
    IsFinalized = true;
    BaseData.SigInputElements = static_cast<uint32_t>(InputElements.size());
    BaseData.SigOutputElements = static_cast<uint32_t>(OutputElements.size());
    BaseData.SigPatchOrPrimElements =
        static_cast<uint32_t>(PatchOrPrimElements.size());

    // Build a string table and set associated offsets to be written when
    // write() is called
    ProcessElementList(DXConStrTabBuilder, IndexBuffer, SignatureElements,
                       SemanticNames, InputElements);
    ProcessElementList(DXConStrTabBuilder, IndexBuffer, SignatureElements,
                       SemanticNames, OutputElements);
    ProcessElementList(DXConStrTabBuilder, IndexBuffer, SignatureElements,
                       SemanticNames, PatchOrPrimElements);

    DXConStrTabBuilder.add(EntryFunctionName);

    DXConStrTabBuilder.finalize();
    for (auto ElAndName : zip(SignatureElements, SemanticNames)) {
      llvm::dxbc::PSV::v0::SignatureElement &El = std::get<0>(ElAndName);
      StringRef Name = std::get<1>(ElAndName);
      El.NameOffset = static_cast<uint32_t>(DXConStrTabBuilder.getOffset(Name));
      if (sys::IsBigEndianHost)
        El.swapBytes();
    }

    BaseData.EntryFunctionName =
        static_cast<uint32_t>(DXConStrTabBuilder.getOffset(EntryFunctionName));

    if (!sys::IsBigEndianHost)
      return;
    BaseData.swapBytes();
    BaseData.swapBytes(Stage);
    for (auto &Res : Resources)
      Res.swapBytes();
  }
};

class Signature {
  struct Parameter {
    uint32_t Stream;
    StringRef Name;
    uint32_t Index;
    dxbc::D3DSystemValue SystemValue;
    dxbc::SigComponentType CompType;
    uint32_t Register;
    uint8_t Mask;
    uint8_t ExclusiveMask;
    dxbc::SigMinPrecision MinPrecision;
  };

  SmallVector<Parameter> Params;

public:
  void addParam(uint32_t Stream, StringRef Name, uint32_t Index,
                dxbc::D3DSystemValue SystemValue,
                dxbc::SigComponentType CompType, uint32_t Register,
                uint8_t Mask, uint8_t ExclusiveMask,
                dxbc::SigMinPrecision MinPrecision) {
    Params.push_back(Parameter{Stream, Name, Index, SystemValue, CompType,
                               Register, Mask, ExclusiveMask, MinPrecision});
  }

  void write(raw_ostream &OS);
};

} // namespace mcdxbc
} // namespace llvm

#endif // LLVM_MC_DXCONTAINERPSVINFO_H
