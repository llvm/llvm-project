//===- DXContainer.h - DXContainer file implementation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the DXContainerFile class, which implements the ObjectFile
// interface for DXContainer files.
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_DXCONTAINER_H
#define LLVM_OBJECT_DXCONTAINER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TargetParser/Triple.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace llvm {
namespace object {

namespace detail {
template <typename T>
std::enable_if_t<std::is_arithmetic<T>::value, void> swapBytes(T &value) {
  sys::swapByteOrder(value);
}

template <typename T>
std::enable_if_t<std::is_class<T>::value, void> swapBytes(T &value) {
  value.swapBytes();
}
} // namespace detail

// This class provides a view into the underlying resource array. The Resource
// data is little-endian encoded and may not be properly aligned to read
// directly from. The dereference operator creates a copy of the data and byte
// swaps it as appropriate.
template <typename T> struct ViewArray {
  StringRef Data;
  uint32_t Stride = sizeof(T); // size of each element in the list.

  ViewArray() = default;
  ViewArray(StringRef D, size_t S) : Data(D), Stride(S) {}

  using value_type = T;
  static constexpr uint32_t MaxStride() {
    return static_cast<uint32_t>(sizeof(value_type));
  }

  struct iterator {
    StringRef Data;
    uint32_t Stride; // size of each element in the list.
    const char *Current;

    iterator(const ViewArray &A, const char *C)
        : Data(A.Data), Stride(A.Stride), Current(C) {}
    iterator(const iterator &) = default;

    value_type operator*() {
      // Explicitly zero the structure so that unused fields are zeroed. It is
      // up to the user to know if the fields are used by verifying the PSV
      // version.
      value_type Val;
      std::memset(&Val, 0, sizeof(value_type));
      if (Current >= Data.end())
        return Val;
      memcpy(static_cast<void *>(&Val), Current, std::min(Stride, MaxStride()));
      if (sys::IsBigEndianHost)
        detail::swapBytes(Val);
      return Val;
    }

    iterator operator++() {
      if (Current < Data.end())
        Current += Stride;
      return *this;
    }

    iterator operator++(int) {
      iterator Tmp = *this;
      ++*this;
      return Tmp;
    }

    iterator operator--() {
      if (Current > Data.begin())
        Current -= Stride;
      return *this;
    }

    iterator operator--(int) {
      iterator Tmp = *this;
      --*this;
      return Tmp;
    }

    bool operator==(const iterator I) { return I.Current == Current; }
    bool operator!=(const iterator I) { return !(*this == I); }
  };

  iterator begin() const { return iterator(*this, Data.begin()); }

  iterator end() const { return iterator(*this, Data.end()); }

  size_t size() const { return Data.size() / Stride; }

  bool isEmpty() const { return Data.empty(); }
};

namespace DirectX {
struct RootParameterView {
  const dxbc::RTS0::v1::RootParameterHeader &Header;
  StringRef ParamData;

  RootParameterView(const dxbc::RTS0::v1::RootParameterHeader &H, StringRef P)
      : Header(H), ParamData(P) {}

  template <typename T> Expected<T> readParameter() {
    T Struct;
    if (sizeof(T) != ParamData.size())
      return make_error<GenericBinaryError>(
          "Reading structure out of file bounds", object_error::parse_failed);

    memcpy(&Struct, ParamData.data(), sizeof(T));
    // DXContainer is always little endian
    if (sys::IsBigEndianHost)
      Struct.swapBytes();
    return Struct;
  }
};

struct RootConstantView : RootParameterView {
  static bool classof(const RootParameterView *V) {
    return V->Header.ParameterType ==
           (uint32_t)dxbc::RootParameterType::Constants32Bit;
  }

  llvm::Expected<dxbc::RTS0::v1::RootConstants> read() {
    return readParameter<dxbc::RTS0::v1::RootConstants>();
  }
};

struct RootDescriptorView : RootParameterView {
  static bool classof(const RootParameterView *V) {
    return (V->Header.ParameterType ==
                llvm::to_underlying(dxbc::RootParameterType::CBV) ||
            V->Header.ParameterType ==
                llvm::to_underlying(dxbc::RootParameterType::SRV) ||
            V->Header.ParameterType ==
                llvm::to_underlying(dxbc::RootParameterType::UAV));
  }

  llvm::Expected<dxbc::RTS0::v2::RootDescriptor> read(uint32_t Version) {
    if (Version == 1) {
      auto Descriptor = readParameter<dxbc::RTS0::v1::RootDescriptor>();
      if (Error E = Descriptor.takeError())
        return E;
      return dxbc::RTS0::v2::RootDescriptor(*Descriptor);
    }
    if (Version != 2)
      return make_error<GenericBinaryError>("Invalid Root Signature version: " +
                                                Twine(Version),
                                            object_error::parse_failed);
    return readParameter<dxbc::RTS0::v2::RootDescriptor>();
  }
};
template <typename T> struct DescriptorTable {
  uint32_t NumRanges;
  uint32_t RangesOffset;
  ViewArray<T> Ranges;

  typename ViewArray<T>::iterator begin() const { return Ranges.begin(); }

  typename ViewArray<T>::iterator end() const { return Ranges.end(); }
};

struct DescriptorTableView : RootParameterView {
  static bool classof(const RootParameterView *V) {
    return (V->Header.ParameterType ==
            llvm::to_underlying(dxbc::RootParameterType::DescriptorTable));
  }

  // Define a type alias to access the template parameter from inside classof
  template <typename T> llvm::Expected<DescriptorTable<T>> read() {
    const char *Current = ParamData.begin();
    DescriptorTable<T> Table;

    Table.NumRanges =
        support::endian::read<uint32_t, llvm::endianness::little>(Current);
    Current += sizeof(uint32_t);

    Table.RangesOffset =
        support::endian::read<uint32_t, llvm::endianness::little>(Current);
    Current += sizeof(uint32_t);

    Table.Ranges.Data = ParamData.substr(2 * sizeof(uint32_t),
                                         Table.NumRanges * Table.Ranges.Stride);
    return Table;
  }
};

static Error parseFailed(const Twine &Msg) {
  return make_error<GenericBinaryError>(Msg.str(), object_error::parse_failed);
}

class RootSignature {
private:
  uint32_t Version;
  uint32_t NumParameters;
  uint32_t RootParametersOffset;
  uint32_t NumStaticSamplers;
  uint32_t StaticSamplersOffset;
  uint32_t Flags;
  ViewArray<dxbc::RTS0::v1::RootParameterHeader> ParametersHeaders;
  StringRef PartData;
  ViewArray<dxbc::RTS0::v1::StaticSampler> StaticSamplers;

  using param_header_iterator =
      ViewArray<dxbc::RTS0::v1::RootParameterHeader>::iterator;
  using samplers_iterator = ViewArray<dxbc::RTS0::v1::StaticSampler>::iterator;

public:
  RootSignature(StringRef PD) : PartData(PD) {}

  LLVM_ABI Error parse();
  uint32_t getVersion() const { return Version; }
  uint32_t getNumParameters() const { return NumParameters; }
  uint32_t getRootParametersOffset() const { return RootParametersOffset; }
  uint32_t getNumStaticSamplers() const { return NumStaticSamplers; }
  uint32_t getStaticSamplersOffset() const { return StaticSamplersOffset; }
  uint32_t getNumRootParameters() const { return ParametersHeaders.size(); }
  llvm::iterator_range<param_header_iterator> param_headers() const {
    return llvm::make_range(ParametersHeaders.begin(), ParametersHeaders.end());
  }
  llvm::iterator_range<samplers_iterator> samplers() const {
    return llvm::make_range(StaticSamplers.begin(), StaticSamplers.end());
  }
  uint32_t getFlags() const { return Flags; }

  llvm::Expected<RootParameterView>
  getParameter(const dxbc::RTS0::v1::RootParameterHeader &Header) const {
    size_t DataSize;
    size_t EndOfSectionByte = getNumStaticSamplers() == 0
                                  ? PartData.size()
                                  : getStaticSamplersOffset();

    if (!dxbc::isValidParameterType(Header.ParameterType))
      return parseFailed("invalid parameter type");

    switch (static_cast<dxbc::RootParameterType>(Header.ParameterType)) {
    case dxbc::RootParameterType::Constants32Bit:
      DataSize = sizeof(dxbc::RTS0::v1::RootConstants);
      break;
    case dxbc::RootParameterType::CBV:
    case dxbc::RootParameterType::SRV:
    case dxbc::RootParameterType::UAV:
      if (Version == 1)
        DataSize = sizeof(dxbc::RTS0::v1::RootDescriptor);
      else
        DataSize = sizeof(dxbc::RTS0::v2::RootDescriptor);
      break;
    case dxbc::RootParameterType::DescriptorTable:
      if (Header.ParameterOffset + sizeof(uint32_t) > EndOfSectionByte)
        return parseFailed("Reading structure out of file bounds");

      uint32_t NumRanges =
          support::endian::read<uint32_t, llvm::endianness::little>(
              PartData.begin() + Header.ParameterOffset);
      if (Version == 1)
        DataSize = sizeof(dxbc::RTS0::v1::DescriptorRange) * NumRanges;
      else
        DataSize = sizeof(dxbc::RTS0::v2::DescriptorRange) * NumRanges;

      // 4 bytes for the number of ranges in table and
      // 4 bytes for the ranges offset
      DataSize += 2 * sizeof(uint32_t);
      break;
    }
    if (Header.ParameterOffset + DataSize > EndOfSectionByte)
      return parseFailed("Reading structure out of file bounds");

    StringRef Buff = PartData.substr(Header.ParameterOffset, DataSize);
    RootParameterView View = RootParameterView(Header, Buff);
    return View;
  }
};

class PSVRuntimeInfo {

  using ResourceArray = ViewArray<dxbc::PSV::v2::ResourceBindInfo>;
  using SigElementArray = ViewArray<dxbc::PSV::v0::SignatureElement>;

  StringRef Data;
  uint32_t Size;
  using InfoStruct =
      std::variant<std::monostate, dxbc::PSV::v0::RuntimeInfo,
                   dxbc::PSV::v1::RuntimeInfo, dxbc::PSV::v2::RuntimeInfo,
                   dxbc::PSV::v3::RuntimeInfo>;
  InfoStruct BasicInfo;
  ResourceArray Resources;
  StringRef StringTable;
  SmallVector<uint32_t> SemanticIndexTable;
  SigElementArray SigInputElements;
  SigElementArray SigOutputElements;
  SigElementArray SigPatchOrPrimElements;

  std::array<ViewArray<uint32_t>, 4> OutputVectorMasks;
  ViewArray<uint32_t> PatchOrPrimMasks;
  std::array<ViewArray<uint32_t>, 4> InputOutputMap;
  ViewArray<uint32_t> InputPatchMap;
  ViewArray<uint32_t> PatchOutputMap;

public:
  PSVRuntimeInfo(StringRef D) : Data(D), Size(0) {}

  // Parsing depends on the shader kind
  LLVM_ABI Error parse(uint16_t ShaderKind);

  uint32_t getSize() const { return Size; }
  uint32_t getResourceCount() const { return Resources.size(); }
  ResourceArray getResources() const { return Resources; }

  uint32_t getVersion() const {
    return Size >= sizeof(dxbc::PSV::v3::RuntimeInfo)
               ? 3
               : (Size >= sizeof(dxbc::PSV::v2::RuntimeInfo)     ? 2
                  : (Size >= sizeof(dxbc::PSV::v1::RuntimeInfo)) ? 1
                                                                 : 0);
  }

  uint32_t getResourceStride() const { return Resources.Stride; }

  const InfoStruct &getInfo() const { return BasicInfo; }

  template <typename T> const T *getInfoAs() const {
    if (const auto *P = std::get_if<dxbc::PSV::v3::RuntimeInfo>(&BasicInfo))
      return static_cast<const T *>(P);
    if (std::is_same<T, dxbc::PSV::v3::RuntimeInfo>::value)
      return nullptr;

    if (const auto *P = std::get_if<dxbc::PSV::v2::RuntimeInfo>(&BasicInfo))
      return static_cast<const T *>(P);
    if (std::is_same<T, dxbc::PSV::v2::RuntimeInfo>::value)
      return nullptr;

    if (const auto *P = std::get_if<dxbc::PSV::v1::RuntimeInfo>(&BasicInfo))
      return static_cast<const T *>(P);
    if (std::is_same<T, dxbc::PSV::v1::RuntimeInfo>::value)
      return nullptr;

    if (const auto *P = std::get_if<dxbc::PSV::v0::RuntimeInfo>(&BasicInfo))
      return static_cast<const T *>(P);
    return nullptr;
  }

  StringRef getStringTable() const { return StringTable; }
  ArrayRef<uint32_t> getSemanticIndexTable() const {
    return SemanticIndexTable;
  }

  LLVM_ABI uint8_t getSigInputCount() const;
  LLVM_ABI uint8_t getSigOutputCount() const;
  LLVM_ABI uint8_t getSigPatchOrPrimCount() const;

  SigElementArray getSigInputElements() const { return SigInputElements; }
  SigElementArray getSigOutputElements() const { return SigOutputElements; }
  SigElementArray getSigPatchOrPrimElements() const {
    return SigPatchOrPrimElements;
  }

  ViewArray<uint32_t> getOutputVectorMasks(size_t Idx) const {
    assert(Idx < 4);
    return OutputVectorMasks[Idx];
  }

  ViewArray<uint32_t> getPatchOrPrimMasks() const { return PatchOrPrimMasks; }

  ViewArray<uint32_t> getInputOutputMap(size_t Idx) const {
    assert(Idx < 4);
    return InputOutputMap[Idx];
  }

  ViewArray<uint32_t> getInputPatchMap() const { return InputPatchMap; }
  ViewArray<uint32_t> getPatchOutputMap() const { return PatchOutputMap; }

  uint32_t getSigElementStride() const { return SigInputElements.Stride; }

  bool usesViewID() const {
    if (const auto *P = getInfoAs<dxbc::PSV::v1::RuntimeInfo>())
      return P->UsesViewID != 0;
    return false;
  }

  uint8_t getInputVectorCount() const {
    if (const auto *P = getInfoAs<dxbc::PSV::v1::RuntimeInfo>())
      return P->SigInputVectors;
    return 0;
  }

  ArrayRef<uint8_t> getOutputVectorCounts() const {
    if (const auto *P = getInfoAs<dxbc::PSV::v1::RuntimeInfo>())
      return ArrayRef<uint8_t>(P->SigOutputVectors);
    return ArrayRef<uint8_t>();
  }

  uint8_t getPatchConstOrPrimVectorCount() const {
    if (const auto *P = getInfoAs<dxbc::PSV::v1::RuntimeInfo>())
      return P->GeomData.SigPatchConstOrPrimVectors;
    return 0;
  }
};

class Signature {
  ViewArray<dxbc::ProgramSignatureElement> Parameters;
  uint32_t StringTableOffset;
  StringRef StringTable;

public:
  ViewArray<dxbc::ProgramSignatureElement>::iterator begin() const {
    return Parameters.begin();
  }

  ViewArray<dxbc::ProgramSignatureElement>::iterator end() const {
    return Parameters.end();
  }

  StringRef getName(uint32_t Offset) const {
    assert(Offset >= StringTableOffset &&
           Offset < StringTableOffset + StringTable.size() &&
           "Offset out of range.");
    // Name offsets are from the start of the signature data, not from the start
    // of the string table. The header encodes the start offset of the sting
    // table, so we convert the offset here.
    uint32_t TableOffset = Offset - StringTableOffset;
    return StringTable.slice(TableOffset, StringTable.find('\0', TableOffset));
  }

  bool isEmpty() const { return Parameters.isEmpty(); }

  LLVM_ABI Error initialize(StringRef Part);
};

} // namespace DirectX

class DXContainer {
public:
  using DXILData = std::pair<dxbc::ProgramHeader, const char *>;

private:
  DXContainer(MemoryBufferRef O);

  MemoryBufferRef Data;
  dxbc::Header Header;
  SmallVector<uint32_t, 4> PartOffsets;
  std::optional<DXILData> DXIL;
  std::optional<uint64_t> ShaderFeatureFlags;
  std::optional<dxbc::ShaderHash> Hash;
  std::optional<DirectX::PSVRuntimeInfo> PSVInfo;
  std::optional<DirectX::RootSignature> RootSignature;
  DirectX::Signature InputSignature;
  DirectX::Signature OutputSignature;
  DirectX::Signature PatchConstantSignature;

  Error parseHeader();
  Error parsePartOffsets();
  Error parseDXILHeader(StringRef Part);
  Error parseShaderFeatureFlags(StringRef Part);
  Error parseHash(StringRef Part);
  Error parseRootSignature(StringRef Part);
  Error parsePSVInfo(StringRef Part);
  Error parseSignature(StringRef Part, DirectX::Signature &Array);
  friend class PartIterator;

public:
  // The PartIterator is a wrapper around the iterator for the PartOffsets
  // member of the DXContainer. It contains a refernce to the container, and the
  // current iterator value, as well as storage for a parsed part header.
  class PartIterator {
    const DXContainer &Container;
    SmallVectorImpl<uint32_t>::const_iterator OffsetIt;
    struct PartData {
      dxbc::PartHeader Part;
      uint32_t Offset;
      StringRef Data;
    } IteratorState;

    friend class DXContainer;
    friend class DXContainerObjectFile;

    PartIterator(const DXContainer &C,
                 SmallVectorImpl<uint32_t>::const_iterator It)
        : Container(C), OffsetIt(It) {
      if (OffsetIt == Container.PartOffsets.end())
        updateIteratorImpl(Container.PartOffsets.back());
      else
        updateIterator();
    }

    // Updates the iterator's state data. This results in copying the part
    // header into the iterator and handling any required byte swapping. This is
    // called when incrementing or decrementing the iterator.
    void updateIterator() {
      if (OffsetIt != Container.PartOffsets.end())
        updateIteratorImpl(*OffsetIt);
    }

    // Implementation for updating the iterator state based on a specified
    // offest.
    LLVM_ABI void updateIteratorImpl(const uint32_t Offset);

  public:
    PartIterator &operator++() {
      if (OffsetIt == Container.PartOffsets.end())
        return *this;
      ++OffsetIt;
      updateIterator();
      return *this;
    }

    PartIterator operator++(int) {
      PartIterator Tmp = *this;
      ++(*this);
      return Tmp;
    }

    bool operator==(const PartIterator &RHS) const {
      return OffsetIt == RHS.OffsetIt;
    }

    bool operator!=(const PartIterator &RHS) const {
      return OffsetIt != RHS.OffsetIt;
    }

    const PartData &operator*() { return IteratorState; }
    const PartData *operator->() { return &IteratorState; }
  };

  PartIterator begin() const {
    return PartIterator(*this, PartOffsets.begin());
  }

  PartIterator end() const { return PartIterator(*this, PartOffsets.end()); }

  StringRef getData() const { return Data.getBuffer(); }
  LLVM_ABI static Expected<DXContainer> create(MemoryBufferRef Object);

  const dxbc::Header &getHeader() const { return Header; }

  const std::optional<DXILData> &getDXIL() const { return DXIL; }

  std::optional<uint64_t> getShaderFeatureFlags() const {
    return ShaderFeatureFlags;
  }

  std::optional<dxbc::ShaderHash> getShaderHash() const { return Hash; }

  std::optional<DirectX::RootSignature> getRootSignature() const {
    return RootSignature;
  }

  const std::optional<DirectX::PSVRuntimeInfo> &getPSVInfo() const {
    return PSVInfo;
  };

  const DirectX::Signature &getInputSignature() const { return InputSignature; }
  const DirectX::Signature &getOutputSignature() const {
    return OutputSignature;
  }
  const DirectX::Signature &getPatchConstantSignature() const {
    return PatchConstantSignature;
  }
};

class LLVM_ABI DXContainerObjectFile : public ObjectFile {
private:
  friend class ObjectFile;
  DXContainer Container;

  using PartData = DXContainer::PartIterator::PartData;
  llvm::SmallVector<PartData> Parts;
  using PartIterator = llvm::SmallVector<PartData>::iterator;

  DXContainerObjectFile(DXContainer C)
      : ObjectFile(ID_DXContainer, MemoryBufferRef(C.getData(), "")),
        Container(C) {
    for (auto &P : C)
      Parts.push_back(P);
  }

public:
  const DXContainer &getDXContainer() const { return Container; }

  static bool classof(const Binary *v) { return v->isDXContainer(); }

  const dxbc::Header &getHeader() const { return Container.getHeader(); }

  Expected<StringRef> getSymbolName(DataRefImpl) const override;
  Expected<uint64_t> getSymbolAddress(DataRefImpl Symb) const override;
  uint64_t getSymbolValueImpl(DataRefImpl Symb) const override;
  uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const override;

  Expected<SymbolRef::Type> getSymbolType(DataRefImpl Symb) const override;
  Expected<section_iterator> getSymbolSection(DataRefImpl Symb) const override;
  void moveSectionNext(DataRefImpl &Sec) const override;
  Expected<StringRef> getSectionName(DataRefImpl Sec) const override;
  uint64_t getSectionAddress(DataRefImpl Sec) const override;
  uint64_t getSectionIndex(DataRefImpl Sec) const override;
  uint64_t getSectionSize(DataRefImpl Sec) const override;
  Expected<ArrayRef<uint8_t>>
  getSectionContents(DataRefImpl Sec) const override;

  uint64_t getSectionAlignment(DataRefImpl Sec) const override;
  bool isSectionCompressed(DataRefImpl Sec) const override;
  bool isSectionText(DataRefImpl Sec) const override;
  bool isSectionData(DataRefImpl Sec) const override;
  bool isSectionBSS(DataRefImpl Sec) const override;
  bool isSectionVirtual(DataRefImpl Sec) const override;

  relocation_iterator section_rel_begin(DataRefImpl Sec) const override;
  relocation_iterator section_rel_end(DataRefImpl Sec) const override;

  void moveRelocationNext(DataRefImpl &Rel) const override;
  uint64_t getRelocationOffset(DataRefImpl Rel) const override;
  symbol_iterator getRelocationSymbol(DataRefImpl Rel) const override;
  uint64_t getRelocationType(DataRefImpl Rel) const override;
  void getRelocationTypeName(DataRefImpl Rel,
                             SmallVectorImpl<char> &Result) const override;

  section_iterator section_begin() const override;
  section_iterator section_end() const override;

  uint8_t getBytesInAddress() const override;
  StringRef getFileFormatName() const override;
  Triple::ArchType getArch() const override;
  Expected<SubtargetFeatures> getFeatures() const override;

  void moveSymbolNext(DataRefImpl &Symb) const override {}
  Error printSymbolName(raw_ostream &OS, DataRefImpl Symb) const override;
  Expected<uint32_t> getSymbolFlags(DataRefImpl Symb) const override;
  basic_symbol_iterator symbol_begin() const override {
    return basic_symbol_iterator(SymbolRef());
  }
  basic_symbol_iterator symbol_end() const override {
    return basic_symbol_iterator(SymbolRef());
  }
  bool is64Bit() const override { return false; }

  bool isRelocatableObject() const override { return false; }
};

} // namespace object
} // namespace llvm

#endif // LLVM_OBJECT_DXCONTAINER_H
