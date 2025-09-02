//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;
using namespace llvm::mcdxbc;

static uint32_t writePlaceholder(raw_svector_ostream &Stream) {
  const uint32_t DummyValue = std::numeric_limits<uint32_t>::max();
  uint32_t Offset = Stream.tell();
  support::endian::write(Stream, DummyValue, llvm::endianness::little);
  return Offset;
}

static uint32_t rewriteOffsetToCurrentByte(raw_svector_ostream &Stream,
                                           uint32_t Offset) {
  uint32_t ByteOffset = Stream.tell();
  uint32_t Value =
      support::endian::byte_swap<uint32_t, llvm::endianness::little>(
          ByteOffset);
  Stream.pwrite(reinterpret_cast<const char *>(&Value), sizeof(Value), Offset);
  return ByteOffset;
}

size_t RootSignatureDesc::getSize() const {
  uint32_t StaticSamplersOffset = computeStaticSamplersOffset();
  size_t StaticSamplersSize =
      StaticSamplers.size() * sizeof(dxbc::RTS0::v1::StaticSampler);

  return size_t(StaticSamplersOffset) + StaticSamplersSize;
}

uint32_t RootSignatureDesc::computeRootParametersOffset() const {
  return sizeof(dxbc::RTS0::v1::RootSignatureHeader);
}

uint32_t RootSignatureDesc::computeStaticSamplersOffset() const {
  uint32_t Offset = computeRootParametersOffset();

  for (const RootParameterInfo &I : ParametersContainer) {
    Offset += sizeof(dxbc::RTS0::v1::RootParameterHeader);
    switch (I.Type) {
    case dxbc::RootParameterType::Constants32Bit:
      Offset += sizeof(dxbc::RTS0::v1::RootConstants);
      break;
    case dxbc::RootParameterType::CBV:
    case dxbc::RootParameterType::SRV:
    case dxbc::RootParameterType::UAV:
      if (Version == 1)
        Offset += sizeof(dxbc::RTS0::v1::RootDescriptor);
      else
        Offset += sizeof(dxbc::RTS0::v2::RootDescriptor);

      break;
    case dxbc::RootParameterType::DescriptorTable:
      const DescriptorTable &Table =
          ParametersContainer.getDescriptorTable(I.Location);

      // 4 bytes for the number of ranges in table and
      // 4 bytes for the ranges offset
      Offset += 2 * sizeof(uint32_t);
      if (Version == 1)
        Offset += sizeof(dxbc::RTS0::v1::DescriptorRange) * Table.Ranges.size();
      else
        Offset += sizeof(dxbc::RTS0::v2::DescriptorRange) * Table.Ranges.size();
      break;
    }
  }

  return Offset;
}

void RootSignatureDesc::write(raw_ostream &OS) const {
  SmallString<256> Storage;
  raw_svector_ostream BOS(Storage);
  BOS.reserveExtraSpace(getSize());

  const uint32_t NumParameters = ParametersContainer.size();
  const uint32_t NumSamplers = StaticSamplers.size();
  support::endian::write(BOS, Version, llvm::endianness::little);
  support::endian::write(BOS, NumParameters, llvm::endianness::little);
  support::endian::write(BOS, RootParameterOffset, llvm::endianness::little);
  support::endian::write(BOS, NumSamplers, llvm::endianness::little);
  uint32_t SSO = writePlaceholder(BOS);
  support::endian::write(BOS, Flags, llvm::endianness::little);

  SmallVector<uint32_t> ParamsOffsets;
  for (const RootParameterInfo &I : ParametersContainer) {
    support::endian::write(BOS, I.Type, llvm::endianness::little);
    support::endian::write(BOS, I.Visibility, llvm::endianness::little);

    ParamsOffsets.push_back(writePlaceholder(BOS));
  }

  assert(NumParameters == ParamsOffsets.size());
  for (size_t I = 0; I < NumParameters; ++I) {
    rewriteOffsetToCurrentByte(BOS, ParamsOffsets[I]);
    const RootParameterInfo &Info = ParametersContainer.getInfo(I);
    switch (Info.Type) {
    case dxbc::RootParameterType::Constants32Bit: {
      const mcdxbc::RootConstants &Constants =
          ParametersContainer.getConstant(Info.Location);
      support::endian::write(BOS, Constants.ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(BOS, Constants.RegisterSpace,
                             llvm::endianness::little);
      support::endian::write(BOS, Constants.Num32BitValues,
                             llvm::endianness::little);
      break;
    }
    case dxbc::RootParameterType::CBV:
    case dxbc::RootParameterType::SRV:
    case dxbc::RootParameterType::UAV: {
      const mcdxbc::RootDescriptor &Descriptor =
          ParametersContainer.getRootDescriptor(Info.Location);

      support::endian::write(BOS, Descriptor.ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(BOS, Descriptor.RegisterSpace,
                             llvm::endianness::little);
      if (Version > 1)
        support::endian::write(BOS, Descriptor.Flags, llvm::endianness::little);
      break;
    }
    case dxbc::RootParameterType::DescriptorTable: {
      const DescriptorTable &Table =
          ParametersContainer.getDescriptorTable(Info.Location);
      support::endian::write(BOS, (uint32_t)Table.Ranges.size(),
                             llvm::endianness::little);
      rewriteOffsetToCurrentByte(BOS, writePlaceholder(BOS));
      for (const auto &Range : Table) {
        support::endian::write(BOS, Range.RangeType, llvm::endianness::little);
        support::endian::write(BOS, Range.NumDescriptors,
                               llvm::endianness::little);
        support::endian::write(BOS, Range.BaseShaderRegister,
                               llvm::endianness::little);
        support::endian::write(BOS, Range.RegisterSpace,
                               llvm::endianness::little);
        if (Version > 1)
          support::endian::write(BOS, Range.Flags, llvm::endianness::little);
        support::endian::write(BOS, Range.OffsetInDescriptorsFromTableStart,
                               llvm::endianness::little);
      }
      break;
    }
    }
  }
  [[maybe_unused]] uint32_t Offset = rewriteOffsetToCurrentByte(BOS, SSO);
  assert(Offset == computeStaticSamplersOffset() &&
         "Computed offset does not match written offset");
  for (const auto &S : StaticSamplers) {
    support::endian::write(BOS, S.Filter, llvm::endianness::little);
    support::endian::write(BOS, S.AddressU, llvm::endianness::little);
    support::endian::write(BOS, S.AddressV, llvm::endianness::little);
    support::endian::write(BOS, S.AddressW, llvm::endianness::little);
    support::endian::write(BOS, S.MipLODBias, llvm::endianness::little);
    support::endian::write(BOS, S.MaxAnisotropy, llvm::endianness::little);
    support::endian::write(BOS, S.ComparisonFunc, llvm::endianness::little);
    support::endian::write(BOS, S.BorderColor, llvm::endianness::little);
    support::endian::write(BOS, S.MinLOD, llvm::endianness::little);
    support::endian::write(BOS, S.MaxLOD, llvm::endianness::little);
    support::endian::write(BOS, S.ShaderRegister, llvm::endianness::little);
    support::endian::write(BOS, S.RegisterSpace, llvm::endianness::little);
    support::endian::write(BOS, S.ShaderVisibility, llvm::endianness::little);
  }
  assert(Storage.size() == getSize());
  OS.write(Storage.data(), Storage.size());
}
