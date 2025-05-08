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

static void rewriteOffsetToCurrentByte(raw_svector_ostream &Stream,
                                       uint32_t Offset) {
  uint32_t Value =
      support::endian::byte_swap<uint32_t, llvm::endianness::little>(
          Stream.tell());
  Stream.pwrite(reinterpret_cast<const char *>(&Value), sizeof(Value), Offset);
}

size_t RootSignatureDesc::getSize() const {
  size_t Size = sizeof(dxbc::RootSignatureHeader) +
                ParametersContainer.size() * sizeof(dxbc::RootParameterHeader);

  for (const auto &I : ParametersContainer) {
    std::optional<ParametersView> P = ParametersContainer.getParameter(&I);
    if (!P)
      continue;
    std::visit(
        [&Size](auto &Value) -> void {
          using T = std::decay_t<decltype(*Value)>;
          Size += sizeof(T);
        },
        *P);
  }

  return Size;
}

void RootSignatureDesc::write(raw_ostream &OS) const {
  SmallString<256> Storage;
  raw_svector_ostream BOS(Storage);
  BOS.reserveExtraSpace(getSize());

  const uint32_t NumParameters = ParametersContainer.size();

  support::endian::write(BOS, Version, llvm::endianness::little);
  support::endian::write(BOS, NumParameters, llvm::endianness::little);
  support::endian::write(BOS, RootParameterOffset, llvm::endianness::little);
  support::endian::write(BOS, NumStaticSamplers, llvm::endianness::little);
  support::endian::write(BOS, StaticSamplersOffset, llvm::endianness::little);
  support::endian::write(BOS, Flags, llvm::endianness::little);

  SmallVector<uint32_t> ParamsOffsets;
  for (const auto &P : ParametersContainer) {
    support::endian::write(BOS, P.Header.ParameterType,
                           llvm::endianness::little);
    support::endian::write(BOS, P.Header.ShaderVisibility,
                           llvm::endianness::little);

    ParamsOffsets.push_back(writePlaceholder(BOS));
  }

  assert(NumParameters == ParamsOffsets.size());
  const RootParameterInfo *H = ParametersContainer.begin();
  for (size_t I = 0; I < NumParameters; ++I, H++) {
    rewriteOffsetToCurrentByte(BOS, ParamsOffsets[I]);
    auto P = ParametersContainer.getParameter(H);
    if (!P)
      continue;
    if (std::holds_alternative<const dxbc::RootConstants *>(P.value())) {
      auto *Constants = std::get<const dxbc::RootConstants *>(P.value());
      support::endian::write(BOS, Constants->ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(BOS, Constants->RegisterSpace,
                             llvm::endianness::little);
      support::endian::write(BOS, Constants->Num32BitValues,
                             llvm::endianness::little);
    } else if (std::holds_alternative<const dxbc::RST0::v0::RootDescriptor *>(
                   *P)) {
      auto *Descriptor =
          std::get<const dxbc::RST0::v0::RootDescriptor *>(P.value());
      support::endian::write(BOS, Descriptor->ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(BOS, Descriptor->RegisterSpace,
                             llvm::endianness::little);
    } else if (std::holds_alternative<const dxbc::RST0::v1::RootDescriptor *>(
                   *P)) {
      auto *Descriptor =
          std::get<const dxbc::RST0::v1::RootDescriptor *>(P.value());

      support::endian::write(BOS, Descriptor->ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(BOS, Descriptor->RegisterSpace,
                             llvm::endianness::little);
      support::endian::write(BOS, Descriptor->Flags, llvm::endianness::little);
    }
  }
  assert(Storage.size() == getSize());
  OS.write(Storage.data(), Storage.size());
}
