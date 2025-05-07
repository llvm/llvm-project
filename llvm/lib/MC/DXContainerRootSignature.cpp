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
                Parameters.size() * sizeof(dxbc::RootParameterHeader);

  for (const mcdxbc::RootParameter &P : Parameters) {
    switch (P.Header.ParameterType) {
    case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit):
      Size += sizeof(dxbc::RootConstants);
      break;
    }
  }
  return Size;
}

void RootSignatureDesc::write(raw_ostream &OS) const {
  SmallString<256> Storage;
  raw_svector_ostream BOS(Storage);
  BOS.reserveExtraSpace(getSize());

  const uint32_t NumParameters = Parameters.size();

  support::endian::write(BOS, Version, llvm::endianness::little);
  support::endian::write(BOS, NumParameters, llvm::endianness::little);
  support::endian::write(BOS, RootParameterOffset, llvm::endianness::little);
  support::endian::write(BOS, NumStaticSamplers, llvm::endianness::little);
  support::endian::write(BOS, StaticSamplersOffset, llvm::endianness::little);
  support::endian::write(BOS, Flags, llvm::endianness::little);

  SmallVector<uint32_t> ParamsOffsets;
  for (const mcdxbc::RootParameter &P : Parameters) {
    support::endian::write(BOS, P.Header.ParameterType,
                           llvm::endianness::little);
    support::endian::write(BOS, P.Header.ShaderVisibility,
                           llvm::endianness::little);

    ParamsOffsets.push_back(writePlaceholder(BOS));
  }

  assert(NumParameters == ParamsOffsets.size());
  for (size_t I = 0; I < NumParameters; ++I) {
    rewriteOffsetToCurrentByte(BOS, ParamsOffsets[I]);
    const mcdxbc::RootParameter &P = Parameters[I];

    switch (P.Header.ParameterType) {
    case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit):
      support::endian::write(BOS, P.Constants.ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(BOS, P.Constants.RegisterSpace,
                             llvm::endianness::little);
      support::endian::write(BOS, P.Constants.Num32BitValues,
                             llvm::endianness::little);
      break;
    }
  }
  assert(Storage.size() == getSize());
  OS.write(Storage.data(), Storage.size());
}
