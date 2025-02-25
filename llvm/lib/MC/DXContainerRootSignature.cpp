//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;
using namespace llvm::mcdxbc;

void setRewrite(raw_ostream &Stream, uint32_t &Offset) {
  const uint32_t DummyValue = std::numeric_limits<uint32_t>::max();
  Offset = Stream.tell();
  support::endian::write(Stream, DummyValue, llvm::endianness::little);
}

void rewriteOffset(buffer_ostream &Stream, uint32_t Offset) {
  uint32_t Value = Stream.tell();
  auto *InsertPoint = &Stream.buffer()[Offset];
  support::endian::write(InsertPoint, Value, llvm::endianness::little);
}

void RootSignatureDesc::write(raw_ostream &OS) const {
  buffer_ostream Writer(OS);
  const uint32_t NumParameters = Parameters.size();
  const uint32_t Zero = 0;

  support::endian::write(Writer, Header.Version, llvm::endianness::little);
  support::endian::write(Writer, NumParameters, llvm::endianness::little);

  uint32_t HeaderPoint;
  setRewrite(Writer, HeaderPoint);

  support::endian::write(Writer, Zero, llvm::endianness::little);
  support::endian::write(Writer, Zero, llvm::endianness::little);
  support::endian::write(Writer, Header.Flags, llvm::endianness::little);

  rewriteOffset(Writer, HeaderPoint);

  SmallVector<uint32_t> ParamsOffset;
  for (const auto &P : Parameters) {
    support::endian::write(Writer, P.ParameterType, llvm::endianness::little);
    support::endian::write(Writer, P.ShaderVisibility,
                           llvm::endianness::little);

    uint32_t Offset;
    setRewrite(Writer, Offset);

    ParamsOffset.push_back(Offset);
  }

  assert(NumParameters == ParamsOffset.size());
  for (size_t I = 0; I < NumParameters; ++I) {
    rewriteOffset(Writer, ParamsOffset[I]);
    const auto &P = Parameters[I];

    switch (P.ParameterType) {
    case dxbc::RootParameterType::Constants32Bit: {
      support::endian::write(Writer, P.Constants.ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(Writer, P.Constants.RegisterSpace,
                             llvm::endianness::little);
      support::endian::write(Writer, P.Constants.Num32BitValues,
                             llvm::endianness::little);
    } break;
    case dxbc::RootParameterType::Empty:
      llvm_unreachable("Invalid RootParameterType");
    }
  }
}
