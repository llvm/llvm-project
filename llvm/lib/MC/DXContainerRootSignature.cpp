//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include <cstdint>
#include <sys/types.h>

using namespace llvm;
using namespace llvm::mcdxbc;

Error setRewrite(BinaryStreamWriter &Stream, uint32_t &Offset) {
  const uint32_t DummyValue = std::numeric_limits<uint32_t>::max();

  Offset = Stream.getOffset();

  if (Error Err = Stream.writeInteger(DummyValue))
    return Err;

  return Error::success();
}

Error rewriteOffset(BinaryStreamWriter &Stream, uint32_t Offset) {
  uint64_t Value = Stream.getOffset();
  Stream.setOffset(Offset);
  if (Error Err = Stream.writeInteger((uint32_t)Value))
    return Err;

  Stream.setOffset(Value);

  return Error::success();
}

Error RootSignatureDesc::write(raw_ostream &OS) const {
  std::vector<uint8_t> Buffer(getSizeInBytes());
  BinaryStreamWriter Writer(Buffer, llvm::endianness::little);

  const uint32_t NumParameters = Parameters.size();
  const uint32_t Zero = 0;

  if (Error Err = Writer.writeInteger(Header.Version))
    return Err;

  if (Error Err = Writer.writeInteger(NumParameters))
    return Err;

  uint32_t HeaderPoint;
  if (Error Err = setRewrite(Writer, HeaderPoint))
    return Err;

  // Static samplers still not implemented
  if (Error Err = Writer.writeInteger(Zero))
    return Err;

  if (Error Err = Writer.writeInteger(Zero))
    return Err;

  if (Error Err = Writer.writeInteger(Header.Flags))
    return Err;

  if (Error Err = rewriteOffset(Writer, HeaderPoint))
    return Err;

  SmallVector<uint32_t> ParamsOffset;
  for (const auto &P : Parameters) {

    if (Error Err = Writer.writeEnum(P.ParameterType))
      return Err;

    if (Error Err = Writer.writeEnum(P.ShaderVisibility))
      return Err;

    uint32_t Offset;
    if (Error Err = setRewrite(Writer, Offset))
      return Err;
    ParamsOffset.push_back(Offset);
  }

  assert(NumParameters == ParamsOffset.size());
  for (size_t I = 0; I < NumParameters; ++I) {
    if (Error Err = rewriteOffset(Writer, ParamsOffset[I]))
      return Err;

    const auto &P = Parameters[I];

    switch (P.ParameterType) {
    case dxbc::RootParameterType::Constants32Bit: {
      if (Error Err = Writer.writeInteger(P.Constants.ShaderRegister))
        return Err;
      if (Error Err = Writer.writeInteger(P.Constants.RegisterSpace))
        return Err;
      if (Error Err = Writer.writeInteger(P.Constants.Num32BitValues))
        return Err;
    } break;
    case dxbc::RootParameterType::Empty:
      llvm_unreachable("Invalid RootParameterType");
    }
  }

  llvm::ArrayRef<char> BufferRef(reinterpret_cast<char *>(Buffer.data()),
                                 Buffer.size());
  OS.write(BufferRef.data(), BufferRef.size());

  return Error::success();
}
