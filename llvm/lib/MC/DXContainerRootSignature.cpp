//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/BinaryStreamWriter.h"

using namespace llvm;
using namespace llvm::mcdxbc;

Error StreamOffsetHelper::addOffset(std::string Key) {
  const uint32_t DummyValue = std::numeric_limits<uint32_t>::max();

  uint32_t Offset = Stream.getOffset();
  auto Value = std::make_pair(Offset, DummyValue);

  OffsetsMaping.insert_or_assign(Key, Value);

  if (Error Err = Stream.writeInteger(DummyValue))
    return Err;

  return Error::success();
}

void StreamOffsetHelper::addRewriteValue(std::string Key) {
  auto It = OffsetsMaping.find(Key);
  assert(It != OffsetsMaping.end() && "Offset address was not found.");
  auto [Offset, _] = It->second;

  uint32_t Value = Stream.getOffset();

  std::pair<uint32_t, uint32_t> NewValue = std::make_pair(Offset, Value);
  OffsetsMaping.insert_or_assign(Key, NewValue);
}

Error StreamOffsetHelper::rewrite() {
  for (auto &[Key, RewriteInfo] : OffsetsMaping) {
    auto [Position, Value] = RewriteInfo;
    assert(Value != std::numeric_limits<uint32_t>::max());

    Stream.setOffset(Position);
    if (Error Err = Stream.writeInteger(Value))
      return Err;
  }

  return Error::success();
}

Error RootSignatureDesc::write(raw_ostream &OS) const {
  std::vector<uint8_t> Buffer(getSizeInBytes());
  BinaryStreamWriter Writer(Buffer, llvm::endianness::little);

  StreamOffsetHelper OffsetMap(Writer);

  const uint32_t NumParameters = Parameters.size();
  const uint32_t Zero = 0;

  if (Error Err = Writer.writeInteger(Header.Version))
    return Err;

  if (Error Err = Writer.writeInteger(NumParameters))
    return Err;

  if (Error Err = OffsetMap.addOffset("header"))
    return Err;

  // Static samplers still not implemented
  if (Error Err = Writer.writeInteger(Zero))
    return Err;

  if (Error Err = Writer.writeInteger(Zero))
    return Err;

  if (Error Err = Writer.writeInteger(Header.Flags))
    return Err;

  OffsetMap.addRewriteValue("header");

  for (size_t It = 0; It < Parameters.size(); It++) {
    const auto &P = Parameters[It];

    if (Error Err = Writer.writeEnum(P.ParameterType))
      return Err;

    if (Error Err = Writer.writeEnum(P.ShaderVisibility))
      return Err;

    std::string Key = ("parameters" + Twine(It)).str();
    if (Error Err = OffsetMap.addOffset(Key))
      return Err;
  }

  for (size_t It = 0; It < Parameters.size(); It++) {
    const auto &P = Parameters[It];

    std::string Key = ("parameters" + Twine(It)).str();
    OffsetMap.addRewriteValue(Key);

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

  if (Error Err = OffsetMap.rewrite())
    return Err;

  llvm::ArrayRef<char> BufferRef(reinterpret_cast<char *>(Buffer.data()),
                                 Buffer.size());
  OS.write(BufferRef.data(), BufferRef.size());

  return Error::success();
}
