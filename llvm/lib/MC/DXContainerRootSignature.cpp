//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include <vector>

using namespace llvm;
using namespace llvm::mcdxbc;

Error RootSignatureDesc::write(raw_ostream &OS) const {
  // Header Size + accounting for parameter offset + parameters size
  std::vector<uint8_t> Buffer(24 + (Parameters.size() * 4) +
                              Parameters.size_in_bytes());
  BinaryStreamWriter Writer(Buffer, llvm::endianness::little);

  SmallVector<uint64_t> OffsetsToReplace;
  SmallVector<uint32_t> ValuesToReplaceOffsetsWith;
  const uint32_t Dummy = std::numeric_limits<uint32_t>::max();

  const uint32_t NumParameters = Parameters.size();
  const uint32_t Zero = 0;

  if (Error Err = Writer.writeInteger(Header.Version))
    return Err;

  if (Error Err = Writer.writeInteger(NumParameters))
    return Err;

  OffsetsToReplace.push_back(Writer.getOffset());
  if (Error Err = Writer.writeInteger(Dummy))
    return Err;

  // Static samplers still not implemented
  if (Error Err = Writer.writeInteger(Zero))
    return Err;

  if (Error Err = Writer.writeInteger(Zero))
    return Err;

  if (Error Err = Writer.writeInteger(Header.Flags))
    return Err;

  ValuesToReplaceOffsetsWith.push_back(Writer.getOffset());

  for (const dxbc::RootParameter &P : Parameters) {
    if (Error Err = Writer.writeEnum(P.ParameterType))
      return Err;
    if (Error Err = Writer.writeEnum(P.ShaderVisibility))
      return Err;

    OffsetsToReplace.push_back(Writer.getOffset());
    if (Error Err = Writer.writeInteger(Dummy))
      return Err;
  }

  for (const dxbc::RootParameter &P : Parameters) {
    ValuesToReplaceOffsetsWith.push_back(Writer.getOffset());
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

  assert(ValuesToReplaceOffsetsWith.size() == OffsetsToReplace.size() &&
         "Offset missing value to replace with.");

  for (size_t It = 0; It < ValuesToReplaceOffsetsWith.size(); It++) {
    uint32_t Position = OffsetsToReplace[It];
    uint32_t Value = ValuesToReplaceOffsetsWith[It];

    Writer.setOffset(Position);
    if (Error Err = Writer.writeInteger(Value))
      return Err;
  }

  llvm::ArrayRef<char> BufferRef(reinterpret_cast<char *>(Buffer.data()),
                                 Buffer.size());
  OS.write(BufferRef.data(), BufferRef.size());

  return Error::success();
}
