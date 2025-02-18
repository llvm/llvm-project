//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/ADT/bit.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/EndianStream.h"
#include <cstdint>
#include <sys/types.h>

using namespace llvm;
using namespace llvm::mcdxbc;

template <typename T> static uint32_t getSizeOf() {
  return static_cast<uint32_t>(sizeof(T));
}

void RootSignatureDesc::write(raw_ostream &OS) const {
  uint32_t Offset = 16;
  const uint32_t ParametersOffset =
      getSizeOf<dxbc::RootSignatureHeader>() + Offset;
  const uint32_t ParameterByteSize = Parameters.size_in_bytes();

  // Writing header information
  support::endian::write(OS, Header.Version, llvm::endianness::little);
  Offset += getSizeOf<uint32_t>();

  support::endian::write(OS, (uint32_t)Parameters.size(),
                         llvm::endianness::little);
  Offset += getSizeOf<uint32_t>();

  support::endian::write(OS, ParametersOffset, llvm::endianness::little);
  Offset += getSizeOf<uint32_t>();

  support::endian::write(OS, ((uint32_t)0), llvm::endianness::little);
  Offset += getSizeOf<uint32_t>();

  support::endian::write(OS, ParameterByteSize + ParametersOffset,
                         llvm::endianness::little);
  Offset += getSizeOf<uint32_t>();

  support::endian::write(OS, Header.Flags, llvm::endianness::little);

  for (const dxbc::RootParameter &P : Parameters) {
    support::endian::write(OS, P.ParameterType, llvm::endianness::little);
    support::endian::write(OS, P.ShaderVisibility, llvm::endianness::little);
    support::endian::write(OS, Offset, llvm::endianness::little);

    switch (P.ParameterType) {
    case dxbc::RootParameterType::Constants32Bit: {
      support::endian::write(OS, P.Constants.ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(OS, P.Constants.RegisterSpace,
                             llvm::endianness::little);
      support::endian::write(OS, P.Constants.Num32BitValues,
                             llvm::endianness::little);
      Offset += getSizeOf<dxbc::RootConstants>() + 3 * getSizeOf<uint32_t>();

    } break;
    }
  }
}
