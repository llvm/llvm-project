//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/EndianStream.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::mcdxbc;

void RootSignatureDesc::write(raw_ostream &OS) const {
  // Root signature header in dxcontainer has 6 uint_32t values.
  const uint32_t HeaderSize = 24;
  const uint32_t ParameterByteSize = Parameters.size_in_bytes();
  const uint32_t NumParametes = Parameters.size();
  const uint32_t Zero = 0;

  // Writing header information
  support::endian::write(OS, Header.Version, llvm::endianness::little);
  support::endian::write(OS, NumParametes, llvm::endianness::little);
  support::endian::write(OS, HeaderSize, llvm::endianness::little);

  // Static samplers still not implemented
  support::endian::write(OS, Zero, llvm::endianness::little);
  support::endian::write(OS, ParameterByteSize + HeaderSize,
                         llvm::endianness::little);

  support::endian::write(OS, Header.Flags, llvm::endianness::little);

  uint32_t ParamsOffset =
      HeaderSize + (3 * sizeof(uint32_t) * Parameters.size());
  for (const dxbc::RootParameter &P : Parameters) {
    support::endian::write(OS, P.ParameterType, llvm::endianness::little);
    support::endian::write(OS, P.ShaderVisibility, llvm::endianness::little);
    support::endian::write(OS, ParamsOffset, llvm::endianness::little);

    // Size of root parameter, removing the ParameterType and ShaderVisibility.
    ParamsOffset += sizeof(dxbc::RootParameter) - 2 * sizeof(uint32_t);
  }

  for (const dxbc::RootParameter &P : Parameters) {
    switch (P.ParameterType) {
    case dxbc::RootParameterType::Constants32Bit: {
      support::endian::write(OS, P.Constants.ShaderRegister,
                             llvm::endianness::little);
      support::endian::write(OS, P.Constants.RegisterSpace,
                             llvm::endianness::little);
      support::endian::write(OS, P.Constants.Num32BitValues,
                             llvm::endianness::little);
    } break;
    case dxbc::RootParameterType::Empty:
      llvm_unreachable("Invalid RootParameterType");
    }
  }
}
