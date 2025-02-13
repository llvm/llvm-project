//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;
using namespace llvm::mcdxbc;

void RootSignatureDesc::write(raw_ostream &OS) const {

  support::endian::write(OS, Version, llvm::endianness::little);
  support::endian::write(OS, NumParameters, llvm::endianness::little);
  support::endian::write(OS, RootParametersOffset, llvm::endianness::little);
  support::endian::write(OS, NumStaticSamplers, llvm::endianness::little);
  support::endian::write(OS, StaticSamplersOffset, llvm::endianness::little);
  support::endian::write(OS, Flags, llvm::endianness::little);
}

void RootParameter::write(raw_ostream &OS) {
  support::endian::write(OS, ParameterType, llvm::endianness::little);
  support::endian::write(OS, ShaderVisibility, llvm::endianness::little);

  switch(ParameterType){
  case dxbc::RootParameterType::Constants32Bit:
    Constants.write(OS);
    break;
  }
}

void RootConstants::write(raw_ostream &OS) {
  support::endian::write(OS, Num32BitValues, llvm::endianness::little);
  support::endian::write(OS, RegisterSpace, llvm::endianness::little);
  support::endian::write(OS, ShaderRegister, llvm::endianness::little);
}
