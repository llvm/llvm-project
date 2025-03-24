//===-- llvm/BinaryFormat/RootSignaturesValidation.h ------------*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Validation logic for Root Signatures
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ROOTSIGNATURE_VALIDATION_H
#define LLVM_ROOTSIGNATURE_VALIDATION_H

#include "llvm/BinaryFormat/DXContainer.h"
namespace llvm {

namespace dxbc {

struct RootSignatureValidations {

  static bool isValidRootFlag(uint32_t Flags) { return (Flags & ~0xfff) == 0; }

  static bool isValidVersion(uint32_t Version) {
    return (Version == 1 || Version == 2);
  }

  static bool isValidParameterType(dxbc::RootParameterType Type) {
    switch (Type) {
    case dxbc::RootParameterType::Constants32Bit:
      return true;
    default:
      return false;
    }
  }

  static bool isValidShaderVisibility(dxbc::ShaderVisibility Visibility) {
    switch (Visibility) {

    case ShaderVisibility::All:
    case ShaderVisibility::Vertex:
    case ShaderVisibility::Hull:
    case ShaderVisibility::Domain:
    case ShaderVisibility::Geometry:
    case ShaderVisibility::Pixel:
    case ShaderVisibility::Amplification:
    case ShaderVisibility::Mesh:
      return true;
    default:
      return false;
    }
  }
};
} // namespace dxbc
} // namespace llvm
#endif // LLVM_ROOTSIGNATURE_VALIDATION_H
