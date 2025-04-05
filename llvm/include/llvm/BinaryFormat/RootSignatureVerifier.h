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

struct RootSignatureVerifier {

  static bool verifyRootFlag(uint32_t Flags) { return (Flags & ~0xfff) == 0; }

  static bool verifyVersion(uint32_t Version) {
    return (Version == 1 || Version == 2);
  }

  static bool verifyParameterType(dxbc::RootParameterType Type) {
    switch (Type) {
    case dxbc::RootParameterType::Constants32Bit:
      return true;
    }
    return false;
  }

  static bool verifyShaderVisibility(dxbc::ShaderVisibility Visibility) {
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
    }
    return false;
  };
};
} // namespace dxbc
} // namespace llvm
#endif // LLVM_ROOTSIGNATURE_VALIDATION_H
