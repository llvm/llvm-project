//===-- DXILABI.h - ABI Sensitive Values for DXIL ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of various constants and enums that are
// required to remain stable as per the DXIL format's requirements.
//
// Documentation for DXIL can be found in
// https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DXILABI_H
#define LLVM_SUPPORT_DXILABI_H

#include "llvm/ADT/StringSwitch.h"

namespace llvm {
namespace dxil {

enum class ParameterKind : uint8_t {
  INVALID = 0,
  VOID,
  HALF,
  FLOAT,
  DOUBLE,
  I1,
  I8,
  I16,
  I32,
  I64,
  OVERLOAD,
  CBUFFER_RET,
  RESOURCE_RET,
  DXIL_HANDLE,
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_SUPPORT_DXILABI_H
