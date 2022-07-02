//===-- DXILOperationCommon.h - DXIL Operation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is created to share common definitions used by both the
// DXILOpBuilder and the table
//  generator.
// Documentation for DXIL can be found in
// https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DXILOPERATIONCOMMON_H
#define LLVM_SUPPORT_DXILOPERATIONCOMMON_H

#include "llvm/ADT/StringSwitch.h"

namespace llvm {
namespace DXIL {

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

inline ParameterKind parameterTypeNameToKind(StringRef Name) {
  return StringSwitch<ParameterKind>(Name)
      .Case("void", ParameterKind::VOID)
      .Case("half", ParameterKind::HALF)
      .Case("float", ParameterKind::FLOAT)
      .Case("double", ParameterKind::DOUBLE)
      .Case("i1", ParameterKind::I1)
      .Case("i8", ParameterKind::I8)
      .Case("i16", ParameterKind::I16)
      .Case("i32", ParameterKind::I32)
      .Case("i64", ParameterKind::I64)
      .Case("$o", ParameterKind::OVERLOAD)
      .Case("dx.types.Handle", ParameterKind::DXIL_HANDLE)
      .Case("dx.types.CBufRet", ParameterKind::CBUFFER_RET)
      .Case("dx.types.ResRet", ParameterKind::RESOURCE_RET)
      .Default(ParameterKind::INVALID);
}

} // namespace DXIL
} // namespace llvm

#endif
