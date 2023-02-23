//===-- SIModeRegisterDefaults.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SIModeRegisterDefaults.h"

using namespace llvm;

SIModeRegisterDefaults::SIModeRegisterDefaults(const Function &F) {
  *this = getDefaultForCallingConv(F.getCallingConv());

  StringRef IEEEAttr = F.getFnAttribute("amdgpu-ieee").getValueAsString();
  if (!IEEEAttr.empty())
    IEEE = IEEEAttr == "true";

  StringRef DX10ClampAttr =
      F.getFnAttribute("amdgpu-dx10-clamp").getValueAsString();
  if (!DX10ClampAttr.empty())
    DX10Clamp = DX10ClampAttr == "true";

  StringRef DenormF32Attr =
      F.getFnAttribute("denormal-fp-math-f32").getValueAsString();
  if (!DenormF32Attr.empty())
    FP32Denormals = parseDenormalFPAttribute(DenormF32Attr);

  StringRef DenormAttr =
      F.getFnAttribute("denormal-fp-math").getValueAsString();
  if (!DenormAttr.empty()) {
    DenormalMode DenormMode = parseDenormalFPAttribute(DenormAttr);
    if (DenormF32Attr.empty())
      FP32Denormals = DenormMode;
    FP64FP16Denormals = DenormMode;
  }
}
