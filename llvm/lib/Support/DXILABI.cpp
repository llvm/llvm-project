//===-- DXILABI.cpp - ABI Sensitive Values for DXIL -----------------------===//
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

#include "llvm/Support/DXILABI.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

StringRef dxil::getResourceClassName(dxil::ResourceClass RC) {
  switch (RC) {
  case dxil::ResourceClass::SRV:
    return "SRV";
  case dxil::ResourceClass::UAV:
    return "UAV";
  case dxil::ResourceClass::CBuffer:
    return "CBV";
  case dxil::ResourceClass::Sampler:
    return "Sampler";
  }
  llvm_unreachable("Invalid ResourceClass enum value");
}
