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
#include "llvm/Support/ScopedPrinter.h"
using namespace llvm;

static const EnumEntry<dxil::ResourceClass> ResourceClassNames[] = {
    {"SRV", llvm::dxil::ResourceClass::SRV},
    {"UAV", llvm::dxil::ResourceClass::UAV},
    {"CBV", llvm::dxil::ResourceClass::CBuffer},
    {"Sampler", llvm::dxil::ResourceClass::Sampler},
};

ArrayRef<EnumEntry<llvm::dxil::ResourceClass>> dxil::getResourceClasses() {
  return ArrayRef(ResourceClassNames);
}

StringRef dxil::getResourceClassName(dxil::ResourceClass RC) {
  return enumToStringRef(RC, getResourceClasses());
}
