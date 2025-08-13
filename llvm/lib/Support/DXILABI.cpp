//===-- DXILABI.cpp - ABI Sensitive Values for DXIL --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions that can be reused accross different stages
// dxil generation.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/DXContainer.h"
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
