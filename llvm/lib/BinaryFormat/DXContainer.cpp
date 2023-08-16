
//===-- llvm/BinaryFormat/DXContainer.cpp - DXContainer Utils ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for working with DXContainers.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;
using namespace llvm::dxbc;

dxbc::PartType dxbc::parsePartType(StringRef S) {
#define CONTAINER_PART(PartName) .Case(#PartName, PartType::PartName)
  return StringSwitch<dxbc::PartType>(S)
#include "llvm/BinaryFormat/DXContainerConstants.def"
      .Default(dxbc::PartType::Unknown);
}

bool ShaderHash::isPopulated() {
  static uint8_t Zeros[16] = {0};
  return Flags > 0 || 0 != memcmp(&Digest, &Zeros, 16);
}
