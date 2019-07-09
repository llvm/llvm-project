//===- MC/MCAsmInfoXCOFF.cpp - XCOFF asm properties ------------ *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoXCOFF.h"

using namespace llvm;

void MCAsmInfoXCOFF::anchor() {}

MCAsmInfoXCOFF::MCAsmInfoXCOFF() {
  IsLittleEndian = false;
  HasDotTypeDotSizeDirective = false;
}
