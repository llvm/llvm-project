//===-- SuperHMCAsmInfo.cpp - SuperH Asm Info -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the SuperHAsmInfo class.
///
//===----------------------------------------------------------------------===//

#include "SuperHMCAsmInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void SuperHMCAsmInfo::anchor() {}

SuperHMCAsmInfo::SuperHMCAsmInfo(const Triple &TheTriple,
                                 const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  this->IsLittleEndian = TheTriple.isLittleEndian();
  this->CommentString = ";";
}