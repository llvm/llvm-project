//===- PIC16Subtarget.cpp - PIC16 Subtarget Information -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PIC16 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "PIC16Subtarget.h"
#include "PIC16GenSubtarget.inc"

using namespace llvm;

PIC16Subtarget::PIC16Subtarget(const Module &M, const std::string &FS, 
                               bool Cooper)
  :IsCooper(Cooper)
{
  std::string CPU = "generic";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
