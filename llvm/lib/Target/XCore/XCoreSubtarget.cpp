//===- XCoreSubtarget.cpp - XCore Subtarget Information -----------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the XCore specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "XCoreSubtarget.h"
#include "XCore.h"
#include "XCoreGenSubtarget.inc"
using namespace llvm;

XCoreSubtarget::XCoreSubtarget(const TargetMachine &TM, const Module &M, 
                             const std::string &FS)
  : IsXS1A(false),
    IsXS1B(false)
{
  std::string CPU = "xs1a-generic";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
