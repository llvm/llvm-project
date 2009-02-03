//===-- ARMSubtarget.cpp - ARM Subtarget Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "ARMSubtarget.h"
#include "ARMGenSubtarget.inc"
#include "llvm/Module.h"
using namespace llvm;

ARMSubtarget::ARMSubtarget(const Module &M, const std::string &FS, bool thumb)
  : ARMArchVersion(V4T)
  , HasVFP2(false)
  , IsThumb(thumb)
  , UseThumbBacktraces(false)
  , IsR9Reserved(false)
  , stackAlignment(4)
  , TargetType(isELF) // Default to ELF unless otherwise specified.
  , TargetABI(ARM_ABI_APCS) {

  // Determine default and user specified characteristics
  std::string CPU = "generic";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
  if (TT.length() > 5) {
    if (TT.find("-darwin") != std::string::npos)
      TargetType = isDarwin;
  } else if (TT.empty()) {
#if defined(__APPLE__)
    TargetType = isDarwin;
#endif
  }

  if (TT.find("eabi") != std::string::npos)
    TargetABI = ARM_ABI_AAPCS;

  if (isAAPCS_ABI())
    stackAlignment = 8;

  if (isTargetDarwin()) {
    UseThumbBacktraces = true;
    IsR9Reserved = true;
  }
}
