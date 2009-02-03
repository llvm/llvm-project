//===- SparcSubtarget.cpp - SPARC Subtarget Information -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPARC specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "SparcSubtarget.h"
#include "SparcGenSubtarget.inc"
using namespace llvm;

// FIXME: temporary.
#include "llvm/Support/CommandLine.h"
namespace {
  cl::opt<bool> EnableV9("enable-sparc-v9-insts", cl::Hidden,
                          cl::desc("Enable V9 instructions in the V8 target"));
}

SparcSubtarget::SparcSubtarget(const Module &M, const std::string &FS) {
  // Set the default features.
  IsV9 = false;
  V8DeprecatedInsts = false;
  IsVIS = false;
  
  // Determine default and user specified characteristics
  std::string CPU = "generic";

  // FIXME: autodetect host here!
  CPU = "v9";   // What is a good way to detect V9?
  
  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);

  // Unless explicitly enabled, disable the V9 instructions.
  if (!EnableV9)
    IsV9 = false;
}
