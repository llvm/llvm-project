//===-- WebAssemblyMCAsmInfo.cpp - WebAssembly asm properties -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declarations of the WebAssemblyMCAsmInfo
/// properties.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCAsmInfo.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-mc-asm-info"

WebAssemblyMCAsmInfoELF::~WebAssemblyMCAsmInfoELF() {}

WebAssemblyMCAsmInfoELF::WebAssemblyMCAsmInfoELF(const Triple &T) {
  CodePointerSize = CalleeSaveStackSlotSize = T.isArch64Bit() ? 8 : 4;

  // TODO: What should MaxInstLength be?

  UseDataRegionDirectives = true;

  // Use .skip instead of .zero because .zero is confusing when used with two
  // arguments (it doesn't actually zero things out).
  ZeroDirective = "\t.skip\t";

  Data8bitsDirective = "\t.int8\t";
  Data16bitsDirective = "\t.int16\t";
  Data32bitsDirective = "\t.int32\t";
  Data64bitsDirective = "\t.int64\t";

  AlignmentIsInBytes = false;
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::Log2Alignment;

  SupportsDebugInformation = true;

  // TODO: UseIntegratedAssembler?

  // WebAssembly's stack is never executable.
  UsesNonexecutableStackSection = false;
}

WebAssemblyMCAsmInfo::~WebAssemblyMCAsmInfo() {}

WebAssemblyMCAsmInfo::WebAssemblyMCAsmInfo(const Triple &T) {
  CodePointerSize = CalleeSaveStackSlotSize = T.isArch64Bit() ? 8 : 4;

  // TODO: What should MaxInstLength be?

  UseDataRegionDirectives = true;

  // Use .skip instead of .zero because .zero is confusing when used with two
  // arguments (it doesn't actually zero things out).
  ZeroDirective = "\t.skip\t";

  Data8bitsDirective = "\t.int8\t";
  Data16bitsDirective = "\t.int16\t";
  Data32bitsDirective = "\t.int32\t";
  Data64bitsDirective = "\t.int64\t";

  AlignmentIsInBytes = false;
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::Log2Alignment;

  SupportsDebugInformation = true;

  // TODO: UseIntegratedAssembler?
}
