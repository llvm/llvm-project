//===-- NVPTXMCAsmInfo.cpp - NVPTX asm properties -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the NVPTXMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "NVPTXMCAsmInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// -debug-compile - Command line option to inform opt and llc passes to
// compile for debugging
static cl::opt<bool> CompileForDebugging("debug-compile",
                                         cl::desc("Compile for debugging"),
                                         cl::Hidden, cl::init(false));

void NVPTXMCAsmInfo::anchor() {}

NVPTXMCAsmInfo::NVPTXMCAsmInfo(const Triple &TheTriple) {
  if (TheTriple.getArch() == Triple::nvptx64) {
    CodePointerSize = CalleeSaveStackSlotSize = 8;
  }

  CommentString = "//";

  HasSingleParameterDotFile = false;

  InlineAsmStart = " begin inline asm";
  InlineAsmEnd = " end inline asm";

  SupportsDebugInformation = CompileForDebugging;
  // PTX does not allow .align on functions.
  HasFunctionAlignment = false;
  HasDotTypeDotSizeDirective = false;
  // PTX does not allow .hidden or .protected
  HiddenDeclarationVisibilityAttr = HiddenVisibilityAttr = MCSA_Invalid;
  ProtectedVisibilityAttr = MCSA_Invalid;

  Data8bitsDirective = " .b8 ";
  Data16bitsDirective = " .b16 ";
  Data32bitsDirective = " .b32 ";
  Data64bitsDirective = " .b64 ";
  ZeroDirective = " .b8";
  AsciiDirective = " .b8";
  AscizDirective = " .b8";

  // @TODO: Can we just disable this?
  WeakDirective = "\t// .weak\t";
  GlobalDirective = "\t// .globl\t";
}
