//===-- NVPTXTargetInfo.cpp - NVPTX Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheNVPTXTarget32;
Target llvm::TheNVPTXTarget64;

extern "C" void LLVMInitializeNVPTXTargetInfo() {
  RegisterTarget<Triple::nvptx> X(TheNVPTXTarget32, "nvptx",
    "NVIDIA PTX 32-bit");
  RegisterTarget<Triple::nvptx64> Y(TheNVPTXTarget64, "nvptx64",
    "NVIDIA PTX 64-bit");
}
