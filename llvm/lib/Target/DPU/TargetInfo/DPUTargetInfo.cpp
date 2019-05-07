//===-- DPUTargetInfo.cpp - DPU Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace llvm {
Target TheDPUTarget;
}

extern "C" void LLVMInitializeDPUTargetInfo() {
  RegisterTarget<Triple::dpu, /*HasJIT=*/false> X(TheDPUTarget, "dpu",
                                                  "UPMEM DPU", "DPU");
}
