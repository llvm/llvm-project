//===-- Next32TargetInfo.cpp - Next32 Target Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "llvm/MC/TargetRegistry.h"
using namespace llvm;

namespace llvm {
Target &getTheNext32Target() {
  static Target TheNext32Target;
  return TheNext32Target;
}
} // namespace llvm

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNext32TargetInfo() {
  RegisterTarget<Triple::next32, /*HasJIT=*/true> X(
      getTheNext32Target(), "next32", "Next32 (little endian)", "Next32");
}
