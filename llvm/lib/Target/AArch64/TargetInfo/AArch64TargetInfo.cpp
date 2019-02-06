//===-- AArch64TargetInfo.cpp - AArch64 Target Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;
namespace llvm {
Target &getTheAArch64leTarget() {
  static Target TheAArch64leTarget;
  return TheAArch64leTarget;
}
Target &getTheAArch64beTarget() {
  static Target TheAArch64beTarget;
  return TheAArch64beTarget;
}
Target &getTheARM64Target() {
  static Target TheARM64Target;
  return TheARM64Target;
}
} // namespace llvm

extern "C" void LLVMInitializeAArch64TargetInfo() {
  // Now register the "arm64" name for use with "-march". We don't want it to
  // take possession of the Triple::aarch64 tag though.
  TargetRegistry::RegisterTarget(getTheARM64Target(), "arm64",
                                 "ARM64 (little endian)", "AArch64",
                                 [](Triple::ArchType) { return false; }, true);

  RegisterTarget<Triple::aarch64, /*HasJIT=*/true> Z(
      getTheAArch64leTarget(), "aarch64", "AArch64 (little endian)", "AArch64");
  RegisterTarget<Triple::aarch64_be, /*HasJIT=*/true> W(
      getTheAArch64beTarget(), "aarch64_be", "AArch64 (big endian)", "AArch64");
}
