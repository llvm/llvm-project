//===-- driver-template.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LLVMDriver.h"

int llvm_cxxfilt_main(int argc, char **, const llvm::ToolContext &);

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);
  return llvm_cxxfilt_main(argc, argv, {argv[0], nullptr, false});
}
