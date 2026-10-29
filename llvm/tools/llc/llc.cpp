//===-- llc.cpp - LLVM Native Code Generator entry point -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/InitLLVM.h"

extern "C" int llcMain(int argc, char **argv);

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);
  return llcMain(argc, argv);
}
