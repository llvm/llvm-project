//===--- llvm-advisor.cpp - LLVM Advisor ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

cl::OptionCategory AdvisorCategory("llvm-advisor options");

} // namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(AdvisorCategory);
  cl::ParseCommandLineOptions(argc, argv, "LLVM Advisor\n");
  outs() << "llvm-advisor foundation is available.\n";
  return 0;
}
