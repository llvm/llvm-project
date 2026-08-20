//===- orc-rt-process-info-check.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-utils/CommandLine.h"
#include "orc-rt/ExecutorProcessInfo.h"
#include <iostream>

int main(int argc, char *argv[]) {

  bool PrintTriple = false;
  bool PrintPageSize = false;
  bool PrintCPUFeatures = false;
  bool PrintHelp = false;

  {
    orc_rt::CommandLineParser P;
    P.addFlag("print-triple", "Print the detected target triple", false,
              PrintTriple)
        .addFlag("print-page-size", "Print the detected page size", false,
                 PrintPageSize)
        .addFlag("print-cpu-features",
                 "Print the detected LLVM target-feature string", false,
                 PrintCPUFeatures)
        .addFlag("help", "Print help", false, PrintHelp);

    if (auto Err = P.parse(argc, argv)) {
      std::cerr << "error: " << orc_rt::toString(std::move(Err)) << "\n";
      std::cerr << P.formatHelp(argv[0]);
      return 1;
    }

    if (PrintHelp) {
      std::cerr << P.formatHelp(argv[0]);
      return 0;
    }
  }

  auto EPI = orc_rt::ExecutorProcessInfo::Detect();
  if (!EPI) {
    std::cerr << "error: " << orc_rt::toString(EPI.takeError()) << "\n";
    return 1;
  }

  if (PrintTriple)
    std::cout << EPI->targetTriple() << "\n";

  if (PrintPageSize)
    std::cout << EPI->pageSize() << "\n";

  if (PrintCPUFeatures)
    std::cout << EPI->targetCPUFeatures() << "\n";

  if (PrintTriple || PrintPageSize || PrintCPUFeatures)
    return 0;

  return 1;
}
