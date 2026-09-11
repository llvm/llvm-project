//===- ogre.cpp - ORC Generic Runtime Environment -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The ORC Generic Runtime Environment (OGRE) is intended as a canonical
// "blank executor".
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/tools/OptionParser.h"

#include <optional>
#include <stdio.h>

using namespace orc_rt;

struct Options {
  bool Verbose = false;
};

static std::optional<Options> parseArgs(int argc, char *argv[]) noexcept {
  Options O;
  bool ShowHelp = false;

  OptionParser P;
  P.addFlag("verbose", "Print verbose output", false, O.Verbose, 'v')
      .addFlag("help", "Display this help message", false, ShowHelp, 'h');

  auto PrintHelp = [&]() -> std::nullopt_t {
    const char *ProgName = argc != 0 ? argv[0] : "ogre";
    fprintf(stderr, "%s usage:\n%s\n", ProgName,
            P.formatHelp(ProgName).c_str());
    return std::nullopt;
  };

  if (auto Err = P.parseAsMainArgs(argc, argv)) {
    fprintf(stderr, "error: %s\n", toString(std::move(Err)).c_str());
    return PrintHelp();
  }

  if (ShowHelp)
    return PrintHelp();

  return O;
}

int main(int argc, char *argv[]) {
  auto Opts = parseArgs(argc, argv);
  if (!Opts)
    return 1;

  if (Opts->Verbose)
    printf("*unintelligible grunts*\n");

  return 0;
}
