//===- FormatVariadicBM.cpp - formatv() benchmark ---------- --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <string>
#include <vector>

using namespace llvm;
using namespace std;

// Generate a list of format strings that have `NumReplacements` replacements
// by permuting the replacements and some literal text.
static vector<string> getFormatStrings(int NumReplacements) {
  vector<string> Components;
  for (int I = 0; I < NumReplacements; I++)
    Components.push_back("{" + to_string(I) + "}");
  // Intersperse these with some other literal text (_).
  const string_view Literal = "____";
  for (char C : Literal)
    Components.push_back(string(1, C));

  vector<string> Formats;
  do {
    string Concat;
    for (const string &C : Components)
      Concat += C;
    Formats.emplace_back(Concat);
  } while (next_permutation(Components.begin(), Components.end()));
  return Formats;
}

// Generate the set of formats to exercise outside the benchmark code.
static const vector<vector<string>> Formats = {
    getFormatStrings(1), getFormatStrings(2), getFormatStrings(3),
    getFormatStrings(4), getFormatStrings(5),
};

// Benchmark formatv() for a variety of format strings and 1-5 replacements.
static void BM_FormatVariadic(benchmark::State &state) {
  for (auto _ : state) {
    for (const string &Fmt : Formats[0])
      formatv(Fmt.c_str(), 1).str();
    for (const string &Fmt : Formats[1])
      formatv(Fmt.c_str(), 1, 2).str();
    for (const string &Fmt : Formats[2])
      formatv(Fmt.c_str(), 1, 2, 3).str();
    for (const string &Fmt : Formats[3])
      formatv(Fmt.c_str(), 1, 2, 3, 4).str();
    for (const string &Fmt : Formats[4])
      formatv(Fmt.c_str(), 1, 2, 3, 4, 5).str();
  }
}

BENCHMARK(BM_FormatVariadic);

BENCHMARK_MAIN();
