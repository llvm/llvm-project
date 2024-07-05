//===- lit-cpuid.cpp - Get CPU feature flags for lit exported features ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// lit-cpuid obtains the feature list for the currently running CPU, and outputs
// those flags that are interesting for LLDB lit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <optional>

using namespace llvm;

int main(int argc, char **argv) {
#if defined(__i386__) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(_M_X64)
  if (const std::optional<StringMap<bool>> features =
          sys::getHostCPUFeatures(features)) {
    if (features->contains("sse"))
      outs() << "sse\n";
    if (features->contains("avx"))
      outs() << "avx\n";
    if (features->contains("avx512f"))
      outs() << "avx512f\n";
  } else
    return 1;
#endif

  return 0;
}
