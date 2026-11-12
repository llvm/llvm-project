//===- GetIntrinsicInfoTableEntries.cpp - IIT signature benchmark ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;
using namespace Intrinsic;

static void BM_GetIntrinsicInfoTableEntries(benchmark::State &state) {
  SmallVector<IITDescriptor> Table;
  for (auto _ : state) {
    for (ID ID = 1; ID < num_intrinsics; ++ID) {
      // This makes sure the vector does not keep growing, as well as after the
      // first iteration does not result in additional allocations.
      Table.clear();
      getIntrinsicInfoTableEntries(ID, Table);
    }
  }
}

BENCHMARK(BM_GetIntrinsicInfoTableEntries);

BENCHMARK_MAIN();
