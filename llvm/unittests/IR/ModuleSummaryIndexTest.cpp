//===- ModuleSummaryIndexTest.cpp - ModuleSummaryIndex Unit Tests-============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<ModuleSummaryIndex> makeLLVMIndex(const char *Summary) {
  SMDiagnostic Err;
  std::unique_ptr<ModuleSummaryIndex> Index =
      parseSummaryIndexAssemblyString(Summary, Err);
  if (!Index)
    Err.print("ModuleSummaryIndexTest", errs());
  return Index;
}

TEST(ModuleSummaryIndexTest, MemProfSummaryPrinting) {
  std::unique_ptr<ModuleSummaryIndex> Index = makeLLVMIndex(R"Summary(
^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 23, summaries: (function: (module: ^0, flags: (linkage: external), insts: 2, allocs: ((versions: (none), memProf: ((type: notcold, stackIds: (1, 2, 3, 4)), (type: cold, stackIds: (1, 2, 3, 5))))))))
^2 = gv: (guid: 25, summaries: (function: (module: ^0, flags: (linkage: external), insts: 22, calls: ((callee: ^1)), callsites: ((callee: ^1, clones: (0), stackIds: (3, 4)), (callee: ^1, clones: (0), stackIds: (3, 5))))))
)Summary");

  std::string Data;
  raw_string_ostream OS(Data);

  ASSERT_NE(Index, nullptr);
  auto *CallsiteSummary =
      cast<FunctionSummary>(Index->getGlobalValueSummary(/*guid=*/25));
  for (auto &CI : CallsiteSummary->callsites())
    OS << "\n" << CI;

  auto *AllocSummary =
      cast<FunctionSummary>(Index->getGlobalValueSummary(/*guid=*/23));
  for (auto &AI : AllocSummary->allocs())
    OS << "\n" << AI;

  EXPECT_EQ(Data, R"(
Callee: 23 Clones: 0 StackIds: 2, 3
Callee: 23 Clones: 0 StackIds: 2, 4
Versions: 0 MIB:
		AllocType 1 StackIds: 0, 1, 2, 3
		AllocType 2 StackIds: 0, 1, 2, 4
)");
}
} // end anonymous namespace
