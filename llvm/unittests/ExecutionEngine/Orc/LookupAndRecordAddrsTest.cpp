//===- LookupAndRecordAddrsTest.cpp - Unit tests for LookupAndRecordAddrs -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Testing/Support/Error.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;

class LookupAndRecordAddrsTest : public CoreAPIsBasedStandardTest {};

namespace {

TEST_F(LookupAndRecordAddrsTest, AsyncRequiredSuccess) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  ExecutorAddr ReturnedFooAddr, ReturnedBarAddr;
  std::promise<MSVCPError> ErrP;

  lookupAndRecordAddrs([&](Error Err) { ErrP.set_value(std::move(Err)); }, ES,
                       LookupKind::Static, makeJITDylibSearchOrder(&JD),
                       {{Foo, &ReturnedFooAddr}, {Bar, &ReturnedBarAddr}});

  Error Err = ErrP.get_future().get();

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(ReturnedFooAddr, FooAddr);
  EXPECT_EQ(ReturnedBarAddr, BarAddr);
}

TEST_F(LookupAndRecordAddrsTest, AsyncRequiredFailure) {
  ExecutorAddr RecordedFooAddr, RecordedBarAddr;
  std::promise<MSVCPError> ErrP;

  lookupAndRecordAddrs([&](Error Err) { ErrP.set_value(std::move(Err)); }, ES,
                       LookupKind::Static, makeJITDylibSearchOrder(&JD),
                       {{Foo, &RecordedFooAddr}, {Bar, &RecordedBarAddr}});

  Error Err = ErrP.get_future().get();

  EXPECT_THAT_ERROR(std::move(Err), Failed());
}

TEST_F(LookupAndRecordAddrsTest, AsyncWeakReference) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  ExecutorAddr RecordedFooAddr, RecordedBarAddr;
  std::promise<MSVCPError> ErrP;

  lookupAndRecordAddrs([&](Error Err) { ErrP.set_value(std::move(Err)); }, ES,
                       LookupKind::Static, makeJITDylibSearchOrder(&JD),
                       {{Foo, &RecordedFooAddr}, {Bar, &RecordedBarAddr}},
                       SymbolLookupFlags::WeaklyReferencedSymbol);

  Error Err = ErrP.get_future().get();

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(RecordedFooAddr, FooAddr);
  EXPECT_EQ(RecordedBarAddr, ExecutorAddr());
}

TEST_F(LookupAndRecordAddrsTest, BlockingRequiredSuccess) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  ExecutorAddr RecordedFooAddr, RecordedBarAddr;
  auto Err =
      lookupAndRecordAddrs(ES, LookupKind::Static, makeJITDylibSearchOrder(&JD),
                           {{Foo, &RecordedFooAddr}, {Bar, &RecordedBarAddr}});

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(RecordedFooAddr, FooAddr);
  EXPECT_EQ(RecordedBarAddr, BarAddr);
}

TEST_F(LookupAndRecordAddrsTest, BlockingRequiredFailure) {
  ExecutorAddr RecordedFooAddr, RecordedBarAddr;
  auto Err =
      lookupAndRecordAddrs(ES, LookupKind::Static, makeJITDylibSearchOrder(&JD),
                           {{Foo, &RecordedFooAddr}, {Bar, &RecordedBarAddr}});

  EXPECT_THAT_ERROR(std::move(Err), Failed());
}

TEST_F(LookupAndRecordAddrsTest, BlockingWeakReference) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  ExecutorAddr RecordedFooAddr, RecordedBarAddr;
  auto Err =
      lookupAndRecordAddrs(ES, LookupKind::Static, makeJITDylibSearchOrder(&JD),
                           {{Foo, &RecordedFooAddr}, {Bar, &RecordedBarAddr}},
                           SymbolLookupFlags::WeaklyReferencedSymbol);

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(RecordedFooAddr, FooAddr);
  EXPECT_EQ(RecordedBarAddr, ExecutorAddr());
}

} // namespace
