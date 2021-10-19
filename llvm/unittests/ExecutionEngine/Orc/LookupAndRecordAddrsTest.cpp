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

  ExecutorAddr FooAddress, BarAddress;
  std::promise<MSVCPError> ErrP;

  lookupAndRecordAddrs([&](Error Err) { ErrP.set_value(std::move(Err)); }, ES,
                       LookupKind::Static, makeJITDylibSearchOrder(&JD),
                       {{Foo, &FooAddress}, {Bar, &BarAddress}});

  Error Err = ErrP.get_future().get();

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(FooAddress.getValue(), FooAddr);
  EXPECT_EQ(BarAddress.getValue(), BarAddr);
}

TEST_F(LookupAndRecordAddrsTest, AsyncRequiredFailure) {
  ExecutorAddr FooAddress, BarAddress;
  std::promise<MSVCPError> ErrP;

  lookupAndRecordAddrs([&](Error Err) { ErrP.set_value(std::move(Err)); }, ES,
                       LookupKind::Static, makeJITDylibSearchOrder(&JD),
                       {{Foo, &FooAddress}, {Bar, &BarAddress}});

  Error Err = ErrP.get_future().get();

  EXPECT_THAT_ERROR(std::move(Err), Failed());
}

TEST_F(LookupAndRecordAddrsTest, AsyncWeakReference) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  ExecutorAddr FooAddress, BarAddress;
  std::promise<MSVCPError> ErrP;

  lookupAndRecordAddrs([&](Error Err) { ErrP.set_value(std::move(Err)); }, ES,
                       LookupKind::Static, makeJITDylibSearchOrder(&JD),
                       {{Foo, &FooAddress}, {Bar, &BarAddress}},
                       SymbolLookupFlags::WeaklyReferencedSymbol);

  Error Err = ErrP.get_future().get();

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(FooAddress.getValue(), FooAddr);
  EXPECT_EQ(BarAddress.getValue(), 0U);
}

TEST_F(LookupAndRecordAddrsTest, BlockingRequiredSuccess) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}, {Bar, BarSym}})));

  ExecutorAddr FooAddress, BarAddress;
  auto Err =
      lookupAndRecordAddrs(ES, LookupKind::Static, makeJITDylibSearchOrder(&JD),
                           {{Foo, &FooAddress}, {Bar, &BarAddress}});

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(FooAddress.getValue(), FooAddr);
  EXPECT_EQ(BarAddress.getValue(), BarAddr);
}

TEST_F(LookupAndRecordAddrsTest, BlockingRequiredFailure) {
  ExecutorAddr FooAddress, BarAddress;
  auto Err =
      lookupAndRecordAddrs(ES, LookupKind::Static, makeJITDylibSearchOrder(&JD),
                           {{Foo, &FooAddress}, {Bar, &BarAddress}});

  EXPECT_THAT_ERROR(std::move(Err), Failed());
}

TEST_F(LookupAndRecordAddrsTest, BlockingWeakReference) {
  cantFail(JD.define(absoluteSymbols({{Foo, FooSym}})));

  ExecutorAddr FooAddress, BarAddress;
  auto Err =
      lookupAndRecordAddrs(ES, LookupKind::Static, makeJITDylibSearchOrder(&JD),
                           {{Foo, &FooAddress}, {Bar, &BarAddress}},
                           SymbolLookupFlags::WeaklyReferencedSymbol);

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(FooAddress.getValue(), FooAddr);
  EXPECT_EQ(BarAddress.getValue(), 0U);
}

} // namespace
