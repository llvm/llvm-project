//===- LookupAndApplyTest.cpp - Test lookupAndApply -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LookupAndApply.h"

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Testing/Support/Error.h"

#include <future>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

// Two arbitrary, distinct addresses to resolve symbols to.
constexpr uint64_t AddrAValue = 0x1000;
constexpr uint64_t AddrBValue = 0x2000;

// Define Name -> Addr in JD as an exported absolute symbol.
static void defineAddr(JITDylib &JD, StringRef Name, ExecutorAddr Addr) {
  auto &ES = JD.getExecutionSession();
  cantFail(JD.define(
      absoluteSymbols({{ES.intern(Name), {Addr, JITSymbolFlags::Exported}}})));
}

} // namespace

// recordAddr writes the resolved address of a required symbol.
TEST(LookupAndApplyTest, RecordAddr) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.getBootstrapJITDylib();
  defineAddr(JD, "addr_a", ExecutorAddr(AddrAValue));

  ExecutorAddr A;
  cantFail(lookupAndApply(JD, {recordAddr("addr_a", &A)}));
  EXPECT_EQ(A, ExecutorAddr(AddrAValue));

  cantFail(ES.endSession());
}

// A required symbol that is missing fails the lookup.
TEST(LookupAndApplyTest, RecordAddrRequiredAbsentFails) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  ExecutorAddr A(AddrAValue);
  EXPECT_THAT_ERROR(
      lookupAndApply(ES.getBootstrapJITDylib(), {recordAddr("absent", &A)}),
      Failed());

  cantFail(ES.endSession());
}

// A weakly-referenced symbol that is missing records a null address, rather
// than failing the lookup.
TEST(LookupAndApplyTest, RecordAddrWeaklyReferencedAbsent) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));

  ExecutorAddr A(AddrAValue);
  cantFail(lookupAndApply(
      ES.getBootstrapJITDylib(),
      {recordAddr("absent", &A, SymbolLookupFlags::WeaklyReferencedSymbol)}));
  EXPECT_EQ(A, ExecutorAddr());

  cantFail(ES.endSession());
}

// Several prepare functions in one call are all applied, and a single one may
// contribute more than one symbol.
TEST(LookupAndApplyTest, MultiplePrepareFns) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.getBootstrapJITDylib();
  defineAddr(JD, "addr_a", ExecutorAddr(AddrAValue));
  defineAddr(JD, "addr_b", ExecutorAddr(AddrBValue));

  ExecutorAddr A, B, C, D;

  // A composite prepare fn: contributes both names, records both results.
  auto RecordBoth = [&C, &D](SymbolLookupSet &LS,
                             ExecutionSession &ES) -> LookupApplyFn {
    auto NA = ES.intern("addr_a");
    auto NB = ES.intern("addr_b");
    LS.add(NA);
    LS.add(NB);
    return
        [&C, &D, NA = std::move(NA), NB = std::move(NB)](const SymbolMap &M) {
          C = M.lookup(NA).getAddress();
          D = M.lookup(NB).getAddress();
        };
  };

  cantFail(lookupAndApply(
      JD, {recordAddr("addr_a", &A), recordAddr("addr_b", &B), RecordBoth}));

  EXPECT_EQ(A, ExecutorAddr(AddrAValue));
  EXPECT_EQ(B, ExecutorAddr(AddrBValue));
  EXPECT_EQ(C, ExecutorAddr(AddrAValue));
  EXPECT_EQ(D, ExecutorAddr(AddrBValue));

  cantFail(ES.endSession());
}

// If the lookup fails then no applicator runs: a failed lookup must not
// leave some variables written and others not.
TEST(LookupAndApplyTest, NoApplyOnLookupFailure) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.getBootstrapJITDylib();
  defineAddr(JD, "addr_a", ExecutorAddr(AddrAValue));

  ExecutorAddr A, B;
  // "addr_a" resolves, "absent" does not, so the whole lookup fails.
  EXPECT_THAT_ERROR(
      lookupAndApply(JD, {recordAddr("addr_a", &A), recordAddr("absent", &B)}),
      Failed());
  EXPECT_EQ(A, ExecutorAddr());
  EXPECT_EQ(B, ExecutorAddr());

  cantFail(ES.endSession());
}

// The asynchronous form delivers success once every applicator has run.
TEST(LookupAndApplyTest, Async) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.getBootstrapJITDylib();
  defineAddr(JD, "addr_a", ExecutorAddr(AddrAValue));

  ExecutorAddr A;
  std::promise<MSVCPError> P;
  auto F = P.get_future();
  lookupAndApply([&](Error Err) { P.set_value(std::move(Err)); }, JD,
                 {recordAddr("addr_a", &A)});
  EXPECT_THAT_ERROR(F.get(), Succeeded());
  EXPECT_EQ(A, ExecutorAddr(AddrAValue));

  cantFail(ES.endSession());
}

// A lookup failure is delivered through the callback rather than returned.
TEST(LookupAndApplyTest, AsyncFailure) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.getBootstrapJITDylib();

  ExecutorAddr A(AddrAValue);
  std::promise<MSVCPError> P;
  auto F = P.get_future();
  lookupAndApply([&](Error Err) { P.set_value(std::move(Err)); }, JD,
                 {recordAddr("absent", &A)});
  EXPECT_THAT_ERROR(F.get(), Failed());
  EXPECT_EQ(A, ExecutorAddr(AddrAValue));

  cantFail(ES.endSession());
}

// Weak references are resolved per symbol: within one lookup a symbol that is
// found is recorded, and one that is not is left null.
TEST(LookupAndApplyTest, WeaklyReferencedMixed) {
  ExecutionSession ES(cantFail(SelfExecutorProcessControl::Create()));
  auto &JD = ES.getBootstrapJITDylib();
  defineAddr(JD, "addr_a", ExecutorAddr(AddrAValue));

  ExecutorAddr A, B(AddrBValue);
  cantFail(lookupAndApply(
      JD,
      {recordAddr("addr_a", &A, SymbolLookupFlags::WeaklyReferencedSymbol),
       recordAddr("absent", &B, SymbolLookupFlags::WeaklyReferencedSymbol)}));
  EXPECT_EQ(A, ExecutorAddr(AddrAValue));
  EXPECT_EQ(B, ExecutorAddr());

  cantFail(ES.endSession());
}
