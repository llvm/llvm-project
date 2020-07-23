//===- llvm/unittest/Support/CrashRecoveryTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"
#include "gtest/gtest.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#endif

using namespace llvm;
using namespace llvm::sys;

static int GlobalInt = 0;
static void nullDeref() { *(volatile int *)0x10 = 0; }
static void incrementGlobal() { ++GlobalInt; }
static void llvmTrap() { LLVM_BUILTIN_TRAP; }
static void incrementGlobalWithParam(void *) { ++GlobalInt; }

TEST(CrashRecoveryTest, Basic) {
  llvm::CrashRecoveryContext::Enable();
  GlobalInt = 0;
  EXPECT_TRUE(CrashRecoveryContext().RunSafely(incrementGlobal));
  EXPECT_EQ(1, GlobalInt);
  EXPECT_FALSE(CrashRecoveryContext().RunSafely(nullDeref));
  EXPECT_FALSE(CrashRecoveryContext().RunSafely(llvmTrap));
}

struct IncrementGlobalCleanup : CrashRecoveryContextCleanup {
  IncrementGlobalCleanup(CrashRecoveryContext *CRC)
      : CrashRecoveryContextCleanup(CRC) {}
  void recoverResources() override { ++GlobalInt; }
};

static void noop() {}

TEST(CrashRecoveryTest, Cleanup) {
  llvm::CrashRecoveryContext::Enable();
  GlobalInt = 0;
  {
    CrashRecoveryContext CRC;
    CRC.registerCleanup(new IncrementGlobalCleanup(&CRC));
    EXPECT_TRUE(CRC.RunSafely(noop));
  } // run cleanups
  EXPECT_EQ(1, GlobalInt);

  GlobalInt = 0;
  {
    CrashRecoveryContext CRC;
    CRC.registerCleanup(new IncrementGlobalCleanup(&CRC));
    EXPECT_FALSE(CRC.RunSafely(nullDeref));
  } // run cleanups
  EXPECT_EQ(1, GlobalInt);
  llvm::CrashRecoveryContext::Disable();
}

TEST(CrashRecoveryTest, DumpStackCleanup) {
  SmallString<128> Filename;
  std::error_code EC = sys::fs::createTemporaryFile("crash", "test", Filename);
  EXPECT_FALSE(EC);
  sys::RemoveFileOnSignal(Filename);
  llvm::sys::AddSignalHandler(incrementGlobalWithParam, nullptr);
  GlobalInt = 0;
  llvm::CrashRecoveryContext::Enable();
  {
    CrashRecoveryContext CRC;
    CRC.DumpStackAndCleanupOnFailure = true;
    EXPECT_TRUE(CRC.RunSafely(noop));
  }
  EXPECT_TRUE(sys::fs::exists(Filename));
  EXPECT_EQ(GlobalInt, 0);
  {
    CrashRecoveryContext CRC;
    CRC.DumpStackAndCleanupOnFailure = true;
    EXPECT_FALSE(CRC.RunSafely(nullDeref));
    EXPECT_NE(CRC.RetCode, 0);
  }
  EXPECT_FALSE(sys::fs::exists(Filename));
  EXPECT_EQ(GlobalInt, 1);
  llvm::CrashRecoveryContext::Disable();
}

#ifdef _WIN32
static void raiseIt() {
  RaiseException(123, EXCEPTION_NONCONTINUABLE, 0, NULL);
}

TEST(CrashRecoveryTest, RaiseException) {
  llvm::CrashRecoveryContext::Enable();
  EXPECT_FALSE(CrashRecoveryContext().RunSafely(raiseIt));
}

static void outputString() {
  OutputDebugStringA("output for debugger\n");
}

TEST(CrashRecoveryTest, CallOutputDebugString) {
  llvm::CrashRecoveryContext::Enable();
  EXPECT_TRUE(CrashRecoveryContext().RunSafely(outputString));
}

#endif
