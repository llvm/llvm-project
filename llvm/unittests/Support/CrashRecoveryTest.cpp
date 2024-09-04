//===- llvm/unittest/Support/CrashRecoveryTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#endif

#ifdef LLVM_ON_UNIX
#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif
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

TEST(CrashRecoveryTest, LimitedStackTrace) {
  // FIXME: Handle "Depth" parameter in PrintStackTrace() function
  // to print stack trace upto a specified Depth.
  if (Triple(sys::getProcessTriple()).isOSWindows())
    GTEST_SKIP();
  std::string Res;
  llvm::raw_string_ostream RawStream(Res);
  PrintStackTrace(RawStream, 1);
  std::string Str = RawStream.str();
  EXPECT_EQ(std::string::npos, Str.find("#1"));
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

TEST(CrashRecoveryTest, Abort) {
  llvm::CrashRecoveryContext::Enable();
  auto A = []() { abort(); };
  EXPECT_FALSE(CrashRecoveryContext().RunSafely(A));
  // Test a second time to ensure we reinstall the abort signal handler.
  EXPECT_FALSE(CrashRecoveryContext().RunSafely(A));
}
#endif

// Specifically ensure that programs that signal() or abort() through the
// CrashRecoveryContext can re-throw again their signal, so that `not --crash`
// succeeds.
#ifdef LLVM_ON_UNIX
// See llvm/utils/unittest/UnitTestMain/TestMain.cpp
extern const char *TestMainArgv0;

// Just a reachable symbol to ease resolving of the executable's path.
static cl::opt<std::string> CrashTestStringArg1("crash-test-string-arg1");

TEST(CrashRecoveryTest, UnixCRCReturnCode) {
  using namespace llvm::sys;
  if (getenv("LLVM_CRC_UNIXCRCRETURNCODE")) {
    llvm::CrashRecoveryContext::Enable();
    CrashRecoveryContext CRC;
    // This path runs in a subprocess that exits by signalling, so don't use
    // the googletest macros to verify things as they won't report properly.
    if (CRC.RunSafely(abort))
      llvm_unreachable("RunSafely returned true!");
    if (CRC.RetCode != 128 + SIGABRT)
      llvm_unreachable("Unexpected RetCode!");
    // re-throw signal
    llvm::sys::unregisterHandlers();
    raise(CRC.RetCode - 128);
    llvm_unreachable("Should have exited already!");
  }

  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &CrashTestStringArg1);
  StringRef argv[] = {
      Executable, "--gtest_filter=CrashRecoveryTest.UnixCRCReturnCode"};

  // Add LLVM_CRC_UNIXCRCRETURNCODE to the environment of the child process.
  int Res = setenv("LLVM_CRC_UNIXCRCRETURNCODE", "1", 0);
  ASSERT_EQ(Res, 0);

  Res = unsetenv("GTEST_SHARD_INDEX");
  ASSERT_EQ(Res, 0);
  Res = unsetenv("GTEST_TOTAL_SHARDS");
  ASSERT_EQ(Res, 0);

  std::string Error;
  bool ExecutionFailed;
  int RetCode = ExecuteAndWait(Executable, argv, {}, {}, 0, 0, &Error,
                               &ExecutionFailed);
  ASSERT_EQ(-2, RetCode);
}
#endif
