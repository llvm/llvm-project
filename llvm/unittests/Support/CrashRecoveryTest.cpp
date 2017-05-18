//===- llvm/unittest/Support/CrashRecoveryTest.cpp ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Compiler.h"
#include "gtest/gtest.h"

#ifdef LLVM_ON_WIN32
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#endif

extern "C" const char *__asan_default_options() {
  return "allow_user_segv_handler=1";
}

using namespace llvm;
using namespace llvm::sys;

static int GlobalInt = 0;
static void nullDeref() { *(volatile int *)0x10 = 0; }
static void incrementGlobal() { ++GlobalInt; }
static void llvmTrap() { LLVM_BUILTIN_TRAP; }

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
  virtual void recoverResources() { ++GlobalInt; }
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
}

#ifdef LLVM_ON_WIN32
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
