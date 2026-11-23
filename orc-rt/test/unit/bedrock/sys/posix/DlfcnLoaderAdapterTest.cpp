//===- DlfcnLoaderAdapterTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's bedrock/sys/posix/DlfcnLoaderAdapter.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sys/posix/DlfcnLoaderAdapter.h"
#include "orc-rt/support/move_only_function.h"
#include "orc-rt/support/scope_exit.h"

#include "ErrorMatchers.h"
#include "gtest/gtest.h"

#include <string>
#include <thread>
#include <vector>

using namespace orc_rt;
using namespace orc_rt::test;

namespace {

/// A DlfcnLoaderAdapter whose operations record their arguments, and either
/// succeed with configurable results or fail with a configurable message.
class MockAdapter : public DlfcnLoaderAdapter {
public:
  MockAdapter(SimpleSymbolTable &ST, Error &Err)
      : DlfcnLoaderAdapter(ST, Err) {}

  /// If non-null, every operation fails with this message.
  const char *FailWith = nullptr;

  void *OpenResult = reinterpret_cast<void *>(0x1000);
  void *SymResult = reinterpret_cast<void *>(0x2000);
  bool AddrFound = true;

  const char *LastPath = nullptr;
  int LastMode = 0;
  void *LastHandle = nullptr;
  const char *LastSymbol = nullptr;
  const void *LastAddr = nullptr;

private:
  Error failure() { return make_error<StringError>(FailWith); }

  Expected<void *> dlopenImpl(const char *Path, int Mode) noexcept override {
    LastPath = Path;
    LastMode = Mode;
    if (FailWith)
      return failure();
    return OpenResult;
  }

  Error dlcloseImpl(void *Handle) noexcept override {
    LastHandle = Handle;
    if (FailWith)
      return failure();
    return Error::success();
  }

  Expected<void *> dlsymImpl(void *Handle,
                             const char *Symbol) noexcept override {
    LastHandle = Handle;
    LastSymbol = Symbol;
    if (FailWith)
      return failure();
    return SymResult;
  }

  Expected<bool> dladdrImpl(const void *Addr,
                            orc_rt_Dl_info *Info) noexcept override {
    LastAddr = Addr;
    if (FailWith)
      return failure();
    if (!AddrFound)
      return false;
    Info->dli_fname = "mock-image";
    Info->dli_fbase = reinterpret_cast<void *>(0x3000);
    Info->dli_sname = "mock-symbol";
    Info->dli_saddr = reinterpret_cast<void *>(0x3100);
    return true;
  }
};

/// The dlerror state is per-thread, and may be shared between adapters, so
/// each test starts by clearing anything an earlier test left on this thread.
/// Any successful operation does that.
class DlfcnLoaderAdapterTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_THAT_ERROR(std::move(Err), Succeeded());
    A.dlclose(nullptr);
  }

  SimpleSymbolTable ST;
  Error Err = Error::success();
  MockAdapter A{ST, Err};
};

} // namespace

// --- Instance symbol ---

TEST_F(DlfcnLoaderAdapterTest, RegistersInstanceSymbol) {
  // The symbol's address must be the adapter as a DlfcnLoaderAdapter, since
  // JIT'd code passes it through the C API as an orc_rt_DlfcnLoaderAdapterRef.
  SymbolNameSpec Name = SymbolNameSpec::c("orc_rt_DlfcnLoaderAdapter_Instance");
  ASSERT_EQ(ST.count(Name), 1U);
  EXPECT_EQ(ST.at(Name),
            static_cast<const void *>(static_cast<DlfcnLoaderAdapter *>(&A)));
}

TEST_F(DlfcnLoaderAdapterTest, SecondAdapterInSameTableFails) {
  Error Err2 = Error::success();
  MockAdapter B(ST, Err2);
  EXPECT_THAT_ERROR(std::move(Err2), Failed());
  EXPECT_EQ(ST.at(SymbolNameSpec::c("orc_rt_DlfcnLoaderAdapter_Instance")),
            static_cast<const void *>(static_cast<DlfcnLoaderAdapter *>(&A)));
}

// --- Forwarding to the *Impl methods ---

TEST_F(DlfcnLoaderAdapterTest, DlopenForwards) {
  const char *Path = "libfoo.so";
  EXPECT_EQ(A.dlopen(Path, ORC_RT_RTLD_NOW), A.OpenResult);
  EXPECT_EQ(A.LastPath, Path);
  EXPECT_EQ(A.LastMode, ORC_RT_RTLD_NOW);
}

TEST_F(DlfcnLoaderAdapterTest, DlcloseForwards) {
  void *Handle = reinterpret_cast<void *>(0x1234);
  EXPECT_EQ(A.dlclose(Handle), 0);
  EXPECT_EQ(A.LastHandle, Handle);
}

TEST_F(DlfcnLoaderAdapterTest, DlsymForwards) {
  void *Handle = reinterpret_cast<void *>(0x1234);
  const char *Symbol = "foo";
  EXPECT_EQ(A.dlsym(Handle, Symbol), A.SymResult);
  EXPECT_EQ(A.LastHandle, Handle);
  EXPECT_EQ(A.LastSymbol, Symbol);
}

TEST_F(DlfcnLoaderAdapterTest, DladdrForwards) {
  const void *Addr = reinterpret_cast<const void *>(0x3110);
  orc_rt_Dl_info Info = {};
  EXPECT_NE(A.dladdr(Addr, &Info), 0);
  EXPECT_EQ(A.LastAddr, Addr);
  EXPECT_STREQ(Info.dli_fname, "mock-image");
  EXPECT_EQ(Info.dli_fbase, reinterpret_cast<void *>(0x3000));
  EXPECT_STREQ(Info.dli_sname, "mock-symbol");
  EXPECT_EQ(Info.dli_saddr, reinterpret_cast<void *>(0x3100));
}

TEST_F(DlfcnLoaderAdapterTest, FailuresReturnPOSIXFailureValues) {
  orc_rt_Dl_info Info = {};
  A.FailWith = "failed";
  EXPECT_EQ(A.dlopen("libfoo.so", ORC_RT_RTLD_NOW), nullptr);
  EXPECT_EQ(A.dlclose(reinterpret_cast<void *>(0x1234)), -1);
  EXPECT_EQ(A.dlsym(reinterpret_cast<void *>(0x1234), "foo"), nullptr);
  EXPECT_EQ(A.dladdr(reinterpret_cast<const void *>(0x3110), &Info), 0);
}

// --- dlerror ---

TEST_F(DlfcnLoaderAdapterTest, DlerrorIsNullWithoutError) {
  EXPECT_EQ(A.dlerror(), nullptr);
}

TEST_F(DlfcnLoaderAdapterTest, DlerrorReportsMessageOnce) {
  A.FailWith = "no such file";
  A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
  EXPECT_STREQ(A.dlerror(), "no such file");
  EXPECT_EQ(A.dlerror(), nullptr);
}

TEST_F(DlfcnLoaderAdapterTest, EmptyErrorMessageIsReplaced) {
  // dlerror must report something after a failure.
  A.FailWith = "";
  EXPECT_EQ(A.dlopen("libfoo.so", ORC_RT_RTLD_NOW), nullptr);
  const char *Msg = A.dlerror();
  ASSERT_NE(Msg, nullptr);
  EXPECT_STRNE(Msg, "");
}

TEST_F(DlfcnLoaderAdapterTest, DlsymNullResultIsNotAnError) {
  // A symbol can legitimately resolve to null: dlerror is how callers tell
  // that apart from a failed lookup. The earlier failure leaves an unread
  // message, which the successful lookup should clear.
  A.FailWith = "earlier failure";
  A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
  A.FailWith = nullptr;
  A.SymResult = nullptr;
  EXPECT_EQ(A.dlsym(reinterpret_cast<void *>(0x1234), "weak_undefined"),
            nullptr);
  EXPECT_EQ(A.dlerror(), nullptr);
}

TEST_F(DlfcnLoaderAdapterTest, SuccessClearsUnreadError) {
  // As with the native dlerror on Darwin and glibc.
  A.FailWith = "no such file";
  A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
  A.FailWith = nullptr;
  A.dlsym(reinterpret_cast<void *>(0x1234), "foo");
  EXPECT_EQ(A.dlerror(), nullptr);
}

TEST_F(DlfcnLoaderAdapterTest, DladdrNotFoundIsNotAnError) {
  orc_rt_Dl_info Info = {};
  A.AddrFound = false;
  EXPECT_EQ(A.dladdr(reinterpret_cast<const void *>(0x3110), &Info), 0);
  EXPECT_EQ(A.dlerror(), nullptr);
}

TEST_F(DlfcnLoaderAdapterTest, DladdrLeavesUnreadErrorAlone) {
  // As with native dladdr: a completed lookup, whether or not it finds the
  // address, neither clears nor replaces an unread error.
  orc_rt_Dl_info Info = {};
  A.FailWith = "earlier failure";
  A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
  A.FailWith = nullptr;
  EXPECT_NE(A.dladdr(reinterpret_cast<const void *>(0x3110), &Info), 0);
  A.AddrFound = false;
  EXPECT_EQ(A.dladdr(reinterpret_cast<const void *>(0x3110), &Info), 0);
  EXPECT_STREQ(A.dlerror(), "earlier failure");
}

TEST_F(DlfcnLoaderAdapterTest, DladdrLookupFailureSetsError) {
  orc_rt_Dl_info Info = {};
  A.FailWith = "symbol table unavailable";
  EXPECT_EQ(A.dladdr(reinterpret_cast<const void *>(0x3110), &Info), 0);
  EXPECT_STREQ(A.dlerror(), "symbol table unavailable");
}

TEST_F(DlfcnLoaderAdapterTest, LatestErrorIsReported) {
  A.FailWith = "first";
  A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
  A.FailWith = "second";
  A.dlsym(reinterpret_cast<void *>(0x1234), "foo");
  EXPECT_STREQ(A.dlerror(), "second");
}

TEST_F(DlfcnLoaderAdapterTest, ErrorStateIsPerThread) {
  A.FailWith = "main thread";
  A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);

  bool OtherSawNoError = false;
  std::string OtherMsg;
  std::thread T([&] {
    OtherSawNoError = A.dlerror() == nullptr;
    A.FailWith = "other thread";
    A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
    if (const char *M = A.dlerror())
      OtherMsg = M;
  });
  T.join();

  EXPECT_TRUE(OtherSawNoError);
  EXPECT_EQ(OtherMsg, "other thread");
  EXPECT_STREQ(A.dlerror(), "main thread");
}

TEST_F(DlfcnLoaderAdapterTest, DlerrorResultUnaffectedByOtherThreads) {
  A.FailWith = "main thread";
  A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
  const char *Msg = A.dlerror();
  ASSERT_STREQ(Msg, "main thread");

  std::vector<std::thread> Threads;
  for (int I = 0; I != 8; ++I)
    Threads.emplace_back([] {
      // Each thread uses its own adapter, so that the threads don't race on
      // the mock's recorded arguments.
      SimpleSymbolTable ST;
      Error Err = Error::success();
      MockAdapter B(ST, Err);
      EXPECT_THAT_ERROR(std::move(Err), Succeeded());
      B.FailWith = "other thread, with a message long enough to allocate";
      for (int J = 0; J != 100; ++J)
        B.dlsym(nullptr, "foo");
    });
  for (auto &T : Threads)
    T.join();

  EXPECT_STREQ(Msg, "main thread");
}

// --- Thread exit ---
//
// These tests run code from a thread_local destructor that runs after the
// adapter's per-thread error state has been destroyed. thread_locals are
// destroyed in reverse order of construction, so a thread_local scope_exit
// constructed before the thread first uses an adapter runs after the error
// state has been destroyed.

namespace {

/// Runs Body on a new thread, then OnThreadExit while that thread exits, after
/// everything Body constructed has been destroyed.
void runWithTeardownProbe(move_only_function<void()> Body,
                          move_only_function<void()> OnThreadExit) {
  std::thread T([&] {
    // Constructed first, so destroyed last. One instance per thread.
    static thread_local scope_exit RunOnThreadExit(std::move(OnThreadExit));
    Body();
  });
  T.join();
}

bool isThreadExitFallback(const std::string &Msg) {
  return Msg.find("thread exit") != std::string::npos;
}

} // namespace

TEST_F(DlfcnLoaderAdapterTest, ErrorAfterStateDestroyedReportedOnce) {
  void *Result = reinterpret_cast<void *>(1);
  std::string Msg;
  bool SecondReadNull = false;
  A.FailWith = "failed";
  runWithTeardownProbe([&] { A.dlopen("libfoo.so", ORC_RT_RTLD_NOW); },
                       [&] {
                         Result = A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
                         if (const char *M = A.dlerror())
                           Msg = M;
                         SecondReadNull = A.dlerror() == nullptr;
                       });
  EXPECT_EQ(Result, nullptr);
  EXPECT_TRUE(isThreadExitFallback(Msg)) << Msg;
  EXPECT_TRUE(SecondReadNull);
}

TEST_F(DlfcnLoaderAdapterTest, ReadErrorNotReportedAgainDuringThreadExit) {
  std::string FirstRead;
  bool TeardownReadNull = false;
  A.FailWith = "failed";
  runWithTeardownProbe(
      [&] {
        A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
        if (const char *M = A.dlerror())
          FirstRead = M;
      },
      [&] { TeardownReadNull = A.dlerror() == nullptr; });
  EXPECT_EQ(FirstRead, "failed");
  EXPECT_TRUE(TeardownReadNull);
}

TEST_F(DlfcnLoaderAdapterTest, UnreadErrorReportedAsFallbackDuringThreadExit) {
  std::string Msg;
  bool SecondReadNull = false;
  A.FailWith = "failed";
  runWithTeardownProbe([&] { A.dlopen("libfoo.so", ORC_RT_RTLD_NOW); },
                       [&] {
                         if (const char *M = A.dlerror())
                           Msg = M;
                         SecondReadNull = A.dlerror() == nullptr;
                       });
  EXPECT_TRUE(isThreadExitFallback(Msg)) << Msg;
  EXPECT_TRUE(SecondReadNull);
}

TEST_F(DlfcnLoaderAdapterTest, SuccessDuringThreadExitClearsLostError) {
  bool ReadNull = false;
  runWithTeardownProbe(
      [&] { A.dlclose(nullptr); }, // Construct the error state.
      [&] {
        A.FailWith = "failed";
        A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
        A.FailWith = nullptr;
        A.dlclose(nullptr);
        ReadNull = A.dlerror() == nullptr;
      });
  EXPECT_TRUE(ReadNull);
}

TEST_F(DlfcnLoaderAdapterTest, FirstUseDuringThreadExitStoresMessage) {
  // The thread doesn't use an adapter until thread exit, so the error state is
  // constructed then, and holds the real message.
  std::string Msg;
  bool SecondReadNull = false;
  A.FailWith = "failed";
  runWithTeardownProbe([] {},
                       [&] {
                         A.dlopen("libfoo.so", ORC_RT_RTLD_NOW);
                         if (const char *M = A.dlerror())
                           Msg = M;
                         SecondReadNull = A.dlerror() == nullptr;
                       });
  EXPECT_EQ(Msg, "failed");
  EXPECT_TRUE(SecondReadNull);
}

// --- C API ---

// Defined in DlfcnLoaderAdapterCTest.c, which is compiled as C.
extern "C" {
void *
orc_rt_test_DlfcnLoaderAdapter_dlopenFromC(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                           const char *Path, int Mode);
int orc_rt_test_DlfcnLoaderAdapter_dlcloseFromC(
    orc_rt_DlfcnLoaderAdapterRef Ldr, void *Handle);
void *
orc_rt_test_DlfcnLoaderAdapter_dlsymFromC(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                          void *Handle, const char *Symbol);
int orc_rt_test_DlfcnLoaderAdapter_dladdrFromC(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                               const void *Addr,
                                               orc_rt_Dl_info *Info);
char *
orc_rt_test_DlfcnLoaderAdapter_dlerrorFromC(orc_rt_DlfcnLoaderAdapterRef Ldr);
}

TEST_F(DlfcnLoaderAdapterTest, CAPIForwards) {
  orc_rt_DlfcnLoaderAdapterRef Ref = wrap(&A);
  void *Handle = reinterpret_cast<void *>(0x1234);
  const void *Addr = reinterpret_cast<const void *>(0x3110);
  orc_rt_Dl_info Info = {};

  EXPECT_EQ(orc_rt_test_DlfcnLoaderAdapter_dlopenFromC(Ref, "libfoo.so",
                                                       ORC_RT_RTLD_NOW),
            A.OpenResult);
  EXPECT_STREQ(A.LastPath, "libfoo.so");
  EXPECT_EQ(orc_rt_test_DlfcnLoaderAdapter_dlcloseFromC(Ref, Handle), 0);
  EXPECT_EQ(A.LastHandle, Handle);
  EXPECT_EQ(orc_rt_test_DlfcnLoaderAdapter_dlsymFromC(Ref, Handle, "foo"),
            A.SymResult);
  EXPECT_STREQ(A.LastSymbol, "foo");
  EXPECT_NE(orc_rt_test_DlfcnLoaderAdapter_dladdrFromC(Ref, Addr, &Info), 0);
  EXPECT_EQ(A.LastAddr, Addr);
  EXPECT_STREQ(Info.dli_sname, "mock-symbol");

  A.FailWith = "failed";
  EXPECT_EQ(orc_rt_test_DlfcnLoaderAdapter_dlopenFromC(Ref, "libfoo.so",
                                                       ORC_RT_RTLD_NOW),
            nullptr);
  EXPECT_STREQ(orc_rt_test_DlfcnLoaderAdapter_dlerrorFromC(Ref), "failed");
}
