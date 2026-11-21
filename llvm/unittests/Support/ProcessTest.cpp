//===- unittest/Support/ProcessTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Process.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ENABLE_THREADS
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
// psapi.h needs windows.h.
#include <psapi.h>
#endif

namespace {

using namespace llvm;
using namespace sys;

TEST(ProcessTest, GetProcessIdTest) {
  const Process::Pid pid = Process::getProcessId();

#ifdef _WIN32
  EXPECT_EQ((DWORD)pid, ::GetCurrentProcessId());
#else
  EXPECT_EQ(pid, ::getpid());
#endif
}

TEST(ProcessTest, GetRandomNumberTest) {
  const unsigned r1 = Process::GetRandomNumber();
  const unsigned r2 = Process::GetRandomNumber();
  // It should be extremely unlikely that both r1 and r2 are 0.
  EXPECT_NE((r1 | r2), 0u);
}

#ifdef _MSC_VER
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

#if HAVE_SETENV || _MSC_VER
TEST(ProcessTest, Basic) {
  setenv("__LLVM_TEST_ENVIRON_VAR__", "abc", true);
  std::optional<std::string> val(Process::GetEnv("__LLVM_TEST_ENVIRON_VAR__"));
  EXPECT_TRUE(val.has_value());
  EXPECT_STREQ("abc", val->c_str());
}

TEST(ProcessTest, None) {
  std::optional<std::string> val(
      Process::GetEnv("__LLVM_TEST_ENVIRON_NO_SUCH_VAR__"));
  EXPECT_FALSE(val.has_value());
}
#endif

#ifdef _WIN32

TEST(ProcessTest, EmptyVal) {
  SetEnvironmentVariableA("__LLVM_TEST_ENVIRON_VAR__", "");
  std::optional<std::string> val(Process::GetEnv("__LLVM_TEST_ENVIRON_VAR__"));
  EXPECT_TRUE(val.has_value());
  EXPECT_STREQ("", val->c_str());
}

TEST(ProcessTest, Wchar) {
  SetEnvironmentVariableW(L"__LLVM_TEST_ENVIRON_VAR__", L"abcdefghijklmnopqrs");
  std::optional<std::string> val(Process::GetEnv("__LLVM_TEST_ENVIRON_VAR__"));
  EXPECT_TRUE(val.has_value());
  EXPECT_STREQ("abcdefghijklmnopqrs", val->c_str());
}
#endif

class PageSizeTest : public testing::Test {
  Triple Host;

protected:
  PageSizeTest() : Host(Triple::normalize(sys::getProcessTriple())) {}

  bool isSupported() const {
    // For now just on X86-64 and Aarch64. This can be expanded in the future.
    return (Host.getArch() == Triple::x86_64 ||
            Host.getArch() == Triple::aarch64) &&
           Host.getOS() == Triple::Linux;
  }

  bool pageSizeAsExpected(unsigned PageSize) const {
    switch (Host.getArch()) {
    case Triple::x86_64:
      return PageSize == 4096;
    case Triple::aarch64:
      // supported granule sizes are 4k, 16k and 64k
      return PageSize == 4096 || PageSize == 16384 || PageSize == 65536;
    default:
      llvm_unreachable("unexpected arch!");
    }
  }
};

TEST_F(PageSizeTest, PageSize) {
  if (!isSupported())
    GTEST_SKIP();

  llvm::Expected<unsigned> Result = llvm::sys::Process::getPageSize();
  ASSERT_FALSE(!Result);
  ASSERT_TRUE(pageSizeAsExpected(*Result));
}

#if LLVM_ENABLE_THREADS
TEST(ProcessTest, ReleaseHeapCaches) {
  // Callable on any thread, whether or not the allocator caches anything.
  Process::releaseHeapCaches(/*IncludeShared=*/false);
  Process::releaseHeapCaches(/*IncludeShared=*/true);
  std::thread([] {
    Process::releaseHeapCaches(/*IncludeShared=*/true);
  }).join();
  void *P = std::malloc(100);
  EXPECT_NE(P, nullptr);
  std::free(P);
}

#if defined(_WIN32) && defined(LLVM_ENABLE_RPMALLOC)
// Threads allocate blocks of all size classes, free each other's blocks and
// release heap caches concurrently. Checks that no block is corrupted.
TEST(ProcessTest, ReleaseHeapCachesConcurrent) {
  constexpr unsigned Threads = 8, Iterations = 2000;
  struct Block {
    unsigned char *Data;
    size_t Size;
    unsigned char Fill;
  };
  std::mutex Mu;
  std::vector<Block> Shared;
  std::atomic<unsigned> Corrupted(0);
  auto Check = [&](const Block &B) {
    if (B.Data[0] != B.Fill || B.Data[B.Size / 2] != B.Fill ||
        B.Data[B.Size - 1] != B.Fill)
      ++Corrupted;
    std::free(B.Data);
  };
  std::vector<std::thread> Workers;
  for (unsigned T = 0; T < Threads; ++T)
    Workers.emplace_back([&, T] {
      std::mt19937 Rng(T);
      for (unsigned I = 0; I < Iterations; ++I) {
        // Mostly small blocks, some medium, large and huge ones.
        size_t MaxSize = 256;
        if (I % 50 == 0)
          MaxSize = 8 << 20;
        else if (I % 10 == 0)
          MaxSize = 2 << 20;
        else if (I % 4 == 0)
          MaxSize = 32 << 10;
        size_t Size = 1 + Rng() % MaxSize;
        Block B{static_cast<unsigned char *>(std::malloc(Size)), Size,
                static_cast<unsigned char>(Rng())};
        ASSERT_NE(B.Data, nullptr);
        std::memset(B.Data, B.Fill, Size);
        std::optional<Block> Other;
        {
          std::lock_guard<std::mutex> Lock(Mu);
          Shared.push_back(B);
          if (Shared.size() > 64) {
            size_t Pick = Rng() % Shared.size();
            Other = Shared[Pick];
            Shared[Pick] = Shared.back();
            Shared.pop_back();
          }
        }
        if (Other)
          Check(*Other);
        if (I % 256 == 0)
          Process::releaseHeapCaches(/*IncludeShared=*/(I / 256) % 2 == 1);
      }
      Process::releaseHeapCaches(/*IncludeShared=*/true);
    });
  for (std::thread &W : Workers)
    W.join();
  for (const Block &B : Shared)
    Check(B);
  EXPECT_EQ(Corrupted.load(), 0u);
}

// The commit charge of the process. It changes when memory is committed or
// released, not when pages are touched.
int64_t committedBytes() {
  PROCESS_MEMORY_COUNTERS_EX C = {};
  EXPECT_TRUE(::GetProcessMemoryInfo(
      ::GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&C),
      sizeof(C)));
  return static_cast<int64_t>(C.PrivateUsage);
}

// Blocks freed by the thread that allocated them stay in rpmalloc's span
// caches. One call has to return practically all of them.
TEST(ProcessTest, ReleaseHeapCachesFreedBlocks) {
  constexpr size_t Size = 1 << 20, Count = 256;
  // Start with empty caches, so that the blocks need new memory.
  Process::releaseHeapCaches(/*IncludeShared=*/true);
  int64_t Before = committedBytes();
  std::vector<void *> Blocks;
  for (size_t I = 0; I < Count; ++I)
    Blocks.push_back(std::malloc(Size));
  // rpmalloc commits the memory it maps for the blocks.
  EXPECT_GE(committedBytes() - Before, static_cast<int64_t>(Size * Count / 2));
  for (void *B : Blocks)
    std::free(B);
  Process::releaseHeapCaches(/*IncludeShared=*/true);
  EXPECT_LT(committedBytes() - Before, static_cast<int64_t>(Size * Count / 16));
}

// rpmalloc only completes a free of a block that another thread allocated on
// the allocating thread, so the block stays committed until that thread
// allocates again or calls releaseHeapCaches.
TEST(ProcessTest, ReleaseHeapCachesCrossThreadFree) {
  constexpr size_t Size = 256 << 20;
  int64_t AfterFree = 0, AfterRelease = 0;
  std::thread([&] {
    void *Block = std::malloc(Size);
    std::thread([Block] { std::free(Block); }).join();
    AfterFree = committedBytes();
    Process::releaseHeapCaches(/*IncludeShared=*/false);
    AfterRelease = committedBytes();
  }).join();
  EXPECT_GE(AfterFree - AfterRelease, static_cast<int64_t>(Size / 2));
}
#endif // _WIN32 && LLVM_ENABLE_RPMALLOC
#endif // LLVM_ENABLE_THREADS

} // end anonymous namespace
