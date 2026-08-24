//===- unittests/Threading.cpp - Thread tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Threading.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ENABLE_THREADS
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/thread.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

#include <atomic>
#include <condition_variable>

using namespace llvm;

namespace {

static bool isThreadingSupportedArchAndOS() {
#if LLVM_ENABLE_THREADS
  Triple Host(Triple::normalize(sys::getProcessTriple()));

  // Initially this is only testing detection of the number of
  // physical cores, which is currently only supported/tested on
  // some systems.
  return (Host.isOSWindows() && llvm_is_multithreaded()) || Host.isOSDarwin() ||
         (Host.isX86() && Host.isOSLinux()) ||
         (Host.isOSLinux() && !Host.isAndroid()) ||
         (Host.isSystemZ() && Host.isOSzOS()) || Host.isOSAIX();
#else
  return false;
#endif
}

TEST(Threading, PhysicalConcurrency) {
  auto Num = heavyweight_hardware_concurrency();
  // Since Num is unsigned this will also catch us trying to
  // return -1.
  ASSERT_LE(Num.compute_thread_count(),
            hardware_concurrency().compute_thread_count());
}

TEST(Threading, NumPhysicalCoresSupported) {
  if (!isThreadingSupportedArchAndOS())
    GTEST_SKIP();
  int Num = get_physical_cores();
  ASSERT_GT(Num, 0);
}

TEST(Threading, NumPhysicalCoresUnsupported) {
  if (isThreadingSupportedArchAndOS())
    GTEST_SKIP();
  int Num = get_physical_cores();
  ASSERT_EQ(Num, -1);
}

#ifdef __linux__

class CgroupCpuCountTest : public testing::Test {
protected:
  CgroupCpuCountTest()
      : ProcSelfCgroup(Root.path("proc-self-cgroup")),
        ProcSelfMountInfo(Root.path("proc-self-mountinfo")),
        CgroupMount(Root.path("cgroup")) {
    EXPECT_FALSE(sys::fs::create_directories(CgroupMount));
  }

  void write(StringRef Path, StringRef Contents) {
    std::error_code EC;
    raw_fd_ostream OS(Path, EC);
    ASSERT_FALSE(EC) << EC.message();
    OS << Contents;
  }

  void configureV2(StringRef Membership, StringRef MountRoot = "/") {
    write(ProcSelfCgroup, (Twine("0::") + Membership + "\n").str());
    write(ProcSelfMountInfo,
          (Twine("29 23 0:26 ") + MountRoot + " " + CgroupMount +
           " rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n")
              .str());
  }

  void configureV1(StringRef Membership) {
    write(ProcSelfCgroup, (Twine("2:cpu,cpuacct:") + Membership + "\n").str());
    write(ProcSelfMountInfo,
          (Twine("30 23 0:27 / ") + CgroupMount +
           " rw,nosuid,nodev,noexec,relatime - cgroup cgroup "
           "rw,cpu,cpuacct\n")
              .str());
  }

  SmallString<128> path(StringRef Relative) {
    SmallString<128> Result(CgroupMount);
    sys::path::append(Result, Relative);
    return Result;
  }

  detail::CgroupFilePaths paths() const {
    detail::CgroupFilePaths Paths;
    Paths.ProcSelfCgroup = ProcSelfCgroup;
    Paths.ProcSelfMountInfo = ProcSelfMountInfo;
    Paths.V2CpuMax = "";
    Paths.V1CpuQuota = "";
    Paths.V1CpuPeriod = "";
    Paths.V1CpuAcctQuota = "";
    Paths.V1CpuAcctPeriod = "";
    return Paths;
  }

  unittest::TempDir Root{"llvm-cgroup-test", /*Unique=*/true};
  SmallString<128> ProcSelfCgroup;
  SmallString<128> ProcSelfMountInfo;
  SmallString<128> CgroupMount;
};

TEST_F(CgroupCpuCountTest, V2LeafQuota) {
  configureV2("/parent/child");
  ASSERT_FALSE(sys::fs::create_directories(path("parent/child")));
  write(path("parent/child/cpu.max"), "800000 100000\n");

  std::optional<unsigned> Count = detail::get_cgroup_cpu_count(paths());
  ASSERT_TRUE(Count);
  EXPECT_EQ(*Count, 8u);
}

TEST_F(CgroupCpuCountTest, V2TightestAncestorQuota) {
  configureV2("/parent/child");
  ASSERT_FALSE(sys::fs::create_directories(path("parent/child")));
  write(path("parent/child/cpu.max"), "max 100000\n");
  write(path("parent/cpu.max"), "750000 100000\n");
  write(path("cpu.max"), "1200000 100000\n");

  std::optional<unsigned> Count = detail::get_cgroup_cpu_count(paths());
  ASSERT_TRUE(Count);
  EXPECT_EQ(*Count, 8u);
}

TEST_F(CgroupCpuCountTest, V2MountRootMapsMembership) {
  configureV2("/tenant/action", "/tenant");
  ASSERT_FALSE(sys::fs::create_directories(path("action")));
  write(path("action/cpu.max"), "max 100000\n");
  write(path("cpu.max"), "600000 100000\n");

  std::optional<unsigned> Count = detail::get_cgroup_cpu_count(paths());
  ASSERT_TRUE(Count);
  EXPECT_EQ(*Count, 6u);
}

TEST_F(CgroupCpuCountTest, V2SkipsUnrelatedMount) {
  configureV2("/tenant/action", "/tenant");
  SmallString<128> UnrelatedMount(Root.path("unrelated-cgroup"));
  ASSERT_FALSE(sys::fs::create_directories(UnrelatedMount));
  write(ProcSelfMountInfo,
        (Twine("28 23 0:25 /other ") + UnrelatedMount +
         " rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n" +
         "29 23 0:26 /tenant " + CgroupMount +
         " rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n")
            .str());
  ASSERT_FALSE(sys::fs::create_directories(path("action")));
  write(path("action/cpu.max"), "400000 100000\n");

  std::optional<unsigned> Count = detail::get_cgroup_cpu_count(paths());
  ASSERT_TRUE(Count);
  EXPECT_EQ(*Count, 4u);
}

TEST_F(CgroupCpuCountTest, V1ParentQuota) {
  configureV1("/parent/child");
  ASSERT_FALSE(sys::fs::create_directories(path("parent/child")));
  write(path("parent/child/cpu.cfs_quota_us"), "-1\n");
  write(path("parent/child/cpu.cfs_period_us"), "100000\n");
  write(path("parent/cpu.cfs_quota_us"), "250000\n");
  write(path("parent/cpu.cfs_period_us"), "100000\n");

  std::optional<unsigned> Count = detail::get_cgroup_cpu_count(paths());
  ASSERT_TRUE(Count);
  EXPECT_EQ(*Count, 3u);
}

TEST_F(CgroupCpuCountTest, UnlimitedHierarchy) {
  configureV2("/");
  write(path("cpu.max"), "max 100000\n");

  EXPECT_FALSE(detail::get_cgroup_cpu_count(paths()));
}

#endif

#if LLVM_ENABLE_THREADS

class Notification {
public:
  void notify() {
    {
      std::lock_guard<std::mutex> Lock(M);
      Notified = true;
      // Broadcast with the lock held, so it's safe to destroy the Notification
      // after wait() returns.
      CV.notify_all();
    }
  }

  bool wait() {
    std::unique_lock<std::mutex> Lock(M);
    using steady_clock = std::chrono::steady_clock;
    auto Deadline = steady_clock::now() +
                    std::chrono::duration_cast<steady_clock::duration>(
                        std::chrono::duration<double>(5));
    return CV.wait_until(Lock, Deadline, [this] { return Notified; });
  }

private:
  bool Notified = false;
  mutable std::condition_variable CV;
  mutable std::mutex M;
};

TEST(Threading, RunOnThreadSyncAsync) {
  Notification ThreadStarted, ThreadAdvanced, ThreadFinished;

  auto ThreadFunc = [&] {
    ThreadStarted.notify();
    ASSERT_TRUE(ThreadAdvanced.wait());
    ThreadFinished.notify();
  };

  llvm::thread Thread(ThreadFunc);
  Thread.detach();
  ASSERT_TRUE(ThreadStarted.wait());
  ThreadAdvanced.notify();
  ASSERT_TRUE(ThreadFinished.wait());
}

TEST(Threading, RunOnThreadSync) {
  std::atomic_bool Executed(false);
  llvm::thread Thread(
      [](void *Arg) { *static_cast<std::atomic_bool *>(Arg) = true; },
      &Executed);
  Thread.join();
  ASSERT_EQ(Executed, true);
}

#if defined(__APPLE__)
TEST(Threading, AppleStackSize) {
  llvm::thread Thread([] {
    volatile unsigned char Var[8 * 1024 * 1024 - 10240];
    Var[0] = 0xff;
    ASSERT_EQ(Var[0], 0xff);
  });
  Thread.join();
}
#endif
#endif

} // namespace
