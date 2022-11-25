//===-- llvm/Support/Threading.cpp- Control multithreading mode --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helper functions for running LLVM in a multi-threaded
// environment.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Threading.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

using namespace llvm;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

#if LLVM_ENABLE_THREADS == 0 ||                                                \
    (!defined(_WIN32) && !defined(HAVE_PTHREAD_H))
uint64_t llvm::get_threadid() { return 0; }

uint32_t llvm::get_max_thread_name_length() { return 0; }

void llvm::set_thread_name(const Twine &Name) {}

void llvm::get_thread_name(SmallVectorImpl<char> &Name) { Name.clear(); }

llvm::BitVector llvm::get_thread_affinity_mask() { return {}; }

unsigned llvm::ThreadPoolStrategy::compute_thread_count() const {
  // When threads are disabled, ensure clients will loop at least once.
  return 1;
}

#else

static int computeHostNumHardwareThreads();

unsigned llvm::ThreadPoolStrategy::compute_thread_count() const {
  int MaxThreadCount =
      UseHyperThreads ? computeHostNumHardwareThreads() : get_physical_cores();
  if (MaxThreadCount <= 0)
    MaxThreadCount = 1;
  if (ThreadsRequested == 0)
    return MaxThreadCount;
  if (!Limit)
    return ThreadsRequested;
  return std::min((unsigned)MaxThreadCount, ThreadsRequested);
}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Threading.inc"
#endif
#ifdef _WIN32
#include "Windows/Threading.inc"
#endif

// Must be included after Threading.inc to provide definition for llvm::thread
// because FreeBSD's condvar.h (included by user.h) misuses the "thread"
// keyword.
#include "llvm/Support/thread.h"

#if defined(__APPLE__)
  // Darwin's default stack size for threads except the main one is only 512KB,
  // which is not enough for some/many normal LLVM compilations. This implements
  // the same interface as std::thread but requests the same stack size as the
  // main thread (8MB) before creation.
const llvm::Optional<unsigned> llvm::thread::DefaultStackSize = 8 * 1024 * 1024;
#else
const llvm::Optional<unsigned> llvm::thread::DefaultStackSize;
#endif


#endif

Optional<ThreadPoolStrategy>
llvm::get_threadpool_strategy(StringRef Num, ThreadPoolStrategy Default) {
  if (Num == "all")
    return llvm::hardware_concurrency();
  if (Num.empty())
    return Default;
  unsigned V;
  if (Num.getAsInteger(10, V))
    return None; // malformed 'Num' value
  if (V == 0)
    return Default;

  // Do not take the Default into account. This effectively disables
  // heavyweight_hardware_concurrency() if the user asks for any number of
  // threads on the cmd-line.
  ThreadPoolStrategy S = llvm::hardware_concurrency();
  S.ThreadsRequested = V;
  return S;
}

#if defined(__linux__) && (defined(__i386__) || defined(__x86_64__))
// On Linux, the number of physical cores can be computed from /proc/cpuinfo,
// using the number of unique physical/core id pairs. The following
// implementation reads the /proc/cpuinfo format on an x86_64 system.
static int computeHostNumPhysicalCores() {
  // Enabled represents the number of physical id/core id pairs with at least
  // one processor id enabled by the CPU affinity mask.
  cpu_set_t Affinity, Enabled;
  if (sched_getaffinity(0, sizeof(Affinity), &Affinity) != 0)
    return -1;
  CPU_ZERO(&Enabled);

  // Read /proc/cpuinfo as a stream (until EOF reached). It cannot be
  // mmapped because it appears to have 0 size.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
      llvm::MemoryBuffer::getFileAsStream("/proc/cpuinfo");
  if (std::error_code EC = Text.getError()) {
    llvm::errs() << "Can't read "
                 << "/proc/cpuinfo: " << EC.message() << "\n";
    return -1;
  }
  SmallVector<StringRef, 8> strs;
  (*Text)->getBuffer().split(strs, "\n", /*MaxSplit=*/-1,
                             /*KeepEmpty=*/false);
  int CurProcessor = -1;
  int CurPhysicalId = -1;
  int CurSiblings = -1;
  int CurCoreId = -1;
  for (StringRef Line : strs) {
    std::pair<StringRef, StringRef> Data = Line.split(':');
    auto Name = Data.first.trim();
    auto Val = Data.second.trim();
    // These fields are available if the kernel is configured with CONFIG_SMP.
    if (Name == "processor")
      Val.getAsInteger(10, CurProcessor);
    else if (Name == "physical id")
      Val.getAsInteger(10, CurPhysicalId);
    else if (Name == "siblings")
      Val.getAsInteger(10, CurSiblings);
    else if (Name == "core id") {
      Val.getAsInteger(10, CurCoreId);
      // The processor id corresponds to an index into cpu_set_t.
      if (CPU_ISSET(CurProcessor, &Affinity))
        CPU_SET(CurPhysicalId * CurSiblings + CurCoreId, &Enabled);
    }
  }
  return CPU_COUNT(&Enabled);
}
#elif defined(__linux__) && defined(__s390x__)
static int computeHostNumPhysicalCores() {
  return sysconf(_SC_NPROCESSORS_ONLN);
}
#elif defined(__linux__) && !defined(__ANDROID__)
static int computeHostNumPhysicalCores() {
  cpu_set_t Affinity;
  if (sched_getaffinity(0, sizeof(Affinity), &Affinity) == 0)
    return CPU_COUNT(&Affinity);

  // The call to sched_getaffinity() may have failed because the Affinity
  // mask is too small for the number of CPU's on the system (i.e. the
  // system has more than 1024 CPUs). Allocate a mask large enough for
  // twice as many CPUs.
  cpu_set_t *DynAffinity;
  DynAffinity = CPU_ALLOC(2048);
  if (sched_getaffinity(0, CPU_ALLOC_SIZE(2048), DynAffinity) == 0) {
    int NumCPUs = CPU_COUNT(DynAffinity);
    CPU_FREE(DynAffinity);
    return NumCPUs;
  }
  return -1;
}
#elif defined(__APPLE__)
// Gets the number of *physical cores* on the machine.
static int computeHostNumPhysicalCores() {
  uint32_t count;
  size_t len = sizeof(count);
  sysctlbyname("hw.physicalcpu", &count, &len, NULL, 0);
  if (count < 1) {
    int nm[2];
    nm[0] = CTL_HW;
    nm[1] = HW_AVAILCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);
    if (count < 1)
      return -1;
  }
  return count;
}
#elif defined(__MVS__)
static int computeHostNumPhysicalCores() {
  enum {
    // Byte offset of the pointer to the Communications Vector Table (CVT) in
    // the Prefixed Save Area (PSA). The table entry is a 31-bit pointer and
    // will be zero-extended to uintptr_t.
    FLCCVT = 16,
    // Byte offset of the pointer to the Common System Data Area (CSD) in the
    // CVT. The table entry is a 31-bit pointer and will be zero-extended to
    // uintptr_t.
    CVTCSD = 660,
    // Byte offset to the number of live CPs in the LPAR, stored as a signed
    // 32-bit value in the table.
    CSD_NUMBER_ONLINE_STANDARD_CPS = 264,
  };
  char *PSA = 0;
  char *CVT = reinterpret_cast<char *>(
      static_cast<uintptr_t>(reinterpret_cast<unsigned int &>(PSA[FLCCVT])));
  char *CSD = reinterpret_cast<char *>(
      static_cast<uintptr_t>(reinterpret_cast<unsigned int &>(CVT[CVTCSD])));
  return reinterpret_cast<int &>(CSD[CSD_NUMBER_ONLINE_STANDARD_CPS]);
}
#elif defined(_WIN32) && LLVM_ENABLE_THREADS != 0
// Defined in llvm/lib/Support/Windows/Threading.inc
int computeHostNumPhysicalCores();
#else
// On other systems, return -1 to indicate unknown.
static int computeHostNumPhysicalCores() { return -1; }
#endif

int llvm::get_physical_cores() {
  static int NumCores = computeHostNumPhysicalCores();
  return NumCores;
}
