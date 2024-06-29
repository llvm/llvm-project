// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(_MSC_VER)
// FIXME: This must be defined before any other includes to disable deprecation
// warnings for use of codecvt from C++17. We should remove our reliance on
// the deprecated functionality instead.
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "internal_macros.h"

#ifdef BENCHMARK_OS_WINDOWS
#if !defined(WINVER) || WINVER < 0x0600
#undef WINVER
#define WINVER 0x0600
#endif  // WINVER handling
#include <shlwapi.h>
#undef StrCat  // Don't let StrCat in string_util.h be renamed to lstrcatA
#include <versionhelpers.h>
#include <windows.h>

#include <codecvt>
#else
#include <fcntl.h>
#if !defined(BENCHMARK_OS_FUCHSIA) && !defined(BENCHMARK_OS_QURT)
#include <sys/resource.h>
#endif
#include <sys/time.h>
#include <sys/types.h>  // this header must be included before 'sys/sysctl.h' to avoid compilation error on FreeBSD
#include <unistd.h>
#if defined BENCHMARK_OS_FREEBSD || defined BENCHMARK_OS_MACOSX || \
    defined BENCHMARK_OS_NETBSD || defined BENCHMARK_OS_OPENBSD || \
    defined BENCHMARK_OS_DRAGONFLY
#define BENCHMARK_HAS_SYSCTL
#include <sys/sysctl.h>
#endif
#endif
#if defined(BENCHMARK_OS_SOLARIS)
#include <kstat.h>
#include <netdb.h>
#endif
#if defined(BENCHMARK_OS_QNX)
#include <sys/syspage.h>
#endif
#if defined(BENCHMARK_OS_QURT)
#include <qurt.h>
#endif
#if defined(BENCHMARK_HAS_PTHREAD_AFFINITY)
#include <pthread.h>
#endif

#include <algorithm>
#include <array>
#include <bitset>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <memory>
#include <random>
#include <sstream>
#include <utility>

#include "benchmark/benchmark.h"
#include "check.h"
#include "cycleclock.h"
#include "internal_macros.h"
#include "log.h"
#include "string_util.h"
#include "timers.h"

namespace benchmark {
namespace {

void PrintImp(std::ostream& out) { out << std::endl; }

template <class First, class... Rest>
void PrintImp(std::ostream& out, First&& f, Rest&&... rest) {
  out << std::forward<First>(f);
  PrintImp(out, std::forward<Rest>(rest)...);
}

template <class... Args>
BENCHMARK_NORETURN void PrintErrorAndDie(Args&&... args) {
  PrintImp(std::cerr, std::forward<Args>(args)...);
  std::exit(EXIT_FAILURE);
}

#ifdef BENCHMARK_HAS_SYSCTL

/// ValueUnion - A type used to correctly alias the byte-for-byte output of
/// `sysctl` with the result type it's to be interpreted as.
struct ValueUnion {
  union DataT {
    int32_t int32_value;
    int64_t int64_value;
    // For correct aliasing of union members from bytes.
    char bytes[8];
  };
  using DataPtr = std::unique_ptr<DataT, decltype(&std::free)>;

  // The size of the data union member + its trailing array size.
  std::size_t size;
  DataPtr buff;

 public:
  ValueUnion() : size(0), buff(nullptr, &std::free) {}

  explicit ValueUnion(std::size_t buff_size)
      : size(sizeof(DataT) + buff_size),
        buff(::new (std::malloc(size)) DataT(), &std::free) {}

  ValueUnion(ValueUnion&& other) = default;

  explicit operator bool() const { return bool(buff); }

  char* data() const { return buff->bytes; }

  std::string GetAsString() const { return std::string(data()); }

  int64_t GetAsInteger() const {
    if (size == sizeof(buff->int32_value))
      return buff->int32_value;
    else if (size == sizeof(buff->int64_value))
      return buff->int64_value;
    BENCHMARK_UNREACHABLE();
  }

  template <class T, int N>
  std::array<T, N> GetAsArray() {
    const int arr_size = sizeof(T) * N;
    BM_CHECK_LE(arr_size, size);
    std::array<T, N> arr;
    std::memcpy(arr.data(), data(), arr_size);
    return arr;
  }
};

ValueUnion GetSysctlImp(std::string const& name) {
#if defined BENCHMARK_OS_OPENBSD
  int mib[2];

  mib[0] = CTL_HW;
  if ((name == "hw.ncpu") || (name == "hw.cpuspeed")) {
    ValueUnion buff(sizeof(int));

    if (name == "hw.ncpu") {
      mib[1] = HW_NCPU;
    } else {
      mib[1] = HW_CPUSPEED;
    }

    if (sysctl(mib, 2, buff.data(), &buff.Size, nullptr, 0) == -1) {
      return ValueUnion();
    }
    return buff;
  }
  return ValueUnion();
#else
  std::size_t cur_buff_size = 0;
  if (sysctlbyname(name.c_str(), nullptr, &cur_buff_size, nullptr, 0) == -1)
    return ValueUnion();

  ValueUnion buff(cur_buff_size);
  if (sysctlbyname(name.c_str(), buff.data(), &buff.size, nullptr, 0) == 0)
    return buff;
  return ValueUnion();
#endif
}

BENCHMARK_MAYBE_UNUSED
bool GetSysctl(std::string const& name, std::string* out) {
  out->clear();
  auto buff = GetSysctlImp(name);
  if (!buff) return false;
  out->assign(buff.data());
  return true;
}

template <class Tp,
          class = typename std::enable_if<std::is_integral<Tp>::value>::type>
bool GetSysctl(std::string const& name, Tp* out) {
  *out = 0;
  auto buff = GetSysctlImp(name);
  if (!buff) return false;
  *out = static_cast<Tp>(buff.GetAsInteger());
  return true;
}

template <class Tp, size_t N>
bool GetSysctl(std::string const& name, std::array<Tp, N>* out) {
  auto buff = GetSysctlImp(name);
  if (!buff) return false;
  *out = buff.GetAsArray<Tp, N>();
  return true;
}
#endif

template <class ArgT>
bool ReadFromFile(std::string const& fname, ArgT* arg) {
  *arg = ArgT();
  std::ifstream f(fname.c_str());
  if (!f.is_open()) return false;
  f >> *arg;
  return f.good();
}

CPUInfo::Scaling CpuScaling(int num_cpus) {
  // We don't have a valid CPU count, so don't even bother.
  if (num_cpus <= 0) return CPUInfo::Scaling::UNKNOWN;
#if defined(BENCHMARK_OS_QNX)
  return CPUInfo::Scaling::UNKNOWN;
#elif !defined(BENCHMARK_OS_WINDOWS)
  // On Linux, the CPUfreq subsystem exposes CPU information as files on the
  // local file system. If reading the exported files fails, then we may not be
  // running on Linux, so we silently ignore all the read errors.
  std::string res;
  for (int cpu = 0; cpu < num_cpus; ++cpu) {
    std::string governor_file =
        StrCat("/sys/devices/system/cpu/cpu", cpu, "/cpufreq/scaling_governor");
    if (ReadFromFile(governor_file, &res) && res != "performance")
      return CPUInfo::Scaling::ENABLED;
  }
  return CPUInfo::Scaling::DISABLED;
#else
  return CPUInfo::Scaling::UNKNOWN;
#endif
}

int CountSetBitsInCPUMap(std::string val) {
  auto CountBits = [](std::string part) {
    using CPUMask = std::bitset<sizeof(std::uintptr_t) * CHAR_BIT>;
    part = "0x" + part;
    CPUMask mask(benchmark::stoul(part, nullptr, 16));
    return static_cast<int>(mask.count());
  };
  std::size_t pos;
  int total = 0;
  while ((pos = val.find(',')) != std::string::npos) {
    total += CountBits(val.substr(0, pos));
    val = val.substr(pos + 1);
  }
  if (!val.empty()) {
    total += CountBits(val);
  }
  return total;
}

BENCHMARK_MAYBE_UNUSED
std::vector<CPUInfo::CacheInfo> GetCacheSizesFromKVFS() {
  std::vector<CPUInfo::CacheInfo> res;
  std::string dir = "/sys/devices/system/cpu/cpu0/cache/";
  int idx = 0;
  while (true) {
    CPUInfo::CacheInfo info;
    std::string fpath = StrCat(dir, "index", idx++, "/");
    std::ifstream f(StrCat(fpath, "size").c_str());
    if (!f.is_open()) break;
    std::string suffix;
    f >> info.size;
    if (f.fail())
      PrintErrorAndDie("Failed while reading file '", fpath, "size'");
    if (f.good()) {
      f >> suffix;
      if (f.bad())
        PrintErrorAndDie(
            "Invalid cache size format: failed to read size suffix");
      else if (f && suffix != "K")
        PrintErrorAndDie("Invalid cache size format: Expected bytes ", suffix);
      else if (suffix == "K")
        info.size *= 1024;
    }
    if (!ReadFromFile(StrCat(fpath, "type"), &info.type))
      PrintErrorAndDie("Failed to read from file ", fpath, "type");
    if (!ReadFromFile(StrCat(fpath, "level"), &info.level))
      PrintErrorAndDie("Failed to read from file ", fpath, "level");
    std::string map_str;
    if (!ReadFromFile(StrCat(fpath, "shared_cpu_map"), &map_str))
      PrintErrorAndDie("Failed to read from file ", fpath, "shared_cpu_map");
    info.num_sharing = CountSetBitsInCPUMap(map_str);
    res.push_back(info);
  }

  return res;
}

#ifdef BENCHMARK_OS_MACOSX
std::vector<CPUInfo::CacheInfo> GetCacheSizesMacOSX() {
  std::vector<CPUInfo::CacheInfo> res;
  std::array<int, 4> cache_counts{{0, 0, 0, 0}};
  GetSysctl("hw.cacheconfig", &cache_counts);

  struct {
    std::string name;
    std::string type;
    int level;
    int num_sharing;
  } cases[] = {{"hw.l1dcachesize", "Data", 1, cache_counts[1]},
               {"hw.l1icachesize", "Instruction", 1, cache_counts[1]},
               {"hw.l2cachesize", "Unified", 2, cache_counts[2]},
               {"hw.l3cachesize", "Unified", 3, cache_counts[3]}};
  for (auto& c : cases) {
    int val;
    if (!GetSysctl(c.name, &val)) continue;
    CPUInfo::CacheInfo info;
    info.type = c.type;
    info.level = c.level;
    info.size = val;
    info.num_sharing = c.num_sharing;
    res.push_back(std::move(info));
  }
  return res;
}
#elif defined(BENCHMARK_OS_WINDOWS)
std::vector<CPUInfo::CacheInfo> GetCacheSizesWindows() {
  std::vector<CPUInfo::CacheInfo> res;
  DWORD buffer_size = 0;
  using PInfo = SYSTEM_LOGICAL_PROCESSOR_INFORMATION;
  using CInfo = CACHE_DESCRIPTOR;

  using UPtr = std::unique_ptr<PInfo, decltype(&std::free)>;
  GetLogicalProcessorInformation(nullptr, &buffer_size);
  UPtr buff(static_cast<PInfo*>(std::malloc(buffer_size)), &std::free);
  if (!GetLogicalProcessorInformation(buff.get(), &buffer_size))
    PrintErrorAndDie("Failed during call to GetLogicalProcessorInformation: ",
                     GetLastError());

  PInfo* it = buff.get();
  PInfo* end = buff.get() + (buffer_size / sizeof(PInfo));

  for (; it != end; ++it) {
    if (it->Relationship != RelationCache) continue;
    using BitSet = std::bitset<sizeof(ULONG_PTR) * CHAR_BIT>;
    BitSet b(it->ProcessorMask);
    // To prevent duplicates, only consider caches where CPU 0 is specified
    if (!b.test(0)) continue;
    const CInfo& cache = it->Cache;
    CPUInfo::CacheInfo C;
    C.num_sharing = static_cast<int>(b.count());
    C.level = cache.Level;
    C.size = cache.Size;
    C.type = "Unknown";
    switch (cache.Type) {
      case CacheUnified:
        C.type = "Unified";
        break;
      case CacheInstruction:
        C.type = "Instruction";
        break;
      case CacheData:
        C.type = "Data";
        break;
      case CacheTrace:
        C.type = "Trace";
        break;
    }
    res.push_back(C);
  }
  return res;
}
#elif BENCHMARK_OS_QNX
std::vector<CPUInfo::CacheInfo> GetCacheSizesQNX() {
  std::vector<CPUInfo::CacheInfo> res;
  struct cacheattr_entry* cache = SYSPAGE_ENTRY(cacheattr);
  uint32_t const elsize = SYSPAGE_ELEMENT_SIZE(cacheattr);
  int num = SYSPAGE_ENTRY_SIZE(cacheattr) / elsize;
  for (int i = 0; i < num; ++i) {
    CPUInfo::CacheInfo info;
    switch (cache->flags) {
      case CACHE_FLAG_INSTR:
        info.type = "Instruction";
        info.level = 1;
        break;
      case CACHE_FLAG_DATA:
        info.type = "Data";
        info.level = 1;
        break;
      case CACHE_FLAG_UNIFIED:
        info.type = "Unified";
        info.level = 2;
        break;
      case CACHE_FLAG_SHARED:
        info.type = "Shared";
        info.level = 3;
        break;
      default:
        continue;
        break;
    }
    info.size = cache->line_size * cache->num_lines;
    info.num_sharing = 0;
    res.push_back(std::move(info));
    cache = SYSPAGE_ARRAY_ADJ_OFFSET(cacheattr, cache, elsize);
  }
  return res;
}
#endif

std::vector<CPUInfo::CacheInfo> GetCacheSizes() {
#ifdef BENCHMARK_OS_MACOSX
  return GetCacheSizesMacOSX();
#elif defined(BENCHMARK_OS_WINDOWS)
  return GetCacheSizesWindows();
#elif defined(BENCHMARK_OS_QNX)
  return GetCacheSizesQNX();
#elif defined(BENCHMARK_OS_QURT)
  return std::vector<CPUInfo::CacheInfo>();
#else
  return GetCacheSizesFromKVFS();
#endif
}

std::string GetSystemName() {
#if defined(BENCHMARK_OS_WINDOWS)
  std::string str;
  static constexpr int COUNT = MAX_COMPUTERNAME_LENGTH + 1;
  TCHAR hostname[COUNT] = {'\0'};
  DWORD DWCOUNT = COUNT;
  if (!GetComputerName(hostname, &DWCOUNT)) return std::string("");
#ifndef UNICODE
  str = std::string(hostname, DWCOUNT);
#else
  // `WideCharToMultiByte` returns `0` when conversion fails.
  int len = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, hostname,
                                DWCOUNT, NULL, 0, NULL, NULL);
  str.resize(len);
  WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, hostname, DWCOUNT, &str[0],
                      str.size(), NULL, NULL);
#endif
  return str;
#elif defined(BENCHMARK_OS_QURT)
  std::string str = "Hexagon DSP";
  qurt_arch_version_t arch_version_struct;
  if (qurt_sysenv_get_arch_version(&arch_version_struct) == QURT_EOK) {
    str += " v";
    str += std::to_string(arch_version_struct.arch_version);
  }
  return str;
#else
#ifndef HOST_NAME_MAX
#ifdef BENCHMARK_HAS_SYSCTL  // BSD/Mac doesn't have HOST_NAME_MAX defined
#define HOST_NAME_MAX 64
#elif defined(BENCHMARK_OS_NACL)
#define HOST_NAME_MAX 64
#elif defined(BENCHMARK_OS_QNX)
#define HOST_NAME_MAX 154
#elif defined(BENCHMARK_OS_RTEMS)
#define HOST_NAME_MAX 256
#elif defined(BENCHMARK_OS_SOLARIS)
#define HOST_NAME_MAX MAXHOSTNAMELEN
#elif defined(BENCHMARK_OS_ZOS)
#define HOST_NAME_MAX _POSIX_HOST_NAME_MAX
#else
#pragma message("HOST_NAME_MAX not defined. using 64")
#define HOST_NAME_MAX 64
#endif
#endif  // def HOST_NAME_MAX
  char hostname[HOST_NAME_MAX];
  int retVal = gethostname(hostname, HOST_NAME_MAX);
  if (retVal != 0) return std::string("");
  return std::string(hostname);
#endif  // Catch-all POSIX block.
}

int GetNumCPUsImpl() {
#ifdef BENCHMARK_HAS_SYSCTL
  int num_cpu = -1;
  if (GetSysctl("hw.ncpu", &num_cpu)) return num_cpu;
  PrintErrorAndDie("Err: ", strerror(errno));
#elif defined(BENCHMARK_OS_WINDOWS)
  SYSTEM_INFO sysinfo;
  // Use memset as opposed to = {} to avoid GCC missing initializer false
  // positives.
  std::memset(&sysinfo, 0, sizeof(SYSTEM_INFO));
  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;  // number of logical
                                        // processors in the current
                                        // group
#elif defined(BENCHMARK_OS_SOLARIS)
  // Returns -1 in case of a failure.
  long num_cpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (num_cpu < 0) {
    PrintErrorAndDie("sysconf(_SC_NPROCESSORS_ONLN) failed with error: ",
                     strerror(errno));
  }
  return (int)num_cpu;
#elif defined(BENCHMARK_OS_QNX)
  return static_cast<int>(_syspage_ptr->num_cpu);
#elif defined(BENCHMARK_OS_QURT)
  qurt_sysenv_max_hthreads_t hardware_threads;
  if (qurt_sysenv_get_max_hw_threads(&hardware_threads) != QURT_EOK) {
    hardware_threads.max_hthreads = 1;
  }
  return hardware_threads.max_hthreads;
#else
  int num_cpus = 0;
  int max_id = -1;
  std::ifstream f("/proc/cpuinfo");
  if (!f.is_open()) {
    PrintErrorAndDie("Failed to open /proc/cpuinfo");
  }
#if defined(__alpha__)
  const std::string Key = "cpus detected";
#else
  const std::string Key = "processor";
#endif
  std::string ln;
  while (std::getline(f, ln)) {
    if (ln.empty()) continue;
    std::size_t split_idx = ln.find(':');
    std::string value;
#if defined(__s390__)
    // s390 has another format in /proc/cpuinfo
    // it needs to be parsed differently
    if (split_idx != std::string::npos)
      value = ln.substr(Key.size() + 1, split_idx - Key.size() - 1);
#else
    if (split_idx != std::string::npos) value = ln.substr(split_idx + 1);
#endif
    if (ln.size() >= Key.size() && ln.compare(0, Key.size(), Key) == 0) {
      num_cpus++;
      if (!value.empty()) {
        const int cur_id = benchmark::stoi(value);
        max_id = std::max(cur_id, max_id);
      }
    }
  }
  if (f.bad()) {
    PrintErrorAndDie("Failure reading /proc/cpuinfo");
  }
  if (!f.eof()) {
    PrintErrorAndDie("Failed to read to end of /proc/cpuinfo");
  }
  f.close();

  if ((max_id + 1) != num_cpus) {
    fprintf(stderr,
            "CPU ID assignments in /proc/cpuinfo seem messed up."
            " This is usually caused by a bad BIOS.\n");
  }
  return num_cpus;
#endif
  BENCHMARK_UNREACHABLE();
}

int GetNumCPUs() {
  const int num_cpus = GetNumCPUsImpl();
  if (num_cpus < 1) {
    PrintErrorAndDie(
        "Unable to extract number of CPUs.  If your platform uses "
        "/proc/cpuinfo, custom support may need to be added.");
  }
  return num_cpus;
}

class ThreadAffinityGuard final {
 public:
  ThreadAffinityGuard() : reset_affinity(SetAffinity()) {
    if (!reset_affinity)
      std::cerr << "***WARNING*** Failed to set thread affinity. Estimated CPU "
                   "frequency may be incorrect."
                << std::endl;
  }

  ~ThreadAffinityGuard() {
    if (!reset_affinity) return;

#if defined(BENCHMARK_HAS_PTHREAD_AFFINITY)
    int ret = pthread_setaffinity_np(self, sizeof(previous_affinity),
                                     &previous_affinity);
    if (ret == 0) return;
#elif defined(BENCHMARK_OS_WINDOWS_WIN32)
    DWORD_PTR ret = SetThreadAffinityMask(self, previous_affinity);
    if (ret != 0) return;
#endif  // def BENCHMARK_HAS_PTHREAD_AFFINITY
    PrintErrorAndDie("Failed to reset thread affinity");
  }

  ThreadAffinityGuard(ThreadAffinityGuard&&) = delete;
  ThreadAffinityGuard(const ThreadAffinityGuard&) = delete;
  ThreadAffinityGuard& operator=(ThreadAffinityGuard&&) = delete;
  ThreadAffinityGuard& operator=(const ThreadAffinityGuard&) = delete;

 private:
  bool SetAffinity() {
#if defined(BENCHMARK_HAS_PTHREAD_AFFINITY)
    int ret;
    self = pthread_self();
    ret = pthread_getaffinity_np(self, sizeof(previous_affinity),
                                 &previous_affinity);
    if (ret != 0) return false;

    cpu_set_t affinity;
    memcpy(&affinity, &previous_affinity, sizeof(affinity));

    bool is_first_cpu = true;

    for (int i = 0; i < CPU_SETSIZE; ++i)
      if (CPU_ISSET(i, &affinity)) {
        if (is_first_cpu)
          is_first_cpu = false;
        else
          CPU_CLR(i, &affinity);
      }

    if (is_first_cpu) return false;

    ret = pthread_setaffinity_np(self, sizeof(affinity), &affinity);
    return ret == 0;
#elif defined(BENCHMARK_OS_WINDOWS_WIN32)
    self = GetCurrentThread();
    DWORD_PTR mask = static_cast<DWORD_PTR>(1) << GetCurrentProcessorNumber();
    previous_affinity = SetThreadAffinityMask(self, mask);
    return previous_affinity != 0;
#else
    return false;
#endif  // def BENCHMARK_HAS_PTHREAD_AFFINITY
  }

#if defined(BENCHMARK_HAS_PTHREAD_AFFINITY)
  pthread_t self;
  cpu_set_t previous_affinity;
#elif defined(BENCHMARK_OS_WINDOWS_WIN32)
  HANDLE self;
  DWORD_PTR previous_affinity;
#endif  // def BENCHMARK_HAS_PTHREAD_AFFINITY
  bool reset_affinity;
};

double GetCPUCyclesPerSecond(CPUInfo::Scaling scaling) {
  // Currently, scaling is only used on linux path here,
  // suppress diagnostics about it being unused on other paths.
  (void)scaling;

#if defined BENCHMARK_OS_LINUX || defined BENCHMARK_OS_CYGWIN
  long freq;

  // If the kernel is exporting the tsc frequency use that. There are issues
  // where cpuinfo_max_freq cannot be relied on because the BIOS may be
  // exporintg an invalid p-state (on x86) or p-states may be used to put the
  // processor in a new mode (turbo mode). Essentially, those frequencies
  // cannot always be relied upon. The same reasons apply to /proc/cpuinfo as
  // well.
  if (ReadFromFile("/sys/devices/system/cpu/cpu0/tsc_freq_khz", &freq)
      // If CPU scaling is disabled, use the *current* frequency.
      // Note that we specifically don't want to read cpuinfo_cur_freq,
      // because it is only readable by root.
      || (scaling == CPUInfo::Scaling::DISABLED &&
          ReadFromFile("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
                       &freq))
      // Otherwise, if CPU scaling may be in effect, we want to use
      // the *maximum* frequency, not whatever CPU speed some random processor
      // happens to be using now.
      || ReadFromFile("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
                      &freq)) {
    // The value is in kHz (as the file name suggests).  For example, on a
    // 2GHz warpstation, the file contains the value "2000000".
    return static_cast<double>(freq) * 1000.0;
  }

  const double error_value = -1;
  double bogo_clock = error_value;

  std::ifstream f("/proc/cpuinfo");
  if (!f.is_open()) {
    std::cerr << "failed to open /proc/cpuinfo\n";
    return error_value;
  }

  auto StartsWithKey = [](std::string const& Value, std::string const& Key) {
    if (Key.size() > Value.size()) return false;
    auto Cmp = [&](char X, char Y) {
      return std::tolower(X) == std::tolower(Y);
    };
    return std::equal(Key.begin(), Key.end(), Value.begin(), Cmp);
  };

  std::string ln;
  while (std::getline(f, ln)) {
    if (ln.empty()) continue;
    std::size_t split_idx = ln.find(':');
    std::string value;
    if (split_idx != std::string::npos) value = ln.substr(split_idx + 1);
    // When parsing the "cpu MHz" and "bogomips" (fallback) entries, we only
    // accept positive values. Some environments (virtual machines) report zero,
    // which would cause infinite looping in WallTime_Init.
    if (StartsWithKey(ln, "cpu MHz")) {
      if (!value.empty()) {
        double cycles_per_second = benchmark::stod(value) * 1000000.0;
        if (cycles_per_second > 0) return cycles_per_second;
      }
    } else if (StartsWithKey(ln, "bogomips")) {
      if (!value.empty()) {
        bogo_clock = benchmark::stod(value) * 1000000.0;
        if (bogo_clock < 0.0) bogo_clock = error_value;
      }
    }
  }
  if (f.bad()) {
    std::cerr << "Failure reading /proc/cpuinfo\n";
    return error_value;
  }
  if (!f.eof()) {
    std::cerr << "Failed to read to end of /proc/cpuinfo\n";
    return error_value;
  }
  f.close();
  // If we found the bogomips clock, but nothing better, we'll use it (but
  // we're not happy about it); otherwise, fallback to the rough estimation
  // below.
  if (bogo_clock >= 0.0) return bogo_clock;

#elif defined BENCHMARK_HAS_SYSCTL
  constexpr auto* freqStr =
#if defined(BENCHMARK_OS_FREEBSD) || defined(BENCHMARK_OS_NETBSD)
      "machdep.tsc_freq";
#elif defined BENCHMARK_OS_OPENBSD
      "hw.cpuspeed";
#elif defined BENCHMARK_OS_DRAGONFLY
      "hw.tsc_frequency";
#else
      "hw.cpufrequency";
#endif
  unsigned long long hz = 0;
#if defined BENCHMARK_OS_OPENBSD
  if (GetSysctl(freqStr, &hz)) return hz * 1000000;
#else
  if (GetSysctl(freqStr, &hz)) return hz;
#endif
  fprintf(stderr, "Unable to determine clock rate from sysctl: %s: %s\n",
          freqStr, strerror(errno));
  fprintf(stderr,
          "This does not affect benchmark measurements, only the "
          "metadata output.\n");

#elif defined BENCHMARK_OS_WINDOWS_WIN32
  // In NT, read MHz from the registry. If we fail to do so or we're in win9x
  // then make a crude estimate.
  DWORD data, data_size = sizeof(data);
  if (IsWindowsXPOrGreater() &&
      SUCCEEDED(
          SHGetValueA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      "~MHz", nullptr, &data, &data_size)))
    return static_cast<double>(static_cast<int64_t>(data) *
                               static_cast<int64_t>(1000 * 1000));  // was mhz
#elif defined(BENCHMARK_OS_SOLARIS)
  kstat_ctl_t* kc = kstat_open();
  if (!kc) {
    std::cerr << "failed to open /dev/kstat\n";
    return -1;
  }
  kstat_t* ksp = kstat_lookup(kc, const_cast<char*>("cpu_info"), -1,
                              const_cast<char*>("cpu_info0"));
  if (!ksp) {
    std::cerr << "failed to lookup in /dev/kstat\n";
    return -1;
  }
  if (kstat_read(kc, ksp, NULL) < 0) {
    std::cerr << "failed to read from /dev/kstat\n";
    return -1;
  }
  kstat_named_t* knp = (kstat_named_t*)kstat_data_lookup(
      ksp, const_cast<char*>("current_clock_Hz"));
  if (!knp) {
    std::cerr << "failed to lookup data in /dev/kstat\n";
    return -1;
  }
  if (knp->data_type != KSTAT_DATA_UINT64) {
    std::cerr << "current_clock_Hz is of unexpected data type: "
              << knp->data_type << "\n";
    return -1;
  }
  double clock_hz = knp->value.ui64;
  kstat_close(kc);
  return clock_hz;
#elif defined(BENCHMARK_OS_QNX)
  return static_cast<double>(
      static_cast<int64_t>(SYSPAGE_ENTRY(cpuinfo)->speed) *
      static_cast<int64_t>(1000 * 1000));
#elif defined(BENCHMARK_OS_QURT)
  // QuRT doesn't provide any API to query Hexagon frequency.
  return 1000000000;
#endif
  // If we've fallen through, attempt to roughly estimate the CPU clock rate.

  // Make sure to use the same cycle counter when starting and stopping the
  // cycle timer. We just pin the current thread to a cpu in the previous
  // affinity set.
  ThreadAffinityGuard affinity_guard;

  static constexpr double estimate_time_s = 1.0;
  const double start_time = ChronoClockNow();
  const auto start_ticks = cycleclock::Now();

  // Impose load instead of calling sleep() to make sure the cycle counter
  // works.
  using PRNG = std::minstd_rand;
  using Result = PRNG::result_type;
  PRNG rng(static_cast<Result>(start_ticks));

  Result state = 0;

  do {
    static constexpr size_t batch_size = 10000;
    rng.discard(batch_size);
    state += rng();

  } while (ChronoClockNow() - start_time < estimate_time_s);

  DoNotOptimize(state);

  const auto end_ticks = cycleclock::Now();
  const double end_time = ChronoClockNow();

  return static_cast<double>(end_ticks - start_ticks) / (end_time - start_time);
  // Reset the affinity of current thread when the lifetime of affinity_guard
  // ends.
}

std::vector<double> GetLoadAvg() {
#if (defined BENCHMARK_OS_FREEBSD || defined(BENCHMARK_OS_LINUX) ||     \
     defined BENCHMARK_OS_MACOSX || defined BENCHMARK_OS_NETBSD ||      \
     defined BENCHMARK_OS_OPENBSD || defined BENCHMARK_OS_DRAGONFLY) && \
    !(defined(__ANDROID__) && __ANDROID_API__ < 29)
  static constexpr int kMaxSamples = 3;
  std::vector<double> res(kMaxSamples, 0.0);
  const int nelem = getloadavg(res.data(), kMaxSamples);
  if (nelem < 1) {
    res.clear();
  } else {
    res.resize(nelem);
  }
  return res;
#else
  return {};
#endif
}

}  // end namespace

const CPUInfo& CPUInfo::Get() {
  static const CPUInfo* info = new CPUInfo();
  return *info;
}

CPUInfo::CPUInfo()
    : num_cpus(GetNumCPUs()),
      scaling(CpuScaling(num_cpus)),
      cycles_per_second(GetCPUCyclesPerSecond(scaling)),
      caches(GetCacheSizes()),
      load_avg(GetLoadAvg()) {}

const SystemInfo& SystemInfo::Get() {
  static const SystemInfo* info = new SystemInfo();
  return *info;
}

SystemInfo::SystemInfo() : name(GetSystemName()) {}
}  // end namespace benchmark
