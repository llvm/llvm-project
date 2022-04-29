//===-- PerfTests.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __x86_64__

#include "Perf.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>

using namespace lldb_private;
using namespace process_linux;
using namespace llvm;

/// Helper function to read current TSC value.
///
/// This code is based on llvm/xray.
static Expected<uint64_t> readTsc() {

  unsigned int eax, ebx, ecx, edx;

  // We check whether rdtscp support is enabled. According to the x86_64 manual,
  // level should be set at 0x80000001, and we should have a look at bit 27 in
  // EDX. That's 0x8000000 (or 1u << 27).
  __asm__ __volatile__("cpuid"
                       : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                       : "0"(0x80000001));
  if (!(edx & (1u << 27))) {
    return createStringError(inconvertibleErrorCode(),
                             "Missing rdtscp support.");
  }

  unsigned cpu;
  unsigned long rax, rdx;

  __asm__ __volatile__("rdtscp\n" : "=a"(rax), "=d"(rdx), "=c"(cpu)::);

  return (rdx << 32) + rax;
}

// Test TSC to walltime conversion based on perf conversion values.
TEST(Perf, TscConversion) {
  // This test works by first reading the TSC value directly before
  // and after sleeping, then converting these values to nanoseconds, and
  // finally ensuring the difference is approximately equal to the sleep time.
  //
  // There will be slight overhead associated with the sleep call, so it isn't
  // reasonable to expect the difference to be exactly equal to the sleep time.

  const int SLEEP_SECS = 1;
  std::chrono::nanoseconds SLEEP_NANOS{std::chrono::seconds(SLEEP_SECS)};

  Expected<LinuxPerfZeroTscConversion> params =
      LoadPerfTscConversionParameters();

  // Skip the test if the conversion parameters aren't available.
  if (!params)
    GTEST_SKIP() << toString(params.takeError());

  Expected<uint64_t> tsc_before_sleep = readTsc();
  sleep(SLEEP_SECS);
  Expected<uint64_t> tsc_after_sleep = readTsc();

  // Skip the test if we are unable to read the TSC value.
  if (!tsc_before_sleep)
    GTEST_SKIP() << toString(tsc_before_sleep.takeError());
  if (!tsc_after_sleep)
    GTEST_SKIP() << toString(tsc_after_sleep.takeError());

  std::chrono::nanoseconds converted_tsc_diff =
      params->Convert(*tsc_after_sleep) - params->Convert(*tsc_before_sleep);

  std::chrono::microseconds acceptable_overhead(500);

  ASSERT_GE(converted_tsc_diff.count(), SLEEP_NANOS.count());
  ASSERT_LT(converted_tsc_diff.count(),
            (SLEEP_NANOS + acceptable_overhead).count());
}

size_t ReadCylicBufferWrapper(void *buf, size_t buf_size, void *cyc_buf,
                              size_t cyc_buf_size, size_t cyc_start,
                              size_t offset) {
  llvm::MutableArrayRef<uint8_t> dst(reinterpret_cast<uint8_t *>(buf),
                                     buf_size);
  llvm::ArrayRef<uint8_t> src(reinterpret_cast<uint8_t *>(cyc_buf),
                              cyc_buf_size);
  ReadCyclicBuffer(dst, src, cyc_start, offset);
  return dst.size();
}

TEST(CyclicBuffer, EdgeCases) {
  size_t bytes_read;
  uint8_t cyclic_buffer[6] = {'l', 'i', 'c', 'c', 'y', 'c'};

  // We will always leave the last bytes untouched
  // so that string comparisons work.
  char smaller_buffer[4] = {};

  // empty buffer to read into
  bytes_read = ReadCylicBufferWrapper(smaller_buffer, 0, cyclic_buffer,
                                      sizeof(cyclic_buffer), 3, 0);
  ASSERT_EQ(0u, bytes_read);

  // empty cyclic buffer
  bytes_read = ReadCylicBufferWrapper(smaller_buffer, sizeof(smaller_buffer),
                                      cyclic_buffer, 0, 3, 0);
  ASSERT_EQ(0u, bytes_read);

  // bigger offset
  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, sizeof(smaller_buffer),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 6);
  ASSERT_EQ(0u, bytes_read);

  // wrong offset
  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, sizeof(smaller_buffer),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 7);
  ASSERT_EQ(0u, bytes_read);

  // wrong start
  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, sizeof(smaller_buffer),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 7);
  ASSERT_EQ(0u, bytes_read);
}

TEST(CyclicBuffer, EqualSizeBuffer) {
  size_t bytes_read = 0;
  uint8_t cyclic_buffer[6] = {'l', 'i', 'c', 'c', 'y', 'c'};

  char cyclic[] = "cyclic";
  for (size_t i = 0; i < sizeof(cyclic); i++) {
    // We will always leave the last bytes untouched
    // so that string comparisons work.
    char equal_size_buffer[7] = {};
    bytes_read =
        ReadCylicBufferWrapper(equal_size_buffer, sizeof(cyclic_buffer),
                               cyclic_buffer, sizeof(cyclic_buffer), 3, i);
    ASSERT_EQ((sizeof(cyclic) - i - 1), bytes_read);
    ASSERT_STREQ(equal_size_buffer, (cyclic + i));
  }
}

TEST(CyclicBuffer, SmallerSizeBuffer) {
  size_t bytes_read;
  uint8_t cyclic_buffer[6] = {'l', 'i', 'c', 'c', 'y', 'c'};

  // We will always leave the last bytes untouched
  // so that string comparisons work.
  char smaller_buffer[4] = {};
  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 0);
  ASSERT_EQ(3u, bytes_read);
  ASSERT_STREQ(smaller_buffer, "cyc");

  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 1);
  ASSERT_EQ(3u, bytes_read);
  ASSERT_STREQ(smaller_buffer, "ycl");

  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 2);
  ASSERT_EQ(3u, bytes_read);
  ASSERT_STREQ(smaller_buffer, "cli");

  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 3);
  ASSERT_EQ(3u, bytes_read);
  ASSERT_STREQ(smaller_buffer, "lic");

  {
    char smaller_buffer[4] = {};
    bytes_read =
        ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                               cyclic_buffer, sizeof(cyclic_buffer), 3, 4);
    ASSERT_EQ(2u, bytes_read);
    ASSERT_STREQ(smaller_buffer, "ic");
  }
  {
    char smaller_buffer[4] = {};
    bytes_read =
        ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                               cyclic_buffer, sizeof(cyclic_buffer), 3, 5);
    ASSERT_EQ(1u, bytes_read);
    ASSERT_STREQ(smaller_buffer, "c");
  }
}

TEST(CyclicBuffer, BiggerSizeBuffer) {
  size_t bytes_read = 0;
  uint8_t cyclic_buffer[6] = {'l', 'i', 'c', 'c', 'y', 'c'};

  char cyclic[] = "cyclic";
  for (size_t i = 0; i < sizeof(cyclic); i++) {
    // We will always leave the last bytes untouched
    // so that string comparisons work.
    char bigger_buffer[10] = {};
    bytes_read =
        ReadCylicBufferWrapper(bigger_buffer, (sizeof(bigger_buffer) - 1),
                               cyclic_buffer, sizeof(cyclic_buffer), 3, i);
    ASSERT_EQ((sizeof(cyclic) - i - 1), bytes_read);
    ASSERT_STREQ(bigger_buffer, (cyclic + i));
  }
}

#endif // __x86_64__
