//===-- GetProcessList.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/GetProcessList.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorExtras.h"

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <sys/sysctl.h>
#include <sys/types.h>
// struct kinfo_proc lives in <sys/user.h> on some BSDs (e.g. FreeBSD).
#include <sys/user.h>
#include <system_error>

llvm::Error
lldb_private::GetProcessList(std::vector<struct kinfo_proc> &kinfos) {
  int mib[3] = {CTL_KERN, KERN_PROC, KERN_PROC_ALL};

  // How often we retry fetching the process list.
  static constexpr unsigned g_retry_count = 200;
  // The rate at which we increase the adjusted buffer size to
  // account for newly created processes between two sysctl calls.
  static constexpr unsigned g_expected_new_pids = 500;
  // We keep increasing the expected growth rate between the two
  // sysctl calls. Check that the last attempt does not create an
  // unreasonbly large buffer. It is unlikely we run on a system where
  // 100k processes are repeatedly created between each attempted sysctl
  // pair.
  static_assert(
      g_retry_count * g_expected_new_pids <= 100'000,
      "Final retry attempt assumes an unlikely amount of new processes.");

  // This is an inherently racy API. We have to first query the size for our
  // buffer and then pass it back to sysctl. If more processes spawn between the
  // size query and the actual call to fetch, then sysctl returns ENOMEM.
  // We keep retrying until we get a passing result.
  for (unsigned attempt = 1; attempt < g_retry_count; ++attempt) {
    // Fetch the buffer size sysctl would return.
    size_t current_pid_size = 0;
    if (::sysctl(mib, 3, nullptr, &current_pid_size, nullptr, 0) != 0)
      return llvm::errorCodeToError(
          std::error_code(errno, std::generic_category()));

    // Convert the byte length result to number of elements.
    const size_t current_num_processes =
        current_pid_size / sizeof(struct kinfo_proc);

    // Adjust the buffer for new processes that spawned between the
    // previous and next sysctl call. We increase this growth each attempt
    // to account for systems where a lot of new processes spawn between
    // these two calls.
    const size_t expected_growth = attempt * g_expected_new_pids;

    // Allocate the buffer for the sysctl result.
    kinfos.resize(current_num_processes + expected_growth);

    // Fetch the actual process list and let sysctl adjust actual_pid_size.
    size_t actual_pid_size = kinfos.size() * sizeof(struct kinfo_proc);
    if (::sysctl(mib, 3, &kinfos[0], &actual_pid_size, nullptr, 0) == 0) {
      // Shrink the buffer to the actual number of processes returned.
      kinfos.resize(actual_pid_size / sizeof(struct kinfo_proc));
      return llvm::Error::success();
    }

    // Errno is set to ENOMEM if our estimated_pid_size is too small, in which
    // case we retry with a bigger buffer. Any other error is unexpected.
    if (errno != ENOMEM)
      return llvm::errorCodeToError(
          std::error_code(errno, std::generic_category()));
  }

  // The only way to exit the loop above is by repeatedly hitting ENOMEM. The
  // only way this can happen is if the process list somehow grew extremely
  // large between the two sysctl calls.
  assert(errno == ENOMEM &&
         "loop should only be left via the ENOMEM retry path");
  return llvm::createStringErrorV(
      "Failed to read process list: sysctl kept returning ENOMEM after {0} "
      "attempts",
      g_retry_count);
}
