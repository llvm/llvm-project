// Copyright 2021 Google Inc. All rights reserved.
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

#include "perf_counters.h"

#include <cstring>
#include <vector>

#if defined HAVE_LIBPFM
#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"
#endif

namespace benchmark {
namespace internal {

constexpr size_t PerfCounterValues::kMaxCounters;

#if defined HAVE_LIBPFM
const bool PerfCounters::kSupported = true;

bool PerfCounters::Initialize() { return pfm_initialize() == PFM_SUCCESS; }

PerfCounters PerfCounters::Create(
    const std::vector<std::string>& counter_names) {
  if (counter_names.empty()) {
    return NoCounters();
  }
  if (counter_names.size() > PerfCounterValues::kMaxCounters) {
    GetErrorLogInstance()
        << counter_names.size()
        << " counters were requested. The minimum is 1, the maximum is "
        << PerfCounterValues::kMaxCounters << "\n";
    return NoCounters();
  }
  std::vector<int> counter_ids(counter_names.size());

  const int mode = PFM_PLM3;  // user mode only
  for (size_t i = 0; i < counter_names.size(); ++i) {
    const bool is_first = i == 0;
    struct perf_event_attr attr {};
    attr.size = sizeof(attr);
    const int group_id = !is_first ? counter_ids[0] : -1;
    const auto& name = counter_names[i];
    if (name.empty()) {
      GetErrorLogInstance() << "A counter name was the empty string\n";
      return NoCounters();
    }
    pfm_perf_encode_arg_t arg{};
    arg.attr = &attr;

    const int pfm_get =
        pfm_get_os_event_encoding(name.c_str(), mode, PFM_OS_PERF_EVENT, &arg);
    if (pfm_get != PFM_SUCCESS) {
      GetErrorLogInstance() << "Unknown counter name: " << name << "\n";
      return NoCounters();
    }
    attr.disabled = is_first;
    // Note: the man page for perf_event_create suggests inerit = true and
    // read_format = PERF_FORMAT_GROUP don't work together, but that's not the
    // case.
    attr.inherit = true;
    attr.pinned = is_first;
    attr.exclude_kernel = true;
    attr.exclude_user = false;
    attr.exclude_hv = true;
    // Read all counters in one read.
    attr.read_format = PERF_FORMAT_GROUP;

    int id = -1;
    static constexpr size_t kNrOfSyscallRetries = 5;
    // Retry syscall as it was interrupted often (b/64774091).
    for (size_t num_retries = 0; num_retries < kNrOfSyscallRetries;
         ++num_retries) {
      id = perf_event_open(&attr, 0, -1, group_id, 0);
      if (id >= 0 || errno != EINTR) {
        break;
      }
    }
    if (id < 0) {
      GetErrorLogInstance()
          << "Failed to get a file descriptor for " << name << "\n";
      return NoCounters();
    }

    counter_ids[i] = id;
  }
  if (ioctl(counter_ids[0], PERF_EVENT_IOC_ENABLE) != 0) {
    GetErrorLogInstance() << "Failed to start counters\n";
    return NoCounters();
  }

  return PerfCounters(counter_names, std::move(counter_ids));
}

PerfCounters::~PerfCounters() {
  if (counter_ids_.empty()) {
    return;
  }
  ioctl(counter_ids_[0], PERF_EVENT_IOC_DISABLE);
  for (int fd : counter_ids_) {
    close(fd);
  }
}
#else   // defined HAVE_LIBPFM
const bool PerfCounters::kSupported = false;

bool PerfCounters::Initialize() { return false; }

PerfCounters PerfCounters::Create(
    const std::vector<std::string>& counter_names) {
  if (!counter_names.empty()) {
    GetErrorLogInstance() << "Performance counters not supported.";
  }
  return NoCounters();
}

PerfCounters::~PerfCounters() = default;
#endif  // defined HAVE_LIBPFM
}  // namespace internal
}  // namespace benchmark
