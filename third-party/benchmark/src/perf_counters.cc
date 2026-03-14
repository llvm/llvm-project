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
#include <memory>
#include <vector>

#if defined HAVE_LIBPFM
#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"
#endif

namespace benchmark {
namespace internal {

constexpr size_t PerfCounterValues::kMaxCounters;

#if defined HAVE_LIBPFM

size_t PerfCounterValues::Read(const std::vector<int>& leaders) {
  // Create a pointer for multiple reads
  const size_t bufsize = values_.size() * sizeof(values_[0]);
  char* ptr = reinterpret_cast<char*>(values_.data());
  size_t size = bufsize;
  for (int lead : leaders) {
    auto read_bytes = ::read(lead, ptr, size);
    if (read_bytes >= ssize_t(sizeof(uint64_t))) {
      // Actual data bytes are all bytes minus initial padding
      std::size_t data_bytes = read_bytes - sizeof(uint64_t);
      // This should be very cheap since it's in hot cache
      std::memmove(ptr, ptr + sizeof(uint64_t), data_bytes);
      // Increment our counters
      ptr += data_bytes;
      size -= data_bytes;
    } else {
      int err = errno;
      GetErrorLogInstance() << "Error reading lead " << lead << " errno:" << err
                            << " " << ::strerror(err) << "\n";
      return 0;
    }
  }
  return (bufsize - size) / sizeof(uint64_t);
}

const bool PerfCounters::kSupported = true;

// Initializes libpfm only on the first call.  Returns whether that single
// initialization was successful.
bool PerfCounters::Initialize() {
  // Function-scope static gets initialized only once on first call.
  static const bool success = []() {
    return pfm_initialize() == PFM_SUCCESS;
  }();
  return success;
}

bool PerfCounters::IsCounterSupported(const std::string& name) {
  Initialize();
  perf_event_attr_t attr;
  std::memset(&attr, 0, sizeof(attr));
  pfm_perf_encode_arg_t arg;
  std::memset(&arg, 0, sizeof(arg));
  arg.attr = &attr;
  const int mode = PFM_PLM3;  // user mode only
  int ret = pfm_get_os_event_encoding(name.c_str(), mode, PFM_OS_PERF_EVENT_EXT,
                                      &arg);
  return (ret == PFM_SUCCESS);
}

PerfCounters PerfCounters::Create(
    const std::vector<std::string>& counter_names) {
  if (!counter_names.empty()) {
    Initialize();
  }

  // Valid counters will populate these arrays but we start empty
  std::vector<std::string> valid_names;
  std::vector<int> counter_ids;
  std::vector<int> leader_ids;

  // Resize to the maximum possible
  valid_names.reserve(counter_names.size());
  counter_ids.reserve(counter_names.size());

  const int kCounterMode = PFM_PLM3;  // user mode only

  // Group leads will be assigned on demand. The idea is that once we cannot
  // create a counter descriptor, the reason is that this group has maxed out
  // so we set the group_id again to -1 and retry - giving the algorithm a
  // chance to create a new group leader to hold the next set of counters.
  int group_id = -1;

  // Loop through all performance counters
  for (size_t i = 0; i < counter_names.size(); ++i) {
    // we are about to push into the valid names vector
    // check if we did not reach the maximum
    if (valid_names.size() == PerfCounterValues::kMaxCounters) {
      // Log a message if we maxed out and stop adding
      GetErrorLogInstance()
          << counter_names.size() << " counters were requested. The maximum is "
          << PerfCounterValues::kMaxCounters << " and " << valid_names.size()
          << " were already added. All remaining counters will be ignored\n";
      // stop the loop and return what we have already
      break;
    }

    // Check if this name is empty
    const auto& name = counter_names[i];
    if (name.empty()) {
      GetErrorLogInstance()
          << "A performance counter name was the empty string\n";
      continue;
    }

    // Here first means first in group, ie the group leader
    const bool is_first = (group_id < 0);

    // This struct will be populated by libpfm from the counter string
    // and then fed into the syscall perf_event_open
    struct perf_event_attr attr {};
    attr.size = sizeof(attr);

    // This is the input struct to libpfm.
    pfm_perf_encode_arg_t arg{};
    arg.attr = &attr;
    const int pfm_get = pfm_get_os_event_encoding(name.c_str(), kCounterMode,
                                                  PFM_OS_PERF_EVENT, &arg);
    if (pfm_get != PFM_SUCCESS) {
      GetErrorLogInstance()
          << "Unknown performance counter name: " << name << "\n";
      continue;
    }

    // We then proceed to populate the remaining fields in our attribute struct
    // Note: the man page for perf_event_create suggests inherit = true and
    // read_format = PERF_FORMAT_GROUP don't work together, but that's not the
    // case.
    attr.disabled = is_first;
    attr.inherit = true;
    attr.pinned = is_first;
    attr.exclude_kernel = true;
    attr.exclude_user = false;
    attr.exclude_hv = true;

    // Read all counters in a group in one read.
    attr.read_format = PERF_FORMAT_GROUP;

    int id = -1;
    while (id < 0) {
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
        // If the file descriptor is negative we might have reached a limit
        // in the current group. Set the group_id to -1 and retry
        if (group_id >= 0) {
          // Create a new group
          group_id = -1;
        } else {
          // At this point we have already retried to set a new group id and
          // failed. We then give up.
          break;
        }
      }
    }

    // We failed to get a new file descriptor. We might have reached a hard
    // hardware limit that cannot be resolved even with group multiplexing
    if (id < 0) {
      GetErrorLogInstance() << "***WARNING** Failed to get a file descriptor "
                               "for performance counter "
                            << name << ". Ignoring\n";

      // We give up on this counter but try to keep going
      // as the others would be fine
      continue;
    }
    if (group_id < 0) {
      // This is a leader, store and assign it to the current file descriptor
      leader_ids.push_back(id);
      group_id = id;
    }
    // This is a valid counter, add it to our descriptor's list
    counter_ids.push_back(id);
    valid_names.push_back(name);
  }

  // Loop through all group leaders activating them
  // There is another option of starting ALL counters in a process but
  // that would be far reaching an intrusion. If the user is using PMCs
  // by themselves then this would have a side effect on them. It is
  // friendlier to loop through all groups individually.
  for (int lead : leader_ids) {
    if (ioctl(lead, PERF_EVENT_IOC_ENABLE) != 0) {
      // This should never happen but if it does, we give up on the
      // entire batch as recovery would be a mess.
      GetErrorLogInstance() << "***WARNING*** Failed to start counters. "
                               "Claring out all counters.\n";

      // Close all peformance counters
      for (int id : counter_ids) {
        ::close(id);
      }

      // Return an empty object so our internal state is still good and
      // the process can continue normally without impact
      return NoCounters();
    }
  }

  return PerfCounters(std::move(valid_names), std::move(counter_ids),
                      std::move(leader_ids));
}

void PerfCounters::CloseCounters() const {
  if (counter_ids_.empty()) {
    return;
  }
  for (int lead : leader_ids_) {
    ioctl(lead, PERF_EVENT_IOC_DISABLE);
  }
  for (int fd : counter_ids_) {
    close(fd);
  }
}
#else   // defined HAVE_LIBPFM
size_t PerfCounterValues::Read(const std::vector<int>&) { return 0; }

const bool PerfCounters::kSupported = false;

bool PerfCounters::Initialize() { return false; }

bool PerfCounters::IsCounterSupported(const std::string&) { return false; }

PerfCounters PerfCounters::Create(
    const std::vector<std::string>& counter_names) {
  if (!counter_names.empty()) {
    GetErrorLogInstance() << "Performance counters not supported.\n";
  }
  return NoCounters();
}

void PerfCounters::CloseCounters() const {}
#endif  // defined HAVE_LIBPFM

PerfCountersMeasurement::PerfCountersMeasurement(
    const std::vector<std::string>& counter_names)
    : start_values_(counter_names.size()), end_values_(counter_names.size()) {
  counters_ = PerfCounters::Create(counter_names);
}

PerfCounters& PerfCounters::operator=(PerfCounters&& other) noexcept {
  if (this != &other) {
    CloseCounters();

    counter_ids_ = std::move(other.counter_ids_);
    leader_ids_ = std::move(other.leader_ids_);
    counter_names_ = std::move(other.counter_names_);
  }
  return *this;
}
}  // namespace internal
}  // namespace benchmark
