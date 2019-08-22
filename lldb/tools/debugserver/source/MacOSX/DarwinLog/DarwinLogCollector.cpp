//===-- DarwinLogCollector.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DarwinLogCollector.h"
#include "ActivityStreamSPI.h"

#include <dlfcn.h>

#include <cinttypes>
#include <memory>
#include <mutex>
#include <vector>

#include "DNB.h"
#include "DNBLog.h"
#include "DarwinLogTypes.h"
#include "LogFilterChain.h"
#include "LogFilterExactMatch.h"
#include "LogFilterRegex.h"
#include "LogMessageOsLog.h"
#include "MachProcess.h"
#include "RNBContext.h"
#include "RNBDefs.h"
#include "RNBRemote.h"

// Use an anonymous namespace for variables and methods that have no
// reason to leak out through the interface.
namespace {
/// Specify max depth that the activity parent-child chain will search
/// back to get the full activity chain name.  If we do more than this,
/// we assume either we hit a loop or it's just too long.
static const size_t MAX_ACTIVITY_CHAIN_DEPTH = 10;

// Used to tap into and retrieve logs from target process.
// (Consumer of os_log).
static os_activity_stream_for_pid_t s_os_activity_stream_for_pid;
static os_activity_stream_resume_t s_os_activity_stream_resume;
static os_activity_stream_cancel_t s_os_activity_stream_cancel;
static os_log_copy_formatted_message_t s_os_log_copy_formatted_message;
static os_activity_stream_set_event_handler_t
    s_os_activity_stream_set_event_handler;

bool LookupSPICalls() {
  static std::once_flag s_once_flag;
  static bool s_has_spi;

  std::call_once(s_once_flag, [] {
    dlopen ("/System/Library/PrivateFrameworks/LoggingSupport.framework/LoggingSupport", RTLD_NOW);
    s_os_activity_stream_for_pid = (os_activity_stream_for_pid_t)dlsym(
        RTLD_DEFAULT, "os_activity_stream_for_pid");
    s_os_activity_stream_resume = (os_activity_stream_resume_t)dlsym(
        RTLD_DEFAULT, "os_activity_stream_resume");
    s_os_activity_stream_cancel = (os_activity_stream_cancel_t)dlsym(
        RTLD_DEFAULT, "os_activity_stream_cancel");
    s_os_log_copy_formatted_message = (os_log_copy_formatted_message_t)dlsym(
        RTLD_DEFAULT, "os_log_copy_formatted_message");
    s_os_activity_stream_set_event_handler =
        (os_activity_stream_set_event_handler_t)dlsym(
            RTLD_DEFAULT, "os_activity_stream_set_event_handler");

    // We'll indicate we're all set if every function entry point
    // was found.
    s_has_spi = (s_os_activity_stream_for_pid != nullptr) &&
                (s_os_activity_stream_resume != nullptr) &&
                (s_os_activity_stream_cancel != nullptr) &&
                (s_os_log_copy_formatted_message != nullptr) &&
                (s_os_activity_stream_set_event_handler != nullptr);
    if (s_has_spi) {
      DNBLogThreadedIf(LOG_DARWIN_LOG, "Found os_log SPI calls.");
      // Tell LogMessageOsLog how to format messages when search
      // criteria requires it.
      LogMessageOsLog::SetFormatterFunction(s_os_log_copy_formatted_message);
    } else {
      DNBLogThreadedIf(LOG_DARWIN_LOG, "Failed to find os_log SPI "
                                       "calls.");
    }
  });

  return s_has_spi;
}

using Mutex = std::mutex;
static Mutex s_collector_mutex;
static std::vector<DarwinLogCollectorSP> s_collectors;

static void TrackCollector(const DarwinLogCollectorSP &collector_sp) {
  std::lock_guard<Mutex> locker(s_collector_mutex);
  if (std::find(s_collectors.begin(), s_collectors.end(), collector_sp) !=
      s_collectors.end()) {
    DNBLogThreadedIf(LOG_DARWIN_LOG,
                     "attempted to add same collector multiple times");
    return;
  }
  s_collectors.push_back(collector_sp);
}

static void StopTrackingCollector(const DarwinLogCollectorSP &collector_sp) {
  std::lock_guard<Mutex> locker(s_collector_mutex);
  s_collectors.erase(
      std::remove(s_collectors.begin(), s_collectors.end(), collector_sp),
      s_collectors.end());
}

static DarwinLogCollectorSP FindCollectorForProcess(pid_t pid) {
  std::lock_guard<Mutex> locker(s_collector_mutex);
  for (const auto &collector_sp : s_collectors) {
    if (collector_sp && (collector_sp->GetProcessID() == pid))
      return collector_sp;
  }
  return DarwinLogCollectorSP();
}

static FilterTarget TargetStringToEnum(const std::string &filter_target_name) {
  if (filter_target_name == "activity")
    return eFilterTargetActivity;
  else if (filter_target_name == "activity-chain")
    return eFilterTargetActivityChain;
  else if (filter_target_name == "category")
    return eFilterTargetCategory;
  else if (filter_target_name == "message")
    return eFilterTargetMessage;
  else if (filter_target_name == "subsystem")
    return eFilterTargetSubsystem;
  else
    return eFilterTargetInvalid;
}

class Configuration {
public:
  Configuration(const JSONObject &config)
      : m_is_valid(false),
        m_activity_stream_flags(OS_ACTIVITY_STREAM_PROCESS_ONLY),
        m_filter_chain_sp(nullptr) {
    // Parse out activity stream flags
    if (!ParseSourceFlags(config)) {
      m_is_valid = false;
      return;
    }

    // Parse filter rules
    if (!ParseFilterRules(config)) {
      m_is_valid = false;
      return;
    }

    // Everything worked.
    m_is_valid = true;
  }

  bool ParseSourceFlags(const JSONObject &config) {
    // Get the source-flags dictionary.
    auto source_flags_sp = config.GetObject("source-flags");
    if (!source_flags_sp)
      return false;
    if (!JSONObject::classof(source_flags_sp.get()))
      return false;

    const JSONObject &source_flags =
        *static_cast<JSONObject *>(source_flags_sp.get());

    // Parse out the flags.
    bool include_any_process = false;
    bool include_callstacks = false;
    bool include_info_level = false;
    bool include_debug_level = false;
    bool live_stream = false;

    if (!source_flags.GetObjectAsBool("any-process", include_any_process)) {
      DNBLogThreadedIf(LOG_DARWIN_LOG, "Source-flag 'any-process' missing from "
                                       "configuration.");
      return false;
    }
    if (!source_flags.GetObjectAsBool("callstacks", include_callstacks)) {
      // We currently suppress the availability of this on the lldb
      // side.  We include here for devices when we enable in the
      // future.
      // DNBLogThreadedIf(LOG_DARWIN_LOG,
      //                  "Source-flag 'callstacks' missing from "
      //                  "configuration.");

      // OK.  We just skip callstacks.
      // return false;
    }
    if (!source_flags.GetObjectAsBool("info-level", include_info_level)) {
      DNBLogThreadedIf(LOG_DARWIN_LOG, "Source-flag 'info-level' missing from "
                                       "configuration.");
      return false;
    }
    if (!source_flags.GetObjectAsBool("debug-level", include_debug_level)) {
      DNBLogThreadedIf(LOG_DARWIN_LOG, "Source-flag 'debug-level' missing from "
                                       "configuration.");
      return false;
    }
    if (!source_flags.GetObjectAsBool("live-stream", live_stream)) {
      DNBLogThreadedIf(LOG_DARWIN_LOG, "Source-flag 'live-stream' missing from "
                                       "configuration.");
      return false;
    }

    // Setup the SPI flags based on this.
    m_activity_stream_flags = 0;
    if (!include_any_process)
      m_activity_stream_flags |= OS_ACTIVITY_STREAM_PROCESS_ONLY;
    if (include_callstacks)
      m_activity_stream_flags |= OS_ACTIVITY_STREAM_CALLSTACK;
    if (include_info_level)
      m_activity_stream_flags |= OS_ACTIVITY_STREAM_INFO;
    if (include_debug_level)
      m_activity_stream_flags |= OS_ACTIVITY_STREAM_DEBUG;
    if (!live_stream)
      m_activity_stream_flags |= OS_ACTIVITY_STREAM_BUFFERED;

    DNBLogThreadedIf(LOG_DARWIN_LOG, "m_activity_stream_flags = 0x%03x",
                     m_activity_stream_flags);

    return true;
  }

  bool ParseFilterRules(const JSONObject &config) {
    // Retrieve the default rule.
    bool filter_default_accept = true;
    if (!config.GetObjectAsBool("filter-fall-through-accepts",
                                filter_default_accept)) {
      DNBLogThreadedIf(LOG_DARWIN_LOG, "Setting 'filter-fall-through-accepts' "
                                       "missing from configuration.");
      return false;
    }
    m_filter_chain_sp = std::make_shared<LogFilterChain>(filter_default_accept);
    DNBLogThreadedIf(LOG_DARWIN_LOG, "DarwinLog no-match rule: %s.",
                     filter_default_accept ? "accept" : "reject");

    // If we don't have the filter-rules array, we're done.
    auto filter_rules_sp = config.GetObject("filter-rules");
    if (!filter_rules_sp) {
      DNBLogThreadedIf(LOG_DARWIN_LOG,
                       "No 'filter-rules' config element, all log "
                       "entries will use the no-match action (%s).",
                       filter_default_accept ? "accept" : "reject");
      return true;
    }
    if (!JSONArray::classof(filter_rules_sp.get()))
      return false;
    const JSONArray &rules_config =
        *static_cast<JSONArray *>(filter_rules_sp.get());

    // Create the filters.
    for (auto &rule_sp : rules_config.m_elements) {
      if (!JSONObject::classof(rule_sp.get()))
        return false;
      const JSONObject &rule_config = *static_cast<JSONObject *>(rule_sp.get());

      // Get whether this filter accepts or rejects.
      bool filter_accepts = true;
      if (!rule_config.GetObjectAsBool("accept", filter_accepts)) {
        DNBLogThreadedIf(LOG_DARWIN_LOG, "Filter 'accept' element missing.");
        return false;
      }

      // Grab the target log field attribute for the match.
      std::string target_attribute;
      if (!rule_config.GetObjectAsString("attribute", target_attribute)) {
        DNBLogThreadedIf(LOG_DARWIN_LOG, "Filter 'attribute' element missing.");
        return false;
      }
      auto target_enum = TargetStringToEnum(target_attribute);
      if (target_enum == eFilterTargetInvalid) {
        DNBLogThreadedIf(LOG_DARWIN_LOG, "Filter attribute '%s' unsupported.",
                         target_attribute.c_str());
        return false;
      }

      // Handle operation-specific fields and filter creation.
      std::string filter_type;
      if (!rule_config.GetObjectAsString("type", filter_type)) {
        DNBLogThreadedIf(LOG_DARWIN_LOG, "Filter 'type' element missing.");
        return false;
      }
      DNBLogThreadedIf(LOG_DARWIN_LOG, "Reading filter of type '%s'",
                       filter_type.c_str());

      LogFilterSP filter_sp;
      if (filter_type == "regex") {
        // Grab the regex for the match.
        std::string regex;
        if (!rule_config.GetObjectAsString("regex", regex)) {
          DNBLogError("Regex filter missing 'regex' element.");
          return false;
        }
        DNBLogThreadedIf(LOG_DARWIN_LOG, "regex for filter: \"%s\"",
                         regex.c_str());

        // Create the regex filter.
        auto regex_filter =
            new LogFilterRegex(filter_accepts, target_enum, regex);
        filter_sp.reset(regex_filter);

        // Validate that the filter is okay.
        if (!regex_filter->IsValid()) {
          DNBLogError("Invalid regex in filter: "
                      "regex=\"%s\", error=%s",
                      regex.c_str(), regex_filter->GetErrorAsCString());
          return false;
        }
      } else if (filter_type == "match") {
        // Grab the regex for the match.
        std::string exact_text;
        if (!rule_config.GetObjectAsString("exact_text", exact_text)) {
          DNBLogError("Exact match filter missing "
                      "'exact_text' element.");
          return false;
        }

        // Create the filter.
        filter_sp = std::make_shared<LogFilterExactMatch>(
            filter_accepts, target_enum, exact_text);
      }

      // Add the filter to the chain.
      m_filter_chain_sp->AppendFilter(filter_sp);
    }
    return true;
  }

  bool IsValid() const { return m_is_valid; }

  os_activity_stream_flag_t GetActivityStreamFlags() const {
    return m_activity_stream_flags;
  }

  const LogFilterChainSP &GetLogFilterChain() const {
    return m_filter_chain_sp;
  }

private:
  bool m_is_valid;
  os_activity_stream_flag_t m_activity_stream_flags;
  LogFilterChainSP m_filter_chain_sp;
};
}

bool DarwinLogCollector::IsSupported() {
  // We're supported if we have successfully looked up the SPI entry points.
  return LookupSPICalls();
}

bool DarwinLogCollector::StartCollectingForProcess(nub_process_t pid,
                                                   const JSONObject &config) {
  // If we're currently collecting for this process, kill the existing
  // collector.
  if (CancelStreamForProcess(pid)) {
    DNBLogThreadedIf(LOG_DARWIN_LOG,
                     "%s() killed existing DarwinLog collector for pid %d.",
                     __FUNCTION__, pid);
  }

  // If the process isn't alive, we're done.
  if (!DNBProcessIsAlive(pid)) {
    DNBLogThreadedIf(LOG_DARWIN_LOG,
                     "%s() cannot collect for pid %d: process not alive.",
                     __FUNCTION__, pid);
    return false;
  }

  // Validate the configuration.
  auto spi_config = Configuration(config);
  if (!spi_config.IsValid()) {
    DNBLogThreadedIf(LOG_DARWIN_LOG,
                     "%s() invalid configuration, will not enable log "
                     "collection",
                     __FUNCTION__);
    return false;
  }

  // Create the stream collector that will manage collected data
  // for this pid.
  DarwinLogCollectorSP collector_sp(
      new DarwinLogCollector(pid, spi_config.GetLogFilterChain()));
  std::weak_ptr<DarwinLogCollector> collector_wp(collector_sp);

  // Setup the stream handling block.
  os_activity_stream_block_t block =
      ^bool(os_activity_stream_entry_t entry, int error) {
        // Check if our collector is still alive.
        DarwinLogCollectorSP inner_collector_sp = collector_wp.lock();
        if (!inner_collector_sp)
          return false;
        return inner_collector_sp->HandleStreamEntry(entry, error);
      };

  os_activity_stream_event_block_t stream_event_block = ^void(
      os_activity_stream_t stream, os_activity_stream_event_t event) {
    switch (event) {
    case OS_ACTIVITY_STREAM_EVENT_STARTED:
      DNBLogThreadedIf(LOG_DARWIN_LOG,
                       "received stream event: "
                       "OS_ACTIVITY_STREAM_EVENT_STARTED, stream %p.",
                       (void *)stream);
      break;
    case OS_ACTIVITY_STREAM_EVENT_STOPPED:
      DNBLogThreadedIf(LOG_DARWIN_LOG,
                       "received stream event: "
                       "OS_ACTIVITY_STREAM_EVENT_STOPPED, stream %p.",
                       (void *)stream);
      break;
    case OS_ACTIVITY_STREAM_EVENT_FAILED:
      DNBLogThreadedIf(LOG_DARWIN_LOG,
                       "received stream event: "
                       "OS_ACTIVITY_STREAM_EVENT_FAILED, stream %p.",
                       (void *)stream);
      break;
    case OS_ACTIVITY_STREAM_EVENT_CHUNK_STARTED:
      DNBLogThreadedIf(LOG_DARWIN_LOG,
                       "received stream event: "
                       "OS_ACTIVITY_STREAM_EVENT_CHUNK_STARTED, stream %p.",
                       (void *)stream);
      break;
    case OS_ACTIVITY_STREAM_EVENT_CHUNK_FINISHED:
      DNBLogThreadedIf(LOG_DARWIN_LOG,
                       "received stream event: "
                       "OS_ACTIVITY_STREAM_EVENT_CHUNK_FINISHED, stream %p.",
                       (void *)stream);
      break;
    }
  };

  // Create the stream.
  os_activity_stream_t activity_stream = (*s_os_activity_stream_for_pid)(
      pid, spi_config.GetActivityStreamFlags(), block);
  collector_sp->SetActivityStream(activity_stream);

  // Specify the stream-related event handler.
  (*s_os_activity_stream_set_event_handler)(activity_stream,
                                            stream_event_block);

  // Start the stream.
  (*s_os_activity_stream_resume)(activity_stream);

  TrackCollector(collector_sp);
  return true;
}

DarwinLogEventVector
DarwinLogCollector::GetEventsForProcess(nub_process_t pid) {
  auto collector_sp = FindCollectorForProcess(pid);
  if (!collector_sp) {
    // We're not tracking a stream for this process.
    return DarwinLogEventVector();
  }

  return collector_sp->RemoveEvents();
}

bool DarwinLogCollector::CancelStreamForProcess(nub_process_t pid) {
  auto collector_sp = FindCollectorForProcess(pid);
  if (!collector_sp) {
    // We're not tracking a stream for this process.
    return false;
  }

  collector_sp->CancelActivityStream();
  StopTrackingCollector(collector_sp);

  return true;
}

const char *
DarwinLogCollector::GetActivityForID(os_activity_id_t activity_id) const {
  auto find_it = m_activity_map.find(activity_id);
  return (find_it != m_activity_map.end()) ? find_it->second.m_name.c_str()
                                           : nullptr;
}

/// Retrieve the full parent-child chain for activity names.  These
/// can be arbitrarily deep.  This method assumes the caller has already
/// locked the activity mutex.
void DarwinLogCollector::GetActivityChainForID_internal(
    os_activity_id_t activity_id, std::string &result, size_t depth) const {
  if (depth > MAX_ACTIVITY_CHAIN_DEPTH) {
    // Terminating condition - too deeply nested.
    return;
  } else if (activity_id == 0) {
    // Terminating condition - no activity.
    return;
  }

  auto find_it = m_activity_map.find(activity_id);
  if (find_it == m_activity_map.end()) {
    // Terminating condition - no data for activity_id.
    return;
  }

  // Activity name becomes parent activity name chain + ':' + our activity
  // name.
  GetActivityChainForID_internal(find_it->second.m_parent_id, result,
                                 depth + 1);
  if (!result.empty())
    result += ':';
  result += find_it->second.m_name;
}

std::string
DarwinLogCollector::GetActivityChainForID(os_activity_id_t activity_id) const {
  std::string result;
  {
    std::lock_guard<std::mutex> locker(m_activity_info_mutex);
    GetActivityChainForID_internal(activity_id, result, 1);
  }
  return result;
}

DarwinLogCollector::DarwinLogCollector(nub_process_t pid,
                                       const LogFilterChainSP &filter_chain_sp)
    : ActivityStore(), m_pid(pid), m_activity_stream(0), m_events(),
      m_events_mutex(), m_filter_chain_sp(filter_chain_sp),
      m_activity_info_mutex(), m_activity_map() {}

DarwinLogCollector::~DarwinLogCollector() {
  // Cancel the stream.
  if (m_activity_stream) {
    DNBLogThreadedIf(LOG_DARWIN_LOG, "tearing down activity stream "
                                     "collector for %d",
                     m_pid);
    (*s_os_activity_stream_cancel)(m_activity_stream);
    m_activity_stream = 0;
  } else {
    DNBLogThreadedIf(LOG_DARWIN_LOG, "no stream to tear down for %d", m_pid);
  }
}

void DarwinLogCollector::SignalDataAvailable() {
  RNBRemoteSP remoteSP(g_remoteSP);
  if (!remoteSP) {
    // We're done.  This is unexpected.
    StopTrackingCollector(shared_from_this());
    return;
  }

  RNBContext &ctx = remoteSP->Context();
  ctx.Events().SetEvents(RNBContext::event_darwin_log_data_available);
  // Wait for the main thread to consume this notification if it requested
  // we wait for it.
  ctx.Events().WaitForResetAck(RNBContext::event_darwin_log_data_available);
}

void DarwinLogCollector::SetActivityStream(
    os_activity_stream_t activity_stream) {
  m_activity_stream = activity_stream;
}

bool DarwinLogCollector::HandleStreamEntry(os_activity_stream_entry_t entry,
                                           int error) {
  if ((error == 0) && (entry != nullptr)) {
    if (entry->pid != m_pid) {
      // For now, skip messages not originating from our process.
      // Later we might want to keep all messages related to an event
      // that we're tracking, even when it came from another process,
      // possibly doing work on our behalf.
      return true;
    }

    switch (entry->type) {
    case OS_ACTIVITY_STREAM_TYPE_ACTIVITY_CREATE:
      DNBLogThreadedIf(
          LOG_DARWIN_LOG, "received activity create: "
                          "%s, creator aid %" PRIu64 ", unique_pid %" PRIu64
                          "(activity id=%" PRIu64 ", parent id=%" PRIu64 ")",
          entry->activity_create.name, entry->activity_create.creator_aid,
          entry->activity_create.unique_pid, entry->activity_id,
          entry->parent_id);
      {
        std::lock_guard<std::mutex> locker(m_activity_info_mutex);
        m_activity_map.insert(
            std::make_pair(entry->activity_id,
                           ActivityInfo(entry->activity_create.name,
                                        entry->activity_id, entry->parent_id)));
      }
      break;

    case OS_ACTIVITY_STREAM_TYPE_ACTIVITY_TRANSITION:
      DNBLogThreadedIf(
          LOG_DARWIN_LOG, "received activity transition:"
                          "new aid: %" PRIu64 "(activity id=%" PRIu64
                          ", parent id=%" PRIu64 ", tid %" PRIu64 ")",
          entry->activity_transition.transition_id, entry->activity_id,
          entry->parent_id, entry->activity_transition.thread);
      break;

    case OS_ACTIVITY_STREAM_TYPE_LOG_MESSAGE: {
      DNBLogThreadedIf(
          LOG_DARWIN_LOG, "received log message: "
                          "(activity id=%" PRIu64 ", parent id=%" PRIu64 ", "
                          "tid %" PRIu64 "): format %s",
          entry->activity_id, entry->parent_id, entry->log_message.thread,
          entry->log_message.format ? entry->log_message.format
                                    : "<invalid-format>");

      // Do the real work here.
      {
        // Ensure our process is still alive.  If not, we can
        // cancel the collection.
        if (!DNBProcessIsAlive(m_pid)) {
          // We're outta here.  This is the manner in which we
          // stop collecting for a process.
          StopTrackingCollector(shared_from_this());
          return false;
        }

        LogMessageOsLog os_log_message(*this, *entry);
        if (!m_filter_chain_sp ||
            !m_filter_chain_sp->GetAcceptMessage(os_log_message)) {
          // This log message was rejected by the filter,
          // so stop processing it now.
          return true;
        }

        // Copy over the relevant bits from the message.
        const struct os_log_message_s &log_message = entry->log_message;

        DarwinLogEventSP message_sp(new DarwinLogEvent());
        // Indicate this event is a log message event.
        message_sp->AddStringItem("type", "log");

        // Add the message contents (fully expanded).
        // Consider expanding on the remote side.
        // Then we don't pay for expansion until when it is
        // used.
        const char *message_text = os_log_message.GetMessage();
        if (message_text)
          message_sp->AddStringItem("message", message_text);

        // Add some useful data fields.
        message_sp->AddIntegerItem("timestamp", log_message.timestamp);

        // Do we want to do all activity name resolution on this
        // side?  Maybe.  For now, send IDs and ID->name mappings
        // and fix this up on that side.  Later, when we add
        // debugserver-side filtering, we'll want to get the
        // activity names over here, so we should probably
        // just send them as resolved strings.
        message_sp->AddIntegerItem("activity_id", entry->activity_id);
        message_sp->AddIntegerItem("parent_id", entry->parent_id);
        message_sp->AddIntegerItem("thread_id", log_message.thread);
        if (log_message.subsystem && strlen(log_message.subsystem) > 0)
          message_sp->AddStringItem("subsystem", log_message.subsystem);
        if (log_message.category && strlen(log_message.category) > 0)
          message_sp->AddStringItem("category", log_message.category);
        if (entry->activity_id != 0) {
          std::string activity_chain =
              GetActivityChainForID(entry->activity_id);
          if (!activity_chain.empty())
            message_sp->AddStringItem("activity-chain", activity_chain);
        }

        // Add it to the list for later collection.
        {
          std::lock_guard<std::mutex> locker(m_events_mutex);
          m_events.push_back(message_sp);
        }
        SignalDataAvailable();
      }
      break;
    }
    }
  } else {
    DNBLogThreadedIf(LOG_DARWIN_LOG, "HandleStreamEntry: final call, "
                                     "error %d",
                     error);
  }
  return true;
}

DarwinLogEventVector DarwinLogCollector::RemoveEvents() {
  DarwinLogEventVector returned_events;
  {
    std::lock_guard<std::mutex> locker(m_events_mutex);
    returned_events.swap(m_events);
  }
  DNBLogThreadedIf(LOG_DARWIN_LOG, "DarwinLogCollector::%s(): removing %lu "
                                   "queued log entries",
                   __FUNCTION__, returned_events.size());
  return returned_events;
}

void DarwinLogCollector::CancelActivityStream() {
  if (!m_activity_stream)
    return;

  DNBLogThreadedIf(LOG_DARWIN_LOG, "DarwinLogCollector::%s(): canceling "
                                   "activity stream %p",
                   __FUNCTION__, reinterpret_cast<void *>(m_activity_stream));
  (*s_os_activity_stream_cancel)(m_activity_stream);
  m_activity_stream = nullptr;
}
