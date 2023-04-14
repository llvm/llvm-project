//===---------- OmptCallback.cpp - Generic OMPT callbacks --------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OMPT support for PluginInterface
//
//===----------------------------------------------------------------------===//

#ifdef OMPT_SUPPORT
#include <atomic>
#include <cstdio>
#include <string.h>
#include <vector>

#include "llvm/Support/ErrorHandling.h"

#include "Debug.h"
#include "ompt-connector.h"
#include "ompt_device_callbacks.h"

//****************************************************************************
// macros
//****************************************************************************

// Supported device tracing entry points
#define FOREACH_TARGET_FN(macro)                                               \
  macro(ompt_set_trace_ompt) macro(ompt_start_trace) macro(ompt_flush_trace)   \
      macro(ompt_stop_trace) macro(ompt_advance_buffer_cursor)                 \
          macro(ompt_get_record_ompt) macro(ompt_get_device_time)              \
              macro(ompt_get_record_type) macro(ompt_translate_time)

#define fnptr_to_ptr(x) ((void *)(uint64_t)x)

#define ompt_ptr_unknown ((void *)0)

#define OMPT_API_ROUTINE static

//****************************************************************************
// private data
//****************************************************************************

// Mutexes to protect the function pointers
static std::mutex set_trace_mutex;
static std::mutex start_trace_mutex;
static std::mutex flush_trace_mutex;
static std::mutex stop_trace_mutex;
static std::mutex advance_buffer_cursor_mutex;
static std::mutex get_record_type_mutex;

//****************************************************************************
// global data
//****************************************************************************

/// Object maintaining all the callbacks in the plugin
ompt_device_callbacks_t ompt_device_callbacks;

static double HostToDeviceRate = .0;

typedef ompt_set_result_t (*libomptarget_ompt_set_trace_ompt_t)(
    ompt_device_t *device, unsigned int enable, unsigned int etype);
typedef int (*libomptarget_ompt_start_trace_t)(ompt_callback_buffer_request_t,
                                               ompt_callback_buffer_complete_t);
typedef int (*libomptarget_ompt_flush_trace_t)(ompt_device_t *);
typedef int (*libomptarget_ompt_stop_trace_t)(ompt_device_t *);
typedef int (*libomptarget_ompt_advance_buffer_cursor_t)(
    ompt_device_t *, ompt_buffer_t *, size_t, ompt_buffer_cursor_t,
    ompt_buffer_cursor_t *);
typedef ompt_device_time_t (*libomptarget_ompt_get_device_time_t)(
    ompt_device_t *);
typedef ompt_record_t (*libomptarget_ompt_get_record_type_t)(
    ompt_buffer_t *, ompt_buffer_cursor_t);

libomptarget_ompt_set_trace_ompt_t ompt_set_trace_ompt_fn = nullptr;
libomptarget_ompt_start_trace_t ompt_start_trace_fn = nullptr;
libomptarget_ompt_flush_trace_t ompt_flush_trace_fn = nullptr;
libomptarget_ompt_stop_trace_t ompt_stop_trace_fn = nullptr;
libomptarget_ompt_advance_buffer_cursor_t ompt_advance_buffer_cursor_fn =
    nullptr;
libomptarget_ompt_get_record_type_t ompt_get_record_type_fn = nullptr;

extern void setOmptAsyncCopyProfile(bool Enable);
extern void setGlobalOmptKernelProfile(int DeviceId, int Enable);
extern uint64_t getSystemTimestampInNs();

// Runtime entry-points for device tracing

OMPT_API_ROUTINE ompt_set_result_t ompt_set_trace_ompt(ompt_device_t *device,
                                                       unsigned int enable,
                                                       unsigned int etype) {
  DP("Executing ompt_set_trace_ompt\n");

  // TODO handle device

  {
    // protect the function pointer
    std::unique_lock<std::mutex> lck(set_trace_mutex);
    // plugin specific
    ompt_device_callbacks.set_trace_ompt(device, enable, etype);
    // libomptarget specific
    if (!ompt_set_trace_ompt_fn) {
      auto libomptarget_dyn_lib = ompt_device_callbacks.get_parent_dyn_lib();
      if (libomptarget_dyn_lib != nullptr && libomptarget_dyn_lib->isValid()) {
        void *vptr = libomptarget_dyn_lib->getAddressOfSymbol(
            "libomptarget_ompt_set_trace_ompt");
        assert(vptr && "OMPT set trace ompt entry point not found");
        ompt_set_trace_ompt_fn =
            reinterpret_cast<libomptarget_ompt_set_trace_ompt_t>(vptr);
      }
    }
  }
  return ompt_set_trace_ompt_fn(device, enable, etype);
}

OMPT_API_ROUTINE int
ompt_start_trace(ompt_device_t *device, ompt_callback_buffer_request_t request,
                 ompt_callback_buffer_complete_t complete) {
  DP("OMPT: Executing ompt_start_trace\n");

  // TODO handle device

  {
    // protect the function pointer
    std::unique_lock<std::mutex> lck(start_trace_mutex);
    // plugin specific
    ompt_device_callbacks.set_buffer_request(request);
    ompt_device_callbacks.set_buffer_complete(complete);
    if (request && complete) {
      ompt_device_callbacks.set_tracing_enabled(true);
      // Enable asynchronous memory copy profiling
      setOmptAsyncCopyProfile(/*Enable=*/true);
      // Enable queue dispatch profiling
      setGlobalOmptKernelProfile(
          device != nullptr
              ? ompt_device_callbacks.lookup_device_id((ompt_device *)device)
              : 0,
          /*Enable=*/1);
    }

    // libomptarget specific
    if (!ompt_start_trace_fn) {
      auto libomptarget_dyn_lib = ompt_device_callbacks.get_parent_dyn_lib();
      if (libomptarget_dyn_lib != nullptr && libomptarget_dyn_lib->isValid()) {
        void *vptr = libomptarget_dyn_lib->getAddressOfSymbol(
            "libomptarget_ompt_start_trace");
        assert(vptr && "OMPT start trace entry point not found");
        ompt_start_trace_fn =
            reinterpret_cast<libomptarget_ompt_start_trace_t>(vptr);
      }
    }
  }
  return ompt_start_trace_fn(request, complete);
}

OMPT_API_ROUTINE int ompt_flush_trace(ompt_device_t *device) {
  DP("OMPT: Executing ompt_flush_trace\n");

  // TODO handle device

  {
    // Protect the function pointer
    std::unique_lock<std::mutex> lck(flush_trace_mutex);
    if (!ompt_flush_trace_fn) {
      auto libomptarget_dyn_lib = ompt_device_callbacks.get_parent_dyn_lib();
      if (libomptarget_dyn_lib != nullptr && libomptarget_dyn_lib->isValid()) {
        void *vptr = libomptarget_dyn_lib->getAddressOfSymbol(
            "libomptarget_ompt_flush_trace");
        assert(vptr && "OMPT flush trace entry point not found");
        ompt_flush_trace_fn =
            reinterpret_cast<libomptarget_ompt_flush_trace_t>(vptr);
      }
    }
  }
  return ompt_flush_trace_fn(device);
}

OMPT_API_ROUTINE int ompt_stop_trace(ompt_device_t *device) {
  DP("OMPT: Executing ompt_stop_trace\n");

  // TODO handle device
  {
    // Protect the function pointer
    std::unique_lock<std::mutex> lck(stop_trace_mutex);
    ompt_device_callbacks.set_tracing_enabled(false);
    // Disable asynchronous memory copy profiling
    setOmptAsyncCopyProfile(/*Enable=*/false);
    // Disable queue dispatch profiling
    setGlobalOmptKernelProfile(0, /*Enable=*/0);

    if (!ompt_stop_trace_fn) {
      auto libomptarget_dyn_lib = ompt_device_callbacks.get_parent_dyn_lib();
      if (libomptarget_dyn_lib != nullptr && libomptarget_dyn_lib->isValid()) {
        void *vptr = libomptarget_dyn_lib->getAddressOfSymbol(
            "libomptarget_ompt_stop_trace");
        assert(vptr && "OMPT stop trace entry point not found");
        ompt_stop_trace_fn =
            reinterpret_cast<libomptarget_ompt_stop_trace_t>(vptr);
      }
    }
  }
  return ompt_stop_trace_fn(device);
}

OMPT_API_ROUTINE ompt_record_ompt_t *
ompt_get_record_ompt(ompt_buffer_t *buffer, ompt_buffer_cursor_t current) {
  // TODO In debug mode, get the metadata associated with this buffer
  // and assert that there are enough bytes for the current record

  // Currently, no synchronization required since a disjoint set of
  // trace records is handed over to a thread.

  // Note that current can be nullptr. In that case, we return
  // nullptr. The tool has to handle that properly.
  return (ompt_record_ompt_t *)current;
}

OMPT_API_ROUTINE int
ompt_advance_buffer_cursor(ompt_device_t *device, ompt_buffer_t *buffer,
                           size_t size, /* bytes returned in the corresponding
                                           callback, unused here */
                           ompt_buffer_cursor_t current,
                           ompt_buffer_cursor_t *next) {
  // Advance can be called concurrently, so synchronize setting the
  // function pointer. The actual libomptarget function does not need
  // to be synchronized since it must be working on logically disjoint
  // buffers.
  {
    std::unique_lock<std::mutex> lck(advance_buffer_cursor_mutex);
    if (!ompt_advance_buffer_cursor_fn) {
      auto libomptarget_dyn_lib = ompt_device_callbacks.get_parent_dyn_lib();
      if (libomptarget_dyn_lib != nullptr && libomptarget_dyn_lib->isValid()) {
        void *vptr = libomptarget_dyn_lib->getAddressOfSymbol(
            "libomptarget_ompt_advance_buffer_cursor");
        assert(vptr && "OMPT advance buffer cursor entry point not found");
        ompt_advance_buffer_cursor_fn =
            reinterpret_cast<libomptarget_ompt_advance_buffer_cursor_t>(vptr);
      }
    }
  }
  return ompt_advance_buffer_cursor_fn(device, buffer, size, current, next);
}

OMPT_API_ROUTINE ompt_record_t
ompt_get_record_type(ompt_buffer_t *buffer, ompt_buffer_cursor_t current) {
  {
    std::unique_lock<std::mutex> lck(get_record_type_mutex);
    if (!ompt_get_record_type_fn) {
      auto libomptarget_dyn_lib = ompt_device_callbacks.get_parent_dyn_lib();
      if (libomptarget_dyn_lib != nullptr && libomptarget_dyn_lib->isValid()) {
        void *vptr = libomptarget_dyn_lib->getAddressOfSymbol(
            "libomptarget_ompt_get_record_type");
        assert(vptr && "OMPT get record type entry point not found");
        ompt_get_record_type_fn =
            reinterpret_cast<libomptarget_ompt_get_record_type_t>(vptr);
      }
    }
  }
  return ompt_get_record_type_fn(buffer, current);
}

OMPT_API_ROUTINE ompt_device_time_t
ompt_get_device_time(ompt_device_t *device) {
  DP("OMPT: Executing ompt_get_device_time\n");
  return getSystemTimestampInNs();
}

// Translates a device time to a meaningful timepoint in host time
OMPT_API_ROUTINE double ompt_translate_time(ompt_device_t *device,
                                            ompt_device_time_t device_time) {
  // We do not need to account for clock-skew / drift. So simple linear
  // translation using the host to device rate we obtained.
  double TranslatedTime = device_time * HostToDeviceRate;
  DP("OMPT: Translate time: %f\n", TranslatedTime);

  return TranslatedTime;
}

// End of runtime entry-points for trace records

/// Libomptarget function that will be used to set timestamps in trace records.
typedef void (*libomptarget_ompt_set_timestamp_t)(uint64_t start, uint64_t end);
libomptarget_ompt_set_timestamp_t ompt_set_timestamp_fn = nullptr;
std::mutex ompt_set_timestamp_mtx;

/// Set timestamps in trace records.
void setOmptTimestamp(uint64_t StartTime, uint64_t EndTime) {
  {
    std::unique_lock<std::mutex> timestamp_fn_lck(ompt_set_timestamp_mtx);
    if (!ompt_set_timestamp_fn) {
      auto libomptarget_dyn_lib = ompt_device_callbacks.get_parent_dyn_lib();
      if (libomptarget_dyn_lib == nullptr || !libomptarget_dyn_lib->isValid())
        return;
      void *vptr = libomptarget_dyn_lib->getAddressOfSymbol(
          "libomptarget_ompt_set_timestamp");
      if (!vptr)
        return;
      ompt_set_timestamp_fn =
          reinterpret_cast<libomptarget_ompt_set_timestamp_t>(vptr);
    }
  }
  // No need to hold a lock
  ompt_set_timestamp_fn(StartTime, EndTime);
}

void setOmptHostToDeviceRate(double Rate) { HostToDeviceRate = Rate; }

/// Libomptarget function that will be used to set num_teams in trace records.
typedef void (*libomptarget_ompt_set_granted_teams_t)(uint32_t);
libomptarget_ompt_set_granted_teams_t ompt_set_granted_teams_fn = nullptr;
std::mutex granted_teams_mtx;

/// Set granted number of teams in trace records.
void setOmptGrantedNumTeams(uint64_t NumTeams) {
  {
    std::unique_lock<std::mutex> granted_teams_fn_lck(granted_teams_mtx);
    if (!ompt_set_granted_teams_fn) {
      auto libomptarget_dyn_lib = ompt_device_callbacks.get_parent_dyn_lib();
      if (libomptarget_dyn_lib == nullptr || !libomptarget_dyn_lib->isValid())
        return;
      void *vptr = libomptarget_dyn_lib->getAddressOfSymbol(
          "libomptarget_ompt_set_granted_teams");
      if (!vptr)
        return;
      ompt_set_granted_teams_fn =
          reinterpret_cast<libomptarget_ompt_set_granted_teams_t>(vptr);
    }
  }
  // No need to hold a lock
  ompt_set_granted_teams_fn(NumTeams);
}

//****************************************************************************
// private data
//****************************************************************************

static ompt_get_target_info_t LIBOMPTARGET_GET_TARGET_OPID;
const char *ompt_device_callbacks_t::documentation = 0;
static ompt_device *devices = 0;

//****************************************************************************
// private operations
//****************************************************************************

void ompt_device_callbacks_t::resize(int number_of_devices) {
  devices = new ompt_device[number_of_devices];
}

ompt_device *ompt_device_callbacks_t::lookup_device(int device_num) {
  return &devices[device_num];
}

int ompt_device_callbacks_t::lookup_device_id(ompt_device *device) {
  for (int i = 0; i < num_devices; ++i)
    if (device == &devices[i])
      return i;
  llvm_unreachable("Lookup device id failed");
}

/// Lookup function used for querying callback functions maintained
/// by the plugin
ompt_interface_fn_t
ompt_device_callbacks_t::lookup(const char *InterfaceFunctionName) {
#define macro(fn)                                                              \
  if (strcmp(InterfaceFunctionName, #fn) == 0)                                 \
    return (ompt_interface_fn_t)fn;

  FOREACH_TARGET_FN(macro);

#undef macro

  return (ompt_interface_fn_t) nullptr;
}

/// Used to indicate whether OMPT was enabled for this library
bool OmptEnabled = false;

/// This function is passed to libomptarget as part of the OMPT connector
/// object. It is called by libomptarget during initialization of OMPT in the
/// plugin. \p lookup to be used to query callbacks registered with libomptarget
/// \p initial_device_num Initial device num provided by libomptarget
/// \p tool_data as provided by the tool
static int OmptDeviceInit(ompt_function_lookup_t lookup, int initial_device_num,
                          ompt_data_t *tool_data) {
  DP("OMPT: Enter OmptDeviceInit\n");
  OmptEnabled = true;

  LIBOMPTARGET_GET_TARGET_OPID =
      (ompt_get_target_info_t)lookup(stringify(LIBOMPTARGET_GET_TARGET_OPID));

  DP("OMPT: libomptarget_get_target_info = %p\n",
     fnptr_to_ptr(LIBOMPTARGET_GET_TARGET_OPID));

  // The lookup parameter is provided by libomptarget which already has the tool
  // callbacks registered at this point. The registration call below causes the
  // same callback functions to be registered in the plugin as well.
  ompt_device_callbacks.register_callbacks(lookup);

  DP("OMPT: Exit OmptDeviceInit\n");
  return 0;
}

/// This function is passed to libomptarget as part of the OMPT connector
/// object. It is called by libomptarget during finalization of OMPT in the
/// plugin.
static void OmptDeviceFini(ompt_data_t *tool_data) {
  DP("OMPT: Executing OmptDeviceFini\n");
}

//****************************************************************************
// constructor
//****************************************************************************
/// Used to initialize callbacks implemented by the tool. This interface will
/// lookup the callbacks table in libomptarget and assign them to the callbacks
/// table maintained in the calling plugin library.
void OmptCallbackInit() {
  DP("OMPT: Entering OmptCallbackInit\n");
  /// Connect plugin instance with libomptarget
  static library_ompt_connector_t libomptarget_connector("libomptarget");
  static ompt_start_tool_result_t OmptResult;

  // Initialize OmptResult with the init and fini functions that will be
  // called by the connector
  OmptResult.initialize = OmptDeviceInit;
  OmptResult.finalize = OmptDeviceFini;
  OmptResult.tool_data.value = 0;

  // Initialize the device callbacks first
  ompt_device_callbacks.init();

  // Now call connect that causes the above init/fini functions to be called
  libomptarget_connector.connect(&OmptResult);
  DP("OMPT: Exiting OmptCallbackInit\n");
}
#endif
