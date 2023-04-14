//===------ ompt_callback.cpp - Target RTLs Implementation -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OMPT support for AMDGPU
//
//===----------------------------------------------------------------------===//

//****************************************************************************
// global includes
//****************************************************************************

#include <atomic>
#include <mutex>
#include <string.h>
#include <sys/time.h>

//****************************************************************************
// local includes
//****************************************************************************

#include "llvm/Support/ErrorHandling.h"

#include <hsa/hsa_ext_amd.h>

#include <Debug.h>
#include <internal.h>
#include <ompt-connector.h>
#include <ompt_device_callbacks.h>

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

ompt_device_callbacks_t ompt_device_callbacks;

static double HostToDeviceRate = .0;
static double HostToDeviceSlope = .0;

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

/// Global function to enable/disable queue profiling for all devices
extern void ompt_enable_queue_profiling(int enable);

// These are the implementations in the device plugin/RTL
extern ompt_device_time_t devrtl_ompt_get_device_time(ompt_device_t *device);

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
      hsa_status_t err = hsa_amd_profiling_async_copy_enable(true /* enable */);
      if (err != HSA_STATUS_SUCCESS) {
        DP("Enabling profiling_async_copy returned %s, continuing\n",
           get_error_string(err));
      }
      // Enable queue dispatch profiling
      ompt_enable_queue_profiling(true /* enable */);
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
    hsa_status_t err = hsa_amd_profiling_async_copy_enable(false /* enable */);
    if (err != HSA_STATUS_SUCCESS) {
      DP("Disabling profiling_async_copy returned %s, continuing\n",
         get_error_string(err));
    }
    // Disable queue dispatch profiling
    ompt_enable_queue_profiling(false /* enable */);

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
  return devrtl_ompt_get_device_time(device);
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

//****************************************************************************
// private data
//****************************************************************************

static bool ompt_enabled = false;

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

ompt_interface_fn_t
ompt_device_callbacks_t::lookup(const char *interface_function_name) {
#define macro(fn)                                                              \
  if (strcmp(interface_function_name, #fn) == 0)                               \
    return (ompt_interface_fn_t)fn;

  FOREACH_TARGET_FN(macro);

#undef macro

  return (ompt_interface_fn_t)0;
}

/// @brief Helper to get the host time
/// @return  CLOCK_REALTIME seconds as double
static double getTimeOfDay() {
  double TimeVal = .0;
  struct timeval tval;
  int rc = gettimeofday(&tval, NULL);
  if (rc) {
    // XXX: Error case: What to do?
  } else {
    TimeVal = static_cast<double>(tval.tv_sec) +
              1.0E-06 * static_cast<double>(tval.tv_usec);
  }
  return TimeVal;
}

static int ompt_device_init(ompt_function_lookup_t lookup,
                            int initial_device_num, ompt_data_t *tool_data) {
  DP("OMPT: Enter ompt_device_init\n");
  ompt_device_t *dev = NULL; // TODO: Pass actual device

  // At init we capture two time points for host and device to calculate the
  // "average time" of those two times.
  // libomp uses the CLOCK_REALTIME (via gettimeofday) to get
  // the value for omp_get_wtime. So we use the same clock here to calculate
  // the rate and convert device time to omp_get_wtime via translate_time.
  double host_ref_a = getTimeOfDay();
  uint64_t device_ref_a =
      devrtl_ompt_get_device_time(dev) + 1; // +1 to erase potential div by zero

  ompt_enabled = true;

  LIBOMPTARGET_GET_TARGET_OPID =
      (ompt_get_target_info_t)lookup(stringify(LIBOMPTARGET_GET_TARGET_OPID));

  DP("OMPT: libomptarget_get_target_info = %p\n",
     fnptr_to_ptr(LIBOMPTARGET_GET_TARGET_OPID));

  ompt_device_callbacks.register_callbacks(lookup);

  double host_ref_b = getTimeOfDay();
  uint64_t device_ref_b =
      devrtl_ompt_get_device_time(dev) + 1; // +1 to erase potential div by zero

  // Multiply with .5 to reduce value range and potential risk of potential
  // overflow
  double host_avg = host_ref_b * 0.5 + host_ref_a * 0.5;
  uint64_t device_avg = device_ref_b * 0.5 + device_ref_a * 0.5;
  DP("Translate time: H1=%f D1=%lu H2=%f D2=%lu\n", host_ref_a, device_ref_a,
     host_ref_b, device_ref_b);
  HostToDeviceRate = host_avg / device_avg;
  DP("OMPT: Translate time HostToDeviceSlope: %f\n", HostToDeviceSlope);

  DP("OMPT: Exit ompt_device_init\n");

  return 0;
}

static void ompt_device_fini(ompt_data_t *tool_data) {
  DP("OMPT: executing amdgpu_ompt_device_fini\n");
}

#ifdef OMPT_SUPPORT
//****************************************************************************
// constructor
//****************************************************************************

__attribute__((constructor)) static void ompt_init(void) {
  DP("OMPT: Entering ompt_init\n");
  static library_ompt_connector_t libomptarget_connector("libomptarget");
  static ompt_start_tool_result_t ompt_result;

  ompt_result.initialize = ompt_device_init;
  ompt_result.finalize = ompt_device_fini;
  ompt_result.tool_data.value = 0;

  ompt_device_callbacks.init();
  libomptarget_connector.connect(&ompt_result);

  DP("OMPT: Exiting ompt_init\n");
}
#endif
