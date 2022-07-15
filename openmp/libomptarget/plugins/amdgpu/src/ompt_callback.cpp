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
#include <vector>

#include <dlfcn.h>
#include <string.h>

//****************************************************************************
// local includes
//****************************************************************************

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
          macro(ompt_get_record_ompt)

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

//****************************************************************************
// global data
//****************************************************************************

ompt_device_callbacks_t ompt_device_callbacks;

typedef ompt_set_result_t (*libomptarget_ompt_set_trace_ompt_t)(
    ompt_device_t *device, unsigned int enable, unsigned int etype);
typedef int (*libomptarget_ompt_start_trace_t)(ompt_callback_buffer_request_t,
                                               ompt_callback_buffer_complete_t);
typedef int (*libomptarget_ompt_flush_trace_t)(ompt_device_t *);
typedef int (*libomptarget_ompt_stop_trace_t)(ompt_device_t *);
typedef int (*libomptarget_ompt_advance_buffer_cursor_t)(
    ompt_device_t *, ompt_buffer_t *, size_t, ompt_buffer_cursor_t,
    ompt_buffer_cursor_t *);

libomptarget_ompt_set_trace_ompt_t ompt_set_trace_ompt_fn = nullptr;
libomptarget_ompt_start_trace_t ompt_start_trace_fn = nullptr;
libomptarget_ompt_flush_trace_t ompt_flush_trace_fn = nullptr;
libomptarget_ompt_stop_trace_t ompt_stop_trace_fn = nullptr;
libomptarget_ompt_advance_buffer_cursor_t ompt_advance_buffer_cursor_fn =
    nullptr;

/// Global function to enable/disable queue profiling for all devices
extern void ompt_enable_queue_profiling(int enable);

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
      void *vptr = dlsym(NULL, "libomptarget_ompt_set_trace_ompt");
      assert(vptr && "OMPT set trace ompt entry point not found");
      ompt_set_trace_ompt_fn =
          reinterpret_cast<libomptarget_ompt_set_trace_ompt_t>(vptr);
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
      void *vptr = dlsym(NULL, "libomptarget_ompt_start_trace");
      assert(vptr && "OMPT start trace entry point not found");
      ompt_start_trace_fn =
          reinterpret_cast<libomptarget_ompt_start_trace_t>(vptr);
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
      void *vptr = dlsym(NULL, "libomptarget_ompt_flush_trace");
      assert(vptr && "OMPT flush trace entry point not found");
      ompt_flush_trace_fn =
          reinterpret_cast<libomptarget_ompt_flush_trace_t>(vptr);
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
      void *vptr = dlsym(NULL, "libomptarget_ompt_stop_trace");
      assert(vptr && "OMPT stop trace entry point not found");
      ompt_stop_trace_fn =
          reinterpret_cast<libomptarget_ompt_stop_trace_t>(vptr);
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
      void *vptr = dlsym(NULL, "libomptarget_ompt_advance_buffer_cursor");
      assert(vptr && "OMPT advance buffer cursor entry point not found");
      ompt_advance_buffer_cursor_fn =
          reinterpret_cast<libomptarget_ompt_advance_buffer_cursor_t>(vptr);
    }
  }
  return ompt_advance_buffer_cursor_fn(device, buffer, size, current, next);
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

ompt_interface_fn_t
ompt_device_callbacks_t::lookup(const char *interface_function_name) {
#define macro(fn)                                                              \
  if (strcmp(interface_function_name, #fn) == 0)                               \
    return (ompt_interface_fn_t)fn;

  FOREACH_TARGET_FN(macro);

#undef macro

  return (ompt_interface_fn_t)0;
}

static int ompt_device_init(ompt_function_lookup_t lookup,
                            int initial_device_num, ompt_data_t *tool_data) {
  DP("OMPT: Enter ompt_device_init\n");

  ompt_enabled = true;

  LIBOMPTARGET_GET_TARGET_OPID =
      (ompt_get_target_info_t)lookup(stringify(LIBOMPTARGET_GET_TARGET_OPID));

  DP("OMPT: libomptarget_get_target_info = %p\n",
     fnptr_to_ptr(LIBOMPTARGET_GET_TARGET_OPID));

  ompt_device_callbacks.register_callbacks(lookup);

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
  ;

  ompt_device_callbacks.init();

  libomptarget_connector.connect(&ompt_result);
  DP("OMPT: Exiting ompt_init\n");
}
#endif
