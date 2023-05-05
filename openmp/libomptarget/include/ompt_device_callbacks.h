//=== ompt_device_callbacks.h - Target independent OpenMP target RTL - C++ ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface used by both target-independent and device-dependent runtimes
// to coordinate registration and invocation of OMPT callbacks
//
//===----------------------------------------------------------------------===//

#ifndef _OMPT_DEVICE_CALLBACKS_H
#define _OMPT_DEVICE_CALLBACKS_H

#include "llvm/Support/DynamicLibrary.h"

//****************************************************************************
// local includes
//****************************************************************************

#include <atomic>
#include <cassert>
#include <stdlib.h>
#include <string.h>

#include <omp-tools.h>

//****************************************************************************
// macros
//****************************************************************************
#define FOREACH_OMPT_TARGET_CALLBACK(macro)                                    \
  FOREACH_OMPT_DEVICE_EVENT(macro)                                             \
  FOREACH_OMPT_NOEMI_EVENT(macro)                                              \
  FOREACH_OMPT_EMI_EVENT(macro)

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

/*****************************************************************************
 * implementation specific types
 *****************************************************************************/

//****************************************************************************
// types
//****************************************************************************

typedef uint64_t (*id_interface_t)();

class ompt_device {
public:
  ompt_device() { atomic_store(&enabled, false); };
  bool do_initialize() {
    bool old = false;
    return atomic_compare_exchange_strong(&enabled, &old, true);
  };
  bool do_finalize() {
    bool old = true;
    return atomic_compare_exchange_strong(&enabled, &old, false);
  };

private:
  std::atomic<bool> enabled;
};

/// Internal representation for OMPT device callback functions.
class OmptDeviceCallbacksTy {
public:
  /// Initialize the enabled flag and all the callbacks
  void init() {
    Enabled = false;
    num_devices = 0;
    tracing_enabled = false;
    tracing_type_enabled = 0;
    parent_dyn_lib = nullptr;
#define initName(Name, Type, Code) Name##_fn = 0;
    FOREACH_OMPT_TARGET_CALLBACK(initName)
#undef initName
    ompt_callback_buffer_request_fn = 0;
    ompt_callback_buffer_complete_fn = 0;
  }

  /// Used to register callbacks. \p Lookup is used to query a given callback
  /// by name and the result is assigned to the corresponding callback function.
  void registerCallbacks(ompt_function_lookup_t Lookup) {
    Enabled = true;
#define OmptBindCallback(Name, Type, Code)                                     \
  Name##_fn = (Name##_t)Lookup(#Name);                                         \
  DP("OMPT: class bound %s=%p\n", #Name, ((void *)(uint64_t)Name##_fn));

    FOREACH_OMPT_TARGET_CALLBACK(OmptBindCallback);
#undef OmptBindCallback
  }

  /// Used to find a callback given its name
  ompt_interface_fn_t lookupCallback(const char *InterfaceFunctionName) {
#define OmptLookup(Name, Type, Code)                                           \
  if (strcmp(InterfaceFunctionName, #Name) == 0)                               \
    return (ompt_interface_fn_t)Name##_fn;

    FOREACH_OMPT_TARGET_CALLBACK(OmptLookup);
#undef OmptLookup
    return (ompt_interface_fn_t) nullptr;
  }

  /// Wrapper function to find a callback given its name
  static ompt_interface_fn_t doLookup(const char *InterfaceFunctionName);

  void ompt_callback_device_initialize(int device_num, const char *type) {
    if (ompt_callback_device_initialize_fn) {
      ompt_device *device = lookup_device(device_num);
      if (device->do_initialize()) {
        ompt_callback_device_initialize_fn(
            device_num, type, (ompt_device_t *)device, doLookup, nullptr);
      }
    }
  };

  void ompt_callback_device_finalize(int device_num) {
    if (ompt_callback_device_finalize_fn) {
      ompt_device *device = lookup_device(device_num);
      if (device->do_finalize()) {
        ompt_callback_device_finalize_fn(device_num);
      }
    }
  };

  void ompt_callback_device_load(int device_num, const char *filename,
                                         int64_t offset_in_file,
                                         void *vma_in_file, size_t bytes,
                                         void *host_addr, void *device_addr,
                                         uint64_t module_id) {
    if (ompt_callback_device_load_fn) {
      ompt_callback_device_load_fn(device_num, filename, offset_in_file,
                                   vma_in_file, bytes, host_addr, device_addr,
                                   module_id);
    }
  };

  void ompt_callback_device_unload(int device_num, uint64_t module_id) {
    if (ompt_callback_device_unload_fn) {
      ompt_callback_device_unload_fn(device_num, module_id);
    }
  };

  void ompt_callback_target_data_op_emi(
      ompt_scope_endpoint_t endpoint, ompt_data_t *target_task_data,
      ompt_data_t *target_data, ompt_target_data_op_t optype, void *src_addr,
      int src_device_num, void *dest_addr, int dest_device_num, size_t bytes,
      const void *codeptr_ra, id_interface_t id_interface,
      ompt_id_t *host_op_id) {
    if (ompt_callback_target_data_op_emi_fn) {
      ompt_callback_target_data_op_emi_fn(
          endpoint, target_task_data, target_data, host_op_id, optype, src_addr,
          src_device_num, dest_addr, dest_device_num, bytes, codeptr_ra);
    } else if (endpoint == ompt_scope_begin) {
      ompt_callback_target_data_op(target_data->value, optype, src_addr,
                                   src_device_num, dest_addr, dest_device_num,
                                   bytes, codeptr_ra, id_interface, host_op_id);
    }
  };

  void ompt_callback_target_data_op(
      ompt_id_t target_id, ompt_target_data_op_t optype, void *src_addr,
      int src_device_num, void *dest_addr, int dest_device_num, size_t bytes,
      const void *codeptr_ra, id_interface_t id_interface,
      ompt_id_t *host_op_id) {
    if (ompt_callback_target_data_op_fn) {
      *host_op_id = id_interface();
      ompt_callback_target_data_op_fn(target_id, *host_op_id, optype, src_addr,
                                      src_device_num, dest_addr,
                                      dest_device_num, bytes, codeptr_ra);
    }
  };

  void ompt_callback_target_emi(ompt_target_t kind,
                                        ompt_scope_endpoint_t endpoint,
                                        int device_num, ompt_data_t *task_data,
                                        ompt_data_t *target_task_data,
                                        ompt_data_t *target_data,
                                        const void *codeptr_ra,
                                        id_interface_t id_interface) {
    if (ompt_callback_target_emi_fn) {
      ompt_callback_target_emi_fn(kind, endpoint, device_num, task_data,
                                  target_task_data, target_data, codeptr_ra);
    } else {
      ompt_callback_target(kind, endpoint, device_num, task_data, codeptr_ra,
                           target_data, id_interface);
    }
  };

  void ompt_callback_target(ompt_target_t kind,
                                    ompt_scope_endpoint_t endpoint,
                                    int device_num, ompt_data_t *task_data,
                                    const void *codeptr_ra,
                                    ompt_data_t *target_data,
                                    id_interface_t id_interface) {
    // if we reach this point, ompt_callback_target_emi was not
    // invoked so a tool didn't provide a target id. thus, we must
    // unconditionally get an id here. even if there is no
    // ompt_callback_target, we need to have an id for use by other
    // callbacks.
    // note:
    // on a scope_begin callback, id_interface will generate an id.
    // on a scope_end callback, id_interface will return the existing
    // id. it is safe to do the assignment again.
    target_data->value = id_interface();
    if (ompt_callback_target_fn) {
      ompt_callback_target_fn(kind, endpoint, device_num, task_data,
                              target_data->value, codeptr_ra);
    }
  };

  void ompt_callback_target_map_emi(ompt_data_t *target_data,
                                            unsigned int nitems,
                                            void **host_addr,
                                            void **device_addr, size_t *bytes,
                                            unsigned int *mapping_flags,
                                            const void *codeptr_ra) {
    if (ompt_callback_target_map_emi_fn) {
      ompt_callback_target_map_emi_fn(target_data, nitems, host_addr,
                                      device_addr, bytes, mapping_flags,
                                      codeptr_ra);
    } else {
      ompt_callback_target_map(target_data->value, nitems, host_addr,
                               device_addr, bytes, mapping_flags, codeptr_ra);
    }
  };

  void ompt_callback_target_map(ompt_id_t target_id,
                                        unsigned int nitems, void **host_addr,
                                        void **device_addr, size_t *bytes,
                                        unsigned int *mapping_flags,
                                        const void *codeptr_ra) {
    if (ompt_callback_target_map_fn) {
      ompt_callback_target_map_fn(target_id, nitems, host_addr, device_addr,
                                  bytes, mapping_flags, codeptr_ra);
    }
  };

  void ompt_callback_target_submit_emi(ompt_scope_endpoint_t endpoint,
                                               ompt_data_t *target_data,
                                               unsigned int requested_num_teams,
                                               id_interface_t id_interface,
                                               ompt_id_t *host_op_id) {
    if (ompt_callback_target_submit_emi_fn) {
      ompt_callback_target_submit_emi_fn(endpoint, target_data, host_op_id,
                                         requested_num_teams);
    } else if (endpoint == ompt_scope_begin) {
      return ompt_callback_target_submit(
          target_data->value, requested_num_teams, id_interface, host_op_id);
    }
  };

  void ompt_callback_target_submit(ompt_id_t target_id,
                                           unsigned int requested_num_teams,
                                           id_interface_t id_interface,
                                           ompt_id_t *host_op_id) {
    if (ompt_callback_target_submit_fn) {
      *host_op_id = id_interface();
      ompt_callback_target_submit_fn(target_id, *host_op_id,
                                     requested_num_teams);
    }
  };

  void ompt_callback_buffer_request(int device_num, ompt_buffer_t **buffer,
                                    size_t *bytes) {
    if (ompt_callback_buffer_request_fn) {
      ompt_callback_buffer_request_fn(device_num, buffer, bytes);
    }
  }

  void ompt_callback_buffer_complete(int device_num, ompt_buffer_t *buffer,
                                     size_t bytes, ompt_buffer_cursor_t begin,
                                     int buffer_owned) {
    if (ompt_callback_buffer_complete_fn) {
      ompt_callback_buffer_complete_fn(device_num, buffer, bytes, begin,
                                       buffer_owned);
    }
  }

  bool is_enabled() { return Enabled; }

  bool is_tracing_enabled() { return tracing_enabled; }
  void set_tracing_enabled(bool b) { tracing_enabled = b; }

  bool is_tracing_type_enabled(unsigned int etype) {
    assert(etype < 64);
    if (etype < 64)
      return (tracing_type_enabled & (1UL << etype)) != 0;
    return false;
  }

  void set_tracing_type_enabled(unsigned int etype, bool b) {
    assert(etype < 64);
    if (etype < 64) {
      if (b)
        tracing_type_enabled |= (1UL << etype);
      else
        tracing_type_enabled &= ~(1UL << etype);
    }
  }

  ompt_set_result_t set_trace_ompt(ompt_device_t *device, unsigned int enable,
                                   unsigned int etype) {
    // TODO handle device

    DP("set_trace_ompt: %d %d\n", etype, enable);

    bool is_event_enabled = enable > 0;
    if (etype == 0) {
      /* set/reset all supported types */
      set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target,
                               is_event_enabled);
      set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target_data_op,
                               is_event_enabled);
      set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target_submit,
                               is_event_enabled);
      set_tracing_type_enabled(ompt_callbacks_t::ompt_callback_target_emi,
                               is_event_enabled);
      set_tracing_type_enabled(
          ompt_callbacks_t::ompt_callback_target_data_op_emi, is_event_enabled);
      set_tracing_type_enabled(
          ompt_callbacks_t::ompt_callback_target_submit_emi, is_event_enabled);

      if (is_event_enabled)
        return ompt_set_sometimes; // a subset is enabled
      else
        return ompt_set_always; // we can disable for all
    }
    switch (etype) {
    case ompt_callbacks_t::ompt_callback_target:
    case ompt_callbacks_t::ompt_callback_target_data_op:
    case ompt_callbacks_t::ompt_callback_target_submit:
    case ompt_callbacks_t::ompt_callback_target_emi:
    case ompt_callbacks_t::ompt_callback_target_data_op_emi:
    case ompt_callbacks_t::ompt_callback_target_submit_emi: {
      set_tracing_type_enabled(etype, is_event_enabled);
      return ompt_set_always;
    }
    default: {
      if (is_event_enabled)
        return ompt_set_never; // unimplemented
      else
        return ompt_set_always; // always disabled anyways
    }
    }
  }

  void compute_parent_dyn_lib(const char *lib_name) {
    if (parent_dyn_lib)
      return;
    std::string err_msg;
    parent_dyn_lib = std::make_shared<llvm::sys::DynamicLibrary>(
        llvm::sys::DynamicLibrary::getPermanentLibrary(lib_name, &err_msg));
  }

  std::shared_ptr<llvm::sys::DynamicLibrary> get_parent_dyn_lib() {
    return parent_dyn_lib;
  }

  void prepare_devices(int number_of_devices) {
    num_devices = number_of_devices;
    resize(number_of_devices);
  }

  void set_buffer_request(ompt_callback_buffer_request_t callback) {
    ompt_callback_buffer_request_fn = callback;
  }

  void set_buffer_complete(ompt_callback_buffer_complete_t callback) {
    ompt_callback_buffer_complete_fn = callback;
  }

  int lookup_device_id(ompt_device *device);

private:
  /// Set to true if callbacks for this library have been initialized
  bool Enabled;

  int num_devices;
  std::atomic<bool> tracing_enabled;
  std::atomic<uint64_t> tracing_type_enabled;
  std::shared_ptr<llvm::sys::DynamicLibrary> parent_dyn_lib;

  ompt_callback_buffer_request_t ompt_callback_buffer_request_fn;
  ompt_callback_buffer_complete_t ompt_callback_buffer_complete_fn;

  static ompt_device *lookup_device(int device_num);

  static void resize(int number_of_devices);

  /// Callback functions
#define DeclareName(Name, Type, Code) Name##_t Name##_fn;
  FOREACH_OMPT_TARGET_CALLBACK(DeclareName)
#undef DeclareName
};

/// Device callbacks object for the library that performs the instantiation
extern OmptDeviceCallbacksTy OmptDeviceCallbacks;

#pragma pop_macro("DEBUG_PREFIX")

#endif
