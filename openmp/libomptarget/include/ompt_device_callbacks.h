//=== ompt_device_callbacks.h - Target independent OpenMP target RTL -- C++
//-===//
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

//****************************************************************************
// local includes
//****************************************************************************

#include <atomic>
#include <string.h>

#include <omp-tools.h>

//****************************************************************************
// macros
//****************************************************************************
#define FOREACH_OMPT_TARGET_CALLBACK(macro)                                    \
  FOREACH_OMPT_DEVICE_EVENT(macro)                                             \
  FOREACH_OMPT_NOEMI_EVENT(macro)                                              \
  FOREACH_OMPT_EMI_EVENT(macro)

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

class ompt_device_callbacks_t {
public:
  void ompt_callback_device_initialize(int device_num,
                                               const char *type) {
    if (ompt_callback_device_initialize_fn) {
      ompt_device *device = lookup_device(device_num);
      if (device->do_initialize()) {
        ompt_callback_device_initialize_fn(
            device_num, type, (ompt_device_t *)device, lookup, documentation);
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

  void init() {
    enabled = false;
#define init_name(name, type, code) name##_fn = 0;
    FOREACH_OMPT_TARGET_CALLBACK(init_name)
#undef init_name
  };

  bool is_enabled() { return enabled; }

  void prepare_devices(int number_of_devices) { resize(number_of_devices); };

  void register_callbacks(ompt_function_lookup_t lookup) {
    enabled = true;
#define ompt_bind_callback(fn, type, code)                                     \
  fn##_fn = (fn##_t)lookup(#fn);                                               \
  DP("OMPT: class bound %s=%p\n", #fn, ((void *)(uint64_t)fn##_fn));
    FOREACH_OMPT_TARGET_CALLBACK(ompt_bind_callback);
#undef ompt_bind_callback
  };

  ompt_interface_fn_t lookup_callback(const char *interface_function_name) {
#define ompt_dolookup(fn, type, code)                                          \
  if (strcmp(interface_function_name, #fn) == 0)                               \
    return (ompt_interface_fn_t)fn##_fn;

    FOREACH_OMPT_TARGET_CALLBACK(ompt_dolookup);
#undef ompt_dolookup

    return (ompt_interface_fn_t)0;
  };

  static ompt_interface_fn_t lookup(const char *interface_function_name);

private:
  bool enabled;

#define declare_name(name, type, code) name##_t name##_fn;
  FOREACH_OMPT_TARGET_CALLBACK(declare_name)
#undef declare_name

  static void resize(int number_of_devices);
  static ompt_device *lookup_device(int device_num);
  static const char *documentation;
};

extern ompt_device_callbacks_t ompt_device_callbacks;

#endif
