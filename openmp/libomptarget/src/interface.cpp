//===-------- interface.cpp - Target independent OpenMP target RTL --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <omptarget.h>

#include "device.h"
#include "omptarget.h"
#include "private.h"
#include "rtl.h"

#include <stdarg.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#ifdef OMPT_SUPPORT
#include "ompt_callback.h"
#define OMPT_IF_ENABLED(stmts)                                                 \
  do {                                                                         \
    if (ompt_enabled) {                                                        \
      stmts                                                                    \
    }                                                                          \
  } while (0)
#else
#define OMPT_IF_ENABLED(stmts)
#endif

////////////////////////////////////////////////////////////////////////////////
/// adds requires flags
EXTERN void __tgt_register_requires(int64_t flags) {
  TIMESCOPE();
  PM->RTLs.RegisterRequires(flags);
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {
  TIMESCOPE();
  std::call_once(PM->RTLs.initFlag, &RTLsTy::LoadRTLs, &PM->RTLs);
  for (auto &RTL : PM->RTLs.AllRTLs) {
    if (RTL.register_lib) {
      if ((*RTL.register_lib)(desc) != OFFLOAD_SUCCESS) {
        DP("Could not register library with %s", RTL.RTLName.c_str());
      }
    }
  }
  PM->RTLs.RegisterLib(desc);
}

static __tgt_image_info **__tgt_AllImageInfos;
static int __tgt_num_registered_images = 0;
EXTERN void __tgt_register_image_info(__tgt_image_info *imageInfo) {

  DP(" register_image_info image %d of %d offload-arch:%s VERSION:%d\n",
     imageInfo->image_number, imageInfo->number_images, imageInfo->offload_arch,
     imageInfo->version);
  if (!__tgt_AllImageInfos)
    __tgt_AllImageInfos = (__tgt_image_info **)malloc(
        sizeof(__tgt_image_info *) * imageInfo->number_images);
  __tgt_AllImageInfos[imageInfo->image_number] = imageInfo;
  __tgt_num_registered_images = imageInfo->number_images;
}

////////////////////////////////////////////////////////////////////////////////
/// Return pointer to image information if it was registered
EXTERN __tgt_image_info *__tgt_get_image_info(unsigned image_number) {
  if (__tgt_num_registered_images)
    return __tgt_AllImageInfos[image_number];
  else
    return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize all available devices without registering any image
EXTERN void __tgt_init_all_rtls() { PM->RTLs.initAllRTLs(); }

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  TIMESCOPE();
  PM->RTLs.UnregisterLib(desc);
  for (auto &RTL : PM->RTLs.UsedRTLs) {
    if (RTL->unregister_lib) {
      if ((*RTL->unregister_lib)(desc) != OFFLOAD_SUCCESS) {
        DP("Could not register library with %s", RTL->RTLName.c_str());
      }
    }
  }
  if (__tgt_num_registered_images) {
    free(__tgt_AllImageInfos);
    __tgt_num_registered_images = 0;
  }
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
                                    void **args_base, void **args,
                                    int64_t *arg_sizes, int64_t *arg_types) {
  TIMESCOPE();
  __tgt_target_data_begin_mapper(nullptr, device_id, arg_num, args_base, args,
                                 arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_begin_with_deps(int64_t device_id, int32_t arg_num,
                                              void **args_base, void **args,
                                              int64_t *arg_sizes, int64_t *arg_types,
                                              int32_t depNum, int nargs, ...) {
  TIMESCOPE();
  int *dependinfo;
  if (depNum > 0) {
    dependinfo = (int*)malloc(nargs*sizeof(int));
    va_list valist;
    va_start(valist, nargs);
    for (int k = 0; k < nargs; k++) {
      dependinfo[k] = va_arg(valist, int);
    }
    va_end(valist);

    kmp_depend_info_t *deplist = (kmp_depend_info_t*)malloc(depNum*sizeof(kmp_depend_info_t));

    for (int i = 0, j = 0; i < depNum && j < depNum*3; i++, j+=3) {
      kmp_depend_info_t depinfo;
      depinfo.base_addr = dependinfo[j+2];
      depinfo.len = dependinfo[j+1];
      int deptype = dependinfo[j];
      depinfo.flags.mtx = 1;
      if (deptype == DI_DEP_TYPE_INOUT) {
        depinfo.flags.in = 1;
        depinfo.flags.out = 1;
      } else if (deptype == DI_DEP_TYPE_IN) {
        depinfo.flags.in = 1;
      } else if (deptype == DI_DEP_TYPE_OUT) {
        depinfo.flags.out = 1;
      }
      deplist[i] = depinfo;
    }
    free(dependinfo);

    __kmpc_omp_wait_deps(NULL, __kmpc_global_thread_num(NULL), depNum, deplist, 0, deplist);
    free(deplist);
  }

  __tgt_target_data_begin_mapper(nullptr, device_id, arg_num, args_base, args,
                                 arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_begin_nowait(int64_t device_id, int32_t arg_num,
                                           void **args_base, void **args,
                                           int64_t *arg_sizes,
                                           int64_t *arg_types, int32_t depNum,
                                           void *depList, int32_t noAliasDepNum,
                                           void *noAliasDepList) {
  TIMESCOPE();

  __tgt_target_data_begin_mapper(nullptr, device_id, arg_num, args_base, args,
                                 arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_begin_mapper(ident_t *loc, int64_t device_id,
                                           int32_t arg_num, void **args_base,
                                           void **args, int64_t *arg_sizes,
                                           int64_t *arg_types,
                                           map_var_info_t *arg_names,
                                           void **arg_mappers) {
  TIMESCOPE_WITH_IDENT(loc);
  DP("Entering data begin region for device %" PRId64 " with %d mappings\n",
     device_id, arg_num);

  void *codeptr = nullptr;
  OMPT_IF_ENABLED(
      codeptr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), codeptr);
      ompt_interface.target_data_enter_begin(device_id, codeptr);
      ompt_interface.target_trace_record_gen(device_id, ompt_target_enter_data,
                                             ompt_scope_begin, codeptr););

  if (checkDeviceAndCtors(device_id, loc)) {
    DP("Not offloading to device %" PRId64 "\n", device_id);
    return;
  }

  DeviceTy &Device = *PM->Devices[device_id];

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(loc, device_id, arg_num, arg_sizes, arg_types,
                         arg_names, "Entering OpenMP data region");
#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i],
       (arg_names) ? getNameFromMapping(arg_names[i]).c_str() : "unknown");
  }
#endif

  AsyncInfoTy AsyncInfo(Device);
  int rc = targetDataBegin(loc, Device, arg_num, args_base, args, arg_sizes,
                           arg_types, arg_names, arg_mappers, AsyncInfo);
  if (rc == OFFLOAD_SUCCESS)
    rc = AsyncInfo.synchronize();
  handleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      device_id, ompt_target_enter_data, ompt_scope_end, codeptr);
                  ompt_interface.target_data_enter_end(device_id, codeptr);
                  ompt_interface.ompt_state_clear(););
}

EXTERN void __tgt_target_data_begin_nowait_mapper(
    ident_t *loc, int64_t device_id, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  TIMESCOPE_WITH_IDENT(loc);

  __tgt_target_data_begin_mapper(loc, device_id, arg_num, args_base, args,
                                 arg_sizes, arg_types, arg_names, arg_mappers);
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int64_t device_id, int32_t arg_num,
                                  void **args_base, void **args,
                                  int64_t *arg_sizes, int64_t *arg_types) {
  TIMESCOPE();
  __tgt_target_data_end_mapper(nullptr, device_id, arg_num, args_base, args,
                               arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_end_with_deps(int64_t device_id, int32_t arg_num,
                                            void **args_base, void **args,
                                            int64_t *arg_sizes, int64_t *arg_types,
                                           int32_t depNum, int nargs, ...) {
  TIMESCOPE();
  int *dependinfo;
  if (depNum > 0) {
    dependinfo = (int*)malloc(nargs*sizeof(int));
    va_list valist;
    va_start(valist, nargs);
    for (int k = 0; k < nargs; k++) {
      dependinfo[k] = va_arg(valist, int);
    }
    va_end(valist);

    kmp_depend_info_t *deplist = (kmp_depend_info_t*)malloc(depNum*sizeof(kmp_depend_info_t));

    for (int i = 0, j = 0; i < depNum && j < depNum*3; i++, j+=3) {
      kmp_depend_info_t depinfo;
      depinfo.base_addr = dependinfo[j+2];
      depinfo.len = dependinfo[j+1];
      int deptype = dependinfo[j];
      depinfo.flags.mtx = 1;
      if (deptype == DI_DEP_TYPE_INOUT) {
        depinfo.flags.in = 1;
        depinfo.flags.out = 1;
      } else if (deptype == DI_DEP_TYPE_IN) {
        depinfo.flags.in = 1;
      } else if (deptype == DI_DEP_TYPE_OUT) {
        depinfo.flags.out = 1;
      }
      deplist[i] = depinfo;
    }
    free(dependinfo);

    __kmpc_omp_wait_deps(NULL, __kmpc_global_thread_num(NULL), depNum, deplist, 0, deplist);
    free(deplist);
  }

  __tgt_target_data_end_mapper(nullptr, device_id, arg_num, args_base, args,
                               arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_end_nowait(int64_t device_id, int32_t arg_num,
                                         void **args_base, void **args,
                                         int64_t *arg_sizes, int64_t *arg_types,
                                         int32_t depNum, void *depList,
                                         int32_t noAliasDepNum,
                                         void *noAliasDepList) {
  TIMESCOPE();

  __tgt_target_data_end_mapper(nullptr, device_id, arg_num, args_base, args,
                               arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_end_mapper(ident_t *loc, int64_t device_id,
                                         int32_t arg_num, void **args_base,
                                         void **args, int64_t *arg_sizes,
                                         int64_t *arg_types,
                                         map_var_info_t *arg_names,
                                         void **arg_mappers) {
  TIMESCOPE_WITH_IDENT(loc);
  DP("Entering data end region with %d mappings\n", arg_num);
  if (checkDeviceAndCtors(device_id, loc)) {
    DP("Not offloading to device %" PRId64 "\n", device_id);
    return;
  }

  DeviceTy &Device = *PM->Devices[device_id];

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(loc, device_id, arg_num, arg_sizes, arg_types,
                         arg_names, "Exiting OpenMP data region");
#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i],
       (arg_names) ? getNameFromMapping(arg_names[i]).c_str() : "unknown");
  }
#endif

  AsyncInfoTy AsyncInfo(Device);

  void *codeptr = nullptr;
  OMPT_IF_ENABLED(
      codeptr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), codeptr);
      ompt_interface.target_data_exit_begin(device_id, codeptr);
      ompt_interface.target_trace_record_gen(device_id, ompt_target_exit_data,
                                             ompt_scope_begin, codeptr););

  int rc = targetDataEnd(loc, Device, arg_num, args_base, args, arg_sizes,
                         arg_types, arg_names, arg_mappers, AsyncInfo);
  if (rc == OFFLOAD_SUCCESS)
    rc = AsyncInfo.synchronize();
  handleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      device_id, ompt_target_exit_data, ompt_scope_end, codeptr);
                  ompt_interface.target_data_exit_end(device_id, codeptr);
                  ompt_interface.ompt_state_clear(););
}

EXTERN void __tgt_target_data_end_nowait_mapper(
    ident_t *loc, int64_t device_id, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  TIMESCOPE_WITH_IDENT(loc);

  __tgt_target_data_end_mapper(loc, device_id, arg_num, args_base, args,
                               arg_sizes, arg_types, arg_names, arg_mappers);
}

EXTERN void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
                                     void **args_base, void **args,
                                     int64_t *arg_sizes, int64_t *arg_types) {
  TIMESCOPE();
  __tgt_target_data_update_mapper(nullptr, device_id, arg_num, args_base, args,
                                  arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_update_with_deps(
    int64_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t depNum, int nargs, ...) {
  TIMESCOPE();
  int *dependinfo;
  if (depNum > 0) {
    dependinfo = (int*)malloc(nargs*sizeof(int));
    va_list valist;
    va_start(valist, nargs);
    for (int k = 0; k < nargs; k++) {
      dependinfo[k] = va_arg(valist, int);
    }
    va_end(valist);
 
    kmp_depend_info_t *deplist = (kmp_depend_info_t*)malloc(depNum*sizeof(kmp_depend_info_t));

    for (int i = 0, j = 0; i < depNum && j < depNum*3; i++, j+=3) {
      kmp_depend_info_t depinfo;
      depinfo.base_addr = dependinfo[j+2];
      depinfo.len = dependinfo[j+1];
      int deptype = dependinfo[j];
      depinfo.flags.mtx = 1;
      if (deptype == DI_DEP_TYPE_INOUT) {
        depinfo.flags.in = 1;
        depinfo.flags.out = 1;
      } else if (deptype == DI_DEP_TYPE_IN) {
        depinfo.flags.in = 1;
      } else if (deptype == DI_DEP_TYPE_OUT) {
        depinfo.flags.out = 1;
      }
      deplist[i] = depinfo;
    }
    free(dependinfo);

    __kmpc_omp_wait_deps(NULL, __kmpc_global_thread_num(NULL), depNum, deplist, 0, deplist);
    free(deplist);
  }
  return __tgt_target_data_update_mapper(nullptr, device_id, arg_num, args_base, args,
                                         arg_sizes, arg_types, nullptr, nullptr);
}


EXTERN void __tgt_target_data_update_nowait(
    int64_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  TIMESCOPE();

  __tgt_target_data_update_mapper(nullptr, device_id, arg_num, args_base, args,
                                  arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN void __tgt_target_data_update_mapper(ident_t *loc, int64_t device_id,
                                            int32_t arg_num, void **args_base,
                                            void **args, int64_t *arg_sizes,
                                            int64_t *arg_types,
                                            map_var_info_t *arg_names,
                                            void **arg_mappers) {
  TIMESCOPE_WITH_IDENT(loc);
  DP("Entering data update with %d mappings\n", arg_num);
  if (checkDeviceAndCtors(device_id, loc)) {
    DP("Not offloading to device %" PRId64 "\n", device_id);
    return;
  }

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(loc, device_id, arg_num, arg_sizes, arg_types,
                         arg_names, "Updating OpenMP data");

  void *codeptr = nullptr;
  OMPT_IF_ENABLED(
      codeptr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), codeptr);
      ompt_interface.target_update_begin(device_id, codeptr);
      ompt_interface.target_trace_record_gen(device_id, ompt_target_update,
                                             ompt_scope_begin, codeptr););

  DeviceTy &Device = *PM->Devices[device_id];
  AsyncInfoTy AsyncInfo(Device);
  int rc = targetDataUpdate(loc, Device, arg_num, args_base, args, arg_sizes,
                            arg_types, arg_names, arg_mappers, AsyncInfo);
  if (rc == OFFLOAD_SUCCESS)
    rc = AsyncInfo.synchronize();
  handleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      device_id, ompt_target_update, ompt_scope_end, codeptr);
                  ompt_interface.target_update_end(device_id, codeptr);
                  ompt_interface.ompt_state_clear(););
}

EXTERN void __tgt_target_data_update_nowait_mapper(
    ident_t *loc, int64_t device_id, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  TIMESCOPE_WITH_IDENT(loc);

  __tgt_target_data_update_mapper(loc, device_id, arg_num, args_base, args,
                                  arg_sizes, arg_types, arg_names, arg_mappers);
}

EXTERN int __tgt_target(int64_t device_id, void *host_ptr, int32_t arg_num,
                        void **args_base, void **args, int64_t *arg_sizes,
                        int64_t *arg_types) {
  TIMESCOPE();
  return __tgt_target_mapper(nullptr, device_id, host_ptr, arg_num, args_base,
                             args, arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN int __tgt_target_with_deps(int64_t device_id, void *host_ptr, int32_t arg_num,
                                  void **args_base, void **args, int64_t *arg_sizes,
                                  int64_t *arg_types, int32_t depNum, int nargs, ...) {
  TIMESCOPE();
  int *dependinfo;
  if (depNum > 0) {
    dependinfo = (int*)malloc(nargs*sizeof(int));
    va_list valist;
    va_start(valist, nargs);
    for (int k = 0; k < nargs; k++) {
      dependinfo[k] = va_arg(valist, int);
    }
    va_end(valist);

    kmp_depend_info_t *deplist = (kmp_depend_info_t*)malloc(depNum*sizeof(kmp_depend_info_t));
    
    for (int i = 0, j = 0; i < depNum && j < depNum*3; i++, j+=3) {
      kmp_depend_info_t depinfo;
      depinfo.base_addr = dependinfo[j+2];
      depinfo.len = dependinfo[j+1];
      int deptype = dependinfo[j];
      depinfo.flags.mtx = 1;
      if (deptype == DI_DEP_TYPE_INOUT) {
        depinfo.flags.in = 1;
        depinfo.flags.out = 1;
      } else if (deptype == DI_DEP_TYPE_IN) {
        depinfo.flags.in = 1;
      } else if (deptype == DI_DEP_TYPE_OUT) {
        depinfo.flags.out = 1;
      }
      deplist[i] = depinfo;
    }
    free(dependinfo);

    __kmpc_omp_task_with_deps(NULL, __kmpc_global_thread_num(NULL), host_ptr, depNum, deplist, 0, deplist);
  }
   
  return __tgt_target_mapper(nullptr, device_id, host_ptr, arg_num, args_base,
                               args, arg_sizes, arg_types, nullptr, nullptr);
}
   
  

EXTERN int __tgt_target_nowait(int64_t device_id, void *host_ptr,
                               int32_t arg_num, void **args_base, void **args,
                               int64_t *arg_sizes, int64_t *arg_types,
                               int32_t depNum, void *depList,
                               int32_t noAliasDepNum, void *noAliasDepList) {
  TIMESCOPE();

  return __tgt_target_mapper(nullptr, device_id, host_ptr, arg_num, args_base,
                             args, arg_sizes, arg_types, nullptr, nullptr);
}

EXTERN int __tgt_target_mapper(ident_t *loc, int64_t device_id, void *host_ptr,
                               int32_t arg_num, void **args_base, void **args,
                               int64_t *arg_sizes, int64_t *arg_types,
                               map_var_info_t *arg_names, void **arg_mappers) {
  TIMESCOPE_WITH_IDENT(loc);
  DP("Entering target region with entry point " DPxMOD " and device Id %" PRId64
     "\n",
     DPxPTR(host_ptr), device_id);
  if (checkDeviceAndCtors(device_id, loc)) {
    DP("Not offloading to device %" PRId64 "\n", device_id);
    return OMP_TGT_FAIL;
  }

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(loc, device_id, arg_num, arg_sizes, arg_types,
                         arg_names, "Entering OpenMP kernel");
#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i],
       (arg_names) ? getNameFromMapping(arg_names[i]).c_str() : "unknown");
  }
#endif

  DeviceTy &Device = *PM->Devices[device_id];
  AsyncInfoTy AsyncInfo(Device);

  void *codeptr = nullptr;
  OMPT_IF_ENABLED(
      codeptr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), codeptr);
      ompt_interface.target_begin(device_id, codeptr);
      ompt_interface.target_trace_record_gen(device_id, ompt_target,
                                             ompt_scope_begin, codeptr););

  int rc = target(loc, Device, host_ptr, arg_num, args_base, args, arg_sizes,
                  arg_types, arg_names, arg_mappers, 0, 0, false /*team*/,
                  AsyncInfo);
  if (rc == OFFLOAD_SUCCESS)
    rc = AsyncInfo.synchronize();
  handleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      device_id, ompt_target, ompt_scope_end, codeptr);
                  ompt_interface.target_end(device_id, codeptr);
                  ompt_interface.ompt_state_clear(););

  assert(rc == OFFLOAD_SUCCESS && "__tgt_target_mapper unexpected failure!");
  return OMP_TGT_SUCCESS;
}

EXTERN int __tgt_target_nowait_mapper(
    ident_t *loc, int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  TIMESCOPE_WITH_IDENT(loc);

  return __tgt_target_mapper(loc, device_id, host_ptr, arg_num, args_base, args,
                             arg_sizes, arg_types, arg_names, arg_mappers);
}

EXTERN int __tgt_target_teams(int64_t device_id, void *host_ptr,
                              int32_t arg_num, void **args_base, void **args,
                              int64_t *arg_sizes, int64_t *arg_types,
                              int32_t team_num, int32_t thread_limit) {
  TIMESCOPE();
  return __tgt_target_teams_mapper(nullptr, device_id, host_ptr, arg_num,
                                   args_base, args, arg_sizes, arg_types,
                                   nullptr, nullptr, team_num, thread_limit);
}

EXTERN int __tgt_target_teams_nowait(int64_t device_id, void *host_ptr,
                                     int32_t arg_num, void **args_base,
                                     void **args, int64_t *arg_sizes,
                                     int64_t *arg_types, int32_t team_num,
                                     int32_t thread_limit, int32_t depNum,
                                     void *depList, int32_t noAliasDepNum,
                                     void *noAliasDepList) {
  TIMESCOPE();

  return __tgt_target_teams_mapper(nullptr, device_id, host_ptr, arg_num,
                                   args_base, args, arg_sizes, arg_types,
                                   nullptr, nullptr, team_num, thread_limit);
}

EXTERN int __tgt_target_teams_mapper(ident_t *loc, int64_t device_id,
                                     void *host_ptr, int32_t arg_num,
                                     void **args_base, void **args,
                                     int64_t *arg_sizes, int64_t *arg_types,
                                     map_var_info_t *arg_names,
                                     void **arg_mappers, int32_t team_num,
                                     int32_t thread_limit) {
  DP("Entering target region with entry point " DPxMOD " and device Id %" PRId64
     "\n",
     DPxPTR(host_ptr), device_id);
  if (checkDeviceAndCtors(device_id, loc)) {
    DP("Not offloading to device %" PRId64 "\n", device_id);
    return OMP_TGT_FAIL;
  }

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(loc, device_id, arg_num, arg_sizes, arg_types,
                         arg_names, "Entering OpenMP kernel");
#ifdef OMPTARGET_DEBUG
  for (int i = 0; i < arg_num; ++i) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       i, DPxPTR(args_base[i]), DPxPTR(args[i]), arg_sizes[i], arg_types[i],
       (arg_names) ? getNameFromMapping(arg_names[i]).c_str() : "unknown");
  }
#endif

  DeviceTy &Device = *PM->Devices[device_id];
  AsyncInfoTy AsyncInfo(Device);

  void *codeptr = nullptr;
  OMPT_IF_ENABLED(
      codeptr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), codeptr);
      ompt_interface.target_begin(device_id, codeptr);
      ompt_interface.target_trace_record_gen(device_id, ompt_target,
                                             ompt_scope_begin, codeptr););

  int rc = target(loc, Device, host_ptr, arg_num, args_base, args, arg_sizes,
                  arg_types, arg_names, arg_mappers, team_num, thread_limit,
                  true /*team*/, AsyncInfo);
  if (rc == OFFLOAD_SUCCESS)
    rc = AsyncInfo.synchronize();
  handleTargetOutcome(rc == OFFLOAD_SUCCESS, loc);

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      device_id, ompt_target, ompt_scope_end, codeptr);
                  ompt_interface.target_end(device_id, codeptr);
                  ompt_interface.ompt_state_clear(););

  assert(rc == OFFLOAD_SUCCESS &&
         "__tgt_target_teams_mapper unexpected failure!");
  return OMP_TGT_SUCCESS;
}

EXTERN int __tgt_target_teams_nowait_mapper(
    ident_t *loc, int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    map_var_info_t *arg_names, void **arg_mappers, int32_t team_num,
    int32_t thread_limit, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  TIMESCOPE_WITH_IDENT(loc);

  return __tgt_target_teams_mapper(loc, device_id, host_ptr, arg_num, args_base,
                                   args, arg_sizes, arg_types, arg_names,
                                   arg_mappers, team_num, thread_limit);
}

// Get the current number of components for a user-defined mapper.
EXTERN int64_t __tgt_mapper_num_components(void *rt_mapper_handle) {
  TIMESCOPE();
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)rt_mapper_handle;
  int64_t size = MapperComponentsPtr->Components.size();
  DP("__tgt_mapper_num_components(Handle=" DPxMOD ") returns %" PRId64 "\n",
     DPxPTR(rt_mapper_handle), size);
  return size;
}

// Push back one component for a user-defined mapper.
EXTERN void __tgt_push_mapper_component(void *rt_mapper_handle, void *base,
                                        void *begin, int64_t size, int64_t type,
                                        void *name) {
  TIMESCOPE();
  DP("__tgt_push_mapper_component(Handle=" DPxMOD
     ") adds an entry (Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
     ", Type=0x%" PRIx64 ", Name=%s).\n",
     DPxPTR(rt_mapper_handle), DPxPTR(base), DPxPTR(begin), size, type,
     (name) ? getNameFromMapping(name).c_str() : "unknown");
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)rt_mapper_handle;
  MapperComponentsPtr->Components.push_back(
      MapComponentInfoTy(base, begin, size, type, name));
}

EXTERN void __kmpc_push_target_tripcount(ident_t *loc, int64_t device_id,
                                         uint64_t loop_tripcount) {
  __kmpc_push_target_tripcount_mapper(loc, device_id, loop_tripcount);
}

EXTERN void __kmpc_push_target_tripcount_mapper(ident_t *loc, int64_t device_id,
                                                uint64_t loop_tripcount) {
  TIMESCOPE_WITH_IDENT(loc);
  if (checkDeviceAndCtors(device_id, loc)) {
    DP("Not offloading to device %" PRId64 "\n", device_id);
    return;
  }

  DP("__kmpc_push_target_tripcount(%" PRId64 ", %" PRIu64 ")\n", device_id,
     loop_tripcount);
  PM->TblMapMtx.lock();
  PM->Devices[device_id]->LoopTripCnt.emplace(__kmpc_global_thread_num(NULL),
                                              loop_tripcount);
  PM->TblMapMtx.unlock();
}

EXTERN void __tgt_set_info_flag(uint32_t NewInfoLevel) {
  std::atomic<uint32_t> &InfoLevel = getInfoLevelInternal();
  InfoLevel.store(NewInfoLevel);
  for (auto &R : PM->RTLs.AllRTLs) {
    if (R.set_info_flag)
      R.set_info_flag(NewInfoLevel);
  }
}

EXTERN int __tgt_print_device_info(int64_t device_id) {
  return PM->Devices[device_id]->printDeviceInfo(
      PM->Devices[device_id]->RTLDeviceID);
}
