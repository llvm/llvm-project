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
EXTERN void __tgt_register_requires(int64_t Flags) {
  TIMESCOPE();
  PM->RTLs.registerRequires(Flags);
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *Desc) {
  TIMESCOPE();
  std::call_once(PM->RTLs.InitFlag, &RTLsTy::loadRTLs, &PM->RTLs);
  for (auto &RTL : PM->RTLs.AllRTLs) {
    if (RTL.register_lib) {
      if ((*RTL.register_lib)(Desc) != OFFLOAD_SUCCESS) {
        DP("Could not register library with %s", RTL.RTLName.c_str());
      }
    }
  }
  PM->RTLs.registerLib(Desc);
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
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *Desc) {
  TIMESCOPE();
  PM->RTLs.unregisterLib(Desc);
  for (auto &RTL : PM->RTLs.UsedRTLs) {
    if (RTL->unregister_lib) {
      if ((*RTL->unregister_lib)(Desc) != OFFLOAD_SUCCESS) {
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
EXTERN void __tgt_target_data_begin(int64_t DeviceId, int32_t ArgNum,
                                    void **ArgsBase, void **Args,
                                    int64_t *ArgSizes, int64_t *ArgTypes) {
  TIMESCOPE();
  __tgt_target_data_begin_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                                 ArgSizes, ArgTypes, nullptr, nullptr);
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

EXTERN void __tgt_target_data_begin_nowait(int64_t DeviceId, int32_t ArgNum,
                                           void **ArgsBase, void **Args,
                                           int64_t *ArgSizes, int64_t *ArgTypes,
                                           int32_t DepNum, void *DepList,
                                           int32_t NoAliasDepNum,
                                           void *NoAliasDepList) {
  TIMESCOPE();

  __tgt_target_data_begin_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                                 ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN void __tgt_target_data_begin_mapper(ident_t *Loc, int64_t DeviceId,
                                           int32_t ArgNum, void **ArgsBase,
                                           void **Args, int64_t *ArgSizes,
                                           int64_t *ArgTypes,
                                           map_var_info_t *ArgNames,
                                           void **ArgMappers) {
  TIMESCOPE_WITH_IDENT(Loc);
  DP("Entering data begin region for device %" PRId64 " with %d mappings\n",
     DeviceId, ArgNum);
  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return;
  }

  DeviceTy &Device = *PM->Devices[DeviceId];

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, ArgNum, ArgSizes, ArgTypes, ArgNames,
                         "Entering OpenMP data region");
#ifdef OMPTARGET_DEBUG
  for (int I = 0; I < ArgNum; ++I) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       I, DPxPTR(ArgsBase[I]), DPxPTR(Args[I]), ArgSizes[I], ArgTypes[I],
       (ArgNames) ? getNameFromMapping(ArgNames[I]).c_str() : "unknown");
  }
#endif

  void *CodePtr = nullptr;
  OMPT_IF_ENABLED(
      CodePtr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), CodePtr);
      ompt_interface.target_data_enter_begin(DeviceId, CodePtr);
      ompt_interface.target_trace_record_gen(DeviceId, ompt_target_enter_data,
                                             ompt_scope_begin, CodePtr););

  AsyncInfoTy AsyncInfo(Device);
  int Rc = targetDataBegin(Loc, Device, ArgNum, ArgsBase, Args, ArgSizes,
                           ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();
  handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);
  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      DeviceId, ompt_target_enter_data, ompt_scope_end, CodePtr);
                  ompt_interface.target_data_enter_end(DeviceId, CodePtr);
                  ompt_interface.ompt_state_clear(););
}

EXTERN void __tgt_target_data_begin_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {
  TIMESCOPE_WITH_IDENT(Loc);

  __tgt_target_data_begin_mapper(Loc, DeviceId, ArgNum, ArgsBase, Args,
                                 ArgSizes, ArgTypes, ArgNames, ArgMappers);
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int64_t DeviceId, int32_t ArgNum,
                                  void **ArgsBase, void **Args,
                                  int64_t *ArgSizes, int64_t *ArgTypes) {
  TIMESCOPE();
  __tgt_target_data_end_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                               ArgSizes, ArgTypes, nullptr, nullptr);
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

EXTERN void __tgt_target_data_end_nowait(int64_t DeviceId, int32_t ArgNum,
                                         void **ArgsBase, void **Args,
                                         int64_t *ArgSizes, int64_t *ArgTypes,
                                         int32_t DepNum, void *DepList,
                                         int32_t NoAliasDepNum,
                                         void *NoAliasDepList) {
  TIMESCOPE();

  __tgt_target_data_end_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                               ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN void __tgt_target_data_end_mapper(ident_t *Loc, int64_t DeviceId,
                                         int32_t ArgNum, void **ArgsBase,
                                         void **Args, int64_t *ArgSizes,
                                         int64_t *ArgTypes,
                                         map_var_info_t *ArgNames,
                                         void **ArgMappers) {
  TIMESCOPE_WITH_IDENT(Loc);
  DP("Entering data end region with %d mappings\n", ArgNum);
  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return;
  }

  DeviceTy &Device = *PM->Devices[DeviceId];

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, ArgNum, ArgSizes, ArgTypes, ArgNames,
                         "Exiting OpenMP data region");
#ifdef OMPTARGET_DEBUG
  for (int I = 0; I < ArgNum; ++I) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       I, DPxPTR(ArgsBase[I]), DPxPTR(Args[I]), ArgSizes[I], ArgTypes[I],
       (ArgNames) ? getNameFromMapping(ArgNames[I]).c_str() : "unknown");
  }
#endif

  AsyncInfoTy AsyncInfo(Device);

  void *CodePtr = nullptr;
  OMPT_IF_ENABLED(
      CodePtr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), CodePtr);
      ompt_interface.target_data_exit_begin(DeviceId, CodePtr);
      ompt_interface.target_trace_record_gen(DeviceId, ompt_target_exit_data,
                                             ompt_scope_begin, CodePtr););

  int Rc = targetDataEnd(Loc, Device, ArgNum, ArgsBase, Args, ArgSizes,
                         ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();
  handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      DeviceId, ompt_target_exit_data, ompt_scope_end, CodePtr);
                  ompt_interface.target_data_exit_end(DeviceId, CodePtr);
                  ompt_interface.ompt_state_clear(););
}

EXTERN void __tgt_target_data_end_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {
  TIMESCOPE_WITH_IDENT(Loc);

  __tgt_target_data_end_mapper(Loc, DeviceId, ArgNum, ArgsBase, Args, ArgSizes,
                               ArgTypes, ArgNames, ArgMappers);
}

EXTERN void __tgt_target_data_update(int64_t DeviceId, int32_t ArgNum,
                                     void **ArgsBase, void **Args,
                                     int64_t *ArgSizes, int64_t *ArgTypes) {
  TIMESCOPE();
  __tgt_target_data_update_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                                  ArgSizes, ArgTypes, nullptr, nullptr);
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
    int64_t DeviceId, int32_t ArgNum, void **ArgsBase, void **Args,
    int64_t *ArgSizes, int64_t *ArgTypes, int32_t DepNum, void *DepList,
    int32_t NoAliasDepNum, void *NoAliasDepList) {
  TIMESCOPE();

  __tgt_target_data_update_mapper(nullptr, DeviceId, ArgNum, ArgsBase, Args,
                                  ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN void __tgt_target_data_update_mapper(ident_t *Loc, int64_t DeviceId,
                                            int32_t ArgNum, void **ArgsBase,
                                            void **Args, int64_t *ArgSizes,
                                            int64_t *ArgTypes,
                                            map_var_info_t *ArgNames,
                                            void **ArgMappers) {
  TIMESCOPE_WITH_IDENT(Loc);
  DP("Entering data update with %d mappings\n", ArgNum);
  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return;
  }

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, ArgNum, ArgSizes, ArgTypes, ArgNames,
                         "Updating OpenMP data");

  void *CodePtr = nullptr;
  OMPT_IF_ENABLED(
      CodePtr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), CodePtr);
      ompt_interface.target_update_begin(DeviceId, CodePtr);
      ompt_interface.target_trace_record_gen(DeviceId, ompt_target_update,
                                             ompt_scope_begin, CodePtr););

  DeviceTy &Device = *PM->Devices[DeviceId];
  AsyncInfoTy AsyncInfo(Device);
  int Rc = targetDataUpdate(Loc, Device, ArgNum, ArgsBase, Args, ArgSizes,
                            ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();
  handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      DeviceId, ompt_target_update, ompt_scope_end, CodePtr);
                  ompt_interface.target_update_end(DeviceId, CodePtr);
                  ompt_interface.ompt_state_clear(););
}

EXTERN void __tgt_target_data_update_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, int32_t ArgNum, void **ArgsBase,
    void **Args, int64_t *ArgSizes, int64_t *ArgTypes, map_var_info_t *ArgNames,
    void **ArgMappers, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {
  TIMESCOPE_WITH_IDENT(Loc);

  __tgt_target_data_update_mapper(Loc, DeviceId, ArgNum, ArgsBase, Args,
                                  ArgSizes, ArgTypes, ArgNames, ArgMappers);
}

EXTERN int __tgt_target(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                        void **ArgsBase, void **Args, int64_t *ArgSizes,
                        int64_t *ArgTypes) {
  TIMESCOPE();
  return __tgt_target_mapper(nullptr, DeviceId, HostPtr, ArgNum, ArgsBase, Args,
                             ArgSizes, ArgTypes, nullptr, nullptr);
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
   
EXTERN int __tgt_target_nowait(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                               void **ArgsBase, void **Args, int64_t *ArgSizes,
                               int64_t *ArgTypes, int32_t DepNum, void *DepList,
                               int32_t NoAliasDepNum, void *NoAliasDepList) {
  TIMESCOPE();

  return __tgt_target_mapper(nullptr, DeviceId, HostPtr, ArgNum, ArgsBase, Args,
                             ArgSizes, ArgTypes, nullptr, nullptr);
}

EXTERN int __tgt_target_mapper(ident_t *Loc, int64_t DeviceId, void *HostPtr,
                               int32_t ArgNum, void **ArgsBase, void **Args,
                               int64_t *ArgSizes, int64_t *ArgTypes,
                               map_var_info_t *ArgNames, void **ArgMappers) {
  TIMESCOPE_WITH_IDENT(Loc);
  DP("Entering target region with entry point " DPxMOD " and device Id %" PRId64
     "\n",
     DPxPTR(HostPtr), DeviceId);
  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return OMP_TGT_FAIL;
  }

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, ArgNum, ArgSizes, ArgTypes, ArgNames,
                         "Entering OpenMP kernel");
#ifdef OMPTARGET_DEBUG
  for (int I = 0; I < ArgNum; ++I) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       I, DPxPTR(ArgsBase[I]), DPxPTR(Args[I]), ArgSizes[I], ArgTypes[I],
       (ArgNames) ? getNameFromMapping(ArgNames[I]).c_str() : "unknown");
  }
#endif

  DeviceTy &Device = *PM->Devices[DeviceId];
  AsyncInfoTy AsyncInfo(Device);

  void *CodePtr = nullptr;
  OMPT_IF_ENABLED(
      CodePtr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), CodePtr);
      ompt_interface.target_begin(DeviceId, CodePtr);
      ompt_interface.target_trace_record_gen(DeviceId, ompt_target,
                                             ompt_scope_begin, CodePtr););

  int Rc =
      target(Loc, Device, HostPtr, ArgNum, ArgsBase, Args, ArgSizes, ArgTypes,
             ArgNames, ArgMappers, 0, 0, false /*team*/, AsyncInfo);
  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();
  handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);
  assert(Rc == OFFLOAD_SUCCESS && "__tgt_target_mapper unexpected failure!");

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      DeviceId, ompt_target, ompt_scope_end, CodePtr);
                  ompt_interface.target_end(DeviceId, CodePtr);
                  ompt_interface.ompt_state_clear(););

  assert(Rc == OFFLOAD_SUCCESS && "__tgt_target_mapper unexpected failure!");
  return OMP_TGT_SUCCESS;
}

EXTERN int __tgt_target_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, void *HostPtr, int32_t ArgNum,
    void **ArgsBase, void **Args, int64_t *ArgSizes, int64_t *ArgTypes,
    map_var_info_t *ArgNames, void **ArgMappers, int32_t DepNum, void *DepList,
    int32_t NoAliasDepNum, void *NoAliasDepList) {
  TIMESCOPE_WITH_IDENT(Loc);

  return __tgt_target_mapper(Loc, DeviceId, HostPtr, ArgNum, ArgsBase, Args,
                             ArgSizes, ArgTypes, ArgNames, ArgMappers);
}

EXTERN int __tgt_target_teams(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                              void **ArgsBase, void **Args, int64_t *ArgSizes,
                              int64_t *ArgTypes, int32_t TeamNum,
                              int32_t ThreadLimit) {
  TIMESCOPE();
  return __tgt_target_teams_mapper(nullptr, DeviceId, HostPtr, ArgNum, ArgsBase,
                                   Args, ArgSizes, ArgTypes, nullptr, nullptr,
                                   TeamNum, ThreadLimit);
}

EXTERN int __tgt_target_teams_nowait(int64_t DeviceId, void *HostPtr,
                                     int32_t ArgNum, void **ArgsBase,
                                     void **Args, int64_t *ArgSizes,
                                     int64_t *ArgTypes, int32_t TeamNum,
                                     int32_t ThreadLimit, int32_t DepNum,
                                     void *DepList, int32_t NoAliasDepNum,
                                     void *NoAliasDepList) {
  TIMESCOPE();

  return __tgt_target_teams_mapper(nullptr, DeviceId, HostPtr, ArgNum, ArgsBase,
                                   Args, ArgSizes, ArgTypes, nullptr, nullptr,
                                   TeamNum, ThreadLimit);
}

EXTERN int __tgt_target_teams_mapper(ident_t *Loc, int64_t DeviceId,
                                     void *HostPtr, int32_t ArgNum,
                                     void **ArgsBase, void **Args,
                                     int64_t *ArgSizes, int64_t *ArgTypes,
                                     map_var_info_t *ArgNames,
                                     void **ArgMappers, int32_t TeamNum,
                                     int32_t ThreadLimit) {
  DP("Entering target region with entry point " DPxMOD " and device Id %" PRId64
     "\n",
     DPxPTR(HostPtr), DeviceId);
  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return OMP_TGT_FAIL;
  }

  if (getInfoLevel() & OMP_INFOTYPE_KERNEL_ARGS)
    printKernelArguments(Loc, DeviceId, ArgNum, ArgSizes, ArgTypes, ArgNames,
                         "Entering OpenMP kernel");
#ifdef OMPTARGET_DEBUG
  for (int I = 0; I < ArgNum; ++I) {
    DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
       ", Type=0x%" PRIx64 ", Name=%s\n",
       I, DPxPTR(ArgsBase[I]), DPxPTR(Args[I]), ArgSizes[I], ArgTypes[I],
       (ArgNames) ? getNameFromMapping(ArgNames[I]).c_str() : "unknown");
  }
#endif

  DeviceTy &Device = *PM->Devices[DeviceId];
  AsyncInfoTy AsyncInfo(Device);

  void *CodePtr = nullptr;
  OMPT_IF_ENABLED(
      CodePtr = OMPT_GET_RETURN_ADDRESS(0);
      ompt_interface.ompt_state_set(OMPT_GET_FRAME_ADDRESS(0), CodePtr);
      ompt_interface.target_begin(DeviceId, CodePtr);
      ompt_interface.target_trace_record_gen(DeviceId, ompt_target,
                                             ompt_scope_begin, CodePtr););

  int Rc = target(Loc, Device, HostPtr, ArgNum, ArgsBase, Args, ArgSizes,
                  ArgTypes, ArgNames, ArgMappers, TeamNum, ThreadLimit,
                  true /*team*/, AsyncInfo);
  if (Rc == OFFLOAD_SUCCESS)
    Rc = AsyncInfo.synchronize();
  handleTargetOutcome(Rc == OFFLOAD_SUCCESS, Loc);
  assert(Rc == OFFLOAD_SUCCESS && "offload failed");

  OMPT_IF_ENABLED(ompt_interface.target_trace_record_gen(
      DeviceId, ompt_target, ompt_scope_end, CodePtr);
                  ompt_interface.target_end(DeviceId, CodePtr);
                  ompt_interface.ompt_state_clear(););

  assert(Rc == OFFLOAD_SUCCESS &&
         "__tgt_target_teams_mapper unexpected failure!");
  return OMP_TGT_SUCCESS;
}

EXTERN int __tgt_target_teams_nowait_mapper(
    ident_t *Loc, int64_t DeviceId, void *HostPtr, int32_t ArgNum,
    void **ArgsBase, void **Args, int64_t *ArgSizes, int64_t *ArgTypes,
    map_var_info_t *ArgNames, void **ArgMappers, int32_t TeamNum,
    int32_t ThreadLimit, int32_t DepNum, void *DepList, int32_t NoAliasDepNum,
    void *NoAliasDepList) {
  TIMESCOPE_WITH_IDENT(Loc);

  return __tgt_target_teams_mapper(Loc, DeviceId, HostPtr, ArgNum, ArgsBase,
                                   Args, ArgSizes, ArgTypes, ArgNames,
                                   ArgMappers, TeamNum, ThreadLimit);
}

// Get the current number of components for a user-defined mapper.
EXTERN int64_t __tgt_mapper_num_components(void *RtMapperHandle) {
  TIMESCOPE();
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)RtMapperHandle;
  int64_t Size = MapperComponentsPtr->Components.size();
  DP("__tgt_mapper_num_components(Handle=" DPxMOD ") returns %" PRId64 "\n",
     DPxPTR(RtMapperHandle), Size);
  return Size;
}

// Push back one component for a user-defined mapper.
EXTERN void __tgt_push_mapper_component(void *RtMapperHandle, void *Base,
                                        void *Begin, int64_t Size, int64_t Type,
                                        void *Name) {
  TIMESCOPE();
  DP("__tgt_push_mapper_component(Handle=" DPxMOD
     ") adds an entry (Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
     ", Type=0x%" PRIx64 ", Name=%s).\n",
     DPxPTR(RtMapperHandle), DPxPTR(Base), DPxPTR(Begin), Size, Type,
     (Name) ? getNameFromMapping(Name).c_str() : "unknown");
  auto *MapperComponentsPtr = (struct MapperComponentsTy *)RtMapperHandle;
  MapperComponentsPtr->Components.push_back(
      MapComponentInfoTy(Base, Begin, Size, Type, Name));
}

EXTERN void __kmpc_push_target_tripcount(ident_t *Loc, int64_t DeviceId,
                                         uint64_t LoopTripcount) {
  __kmpc_push_target_tripcount_mapper(Loc, DeviceId, LoopTripcount);
}

EXTERN void __kmpc_push_target_tripcount_mapper(ident_t *Loc, int64_t DeviceId,
                                                uint64_t LoopTripcount) {
  TIMESCOPE_WITH_IDENT(Loc);
  if (checkDeviceAndCtors(DeviceId, Loc)) {
    DP("Not offloading to device %" PRId64 "\n", DeviceId);
    return;
  }

  DP("__kmpc_push_target_tripcount(%" PRId64 ", %" PRIu64 ")\n", DeviceId,
     LoopTripcount);
  PM->TblMapMtx.lock();
  PM->Devices[DeviceId]->LoopTripCnt.emplace(__kmpc_global_thread_num(NULL),
                                             LoopTripcount);
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

EXTERN int __tgt_print_device_info(int64_t DeviceId) {
  return PM->Devices[DeviceId]->printDeviceInfo(
      PM->Devices[DeviceId]->RTLDeviceID);
}
