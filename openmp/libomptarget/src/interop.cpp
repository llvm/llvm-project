//===---------------interop.cpp - Implementation of interop directive -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "interop.h"
#include "private.h"

namespace {
omp_interop_rc_t getPropertyErrorType(omp_interop_property_t Property) {
  switch (Property) {
  case omp_ipr_fr_id:
    return omp_irc_type_int;
  case omp_ipr_fr_name:
    return omp_irc_type_str;
  case omp_ipr_vendor:
    return omp_irc_type_int;
  case omp_ipr_vendor_name:
    return omp_irc_type_str;
  case omp_ipr_device_num:
    return omp_irc_type_int;
  case omp_ipr_platform:
    return omp_irc_type_int;
  case omp_ipr_device:
    return omp_irc_type_ptr;
  case omp_ipr_device_context:
    return omp_irc_type_ptr;
  case omp_ipr_targetsync:
    return omp_irc_type_ptr;
  };
  return omp_irc_no_value;
}

void getTypeMismatch(omp_interop_property_t Property, int *Err) {
  if (Err)
    *Err = getPropertyErrorType(Property);
}

const char *getVendorIdToStr(const omp_foreign_runtime_ids_t VendorId) {
  switch (VendorId) {
  case cuda:
    return ("cuda");
  case cuda_driver:
    return ("cuda_driver");
  case opencl:
    return ("opencl");
  case sycl:
    return ("sycl");
  case hip:
    return ("hip");
  case level_zero:
    return ("level_zero");
  }
  return ("unknown");
}

template <typename PropertyTy>
PropertyTy getProperty(omp_interop_val_t &InteropVal,
                       omp_interop_property_t Property, int *Err);

template <>
intptr_t getProperty<intptr_t>(omp_interop_val_t &interop_val,
                               omp_interop_property_t property, int *err) {
  switch (property) {
  case omp_ipr_fr_id:
    return interop_val.backend_type_id;
  case omp_ipr_vendor:
    return interop_val.vendor_id;
  case omp_ipr_device_num:
    return interop_val.device_id;
  default:;
  }
  getTypeMismatch(property, err);
  return 0;
}

template <>
const char *getProperty<const char *>(omp_interop_val_t &interop_val,
                                      omp_interop_property_t property,
                                      int *err) {
  switch (property) {
  case omp_ipr_fr_id:
    return interop_val.interop_type == kmp_interop_type_tasksync
               ? "tasksync"
               : "device+context";
  case omp_ipr_vendor_name:
    return getVendorIdToStr(interop_val.vendor_id);
  default:
    getTypeMismatch(property, err);
    return nullptr;
  }
}

template <>
void *getProperty<void *>(omp_interop_val_t &interop_val,
                          omp_interop_property_t property, int *err) {
  switch (property) {
  case omp_ipr_device:
    if (interop_val.device_info.Device)
      return interop_val.device_info.Device;
    *err = omp_irc_no_value;
    return const_cast<char *>(interop_val.err_str);
  case omp_ipr_device_context:
    return interop_val.device_info.Context;
  case omp_ipr_targetsync:
    return interop_val.async_info->Queue;
  default:;
  }
  getTypeMismatch(property, err);
  return nullptr;
}

bool getPropertyCheck(omp_interop_val_t **interop_ptr,
                      omp_interop_property_t property, int *err) {
  if (err)
    *err = omp_irc_success;
  if (!interop_ptr) {
    if (err)
      *err = omp_irc_empty;
    return false;
  }
  if (property >= 0 || property < omp_ipr_first) {
    if (err)
      *err = omp_irc_out_of_range;
    return false;
  }
  if (property == omp_ipr_targetsync &&
      (*interop_ptr)->interop_type != kmp_interop_type_tasksync) {
    if (err)
      *err = omp_irc_other;
    return false;
  }
  if ((property == omp_ipr_device || property == omp_ipr_device_context) &&
      (*interop_ptr)->interop_type == kmp_interop_type_tasksync) {
    if (err)
      *err = omp_irc_other;
    return false;
  }
  return true;
}

} // namespace

#define __OMP_GET_INTEROP_TY(RETURN_TYPE, SUFFIX)                              \
  RETURN_TYPE omp_get_interop_##SUFFIX(const omp_interop_t interop,            \
                                       omp_interop_property_t property_id,     \
                                       int *err) {                             \
    omp_interop_val_t *interop_val = (omp_interop_val_t *)interop;             \
    assert((interop_val)->interop_type == kmp_interop_type_tasksync);          \
    if (!getPropertyCheck(&interop_val, property_id, err)) {                   \
      return (RETURN_TYPE)(0);                                                 \
    }                                                                          \
    return getProperty<RETURN_TYPE>(*interop_val, property_id, err);           \
  }
__OMP_GET_INTEROP_TY(intptr_t, int)
__OMP_GET_INTEROP_TY(void *, ptr)
__OMP_GET_INTEROP_TY(const char *, str)
#undef __OMP_GET_INTEROP_TY

#define __OMP_GET_INTEROP_TY3(RETURN_TYPE, SUFFIX)                             \
  RETURN_TYPE omp_get_interop_##SUFFIX(const omp_interop_t interop,            \
                                       omp_interop_property_t property_id) {   \
    int err;                                                                   \
    omp_interop_val_t *interop_val = (omp_interop_val_t *)interop;             \
    if (!getPropertyCheck(&interop_val, property_id, &err)) {                  \
      return (RETURN_TYPE)(0);                                                 \
    }                                                                          \
    return nullptr;                                                            \
    return getProperty<RETURN_TYPE>(*interop_val, property_id, &err);          \
  }
__OMP_GET_INTEROP_TY3(const char *, name)
__OMP_GET_INTEROP_TY3(const char *, type_desc)
__OMP_GET_INTEROP_TY3(const char *, rc_desc)
#undef __OMP_GET_INTEROP_TY3

typedef int64_t kmp_int64;

#ifdef __cplusplus
extern "C" {
#endif
void __tgt_interop_init(ident_t *loc_ref, kmp_int32 gtid,
                        omp_interop_val_t *&interop_ptr,
                        kmp_interop_type_t interop_type, kmp_int32 device_id,
                        kmp_int64 ndeps, kmp_depend_info_t *dep_list,
                        kmp_int32 have_nowait) {
  kmp_int32 ndeps_noalias = 0;
  kmp_depend_info_t *noalias_dep_list = NULL;
  assert(interop_type != kmp_interop_type_unknown &&
         "Cannot initialize with unknown interop_type!");
  if (device_id == -1) {
    device_id = omp_get_default_device();
  }

  if (interop_type == kmp_interop_type_tasksync) {
    __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
  }

  interop_ptr = new omp_interop_val_t(device_id, interop_type);
  if (!device_is_ready(device_id)) {
    interop_ptr->err_str = "Device not ready!";
    return;
  }

  DeviceTy &Device = *PM->Devices[device_id];
  if (!Device.RTL || !Device.RTL->init_device_info ||
      Device.RTL->init_device_info(device_id, &(interop_ptr)->device_info,
                                   &(interop_ptr)->err_str)) {
    delete interop_ptr;
    interop_ptr = omp_interop_none;
  }
  if (interop_type == kmp_interop_type_tasksync) {
    if (!Device.RTL || !Device.RTL->init_async_info ||
        Device.RTL->init_async_info(device_id, &(interop_ptr)->async_info)) {
      delete interop_ptr;
      interop_ptr = omp_interop_none;
    }
  }
}

void __tgt_interop_use(ident_t *loc_ref, kmp_int32 gtid,
                       omp_interop_val_t *&interop_ptr, kmp_int32 device_id,
                       kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                       kmp_int32 have_nowait) {
  kmp_int32 ndeps_noalias = 0;
  kmp_depend_info_t *noalias_dep_list = NULL;
  assert(interop_ptr && "Cannot use nullptr!");
  omp_interop_val_t *interop_val = interop_ptr;
  if (device_id == -1) {
    device_id = omp_get_default_device();
  }
  assert(interop_val != omp_interop_none &&
         "Cannot use uninitialized interop_ptr!");
  assert((device_id == -1 || interop_val->device_id == device_id) &&
         "Inconsistent device-id usage!");

  if (!device_is_ready(device_id)) {
    interop_ptr->err_str = "Device not ready!";
    return;
  }

  if (interop_val->interop_type == kmp_interop_type_tasksync) {
    __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
  }
  // TODO Flush the queue associated with the interop through the plugin
}

void __tgt_interop_destroy(ident_t *loc_ref, kmp_int32 gtid,
                           omp_interop_val_t *&interop_ptr, kmp_int32 device_id,
                           kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                           kmp_int32 have_nowait) {
  kmp_int32 ndeps_noalias = 0;
  kmp_depend_info_t *noalias_dep_list = NULL;
  assert(interop_ptr && "Cannot use nullptr!");
  omp_interop_val_t *interop_val = interop_ptr;
  if (device_id == -1) {
    device_id = omp_get_default_device();
  }

  if (interop_val == omp_interop_none)
    return;

  assert((device_id == -1 || interop_val->device_id == device_id) &&
         "Inconsistent device-id usage!");
  if (!device_is_ready(device_id)) {
    interop_ptr->err_str = "Device not ready!";
    return;
  }

  if (interop_val->interop_type == kmp_interop_type_tasksync) {
    __kmpc_omp_wait_deps(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list);
  }
  // TODO Flush the queue associated with the interop through the plugin
  // TODO Signal out dependences

  delete interop_ptr;
  interop_ptr = omp_interop_none;
}
#ifdef __cplusplus
} // extern "C"
#endif
