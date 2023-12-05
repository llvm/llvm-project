//===-- InteropAPI.cpp - Implementation of OpenMP interoperability API ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpenMP/InteropAPI.h"
#include "OpenMP/InternalTypes.h"
#include "OpenMP/omp.h"

#include "PluginManager.h"
#include "device.h"
#include "omptarget.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <cstring>

extern "C" {

void __kmpc_omp_wait_deps(ident_t *loc_ref, int32_t gtid, int32_t ndeps,
                          kmp_depend_info_t *dep_list, int32_t ndeps_noalias,
                          kmp_depend_info_t *noalias_dep_list)
    __attribute__((weak));

} // extern "C"

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
intptr_t getProperty<intptr_t>(omp_interop_val_t &InteropVal,
                               omp_interop_property_t Property, int *Err) {
  switch (Property) {
  case omp_ipr_fr_id:
    return InteropVal.backend_type_id;
  case omp_ipr_vendor:
    return InteropVal.vendor_id;
  case omp_ipr_device_num:
    return InteropVal.device_id;
  default:;
  }
  getTypeMismatch(Property, Err);
  return 0;
}

template <>
const char *getProperty<const char *>(omp_interop_val_t &InteropVal,
                                      omp_interop_property_t Property,
                                      int *Err) {
  switch (Property) {
  case omp_ipr_fr_id:
    return InteropVal.interop_type == kmp_interop_type_tasksync
               ? "tasksync"
               : "device+context";
  case omp_ipr_vendor_name:
    return getVendorIdToStr(InteropVal.vendor_id);
  default:
    getTypeMismatch(Property, Err);
    return nullptr;
  }
}

template <>
void *getProperty<void *>(omp_interop_val_t &InteropVal,
                          omp_interop_property_t Property, int *Err) {
  switch (Property) {
  case omp_ipr_device:
    if (InteropVal.device_info.Device)
      return InteropVal.device_info.Device;
    *Err = omp_irc_no_value;
    return const_cast<char *>(InteropVal.err_str);
  case omp_ipr_device_context:
    return InteropVal.device_info.Context;
  case omp_ipr_targetsync:
    return InteropVal.async_info->Queue;
  default:;
  }
  getTypeMismatch(Property, Err);
  return nullptr;
}

bool getPropertyCheck(omp_interop_val_t **InteropPtr,
                      omp_interop_property_t Property, int *Err) {
  if (Err)
    *Err = omp_irc_success;
  if (!InteropPtr) {
    if (Err)
      *Err = omp_irc_empty;
    return false;
  }
  if (Property >= 0 || Property < omp_ipr_first) {
    if (Err)
      *Err = omp_irc_out_of_range;
    return false;
  }
  if (Property == omp_ipr_targetsync &&
      (*InteropPtr)->interop_type != kmp_interop_type_tasksync) {
    if (Err)
      *Err = omp_irc_other;
    return false;
  }
  if ((Property == omp_ipr_device || Property == omp_ipr_device_context) &&
      (*InteropPtr)->interop_type == kmp_interop_type_tasksync) {
    if (Err)
      *Err = omp_irc_other;
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

static const char *copyErrorString(llvm::Error &&Err) {
  // TODO: Use the error string while avoiding leaks.
  std::string ErrMsg = llvm::toString(std::move(Err));
  char *UsrMsg = reinterpret_cast<char *>(malloc(ErrMsg.size() + 1));
  strcpy(UsrMsg, ErrMsg.c_str());
  return UsrMsg;
};

extern "C" {

void __tgt_interop_init(ident_t *LocRef, int32_t Gtid,
                        omp_interop_val_t *&InteropPtr,
                        kmp_interop_type_t InteropType, int32_t DeviceId,
                        int32_t Ndeps, kmp_depend_info_t *DepList,
                        int32_t HaveNowait) {
  int32_t NdepsNoalias = 0;
  kmp_depend_info_t *NoaliasDepList = NULL;
  assert(InteropType != kmp_interop_type_unknown &&
         "Cannot initialize with unknown interop_type!");
  if (DeviceId == -1) {
    DeviceId = omp_get_default_device();
  }

  if (InteropType == kmp_interop_type_tasksync) {
    __kmpc_omp_wait_deps(LocRef, Gtid, Ndeps, DepList, NdepsNoalias,
                         NoaliasDepList);
  }

  InteropPtr = new omp_interop_val_t(DeviceId, InteropType);

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr) {
    InteropPtr->err_str = copyErrorString(DeviceOrErr.takeError());
    return;
  }

  DeviceTy &Device = *DeviceOrErr;
  if (!Device.RTL || !Device.RTL->init_device_info ||
      Device.RTL->init_device_info(DeviceId, &(InteropPtr)->device_info,
                                   &(InteropPtr)->err_str)) {
    delete InteropPtr;
    InteropPtr = omp_interop_none;
  }
  if (InteropType == kmp_interop_type_tasksync) {
    if (!Device.RTL || !Device.RTL->init_async_info ||
        Device.RTL->init_async_info(DeviceId, &(InteropPtr)->async_info)) {
      delete InteropPtr;
      InteropPtr = omp_interop_none;
    }
  }
}

void __tgt_interop_use(ident_t *LocRef, int32_t Gtid,
                       omp_interop_val_t *&InteropPtr, int32_t DeviceId,
                       int32_t Ndeps, kmp_depend_info_t *DepList,
                       int32_t HaveNowait) {
  int32_t NdepsNoalias = 0;
  kmp_depend_info_t *NoaliasDepList = NULL;
  assert(InteropPtr && "Cannot use nullptr!");
  omp_interop_val_t *InteropVal = InteropPtr;
  if (DeviceId == -1) {
    DeviceId = omp_get_default_device();
  }
  assert(InteropVal != omp_interop_none &&
         "Cannot use uninitialized interop_ptr!");
  assert((DeviceId == -1 || InteropVal->device_id == DeviceId) &&
         "Inconsistent device-id usage!");

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr) {
    InteropPtr->err_str = copyErrorString(DeviceOrErr.takeError());
    return;
  }

  if (InteropVal->interop_type == kmp_interop_type_tasksync) {
    __kmpc_omp_wait_deps(LocRef, Gtid, Ndeps, DepList, NdepsNoalias,
                         NoaliasDepList);
  }
  // TODO Flush the queue associated with the interop through the plugin
}

void __tgt_interop_destroy(ident_t *LocRef, int32_t Gtid,
                           omp_interop_val_t *&InteropPtr, int32_t DeviceId,
                           int32_t Ndeps, kmp_depend_info_t *DepList,
                           int32_t HaveNowait) {
  int32_t NdepsNoalias = 0;
  kmp_depend_info_t *NoaliasDepList = NULL;
  assert(InteropPtr && "Cannot use nullptr!");
  omp_interop_val_t *InteropVal = InteropPtr;
  if (DeviceId == -1) {
    DeviceId = omp_get_default_device();
  }

  if (InteropVal == omp_interop_none)
    return;

  assert((DeviceId == -1 || InteropVal->device_id == DeviceId) &&
         "Inconsistent device-id usage!");
  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr) {
    InteropPtr->err_str = copyErrorString(DeviceOrErr.takeError());
    return;
  }

  if (InteropVal->interop_type == kmp_interop_type_tasksync) {
    __kmpc_omp_wait_deps(LocRef, Gtid, Ndeps, DepList, NdepsNoalias,
                         NoaliasDepList);
  }
  // TODO Flush the queue associated with the interop through the plugin
  // TODO Signal out dependences

  delete InteropPtr;
  InteropPtr = omp_interop_none;
}

} // extern "C"
