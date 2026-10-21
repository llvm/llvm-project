//===-- InteropAPI.cpp - Implementation of OpenMP interoperability API ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpenMP/InteropAPI.h"
#include "OpenMP/InternalTypes.h"
#include "OpenMP/OffloadRTL.h"
#include "OpenMP/omp.h"

#include "OffloadPolicy.h"
#include "PluginManager.h"
#include "device.h"
#include "omptarget.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <cstring>

using namespace llvm::omp::target::debug;

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

static const char *VendorStrTbl[] = {
    "unknown", "amd",   "arm",  "bsc", "fujitsu", "gnu", "hpe",
    "ibm",     "intel", "llvm", "nec", "nvidia",  "ti"};
const char *getVendorIdToStr(const omp_vendor_id_t VendorId) {
  if (VendorId < omp_vendor_unknown || VendorId >= omp_vendor_last)
    return ("unknown");
  return VendorStrTbl[VendorId];
}

static const char *ForeignRuntimeStrTbl[] = {
    "none", "cuda", "cuda_driver", "opencl",
    "sycl", "hip",  "level_zero",  "hsa"};
const char *getForeignRuntimeIdToStr(const omp_foreign_runtime_id_t FrId) {
  if (FrId < omp_fr_none || FrId >= omp_fr_last)
    return ("unknown");
  return ForeignRuntimeStrTbl[FrId];
}

constexpr interop_spec_t NoInteropSpec{omp_fr_none, {true, 0}, 0};

/// Return the interop specification supported by \p Device for the given
/// interop type, or NoInteropSpec if the device supports none.
/// TODO: the user-provided preference list is not honored yet.
interop_spec_t selectInteropPreference(DeviceTy &Device, int32_t InteropType,
                                       int32_t NumPrefers,
                                       interop_spec_t *Prefers) {

  interop_spec_t SelectedSpec = NoInteropSpec;
  switch (Device.getBackend()) {
  case OL_PLATFORM_BACKEND_CUDA:
    SelectedSpec.fr_id = omp_fr_cuda;
    break;
  case OL_PLATFORM_BACKEND_AMDGPU:
    // TODO: update once targetsync is supported.
    if (InteropType == kmp_interop_type_targetsync)
      return NoInteropSpec;
    SelectedSpec.fr_id = omp_fr_hsa;
    break;
  case OL_PLATFORM_BACKEND_LEVEL_ZERO:
    SelectedSpec.fr_id = omp_fr_level_zero;
    break;
  default:;
  }
  return SelectedSpec;
}

omp_vendor_id_t getVendorForForeignRuntime(int32_t FrId) {
  // Vendor of each foreign runtime, indexed by omp_foreign_runtime_id_t.
  static constexpr omp_vendor_id_t VendorTbl[omp_fr_last] = {
      /*none*/ omp_vendor_unknown,
      /*cuda*/ omp_vendor_nvidia,
      /*cuda_driver*/ omp_vendor_unknown,
      /*opencl*/ omp_vendor_unknown,
      /*sycl*/ omp_vendor_unknown,
      /*hip*/ omp_vendor_unknown,
      /*level_zero*/ omp_vendor_intel,
      /*hsa*/ omp_vendor_amd};
  if (FrId < omp_fr_none || FrId >= omp_fr_last)
    return omp_vendor_unknown;
  return VendorTbl[FrId];
}

/// Query a backend-native handle, returning null when the backend exposes
/// none.
template <typename HandleTy, typename InfoTy, typename GetInfoTy>
void *getNativeHandle(GetInfoTy GetInfo, HandleTy Handle, InfoTy Info) {
  void *Native = nullptr;
  if (auto *Res = GetInfo(Handle, Info, sizeof(Native), &Native)) {
    ODBG(ODT_Interface) << "Could not query native handle: " << Res->Details;
    return nullptr;
  }
  return Native;
}

/// Build an interop object for \p Device following \p Spec. Returns
/// omp_interop_none if it could not be created.
omp_interop_val_t *createInterop(DeviceTy &Device, int32_t InteropType,
                                 const interop_spec_t &Spec) {
  auto *Interop = new omp_interop_val_t(
      Device.DeviceID, static_cast<kmp_interop_type_t>(InteropType));
  Interop->fr_id = static_cast<omp_foreign_runtime_id_t>(Spec.fr_id);
  Interop->vendor_id = getVendorForForeignRuntime(Spec.fr_id);
  Interop->attrs = Spec.attrs;
  Interop->impl_attrs = Spec.impl_attrs;

  Interop->native_device = getNativeHandle(olGetDeviceInfo, Device.DeviceHandle,
                                           OL_DEVICE_INFO_NATIVE_HANDLE);
  if (!Interop->native_device)
    Interop->err_str = "the device does not expose a native handle";
  Interop->native_platform =
      getNativeHandle(olGetDeviceInfo, Device.DeviceHandle,
                      OL_DEVICE_INFO_NATIVE_PLATFORM_HANDLE);
  Interop->native_context = getNativeHandle(olGetContextInfo, Device.Context,
                                            OL_CONTEXT_INFO_NATIVE_HANDLE);

  if (InteropType == kmp_interop_type_targetsync) {
    if (auto *Res = olCreateQueue(Device.Context, Device.DeviceHandle,
                                  &Interop->queue)) {
      REPORT() << "Failed to create queue for interop on device "
               << Device.DeviceID << ": " << Res->Details;
      delete Interop;
      return omp_interop_none;
    }
    Interop->native_queue = getNativeHandle(olGetQueueInfo, Interop->queue,
                                            OL_QUEUE_INFO_NATIVE_HANDLE);
    if (!Interop->native_queue) {
      REPORT() << "The queue of device " << Device.DeviceID
               << " does not expose a native handle for targetsync";
      if (auto *Res = olDestroyQueue(Interop->queue))
        REPORT() << "Failed to destroy queue: " << Res->Details;
      delete Interop;
      return omp_interop_none;
    }
  }

  return Interop;
}

template <typename PropertyTy>
PropertyTy getProperty(omp_interop_val_t &InteropVal,
                       omp_interop_property_t Property, int *Err);

template <>
intptr_t getProperty<intptr_t>(omp_interop_val_t &InteropVal,
                               omp_interop_property_t Property, int *Err) {
  switch (Property) {
  case omp_ipr_fr_id:
    return InteropVal.fr_id;
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
  case omp_ipr_fr_name:
    return getForeignRuntimeIdToStr(InteropVal.fr_id);
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
    if (InteropVal.native_device)
      return InteropVal.native_device;
    *Err = omp_irc_no_value;
    return const_cast<char *>(InteropVal.err_str);
  case omp_ipr_platform:
    return InteropVal.native_platform;
  case omp_ipr_device_context:
    return InteropVal.native_context;
  case omp_ipr_targetsync:
    return InteropVal.native_queue;
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
      (*InteropPtr)->interop_type != kmp_interop_type_targetsync) {
    if (Err)
      *Err = omp_irc_other;
    return false;
  }
  if ((Property == omp_ipr_device || Property == omp_ipr_device_context)) {
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

extern "C" {

omp_interop_val_t *__tgt_interop_get(ident_t *LocRef, int32_t InteropType,
                                     int64_t DeviceNum, int32_t NumPrefers,
                                     interop_spec_t *Prefers,
                                     interop_ctx_t *Ctx, dep_pack_t *Deps) {

  ODBG(ODT_Interface) << "Call to " << __func__ << " with device_num "
                      << DeviceNum << ", interop type " << InteropType
                      << ", number of preferred specs " << NumPrefers
                      << (Ctx->flags.implicit ? " (implicit)" : "")
                      << (Ctx->flags.nowait ? " (nowait)" : "");

  if (OffloadPolicy::get(*PM).Kind == OffloadPolicy::DISABLED)
    return omp_interop_none;

  // Now, try to create an interop with device_num.
  if (DeviceNum == OFFLOAD_DEVICE_DEFAULT)
    DeviceNum = omp_get_default_device();

  auto gtid = Ctx->gtid;

  if (InteropType == kmp_interop_type_targetsync) {
    if (Ctx->flags.nowait)
      ODBG(ODT_Interface) << "Warning: nowait flag on interop creation not "
                             "supported yet. Ignored";
    if (Deps)
      __kmpc_omp_wait_deps(LocRef, gtid, Deps->ndeps, Deps->deplist,
                           Deps->ndeps_noalias, Deps->noalias_deplist);
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr) {
    std::string ErrStr = toString(DeviceOrErr.takeError());
    ODBG(ODT_Interface) << "Couldn't find device " << DeviceNum
                        << " while constructing interop object: " << ErrStr;
    return omp_interop_none;
  }
  auto &Device = *DeviceOrErr;
  omp_interop_val_t *Interop = omp_interop_none;
  auto InteropSpec =
      selectInteropPreference(Device, InteropType, NumPrefers, Prefers);
  if (InteropSpec.fr_id == omp_fr_none) {
    ODBG(ODT_Interface) << "Interop request not supported by device "
                        << DeviceNum;
    return omp_interop_none;
  }
  ODBG(ODT_Interface) << "Selected interop preference is fr_id="
                      << getForeignRuntimeIdToStr(
                             (omp_foreign_runtime_id_t)InteropSpec.fr_id)
                      << (InteropSpec.attrs.inorder ? " inorder" : "")
                      << " impl_attrs=" << InteropSpec.impl_attrs;

  if (Ctx->flags.implicit) {
    // This is a request for an RTL managed interop object.
    // Get it from the InteropTbl if possible
    for (auto iop : PM->InteropTbl) {
      if (iop->isCompatibleWith(InteropType, InteropSpec, DeviceNum, gtid)) {
        Interop = iop;
        Interop->markDirty();
        ODBG(ODT_Interface)
            << "Reused interop " << Interop << " from device number "
            << DeviceNum << " for gtid " << gtid;
        return Interop;
      }
    }
  }

  Interop = createInterop(Device, InteropType, InteropSpec);
  if (Interop == omp_interop_none)
    return omp_interop_none;

  ODBG(ODT_Interface) << "Created an interop " << Interop
                      << " from device number " << DeviceNum;

  if (Ctx->flags.implicit) {
    // register the new implicit interop in the RTL
    Interop->setOwner(gtid);
    Interop->markDirty();
    PM->InteropTbl.add(Interop);
  } else {
    Interop->setOwner(omp_interop_val_t::no_owner);
  }

  return Interop;
}

int __tgt_interop_use60(ident_t *LocRef, omp_interop_val_t *Interop,
                        interop_ctx_t *Ctx, dep_pack_t *Deps) {
  bool Nowait = Ctx->flags.nowait;
  ODBG(ODT_Interface) << "Call to " << __func__ << " with interop " << Interop
                      << ", nowait " << Nowait;
  if (OffloadPolicy::get(*PM).Kind == OffloadPolicy::DISABLED || !Interop)
    return OFFLOAD_FAIL;

  if (Interop->interop_type == kmp_interop_type_targetsync) {
    if (Deps) {
      if (Nowait) {
        ODBG(ODT_Interface)
            << "Warning: nowait flag on interop use with dependences "
               "not supported yet. Ignored";
        Nowait = false;
      }

      __kmpc_omp_wait_deps(LocRef, Ctx->gtid, Deps->ndeps, Deps->deplist,
                           Deps->ndeps_noalias, Deps->noalias_deplist);
    }
  }

  // A nowait use needs no barrier: the queue is in-order, so it already
  // orders the OpenMP work against the work the foreign runtime submits.
  if (!Nowait && Interop->queue) {
    Interop->sync_barrier();
    Interop->markClean();
  }

  return OFFLOAD_SUCCESS;
}

int __tgt_interop_release(ident_t *LocRef, omp_interop_val_t *Interop,
                          interop_ctx_t *Ctx, dep_pack_t *Deps) {
  ODBG(ODT_Interface) << "Call to " << __func__ << " with interop " << Interop;

  if (OffloadPolicy::get(*PM).Kind == OffloadPolicy::DISABLED || !Interop)
    return OFFLOAD_FAIL;

  if (Interop->interop_type == kmp_interop_type_targetsync) {
    if (Ctx->flags.nowait)
      ODBG(ODT_Interface)
          << "Warning: nowait flag on interop destroy not supported yet. "
             "Ignored";
    if (Deps) {
      __kmpc_omp_wait_deps(LocRef, Ctx->gtid, Deps->ndeps, Deps->deplist,
                           Deps->ndeps_noalias, Deps->noalias_deplist);
    }
  }

  return Interop->release();
}

EXTERN int ompx_interop_add_completion_callback(omp_interop_val_t *Interop,
                                                ompx_interop_cb_t *CB,
                                                void *Data) {
  ODBG(ODT_Interface) << "Call to " << __func__ << " with interop " << Interop
                      << ", property callback " << reinterpret_cast<void *>(CB)
                      << " and data " << Data;

  if (OffloadPolicy::get(*PM).Kind == OffloadPolicy::DISABLED || !Interop)
    return omp_irc_other;

  Interop->addCompletionCb(CB, Data);

  return omp_irc_success;
}

// Backwards compatibility wrappers
void __tgt_interop_init(ident_t *LocRef, int32_t Gtid,
                        omp_interop_val_t *&InteropPtr, int32_t InteropType,
                        int32_t DeviceId, int32_t Ndeps,
                        kmp_depend_info_t *DepList, int32_t HaveNowait) {
  constexpr int32_t old_kmp_interop_type_targetsync = 2;
  interop_ctx_t Ctx = {0, {false, (bool)HaveNowait, 0}, Gtid};
  dep_pack_t Deps = {Ndeps, 0, DepList, nullptr};
  InteropPtr =
      __tgt_interop_get(LocRef,
                        InteropType == old_kmp_interop_type_targetsync
                            ? kmp_interop_type_targetsync
                            : kmp_interop_type_target,
                        DeviceId, 0, nullptr, &Ctx, Ndeps ? &Deps : nullptr);
}

void __tgt_interop_use(ident_t *LocRef, int32_t Gtid,
                       omp_interop_val_t *&InteropPtr, int32_t DeviceId,
                       int32_t Ndeps, kmp_depend_info_t *DepList,
                       int32_t HaveNowait) {
  interop_ctx_t Ctx = {0, {false, (bool)HaveNowait, 0}, Gtid};
  dep_pack_t Deps = {Ndeps, 0, DepList, nullptr};
  __tgt_interop_use60(LocRef, InteropPtr, &Ctx, Ndeps ? &Deps : nullptr);
}

void __tgt_interop_destroy(ident_t *LocRef, int32_t Gtid,
                           omp_interop_val_t *&InteropPtr, int32_t DeviceId,
                           int32_t Ndeps, kmp_depend_info_t *DepList,
                           int32_t HaveNowait) {
  interop_ctx_t Ctx = {0, {false, (bool)HaveNowait, 0}, Gtid};
  dep_pack_t Deps = {Ndeps, 0, DepList, nullptr};
  __tgt_interop_release(LocRef, InteropPtr, &Ctx, Ndeps ? &Deps : nullptr);
}

} // extern "C"

bool omp_interop_val_t::isCompatibleWith(int32_t InteropType,
                                         const interop_spec_t &Spec) {
  if (interop_type != InteropType)
    return false;
  if (Spec.fr_id != fr_id)
    return false;
  if (Spec.attrs.inorder != attrs.inorder)
    return false;
  if (Spec.impl_attrs != impl_attrs)
    return false;

  return true;
}

bool omp_interop_val_t::isCompatibleWith(int32_t InteropType,
                                         const interop_spec_t &Spec,
                                         int64_t DeviceNum, int GTID) {
  if (device_id != DeviceNum)
    return false;

  if (GTID != owner_gtid)
    return false;

  return isCompatibleWith(InteropType, Spec);
}

int32_t omp_interop_val_t::sync_barrier() {
  if (queue)
    if (auto *Res = olSyncQueue(queue)) {
      FATAL_MESSAGE(device_id,
                    "Interop sync barrier failed for %p object: %s\n", this,
                    Res->Details);
    }
  ODBG(ODT_Sync) << "Calling completion callbacks for " << this;
  runCompletionCbs();
  return OFFLOAD_SUCCESS;
}

int32_t omp_interop_val_t::release() {
  if (queue && (!hasOwner() || !isClean()))
    sync_barrier();

  int32_t Ret = OFFLOAD_SUCCESS;
  if (queue)
    if (auto *Res = olDestroyQueue(queue)) {
      REPORT() << "Failed to destroy queue of interop " << this << ": "
               << Res->Details;
      Ret = OFFLOAD_FAIL;
    }
  delete this;
  return Ret;
}

void syncImplicitInterops(int Gtid, void *Event) {
  if (PM->InteropTbl.size() == 0)
    return;

  ODBG(ODT_Sync) << "target_sync: syncing interops for gtid " << Gtid
                 << ", event " << Event;

  for (auto iop : PM->InteropTbl) {
    if (iop->queue && iop->isOwnedBy(Gtid) && !iop->isClean()) {
      iop->sync_barrier();
      iop->markClean();

      // Alternate implementation option in case using barriers is not
      // efficient enough:
      //
      // Instead of using a synchronous barrier, queue an asynchronous
      // barrier and create a proxy task associated to the event to handle
      // OpenMP synchronizations.
      // When the event is completed, fulfill the proxy task to notify the
      // OpenMP runtime.
      // event = iop->asyncBarrier();
      // ptask = createProxyTask();
      // Events->add(event,ptask);
    }
  }
  // This would be needed for the alternate implementation
  // processEvents();
}

void InteropTblTy::clear() {
  ODBG(ODT_Deinit) << "Clearing Interop Table";
  PerThreadTable::clear([](auto &IOP) { IOP->release(); });
}
