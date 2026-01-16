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
const char *getForeignRuntimeIdToStr(const tgt_foreign_runtime_id_t FrId) {
  if (FrId < tgt_fr_none || FrId >= tgt_fr_last)
    return ("unknown");
  return ForeignRuntimeStrTbl[FrId];
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
    if (InteropVal.device_info.Device)
      return InteropVal.device_info.Device;
    *Err = omp_irc_no_value;
    return const_cast<char *>(InteropVal.err_str);
  case omp_ipr_platform:
    return InteropVal.device_info.Platform;
  case omp_ipr_device_context:
    return InteropVal.device_info.Context;
  case omp_ipr_targetsync:
    return InteropVal.async_info ? InteropVal.async_info->Queue : nullptr;
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
  if ((Property == omp_ipr_device || Property == omp_ipr_device_context) &&
      (*InteropPtr)->interop_type == kmp_interop_type_targetsync) {
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
  auto InteropSpec = Device.RTL->select_interop_preference(
      DeviceNum, InteropType, NumPrefers, Prefers);
  if (InteropSpec.fr_id == tgt_fr_none) {
    ODBG(ODT_Interface) << "Interop request not supported by device "
                        << DeviceNum;
    return omp_interop_none;
  }
  ODBG(ODT_Interface) << "Selected interop preference is fr_id="
                      << getForeignRuntimeIdToStr(
                             (tgt_foreign_runtime_id_t)InteropSpec.fr_id)
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

  Interop = Device.RTL->create_interop(DeviceNum, InteropType, &InteropSpec);
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

  auto DeviceOrErr = Interop->getDevice();
  if (!DeviceOrErr) {
    REPORT() << "Failed to get device for interop " << Interop << ": "
             << toString(DeviceOrErr.takeError());
    return OFFLOAD_FAIL;
  }
  auto &IOPDevice = *DeviceOrErr;

  if (Interop->async_info && Interop->async_info->Queue) {
    if (Nowait)
      Interop->async_barrier(IOPDevice);
    else {
      Interop->flush(IOPDevice);
      Interop->sync_barrier(IOPDevice);
      Interop->markClean();
    }
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

  auto DeviceOrErr = Interop->getDevice();
  if (!DeviceOrErr) {
    REPORT() << "Failed to get device for interop " << Interop << ": "
             << toString(DeviceOrErr.takeError());
    return OFFLOAD_FAIL;
  }

  return Interop->release(*DeviceOrErr);
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

llvm::Expected<DeviceTy &> omp_interop_val_t::getDevice() const {
  return PM->getDevice(device_id);
}

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

int32_t omp_interop_val_t::flush(DeviceTy &Device) {
  return Device.RTL->flush_queue(this);
}

int32_t omp_interop_val_t::sync_barrier(DeviceTy &Device) {
  if (Device.RTL->sync_barrier(this) != OFFLOAD_SUCCESS) {
    FATAL_MESSAGE(device_id, "Interop sync barrier failed for %p object\n",
                  this);
  }
  ODBG(ODT_Sync) << "Calling completion callbacks for " << this;
  runCompletionCbs();
  return OFFLOAD_SUCCESS;
}

int32_t omp_interop_val_t::async_barrier(DeviceTy &Device) {
  return Device.RTL->async_barrier(this);
}

int32_t omp_interop_val_t::release(DeviceTy &Device) {
  if (async_info != nullptr && (!hasOwner() || !isClean())) {
    flush(Device);
    sync_barrier(Device);
  }
  return Device.RTL->release_interop(device_id, this);
}

void syncImplicitInterops(int Gtid, void *Event) {
  if (PM->InteropTbl.size() == 0)
    return;

  ODBG(ODT_Sync) << "target_sync: syncing interops for gtid " << Gtid
                 << ", event " << Event;

  for (auto iop : PM->InteropTbl) {
    if (iop->async_info && iop->async_info->Queue && iop->isOwnedBy(Gtid) &&
        !iop->isClean()) {

      auto DeviceOrErr = iop->getDevice();
      if (!DeviceOrErr) {
        REPORT() << "Failed to get device for interop " << iop << ": "
                 << toString(DeviceOrErr.takeError());
        continue;
      }
      auto &IOPDevice = *DeviceOrErr;

      iop->flush(IOPDevice);
      iop->sync_barrier(IOPDevice);
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
  PerThreadTable::clear([](auto &IOP) {
    auto DeviceOrErr = IOP->getDevice();
    if (!DeviceOrErr) {
      REPORT() << "Failed to get device for interop " << IOP << ": "
               << toString(DeviceOrErr.takeError());
      return;
    }
    IOP->release(*DeviceOrErr);
  });
}
