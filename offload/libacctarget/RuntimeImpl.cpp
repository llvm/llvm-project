//===- AccImpl.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "DeviceManager.h"
#include "Logger.h"
#include "PluginManager.h"
#include "Private.h"
#include "Shared/Debug.h"

using namespace llvm::acc::target::debug;

namespace llvm::acc::target {
int accIsPresent(void *Ptr) {
  FUNC_LOGGER();
  ODBG(ADT_Interface) << "Address " << Ptr;

  auto DeviceOrErr = DM->getDevice();
  if (!DeviceOrErr)
    REPORT_FATAL() << toString(DeviceOrErr.takeError()).c_str();

  TargetPointerResultTy TPR =
      DeviceOrErr->getMappingInfo().getTgtPtrBegin(const_cast<void *>(Ptr), 1,
                                                   /*UpdateRefCount=*/false,
                                                   /*UseHoldRefCount=*/false);
  int Rc = TPR.isPresent();
  ODBG(ADT_Interface) << "Result " << Rc;
  return Rc;
}

void *accAlloc(size_t Size) {
  FUNC_LOGGER();
  ODBG(ADT_Interface) << "Allocating " << Size << " bytes";

  if (Size <= 0) {
    ODBG(ADT_Interface) << "Non-positive length";
    return NULL;
  }

  void *Rc = NULL;
  auto DeviceOrErr = DM->getDevice();
  if (!DeviceOrErr)
    REPORT_FATAL() << toString(DeviceOrErr.takeError()).c_str();

  Rc = DeviceOrErr->allocData(Size, nullptr);
  ODBG(ADT_Interface) << "Device ptr " << Rc;
  return Rc;
}

void accFree(void *DevicePtr) {
  FUNC_LOGGER();
  ODBG(ADT_Interface) << "Address " << DevicePtr;
  auto DeviceOrErr = DM->getDevice();
  if (!DeviceOrErr)
    REPORT_FATAL() << toString(DeviceOrErr.takeError()).c_str();

  if (DeviceOrErr->deleteData(DevicePtr) == OFFLOAD_FAIL)
    REPORT_FATAL() << "Failed to deallocate device ptr. Set "
                      "OFFLOAD_TRACK_ALLOCATION_TRACES=1 to track allocations.";
}

void accMemcpyToDevice(void *Dst, void *Src, size_t Bytes) {
  FUNC_LOGGER();
  ODBG(ADT_Interface) << Dst << " <- " << Src << ", " << Bytes << " bytes";

  if (!Dst || !Src || Bytes <= 0) {
    if (Bytes == 0) {
      ODBG(ADT_Interface) << "Zero bytes, nothing to do";
      return;
    }
    REPORT() << "Invalid arguments";
    return;
  }

  auto DeviceOrErr = DM->getDevice();
  if (!DeviceOrErr)
    REPORT_FATAL() << toString(DeviceOrErr.takeError()).c_str();
  AsyncInfoTy AsyncInfo(*DeviceOrErr);
  int Rc = DeviceOrErr->submitData(Dst, Src, Bytes, AsyncInfo);
  ODBG(ADT_Interface) << "Result " << Rc;
}

void accMemcpyFromDevice(void *Dst, void *Src, size_t Bytes) {
  FUNC_LOGGER();
  ODBG(ADT_Interface) << Dst << " <- " << Src << ", " << Bytes << " bytes";

  if (!Dst || !Src || Bytes <= 0) {
    if (Bytes == 0) {
      ODBG(ADT_Interface) << "Zero bytes, nothing to do";
      return;
    }
    REPORT() << "Invalid arguments";
    return;
  }

  auto DeviceOrErr = DM->getDevice();
  if (!DeviceOrErr)
    REPORT_FATAL() << toString(DeviceOrErr.takeError()).c_str();
  AsyncInfoTy AsyncInfo(*DeviceOrErr);
  int Rc = DeviceOrErr->retrieveData(Dst, Src, Bytes, AsyncInfo);
  ODBG(ADT_Interface) << "Result " << Rc;
}

void accMemcpyD2D(void *Dst, void *Src, size_t Bytes, int DstDevice,
                  int SrcDevice) {
  FUNC_LOGGER();
  ODBG(ADT_Interface) << Dst << " <- " << Src << ", " << Bytes << " bytes";

  if (!Dst || !Src || Bytes <= 0) {
    if (Bytes == 0) {
      ODBG(ADT_Interface) << "Zero bytes, nothing to do";
      return;
    }
    REPORT() << "Invalid arguments";
    return;
  }

  auto DstDeviceOrErr = DM->getDevice();
  if (!DstDeviceOrErr)
    REPORT_FATAL() << toString(DstDeviceOrErr.takeError()).c_str();
  auto SrcDeviceOrErr = DM->getDevice();
  if (!SrcDeviceOrErr)
    REPORT_FATAL() << toString(SrcDeviceOrErr.takeError()).c_str();
  if (!SrcDeviceOrErr->isDataExchangable(*DstDeviceOrErr)) {
    REPORT() << "D2D not allowed for current device type";
    return;
  }

  AsyncInfoTy AsyncInfo(*SrcDeviceOrErr);
  int Rc =
      SrcDeviceOrErr->dataExchange(Src, *DstDeviceOrErr, Dst, Bytes, AsyncInfo);
  ODBG(ADT_Interface) << "Result " << Rc;
}

void accMapData(void *Hst, void *Dev, size_t Bytes) {
  FUNC_LOGGER();
  ODBG(ADT_Interface) << Hst << " <-> " << Dev << ", " << Bytes << " bytes";

  if (!Hst || !Dev || Bytes <= 0) {
    REPORT() << "Invalid arguments";
    return;
  }

  auto DeviceOrErr = DM->getDevice();
  if (!DeviceOrErr)
    REPORT_FATAL() << toString(DeviceOrErr.takeError()).c_str();

  int Rc = DeviceOrErr->getMappingInfo().associatePtr(
      const_cast<void *>(Hst), const_cast<void *>(Dev), Bytes);
  ODBG(ADT_Interface) << "Result " << Rc;
}

void accUnmapData(void *Hst) {
  FUNC_LOGGER();
  ODBG(ADT_Interface) << Hst;

  if (!Hst) {
    REPORT() << "Invalid arguments";
    return;
  }

  auto DeviceOrErr = DM->getDevice();
  if (!DeviceOrErr)
    REPORT_FATAL() << toString(DeviceOrErr.takeError()).c_str();
  int Rc =
      DeviceOrErr->getMappingInfo().disassociatePtr(const_cast<void *>(Hst));
  ODBG(ADT_Interface) << "Result " << Rc;
}

} // namespace llvm::acc::target
