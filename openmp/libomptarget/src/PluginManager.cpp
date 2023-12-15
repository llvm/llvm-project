//===-- PluginManager.cpp - Plugin loading and communication API ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for handling plugins.
//
//===----------------------------------------------------------------------===//

#include "PluginManager.h"
#include "Shared/Debug.h"
#include "Shared/Profile.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

using namespace llvm;
using namespace llvm::sys;

PluginManager *PM;

// List of all plugins that can support offloading.
static const char *RTLNames[] = {ENABLED_OFFLOAD_PLUGINS};

Expected<std::unique_ptr<PluginAdaptorTy>>
PluginAdaptorTy::create(const std::string &Name) {
  DP("Attempting to load library '%s'...\n", Name.c_str());
  TIMESCOPE_WITH_NAME_AND_IDENT(Name, (const ident_t *)nullptr);

  std::string ErrMsg;
  auto LibraryHandler = std::make_unique<DynamicLibrary>(
      DynamicLibrary::getPermanentLibrary(Name.c_str(), &ErrMsg));

  if (!LibraryHandler->isValid()) {
    // Library does not exist or cannot be found.
    return createStringError(inconvertibleErrorCode(),
                             "Unable to load library '%s': %s!\n", Name.c_str(),
                             ErrMsg.c_str());
  }

  DP("Successfully loaded library '%s'!\n", Name.c_str());
  auto PluginAdaptor = std::unique_ptr<PluginAdaptorTy>(
      new PluginAdaptorTy(Name, std::move(LibraryHandler)));
  if (auto Err = PluginAdaptor->init())
    return Err;
  return std::move(PluginAdaptor);
}

PluginAdaptorTy::PluginAdaptorTy(const std::string &Name,
                                 std::unique_ptr<llvm::sys::DynamicLibrary> DL)
    : Name(Name), LibraryHandler(std::move(DL)) {}

Error PluginAdaptorTy::init() {

#define PLUGIN_API_HANDLE(NAME, MANDATORY)                                     \
  NAME = reinterpret_cast<decltype(NAME)>(                                     \
      LibraryHandler->getAddressOfSymbol(GETNAME(__tgt_rtl_##NAME)));          \
  if (MANDATORY && !NAME) {                                                    \
    return createStringError(inconvertibleErrorCode(),                         \
                             "Invalid plugin as necessary interface function " \
                             "(%s) was not found.\n",                          \
                             std::string(#NAME).c_str());                      \
  }

#include "Shared/PluginAPI.inc"
#undef PLUGIN_API_HANDLE

  // Remove plugin on failure to call optional init_plugin
  int32_t Rc = init_plugin();
  if (Rc != OFFLOAD_SUCCESS) {
    return createStringError(inconvertibleErrorCode(),
                             "Unable to initialize library '%s': %u!\n",
                             Name.c_str(), Rc);
  }

  // No devices are supported by this RTL?
  NumberOfPluginDevices = number_of_devices();
  if (!NumberOfPluginDevices) {
    return createStringError(inconvertibleErrorCode(),
                             "No devices supported in this RTL\n");
  }

  DP("Registered '%s' with %d plugin visible devices!\n", Name.c_str(),
     NumberOfPluginDevices);
  return Error::success();
}

void PluginAdaptorTy::addOffloadEntries(DeviceImageTy &DI) {
  for (int32_t I = 0, E = getNumberOfUserDevices(); I < E; ++I) {
    auto DeviceOrErr = PM->getDevice(DeviceOffset + I);
    if (!DeviceOrErr)
      FATAL_MESSAGE(DeviceOffset + I, "%s",
                    toString(DeviceOrErr.takeError()).c_str());

    DeviceTy &Device = *DeviceOrErr;
    for (OffloadEntryTy &Entry : DI.entries())
      Device.addOffloadEntry(Entry);
  }
}

void PluginManager::init() {
  TIMESCOPE();
  DP("Loading RTLs...\n");

  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (const char *Name : RTLNames) {
    auto PluginAdaptorOrErr =
        PluginAdaptorTy::create(std::string(Name) + ".so");
    if (!PluginAdaptorOrErr) {
      [[maybe_unused]] std::string InfoMsg =
          toString(PluginAdaptorOrErr.takeError());
      DP("%s", InfoMsg.c_str());
    } else {
      PluginAdaptors.push_back(std::move(*PluginAdaptorOrErr));
    }
  }

  DP("RTLs loaded!\n");
}

void PluginAdaptorTy::initDevices(PluginManager &PM) {
  if (isUsed())
    return;
  TIMESCOPE();

  // If this RTL is not already in use, initialize it.
  assert(getNumberOfPluginDevices() > 0 &&
         "Tried to initialize useless plugin adaptor");

  // Initialize the device information for the RTL we are about to use.
  auto ExclusiveDevicesAccessor = PM.getExclusiveDevicesAccessor();

  // Initialize the index of this RTL and save it in the used RTLs.
  DeviceOffset = ExclusiveDevicesAccessor->size();

  // If possible, set the device identifier offset in the plugin.
  if (set_device_offset)
    set_device_offset(DeviceOffset);

  int32_t NumPD = getNumberOfPluginDevices();
  ExclusiveDevicesAccessor->reserve(DeviceOffset + NumPD);
  for (int32_t PDevI = 0, UserDevId = DeviceOffset; PDevI < NumPD; PDevI++) {
    auto Device = std::make_unique<DeviceTy>(this, UserDevId, PDevI);
    if (auto Err = Device->init()) {
      DP("Skip plugin known device %d: %s\n", PDevI,
         toString(std::move(Err)).c_str());
      continue;
    }

    ExclusiveDevicesAccessor->push_back(std::move(Device));
    ++NumberOfUserDevices;
    ++UserDevId;
  }

  DP("Plugin adaptor " DPxMOD " has index %d, exposes %d out of %d devices!\n",
     DPxPTR(LibraryHandler.get()), DeviceOffset, NumberOfUserDevices,
     NumberOfPluginDevices);
}

void PluginManager::initAllPlugins() {
  for (auto &R : PluginAdaptors)
    R->initDevices(*this);
}

static void registerImageIntoTranslationTable(TranslationTable &TT,
                                              PluginAdaptorTy &RTL,
                                              __tgt_device_image *Image) {

  // same size, as when we increase one, we also increase the other.
  assert(TT.TargetsTable.size() == TT.TargetsImages.size() &&
         "We should have as many images as we have tables!");

  // Resize the Targets Table and Images to accommodate the new targets if
  // required
  unsigned TargetsTableMinimumSize =
      RTL.DeviceOffset + RTL.getNumberOfUserDevices();

  if (TT.TargetsTable.size() < TargetsTableMinimumSize) {
    TT.TargetsImages.resize(TargetsTableMinimumSize, 0);
    TT.TargetsTable.resize(TargetsTableMinimumSize, 0);
  }

  // Register the image in all devices for this target type.
  for (int32_t I = 0; I < RTL.getNumberOfUserDevices(); ++I) {
    // If we are changing the image we are also invalidating the target table.
    if (TT.TargetsImages[RTL.DeviceOffset + I] != Image) {
      TT.TargetsImages[RTL.DeviceOffset + I] = Image;
      TT.TargetsTable[RTL.DeviceOffset + I] =
          0; // lazy initialization of target table.
    }
  }
}

void PluginManager::registerLib(__tgt_bin_desc *Desc) {
  PM->RTLsMtx.lock();

  // Extract the exectuable image and extra information if availible.
  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i)
    PM->addDeviceImage(*Desc, Desc->DeviceImages[i]);

  // Register the images with the RTLs that understand them, if any.
  for (DeviceImageTy &DI : PM->deviceImages()) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &DI.getExecutableImage();
    __tgt_image_info *Info = &DI.getImageInfo();

    PluginAdaptorTy *FoundRTL = nullptr;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : PM->pluginAdaptors()) {
      if (R.is_valid_binary_info) {
        if (!R.is_valid_binary_info(Img, Info)) {
          DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
             DPxPTR(Img->ImageStart), R.Name.c_str());
          continue;
        }
      } else if (!R.is_valid_binary(Img)) {
        DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
           DPxPTR(Img->ImageStart), R.Name.c_str());
        continue;
      }

      DP("Image " DPxMOD " is compatible with RTL %s!\n",
         DPxPTR(Img->ImageStart), R.Name.c_str());

      R.initDevices(*this);

      // Initialize (if necessary) translation table for this library.
      PM->TrlTblMtx.lock();
      if (!PM->HostEntriesBeginToTransTable.count(Desc->HostEntriesBegin)) {
        PM->HostEntriesBeginRegistrationOrder.push_back(Desc->HostEntriesBegin);
        TranslationTable &TransTable =
            (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];
        TransTable.HostTable.EntriesBegin = Desc->HostEntriesBegin;
        TransTable.HostTable.EntriesEnd = Desc->HostEntriesEnd;
      }

      // Retrieve translation table for this library.
      TranslationTable &TransTable =
          (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];

      DP("Registering image " DPxMOD " with RTL %s!\n", DPxPTR(Img->ImageStart),
         R.Name.c_str());
      registerImageIntoTranslationTable(TransTable, R, Img);
      R.UsedImages.insert(Img);

      PM->TrlTblMtx.unlock();
      FoundRTL = &R;

      // Register all offload entries with the devices handled by the plugin.
      R.addOffloadEntries(DI);

      // if an RTL was found we are done - proceed to register the next image
      break;
    }

    if (!FoundRTL) {
      DP("No RTL found for image " DPxMOD "!\n", DPxPTR(Img->ImageStart));
    }
  }
  PM->RTLsMtx.unlock();

  DP("Done registering entries!\n");
}

// Temporary forward declaration, old style CTor/DTor handling is going away.
int target(ident_t *Loc, DeviceTy &Device, void *HostPtr,
           KernelArgsTy &KernelArgs, AsyncInfoTy &AsyncInfo);

void PluginManager::unregisterLib(__tgt_bin_desc *Desc) {
  DP("Unloading target library!\n");

  PM->RTLsMtx.lock();
  // Find which RTL understands each image, if any.
  for (DeviceImageTy &DI : PM->deviceImages()) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &DI.getExecutableImage();

    PluginAdaptorTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto &R : PM->pluginAdaptors()) {
      if (!R.isUsed())
        continue;

      // Ensure that we do not use any unused images associated with this RTL.
      if (!R.UsedImages.contains(Img))
        continue;

      FoundRTL = &R;

      // Execute dtors for static objects if the device has been used, i.e.
      // if its PendingCtors list has been emptied.
      for (int32_t I = 0; I < FoundRTL->getNumberOfUserDevices(); ++I) {
        auto DeviceOrErr = PM->getDevice(FoundRTL->DeviceOffset + I);
        if (!DeviceOrErr)
          FATAL_MESSAGE(FoundRTL->DeviceOffset + I, "%s",
                        toString(DeviceOrErr.takeError()).c_str());

        DeviceTy &Device = *DeviceOrErr;
        Device.PendingGlobalsMtx.lock();
        if (Device.PendingCtorsDtors[Desc].PendingCtors.empty()) {
          AsyncInfoTy AsyncInfo(Device);
          for (auto &Dtor : Device.PendingCtorsDtors[Desc].PendingDtors) {
            int Rc =
                target(nullptr, Device, Dtor, CTorDTorKernelArgs, AsyncInfo);
            if (Rc != OFFLOAD_SUCCESS) {
              DP("Running destructor " DPxMOD " failed.\n", DPxPTR(Dtor));
            }
          }
          // Remove this library's entry from PendingCtorsDtors
          Device.PendingCtorsDtors.erase(Desc);
          // All constructors have been issued, wait for them now.
          if (AsyncInfo.synchronize() != OFFLOAD_SUCCESS)
            DP("Failed synchronizing destructors kernels.\n");
        }
        Device.PendingGlobalsMtx.unlock();
      }

      DP("Unregistered image " DPxMOD " from RTL " DPxMOD "!\n",
         DPxPTR(Img->ImageStart), DPxPTR(R.LibraryHandler.get()));

      break;
    }

    // if no RTL was found proceed to unregister the next image
    if (!FoundRTL) {
      DP("No RTLs in use support the image " DPxMOD "!\n",
         DPxPTR(Img->ImageStart));
    }
  }
  PM->RTLsMtx.unlock();
  DP("Done unregistering images!\n");

  // Remove entries from PM->HostPtrToTableMap
  PM->TblMapMtx.lock();
  for (__tgt_offload_entry *Cur = Desc->HostEntriesBegin;
       Cur < Desc->HostEntriesEnd; ++Cur) {
    PM->HostPtrToTableMap.erase(Cur->addr);
  }

  // Remove translation table for this descriptor.
  auto TransTable =
      PM->HostEntriesBeginToTransTable.find(Desc->HostEntriesBegin);
  if (TransTable != PM->HostEntriesBeginToTransTable.end()) {
    DP("Removing translation table for descriptor " DPxMOD "\n",
       DPxPTR(Desc->HostEntriesBegin));
    PM->HostEntriesBeginToTransTable.erase(TransTable);
  } else {
    DP("Translation table for descriptor " DPxMOD " cannot be found, probably "
       "it has been already removed.\n",
       DPxPTR(Desc->HostEntriesBegin));
  }

  PM->TblMapMtx.unlock();

  DP("Done unregistering library!\n");
}

Expected<DeviceTy &> PluginManager::getDevice(uint32_t DeviceNo) {
  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
  if (DeviceNo >= ExclusiveDevicesAccessor->size())
    return createStringError(
        inconvertibleErrorCode(),
        "Device number '%i' out of range, only %i devices available", DeviceNo,
        ExclusiveDevicesAccessor->size());

  return *(*ExclusiveDevicesAccessor)[DeviceNo];
}
